import logging

from fastapi import APIRouter, Depends, HTTPException

from api.config import load_deployments
from api.services.auth import AuthenticatedUser, get_current_user
from api.services.blockchain import get_blockchain_service
from api.services.cache import get_cache

logger = logging.getLogger(__name__)

HARDHAT_CHAIN_ID = 31337
SNAPSHOT_KEY = "_meta:reset_snapshot"

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/overview", summary="Admin overview of all consent declarations, access requests, and requester profiles")
async def admin_overview(user: AuthenticatedUser = Depends(get_current_user)) -> dict:
    cache = get_cache()

    all_consents = await cache.get_all_consents()

    consents_out = []
    seen_requester_hashes: set[str] = set()

    for c in all_consents:
        cohort_hash = (c.get("cohort_hash") or "").lstrip("0x")
        raw_grants = await cache.get_cohort_access_grants(cohort_hash) if cohort_hash else []

        enriched_grants = []
        for g in raw_grants:
            req_hash = g.get("requester_hash") or ""
            profile = None
            if req_hash:
                seen_requester_hashes.add(req_hash)
                profile = await cache.get_authorization_token(f"requester:{req_hash}")
            enriched_grants.append({
                "requester": g.get("requester", ""),
                "requester_hash": req_hash,
                "status": g.get("status", "approved" if g.get("approved") else "pending"),
                "intended_use": g.get("intended_use"),
                "disease_code": g.get("disease_code"),
                "disease_codes": g.get("disease_codes") or ([g["disease_code"]] if g.get("disease_code") else []),
                "reason": g.get("reason"),
                "reason_detail": g.get("reason_detail"),
                "decided_at": g.get("decided_at"),
                "project_id": g.get("project_id"),
                "abstract": g.get("abstract"),
                "requested_at": g.get("requested_at"),
                "granted_at": g.get("granted_at"),
                "request_id": g.get("request_id"),
                "tx_hash": g.get("tx_hash"),
                "profile": {
                    "institution_id": profile.get("institution_id") if profile else None,
                    "requester_type": profile.get("requester_type") if profile else None,
                    "country_code": profile.get("country_code") if profile else None,
                } if profile else None,
            })

        consents_out.append({
            "cohort_id": c.get("cohort_id", ""),
            "cohort_hash": c.get("cohort_hash", ""),
            "permission": c.get("permission", ""),
            "modifiers": c.get("modifiers", []),
            "disease_code": c.get("disease_code"),
            "disease_codes": c.get("disease_codes") or ([c["disease_code"]] if c.get("disease_code") else []),
            "data_use_description": c.get("data_use_description"),
            "additional_restrictions": c.get("additional_restrictions"),
            "research_scope": c.get("research_scope"),
            "allowed_countries": c.get("allowed_countries", []),
            "allowed_institutions": c.get("allowed_institutions", []),
            "moratorium_months": c.get("moratorium_months"),
            "active": bool(c.get("active", False)),
            "valid_until": c.get("valid_until"),
            "recorded_at": c.get("recorded_at"),
            "owners": c.get("owners", []),
            "access_grants": enriched_grants,
        })

    profiles_out = []
    for rh in seen_requester_hashes:
        profile = await cache.get_authorization_token(f"requester:{rh}")
        if profile:
            profiles_out.append({
                "email_hash": rh,
                "address": profile.get("address", ""),
                "institution_id": profile.get("institution_id", ""),
                "requester_type": profile.get("requester_type", ""),
                "country_code": profile.get("country_code"),
                "public_profile": profile.get("public_profile", False),
                "updated_at": profile.get("updated_at"),
            })

    profiles_out.sort(key=lambda p: p.get("updated_at") or "", reverse=True)

    return {
        "consents": consents_out,
        "requester_profiles": profiles_out,
        "stats": {
            "total_consents": len(all_consents),
            "active_consents": sum(1 for c in all_consents if c.get("active")),
            "total_requester_profiles": len(profiles_out),
            "total_access_requests": sum(len(c["access_grants"]) for c in consents_out),
            "approved_access_requests": sum(
                sum(1 for g in c["access_grants"] if g["status"] == "approved")
                for c in consents_out
            ),
            "pending_access_requests": sum(
                sum(1 for g in c["access_grants"] if g["status"] == "pending")
                for c in consents_out
            ),
            "rejected_access_requests": sum(
                sum(1 for g in c["access_grants"] if g["status"] in ("rejected", "denied"))
                for c in consents_out
            ),
        },
    }


@router.post("/reset", summary="DEV ONLY: revert the local Hardhat chain to its post-deploy snapshot and flush the cache")
async def admin_reset(user: AuthenticatedUser = Depends(get_current_user)) -> dict:
    service = get_blockchain_service()
    w3 = service.w3

    try:
        chain_id = w3.eth.chain_id
    except Exception as e:
        raise HTTPException(502, f"Cannot reach RPC node: {e}")
    if chain_id != HARDHAT_CHAIN_ID:
        raise HTTPException(403, f"Chain reset is only allowed on the local Hardhat chain (chainId {HARDHAT_CHAIN_ID}); connected to {chain_id}")

    cache = get_cache()
    redis = getattr(cache, "client", None)

    def rpc(method: str, params: list):
        resp = w3.provider.make_request(method, params)
        if "error" in resp and resp["error"]:
            raise HTTPException(502, f"{method} failed: {resp['error']}")
        return resp.get("result")

    # Snapshot ids are single-use: prefer the one saved after the last reset, then the deployer's.
    candidates: list[str] = []
    if redis is not None:
        saved = await redis.get(SNAPSHOT_KEY)
        if saved:
            candidates.append(saved.decode() if isinstance(saved, bytes) else str(saved))
    base = load_deployments().get("snapshotId")
    if base:
        candidates.append(str(base))

    reverted_to = None
    for sid in candidates:
        if rpc("evm_revert", [sid]) is True:
            reverted_to = sid
            break
    if reverted_to is None:
        raise HTTPException(
            409,
            "No usable snapshot on the Hardhat node. Restart the stack (docker compose down && docker compose up -d) "
            "so the deployer records a fresh snapshot.",
        )

    new_snapshot = rpc("evm_snapshot", [])

    # Chain state is gone; everything cached from it is now stale.
    if redis is not None:
        await redis.flushdb()
        await redis.set(SNAPSHOT_KEY, str(new_snapshot))
    else:
        await cache.clear()

    logger.warning("Local chain reset by %s: reverted to snapshot %s, new snapshot %s", user.email, reverted_to, new_snapshot)
    return {"success": True, "revertedTo": reverted_to, "snapshotId": new_snapshot, "blockNumber": w3.eth.block_number}
