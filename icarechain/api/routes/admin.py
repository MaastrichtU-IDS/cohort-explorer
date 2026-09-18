import json
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
            "allowed_projects": c.get("allowed_projects", []),
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

    # Snapshot ids are single-use, and a snapshot is only meaningful for the deployment it was taken
    # under: reverting to one that predates the current deployment erases the very contracts this API
    # points at. So the id saved after a reset is stored together with the deployment's timestamp and
    # is trusted only while that still matches deployments.json; otherwise fall back to the deployer's.
    deployments = load_deployments()
    deployment_tag = str(deployments.get("timestamp") or "")
    vault_address = (deployments.get("contracts") or {}).get("duoConsentVaultV2")

    def has_code(addr: str | None) -> bool:
        if not addr:
            return False
        code = rpc("eth_getCode", [addr, "latest"])
        return bool(code) and code != "0x"

    if not has_code(vault_address):
        raise HTTPException(
            409,
            "The contracts in deployments.json are not on the Hardhat node (the chain is older than the "
            "current deployment). Recreate the chain: docker compose rm -sf icarechain-hardhat "
            "icarechain-deployer icarechain-api && docker compose up -d",
        )

    # NOTE on ordering: Hardhat (EDR) drops every snapshot with an id >= the requested one *before*
    # checking that it exists, so a failed evm_revert to a stale LOWER id destroys valid later
    # snapshots. A record whose tag matches means resets already happened under this deployment and
    # the deployer's base id is consumed: try only the saved id. A missing or mismatched record means
    # no reset has happened under this deployment yet: the base id is fresh, try only that.
    candidates: list[str] = []
    if redis is not None:
        saved = await redis.get(SNAPSHOT_KEY)
        if saved:
            raw = saved.decode() if isinstance(saved, bytes) else str(saved)
            try:
                rec = json.loads(raw)
            except ValueError:
                rec = None  # legacy plain id with no deployment tag: not trustworthy
            if isinstance(rec, dict) and rec.get("deployment") == deployment_tag and rec.get("id"):
                candidates.append(str(rec["id"]))
    base = deployments.get("snapshotId")
    if not candidates and base:
        candidates.append(str(base))

    reverted_to = None
    for sid in candidates:
        if rpc("evm_revert", [sid]) is True:
            reverted_to = sid
            break
    if reverted_to is None:
        raise HTTPException(
            409,
            "No usable snapshot on the Hardhat node. Recreate the chain: docker compose rm -sf "
            "icarechain-hardhat icarechain-deployer icarechain-api && docker compose up -d",
        )

    if not has_code(vault_address):
        # Should be impossible given the tag check above; fail loudly rather than leave a silent broken state.
        raise HTTPException(
            500,
            f"Reverted to snapshot {reverted_to} but the deployed contracts are gone. Recreate the chain: "
            "docker compose rm -sf icarechain-hardhat icarechain-deployer icarechain-api && docker compose up -d",
        )

    new_snapshot = rpc("evm_snapshot", [])

    # Chain state is gone; everything cached from it is now stale.
    if redis is not None:
        await redis.flushdb()
        await redis.set(SNAPSHOT_KEY, json.dumps({"id": str(new_snapshot), "deployment": deployment_tag}))
    else:
        await cache.clear()

    logger.warning("Local chain reset by %s: reverted to snapshot %s, new snapshot %s", user.email or user.email_hash, reverted_to, new_snapshot)
    return {"success": True, "revertedTo": reverted_to, "snapshotId": new_snapshot, "blockNumber": w3.eth.block_number}
