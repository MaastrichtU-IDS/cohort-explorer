from fastapi import APIRouter, Depends

from api.services.auth import AuthenticatedUser, get_current_user
from api.services.cache import get_cache

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
        },
    }
