import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from api.models.duo import (
    PERMISSION_VALUES,
    is_permission_compatible,
)
from api.services.auth import (
    AuthenticatedUser,
    fire_ibis_lifecycle,
    get_current_user,
    require_email_match,
)
from api.services.blockchain import get_blockchain_service
from api.services.cache import get_cache
from api.services.ibis import OperationType
from api.services.ontology import icd10

router = APIRouter(prefix="/requesters", tags=["requesters"])

REQUESTER_TYPES = {"PROFIT", "NONPROFIT", "ACADEMIC", "GOVERNMENT", "INDIVIDUAL"}

_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{1,63}$")
_NONPROFIT_TYPES = {"ACADEMIC", "NONPROFIT", "GOVERNMENT"}
_COMMERCIAL_TYPES = {"PROFIT", "COMMERCIAL"}

_OBLIGATION_MAP = {
    "PUB": ("PUBLICATION", "PUBLICATION", 730),
    "RTN": ("DATA_RETURN", "RETURN_DATA", 365),
    "MOR": ("MORATORIUM", None, 180),
}

def _validate_request_format(req):
    if not isinstance(req.intendedUse, str):
        return ("INVALID_INTENT", f"intendedUse must be a string; got {type(req.intendedUse).__name__}")
    use = req.intendedUse.strip().upper()
    if use not in PERMISSION_VALUES:
        return (
            "INVALID_INTENT",
            f"intendedUse must be one of {sorted(PERMISSION_VALUES)}; got {req.intendedUse!r}",
        )
    req.intendedUse = use

    if req.diseaseCode is not None:
        d = icd10.normalize(req.diseaseCode)
        if not icd10.is_requester_leaf(d):
            return (
                "INVALID_DISEASE_CODE",
                f"diseaseCode must be a specific bottom-level ICD-10 code (e.g. I50, E11, O24.4); "
                f"blocks/chapters are not allowed for requests (got {req.diseaseCode!r})",
            )
        req.diseaseCode = d

    if req.projectId is not None:
        p = req.projectId.strip()
        if not _PROJECT_ID_RE.match(p):
            return (
                "INVALID_PROJECT_ID",
                f"projectId must be 2-64 chars, alphanumerics with . _ - allowed (got {req.projectId!r})",
            )
        req.projectId = p

    if use == "DS" and not req.diseaseCode:
        return ("DISEASE_CODE_REQUIRED", "DS intent requires a diseaseCode")

    return None


def _match_or_reject(req, cohort: dict, profile: dict, requester_eoa: str, requester_sca: Optional[str]):
    if not cohort.get("active", False):
        return ("CONSENT_INACTIVE", "Consent is not active (revoked or never activated)")
    vu = cohort.get("valid_until")
    if vu:
        try:
            t = datetime.fromisoformat(vu.rstrip("Z"))
            if t <= datetime.utcnow():
                return ("CONSENT_EXPIRED", f"Consent expired at {vu}")
        except ValueError:
            pass

    cons_perm = (cohort.get("permission") or "").upper()
    if not is_permission_compatible(cons_perm, req.intendedUse):
        return (
            "PERMISSION_INCOMPATIBLE",
            f"intendedUse {req.intendedUse!r} not compatible with consent permission {cons_perm!r}",
        )

    if req.intendedUse == "DS":
        cd = cohort.get("disease_code")
        if cd and req.diseaseCode and not icd10.is_compatible(cd, req.diseaseCode):
            return (
                "DISEASE_NOT_COMPATIBLE",
                f"requested {req.diseaseCode!r} is not {cd!r} nor one of its ICD-10 descendants",
            )

    mods = {m.upper() for m in (cohort.get("modifiers") or [])}

    if "NPOA" in mods and req.intendedUse == "POA":
        return ("NPOA_BLOCKED", "POA intent blocked by NPOA modifier")

    if "GSO" in mods and req.intendedUse not in ("DS", "POA") and not req.diseaseCode:
        return ("GSO_REQUIRES_GENETIC", "GSO modifier requires DS/POA intent or a diseaseCode")

    if "GS" in mods:
        allowed = [c.upper() for c in (cohort.get("allowed_countries") or [])]
        country = (profile.get("country_code") or "").upper()
        if not country:
            return ("GS_COUNTRY_REQUIRED", "GS modifier: requester profile must include countryCode")
        if country not in allowed:
            return (
                "GS_COUNTRY_NOT_ALLOWED",
                f"requester country {country!r} not in allowed {allowed}",
            )

    if "IS" in mods:
        allowed = list(cohort.get("allowed_institutions") or [])
        inst = profile.get("institution_id") or ""
        if not inst:
            return ("IS_INSTITUTION_REQUIRED", "IS modifier: requester profile must include institutionId")
        if inst not in allowed:
            return (
                "IS_INSTITUTION_NOT_ALLOWED",
                f"requester institution {inst!r} not in allowed {allowed}",
            )

    if "PS" in mods:
        allowed = list(cohort.get("allowed_projects") or [])
        if not req.projectId:
            return ("PS_PROJECT_REQUIRED", "PS modifier: request must include projectId")
        if req.projectId not in allowed:
            return (
                "PS_PROJECT_NOT_ALLOWED",
                f"projectId {req.projectId!r} not in allowed {allowed}",
            )

    if "US" in mods:
        allowed = {(u or "").lower().removeprefix("0x") for u in (cohort.get("allowed_users") or [])}
        candidates = {a.lower().removeprefix("0x") for a in (requester_eoa, requester_sca) if a}
        if not (allowed & candidates):
            return ("US_USER_NOT_ALLOWED", "requester not in allowed users list")

    req_type = (profile.get("requester_type") or "").upper()
    if "NPU" in mods and req_type not in _NONPROFIT_TYPES:
        return (
            "NPU_NOT_NONPROFIT",
            f"NPU requires ACADEMIC/NONPROFIT/GOVERNMENT; got {req_type or 'unset'}",
        )
    if "NCU" in mods and req_type in _COMMERCIAL_TYPES:
        return ("NCU_BLOCKED", "NCU blocks commercial requesters")
    if "NPUNCU" in mods and req_type not in _NONPROFIT_TYPES:
        return (
            "NPUNCU_NOT_NONPROFIT",
            f"NPUNCU requires ACADEMIC/NONPROFIT/GOVERNMENT; got {req_type or 'unset'}",
        )

    return None


def _obligations_from_cohort(cohort: dict) -> list[dict]:
    modifiers = {m.upper() for m in (cohort.get("modifiers") or [])}
    out: list[dict] = []
    for mod, (commit_type, att_type, default_days) in _OBLIGATION_MAP.items():
        if mod not in modifiers:
            continue
        deadline_days = default_days
        if mod == "MOR" and cohort.get("moratorium_months"):
            deadline_days = int(cohort["moratorium_months"]) * 30
        if mod == "PUB" and cohort.get("publication_deadline_days"):
            deadline_days = int(cohort["publication_deadline_days"])
        out.append({
            "type": commit_type,
            "attestation_type": att_type,
            "deadline_days": deadline_days,
            "source_modifier": mod,
        })
    return out

class RequesterProfile(BaseModel):
    email: Optional[EmailStr] = None
    institutionId: str
    requesterType: str
    irbApprovalId: Optional[str] = None
    countryCode: Optional[str] = None
    publicProfile: bool = False
    gaslessOptIn: bool = False

class RequesterProfileResponse(BaseModel):
    address: str
    emailHash: str
    profile: RequesterProfile
    updatedAt: str

class CohortListing(BaseModel):
    cohortId: str
    cohortHash: str
    permission: str
    modifiers: list[str]
    diseaseCode: Optional[str] = None
    additionalRestrictions: Optional[str] = None
    active: bool

class AccessRequestCreate(BaseModel):
    email: EmailStr
    cohortId: str
    intendedUse: str
    diseaseCode: Optional[str] = None
    projectId: Optional[str] = None
    abstract: Optional[str] = None

class AccessRequestStatus(BaseModel):
    requestId: str
    cohortId: str
    status: str
    autoApproved: bool = False
    complianceScore: Optional[int] = None
    decision: Optional[str] = None
    requestedAt: Optional[str] = None
    decidedAt: Optional[str] = None
    obligationsCreated: list[dict] = Field(default_factory=list)
    pendingObligations: list[dict] = Field(default_factory=list)

class AccessCredential(BaseModel):
    tokenId: str
    cohortId: str
    cohortHash: str
    grantedAt: str
    expiresAt: Optional[str] = None
    revoked: bool = False

@router.put("/profile", response_model=RequesterProfileResponse, summary="Create or update the requester profile (institution, type, country)")
async def upsert_profile(profile: RequesterProfile, user: AuthenticatedUser = Depends(get_current_user)):
    if not profile.email:
        raise HTTPException(400, "email is required on PUT /profile")
    require_email_match(user, profile.email, "email")
    if profile.requesterType.upper() not in REQUESTER_TYPES:
        raise HTTPException(400, f"requesterType must be one of {sorted(REQUESTER_TYPES)}")

    cache = get_cache()
    existed = await cache.get_authorization_token(f"requester:{user.email_hash}")
    await fire_ibis_lifecycle(profile.email, OperationType.RENEW if existed else OperationType.REGISTER)

    now = datetime.utcnow().isoformat() + "Z"
    record = {
        "email_hash": user.email_hash,
        "address": user.address,
        "institution_id": profile.institutionId,
        "requester_type": profile.requesterType.upper(),
        "irb_approval_id": profile.irbApprovalId,
        "country_code": profile.countryCode,
        "public_profile": profile.publicProfile,
        "gasless_opt_in": profile.gaslessOptIn,
        "updated_at": now,
    }
    await cache.set_authorization_token(f"requester:{user.email_hash}", record, ttl=86400 * 365)

    try:
        await get_blockchain_service().set_requester_type(
            user_email=profile.email,
            requester_type=profile.requesterType,
            country_code=profile.countryCode,
        )
    except Exception:
        pass

    return RequesterProfileResponse(
        address=user.address, emailHash=user.email_hash, profile=profile, updatedAt=now,
    )

@router.get("/profile", response_model=RequesterProfileResponse, summary="Get the authenticated requester's profile")
async def get_profile(user: AuthenticatedUser = Depends(get_current_user)):
    record = await get_cache().get_authorization_token(f"requester:{user.email_hash}")
    if not record:
        raise HTTPException(404, "No profile — PUT /api/requesters/profile first")
    return RequesterProfileResponse(
        address=record["address"],
        emailHash=record["email_hash"],
        profile=RequesterProfile(
            email=None,
            institutionId=record["institution_id"],
            requesterType=record["requester_type"],
            irbApprovalId=record.get("irb_approval_id"),
            countryCode=record.get("country_code"),
            publicProfile=record.get("public_profile", False),
            gaslessOptIn=record.get("gasless_opt_in", False),
        ),
        updatedAt=record["updated_at"],
    )

@router.get("/cohorts", response_model=list[CohortListing], summary="Browse active cohorts the requester can apply to")
async def list_cohorts(user: AuthenticatedUser = Depends(get_current_user)):
    consents = await get_cache().get_all_consents()
    return [
        CohortListing(
            cohortId=c.get("cohort_id", ""),
            cohortHash=c.get("cohort_hash", ""),
            permission=c.get("permission", ""),
            modifiers=c.get("modifiers", []),
            diseaseCode=c.get("disease_code"),
            additionalRestrictions=c.get("additional_restrictions"),
            active=c.get("active", False),
        )
        for c in consents
        if c.get("active", False)
    ]

@router.post("/access-requests", summary="Request access; auto-cascades self-attestations and commitments for PUB/RTN/MOR modifiers")
async def create_access_request(req: AccessRequestCreate, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.email, "email")
    cache = get_cache()
    profile = await cache.get_authorization_token(f"requester:{user.email_hash}")
    if not profile:
        raise HTTPException(400, "Requester profile required. PUT /api/requesters/profile first.")

    from api.services.wallet import get_cohort_hash as _gch
    cohort_hash_unprefixed = _gch(req.cohortId).hex()
    cohort = await cache.get_consent(cohort_hash_unprefixed) or await cache.get_consent("0x" + cohort_hash_unprefixed)
    if not cohort:
        raise HTTPException(
            status_code=404,
            detail={
                "matched": False,
                "reason": "NO_CONSENT_DECLARED",
                "detail": "Cohort does not yet have usage permissions specified",
                "cohortId": req.cohortId,
            },
        )

    format_error = _validate_request_format(req)
    if format_error is not None:
        code, detail = format_error
        raise HTTPException(
            status_code=422,
            detail={"matched": False, "reason": code, "detail": detail, "cohortId": req.cohortId},
        )

    if "RS" in set(cohort.get("modifiers") or []) and not (req.abstract and req.abstract.strip()):
        raise HTTPException(
            status_code=422,
            detail={
                "matched": False,
                "reason": "ABSTRACT_REQUIRED",
                "detail": "RS modifier on cohort: abstract is required from requester",
                "cohortId": req.cohortId,
            },
        )

    service = get_blockchain_service()
    requester_info = service.get_role_account_if_attached(user.address, "REQUESTER")
    requester_sca = requester_info["account"] if requester_info else None

    rejection = _match_or_reject(req, cohort, profile, user.address, requester_sca)
    if rejection is not None:
        code, detail = rejection
        raise HTTPException(
            status_code=403,
            detail={"matched": False, "reason": code, "detail": detail, "cohortId": req.cohortId},
        )

    await fire_ibis_lifecycle(req.email, OperationType.AUTHENTICATE)

    obligations = _obligations_from_cohort(cohort)
    created_commitments: list[dict] = []
    submitted_attestations: list[dict] = []
    pending_obligations: list[dict] = []
    for ob in obligations:
        if ob["attestation_type"] in {"PUBLICATION", "RETURN_DATA"}:
            promise_type = "PUBLICATION_PROMISE" if ob["attestation_type"] == "PUBLICATION" else "RETURN_DATA_PROMISE"
            ar = await service.record_commitment_promise(
                requester_email=req.email,
                cohort_id=req.cohortId,
                att_type=promise_type,
            )
            if ar.get("success"):
                submitted_attestations.append({
                    "attestationId": ar.get("attestation_id"),
                    "type": ob["attestation_type"],
                    "sourceModifier": ob["source_modifier"],
                    "txHash": ar.get("tx_hash"),
                })
            else:
                pending_obligations.append({**ob, "attestation_error": ar.get("error")})
                continue

        cr = await service.create_commitment(
            researcher_email=req.email,
            cohort_id=req.cohortId,
            commitment_type=ob["type"],
            deadline_days=ob["deadline_days"],
            description=f"auto-created from {ob['source_modifier']} modifier on cohort {req.cohortId}",
        )
        if cr.get("success"):
            created_commitments.append({
                "commitmentId": cr.get("commitment_id"),
                "type": ob["type"],
                "sourceModifier": ob["source_modifier"],
                "txHash": cr.get("tx_hash"),
            })
        else:
            pending_obligations.append({**ob, "commitment_error": cr.get("error")})

    result = await service.request_access(
        requester_email=req.email,
        cohort_id=req.cohortId,
        intended_use=req.intendedUse,
        disease_code=req.diseaseCode or "",
        project_id=req.projectId or "",
        country_code=profile.get("country_code"),
        institution_id=profile.get("institution_id"),
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=502,
            detail={
                "matched": False,
                "reason": result.get("reason", "CHAIN_REVERTED"),
                "detail": result.get("reason_detail") or result.get("error", "tx reverted"),
                "cohortId": req.cohortId,
                "obligations": created_commitments,
                "attestations": submitted_attestations,
                "pendingObligations": pending_obligations,
            },
        )

    from api.services.wallet import get_cohort_hash
    cohort_hash = get_cohort_hash(req.cohortId).hex()
    approved = bool(result.get("auto_approved"))
    now = datetime.utcnow().isoformat() + "Z"
    principal = result.get("requester_address") or user.address

    await cache.set_access(cohort_hash, principal, {
        "approved": approved,
        "intended_use": req.intendedUse,
        "disease_code": req.diseaseCode,
        "project_id": req.projectId,
        "abstract": req.abstract,
        "requester_address": principal,
        "requester_eoa": user.address,
        "requester_hash": user.email_hash,
        "requested_at": now,
        "granted_at": now if approved else None,
        "request_id": result.get("request_id"),
        "tx_hash": result.get("tx_hash"),
        "status": "approved" if approved else "pending",
        "obligations_created": created_commitments,
        "pending_obligations": pending_obligations,
    })

    result["matched"] = True
    result["obligations"] = created_commitments
    result["attestations"] = submitted_attestations
    result["pendingObligations"] = pending_obligations
    return result

@router.get("/access-requests", response_model=list[AccessRequestStatus], summary="List the authenticated requester's own access requests with obligation status")
async def list_access_requests(user: AuthenticatedUser = Depends(get_current_user)):
    cache = get_cache()
    out: list[AccessRequestStatus] = []
    requester_info = get_blockchain_service().get_role_account_if_attached(user.address, "REQUESTER")
    principals = [user.address]
    if requester_info:
        principals.append(requester_info["account"])
    for c in await cache.get_all_consents():
        cohort_hash = (c.get("cohort_hash") or "").lstrip("0x")
        if not cohort_hash:
            continue
        grant = None
        for p in principals:
            grant = await cache.get_access(cohort_hash, p)
            if grant:
                break
        if not grant:
            continue
        out.append(AccessRequestStatus(
            requestId=grant.get("request_id", ""),
            cohortId=c.get("cohort_id", ""),
            status=grant.get("status", "approved" if grant.get("approved") else "pending"),
            autoApproved=bool(grant.get("approved")),
            complianceScore=grant.get("compliance_score"),
            decision=grant.get("decision"),
            requestedAt=grant.get("requested_at"),
            decidedAt=grant.get("granted_at") or grant.get("decided_at"),
            obligationsCreated=grant.get("obligations_created") or [],
            pendingObligations=grant.get("pending_obligations") or [],
        ))
    return out

@router.get("/access-credentials", response_model=list[AccessCredential], summary="List the soulbound access credential NFTs minted to the requester")
async def list_credentials(user: AuthenticatedUser = Depends(get_current_user)):
    service = get_blockchain_service()
    rows = await service.get_credentials_for_address(user.address) if hasattr(service, "get_credentials_for_address") else []
    return [
        AccessCredential(
            tokenId=str(r.get("token_id", "")),
            cohortId=r.get("cohort_id", ""),
            cohortHash=r.get("cohort_hash", ""),
            grantedAt=r.get("granted_at", ""),
            expiresAt=r.get("expires_at"),
            revoked=r.get("revoked", False),
        )
        for r in rows
    ]
