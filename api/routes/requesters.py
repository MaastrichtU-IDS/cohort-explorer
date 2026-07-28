import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from api.models.duo import PERMISSION_VALUES
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

_RESEARCH_PURPOSES = {
    "general": 1,
    "health": 2,
    "disease": 3,
    "population": 4,
    "methods": 5,
    "genetics": 6,
    "clinical": 7,
}
_PURPOSE_MODIFIERS = {"GSO", "NPOA", "NMDS"}

_OBLIGATION_MAP = {
    "PUB": ("PUBLICATION", "PUBLICATION", 730),
    "RTN": ("DATA_RETURN", "RETURN_DATA", 365),
    "COL": ("COLLABORATION", "COLLABORATION", 365),
    "MOR": ("MORATORIUM", None, 180),
}

_PROMISE_TYPES = {
    "PUBLICATION": "PUBLICATION_PROMISE",
    "RETURN_DATA": "RETURN_DATA_PROMISE",
    "COLLABORATION": "COLLABORATION_PROMISE",
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

    raw_inputs = list(req.diseaseCodes or [])
    if req.diseaseCode:
        raw_inputs.append(req.diseaseCode)
    raw_codes = icd10.split_codes(raw_inputs)
    canonical: list[str] = []
    for raw in raw_codes:
        if not isinstance(raw, str):
            return ("INVALID_DISEASE_CODE", f"diseaseCodes entries must be strings; got {type(raw).__name__}")
        d = icd10.normalize(raw)
        if not icd10.is_well_formed(d) or icd10.level(d) != "leaf":
            return (
                "INVALID_DISEASE_CODE",
                f"diseaseCodes must be specific ICD-10 disease codes (e.g. I50, I21.9, O24.4); "
                f"blocks/chapters are not allowed for requests (got {raw!r})",
            )
        if icd10.is_known_code(d):
            if not icd10.is_requester_leaf(d):
                kids = ", ".join(icd10.children(d))
                return (
                    "INVALID_DISEASE_CODE",
                    f"diseaseCode {raw!r} is not terminal in the supported hierarchy; "
                    f"use a more specific code ({kids})",
                )
            canonical.append(d)
        else:
            rolled = icd10.rollup_to_known(d)
            if rolled is None:
                return (
                    "INVALID_DISEASE_CODE",
                    f"diseaseCode {raw!r} is not in the supported ICD-10 hierarchy",
                )
            canonical.append(rolled)
    req.diseaseCodes = icd10.prune_redundant(canonical)
    if len(req.diseaseCodes) > 16:
        return ("INVALID_DISEASE_CODE", "at most 16 non-redundant disease codes allowed per request")
    req.diseaseCode = req.diseaseCodes[0] if req.diseaseCodes else None

    if req.projectId is not None:
        p = req.projectId.strip()
        if not _PROJECT_ID_RE.match(p):
            return (
                "INVALID_PROJECT_ID",
                f"projectId must be 2-64 chars, alphanumerics with . _ - allowed (got {req.projectId!r})",
            )
        req.projectId = p

    if use == "DS" and not req.diseaseCodes:
        return ("DISEASE_CODE_REQUIRED", "DS intent requires at least one diseaseCode")

    if req.researchPurpose is not None:
        p = req.researchPurpose.strip().lower()
        if p not in _RESEARCH_PURPOSES:
            return (
                "INVALID_RESEARCH_PURPOSE",
                f"researchPurpose must be one of {sorted(_RESEARCH_PURPOSES)}; got {req.researchPurpose!r}",
            )
        req.researchPurpose = p

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
    diseaseCodes: list[str] = []
    additionalRestrictions: Optional[str] = None
    active: bool
    requiresProject: bool = False
    requiresResearchPurpose: bool = False

class AccessRequestCreate(BaseModel):
    email: EmailStr
    cohortId: str
    intendedUse: str
    diseaseCode: Optional[str] = None
    diseaseCodes: Optional[list[str]] = None
    researchPurpose: Optional[str] = None
    projectId: Optional[str] = None
    abstract: Optional[str] = None

class AccessRequestStatus(BaseModel):
    requestId: str
    cohortId: str
    status: str
    autoApproved: bool = False
    complianceScore: Optional[int] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
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
            diseaseCodes=c.get("disease_codes") or ([c["disease_code"]] if c.get("disease_code") else []),
            additionalRestrictions=c.get("additional_restrictions"),
            active=c.get("active", False),
            requiresProject="PS" in {m.upper() for m in (c.get("modifiers") or [])},
            requiresResearchPurpose=bool(_PURPOSE_MODIFIERS & {m.upper() for m in (c.get("modifiers") or [])}),
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
        raise HTTPException(404, "Cohort not found")

    requested_codes = sorted({
        icd10.normalize(c)
        for c in icd10.split_codes([*(req.diseaseCodes or []), *([req.diseaseCode] if req.diseaseCode else [])])
        if isinstance(c, str) and icd10.normalize(c)
    })
    format_error = _validate_request_format(req)
    if format_error is not None:
        code, detail = format_error
        raise HTTPException(
            status_code=422,
            detail={"matched": False, "reason": code, "detail": detail, "cohortId": req.cohortId},
        )

    mods = {m.upper() for m in (cohort.get("modifiers") or [])}

    if (_PURPOSE_MODIFIERS & mods) and not req.researchPurpose:
        raise HTTPException(
            status_code=422,
            detail={
                "matched": False,
                "reason": "RESEARCH_PURPOSE_REQUIRED",
                "detail": f"cohort modifiers {sorted(_PURPOSE_MODIFIERS & mods)} require a researchPurpose "
                          f"(one of {sorted(_RESEARCH_PURPOSES)})",
                "cohortId": req.cohortId,
            },
        )

    if "PS" in mods and not req.projectId:
        raise HTTPException(
            status_code=422,
            detail={
                "matched": False,
                "reason": "PROJECT_REQUIRED",
                "detail": "PS modifier on cohort: projectId is required from requester",
                "cohortId": req.cohortId,
            },
        )

    service = get_blockchain_service()

    await service.ensure_requester_type_onchain(
        req.email,
        profile.get("requester_type"),
        profile.get("country_code"),
    )

    await fire_ibis_lifecycle(req.email, OperationType.AUTHENTICATE)

    obligations = _obligations_from_cohort(cohort)
    created_commitments: list[dict] = []
    submitted_attestations: list[dict] = []
    pending_obligations: list[dict] = []
    for ob in obligations:
        promise_type = _PROMISE_TYPES.get(ob["attestation_type"] or "")
        if promise_type:
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

    purpose_int = _RESEARCH_PURPOSES.get(req.researchPurpose or "general", 1)

    result = await service.request_access(
        requester_email=req.email,
        cohort_id=req.cohortId,
        intended_use=req.intendedUse,
        purpose=purpose_int,
        disease_codes=req.diseaseCodes,
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
    approved = bool(result.get("matched"))
    decision = result.get("decision") or ("approved" if approved else "rejected")
    reason = result.get("reason")
    now = datetime.utcnow().isoformat() + "Z"
    principal = result.get("requester_address") or user.address

    await cache.set_access(cohort_hash, principal, {
        "approved": approved,
        "intended_use": req.intendedUse,
        "research_purpose": req.researchPurpose or "general",
        "disease_codes": req.diseaseCodes,
        "disease_code": req.diseaseCode,
        "disease_codes_requested": requested_codes,
        "project_id": req.projectId,
        "abstract": req.abstract,
        "requester_address": principal,
        "requester_eoa": user.address,
        "requester_hash": user.email_hash,
        "requested_at": now,
        "granted_at": now if approved else None,
        "decided_at": now,
        "request_id": result.get("request_id"),
        "tx_hash": result.get("tx_hash"),
        "status": decision,
        "decision": decision,
        "reason": reason,
        "reason_detail": result.get("reason_detail"),
        "obligations_created": created_commitments,
        "pending_obligations": pending_obligations,
    })

    result["matched"] = approved
    result["cohortId"] = req.cohortId
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
            reason=grant.get("reason"),
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
