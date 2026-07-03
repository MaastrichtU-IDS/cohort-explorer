import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field, model_validator

from api.models.duo import MODIFIER_VALUES, PERMISSION_VALUES
from api.services.iso3166 import all_codes as _iso_country_codes

_ROR_RE = re.compile(r"^https?://ror\.org/[0-9a-z]{6,12}$")
_ETH_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{1,63}$")
_ISO_CODES = set(_iso_country_codes())
_URI_RE = re.compile(r"^(https?|ipfs|ar|s3|file)://\S+$", re.IGNORECASE)
from api.models.requests import (
    AddOwnerRequest,
    ConsentRevokeRequest,
    ConsentSummary,
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
from api.services.wallet import derive_address_from_email, get_cohort_hash

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/providers", tags=["providers"])

class ConsentDeclaration(BaseModel):
    permission: str = Field(..., description="Primary DUO permission: NRES | GRU | HMB | DS | POA")
    modifiers: list[str] = Field(default_factory=list, description="DUO modifier codes")

    diseaseCode: Optional[str] = Field(None, description="ICD-10 code at any level: category (I50), block (I30-I52) or chapter (I00-I99 / IX); required when permission == DS")
    allowedCountries: list[str] = Field(default_factory=list, description="ISO-3166 codes; required when GS modifier set")
    allowedInstitutions: list[str] = Field(default_factory=list, description="ROR IDs; required when IS modifier set")
    allowedProjects: list[str] = Field(default_factory=list, description="Project IDs; required when PS modifier set")
    allowedUsers: list[str] = Field(default_factory=list, description="User addresses or email hashes; required when US modifier set")
    moratoriumMonths: Optional[int] = Field(None, ge=1, description="Months of publication moratorium; required when MOR modifier set")
    researchScope: Optional[str] = Field(None, description="Free-text research scope; required when RS modifier set")
    returnTargetUri: Optional[str] = Field(None, description="Target URI to return derived data; required when RTN modifier set")
    publicationDeadlineDays: Optional[int] = Field(None, ge=1, description="Days for PUB obligation; optional when PUB modifier set")

    expirationDays: int = Field(0, ge=0, description="Days until consent expires; 0 = no expiry")

    dataUseDescription: Optional[str] = Field(None, description="Free-text description of what the data may be used for")
    additionalRestrictions: Optional[str] = Field(None, description="Free-text restrictions beyond modifier list")
    consentDate: Optional[date] = Field(None, description="Date the original consent was obtained from data subjects")
    consentFormUri: Optional[str] = Field(None, description="URI of the signed consent form")
    metadataUri: str = Field("", description="Generic metadata URI")

    @model_validator(mode="after")
    def _check_duo(self):
        perm = self.permission.upper()
        if perm not in PERMISSION_VALUES:
            raise ValueError(f"unknown permission '{self.permission}'; expected one of {sorted(PERMISSION_VALUES)}")
        self.permission = perm
        bad = [m for m in self.modifiers if m.upper() not in MODIFIER_VALUES]
        if bad:
            raise ValueError(f"unknown modifiers: {bad}; expected subset of {sorted(MODIFIER_VALUES)}")
        self.modifiers = sorted({m.upper() for m in self.modifiers})

        mods = set(self.modifiers)

        if {"NPU", "NCU"} <= mods and "NPUNCU" not in mods:
            mods.add("NPUNCU")
            self.modifiers = sorted(mods)

        if perm == "DS" and not self.diseaseCode:
            raise ValueError("DS permission requires diseaseCode (an ICD-10 code, e.g. I50, I30-I52 or IX)")
        if "GS" in mods and not self.allowedCountries:
            raise ValueError("GS modifier requires allowedCountries (ISO-3166)")
        if "IS" in mods and not self.allowedInstitutions:
            raise ValueError("IS modifier requires allowedInstitutions (ROR IDs)")
        if "PS" in mods and not self.allowedProjects:
            raise ValueError("PS modifier requires allowedProjects")
        if "US" in mods and not self.allowedUsers:
            raise ValueError("US modifier requires allowedUsers")
        if "MOR" in mods and self.moratoriumMonths is None:
            raise ValueError("MOR modifier requires moratoriumMonths")
        if "TS" in mods and self.expirationDays <= 0:
            raise ValueError("TS modifier requires expirationDays > 0")
        if "RS" in mods and not (self.researchScope and self.researchScope.strip()):
            raise ValueError("RS modifier requires researchScope (free text describing allowed scope)")
        if "RTN" in mods and not self.returnTargetUri:
            raise ValueError("RTN modifier requires returnTargetUri")

        if self.diseaseCode is not None:
            code = icd10.normalize(self.diseaseCode)
            if not icd10.is_well_formed(code):
                raise ValueError(
                    f"diseaseCode must be an ICD-10 code: a category (I50), block (I30-I52) or chapter (I00-I99 / IX) (got {self.diseaseCode!r})"
                )
            if not icd10.is_known_code(code):
                raise ValueError(
                    f"diseaseCode {self.diseaseCode!r} is not a supported ICD-10 code"
                )
            self.diseaseCode = code

        if self.allowedCountries:
            normalized = []
            for c in self.allowedCountries:
                if not isinstance(c, str):
                    raise ValueError(f"allowedCountries: expected string, got {type(c).__name__}")
                cc = c.strip().upper()
                if cc not in _ISO_CODES:
                    raise ValueError(f"allowedCountries: unknown ISO-3166-1 alpha-2 code {c!r}")
                normalized.append(cc)
            self.allowedCountries = sorted(set(normalized))

        if self.allowedInstitutions:
            normalized = []
            for inst in self.allowedInstitutions:
                if not isinstance(inst, str):
                    raise ValueError(f"allowedInstitutions: expected string, got {type(inst).__name__}")
                s = inst.strip()
                if not _ROR_RE.match(s):
                    raise ValueError(
                        f"allowedInstitutions: must be a ROR URL like https://ror.org/01ej9dk98 (got {inst!r})"
                    )
                normalized.append(s)
            self.allowedInstitutions = sorted(set(normalized))

        if self.allowedProjects:
            normalized = []
            for p in self.allowedProjects:
                if not isinstance(p, str):
                    raise ValueError(f"allowedProjects: expected string, got {type(p).__name__}")
                s = p.strip()
                if not _PROJECT_ID_RE.match(s):
                    raise ValueError(
                        f"allowedProjects: must be 2-64 chars, alphanumerics with . _ - allowed (got {p!r})"
                    )
                normalized.append(s)
            self.allowedProjects = sorted(set(normalized))

        if self.allowedUsers:
            normalized = []
            for u in self.allowedUsers:
                if not isinstance(u, str):
                    raise ValueError(f"allowedUsers: expected string, got {type(u).__name__}")
                s = u.strip()
                if _ETH_ADDR_RE.match(s):
                    normalized.append(s)
                elif _HEX64_RE.match(s):
                    normalized.append(s.lower().removeprefix("0x"))
                else:
                    raise ValueError(
                        f"allowedUsers: each entry must be a 0x-prefixed 40-hex eth address or a 64-hex email hash (got {u!r})"
                    )
            self.allowedUsers = sorted(set(normalized))

        for field_name, value in (
            ("returnTargetUri", self.returnTargetUri),
            ("consentFormUri", self.consentFormUri),
            ("metadataUri", self.metadataUri),
        ):
            if value and not _URI_RE.match(value.strip()):
                raise ValueError(f"{field_name}: must be a URI (https://, ipfs://, ...); got {value!r}")

        return self

class CreateCohortRequest(BaseModel):
    email: EmailStr
    cohortId: str
    consent: ConsentDeclaration

class CohortSummary(BaseModel):
    cohortId: str
    cohortHash: str
    permission: str
    modifiers: list[str]
    diseaseCode: Optional[str] = None
    allowedCountries: list[str] = []
    allowedInstitutions: list[str] = []
    allowedProjects: list[str] = []
    moratoriumMonths: Optional[int] = None
    active: bool
    validUntil: Optional[str] = None
    recordedAt: Optional[str] = None

class CreateCohortResponse(BaseModel):
    success: bool
    cohortId: str
    cohortHash: str
    txHash: str
    ownerAddress: str
    consent: ConsentSummary

class AccessRequestSummary(BaseModel):
    requestId: str
    cohortId: str
    requesterAddress: str
    intendedUse: Optional[str] = None
    diseaseCode: Optional[str] = None
    projectId: Optional[str] = None
    abstract: Optional[str] = None
    status: str
    requestedAt: Optional[str] = None
    grantedAt: Optional[str] = None

class DecisionRequest(BaseModel):
    email: EmailStr
    decision: str = Field(..., pattern="^(approve|deny)$")
    reason: Optional[str] = None

def _provider_principal(user: AuthenticatedUser) -> str | None:
    info = get_blockchain_service().get_role_account_if_attached(user.address, "PROVIDER")
    return info["account"].lower() if info else None

def _own_or_403(user: AuthenticatedUser, c: dict, cohort_id: str = "") -> None:
    owners = [a.lower() for a in c.get("owners", [])]
    principal = _provider_principal(user)
    eoa = user.address.lower()
    if principal in owners or eoa in owners:
        return
    cohort_name = cohort_id or c.get("cohort_id", "this cohort")
    raise HTTPException(403, f"Only the consent owner of '{cohort_name}' can perform this action")

async def _load_cohort(cache, cohort_id: str) -> dict:
    h = get_cohort_hash(cohort_id).hex()
    return await cache.get_consent(h) or await cache.get_consent("0x" + h)

def _maybe_int(v):
    if v in (None, "", "None"):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

def _to_summary(c: dict, fallback_id: str = "") -> CohortSummary:
    return CohortSummary(
        cohortId=c.get("cohort_id", fallback_id),
        cohortHash=c.get("cohort_hash", ""),
        permission=c.get("permission", ""),
        modifiers=c.get("modifiers", []),
        diseaseCode=c.get("disease_code") or None,
        allowedCountries=c.get("allowed_countries") or [],
        allowedInstitutions=c.get("allowed_institutions") or [],
        allowedProjects=c.get("allowed_projects") or [],
        moratoriumMonths=_maybe_int(c.get("moratorium_months")),
        active=bool(c.get("active", False)),
        validUntil=c.get("valid_until") or None,
        recordedAt=c.get("recorded_at") or None,
    )

@router.post("/consents", response_model=CreateCohortResponse, summary="Record consent for a cohort")
async def create_consent(req: CreateCohortRequest, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.email, "email")
    await fire_ibis_lifecycle(req.email, OperationType.REGISTER)

    consent = req.consent
    service = get_blockchain_service()
    cache = get_cache()

    if await service.consent_exists_on_chain(req.cohortId):
        raise HTTPException(409, f"Consent for cohortId '{req.cohortId}' already exists. Use PATCH to update.")

    metadata_blob = {
        "dataUseDescription": consent.dataUseDescription,
        "additionalRestrictions": consent.additionalRestrictions,
        "researchScope": consent.researchScope,
        "returnTargetUri": consent.returnTargetUri,
        "consentDate": consent.consentDate.isoformat() if consent.consentDate else None,
        "consentFormUri": consent.consentFormUri,
        "metadataUri": consent.metadataUri,
    }
    result = await service.record_consent(
        owner_email=req.email,
        cohort_id=req.cohortId,
        permission=consent.permission,
        modifiers=consent.modifiers,
        disease_code=consent.diseaseCode or "",
        allowed_countries=consent.allowedCountries,
        allowed_institutions=consent.allowedInstitutions,
        allowed_projects=consent.allowedProjects,
        allowed_users=consent.allowedUsers,
        moratorium_months=consent.moratoriumMonths or 0,
        publication_deadline_days=consent.publicationDeadlineDays or 0,
        expiration_days=consent.expirationDays,
        metadata=metadata_blob,
        signature=None,
    )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "record_consent failed"))

    cohort_hash = result["cohort_hash"]
    expires_at = (datetime.utcnow() + timedelta(days=consent.expirationDays)) if consent.expirationDays > 0 else None

    await cache.set_consent(cohort_hash, {
        "cohort_id": req.cohortId,
        "cohort_hash": cohort_hash,
        "owners": [result["owner_address"]],
        "permission": consent.permission,
        "modifiers": consent.modifiers,
        "disease_code": consent.diseaseCode,
        "allowed_countries": consent.allowedCountries,
        "allowed_institutions": consent.allowedInstitutions,
        "allowed_projects": consent.allowedProjects,
        "allowed_users": consent.allowedUsers,
        "moratorium_months": consent.moratoriumMonths,
        "research_scope": consent.researchScope,
        "return_target_uri": consent.returnTargetUri,
        "publication_deadline_days": consent.publicationDeadlineDays,
        "data_use_description": consent.dataUseDescription,
        "additional_restrictions": consent.additionalRestrictions,
        "consent_date": consent.consentDate.isoformat() if consent.consentDate else None,
        "consent_form_uri": consent.consentFormUri,
        "metadata_uri": consent.metadataUri,
        "active": True,
        "valid_until": expires_at.isoformat() + "Z" if expires_at else None,
        "recorded_at": datetime.utcnow().isoformat() + "Z",
    })

    await cache.store_transaction(result["tx_hash"], {
        "type": "consent",
        "cohort_id": req.cohortId,
        "cohort_hash": cohort_hash,
        "owner_hash": user.email_hash,
        "owner_address": result["owner_address"],
        "permission": consent.permission,
        "modifiers": consent.modifiers,
        "tx_success": True,
        "status": "success",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    try:
        await service.mint_consent_nft(
            owner_email=req.email,
            cohort_id=req.cohortId,
            permission=consent.permission,
            modifiers=consent.modifiers,
            disease_code=consent.diseaseCode or "",
            valid_days=consent.expirationDays,
            metadata_uri=consent.metadataUri,
        )
    except Exception as e:
        logger.warning(f"NFT mint failed: {e}")

    return CreateCohortResponse(
        success=True,
        cohortId=req.cohortId,
        cohortHash=cohort_hash,
        txHash=result["tx_hash"],
        ownerAddress=result["owner_address"],
        consent=ConsentSummary(
            permission=result["permission"],
            permission_label=result["permission_label"],
            modifiers=result["modifiers"],
            modifier_details=result["modifier_details"],
            expires_at=expires_at,
        ),
    )

@router.get("/consents", response_model=list[CohortSummary], summary="List consents owned by the authenticated PROVIDER")
async def list_my_consents(user: AuthenticatedUser = Depends(get_current_user)):
    consents = await get_cache().get_all_consents()
    principal = _provider_principal(user)
    eoa = user.address.lower()
    mine = [
        c for c in consents
        if eoa in [a.lower() for a in c.get("owners", [])]
        or (principal and principal in [a.lower() for a in c.get("owners", [])])
    ]
    return [_to_summary(c) for c in mine]

@router.get("/consents/{cohort_id}", response_model=CohortSummary, summary="Get consent for a cohort")
async def get_my_consent(cohort_id: str, user: AuthenticatedUser = Depends(get_current_user)):
    c = await _load_cohort(get_cache(), cohort_id)
    if not c:
        raise HTTPException(404, f"No consent record exists for cohort '{cohort_id}'")
    _own_or_403(user, c, cohort_id)
    return _to_summary(c, fallback_id=cohort_id)

@router.patch("/consents/{cohort_id}", summary="Update consent terms for a cohort")
async def update_consent(cohort_id: str, req: CreateCohortRequest, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.email, "email")
    if req.cohortId != cohort_id:
        raise HTTPException(400, "cohortId in path and body must match")

    service = get_blockchain_service()
    cache = get_cache()

    if not await service.consent_exists_on_chain(cohort_id):
        raise HTTPException(404, f"No consent record exists for cohort '{cohort_id}'")

    c = await _load_cohort(cache, cohort_id)
    if c:
        _own_or_403(user, c, cohort_id)

    await fire_ibis_lifecycle(req.email, OperationType.RENEW)
    consent = req.consent

    metadata_blob = {
        "dataUseDescription": consent.dataUseDescription,
        "additionalRestrictions": consent.additionalRestrictions,
        "researchScope": consent.researchScope,
        "returnTargetUri": consent.returnTargetUri,
        "consentDate": consent.consentDate.isoformat() if consent.consentDate else None,
        "consentFormUri": consent.consentFormUri,
        "metadataUri": consent.metadataUri,
    }
    result = await service.record_consent(
        owner_email=req.email,
        cohort_id=req.cohortId,
        permission=consent.permission,
        modifiers=consent.modifiers,
        disease_code=consent.diseaseCode or "",
        allowed_countries=consent.allowedCountries,
        allowed_institutions=consent.allowedInstitutions,
        allowed_projects=consent.allowedProjects,
        allowed_users=consent.allowedUsers,
        moratorium_months=consent.moratoriumMonths or 0,
        publication_deadline_days=consent.publicationDeadlineDays or 0,
        expiration_days=consent.expirationDays,
        metadata=metadata_blob,
        signature=None,
    )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "record_consent failed"))

    cohort_hash = result["cohort_hash"]
    expires_at = (datetime.utcnow() + timedelta(days=consent.expirationDays)) if consent.expirationDays > 0 else None
    now = datetime.utcnow().isoformat() + "Z"

    existing_owners = c.get("owners", [result["owner_address"]]) if c else [result["owner_address"]]
    recorded_at = (c or {}).get("recorded_at") or now

    await cache.set_consent(cohort_hash, {
        "cohort_id": req.cohortId,
        "cohort_hash": cohort_hash,
        "owners": existing_owners,
        "permission": consent.permission,
        "modifiers": consent.modifiers,
        "disease_code": consent.diseaseCode,
        "allowed_countries": consent.allowedCountries,
        "allowed_institutions": consent.allowedInstitutions,
        "allowed_projects": consent.allowedProjects,
        "allowed_users": consent.allowedUsers,
        "moratorium_months": consent.moratoriumMonths,
        "research_scope": consent.researchScope,
        "return_target_uri": consent.returnTargetUri,
        "publication_deadline_days": consent.publicationDeadlineDays,
        "data_use_description": consent.dataUseDescription,
        "additional_restrictions": consent.additionalRestrictions,
        "consent_date": consent.consentDate.isoformat() if consent.consentDate else None,
        "consent_form_uri": consent.consentFormUri,
        "metadata_uri": consent.metadataUri,
        "active": True,
        "valid_until": expires_at.isoformat() + "Z" if expires_at else None,
        "recorded_at": recorded_at,
        "updated_at": now,
    })

    await cache.store_transaction(result["tx_hash"], {
        "type": "consent_update",
        "cohort_id": req.cohortId,
        "cohort_hash": cohort_hash,
        "owner_hash": user.email_hash,
        "owner_address": result["owner_address"],
        "permission": consent.permission,
        "modifiers": consent.modifiers,
        "tx_success": True,
        "status": "success",
        "timestamp": now,
    })

    return {
        "success": True,
        "cohortId": cohort_id,
        "cohortHash": cohort_hash,
        "txHash": result["tx_hash"],
        "ownerAddress": result["owner_address"],
        "permission": consent.permission,
        "modifiers": consent.modifiers,
    }

@router.post("/consents/{cohort_id}/revoke", summary="Revoke consent for a cohort")
async def revoke_consent(cohort_id: str, req: ConsentRevokeRequest, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.owner_email, "owner_email")
    if req.cohort_id != cohort_id:
        raise HTTPException(400, "cohort_id in path and body must match")

    cache = get_cache()
    c = await _load_cohort(cache, cohort_id)
    if not c:
        raise HTTPException(404, f"No consent record exists for cohort '{cohort_id}'")
    _own_or_403(user, c, cohort_id)

    await fire_ibis_lifecycle(req.owner_email, OperationType.RECOVER)
    result = await get_blockchain_service().revoke_consent(owner_email=req.owner_email, cohort_id=cohort_id)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "revoke_consent failed"))

    c["active"] = False
    c["revoked_at"] = datetime.utcnow().isoformat() + "Z"
    await cache.set_consent(c.get("cohort_hash") or get_cohort_hash(cohort_id).hex(), c)
    return result

@router.post("/consents/{cohort_id}/owners", summary="Add a co-owner to the consent")
async def add_co_owner(cohort_id: str, req: AddOwnerRequest, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.owner_email, "owner_email")
    if req.cohort_id != cohort_id:
        raise HTTPException(400, "cohort_id in path and body must match")

    cache = get_cache()
    c = await _load_cohort(cache, cohort_id)
    if not c:
        raise HTTPException(404, f"No consent record exists for cohort '{cohort_id}'")
    _own_or_403(user, c, cohort_id)

    await fire_ibis_lifecycle(req.owner_email, OperationType.RENEW)
    result = await get_blockchain_service().add_owner(
        owner_email=req.owner_email, cohort_id=cohort_id, new_owner_email=req.new_owner_email,
    )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "add_owner failed"))

    new_addr = derive_address_from_email(req.new_owner_email)
    if new_addr.lower() not in [a.lower() for a in c.get("owners", [])]:
        c.setdefault("owners", []).append(new_addr)
        await cache.set_consent(c.get("cohort_hash") or get_cohort_hash(cohort_id).hex(), c)
    return result

@router.get("/consents/{cohort_id}/access-requests", response_model=list[AccessRequestSummary], summary="List access requests against the consent")
async def list_access_requests(cohort_id: str, user: AuthenticatedUser = Depends(get_current_user)):
    cache = get_cache()
    c = await _load_cohort(cache, cohort_id)
    if not c:
        raise HTTPException(404, f"No consent record exists for cohort '{cohort_id}'")
    _own_or_403(user, c, cohort_id)

    cohort_hash = (c.get("cohort_hash") or get_cohort_hash(cohort_id).hex()).lstrip("0x")
    grants = await cache.get_cohort_access_grants(cohort_hash) if hasattr(cache, "get_cohort_access_grants") else []
    return [
        AccessRequestSummary(
            requestId=g.get("request_id", ""),
            cohortId=cohort_id,
            requesterAddress=g.get("requester_address") or g.get("requester", ""),
            intendedUse=g.get("intended_use"),
            diseaseCode=g.get("disease_code"),
            projectId=g.get("project_id"),
            abstract=g.get("abstract"),
            status=g.get("status", "approved" if g.get("approved") else "pending"),
            requestedAt=g.get("requested_at"),
            grantedAt=g.get("granted_at"),
        )
        for g in grants
    ]

@router.post("/consents/{cohort_id}/access-requests/{request_id}/decision", summary="Approve or deny an access request")
async def decide_access_request(
    cohort_id: str,
    request_id: str,
    req: DecisionRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    require_email_match(user, req.email, "email")
    cache = get_cache()
    c = await _load_cohort(cache, cohort_id)
    if not c:
        raise HTTPException(404, f"No consent record exists for cohort '{cohort_id}'")
    _own_or_403(user, c, cohort_id)
    await fire_ibis_lifecycle(req.email, OperationType.RENEW)

    if hasattr(cache, "set_access_request_decision"):
        await cache.set_access_request_decision(
            request_id=request_id,
            decision=req.decision,
            reason=req.reason,
            decided_by_hash=user.email_hash,
            decided_at=datetime.utcnow().isoformat() + "Z",
        )
    return {"success": True, "requestId": request_id, "decision": req.decision, "decidedBy": user.email_hash}
