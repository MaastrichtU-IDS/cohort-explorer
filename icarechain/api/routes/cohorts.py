import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api.models.duo import PERMISSION_LABELS, get_modifier_details
from api.services.blockchain import get_blockchain_service
from api.services.cache import get_cache
from api.services.compliance import get_compliance_service
from api.services.wallet import derive_address_from_email, get_cohort_hash

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cohorts", tags=["cohorts"])

_REQUIRES = {
    "GS": "requiresZk",
    "IS": "requiresZk",
    "IRB": "requiresIrb",
    "COL": "requiresCollaboration",
    "PUB": "requiresPublication",
    "RTN": "requiresReturnData",
}

class CohortDetail(BaseModel):
    cohortId: str
    cohortHash: str
    active: bool
    owners: list[str]
    permission: str
    permissionLabel: str
    modifiers: list[str]
    modifierDetails: list[dict]
    diseaseCode: Optional[str] = None
    additionalRestrictions: Optional[str] = None
    allowedCountries: list[str] = Field(default_factory=list)
    allowedInstitutions: list[str] = Field(default_factory=list)
    countriesMerkleRoot: Optional[str] = None
    institutionsMerkleRoot: Optional[str] = None
    validUntil: Optional[str] = None
    recordedAt: Optional[str] = None
    requiresZk: bool = False
    requiresIrb: bool = False
    requiresCollaboration: bool = False
    requiresPublication: bool = False
    requiresReturnData: bool = False
    consentTokenId: Optional[str] = None
    consentNftActive: Optional[bool] = None

class VerifyResult(BaseModel):
    cohortId: str
    requesterAddress: str
    hasAccess: bool
    accessType: Optional[str] = None
    grantedAt: Optional[str] = None
    reason: Optional[str] = None

async def _read(cohort_id: str) -> dict:
    cache = get_cache()
    h = get_cohort_hash(cohort_id).hex()
    record = await cache.get_consent(h) or await cache.get_consent("0x" + h)
    if record:
        return record

    chain = await get_blockchain_service().get_consent_status(cohort_id)
    if chain.get("error"):
        raise HTTPException(
            status_code=404,
            detail="Cohort does not yet have usage permissions specified",
        )
    return {
        "cohort_id": cohort_id,
        "cohort_hash": "0x" + h,
        "owners": chain.get("owners", []),
        "permission": chain.get("duo_permission", ""),
        "modifiers": chain.get("duo_modifiers", []),
        "disease_code": chain.get("disease_code"),
        "active": chain.get("active", False),
        "valid_until": chain.get("valid_until"),
    }

@router.get("/{cohort_id}", response_model=CohortDetail, summary="Public cohort detail incl. DUO permission, modifiers and capability flags")
async def get_cohort(cohort_id: str):
    record = await _read(cohort_id)
    perm = record.get("permission", "")
    modifiers = record.get("modifiers") or []

    flags = {flag: False for flag in _REQUIRES.values()}
    for m in modifiers:
        flag = _REQUIRES.get(m.upper())
        if flag:
            flags[flag] = True

    detail = CohortDetail(
        cohortId=record.get("cohort_id", cohort_id),
        cohortHash=record.get("cohort_hash") or ("0x" + get_cohort_hash(cohort_id).hex()),
        active=record.get("active", False),
        owners=record.get("owners", []),
        permission=perm,
        permissionLabel=PERMISSION_LABELS.get(perm, perm),
        modifiers=modifiers,
        modifierDetails=get_modifier_details(modifiers) if modifiers else [],
        diseaseCode=record.get("disease_code"),
        additionalRestrictions=record.get("additional_restrictions"),
        allowedCountries=record.get("allowed_countries") or [],
        allowedInstitutions=record.get("allowed_institutions") or [],
        countriesMerkleRoot=record.get("countries_merkle_root"),
        institutionsMerkleRoot=record.get("institutions_merkle_root"),
        validUntil=record.get("valid_until"),
        recordedAt=record.get("recorded_at"),
        **flags,
    )
    try:
        nft = await get_blockchain_service().get_consent_nft(cohort_id)
        if not nft.get("error"):
            detail.consentTokenId = str(nft.get("token_id")) if nft.get("token_id") else None
            detail.consentNftActive = nft.get("active")
    except Exception:
        pass
    return detail

@router.get("/{cohort_id}/verify", response_model=VerifyResult, summary="Public access check: does this email currently hold a valid grant on this cohort?")
async def verify_access(cohort_id: str, requester_email: str = Query(...)):
    eoa = derive_address_from_email(requester_email)

    svc = get_blockchain_service()
    requester_info = svc.get_role_account_if_attached(eoa, "REQUESTER")
    principal = requester_info["account"] if requester_info else eoa
    cohort_hash = get_cohort_hash(cohort_id).hex()
    result = await get_compliance_service().quick_access_check(cohort_hash, principal)
    if not result.get("has_access") and principal != eoa:

        result = await get_compliance_service().quick_access_check(cohort_hash, eoa)
        if result.get("has_access"):
            principal = eoa
    return VerifyResult(
        cohortId=cohort_id,
        requesterAddress=principal,
        hasAccess=result.get("has_access", False),
        accessType=result.get("access_type"),
        grantedAt=result.get("granted_at"),
        reason=result.get("reason"),
    )
