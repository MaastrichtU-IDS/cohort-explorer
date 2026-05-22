from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from api.services.auth import (
    AuthenticatedUser,
    fire_ibis_lifecycle,
    get_current_user,
    require_email_match,
)
from api.services.blockchain import get_blockchain_service
from api.services.cache import get_cache
from api.services.ibis import OperationType
from api.services.wallet import derive_address_from_email, get_cohort_hash

router = APIRouter(prefix="/attestations", tags=["attestations"])

VALID_TYPES = {
    "IRB_APPROVAL", "PUBLICATION", "RETURN_DATA", "DATA_DESTRUCTION",
    "COLLABORATION", "GEOGRAPHIC", "INSTITUTIONAL",
}

class AttestationCreate(BaseModel):
    email: EmailStr
    type: str
    cohortId: str
    documentHash: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    validDays: int = Field(0, ge=0)
    counterpartyEmail: Optional[EmailStr] = None

class AttestationResponse(BaseModel):
    success: bool
    type: str
    cohortId: str
    txHash: Optional[str] = None
    attestationId: Optional[str] = None
    easUid: Optional[str] = None
    validUntil: Optional[datetime] = None
    submittedAt: datetime

class AttestationRecord(BaseModel):
    type: str
    cohortId: str
    cohortHash: str
    valid: bool
    validUntil: Optional[str] = None
    submittedAt: Optional[str] = None
    revoked: bool = False
    metadata: dict = Field(default_factory=dict)
    easUid: Optional[str] = None

@router.post("", response_model=AttestationResponse, summary="Submit an attestation (IRB, COLLABORATION, etc.) for a subject + cohort")
async def create_attestation(req: AttestationCreate, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.email, "email")
    att_type = req.type.upper()
    if att_type not in VALID_TYPES:
        raise HTTPException(400, f"type must be one of {sorted(VALID_TYPES)}")

    await fire_ibis_lifecycle(req.email, OperationType.REGISTER)
    service = get_blockchain_service()
    cache = get_cache()
    subject_address = derive_address_from_email(req.email)
    cohort_hash = get_cohort_hash(req.cohortId).hex()
    submitted_at = datetime.utcnow()
    valid_until = (submitted_at + timedelta(days=req.validDays)) if req.validDays else None

    if att_type == "COLLABORATION":
        if not req.counterpartyEmail:
            raise HTTPException(400, "counterpartyEmail required for COLLABORATION")
        result = await service.establish_collaboration(
            owner_email=req.email,
            cohort_id=req.cohortId,
            requester_email=req.counterpartyEmail,
            valid_days=req.validDays,
        )
    else:
        result = await service.submit_attestation(
            subject_email=req.email,
            attestation_type=att_type,
            scope=cohort_hash,
            data_hash=req.documentHash,
            valid_days=req.validDays,
            metadata=req.metadata,
        )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "submit failed"))

    cache_record = {
        **req.metadata,
        "document_hash": req.documentHash,
        "valid_until": valid_until.timestamp() if valid_until else None,
        "submitted_at": submitted_at.isoformat(),
        "ea_uid": result.get("eas_uid"),
    }
    if att_type == "COLLABORATION":
        cache_record["counterparty_address"] = derive_address_from_email(req.counterpartyEmail)
    await cache.set_attestation(subject_address, att_type, cohort_hash, cache_record)

    return AttestationResponse(
        success=True,
        type=att_type,
        cohortId=req.cohortId,
        txHash=result.get("tx_hash"),
        attestationId=result.get("attestation_id"),
        easUid=result.get("eas_uid"),
        validUntil=valid_until,
        submittedAt=submitted_at,
    )

@router.get("/me", response_model=list[AttestationRecord], summary="List attestations where the authenticated user is the subject")
async def list_mine(user: AuthenticatedUser = Depends(get_current_user)):
    rows = await get_cache().get_subject_attestations(user.address)
    now = datetime.utcnow().timestamp()
    out = []
    for r in rows:
        valid_until_ts = r.get("valid_until")
        valid = not r.get("revoked", False)
        if valid_until_ts and valid_until_ts < now:
            valid = False
        out.append(AttestationRecord(
            type=r.get("type", ""),
            cohortId=r.get("cohort_id", ""),
            cohortHash=r.get("scope", ""),
            valid=valid,
            validUntil=datetime.fromtimestamp(valid_until_ts).isoformat() if valid_until_ts else None,
            submittedAt=r.get("submitted_at"),
            revoked=bool(r.get("revoked", False)),
            metadata={k: v for k, v in r.items() if k not in {"type", "scope", "valid_until", "submitted_at", "revoked", "ea_uid", "cached_at"}},
            easUid=r.get("ea_uid"),
        ))
    return out

@router.delete("", include_in_schema=False)
async def revoke_attestation(
    email: EmailStr,
    type: str,
    cohort_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    counterparty_email: Optional[EmailStr] = None,
):
    require_email_match(user, email, "email")
    att_type = type.upper()
    if att_type not in VALID_TYPES:
        raise HTTPException(400, f"type must be one of {sorted(VALID_TYPES)}")

    await fire_ibis_lifecycle(email, OperationType.RECOVER)
    service = get_blockchain_service()
    cohort_hash = get_cohort_hash(cohort_id).hex()

    if att_type == "COLLABORATION":
        if not counterparty_email:
            raise HTTPException(400, "counterparty_email required for COLLABORATION")
        result = await service.revoke_collaboration(
            owner_email=email, cohort_id=cohort_id, requester_email=counterparty_email,
        )
    else:
        result = await service.revoke_attestation(
            subject_email=email, attestation_type=att_type, scope=cohort_hash,
        )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "revoke failed"))
    return {"success": True, "type": att_type, "cohortId": cohort_id, "txHash": result.get("tx_hash")}
