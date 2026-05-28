from datetime import datetime
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
from api.services.ibis import OperationType

router = APIRouter(prefix="/commitments", tags=["commitments"])

VALID_TYPES = {"PUBLICATION", "DATA_RETURN", "COLLABORATION", "MORATORIUM"}
DEFAULT_DEADLINE_DAYS = {
    "PUBLICATION": 730,
    "DATA_RETURN": 365,
    "COLLABORATION": 0,
    "MORATORIUM": 180,
}

class CreateCommitment(BaseModel):
    email: EmailStr
    cohortId: str
    type: str
    deadlineDays: Optional[int] = None
    description: str = ""

class CommitmentResponse(BaseModel):
    success: bool
    commitmentId: Optional[str] = None
    txHash: Optional[str] = None
    type: str
    deadline: Optional[datetime] = None

class FulfillCommitment(BaseModel):
    email: EmailStr
    commitmentId: str
    evidenceUri: Optional[str] = None
    evidenceHash: Optional[str] = None

class FulfillResponse(BaseModel):
    success: bool
    commitmentId: str
    txHash: Optional[str] = None
    fulfilledAt: datetime

class CommitmentDetail(BaseModel):
    commitmentId: str
    cohortId: str
    type: str
    description: str
    status: str
    createdAt: Optional[datetime] = None
    deadline: Optional[datetime] = None
    daysRemaining: Optional[int] = None
    evidenceUri: Optional[str] = None
    fulfilledAt: Optional[datetime] = None

def _to_detail(c: dict, cid: str = "") -> CommitmentDetail:
    deadline = datetime.fromtimestamp(c["deadline"]) if c.get("deadline") else None
    days_remaining = None
    if c.get("status") == "ACTIVE" and deadline:
        days_remaining = max(0, (deadline - datetime.utcnow()).days)
    return CommitmentDetail(
        commitmentId=c.get("commitment_id", cid),
        cohortId=c.get("cohort_id", c.get("cohort_hash", "")),
        type=c.get("commitment_type", ""),
        description=c.get("description", ""),
        status=c.get("status", "UNKNOWN"),
        createdAt=datetime.fromtimestamp(c["created_at"]) if c.get("created_at") else None,
        deadline=deadline,
        daysRemaining=days_remaining,
        evidenceUri=c.get("evidence_uri"),
        fulfilledAt=datetime.fromtimestamp(c["fulfilled_at"]) if c.get("fulfilled_at") else None,
    )

@router.post("", response_model=CommitmentResponse, summary="Create an ad-hoc deliverable commitment (PUB/RTN/MOR/COL) for a cohort")
async def create_commitment(req: CreateCommitment, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, req.email, "email")
    t = req.type.upper()
    if t not in VALID_TYPES:
        raise HTTPException(400, f"type must be one of {sorted(VALID_TYPES)}")

    deadline_days = req.deadlineDays if req.deadlineDays is not None else DEFAULT_DEADLINE_DAYS[t]
    await fire_ibis_lifecycle(req.email, OperationType.REGISTER)

    service = get_blockchain_service()
    result = await service.create_commitment(
        researcher_email=req.email,
        cohort_id=req.cohortId,
        commitment_type=req.type,
        deadline_days=deadline_days,
        description=req.description,
    )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "create failed"))

    deadline = None
    if result.get("commitment_id"):
        rec = await service.get_commitment(result["commitment_id"])
        if rec.get("deadline"):
            deadline = datetime.fromtimestamp(rec["deadline"])

    return CommitmentResponse(
        success=True,
        commitmentId=result.get("commitment_id"),
        txHash=result.get("tx_hash"),
        type=t,
        deadline=deadline,
    )

@router.post("/{commitment_id}/fulfill", response_model=FulfillResponse, summary="Mark a commitment fulfilled with on-chain evidence URI / hash")
async def fulfill(
    commitment_id: str,
    req: FulfillCommitment,
    user: AuthenticatedUser = Depends(get_current_user),
):
    require_email_match(user, req.email, "email")
    if req.commitmentId != commitment_id:
        raise HTTPException(400, "commitmentId mismatch")
    if not req.evidenceUri and not req.evidenceHash:
        raise HTTPException(400, "evidenceUri or evidenceHash required")

    await fire_ibis_lifecycle(req.email, OperationType.RENEW)
    result = await get_blockchain_service().fulfill_commitment(
        commitment_id=commitment_id,
        evidence_hash=req.evidenceHash or "",
        evidence_uri=req.evidenceUri or "",
    )
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "fulfill failed"))
    return FulfillResponse(
        success=True, commitmentId=commitment_id, txHash=result.get("tx_hash"), fulfilledAt=datetime.utcnow(),
    )

@router.get("/me", response_model=list[CommitmentDetail], summary="List commitments owned by the authenticated researcher")
async def list_mine(email: EmailStr, user: AuthenticatedUser = Depends(get_current_user)):
    require_email_match(user, email, "email")
    result = await get_blockchain_service().get_researcher_commitments(email)
    rows = result.get("commitments", []) if isinstance(result, dict) else result
    return [_to_detail(c) for c in rows]

@router.get("/{commitment_id}", response_model=CommitmentDetail, summary="Public read of a single commitment by id")
async def get_commitment(commitment_id: str):
    c = await get_blockchain_service().get_commitment(commitment_id)
    if c.get("error"):
        raise HTTPException(404, "Commitment not found")
    return _to_detail(c, cid=commitment_id)
