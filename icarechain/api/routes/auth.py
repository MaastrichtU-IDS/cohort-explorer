import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, EmailStr, Field

from api.services.auth import (
    AuthenticatedUser,
    _hash_email,
    fire_ibis_lifecycle,
    generate_otp,
    get_current_user,
    is_registered,
    verify_otp,
)
from api.services.blockchain import get_blockchain_service
from api.services.cache import get_cache
from api.services.ibis import OperationType
from api.services.wallet import derive_address_from_email

router = APIRouter(prefix="/auth", tags=["auth"])

DEV_MODE = os.environ.get("AUTH_DEV_MODE", "true").lower() in ("true", "1", "yes")
VALID_ROLES = {"PROVIDER", "REQUESTER"}

class RoleAccountInfo(BaseModel):
    commitment: str
    account: str

class RegisterRequest(BaseModel):
    email: EmailStr
    roles: list[str] = Field(default_factory=list, description="Optional: PROVIDER, REQUESTER, or both")

class RegisterResponse(BaseModel):
    emailHash: str
    address: str
    roles: dict[str, RoleAccountInfo] = Field(default_factory=dict)
    otpCode: Optional[str] = None

class VerifyRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)

class VerifyResponse(BaseModel):
    token: str
    emailHash: str
    address: str

class RenewRequest(BaseModel):
    email: EmailStr

class RecoverRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)

class CommitmentBrief(BaseModel):
    commitmentId: str
    cohortHash: str
    type: str
    status: str
    deadline: Optional[int] = None

class AttestationBrief(BaseModel):
    type: str
    cohortId: Optional[str] = None
    cohortHash: Optional[str] = None
    valid: bool
    validUntil: Optional[str] = None

class PendingAccessRequestBrief(BaseModel):
    requestId: str
    cohortId: str
    requesterAddress: str
    requestedAt: Optional[str] = None
    intendedUse: Optional[str] = None

class Dashboard(BaseModel):
    pendingAccessRequests: list[PendingAccessRequestBrief] = Field(default_factory=list)
    openCommitments: list[CommitmentBrief] = Field(default_factory=list)
    recentAttestations: list[AttestationBrief] = Field(default_factory=list)

class MeResponse(BaseModel):
    emailHash: str
    address: str
    registeredAt: str
    roles: dict[str, RoleAccountInfo] = Field(default_factory=dict)
    dashboard: Optional[Dashboard] = None

def _normalize_roles(roles: list[str]) -> list[str]:
    out: list[str] = []
    for r in roles:
        u = r.upper()
        if u not in VALID_ROLES:
            raise HTTPException(400, f"Unknown role '{r}'. Valid: {sorted(VALID_ROLES)}")
        if u not in out:
            out.append(u)
    return out

async def _activate_roles(email: str, roles: list[str]) -> dict[str, RoleAccountInfo]:
    svc = get_blockchain_service()
    out: dict[str, RoleAccountInfo] = {}
    for role in roles:
        result = await svc.activate_role(email, role)
        if not result.get("success"):
            raise HTTPException(500, f"Could not activate role {role}: {result.get('error')}")
        out[role] = RoleAccountInfo(commitment=result["commitment"], account=result["account"])
    return out

@router.post("/register", response_model=RegisterResponse, summary="Issue an OTP for the email; stash any requested roles to activate after verify")
async def register(req: RegisterRequest):

    email = req.email.lower().strip()
    roles = _normalize_roles(req.roles)
    code = await generate_otp(email)

    if roles:
        await get_cache().set_authorization_token(
            f"pending_roles:{_hash_email(email)}",
            {"roles": roles},
            ttl=600,
        )

    return RegisterResponse(
        emailHash=_hash_email(email),
        address=derive_address_from_email(email),
        roles={},
        otpCode=code if DEV_MODE else None,
    )

@router.post("/verify", response_model=VerifyResponse, summary="Trade the OTP for a session token; runs first-time identity setup and activates pending roles")
async def verify(req: VerifyRequest):

    from api.services.blockchain import get_blockchain_service

    email = req.email.lower().strip()
    email_hash = _hash_email(email)
    already_registered = await is_registered(email)
    token = await verify_otp(email, req.code)
    if not token:
        raise HTTPException(401, "Invalid or expired code")

    await fire_ibis_lifecycle(
        email, OperationType.AUTHENTICATE if already_registered else OperationType.REGISTER
    )

    svc = get_blockchain_service()
    reg_result = await svc.register_user_identity(email)
    if not reg_result.get("success"):
        raise HTTPException(500, f"identity registration failed: {reg_result.get('error')}")

    cache = get_cache()
    pending = await cache.get_authorization_token(f"pending_roles:{email_hash}")
    if pending and isinstance(pending, dict) and pending.get("roles"):
        try:
            await _activate_roles(email, pending["roles"])
        finally:
            await cache.invalidate_authorization_token(f"pending_roles:{email_hash}")

    return VerifyResponse(
        token=token,
        emailHash=email_hash,
        address=derive_address_from_email(email),
    )

@router.post("/renew", response_model=VerifyResponse, include_in_schema=False)
async def renew(req: RenewRequest, user: AuthenticatedUser = Depends(get_current_user)):
    email = req.email.lower().strip()
    if _hash_email(email) != user.email_hash:
        raise HTTPException(403, "Cannot renew another user")
    token = secrets.token_urlsafe(48)
    now = datetime.utcnow().isoformat() + "Z"
    await get_cache().set_authorization_token(
        f"session:{token}",
        {
            "email_hash": user.email_hash,
            "address": user.address,
            "registered_at": user.registered_at,
            "last_used": now,
        },
        ttl=86400,
    )
    await fire_ibis_lifecycle(email, OperationType.RENEW)
    return VerifyResponse(token=token, emailHash=user.email_hash, address=user.address)

@router.post("/recover", response_model=VerifyResponse, include_in_schema=False)
async def recover(req: RecoverRequest):
    email = req.email.lower().strip()
    token = await verify_otp(email, req.code)
    if not token:
        raise HTTPException(401, "Invalid or expired code")
    await fire_ibis_lifecycle(email, OperationType.RECOVER)
    return VerifyResponse(
        token=token,
        emailHash=_hash_email(email),
        address=derive_address_from_email(email),
    )

@router.post("/logout", summary="Invalidate the current session token")
async def logout(user: AuthenticatedUser = Depends(get_current_user)):
    return {"success": True}

class ActivateRoleRequest(BaseModel):
    email: EmailStr

@router.post("/roles/{role}/activate", response_model=RoleAccountInfo, summary="Activate PROVIDER or REQUESTER on an already-registered user; deploys the role's smart account")
async def activate_role(
    body: ActivateRoleRequest,
    role: str = Path(..., description="PROVIDER or REQUESTER"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    role_upper = role.upper()
    if role_upper not in VALID_ROLES:
        raise HTTPException(400, f"Unknown role '{role}'. Valid: {sorted(VALID_ROLES)}")
    email = body.email.lower().strip()
    if _hash_email(email) != user.email_hash:
        raise HTTPException(403, "email does not match authenticated identity")
    info = await _activate_roles(email, [role_upper])
    return info[role_upper]

@router.get("/me", response_model=MeResponse, summary="Identity, roles and role-aware dashboard for the authenticated user")
async def me(user: AuthenticatedUser = Depends(get_current_user)):
    svc = get_blockchain_service()
    cache = get_cache()
    role_info: dict[str, RoleAccountInfo] = {}
    for role in sorted(VALID_ROLES):
        existing = svc.get_role_account_if_attached(user.address, role)
        if existing:
            role_info[role] = RoleAccountInfo(
                commitment=existing["commitment"], account=existing["account"]
            )

    pending_reqs: list[PendingAccessRequestBrief] = []
    if "PROVIDER" in role_info:
        principal = role_info["PROVIDER"].account.lower()
        eoa = user.address.lower()
        for c in await cache.get_all_consents():
            owners = [a.lower() for a in c.get("owners", [])]
            if eoa not in owners and principal not in owners:
                continue
            cohort_hash = (c.get("cohort_hash") or "").lstrip("0x")
            if not cohort_hash or not hasattr(cache, "get_cohort_access_grants"):
                continue
            for g in await cache.get_cohort_access_grants(cohort_hash):
                status = g.get("status") or ("approved" if g.get("approved") else "pending")
                if status != "pending":
                    continue
                pending_reqs.append(PendingAccessRequestBrief(
                    requestId=g.get("request_id", ""),
                    cohortId=c.get("cohort_id", ""),
                    requesterAddress=g.get("requester_address") or g.get("requester", ""),
                    requestedAt=g.get("requested_at"),
                    intendedUse=g.get("intended_use"),
                ))

    open_commitments: list[CommitmentBrief] = []
    if "REQUESTER" in role_info:
        commit_result = await svc.get_researcher_commitments_by_address(user.address)
        for c in commit_result.get("commitments", []):
            if c.get("status") == "ACTIVE":
                open_commitments.append(CommitmentBrief(
                    commitmentId=c.get("commitment_id", ""),
                    cohortHash=c.get("cohort_hash", ""),
                    type=c.get("commitment_type", ""),
                    status=c.get("status", ""),
                    deadline=c.get("deadline"),
                ))

    recent_atts: list[AttestationBrief] = []
    try:
        rows = await cache.get_subject_attestations(user.address)
        now_ts = datetime.utcnow().timestamp()
        for r in rows[-10:]:
            vu = r.get("valid_until")
            valid = not r.get("revoked", False) and (vu is None or vu > now_ts)
            recent_atts.append(AttestationBrief(
                type=r.get("type", ""),
                cohortId=r.get("cohort_id"),
                cohortHash=r.get("scope"),
                valid=valid,
                validUntil=datetime.fromtimestamp(vu).isoformat() if vu else None,
            ))
    except Exception:
        pass

    return MeResponse(
        emailHash=user.email_hash,
        address=user.address,
        registeredAt=user.registered_at,
        roles=role_info,
        dashboard=Dashboard(
            pendingAccessRequests=pending_reqs,
            openCommitments=open_commitments,
            recentAttestations=recent_atts,
        ),
    )
