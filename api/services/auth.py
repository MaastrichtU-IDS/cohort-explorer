import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.config import get_settings
from api.services.cache import get_cache
from api.services.wallet import derive_address_from_email

logger = logging.getLogger(__name__)

OTP_TTL = 600
TOKEN_TTL = 86400
OTP_LENGTH = 6

def _hash_email(email: str) -> str:
    salt = get_settings().derivation_salt.encode()
    return hmac.new(salt, email.lower().strip().encode(), hashlib.sha256).hexdigest()

@dataclass
class AuthenticatedUser:
    email: str
    email_hash: str
    address: str
    registered_at: str

async def generate_otp(email: str) -> str:
    cache = get_cache()
    email_hash = _hash_email(email.lower().strip())
    code = "".join(str(secrets.randbelow(10)) for _ in range(OTP_LENGTH))
    await cache.set_authorization_token(
        f"otp:{email_hash}",
        {
            "code": code,
            "email_hash": email_hash,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "attempts": 0,
        },
        ttl=OTP_TTL,
    )
    return code

async def verify_otp(email: str, code: str) -> str | None:
    cache = get_cache()
    email_lower = email.lower().strip()
    email_hash = _hash_email(email_lower)
    otp_key = f"otp:{email_hash}"
    otp_data = await cache.get_authorization_token(otp_key)
    if not otp_data:
        return None

    if otp_data.get("attempts", 0) >= 5:
        await cache.invalidate_authorization_token(otp_key)
        return None

    otp_data["attempts"] = otp_data.get("attempts", 0) + 1
    await cache.set_authorization_token(otp_key, otp_data, ttl=OTP_TTL)
    if str(otp_data.get("code")) != code:
        return None

    await cache.invalidate_authorization_token(otp_key)
    token = secrets.token_urlsafe(48)
    address = derive_address_from_email(email_lower)
    now = datetime.utcnow().isoformat() + "Z"

    await cache.set_authorization_token(
        f"session:{token}",
        {"email_hash": email_hash, "address": address, "registered_at": now, "last_used": now},
        ttl=TOKEN_TTL,
    )

    if not await cache.get_authorization_token(f"user:{email_hash}"):
        await cache.set_authorization_token(
            f"user:{email_hash}",
            {"email_hash": email_hash, "address": address, "registered_at": now},
            ttl=TOKEN_TTL * 365,
        )
    return token

async def is_registered(email: str) -> bool:

    try:
        from api.services.blockchain import get_blockchain_service
        svc = get_blockchain_service()
        if svc.user_identity_registry:
            return svc.is_user_registered_onchain(email)
    except Exception:
        pass
    return (await get_cache().get_authorization_token(f"user:{_hash_email(email)}")) is not None

async def get_user_from_token(token: str) -> AuthenticatedUser | None:
    cache = get_cache()
    session = await cache.get_authorization_token(f"session:{token}")
    if not session:
        return None
    session["last_used"] = datetime.utcnow().isoformat() + "Z"
    await cache.set_authorization_token(f"session:{token}", session, ttl=TOKEN_TTL)
    return AuthenticatedUser(
        email="",
        email_hash=session["email_hash"],
        address=session["address"],
        registered_at=session["registered_at"],
    )

async def revoke_token(token: str) -> bool:
    cache = get_cache()
    session = await cache.get_authorization_token(f"session:{token}")
    if not session:
        return False
    await cache.invalidate_authorization_token(f"session:{token}")
    return True

bearer_scheme = HTTPBearer(
    description="Paste the token returned by POST /api/auth/verify (no 'Bearer ' prefix).",
    auto_error=False,
)

async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedUser:
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    user = await get_user_from_token(creds.credentials)
    if not user:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    return user

def require_email_match(user: AuthenticatedUser, request_email: str, field_name: str = "email") -> None:
    if user.email_hash != _hash_email(request_email):
        raise HTTPException(403, f"{field_name} does not match authenticated identity")

async def fire_ibis_lifecycle(email: str, op_type) -> dict:
    try:
        from api.services.blockchain import get_blockchain_service
        from api.services.ibis import get_ibis_protocol

        protocol = get_ibis_protocol()
        envelope = protocol.create_envelope(email=email, operation=op_type)
        await protocol.process_envelope(envelope)
        try:
            await get_blockchain_service().submit_ibis_envelope(
                commitment=bytes.fromhex(envelope.commitment),
                nullifier=bytes.fromhex(envelope.nullifier),
                blinded_epoch=bytes.fromhex(envelope.blinded_epoch),
                epoch_key_hash=bytes.fromhex(envelope.epoch_key_hash),
                encrypted_payload=bytes.fromhex(envelope.encrypted_payload),
                proof=bytes.fromhex(envelope.proof),
            )
        except Exception as e:
            logger.warning(f"IBIS chain submit failed: {e}")
        return {"ok": True, "operation": op_type.name}
    except Exception as e:
        logger.warning(f"IBIS lifecycle failed op={op_type.name}: {e}")
        return {"ok": False, "operation": op_type.name, "error": str(e)}
