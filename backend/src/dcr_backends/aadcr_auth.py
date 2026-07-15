"""Short-lived service tokens for Cohort Explorer to call local AADCR v2."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt

from src.config import Settings

AADCR_TOKEN_TTL = timedelta(minutes=5)
JWT_ALGORITHM = "HS256"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def mint_aadcr_token(user: dict[str, Any], settings: Settings) -> str:
    """Mint a minimal, short-lived token representing one authenticated user."""
    email = str(user.get("email") or "").strip().lower()
    if not email:
        raise ValueError("AADCR authentication requires an authenticated user email")
    if not settings.aadcrv2_jwt_secret:
        raise ValueError("AADCRV2_JWT_SECRET is required to mint an AADCR token")

    issued_at = _utc_now()
    claims = {
        "sub": email,
        "email": email,
        "email_verified": True,
        "iss": settings.aadcrv2_jwt_issuer,
        "aud": settings.aadcrv2_jwt_audience,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + AADCR_TOKEN_TTL).timestamp()),
    }
    return jwt.encode(claims, settings.aadcrv2_jwt_secret, algorithm=JWT_ALGORITHM)
