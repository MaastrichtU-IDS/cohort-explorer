from datetime import datetime, timezone

import pytest
from jose import jwt

from src.dcr_backends.aadcr_auth import mint_aadcr_token

TEST_AADCR_SECRET = "shared-test-secret"


def test_minted_token_matches_aadcr_claim_contract(settings_factory, monkeypatch):
    fixed_now = datetime(2026, 7, 15, 9, 30, tzinfo=timezone.utc)
    monkeypatch.setattr("src.dcr_backends.aadcr_auth._utc_now", lambda: fixed_now)
    settings = settings_factory(
        dcr_backend="aadcrv2",
        aadcrv2_jwt_secret=TEST_AADCR_SECRET,
        aadcrv2_jwt_issuer="cohort-explorer-local",
        aadcrv2_jwt_audience="aadcrv2-local",
    )
    user = {
        "email": "  Nikolas.Molyndris@Decentriq.CH ",
        "access_token": "must-not-cross-the-boundary",
        "session_cookie": "must-not-cross-the-boundary-either",
    }

    token = mint_aadcr_token(user, settings)

    assert jwt.get_unverified_header(token) == {"alg": "HS256", "typ": "JWT"}
    claims = jwt.decode(
        token,
        settings.aadcrv2_jwt_secret,
        algorithms=["HS256"],
        audience=settings.aadcrv2_jwt_audience,
        issuer=settings.aadcrv2_jwt_issuer,
        options={"verify_exp": False},
    )
    expected_timestamp = int(fixed_now.timestamp())
    assert claims["sub"] == "nikolas.molyndris@decentriq.ch"
    assert claims["email"] == "nikolas.molyndris@decentriq.ch"
    assert claims["email_verified"] is True
    assert claims["iss"] == "cohort-explorer-local"
    assert claims["aud"] == "aadcrv2-local"
    assert claims["iat"] == expected_timestamp
    assert 0 < claims["exp"] - claims["iat"] <= 300
    assert "access_token" not in claims
    assert "session_cookie" not in claims
    assert "cookie" not in claims


@pytest.mark.parametrize("email", [None, "", "   "])
def test_mint_rejects_missing_email(settings_factory, email):
    settings = settings_factory(aadcrv2_jwt_secret=TEST_AADCR_SECRET)

    with pytest.raises(ValueError, match="authenticated user email"):
        mint_aadcr_token({"email": email}, settings)


def test_mint_rejects_missing_shared_secret(settings_factory):
    settings = settings_factory(aadcrv2_jwt_secret="")

    with pytest.raises(ValueError, match="AADCRV2_JWT_SECRET"):
        mint_aadcr_token({"email": "owner@example.test"}, settings)
