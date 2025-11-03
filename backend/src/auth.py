import base64
import json
import time
from datetime import datetime, timezone
from typing import Optional, Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from src.config import settings


router = APIRouter()

JWT_ALGORITHM = "HS256"


# Extend FastAPI auth scheme to use a HTTP-only cookie token
class OAuth2AuthorizationCodeCookie(OAuth2AuthorizationCodeBearer):
    """
    OAuth2 flow for authentication using a HTTP-only cookie token obtained with an OAuth2 code
    flow. An instance of it would be used as a dependency.
    """

    async def __call__(self, request: Request) -> Optional[str]:
        token = request.cookies.get("token")
        if not token:
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Cookie"},
                )
            else:
                return None
        return token


auth_params = {
    "audience": settings.auth_audience,
    "redirect_uri": settings.redirect_uri,
}

oauth2_scheme = OAuth2AuthorizationCodeCookie(
    authorizationUrl=f"{settings.authorization_endpoint}?{urlencode(auth_params)}",
    tokenUrl=settings.token_endpoint,
    scopes={
        "openid": "OpenID Connect",
        "email": "Access user email",
        "read:icare4cvd-dataset-descriptions": "Access iCARE4CVD datasets descriptions",
    },
)


@router.get("/login")
def login() -> RedirectResponse:
    """Redirect to Auth0 login page to authenticate the user and get the code to exchange for a token"""
    login_url = f"""{settings.authorization_endpoint}?{urlencode({
        **auth_params,
        'scope': settings.scope,
        'response_type': settings.response_type,
        'client_id': settings.client_id,
        'prompt': 'login',
    })}"""
    return RedirectResponse(login_url)


def create_access_token(data: dict[str, str], expires_timestamp: int) -> str:
    """Create a JWT token with the given data and expiration time"""
    to_encode = data.copy()
    to_encode.update({"exp": datetime.fromtimestamp(expires_timestamp, timezone.utc)})
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=JWT_ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, str]:
    """Get the logged user decoding its JWT token passed through HTTP-only cookie"""
    if not token:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])
        if payload.get("email") is None:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="User email not found in token")
    except JWTError as e:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail=str(e))
    return payload


@router.get("/cb")
async def auth_callback(code: str) -> RedirectResponse:
    """Callback for auth. Generates JWT token and redirects to frontend if successful"""
    token_payload = {
        "client_id": settings.client_id,
        "client_secret": settings.client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.redirect_uri,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(settings.token_endpoint, data=token_payload)
        response.raise_for_status()
        token = response.json()
        try:
            access_payload = json.loads(base64.urlsafe_b64decode(token["access_token"].split(".")[1] + "==="))
            id_payload = json.loads(base64.urlsafe_b64decode(token["id_token"].split(".")[1] + "==="))
        except Exception as _e:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        # print("ACCESS PAYLOAD", access_payload)
        # NOTE: if needed user info can be retrieved later from the /userinfo endpoint using the provided access token
        # resp = await client.get(f"{settings.authorization_endpoint}/userinfo", headers={"Authorization": f"Bearer {token['access_token']}"})
        # print("user_info", resp.json())

        # Check in payload if logged in user has the required permissions
        if (
            "https://explorer.icare4cvd.eu" in access_payload["aud"]
            and "read:icare4cvd-dataset-descriptions" in access_payload["permissions"]
        ) or settings.dev_mode:
            # TODO: for LUCE blockchain: check if user email has a blockchain address
            # Where? Either a JSON file on the server, or in the triplestore
            # blockchain_addrs = json.load(settings.data_folder / "blockchain_addresses.json")
            # if id_payload["email"] not in blockchain_addrs:
            #     blockchain_addrs[id_payload["email"]] = "0x1234567890"
            #     json.dump(blockchain_addrs, settings.data_folder / "blockchain_addresses.json")

            # Reuse expiration time from decentriq Auth0 access token
            exp_timestamp = access_payload["exp"]
            jwt_token = create_access_token(
                data={"email": id_payload["email"].lower(), "access_token": token["access_token"]},
                expires_timestamp=exp_timestamp,
            )

            # NOTE: Redirect to react frontend
            send_resp = RedirectResponse(url=f"{settings.frontend_url}/cohorts")
            # Send JWT token as HTTP-only cookie to the frontend (will not be available to JS code in the frontend)
            send_resp.set_cookie(
                key="token",
                value=jwt_token,
                httponly=True,
                secure=True,
                max_age=exp_timestamp - int(time.time()),
                samesite="Lax",  # or 'Strict'
            )
            return send_resp
        else:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="User is not authorized",
            )


@router.get("/debug/permissions")
async def debug_permissions() -> dict[str, Any]:
    """
    Compact permission map for emails present in spreadsheet fields.
    For each email that appears as cohort email, administrator_email, or study_contact_person_email,
    list cohorts where that email has access (owner or admin). Cohorts with no access are omitted.
    """
    from src.cohort_cache import get_cohorts_from_cache
    from src.config import settings

    all_cohorts = get_cohorts_from_cache("")

    # Collect emails from spreadsheet-driven fields
    spreadsheet_emails: set[str] = set()
    for c in all_cohorts.values():
        spreadsheet_emails.update(e.lower() for e in c.cohort_email or [] if e)
        if c.administrator_email:
            spreadsheet_emails.add(c.administrator_email.lower())
        if c.study_contact_person_email:
            spreadsheet_emails.add(c.study_contact_person_email.lower())

    admins_lower = {a.lower() for a in settings.admins_list}

    # Build compact mapping email -> [cohort_ids]
    emails_map: dict[str, list[str]] = {}
    for email in sorted(spreadsheet_emails):
        has_admin = email in admins_lower
        accessible: list[str] = []
        for cohort_id, cohort in all_cohorts.items():
            is_owner = email in [e.lower() for e in (cohort.cohort_email or [])] or \
                       (cohort.administrator_email and email == cohort.administrator_email.lower()) or \
                       (cohort.study_contact_person_email and email == cohort.study_contact_person_email.lower())
            if has_admin or is_owner:
                accessible.append(cohort_id)
        if accessible:
            emails_map[email] = sorted(accessible)

    return {
        "admins": sorted(settings.admins_list),
        "emails": emails_map,
    }


@router.post("/logout")
def logout(resp: Response) -> dict[str, str]:
    """Log out the user by deleting the token HTTP-only cookie"""
    resp.delete_cookie(key="token")
    return {"message": "Logged out successfully"}
