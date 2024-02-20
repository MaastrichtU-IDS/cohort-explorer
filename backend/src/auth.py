import base64
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, Request
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt

from src.config import settings

router = APIRouter()


oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{settings.authorization_endpoint()}?response_type=code&client_id={settings.client_id}&redirect_uri={settings.redirect_uri}&scope={settings.scope}",
    tokenUrl=settings.token_endpoint(),
)

SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# async def get_user_info(token: Annotated[str, Depends(oauth2_scheme)]):
async def get_user_info(request: Request):
    """Get the actual user using its token"""
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("email") is None:
            raise HTTPException(status_code=403, detail="User email not found in token")
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return payload

    # async with httpx.AsyncClient() as client:
    #     try:
    #         resp = await client.get(
    #             f"{settings.authorization_endpoint()}/userinfo", headers={"Authorization": f"Bearer {token}"}
    #         )
    #         # resp.raise_for_status()
    #         user_info = resp.json()
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=403,
    #             detail=str(e),
    #         )
    #     return user_info


@router.get("/login")
def login():
    data = {
        "audience": "https://other-ihi-app",
        "response_type": settings.response_type,
        "client_id": settings.client_id,
        "redirect_uri": settings.redirect_uri,
        "scope": settings.scope,
    }
    query = f"{settings.authorization_endpoint()}?{urlencode(data)}"
    return RedirectResponse(query)


@router.get("/cb")
async def auth_callback(code: str):
    """Callback for auth. Redirect to frontend if successful"""
    token_payload = {
        "client_id": settings.client_id,
        "client_secret": settings.client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.redirect_uri,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(settings.token_endpoint(), data=token_payload)
        response.raise_for_status()
        token = response.json()
        access_token = token["access_token"]
        try:
            payload = json.loads(base64.urlsafe_b64decode(access_token.split(".")[1] + "==="))
        except Exception as _e:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )

        if payload["aud"] == "https://other-ihi-app" and "read:datasets-descriptions" in payload["permissions"]:
            # TODO: get user email from payload
            user_email = settings.decentriq_email
            jwt_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            jwt_token = create_access_token(
                data={"email": user_email}, expires_delta=jwt_token_expires
            )

            # NOTE: Redirect to react frontend
            nextjs_redirect_uri = f"{settings.frontend_url}/cohorts"
            send_resp = RedirectResponse(url=nextjs_redirect_uri)
            send_resp.set_cookie(
                key="token",
                value=jwt_token,
                httponly=True,
                secure=True,
                max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                samesite="Lax"  # or 'Strict'
            )
            return send_resp
        else:
            raise HTTPException(
                status_code=403,
                detail="User is not authorized",
            )


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="token")
    return {"message": "Logged out successfully"}
