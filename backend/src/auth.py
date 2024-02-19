import base64
import json
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2AuthorizationCodeBearer

from src.config import settings

router = APIRouter()


oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{settings.authorization_endpoint()}?response_type=code&client_id={settings.client_id}&redirect_uri={settings.redirect_uri}&scope={settings.scope}",
    tokenUrl=settings.token_endpoint(),
)


async def get_user_info(token: Annotated[str, Depends(oauth2_scheme)]):
    """Get the actual user using its token"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{settings.authorization_endpoint()}/userinfo", headers={"Authorization": f"Bearer {token}"}
            )
            # resp.raise_for_status()
            user_info = resp.json()
        except Exception as e:
            raise HTTPException(
                status_code=403,
                detail=str(e),
            )
        return user_info


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
        # print("payload", payload)
        if payload["aud"] == "https://other-ihi-app" and "read:datasets-descriptions" in payload["permissions"]:
            # NOTE: Redirect to react frontend
            nextjs_redirect_uri = f"{settings.frontend_url}/cohorts"
            send_resp = RedirectResponse(url=nextjs_redirect_uri)
            expiration = 60 * 60 * 24 * 7  # 1 week
            # TODO: should we reuse the 'exp': 1708090134 from the payload?
            send_resp.set_cookie(key="token", value=access_token, httponly=True, secure=True, expires=expiration)
            # send_resp.set_cookie(key="email", value=access_token, httponly=True, secure=True, expires=expiration)

            return send_resp
            # return RedirectResponse(url=f"{nextjs_redirect_uri}?token={access_token}")
        else:
            raise HTTPException(
                status_code=403,
                detail="User is not authorized",
            )


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="token")
    return {"message": "Logged out successfully"}
