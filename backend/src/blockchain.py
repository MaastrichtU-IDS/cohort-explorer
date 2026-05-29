"""
Proxy routes for the iCARE4CHAIN blockchain consent API.
These endpoints allow the Cohort Explorer frontend to interact with the
blockchain API (running at icarechain-api:8001) without CORS issues.
"""
import logging
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.auth import get_current_user
from src.config import settings

router = APIRouter(tags=["blockchain"])
logger = logging.getLogger(__name__)

# The icarechain API is accessible within docker network
ICARECHAIN_API_URL = "http://icarechain-api:8001/api"


class BlockchainRegisterResponse(BaseModel):
    emailHash: str
    address: str
    roles: dict = Field(default_factory=dict)
    otpCode: Optional[str] = None


class BlockchainVerifyResponse(BaseModel):
    token: str
    emailHash: str
    address: str


class ConsentDeclarationInput(BaseModel):
    permission: str = Field(..., description="Primary DUO permission: NRES | GRU | HMB | DS | POA")
    modifiers: list[str] = Field(default_factory=list, description="DUO modifier codes")
    diseaseCode: Optional[str] = Field(None, description="Ontology CURIE (MONDO/DOID/HP); required when permission == DS")
    allowedCountries: list[str] = Field(default_factory=list, description="ISO-3166 codes; required when GS modifier set")
    allowedInstitutions: list[str] = Field(default_factory=list, description="ROR IDs; required when IS modifier set")
    allowedProjects: list[str] = Field(default_factory=list, description="Project IDs; required when PS modifier set")
    allowedUsers: list[str] = Field(default_factory=list, description="User addresses or email hashes; required when US modifier set")
    moratoriumMonths: Optional[int] = Field(None, ge=1, description="Months of publication moratorium; required when MOR modifier set")
    researchScope: Optional[str] = Field(None, description="Free-text research scope; required when RS modifier set")
    returnTargetUri: Optional[str] = Field(None, description="Target URI to return derived data; required when RTN modifier set")
    publicationDeadlineDays: Optional[int] = Field(None, ge=1, description="Days for PUB obligation; optional when PUB modifier set")
    expirationDays: int = Field(0, ge=0, description="Days until consent expires; 0 = no expiry")
    dataUseDescription: Optional[str] = Field(None, description="Free-text description of what the data may be used for")
    additionalRestrictions: Optional[str] = Field(None, description="Free-text restrictions beyond modifier list")
    consentDate: Optional[str] = Field(None, description="Date the original consent was obtained (ISO format)")
    consentFormUri: Optional[str] = Field(None, description="URI of the signed consent form")
    metadataUri: str = Field("", description="Generic metadata URI")


class RecordConsentRequest(BaseModel):
    cohortId: str
    consent: ConsentDeclarationInput


@router.post("/blockchain/register", summary="Register user with the blockchain API and get OTP")
async def blockchain_register(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """
    Registers the currently logged-in user with the blockchain API as a PROVIDER.
    Returns the OTP code (in dev mode) and registration details.
    """
    email = user["email"]
    logger.info(f"Blockchain register request for {email}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/register",
                json={"email": email, "roles": ["PROVIDER"]},
            )
            data = resp.json()
            if resp.status_code != 200:
                logger.error(f"Blockchain register failed: {resp.status_code} {data}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=data.get("detail", f"Blockchain register failed: {data}"),
                )
            logger.info(f"Blockchain register success for {email}: address={data.get('address')}")
            return data
        except httpx.RequestError as e:
            logger.error(f"Cannot reach blockchain API: {e}")
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API: {str(e)}")


@router.post("/blockchain/verify", summary="Verify OTP with the blockchain API and get session token")
async def blockchain_verify(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """
    Automatically registers and verifies the user:
    1. Calls /auth/register to get an OTP
    2. Calls /auth/verify with that OTP to get a session token
    Returns the session token for subsequent blockchain API calls.
    """
    email = user["email"]
    logger.info(f"Blockchain verify (full flow) for {email}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Register to get OTP
        try:
            reg_resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/register",
                json={"email": email, "roles": ["PROVIDER"]},
            )
            reg_data = reg_resp.json()
            if reg_resp.status_code != 200:
                raise HTTPException(
                    status_code=reg_resp.status_code,
                    detail=reg_data.get("detail", f"Blockchain register failed: {reg_data}"),
                )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API for register: {str(e)}")

        otp_code = reg_data.get("otpCode")
        if not otp_code:
            raise HTTPException(
                status_code=500,
                detail="Blockchain API did not return OTP code. Is AUTH_DEV_MODE enabled?",
            )

        # Step 2: Verify OTP to get session token
        try:
            verify_resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/verify",
                json={"email": email, "code": otp_code},
            )
            verify_data = verify_resp.json()
            if verify_resp.status_code != 200:
                raise HTTPException(
                    status_code=verify_resp.status_code,
                    detail=verify_data.get("detail", f"Blockchain verify failed: {verify_data}"),
                )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API for verify: {str(e)}")

        logger.info(f"Blockchain verify success for {email}: address={verify_data.get('address')}")
        return {
            "register": reg_data,
            "verify": verify_data,
            "message": "Successfully authenticated with blockchain API",
        }


@router.post("/blockchain/record-consent", summary="Record DUO consent on the blockchain for a cohort")
async def blockchain_record_consent(
    req: RecordConsentRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Records consent constraints for a cohort on the blockchain.
    Requires a valid blockchain session token (obtained via /blockchain/verify).
    This endpoint:
    1. Authenticates with blockchain API (register + verify)
    2. Calls POST /api/providers/consents with the consent declaration
    """
    email = user["email"]
    logger.info(f"Blockchain record-consent for cohort={req.cohortId} by {email}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # First authenticate to get a session token
        try:
            reg_resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/register",
                json={"email": email, "roles": ["PROVIDER"]},
            )
            reg_data = reg_resp.json()
            if reg_resp.status_code != 200:
                raise HTTPException(
                    status_code=reg_resp.status_code,
                    detail=f"Blockchain register failed: {reg_data.get('detail', reg_data)}",
                )
            otp_code = reg_data.get("otpCode")
            if not otp_code:
                raise HTTPException(status_code=500, detail="No OTP returned from blockchain API")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API: {str(e)}")

        try:
            verify_resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/verify",
                json={"email": email, "code": otp_code},
            )
            verify_data = verify_resp.json()
            if verify_resp.status_code != 200:
                raise HTTPException(
                    status_code=verify_resp.status_code,
                    detail=f"Blockchain verify failed: {verify_data.get('detail', verify_data)}",
                )
            token = verify_data.get("token")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API: {str(e)}")

        # Now call the consent endpoint with the session token
        consent_payload = {
            "email": email,
            "cohortId": req.cohortId,
            "consent": req.consent.model_dump(exclude_none=True),
        }

        try:
            consent_resp = await client.post(
                f"{ICARECHAIN_API_URL}/providers/consents",
                json=consent_payload,
                headers={"Authorization": f"Bearer {token}"},
            )
            consent_data = consent_resp.json()
            if consent_resp.status_code != 200:
                logger.error(f"Blockchain consent recording failed: {consent_resp.status_code} {consent_data}")
                raise HTTPException(
                    status_code=consent_resp.status_code,
                    detail=consent_data.get("detail", f"Consent recording failed: {consent_data}"),
                )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API for consent: {str(e)}")

        logger.info(f"Blockchain consent recorded: cohort={req.cohortId}, txHash={consent_data.get('txHash')}")
        return {
            "message": f"Consent successfully recorded on blockchain for cohort '{req.cohortId}'",
            "blockchain_response": consent_data,
        }


@router.get("/blockchain/consent/{cohort_id}", summary="Get the DUO consent declaration for a cohort")
async def blockchain_get_consent(
    cohort_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Fetches the DUO consent declaration for a given cohort from the blockchain API.
    Authenticates with the blockchain API first, then calls GET /api/providers/consents/{cohort_id}.
    """
    email = user["email"]
    logger.info(f"Blockchain get-consent for cohort={cohort_id} by {email}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Authenticate to get a session token
        try:
            reg_resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/register",
                json={"email": email, "roles": ["PROVIDER"]},
            )
            reg_data = reg_resp.json()
            if reg_resp.status_code != 200:
                raise HTTPException(
                    status_code=reg_resp.status_code,
                    detail=f"Blockchain register failed: {reg_data.get('detail', reg_data)}",
                )
            otp_code = reg_data.get("otpCode")
            if not otp_code:
                raise HTTPException(status_code=500, detail="No OTP returned from blockchain API")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API: {str(e)}")

        try:
            verify_resp = await client.post(
                f"{ICARECHAIN_API_URL}/auth/verify",
                json={"email": email, "code": otp_code},
            )
            verify_data = verify_resp.json()
            if verify_resp.status_code != 200:
                raise HTTPException(
                    status_code=verify_resp.status_code,
                    detail=f"Blockchain verify failed: {verify_data.get('detail', verify_data)}",
                )
            token = verify_data.get("token")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API: {str(e)}")

        # Fetch the consent for this cohort
        try:
            consent_resp = await client.get(
                f"{ICARECHAIN_API_URL}/providers/consents/{cohort_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if consent_resp.status_code == 404:
                return {"found": False, "message": f"No consent declaration found for cohort '{cohort_id}'"}
            if consent_resp.status_code == 403:
                return {"found": False, "message": "You are not authorized to view this cohort's consent declaration"}
            consent_data = consent_resp.json()
            if consent_resp.status_code != 200:
                raise HTTPException(
                    status_code=consent_resp.status_code,
                    detail=consent_data.get("detail", f"Failed to fetch consent: {consent_data}"),
                )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach blockchain API: {str(e)}")

        logger.info(f"Blockchain consent fetched for cohort={cohort_id}")
        return {"found": True, "consent": consent_data}
