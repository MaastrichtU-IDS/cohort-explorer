"""Admin settings endpoints — accessible only to users in the ADMINS list."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_403_FORBIDDEN

from src.auth import get_current_user
from src.config import settings
from src.cohort_cache import (
    get_cohorts_from_cache,
    add_cohort_to_cache,
    save_cache_to_disk,
)

router = APIRouter(prefix="/admin", tags=["admin"])

TIMECHF_TEST_EMAIL = "wei.wei@maastrichtuniversity.nl"
TIMECHF_COHORT_ID = "TIME-CHF"


def _require_admin(user: dict[str, str]) -> str:
    """Return user email if admin, otherwise raise 403."""
    email = user["email"].lower()
    if email not in settings.admins_list:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Admin access required")
    return email


# ------------------------------------------------------------------
# GET /admin/check — is the current user an admin?
# ------------------------------------------------------------------
@router.get("/check")
def check_admin(user: Any = Depends(get_current_user)) -> dict:
    email = user["email"].lower()
    return {"is_admin": email in settings.admins_list}


# ------------------------------------------------------------------
# GET /admin/settings — retrieve current admin settings
# ------------------------------------------------------------------
@router.get("/settings")
def get_admin_settings(user: Any = Depends(get_current_user)) -> dict:
    _require_admin(user)

    cohorts = get_cohorts_from_cache("")
    timechf = cohorts.get(TIMECHF_COHORT_ID)

    timechf_testing = False
    if timechf:
        timechf_testing = TIMECHF_TEST_EMAIL in [
            e.lower() for e in (timechf.cohort_email or [])
        ]

    return {
        "timechf_testing_enabled": timechf_testing,
    }


# ------------------------------------------------------------------
# POST /admin/toggle-timechf-testing — flip the switch
# ------------------------------------------------------------------
@router.post("/toggle-timechf-testing")
def toggle_timechf_testing(user: Any = Depends(get_current_user)) -> dict:
    admin_email = _require_admin(user)

    # Operate on the *live* cache (pass empty email to skip can_edit logic)
    cohorts = get_cohorts_from_cache("")
    timechf = cohorts.get(TIMECHF_COHORT_ID)
    if not timechf:
        raise HTTPException(status_code=404, detail=f"Cohort {TIMECHF_COHORT_ID} not found in cache")

    emails_lower = [e.lower() for e in (timechf.cohort_email or [])]
    currently_enabled = TIMECHF_TEST_EMAIL in emails_lower

    if currently_enabled:
        # Remove
        timechf.cohort_email = [
            e for e in timechf.cohort_email if e.lower() != TIMECHF_TEST_EMAIL
        ]
        logging.info(
            "Admin %s DISABLED TIME-CHF testing capacity (removed %s)",
            admin_email, TIMECHF_TEST_EMAIL,
        )
    else:
        # Add
        timechf.cohort_email.append(TIMECHF_TEST_EMAIL)
        logging.info(
            "Admin %s ENABLED TIME-CHF testing capacity (added %s)",
            admin_email, TIMECHF_TEST_EMAIL,
        )

    # Persist to the shared cache file so all workers pick it up
    add_cohort_to_cache(timechf, save_to_disk=True)

    return {
        "timechf_testing_enabled": not currently_enabled,
    }
