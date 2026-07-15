"""Admin settings endpoints — accessible only to users in the ADMINS list."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_403_FORBIDDEN

from src.auth import get_current_user
from src.cohort_cache import (
    add_cohort_to_cache,
    get_cohorts_from_cache,
)
from src.config import settings

router = APIRouter(prefix="/admin", tags=["admin"])
debug_router = APIRouter(prefix="/debug", tags=["debug"])

TIMECHF_TEST_EMAIL = "wei.wei@maastrichtuniversity.nl"
TIMECHF_COHORT_ID = "TIME-CHF"


def _require_admin(user: dict[str, str]) -> str:
    """Return user email if admin, otherwise raise 403."""
    email = user["email"].lower()
    if email not in settings.admins_list:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Admin access required")
    return email


@debug_router.get("/permissions")
async def debug_permissions(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Return the cohort permission map to an authenticated administrator."""
    _require_admin(user)

    from src import cohort_cache

    all_cohorts = cohort_cache.get_cohorts_from_cache("")

    spreadsheet_emails: set[str] = set()
    for cohort in all_cohorts.values():
        spreadsheet_emails.update(email.lower() for email in cohort.cohort_email or [] if email)
        if cohort.administrator_email:
            spreadsheet_emails.add(cohort.administrator_email.lower())
        if cohort.study_contact_person_email:
            spreadsheet_emails.add(cohort.study_contact_person_email.lower())

    admins_lower = {admin.lower() for admin in settings.admins_list}
    emails_map: dict[str, list[str]] = {}
    for email in sorted(spreadsheet_emails):
        accessible: list[str] = []
        for cohort_id, cohort in all_cohorts.items():
            is_owner = (
                email in [owner.lower() for owner in (cohort.cohort_email or [])]
                or bool(cohort.administrator_email and email == cohort.administrator_email.lower())
                or bool(cohort.study_contact_person_email and email == cohort.study_contact_person_email.lower())
            )
            if email in admins_lower or is_owner:
                accessible.append(cohort_id)
        if accessible:
            emails_map[email] = sorted(accessible)

    return {
        "admins": sorted(settings.admins_list),
        "emails": emails_map,
    }


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
