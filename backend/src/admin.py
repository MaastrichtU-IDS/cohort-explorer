"""Admin settings endpoints — accessible only to users in the ADMINS list."""

import json
import logging
import os
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

# Persisted app settings toggled from the admin page (shared across workers via
# a small JSON file in the data folder).
APP_SETTINGS_FILE = os.path.join(settings.data_folder, "app_settings.json")
APP_SETTINGS_DEFAULTS = {
    # Whether the "iCARE-AI" nav button is shown to users.
    "ai_nav_enabled": False,
    # Show the "Flexible DCR / No-code DCR" chooser when creating an analysis DCR.
    # Off = the traditional wizard opens directly.
    "dcr_chooser_enabled": True,
}


def _load_app_settings() -> dict:
    result = dict(APP_SETTINGS_DEFAULTS)
    try:
        with open(APP_SETTINGS_FILE) as fh:
            stored = json.load(fh)
        if isinstance(stored, dict):
            result.update({k: stored[k] for k in APP_SETTINGS_DEFAULTS if k in stored})
    except Exception:
        pass
    return result


def _save_app_settings(values: dict) -> None:
    os.makedirs(os.path.dirname(APP_SETTINGS_FILE), exist_ok=True)
    with open(APP_SETTINGS_FILE, "w") as fh:
        json.dump(values, fh, indent=2)


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
        "ai_nav_enabled": _load_app_settings()["ai_nav_enabled"],
        "dcr_chooser_enabled": _load_app_settings()["dcr_chooser_enabled"],
    }


# ------------------------------------------------------------------
# GET /admin/public-settings — app settings any logged-in user may read
# (used by the nav bar to decide whether to show the iCARE-AI button)
# ------------------------------------------------------------------
@router.get("/public-settings")
def get_public_settings(user: Any = Depends(get_current_user)) -> dict:
    values = _load_app_settings()
    return {"ai_nav_enabled": values["ai_nav_enabled"], "dcr_chooser_enabled": values["dcr_chooser_enabled"]}


# ------------------------------------------------------------------
# POST /admin/toggle-ai-nav — show/hide the iCARE-AI nav button
# ------------------------------------------------------------------
@router.post("/toggle-ai-nav")
def toggle_ai_nav(user: Any = Depends(get_current_user)) -> dict:
    admin_email = _require_admin(user)
    values = _load_app_settings()
    values["ai_nav_enabled"] = not values["ai_nav_enabled"]
    _save_app_settings(values)
    logging.info("Admin %s set ai_nav_enabled=%s", admin_email, values["ai_nav_enabled"])
    return {"ai_nav_enabled": values["ai_nav_enabled"]}


# ------------------------------------------------------------------
# POST /admin/toggle-dcr-chooser — show/hide the Flexible/No-code chooser
# when creating an analysis DCR (off = traditional wizard is the default)
# ------------------------------------------------------------------
@router.post("/toggle-dcr-chooser")
def toggle_dcr_chooser(user: Any = Depends(get_current_user)) -> dict:
    admin_email = _require_admin(user)
    values = _load_app_settings()
    values["dcr_chooser_enabled"] = not values["dcr_chooser_enabled"]
    _save_app_settings(values)
    logging.info("Admin %s set dcr_chooser_enabled=%s", admin_email, values["dcr_chooser_enabled"])
    return {"dcr_chooser_enabled": values["dcr_chooser_enabled"]}


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
