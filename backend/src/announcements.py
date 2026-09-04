"""Announcements shown in the rotating box on the Explorer's front page.

Storage is a small JSON file in the data folder. Any admin can add an
announcement (text + obligatory date, which may be backdated + one tag) and
delete any existing one; there is deliberately no edit (delete and re-add).
The stored record keeps WHO added each announcement; that field is returned
only to admins - the front page never shows it.
"""

import json
import logging
import os
import threading
import uuid
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings

router = APIRouter(tags=["announcements"])

ANNOUNCEMENT_TAGS = ("new cohort", "new feature", "analysis", "event")
MAX_TEXT_LENGTH = 500  # announcements are at most a couple of sentences

_LOCK = threading.Lock()


def _store_path() -> str:
    return os.path.join(settings.data_folder, "announcements.json")


def _load() -> dict:
    """The store: {'enabled': bool, 'items': [...]}. Legacy files held a bare
    list of items; those read as enabled=True."""
    try:
        with open(_store_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return {"enabled": True, "items": data}
        if isinstance(data, dict):
            items = data.get("items")
            return {"enabled": bool(data.get("enabled", True)),
                    "items": items if isinstance(items, list) else []}
        return {"enabled": True, "items": []}
    except FileNotFoundError:
        return {"enabled": True, "items": []}
    except Exception as exc:
        logging.warning("Announcements store unreadable: %s", exc)
        return {"enabled": True, "items": []}


def _save(store: dict) -> None:
    path = _store_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _sorted(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda a: (str(a.get("date") or ""), str(a.get("created_at") or "")),
                  reverse=True)


def _is_admin(user: Any) -> bool:
    return user["email"].lower() in settings.admins_list


@router.get("/announcements", name="List announcements for the front-page box (newest first)")
def list_announcements(user: Any = Depends(get_current_user)) -> list[dict]:
    store = _load()
    # The box is hidden globally: empty list for everyone (admins included -
    # they see the front page too; the manage page uses /announcements/all).
    if not store["enabled"]:
        return []
    items = _sorted(store["items"])
    if _is_admin(user):
        return items
    # Non-admins (the front page) never see who added an announcement.
    return [{k: v for k, v in a.items() if k != "added_by"} for a in items]


@router.get("/announcements/all", name="Full announcements store for the manage page (admins only)")
def list_all_announcements(user: Any = Depends(get_current_user)) -> dict:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    store = _load()
    return {"enabled": store["enabled"], "items": _sorted(store["items"])}


@router.post("/announcements/visibility", name="Show/hide the front-page announcements box (admins only)")
def set_announcements_visibility(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    if not isinstance(body.get("enabled"), bool):
        raise HTTPException(status_code=400, detail="enabled (true/false) is required")
    with _LOCK:
        store = _load()
        store["enabled"] = body["enabled"]
        _save(store)
    logging.info("Announcements box %s by %s", "shown" if body["enabled"] else "hidden", user["email"])
    return {"enabled": store["enabled"]}


@router.post("/announcements", name="Add an announcement (admins only)")
def add_announcement(body: dict[str, Any], user: Any = Depends(get_current_user)) -> dict:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400,
                            detail=f"text too long (max {MAX_TEXT_LENGTH} characters - announcements "
                                   "are meant to be one or two sentences)")
    raw_date = str(body.get("date") or "").strip()
    try:
        date.fromisoformat(raw_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="date is required, in YYYY-MM-DD form (backdating is fine)")
    tag = str(body.get("tag") or "").strip().lower()
    if tag not in ANNOUNCEMENT_TAGS:
        raise HTTPException(status_code=400, detail=f"tag must be one of: {', '.join(ANNOUNCEMENT_TAGS)}")
    item = {
        "id": uuid.uuid4().hex,
        "text": text,
        "date": raw_date,
        "tag": tag,
        "added_by": user["email"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _LOCK:
        store = _load()
        store["items"].append(item)
        _save(store)
    logging.info("Announcement %s added by %s (%s, %s)", item["id"], item["added_by"], tag, raw_date)
    return item


@router.delete("/announcements/{announcement_id}", name="Delete an announcement (admins only)")
def delete_announcement(announcement_id: str, user: Any = Depends(get_current_user)) -> dict:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    with _LOCK:
        store = _load()
        kept = [a for a in store["items"] if a.get("id") != announcement_id]
        if len(kept) == len(store["items"]):
            raise HTTPException(status_code=404, detail="No announcement with that id")
        store["items"] = kept
        _save(store)
    logging.info("Announcement %s deleted by %s", announcement_id, user["email"])
    return {"status": "deleted", "id": announcement_id}
