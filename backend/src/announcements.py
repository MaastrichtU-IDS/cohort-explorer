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

ANNOUNCEMENT_TAGS = ("new cohort", "new feature", "analysis")
MAX_TEXT_LENGTH = 500  # announcements are at most a couple of sentences

_LOCK = threading.Lock()


def _store_path() -> str:
    return os.path.join(settings.data_folder, "announcements.json")


def _load() -> list[dict]:
    try:
        with open(_store_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception as exc:
        logging.warning("Announcements store unreadable: %s", exc)
        return []


def _save(items: list[dict]) -> None:
    path = _store_path()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _is_admin(user: Any) -> bool:
    return user["email"].lower() in settings.admins_list


@router.get("/announcements", name="List announcements (newest first)")
def list_announcements(user: Any = Depends(get_current_user)) -> list[dict]:
    items = sorted(_load(), key=lambda a: (str(a.get("date") or ""), str(a.get("created_at") or "")),
                   reverse=True)
    if _is_admin(user):
        return items
    # Non-admins (the front page) never see who added an announcement.
    return [{k: v for k, v in a.items() if k != "added_by"} for a in items]


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
        items = _load()
        items.append(item)
        _save(items)
    logging.info("Announcement %s added by %s (%s, %s)", item["id"], item["added_by"], tag, raw_date)
    return item


@router.delete("/announcements/{announcement_id}", name="Delete an announcement (admins only)")
def delete_announcement(announcement_id: str, user: Any = Depends(get_current_user)) -> dict:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    with _LOCK:
        items = _load()
        kept = [a for a in items if a.get("id") != announcement_id]
        if len(kept) == len(items):
            raise HTTPException(status_code=404, detail="No announcement with that id")
        _save(kept)
    logging.info("Announcement %s deleted by %s", announcement_id, user["email"])
    return {"status": "deleted", "id": announcement_id}
