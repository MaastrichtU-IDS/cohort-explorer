import json
import re
from functools import lru_cache
from pathlib import Path

_HIERARCHY_PATH = Path(__file__).parent / "icd10_hierarchy.json"

_LEAF_RE = re.compile(r"^[A-Z][0-9]{2}(\.[0-9]{1,4})?$")
_BLOCK_RE = re.compile(r"^[A-Z][0-9]{2}-[A-Z][0-9]{2}$")
_ROMAN_RE = re.compile(r"^(X{0,3})(IX|IV|V?I{0,3})$")

_MAX_DEPTH = 15


@lru_cache(maxsize=1)
def _data() -> dict:
    with open(_HIERARCHY_PATH) as f:
        raw = json.load(f)
    parents = {k.upper(): v.upper() for k, v in raw["parents"].items()}
    leaves = {c.upper() for c in raw["requester_leaves"]}
    labels = {k.upper(): v for k, v in raw.get("labels", {}).items()}
    chapter_ranges = {v.upper() for v in raw.get("chapters", {}).values()}
    all_codes = set(parents) | set(parents.values()) | leaves
    return {
        "parents": parents,
        "leaves": leaves,
        "labels": labels,
        "chapter_ranges": chapter_ranges,
        "all_codes": all_codes,
    }


def normalize(code: str) -> str:
    return (code or "").strip().upper()


def is_roman_chapter(code: str) -> bool:
    c = normalize(code)
    return bool(c) and bool(_ROMAN_RE.match(c))


def is_well_formed(code: str) -> bool:
    c = normalize(code)
    return bool(_LEAF_RE.match(c) or _BLOCK_RE.match(c) or is_roman_chapter(c))


def is_known_code(code: str) -> bool:
    return normalize(code) in _data()["all_codes"]


def is_requester_leaf(code: str) -> bool:
    return normalize(code) in _data()["leaves"]


def label(code: str) -> str | None:
    return _data()["labels"].get(normalize(code))


def level(code: str) -> str:
    c = normalize(code)
    if is_roman_chapter(c) or c in _data()["chapter_ranges"]:
        return "chapter"
    if _BLOCK_RE.match(c):
        return "block"
    return "leaf"


def ancestors(code: str) -> list[str]:
    parents = _data()["parents"]
    out: list[str] = []
    cur = normalize(code)
    for _ in range(_MAX_DEPTH):
        cur = parents.get(cur)
        if not cur or cur in out:
            break
        out.append(cur)
    return out


def is_compatible(consent_code: str | None, requested_code: str | None) -> bool:
    consent = normalize(consent_code)
    requested = normalize(requested_code)
    if not consent:
        return True
    if consent == requested:
        return True
    return consent in ancestors(requested)


def describe(code: str) -> dict:
    c = normalize(code)
    if not is_known_code(c):
        return {"valid": False, "id": c, "error": f"Unknown ICD-10 code: {code!r}"}
    return {
        "valid": True,
        "id": c,
        "name": label(c) or c,
        "level": level(c),
        "is_requester_selectable": is_requester_leaf(c),
        "parents": ancestors(c),
    }


def search(query: str, limit: int = 10) -> list[dict]:
    q = (query or "").strip().lower()
    data = _data()
    hits: list[tuple[int, str]] = []
    for c in data["all_codes"]:
        name = data["labels"].get(c, "")
        haystack = f"{c} {name}".lower()
        if not q or q in haystack:
            if c.lower() == q:
                rank = 0
            elif c.lower().startswith(q):
                rank = 1
            elif q in c.lower():
                rank = 2
            else:
                rank = 3
            hits.append((rank, c))
    hits.sort(key=lambda t: (t[0], t[1]))
    return [
        {
            "id": c,
            "name": data["labels"].get(c, c),
            "level": level(c),
            "is_requester_selectable": c in data["leaves"],
        }
        for _, c in hits[:limit]
    ]


def hierarchy_edges() -> list[tuple[str, str]]:
    return sorted(_data()["parents"].items())
