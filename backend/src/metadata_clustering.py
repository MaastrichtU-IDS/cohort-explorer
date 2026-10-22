"""Metadata clustering: for every study-level metadata element (a cohort's study
objective, population, outcomes, inclusion and exclusion criteria) the closest
matching elements in the other studies.

Two passes, merged symmetrically:

  lexical  token-set similarity (thefuzz) and a TF-IDF cosine over the whole
           element corpus. Same-field pairs (study A's objective vs study B's
           objective) are kept above FIELD_THRESHOLD; cross-field pairs (A's
           inclusion text vs B's exclusion text) only above the stricter
           CROSS_FIELD_THRESHOLD, since the same information can sit under a
           different heading in another study.
  ai       the chat LLM (settings.litellm_*) judges, for each element, a
           candidate list: the same field of every other study plus the best
           cross-field lexical candidates. Optional operator instructions are
           appended to the system prompt. A match the LLM found that the
           lexical pass did not is tagged "ai"; one both found is "both".

Symmetry: a match is a property of the unordered element pair, so if A lists B
then B lists A with the same tag, score and reason.

Results are one JSON file under <data_folder>/metadata_clustering/ with the run
parameters, so the review page and the cohort pages load them instantly.
Endpoints (prefix /api):
  POST /metadata-clustering/run        {instructions?}   admin, starts the job
  GET  /metadata-clustering/status                       job state
  GET  /metadata-clustering/results                      the whole result
"""

import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# The metadata elements compared, grouped as the cohort page shows them
# ---------------------------------------------------------------------------

FIELD_GROUPS: list[dict[str, Any]] = [
    {"group": "Objective", "fields": [("study_objective", "Study Objective")]},
    {"group": "Population", "fields": [("study_population", "Study Population"),
                                       ("morbidity", "Morbidity")]},
    {"group": "Outcomes", "fields": [("primary_outcome_spec", "Primary Outcome Specification"),
                                     ("secondary_outcome_spec", "Secondary Outcome Specification")]},
    {"group": "Inclusion criteria", "fields": [
        ("sex_inclusion", "Sex (inclusion)"),
        ("health_status_inclusion", "Health Status (inclusion)"),
        ("clinically_relevant_exposure_inclusion", "Clinically Relevant Exposure (inclusion)"),
        ("age_group_inclusion", "Age Group (inclusion)"),
        ("bmi_range_inclusion", "BMI Range (inclusion)"),
        ("ethnicity_inclusion", "Ethnicity (inclusion)"),
        ("family_status_inclusion", "Family Status (inclusion)"),
        ("hospital_patient_inclusion", "Hospital Patient (inclusion)"),
        ("use_of_medication_inclusion", "Use of Medication (inclusion)"),
    ]},
    {"group": "Exclusion criteria", "fields": [
        ("health_status_exclusion", "Health Status (exclusion)"),
        ("bmi_range_exclusion", "BMI Range (exclusion)"),
        ("limited_life_expectancy_exclusion", "Limited Life Expectancy (exclusion)"),
        ("need_for_surgery_exclusion", "Need for Surgery (exclusion)"),
        ("surgical_procedure_history_exclusion", "Surgical Procedure History (exclusion)"),
        ("clinically_relevant_exposure_exclusion", "Clinically Relevant Exposure (exclusion)"),
    ]},
]
FIELD_LABELS: dict[str, str] = {f: label for g in FIELD_GROUPS for f, label in g["fields"]}
FIELD_GROUP: dict[str, str] = {f: g["group"] for g in FIELD_GROUPS for f, _ in g["fields"]}
FIELDS: list[str] = list(FIELD_LABELS)

# Spreadsheet cells that mean "nothing here"; they must not match each other.
_EMPTY_VALUES = {"", "n/a", "na", "none", "no", "nan", "null", "-", "not applicable",
                 "not specified", "unknown", "unspecified", "no restriction", "no restrictions"}

# Lexical thresholds (0..1). Same-field pairs are expected to match, so the bar
# is lower; a cross-field pair has to be a near paraphrase to count.
FIELD_THRESHOLD = 0.55
CROSS_FIELD_THRESHOLD = 0.75
# Cross-field candidates handed to the LLM even when below the lexical bar.
AI_CROSS_FIELD_CANDIDATES = 4
AI_CROSS_FIELD_MIN_SCORE = 0.3
LLM_WORKERS = 3

_DIR = os.path.join(settings.data_folder, "metadata_clustering")
_STATE_PATH = os.path.join(_DIR, "job.json")
_RESULT_PATH = os.path.join(_DIR, "results.json")
# One line per finished run (done / failed / stalled), newest last.
_HISTORY_PATH = os.path.join(_DIR, "runs.jsonl")
# Snapshots of earlier results, results_<started>.json; the newest few are kept.
_KEEP_SNAPSHOTS = 10
# A running job whose state file has not been touched for this long did not
# survive (the worker died or the container restarted): it is marked failed.
STALL_MINUTES = 15
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Job state and result files (shared by all uvicorn workers)
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(_DIR, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _update_state(**fields: Any) -> dict[str, Any]:
    """Merge fields into the job state; every update stamps updated_at, which
    the stall detection relies on."""
    with _write_lock:
        state = _read_json(_STATE_PATH)
        state.update(fields)
        state["updated_at"] = datetime.now().isoformat()
        _write_json(_STATE_PATH, state)
        return state


def _append_history(state: dict[str, Any]) -> None:
    """Record a finished run (done, failed or stalled) in the run history."""
    entry = {k: state.get(k) for k in ("started", "finished", "phase", "started_by", "instructions", "model",
                                        "ai_available", "counts", "duration_seconds", "error",
                                        "elements", "cohorts", "lexical_matches", "ai_done", "ai_failed")}
    entry["status"] = state.get("phase")
    try:
        os.makedirs(_DIR, exist_ok=True)
        with _write_lock, open(_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("[metadata_clustering] could not write run history: %s", exc)


def list_history() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(_HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return []
    return list(reversed(out))


def _snapshot_results(started: Optional[str]) -> None:
    """Keep a copy of the results of each run, pruning old snapshots."""
    try:
        if not os.path.exists(_RESULT_PATH):
            return
        tag = re.sub(r"[^0-9T]", "", (started or datetime.now().isoformat())[:19])
        import shutil
        shutil.copyfile(_RESULT_PATH, os.path.join(_DIR, f"results_{tag}.json"))
        snaps = sorted(f for f in os.listdir(_DIR) if f.startswith("results_") and f.endswith(".json"))
        for old in snaps[:-_KEEP_SNAPSHOTS]:
            os.remove(os.path.join(_DIR, old))
    except OSError as exc:
        logger.warning("[metadata_clustering] could not snapshot results: %s", exc)


def check_stalled() -> dict[str, Any]:
    """The job state, after marking a running job failed when it has made no
    progress for STALL_MINUTES. The job runs as a background task inside one
    API worker and does not need anyone on the page; a job that stops
    updating its state is one whose worker died or was restarted."""
    state = _read_json(_STATE_PATH)
    if not state.get("running"):
        return state
    try:
        last = datetime.fromisoformat(state.get("updated_at") or state.get("started"))
    except (TypeError, ValueError):
        last = None
    if last is None or datetime.now() - last > timedelta(minutes=STALL_MINUTES):
        since = last.isoformat(timespec="minutes") if last else "an unknown time"
        state = _update_state(running=False, phase="failed", finished=datetime.now().isoformat(),
                              error=f"stalled: no progress since {since} (more than {STALL_MINUTES} minutes); "
                                    "the worker running it crashed or was restarted")
        _append_history(state)
        logger.warning("[metadata_clustering] run started %s marked as stalled", state.get("started"))
    return state


def _require_admin(user: dict) -> None:
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admins only.")


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return "" if text.lower().strip(" .") in _EMPTY_VALUES else text


def collect_elements() -> list[dict[str, Any]]:
    """Every non-empty metadata element of every cohort:
    {id, cohort, field, label, group, text}."""
    from src.cohort_cache import get_cohorts_from_cache
    admin_email = settings.admins_list[0] if settings.admins_list else ""
    cohorts = get_cohorts_from_cache(admin_email)
    elements = []
    for cohort_id in sorted(cohorts, key=str.lower):
        cohort = cohorts[cohort_id]
        for field in FIELDS:
            text = _clean_text(getattr(cohort, field, None))
            if not text:
                continue
            elements.append({
                "id": f"{cohort_id}::{field}",
                "cohort": cohort_id,
                "field": field,
                "label": FIELD_LABELS[field],
                "group": FIELD_GROUP[field],
                "text": text,
            })
    return elements


# ---------------------------------------------------------------------------
# Lexical pass
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/%\.]*")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "without",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "that", "this", "these",
    "those", "it", "its", "their", "than", "then", "who", "which", "will", "not", "no",
    "any", "all", "other", "patients", "patient", "study", "subjects", "subject",
}


def _tokens(text: str) -> list[str]:
    return [t.strip(".-/") for t in _TOKEN_RE.findall(text.lower())
            if t.strip(".-/") and t.strip(".-/") not in _STOPWORDS]


def _tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    """Length-normalized TF-IDF vectors (sparse dicts) over the element corpus."""
    docs = [_tokens(t) for t in texts]
    df: dict[str, int] = defaultdict(int)
    for toks in docs:
        for term in set(toks):
            df[term] += 1
    n = max(len(docs), 1)
    vectors = []
    for toks in docs:
        tf: dict[str, float] = defaultdict(float)
        for term in toks:
            tf[term] += 1.0
        vec = {term: (1 + math.log(c)) * math.log((1 + n) / (1 + df[term])) + 1e-9 for term, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({term: v / norm for term, v in vec.items()})
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[t] for t, v in a.items() if t in b)


def lexical_pass(elements: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    """Scores for every element pair from different cohorts that clears its
    threshold, keyed by the index pair (i < j). Each value: {score, fuzzy, tfidf,
    same_field}."""
    from thefuzz import fuzz
    vectors = _tfidf_vectors([e["text"] for e in elements])
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    n = len(elements)
    for i in range(n):
        ei = elements[i]
        for j in range(i + 1, n):
            ej = elements[j]
            if ei["cohort"] == ej["cohort"]:
                continue
            same_field = ei["field"] == ej["field"]
            fuzzy = fuzz.token_set_ratio(ei["text"], ej["text"]) / 100.0
            tfidf = _cosine(vectors[i], vectors[j])
            # token_set_ratio is generous on short strings (a shared word can
            # score 100), so it is damped unless TF-IDF agrees somewhat.
            score = max(tfidf, fuzzy if tfidf >= 0.2 else fuzzy * 0.7)
            pairs[(i, j)] = {"score": round(score, 3), "fuzzy": round(fuzzy, 3),
                             "tfidf": round(tfidf, 3), "same_field": same_field}
    return pairs


def _passes_lexical(p: dict[str, Any]) -> bool:
    return p["score"] >= (FIELD_THRESHOLD if p["same_field"] else CROSS_FIELD_THRESHOLD)


# ---------------------------------------------------------------------------
# AI pass
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You compare study-level metadata of clinical cohort studies (heart failure and cardiovascular research).
You are given ONE element of one study (for example its study objective, its study population, an outcome specification, or one inclusion or exclusion criterion) and a numbered list of candidate elements from OTHER studies.

Decide which candidates describe the same or closely overlapping information as the given element:
- "strong": the two elements say essentially the same thing (same objective, same population, same outcome, same criterion), even if worded differently or at a different level of detail.
- "partial": clearly related and overlapping, but one is broader, narrower, or covers only part of the other.
Do not match elements that merely belong to the same kind of field or share generic words. Judge the medical content.
A candidate from a different field (e.g. an exclusion criterion against an inclusion criterion) can still match when it carries the same information.

For every match, name WHAT is shared as one or two short themes of two to five words each: concrete medical or
methodological terms such as "kidney failure", "NYHA class II-IV", "all-cause mortality", "NT-proBNP guided therapy",
"adults over 18". No sentences, no field names, no study names.

Answer with STRICT JSON only, no prose, no code fences:
{"matches": [{"id": <candidate number>, "strength": "strong" | "partial", "themes": ["<2-5 words>", "<2-5 words>"], "reason": "<one short sentence>"}]}
Return {"matches": []} when nothing matches."""

_MERGE_PROMPT = """You organise a vocabulary of short themes that describe what pairs of clinical cohort studies
(heart failure and cardiovascular research) have in common in their metadata. You receive the raw themes with how
often each occurred.

Task 1 - normalise: merge themes that mean the same thing (synonyms, singular/plural, wording variants, different
granularity of the same idea such as "renal impairment" / "kidney failure" / "chronic kidney disease") into one
canonical theme of two to five words, lowercase. Keep genuinely different themes apart. Every raw theme must map to
exactly one canonical theme.

Task 2 - group: sort the canonical themes under five to eight parent themes with short names (two to four words),
e.g. "heart failure phenotype", "comorbidities", "age and eligibility", "outcomes", "treatment context". Every
canonical theme belongs to exactly one parent.

Answer with STRICT JSON only, no prose, no code fences:
{"parents": [{"id": "p1", "label": "<parent name>"}],
 "themes": [{"id": "t1", "label": "<canonical theme>", "parent": "p1", "raw": ["<raw theme>", "..."]}]}"""


def _ai_judge(client: Any, model: str, element: dict[str, Any], candidates: list[dict[str, Any]],
              instructions: str) -> tuple[dict[str, dict[str, str]], Optional[str]]:
    """The LLM's verdict on one element: {candidate_id: {strength, reason}} and an
    error message when the call or its parsing failed."""
    system = _SYSTEM_PROMPT
    if instructions:
        system += "\n\nAdditional instructions from the operator (follow them):\n" + instructions.strip()
    lines = [f"ELEMENT ({element['cohort']} — {element['label']}):\n{element['text']}\n", "CANDIDATES:"]
    for n, c in enumerate(candidates, 1):
        lines.append(f"{n}. [{c['cohort']} — {c['label']}] {c['text']}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": "\n".join(lines)}],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
    except Exception as exc:  # network, parsing: the element simply gets no AI verdict
        return {}, f"{type(exc).__name__}: {str(exc)[:200]}"
    verdicts: dict[str, dict[str, str]] = {}
    for m in parsed.get("matches") or []:
        try:
            idx = int(m.get("id")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= idx < len(candidates):
            strength = str(m.get("strength") or "partial").lower()
            themes = []
            for raw in (m.get("themes") or [])[:3]:
                t = " ".join(str(raw or "").lower().replace("_", " ").split())
                t = re.sub(r"[^a-z0-9 +/%<>=≥≤.-]", "", t).strip(" .")
                if t and 1 <= len(t.split()) <= 6 and t not in themes:
                    themes.append(t[:60])
            verdicts[candidates[idx]["id"]] = {
                "strength": "strong" if strength.startswith("strong") else "partial",
                "reason": " ".join(str(m.get("reason") or "").split())[:400],
                "themes": themes,
            }
    return verdicts, None


def _ai_candidates(i: int, elements: list[dict[str, Any]],
                   pairs: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    """The same field of every other study, plus the best-scoring cross-field
    elements, so the LLM can confirm what the lexical pass only hinted at."""
    e = elements[i]
    same = [c for c in elements if c["field"] == e["field"] and c["cohort"] != e["cohort"]]
    cross = []
    for j, c in enumerate(elements):
        if j == i or c["cohort"] == e["cohort"] or c["field"] == e["field"]:
            continue
        p = pairs.get((min(i, j), max(i, j)))
        if p and p["score"] >= AI_CROSS_FIELD_MIN_SCORE:
            cross.append((p["score"], c))
    cross.sort(key=lambda x: -x[0])
    return same + [c for _, c in cross[:AI_CROSS_FIELD_CANDIDATES]]


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "theme"


def merge_themes(client: Any, model: str, raw_counts: dict[str, int]) -> dict[str, Any]:
    """The two-level vocabulary: {parents: [...], themes: [{id, label, parent, raw}], raw_to_theme: {raw: id}}.
    Without a model (or when its answer is unusable) every raw theme is its own
    canonical theme under one parent, so the views still work."""
    raws = sorted(raw_counts, key=lambda r: -raw_counts[r])
    fallback = {
        "parents": [{"id": "p-unsorted", "label": "unsorted themes"}],
        "themes": [{"id": "t-" + _slug(r), "label": r, "parent": "p-unsorted", "raw": [r]} for r in raws],
    }
    fallback["raw_to_theme"] = {r: "t-" + _slug(r) for r in raws}
    if client is None or not raws:
        return fallback
    listing = "\n".join(f"- {r} ({raw_counts[r]}x)" for r in raws[:400])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _MERGE_PROMPT},
                      {"role": "user", "content": f"RAW THEMES ({len(raws)}):\n{listing}"}],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
    except Exception as exc:
        logger.warning("[metadata_clustering] theme merge failed, keeping raw themes: %s", exc)
        return fallback
    parents, themes, raw_to_theme, seen_p, seen_t = [], [], {}, set(), set()
    for pp in parsed.get("parents") or []:
        label = " ".join(str(pp.get("label") or "").lower().split())
        if label and label not in seen_p:
            seen_p.add(label)
            parents.append({"id": "p-" + _slug(label), "label": label, "_src": str(pp.get("id") or "")})
    src_to_parent = {pp["_src"]: pp["id"] for pp in parents}
    for tt in parsed.get("themes") or []:
        label = " ".join(str(tt.get("label") or "").lower().split())
        if not label or label in seen_t:
            continue
        seen_t.add(label)
        parent = src_to_parent.get(str(tt.get("parent") or ""), parents[0]["id"] if parents else "p-unsorted")
        tid = "t-" + _slug(label)
        raw = [" ".join(str(r).lower().split()) for r in (tt.get("raw") or []) if str(r).strip()]
        themes.append({"id": tid, "label": label, "parent": parent, "raw": raw})
        for r in raw:
            raw_to_theme.setdefault(r, tid)
    if not parents or not themes:
        return fallback
    # Raw themes the model forgot keep their own canonical entry.
    for r in raws:
        if r not in raw_to_theme:
            tid = "t-" + _slug(r)
            if tid not in {t["id"] for t in themes}:
                themes.append({"id": tid, "label": r, "parent": parents[0]["id"], "raw": [r]})
            raw_to_theme[r] = tid
    for pp in parents:
        pp.pop("_src", None)
    if not any(p["id"] == "p-unsorted" for p in parents) and any(t["parent"] == "p-unsorted" for t in themes):
        parents.append({"id": "p-unsorted", "label": "unsorted themes"})
    return {"parents": parents, "themes": themes, "raw_to_theme": raw_to_theme}


_STRENGTH_W = {"strong": 1.0, "partial": 0.6}


def build_study_themes(elements: list[dict[str, Any]], pair_records: list[dict[str, Any]],
                       vocab: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """study -> theme id -> {strength, count, excerpts}. A study carries a theme
    when any of its elements matched another study's element on it; strength is
    the best match that carried it, count how many did."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in pair_records:
        if not rec.get("themes"):
            continue
        w = _STRENGTH_W.get(rec.get("ai_strength") or "partial", 0.6)
        a, b = elements[rec["i"]], elements[rec["j"]]
        for raw in rec["themes"]:
            tid = vocab["raw_to_theme"].get(raw)
            if not tid:
                continue
            for me, other in ((a, b), (b, a)):
                cell = table.setdefault(me["cohort"], {}).setdefault(tid, {"strength": 0.0, "count": 0, "excerpts": []})
                cell["strength"] = max(cell["strength"], w)
                cell["count"] += 1
                if len(cell["excerpts"]) < 6 and not any(x["text"] == me["text"] and x["with"] == other["cohort"] for x in cell["excerpts"]):
                    cell["excerpts"].append({"field": me["label"], "text": me["text"][:400], "with": other["cohort"],
                                             "with_text": other["text"][:400], "strength": rec.get("ai_strength")})
    return table


# Themes an admin took out of the views (not out of the stored results).
_EXCLUDED_PATH = os.path.join(_DIR, "excluded_themes.json")


def excluded_themes() -> set[str]:
    data = _read_json(_EXCLUDED_PATH)
    return set(data.get("excluded") or [])


def theme_balance(n: int, n_total: int) -> float:
    """How informative the split a theme induces is: the binary entropy of the
    carrier fraction p = n / N, in bits. 1 when half the studies carry the theme,
    0 when all or none do, low at both ends - so a theme shared by everyone
    (adults over 18) and a theme shared by one pair are both down-weighted,
    and themes that divide the catalogue in two weigh most. The criterion
    information gain uses to pick a splitting feature."""
    if n_total <= 1 or n <= 0 or n >= n_total:
        return 0.0
    p = n / n_total
    return round(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)), 3)


def theme_specificity(n: int, n_total: int) -> float:
    """Inverse-frequency weight: log(N / n) / log(N), 0 when every study carries
    the theme. Themes come from pairs, so n is at least 2 and the maximum is
    log(N / 2) / log(N). Rarer is always heavier, which over-rewards one-pair
    themes; kept as an alternative to the balance weight."""
    if n <= 0 or n_total <= 1:
        return 0.0
    return round(math.log(n_total / n) / math.log(n_total), 3)


def theme_view(data: dict[str, Any]) -> dict[str, Any]:
    """The study x theme projection for the heatmap and the constellation. Each
    theme carries two weights, balance (default) and specificity; a pair's
    strength is the sum over shared themes of min(strength) x weight, computed
    for both so the views can switch."""
    themes_info = data.get("themes") or {}
    vocab_themes = themes_info.get("themes") or []
    parents = themes_info.get("parents") or []
    table = themes_info.get("table") or {}
    studies = sorted({e["cohort"] for e in data.get("elements") or []} | set(table), key=str.lower)
    n_total = max(len(studies), 1)
    excluded = excluded_themes()
    themes_out = []
    for t in vocab_themes:
        carriers = [s for s in studies if t["id"] in (table.get(s) or {})]
        n = len(carriers)
        themes_out.append({**t, "studies": n, "specificity": theme_specificity(n, n_total),
                           "balance": theme_balance(n, n_total), "excluded": t["id"] in excluded})
    active = {t["id"]: t for t in themes_out if not t["excluded"] and t["studies"]}
    pairs = []
    for i, a in enumerate(studies):
        for b in studies[i + 1:]:
            shared = []
            for tid, ta in (table.get(a) or {}).items():
                tb = (table.get(b) or {}).get(tid)
                if tb and tid in active:
                    shared.append({"theme": tid, "parent": active[tid]["parent"], "w": round(min(ta["strength"], tb["strength"]), 3),
                                   "specificity": active[tid]["specificity"], "balance": active[tid]["balance"]})
            if shared:
                pairs.append({"a": a, "b": b, "shared": shared, "n": len(shared),
                              "strength": round(sum(x["w"] * x["balance"] for x in shared), 3),
                              "strength_specificity": round(sum(x["w"] * x["specificity"] for x in shared), 3),
                              "raw_strength": round(sum(x["w"] for x in shared), 3)})
    return {"finished": data.get("finished"), "model": data.get("model"), "studies": studies,
            "n_studies": n_total, "parents": parents, "themes": themes_out, "table": table, "pairs": pairs,
            "excluded": sorted(excluded), "weights": ["balance", "specificity"]}


def run_clustering(instructions: str = "") -> None:
    started = time.time()
    try:
        elements = collect_elements()
        index = {e["id"]: k for k, e in enumerate(elements)}
        _update_state(phase="lexical", elements=len(elements),
                      cohorts=len({e["cohort"] for e in elements}))
        pairs = lexical_pass(elements)
        lexical_hits = {k for k, p in pairs.items() if _passes_lexical(p)}
        _update_state(phase="ai", lexical_matches=len(lexical_hits), ai_done=0, ai_failed=0)

        # ai verdicts per unordered pair: {(i,j): {strength, reason, by: [i or j...]}}
        ai: dict[tuple[int, int], dict[str, Any]] = {}
        ai_failed: dict[str, str] = {}
        ai_available = bool(settings.litellm_base_url)
        if ai_available:
            from src.chat import _get_openai_client
            client = _get_openai_client()
            model = settings.litellm_model
            done = 0

            def judge(i: int):
                cands = _ai_candidates(i, elements, pairs)
                if not cands:
                    return i, {}, None
                return i, *_ai_judge(client, model, elements[i], cands, instructions)

            with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
                futures = [pool.submit(judge, i) for i in range(len(elements))]
                for fut in as_completed(futures):
                    i, verdicts, err = fut.result()
                    done += 1
                    if err:
                        ai_failed[elements[i]["id"]] = err
                        logger.warning("[metadata_clustering] LLM failed for %s: %s", elements[i]["id"], err)
                    for cid, v in verdicts.items():
                        j = index[cid]
                        key = (min(i, j), max(i, j))
                        cur = ai.setdefault(key, {"strength": v["strength"], "reason": v["reason"], "votes": 0, "themes": []})
                        cur["votes"] += 1
                        for t in v.get("themes") or []:
                            if t not in cur["themes"]:
                                cur["themes"].append(t)
                        # Two directions may disagree; keep the stronger verdict and
                        # the first reason given for it.
                        if v["strength"] == "strong" and cur["strength"] != "strong":
                            cur["strength"], cur["reason"] = "strong", v["reason"]
                    # Every element: the state's updated_at is the heartbeat for stall detection.
                    _update_state(ai_done=done, ai_failed=len(ai_failed))
                    if done % 10 == 0 or done == len(elements):
                        logger.info("[metadata_clustering] LLM %d/%d elements judged", done, len(elements))
        else:
            logger.warning("[metadata_clustering] LITELLM_BASE_URL not set; lexical pass only")

        # Merge, symmetrically: one record per unordered pair, listed on both sides.
        matches: dict[str, list[dict[str, Any]]] = {e["id"]: [] for e in elements}
        counts = {"lexical": 0, "ai": 0, "both": 0}
        pair_records: list[dict[str, Any]] = []
        for key in sorted(set(lexical_hits) | set(ai)):
            i, j = key
            p = pairs.get(key) or {"score": 0.0, "fuzzy": 0.0, "tfidf": 0.0, "same_field": elements[i]["field"] == elements[j]["field"]}
            in_lex, in_ai = key in lexical_hits, key in ai
            source = "both" if (in_lex and in_ai) else ("lexical" if in_lex else "ai")
            counts[source] += 1
            rec = {"source": source, "score": p["score"], "fuzzy": p["fuzzy"], "tfidf": p["tfidf"],
                   "same_field": p["same_field"],
                   "ai_strength": ai[key]["strength"] if in_ai else None,
                   "ai_reason": ai[key]["reason"] if in_ai else None,
                   "ai_votes": ai[key]["votes"] if in_ai else 0,
                   "themes": ai[key]["themes"] if in_ai else []}
            pair_records.append({"i": i, "j": j, **rec})
            for a, b in ((i, j), (j, i)):
                matches[elements[a]["id"]].append({**rec, "cohort": elements[b]["cohort"], "field": elements[b]["field"],
                                                   "label": elements[b]["label"], "text": elements[b]["text"],
                                                   "element_id": elements[b]["id"]})
        for lst in matches.values():
            lst.sort(key=lambda m: (-(m["ai_strength"] == "strong"), -(m["source"] != "lexical"), -m["score"]))

        # The theme vocabulary (merge pass) and the study x theme table.
        _update_state(phase="themes")
        raw_counts: dict[str, int] = {}
        for rec in pair_records:
            for t in rec["themes"]:
                raw_counts[t] = raw_counts.get(t, 0) + 1
        vocab = merge_themes(client if ai_available else None, settings.litellm_model if ai_available else "", raw_counts)
        table = build_study_themes(elements, pair_records, vocab)
        logger.info("[metadata_clustering] themes: %d raw -> %d canonical under %d parents",
                    len(raw_counts), len(vocab["themes"]), len(vocab["parents"]))

        result = {
            "started": datetime.fromtimestamp(started).isoformat(),
            "finished": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - started),
            "model": settings.litellm_model if ai_available else None,
            "ai_available": ai_available,
            "instructions": instructions,
            "fields": FIELD_GROUPS,
            "thresholds": {"same_field": FIELD_THRESHOLD, "cross_field": CROSS_FIELD_THRESHOLD},
            "counts": {**counts, "elements": len(elements), "cohorts": len({e["cohort"] for e in elements}),
                       "pairs": sum(counts.values()), "ai_failed": len(ai_failed)},
            "ai_failed": ai_failed,
            "elements": elements,
            "matches": matches,
            "themes": {"parents": vocab["parents"], "themes": vocab["themes"], "raw_to_theme": vocab["raw_to_theme"],
                       "raw_counts": raw_counts, "table": table},
        }
        _write_json(_RESULT_PATH, result)
        state = _update_state(running=False, phase="done", finished=datetime.now().isoformat(), error=None,
                              counts=result["counts"], duration_seconds=result["duration_seconds"])
        _append_history(state)
        _snapshot_results(state.get("started"))
        logger.info("[metadata_clustering] done in %ds: %s", result["duration_seconds"], result["counts"])
    except Exception as exc:
        logger.exception("[metadata_clustering] run failed")
        state = _update_state(running=False, phase="failed", finished=datetime.now().isoformat(),
                              error=f"{type(exc).__name__}: {exc}")
        _append_history(state)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/metadata-clustering/run")
async def start_run(background: BackgroundTasks, body: Optional[dict[str, Any]] = None,
                    user: Any = Depends(get_current_user)):
    _require_admin(user)
    if check_stalled().get("running"):
        raise HTTPException(status_code=409, detail="A metadata clustering run is already in progress.")
    instructions = str((body or {}).get("instructions") or "").strip()[:4000]
    _update_state(running=True, phase="starting", started=datetime.now().isoformat(), finished=None,
                  error=None, started_by=user["email"], instructions=instructions,
                  ai_available=bool(settings.litellm_base_url), model=settings.litellm_model)
    background.add_task(run_clustering, instructions)
    return {"started": True, "ai_available": bool(settings.litellm_base_url)}


@router.get("/metadata-clustering/status")
async def status(user: Any = Depends(get_current_user)):
    _require_admin(user)
    state = check_stalled()
    state["has_results"] = os.path.exists(_RESULT_PATH)
    state["is_admin"] = True
    state["fields"] = FIELD_GROUPS
    state["stall_minutes"] = STALL_MINUTES
    return state


@router.get("/metadata-clustering/runs")
async def runs(user: Any = Depends(get_current_user)):
    """Previous runs, newest first (done, failed and stalled alike)."""
    _require_admin(user)
    check_stalled()
    return {"items": list_history()}


@router.get("/metadata-clustering/themes")
async def themes(user: Any = Depends(get_current_user)):
    """The study x theme view: vocabulary with parents and specificity, the table
    with excerpts, and the study pairs weighted by specificity."""
    _require_admin(user)
    data = _read_json(_RESULT_PATH)
    if not data:
        raise HTTPException(status_code=404, detail="No metadata clustering results yet.")
    if not data.get("themes"):
        raise HTTPException(status_code=409, detail="The stored results predate theme extraction: run the matching again.")
    return theme_view(data)


@router.post("/metadata-clustering/themes/exclude")
async def exclude_theme(body: dict[str, Any], user: Any = Depends(get_current_user)):
    """Take a theme out of (or back into) the views; the stored results are untouched."""
    _require_admin(user)
    tid = str(body.get("theme_id") or "")
    if not tid:
        raise HTTPException(status_code=400, detail="theme_id is required")
    excluded = excluded_themes()
    if body.get("excluded", True):
        excluded.add(tid)
    else:
        excluded.discard(tid)
    _write_json(_EXCLUDED_PATH, {"excluded": sorted(excluded), "updated_at": datetime.now().isoformat(),
                                 "updated_by": user["email"]})
    return {"excluded": sorted(excluded)}


@router.get("/metadata-clustering/results")
async def results(user: Any = Depends(get_current_user)):
    _require_admin(user)
    data = _read_json(_RESULT_PATH)
    if not data:
        raise HTTPException(status_code=404, detail="No metadata clustering results yet.")
    return data
