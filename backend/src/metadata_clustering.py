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
  GET  /metadata-clustering/cohort/{cohort_id}           one study's matches
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

Answer with STRICT JSON only, no prose, no code fences:
{"matches": [{"id": <candidate number>, "strength": "strong" | "partial", "reason": "<one short sentence>"}]}
Return {"matches": []} when nothing matches."""


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
            verdicts[candidates[idx]["id"]] = {
                "strength": "strong" if strength.startswith("strong") else "partial",
                "reason": " ".join(str(m.get("reason") or "").split())[:400],
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
                        cur = ai.setdefault(key, {"strength": v["strength"], "reason": v["reason"], "votes": 0})
                        cur["votes"] += 1
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
                   "ai_votes": ai[key]["votes"] if in_ai else 0}
            for a, b in ((i, j), (j, i)):
                matches[elements[a]["id"]].append({**rec, "cohort": elements[b]["cohort"], "field": elements[b]["field"],
                                                   "label": elements[b]["label"], "text": elements[b]["text"],
                                                   "element_id": elements[b]["id"]})
        for lst in matches.values():
            lst.sort(key=lambda m: (-(m["ai_strength"] == "strong"), -(m["source"] != "lexical"), -m["score"]))

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


@router.get("/metadata-clustering/results")
async def results(user: Any = Depends(get_current_user)):
    _require_admin(user)
    data = _read_json(_RESULT_PATH)
    if not data:
        raise HTTPException(status_code=404, detail="No metadata clustering results yet.")
    return data


@router.get("/metadata-clustering/cohort/{cohort_id}")
async def cohort_matches(cohort_id: str, user: Any = Depends(get_current_user)):
    """One study's elements with their matches, for the cohort page (admins only:
    the panel hides itself for other users)."""
    _require_admin(user)
    data = _read_json(_RESULT_PATH)
    if not data:
        return {"available": False, "elements": []}
    wanted = cohort_id.lower()
    out = [{**e, "matches": data["matches"].get(e["id"], [])}
           for e in data["elements"] if e["cohort"].lower() == wanted]
    return {"available": True, "finished": data.get("finished"), "model": data.get("model"),
            "elements": out}
