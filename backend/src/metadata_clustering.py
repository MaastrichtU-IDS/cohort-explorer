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
           Every AI match carries a structured explanation written BEFORE its
           tags (shared: what both say, with quotes; differs; basis) and raw
           candidate tags derived from the "shared" part only.

Themes, in three steps after the matching (also runnable alone on stored
results, POST /metadata-clustering/rebuild-themes):
  cleanup     raw tags that only restate a field ("age inclusion criterion",
              "morbidity") are dropped; spelling variants are folded.
  vocabulary  the cleaned terms are merged into canonical themes under at
              most eight parents, in batches of VOCAB_BATCH terms, each batch
              seeing the vocabulary built so far (one big call failed on the
              local model and left 985 raw tags unsorted).
  assignment  every AI match gets one to three canonical themes chosen from
              that closed vocabulary, given its "shared" explanation and the
              themes its raw tags map to as suggestions; ASSIGN_BATCH matches
              per call. The study x theme table is built from these.

Symmetry: a match is a property of the unordered element pair, so if A lists B
then B lists A with the same tag, score and reason.

Results are one JSON file under <data_folder>/metadata_clustering/ with the run
parameters, so the review page and the cohort pages load them instantly.
Endpoints (prefix /api):
  POST /metadata-clustering/run        {instructions?}   admin, starts the job
  POST /metadata-clustering/rebuild-themes               admin, themes only, on stored results
  GET  /metadata-clustering/vocabulary                   the two-level theme vocabulary with carriers
  POST /metadata-clustering/vocabulary/edit  {op, ...}   admin: merge / move / nest / rename / delete_parent
  POST /metadata-clustering/vocabulary/undo              admin: restore the results from before the last edit
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
# Theme vocabulary: cleaned raw terms per merge call, and AI matches per
# theme-assignment call. Small enough for an 8B model to answer in full JSON.
VOCAB_BATCH = 60
ASSIGN_BATCH = 25
MAX_PARENTS = 8

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

For every match, explain it BEFORE naming themes, in this order:
- "shared": two or three sentences on what both elements say, quoting from each element the phrase that carries it (short quotes in double quotes). Only what is written in the two texts: no assumptions, no numbers or conditions that are not there.
- "differs": one or two sentences on what one element says that the other does not, or which one is broader or narrower and how. Write "nothing material" when they say the same.
- "basis": "content" when the match is in the medical or methodological content, "field" when the two merely share the kind of information (then prefer not to match at all).
- "themes": one to three themes of one to five words each, taken ONLY from what "shared" describes: concrete medical or methodological concepts such as "kidney failure", "NYHA class II-IV", "all-cause mortality", "NT-proBNP guided therapy", "adults over 18". Never a field name (not "age inclusion criterion", not "morbidity"), never a difference (not "narrower age range"), no sentences, no study names.

Answer with STRICT JSON only, no prose, no code fences:
{"matches": [{"id": <candidate number>, "strength": "strong" | "partial", "shared": "<text>", "differs": "<text>", "basis": "content" | "field", "themes": ["<1-5 words>"]}]}
Return {"matches": []} when nothing matches."""

_VOCAB_PROMPT = """You build a controlled vocabulary of themes that describe what pairs of clinical cohort studies (heart failure and cardiovascular research) have in common in their metadata.

You receive the PARENTS and CANONICAL THEMES defined so far, and a batch of RAW TERMS with how often each occurred. For every raw term say which canonical theme it belongs to:
- reuse an existing canonical theme when the term means the same thing: synonyms, singular/plural, wording variants, abbreviations ("hfref" = "reduced ejection fraction"), finer wording of one idea ("renal impairment", "chronic kidney disease", "kidney failure");
- otherwise create a new canonical theme: lowercase, two to five words, the clearest medical or methodological name for it.
Every canonical theme has one parent. Reuse an existing parent whenever one fits; create a new one only when none does. There are at most eight parents in total, with short names of two to four words, for example "heart failure phenotype", "comorbidities", "age and eligibility", "outcomes", "treatment context", "study design".
Every raw term of the batch must appear exactly once, spelled exactly as given.

Answer with STRICT JSON only, no prose, no code fences:
{"map": [{"term": "<raw term as given>", "theme": "<canonical theme label>", "parent": "<parent label>"}]}"""

_PARENTS_PROMPT = """You tidy the parent groups of a vocabulary of clinical-study metadata themes. You receive the current parent labels with how many themes each holds. Group them into at most eight parents with short names of two to four words; a parent may keep its name. Every given label must appear in exactly one group.

Answer with STRICT JSON only, no prose, no code fences:
{"groups": [{"label": "<parent name>", "members": ["<given label>", "..."]}]}"""

_ASSIGN_PROMPT = """You label matches between metadata elements of clinical cohort studies (heart failure and cardiovascular research) with themes from a fixed vocabulary.

You receive the VOCABULARY (numbered themes, grouped by parent) and a list of MATCHES. Each match gives the two elements and the explanation of what they share, and may suggest theme numbers from an earlier pass.
For every match choose one to three theme numbers that name what the explanation says is shared. Use only the vocabulary. Prefer the most specific theme that fits. Keep a suggested theme only when the explanation supports it, and add one the suggestions missed when it clearly applies. Choose none ([]) when no theme fits.

Answer with STRICT JSON only, no prose, no code fences:
{"assign": [{"match": <match number>, "themes": [<theme number>, "..."]}]}"""


# Tags that only restate the field or the shape of the overlap carry nothing a
# graph can use; they are dropped before the vocabulary is built.
_FIELD_WORDS = {"criterion", "criteria", "inclusion", "exclusion", "specification", "spec", "requirement",
                "requirements", "restriction", "restrictions"}
_GENERIC_WORDS = {"sex", "age", "group", "range", "ranges", "outcome", "outcomes", "primary", "secondary", "study",
                  "population", "objective", "objectives", "morbidity", "status", "health", "adult", "adults",
                  "threshold", "overlap", "overlapping", "narrower", "broader", "subset", "both", "included",
                  "include", "includes", "minimum", "maximum", "general", "specific", "similar", "same", "matching",
                  "match", "limit", "limits", "lower", "upper", "years", "year", "old", "of", "and", "or", "the",
                  "a", "an", "in", "with", "for", "to", "on", "by", "at", "cohort", "cohorts", "patients",
                  "participants", "subjects", "element", "elements", "field", "fields", "criterion", "condition",
                  "conditions", "comorbidity", "comorbidities", "multiple", "list", "listed"}


def _squash(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _clean_theme(raw: Any) -> Optional[str]:
    """A raw tag normalised (lowercase, one space, safe characters) or None
    when it only restates a field or the shape of the overlap: 'age inclusion
    criterion', 'morbidity', 'narrower age subset', 'multiple comorbidities'
    carry no medical content. 'adults over 18', 'both sexes included',
    'no sex restriction', 'kidney failure' stay."""
    t = " ".join(str(raw or "").lower().replace("_", " ").split())
    t = re.sub(r"[^a-z0-9 +/%<>=≥≤.-]", "", t).strip(" .-")
    words = t.split()
    if not words or len(words) > 6:
        return None
    if all(w in _FIELD_WORDS or w in _GENERIC_WORDS for w in words):
        return None
    return t[:60]


def _llm_json(client: Any, model: str, system: str, user: str) -> dict[str, Any]:
    """One chat call answered as JSON; {} when the call or the parsing fails."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        start, end = content.find("{"), content.rfind("}")
        parsed = json.loads(content[start:end + 1]) if start >= 0 else {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        logger.warning("[metadata_clustering] LLM call failed: %s: %s", type(exc).__name__, str(exc)[:200])
        return {}


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
                t = _clean_theme(raw)
                if t and t not in themes:
                    themes.append(t)
            # The explanation: the structured form (shared / differs / basis), or
            # the single sentence of the earlier prompt when a model still answers that way.
            shared = _squash(m.get("shared"), 900)
            differs = _squash(m.get("differs"), 500)
            basis = "field" if str(m.get("basis") or "").lower().startswith("field") else "content"
            legacy = _squash(m.get("reason"), 400)
            reason = shared or legacy
            if differs and differs.lower().strip(" .") not in ("nothing material", "nothing", "none", "n/a", "no difference", "no material difference"):
                reason = f"{reason} Differs: {differs}".strip()
            verdicts[candidates[idx]["id"]] = {
                "strength": "strong" if strength.startswith("strong") else "partial",
                "reason": reason[:1400],
                "reason_parts": {"shared": shared or legacy, "differs": differs, "basis": basis},
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


def _norm_label(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split()).strip(" .")[:60]


def build_vocabulary(client: Any, model: str, raw_counts: dict[str, int],
                     progress: Any = None) -> dict[str, Any]:
    """The two-level vocabulary {parents, themes: [{id, label, parent, raw}], raw_to_theme},
    built in batches: the cleaned terms, most frequent first, VOCAB_BATCH at a
    time, each batch merged against the canonical themes and parents built so far.
    Terms the model leaves out of a batch are retried once with the next, then
    become their own theme under "unsorted themes". Without a model every cleaned
    term is its own theme under one parent, so the views still work."""
    cleaned: dict[str, list[str]] = {}
    for raw, n in raw_counts.items():
        c = _clean_theme(raw)
        if c:
            cleaned.setdefault(c, []).append(raw)
    term_count = {c: sum(raw_counts[r] for r in rs) for c, rs in cleaned.items()}
    terms = sorted(cleaned, key=lambda c: (-term_count[c], c))

    parents: list[dict[str, Any]] = []
    themes: list[dict[str, Any]] = []
    parent_by_label: dict[str, str] = {}
    theme_by_label: dict[str, str] = {}
    theme_by_id: dict[str, dict[str, Any]] = {}

    def parent_id(label: str) -> str:
        label = _norm_label(label) or "other"
        if label not in parent_by_label:
            pid = "p-" + _slug(label)
            if any(p["id"] == pid for p in parents):
                pid = f"{pid}-{len(parents) + 1}"
            parents.append({"id": pid, "label": label})
            parent_by_label[label] = pid
        return parent_by_label[label]

    def theme_id(label: str, pid: str) -> str:
        label = _norm_label(label)
        if label not in theme_by_label:
            tid = "t-" + _slug(label)
            if tid in theme_by_id:
                tid = f"{tid}-{len(themes) + 1}"
            t = {"id": tid, "label": label, "parent": pid, "raw": []}
            themes.append(t)
            theme_by_id[tid] = t
            theme_by_label[label] = tid
        return theme_by_label[label]

    def attach(term: str, tid: str) -> None:
        for r in cleaned[term]:
            if r not in theme_by_id[tid]["raw"]:
                theme_by_id[tid]["raw"].append(r)

    if client is None or not terms:
        pid = parent_id("unsorted themes")
        for term in terms:
            attach(term, theme_id(term, pid))
    else:
        batches = [terms[k:k + VOCAB_BATCH] for k in range(0, len(terms), VOCAB_BATCH)]
        carry: list[str] = []  # terms a batch's answer left out, retried once
        for bi, batch in enumerate(batches):
            pending = carry + [t for t in batch if t not in carry]
            carry = []
            theme_lines = "\n".join(f"- {t['label']}  [{next(p['label'] for p in parents if p['id'] == t['parent'])}]"
                                    for t in themes[:500]) or "(none yet)"
            user = ("PARENTS SO FAR: " + (", ".join(p["label"] for p in parents) or "(none yet)")
                    + f"\n\nCANONICAL THEMES SO FAR ({len(themes)}):\n{theme_lines}"
                    + f"\n\nRAW TERMS ({len(pending)}):\n" + "\n".join(f"- {t} ({term_count[t]}x)" for t in pending))
            parsed = _llm_json(client, model, _VOCAB_PROMPT, user)
            answered: set[str] = set()
            pending_set = set(pending)
            for row in parsed.get("map") or []:
                if not isinstance(row, dict):
                    continue
                term = _norm_label(row.get("term"))
                if term not in pending_set or term in answered:
                    continue
                label = _norm_label(row.get("theme")) or term
                pid = parent_id(row.get("parent") or "other")
                attach(term, theme_id(label, pid))
                answered.add(term)
            missing = [t for t in pending if t not in answered]
            if bi + 1 < len(batches):
                carry = missing[:VOCAB_BATCH // 2]
                missing = missing[len(carry):]
            if missing:
                pid = parent_id("unsorted themes")
                for term in missing:
                    attach(term, theme_id(term, pid))
            if progress:
                progress(bi + 1, len(batches))
        if len(parents) > MAX_PARENTS:
            _fold_parents(client, model, parents, themes)

    raw_to_theme = {r: t["id"] for t in themes for r in t["raw"]}
    return {"parents": parents, "themes": themes, "raw_to_theme": raw_to_theme}


def _fold_parents(client: Any, model: str, parents: list[dict[str, Any]], themes: list[dict[str, Any]]) -> None:
    """Regroup the parents into at most MAX_PARENTS (one small call); on an
    unusable answer the parents stay as they are."""
    counts = {p["id"]: 0 for p in parents}
    for t in themes:
        counts[t["parent"]] = counts.get(t["parent"], 0) + 1
    user = "PARENT LABELS:\n" + "\n".join(f"- {p['label']} ({counts.get(p['id'], 0)} themes)" for p in parents)
    parsed = _llm_json(client, model, _PARENTS_PROMPT, user)
    by_label = {p["label"]: p["id"] for p in parents}
    new_parents: list[dict[str, Any]] = []
    remap: dict[str, str] = {}
    for g in parsed.get("groups") or []:
        if not isinstance(g, dict):
            continue
        label = _norm_label(g.get("label"))
        members = [_norm_label(m) for m in (g.get("members") or [])]
        members = [m for m in members if m in by_label and by_label[m] not in remap]
        if not label or not members:
            continue
        pid = "p-" + _slug(label)
        if any(p["id"] == pid for p in new_parents):
            pid = f"{pid}-{len(new_parents) + 1}"
        new_parents.append({"id": pid, "label": label})
        for m in members:
            remap[by_label[m]] = pid
    if not new_parents or len(new_parents) > MAX_PARENTS + 2:
        return
    leftovers = [p for p in parents if p["id"] not in remap]
    if leftovers:
        other = next((p for p in new_parents if p["label"] in ("other", "unsorted themes")), None)
        if other is None:
            other = {"id": "p-unsorted", "label": "unsorted themes"}
            new_parents.append(other)
        for p in leftovers:
            remap[p["id"]] = other["id"]
    for t in themes:
        t["parent"] = remap.get(t["parent"], t["parent"])
    parents[:] = new_parents


def assign_themes(client: Any, model: str, elements: list[dict[str, Any]], pair_records: list[dict[str, Any]],
                  vocab: dict[str, Any], progress: Any = None) -> None:
    """Give every AI match its canonical themes (rec["theme_ids"]), chosen by the
    model from the closed vocabulary given the match's "shared" explanation, with
    the themes its raw tags map to as suggestions. ASSIGN_BATCH matches per call,
    grouped by the first element of the pair. Without a model, or when a call
    fails, the suggestions stand."""
    theme_list = vocab["themes"]
    num_to_id = {k + 1: t["id"] for k, t in enumerate(theme_list)}
    id_to_num = {v: k for k, v in num_to_id.items()}
    parent_label = {p["id"]: p["label"] for p in vocab["parents"]}
    by_parent: dict[str, list[str]] = {}
    for k, t in enumerate(theme_list, 1):
        by_parent.setdefault(t["parent"], []).append(f"  T{k} {t['label']}")
    vocab_text = "\n".join(f"{parent_label.get(pid, pid)}:\n" + "\n".join(lines) for pid, lines in by_parent.items())

    ai_records = [r for r in pair_records if r.get("ai_strength")]
    for rec in ai_records:
        rec["theme_ids"] = _suggested_ids(rec, vocab)
    if client is None or not theme_list or not ai_records:
        if progress:
            progress(1, 1)
        return

    groups: dict[int, list[dict[str, Any]]] = {}
    for rec in ai_records:
        groups.setdefault(rec["i"], []).append(rec)
    batches: list[list[dict[str, Any]]] = []
    for recs in groups.values():
        for k in range(0, len(recs), ASSIGN_BATCH):
            batches.append(recs[k:k + ASSIGN_BATCH])

    def run_batch(recs: list[dict[str, Any]]) -> None:
        lines = []
        for n, rec in enumerate(recs, 1):
            a, b = elements[rec["i"]], elements[rec["j"]]
            parts = rec.get("ai_reason_parts") or {}
            shared = parts.get("shared") or rec.get("ai_reason") or ""
            sugg = ", ".join(f"T{id_to_num[t]}" for t in rec["theme_ids"] if t in id_to_num)
            lines.append(f"{n}. [{a['cohort']} — {a['label']}] vs [{b['cohort']} — {b['label']}] ({rec.get('ai_strength')})\n"
                         f"   shared: {shared[:700]}" + (f"\n   suggested: {sugg}" if sugg else ""))
        user = f"VOCABULARY:\n{vocab_text}\n\nMATCHES ({len(recs)}):\n" + "\n".join(lines)
        parsed = _llm_json(client, model, _ASSIGN_PROMPT, user)
        for row in parsed.get("assign") or []:
            if not isinstance(row, dict):
                continue
            try:
                n = int(row.get("match"))
            except (TypeError, ValueError):
                continue
            if not 1 <= n <= len(recs):
                continue
            chosen: list[str] = []
            for v in (row.get("themes") or [])[:3]:
                try:
                    k = int(str(v).strip().lstrip("Tt"))
                except ValueError:
                    continue
                tid = num_to_id.get(k)
                if tid and tid not in chosen:
                    chosen.append(tid)
            recs[n - 1]["theme_ids"] = chosen

    done = 0
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        for _ in as_completed([pool.submit(run_batch, b) for b in batches]):
            done += 1
            if progress and (done % 5 == 0 or done == len(batches)):
                progress(done, len(batches))


def _suggested_ids(rec: dict[str, Any], vocab: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for raw in rec.get("themes") or []:
        tid = vocab["raw_to_theme"].get(raw)
        if tid and tid not in out:
            out.append(tid)
    return out[:3]


_STRENGTH_W = {"strong": 1.0, "partial": 0.6}


def build_study_themes(elements: list[dict[str, Any]], pair_records: list[dict[str, Any]],
                       vocab: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """study -> theme id -> {strength, count, excerpts}. A study carries a theme
    when any of its elements matched another study's element on it; strength is
    the best match that carried it, count how many did."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in pair_records:
        # Canonical themes from the assignment pass; older records fall back to
        # mapping their raw tags through the vocabulary.
        tids = rec["theme_ids"] if "theme_ids" in rec else _suggested_ids(rec, vocab)
        if not tids:
            continue
        w = _STRENGTH_W.get(rec.get("ai_strength") or "partial", 0.6)
        a, b = elements[rec["i"]], elements[rec["j"]]
        parts = rec.get("ai_reason_parts") or {}
        reason = (parts.get("shared") or rec.get("ai_reason") or "")[:500]
        for tid in tids:
            for me, other in ((a, b), (b, a)):
                cell = table.setdefault(me["cohort"], {}).setdefault(tid, {"strength": 0.0, "count": 0, "excerpts": []})
                cell["strength"] = max(cell["strength"], w)
                cell["count"] += 1
                if len(cell["excerpts"]) < 6 and not any(x["text"] == me["text"] and x["with"] == other["cohort"] for x in cell["excerpts"]):
                    cell["excerpts"].append({"field": me["label"], "text": me["text"][:400], "with": other["cohort"],
                                             "with_text": other["text"][:400], "strength": rec.get("ai_strength"),
                                             "reason": reason})
    return table


def _theme_stage(client: Any, model: str, elements: list[dict[str, Any]],
                 pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Cleanup, vocabulary, assignment, table: the "themes" block of the results.
    Sets rec["theme_ids"] on every AI pair record. Progress goes to the job state."""
    raw_counts: dict[str, int] = {}
    for rec in pair_records:
        for t in rec.get("themes") or []:
            raw_counts[t] = raw_counts.get(t, 0) + 1
    _update_state(phase="themes", vocab_done=0, vocab_total=max(1, math.ceil(len(raw_counts) / VOCAB_BATCH)))
    vocab = build_vocabulary(client, model, raw_counts,
                             progress=lambda d, t: _update_state(vocab_done=d, vocab_total=t))
    _update_state(phase="assign", assign_done=0, assign_total=None)
    assign_themes(client, model, elements, pair_records, vocab,
                  progress=lambda d, t: _update_state(assign_done=d, assign_total=t))
    table = build_study_themes(elements, pair_records, vocab)
    logger.info("[metadata_clustering] themes: %d raw -> %d canonical under %d parents",
                len(raw_counts), len(vocab["themes"]), len(vocab["parents"]))
    return {"parents": vocab["parents"], "themes": vocab["themes"], "raw_to_theme": vocab["raw_to_theme"],
            "raw_counts": raw_counts, "table": table, "built": datetime.now().isoformat(),
            "pipeline": "cleanup + batched vocabulary + closed-set assignment"}


# Themes an admin took out of the views (not out of the stored results).
_EXCLUDED_PATH = os.path.join(_DIR, "excluded_themes.json")
# Admin edits of the vocabulary (merge, move, nest, rename, delete a parent):
# applied to the stored results at once and kept as label-based rules, so a
# rebuild or a new run re-applies them (best effort: a rule whose labels no
# longer exist is skipped). Each edit snapshots the results first, for undo.
_EDITS_PATH = os.path.join(_DIR, "theme_edits.json")
_KEEP_EDIT_SNAPSHOTS = 10


def excluded_themes() -> set[str]:
    data = _read_json(_EXCLUDED_PATH)
    return set(data.get("excluded") or [])


# ---------------------------------------------------------------------------
# Admin edits of the vocabulary (two levels: parents and themes)
# ---------------------------------------------------------------------------

def _pair_records_from_matches(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The unordered pair records behind the symmetric match lists."""
    elements = data["elements"]
    index = {e["id"]: k for k, e in enumerate(elements)}
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for eid, lst in data["matches"].items():
        for m in lst:
            i, j = index[eid], index[m["element_id"]]
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            rec = {k: m.get(k) for k in ("source", "score", "fuzzy", "tfidf", "same_field", "ai_strength",
                                          "ai_reason", "ai_reason_parts", "ai_votes")}
            rec["themes"] = list(m.get("themes") or [])
            if "theme_ids" in m:
                rec["theme_ids"] = list(m.get("theme_ids") or [])
            rec["i"], rec["j"] = key
            out.append(rec)
    return out


def _vocab_of(data: dict[str, Any]) -> dict[str, Any]:
    block = data.setdefault("themes", {})
    block.setdefault("parents", [])
    block.setdefault("themes", [])
    block.setdefault("raw_to_theme", {})
    return block


def _unsorted_parent(vocab: dict[str, Any]) -> str:
    for p in vocab["parents"]:
        if p["id"] == "p-unsorted":
            return p["id"]
    vocab["parents"].append({"id": "p-unsorted", "label": "unsorted themes"})
    return "p-unsorted"


def _parent_for_label(vocab: dict[str, Any], label: str) -> str:
    """The parent with this label, created when missing."""
    label = _norm_label(label)
    for p in vocab["parents"]:
        if p["label"] == label:
            return p["id"]
    pid = "p-" + _slug(label)
    if any(p["id"] == pid for p in vocab["parents"]):
        pid = f"{pid}-{len(vocab['parents']) + 1}"
    vocab["parents"].append({"id": pid, "label": label})
    return pid


def _theme_by(vocab: dict[str, Any], theme_id: Optional[str] = None, label: Optional[str] = None) -> Optional[dict[str, Any]]:
    for t in vocab["themes"]:
        if (theme_id and t["id"] == theme_id) or (label and t["label"] == _norm_label(label)):
            return t
    return None


def _parent_by(vocab: dict[str, Any], parent_id: Optional[str] = None, label: Optional[str] = None) -> Optional[dict[str, Any]]:
    for p in vocab["parents"]:
        if (parent_id and p["id"] == parent_id) or (label and p["label"] == _norm_label(label)):
            return p
    return None


def _replace_theme_id(data: dict[str, Any], old: str, new: Optional[str]) -> None:
    """Point every match's theme_ids from old to new (dropped when new is None)."""
    for lst in data["matches"].values():
        for m in lst:
            ids = m.get("theme_ids")
            if not ids or old not in ids:
                continue
            out: list[str] = []
            for t in ids:
                t2 = new if t == old else t
                if t2 and t2 not in out:
                    out.append(t2)
            m["theme_ids"] = out


def apply_vocab_edit(data: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
    """Apply one edit to the results in place and return the label-based rule
    that re-applies it later. Raises ValueError on a bad request. Ops:
      merge         {source, target}        theme source folds into theme target
      move          {theme, parent | parent_label}   theme goes under a parent (created from the label when new)
      nest          {theme, under}          theme goes under a parent named after theme `under`,
                                            which moves there too (a theme cannot hold themes: two levels)
      rename        {theme | parent, label}
      delete_parent {parent}                its themes go to "unsorted themes"
    Ids in the op; the rule records labels."""
    vocab = _vocab_of(data)
    kind = str(op.get("op") or "")
    rule: dict[str, Any] = {"op": kind}
    if kind == "merge":
        src, dst = _theme_by(vocab, op.get("source")), _theme_by(vocab, op.get("target"))
        if not src or not dst or src is dst:
            raise ValueError("merge needs two different existing themes")
        for r in src["raw"]:
            if r not in dst["raw"]:
                dst["raw"].append(r)
            vocab["raw_to_theme"][r] = dst["id"]
        vocab["themes"] = [t for t in vocab["themes"] if t is not src]
        _replace_theme_id(data, src["id"], dst["id"])
        rule.update(source=src["label"], target=dst["label"])
    elif kind == "move":
        t = _theme_by(vocab, op.get("theme"))
        if not t:
            raise ValueError("unknown theme")
        if op.get("parent_label"):
            pid = _parent_for_label(vocab, str(op["parent_label"]))
        else:
            p = _parent_by(vocab, op.get("parent"))
            if not p:
                raise ValueError("unknown parent")
            pid = p["id"]
        t["parent"] = pid
        rule.update(theme=t["label"], parent_label=_parent_by(vocab, pid)["label"])
    elif kind == "nest":
        t, under = _theme_by(vocab, op.get("theme")), _theme_by(vocab, op.get("under"))
        if not t or not under or t is under:
            raise ValueError("nest needs two different existing themes")
        pid = _parent_for_label(vocab, under["label"])
        under["parent"] = pid
        t["parent"] = pid
        rule.update(theme=t["label"], under=under["label"])
    elif kind == "rename":
        label = _norm_label(op.get("label"))
        if not label:
            raise ValueError("a label is required")
        if op.get("theme"):
            t = _theme_by(vocab, op.get("theme"))
            if not t:
                raise ValueError("unknown theme")
            if _theme_by(vocab, label=label):
                raise ValueError(f"a theme named '{label}' exists already: merge instead")
            rule.update(theme=t["label"], label=label)
            t["label"] = label
        else:
            p = _parent_by(vocab, op.get("parent"))
            if not p:
                raise ValueError("unknown parent")
            rule.update(parent=p["label"], label=label)
            p["label"] = label
    elif kind == "delete_parent":
        p = _parent_by(vocab, op.get("parent"))
        if not p:
            raise ValueError("unknown parent")
        if p["id"] == "p-unsorted":
            raise ValueError("the unsorted group cannot be deleted")
        rule.update(parent=p["label"])
        vocab["parents"] = [q for q in vocab["parents"] if q is not p]
        unsorted = _unsorted_parent(vocab)
        for t in vocab["themes"]:
            if t["parent"] == p["id"]:
                t["parent"] = unsorted
    else:
        raise ValueError(f"unknown op '{kind}'")
    # Parents left without themes disappear (the unsorted group stays only while used).
    used = {t["parent"] for t in vocab["themes"]}
    vocab["parents"] = [p for p in vocab["parents"] if p["id"] in used]
    return rule


def _rule_to_op(vocab: dict[str, Any], rule: dict[str, Any]) -> Optional[dict[str, Any]]:
    """A stored label rule as an id-based op against the current vocabulary, or
    None when its labels are gone."""
    kind = rule.get("op")
    if kind == "merge":
        s_, d = _theme_by(vocab, label=rule.get("source")), _theme_by(vocab, label=rule.get("target"))
        return {"op": kind, "source": s_["id"], "target": d["id"]} if s_ and d and s_ is not d else None
    if kind == "move":
        t = _theme_by(vocab, label=rule.get("theme"))
        return {"op": kind, "theme": t["id"], "parent_label": rule.get("parent_label")} if t and rule.get("parent_label") else None
    if kind == "nest":
        t, u = _theme_by(vocab, label=rule.get("theme")), _theme_by(vocab, label=rule.get("under"))
        return {"op": kind, "theme": t["id"], "under": u["id"]} if t and u and t is not u else None
    if kind == "rename":
        if rule.get("theme"):
            t = _theme_by(vocab, label=rule.get("theme"))
            return {"op": kind, "theme": t["id"], "label": rule.get("label")} if t and not _theme_by(vocab, label=rule.get("label")) else None
        p = _parent_by(vocab, label=rule.get("parent"))
        return {"op": kind, "parent": p["id"], "label": rule.get("label")} if p else None
    if kind == "delete_parent":
        p = _parent_by(vocab, label=rule.get("parent"))
        return {"op": kind, "parent": p["id"]} if p else None
    return None


def _rebuild_table(data: dict[str, Any]) -> None:
    data["themes"]["table"] = build_study_themes(data["elements"], _pair_records_from_matches(data), data["themes"])


def apply_theme_edits(data: dict[str, Any]) -> int:
    """Re-apply the stored admin rules to freshly built results; the number applied."""
    rules = (_read_json(_EDITS_PATH).get("rules") or [])
    if not rules or not data.get("themes"):
        return 0
    applied = 0
    for rule in rules:
        op = _rule_to_op(data["themes"], rule)
        if not op:
            continue
        try:
            apply_vocab_edit(data, op)
            applied += 1
        except ValueError as exc:
            logger.info("[metadata_clustering] stored vocabulary rule skipped (%s): %s", exc, rule)
    if applied:
        _rebuild_table(data)
    return applied


def _snapshot_before_edit() -> Optional[str]:
    """Copy the results aside before an edit; returns the snapshot path."""
    if not os.path.exists(_RESULT_PATH):
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = os.path.join(_DIR, f"results_before_edit_{stamp}.json")
    import shutil
    shutil.copyfile(_RESULT_PATH, path)
    olds = sorted(f for f in os.listdir(_DIR) if f.startswith("results_before_edit_"))
    for f in olds[:-_KEEP_EDIT_SNAPSHOTS]:
        try:
            os.remove(os.path.join(_DIR, f))
        except OSError:
            pass
    return path


def vocabulary_view(data: dict[str, Any]) -> dict[str, Any]:
    """Parents with their themes, each theme with its carriers and raw tags."""
    vocab = _vocab_of(data)
    table = vocab.get("table") or {}
    excluded = excluded_themes()
    carriers: dict[str, int] = {}
    for study, row in table.items():
        for tid in row:
            carriers[tid] = carriers.get(tid, 0) + 1
    themes = [{"id": t["id"], "label": t["label"], "parent": t["parent"], "raw": t.get("raw") or [],
               "studies": carriers.get(t["id"], 0), "excluded": t["id"] in excluded} for t in vocab["themes"]]
    edits = _read_json(_EDITS_PATH)
    return {"parents": vocab["parents"], "themes": themes, "n_studies": len({e["cohort"] for e in data.get("elements") or []}),
            "rules": edits.get("rules") or [], "undo_available": bool(edits.get("last_snapshot")) and os.path.exists(str(edits.get("last_snapshot") or "")),
            "built": vocab.get("built"), "rebuilt": data.get("themes_rebuilt")}


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
                        cur = ai.setdefault(key, {"strength": v["strength"], "reason": v["reason"],
                                                  "reason_parts": v.get("reason_parts"), "votes": 0, "themes": []})
                        cur["votes"] += 1
                        for t in v.get("themes") or []:
                            if t not in cur["themes"]:
                                cur["themes"].append(t)
                        # Two directions may disagree; keep the stronger verdict and
                        # the explanation given for it.
                        if v["strength"] == "strong" and cur["strength"] != "strong":
                            cur["strength"], cur["reason"], cur["reason_parts"] = "strong", v["reason"], v.get("reason_parts")
                    # Every element: the state's updated_at is the heartbeat for stall detection.
                    _update_state(ai_done=done, ai_failed=len(ai_failed))
                    if done % 10 == 0 or done == len(elements):
                        logger.info("[metadata_clustering] LLM %d/%d elements judged", done, len(elements))
        else:
            logger.warning("[metadata_clustering] LITELLM_BASE_URL not set; lexical pass only")

        # One record per unordered pair.
        counts = {"lexical": 0, "ai": 0, "both": 0}
        pair_records: list[dict[str, Any]] = []
        for key in sorted(set(lexical_hits) | set(ai)):
            i, j = key
            p = pairs.get(key) or {"score": 0.0, "fuzzy": 0.0, "tfidf": 0.0, "same_field": elements[i]["field"] == elements[j]["field"]}
            in_lex, in_ai = key in lexical_hits, key in ai
            source = "both" if (in_lex and in_ai) else ("lexical" if in_lex else "ai")
            counts[source] += 1
            pair_records.append({"i": i, "j": j, "source": source, "score": p["score"], "fuzzy": p["fuzzy"],
                                 "tfidf": p["tfidf"], "same_field": p["same_field"],
                                 "ai_strength": ai[key]["strength"] if in_ai else None,
                                 "ai_reason": ai[key]["reason"] if in_ai else None,
                                 "ai_reason_parts": ai[key].get("reason_parts") if in_ai else None,
                                 "ai_votes": ai[key]["votes"] if in_ai else 0,
                                 "themes": ai[key]["themes"] if in_ai else []})

        # Cleanup, vocabulary, assignment (sets theme_ids on the records), table.
        themes_block = _theme_stage(client if ai_available else None, settings.litellm_model if ai_available else "",
                                    elements, pair_records)

        # Listed symmetrically: every pair on both of its elements.
        matches: dict[str, list[dict[str, Any]]] = {e["id"]: [] for e in elements}
        for rec in pair_records:
            i, j = rec["i"], rec["j"]
            body = {k: v for k, v in rec.items() if k not in ("i", "j")}
            body.setdefault("theme_ids", [])
            for a, b in ((i, j), (j, i)):
                matches[elements[a]["id"]].append({**body, "cohort": elements[b]["cohort"], "field": elements[b]["field"],
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
            "themes": themes_block,
        }
        applied = apply_theme_edits(result)
        if applied:
            logger.info("[metadata_clustering] %d stored vocabulary edits re-applied", applied)
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


def rebuild_themes() -> None:
    """The theme steps alone, on the stored results: the AI verdicts and their
    explanations are kept, the raw tags are cleaned, the vocabulary rebuilt in
    batches and every match re-assigned from it. Minutes, not the hour of a run."""
    started = time.time()
    try:
        data = _read_json(_RESULT_PATH)
        elements = data["elements"]
        index = {e["id"]: k for k, e in enumerate(elements)}
        seen: set[tuple[int, int]] = set()
        pair_records: list[dict[str, Any]] = []
        for eid, lst in data["matches"].items():
            for m in lst:
                i, j = index[eid], index[m["element_id"]]
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                rec = {k: m.get(k) for k in ("source", "score", "fuzzy", "tfidf", "same_field", "ai_strength",
                                              "ai_reason", "ai_reason_parts", "ai_votes")}
                rec["themes"] = list(m.get("themes") or [])
                rec["i"], rec["j"] = key
                pair_records.append(rec)
        _update_state(elements=len(elements), cohorts=len({e["cohort"] for e in elements}),
                      lexical_matches=None, ai_done=None, ai_failed=None)
        ai_available = bool(settings.litellm_base_url)
        client = None
        if ai_available:
            from src.chat import _get_openai_client
            client = _get_openai_client()
        themes_block = _theme_stage(client, settings.litellm_model if ai_available else "", elements, pair_records)
        by_key = {(r["i"], r["j"]): r for r in pair_records}
        for eid, lst in data["matches"].items():
            for m in lst:
                i, j = index[eid], index[m["element_id"]]
                m["theme_ids"] = by_key[(min(i, j), max(i, j))].get("theme_ids", [])
        data["themes"] = themes_block
        data["themes_rebuilt"] = datetime.now().isoformat()
        applied = apply_theme_edits(data)
        if applied:
            logger.info("[metadata_clustering] %d stored vocabulary edits re-applied", applied)
        _write_json(_RESULT_PATH, data)
        state = _update_state(running=False, phase="done", finished=datetime.now().isoformat(), error=None,
                              counts=data.get("counts"), duration_seconds=round(time.time() - started))
        _append_history(state)
        logger.info("[metadata_clustering] themes rebuilt in %ds: %d themes under %d parents",
                    round(time.time() - started), len(themes_block["themes"]), len(themes_block["parents"]))
    except Exception as exc:
        logger.exception("[metadata_clustering] theme rebuild failed")
        state = _update_state(running=False, phase="failed", finished=datetime.now().isoformat(),
                              error=f"{type(exc).__name__}: {exc}")
        _append_history(state)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/metadata-clustering/rebuild-themes")
async def start_rebuild_themes(background: BackgroundTasks, user: Any = Depends(get_current_user)):
    """Rebuild the theme vocabulary and assignments from the stored results, without
    re-judging the pairs."""
    _require_admin(user)
    if check_stalled().get("running"):
        raise HTTPException(status_code=409, detail="A metadata clustering job is already in progress.")
    if not os.path.exists(_RESULT_PATH):
        raise HTTPException(status_code=404, detail="No metadata clustering results yet.")
    _update_state(running=True, phase="starting", started=datetime.now().isoformat(), finished=None,
                  error=None, started_by=user["email"], instructions="(themes rebuilt from the stored results)",
                  kind="rebuild-themes", ai_available=bool(settings.litellm_base_url), model=settings.litellm_model)
    background.add_task(rebuild_themes)
    return {"started": True, "ai_available": bool(settings.litellm_base_url)}


@router.post("/metadata-clustering/run")
async def start_run(background: BackgroundTasks, body: Optional[dict[str, Any]] = None,
                    user: Any = Depends(get_current_user)):
    _require_admin(user)
    if check_stalled().get("running"):
        raise HTTPException(status_code=409, detail="A metadata clustering run is already in progress.")
    instructions = str((body or {}).get("instructions") or "").strip()[:4000]
    _update_state(running=True, phase="starting", started=datetime.now().isoformat(), finished=None,
                  error=None, started_by=user["email"], instructions=instructions, kind="run",
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


@router.get("/metadata-clustering/vocabulary")
async def vocabulary(user: Any = Depends(get_current_user)):
    _require_admin(user)
    data = _read_json(_RESULT_PATH)
    if not data or not data.get("themes"):
        raise HTTPException(status_code=404, detail="No theme vocabulary yet: run the matching or rebuild the themes.")
    return vocabulary_view(data)


@router.post("/metadata-clustering/vocabulary/edit")
async def edit_vocabulary(body: dict[str, Any], user: Any = Depends(get_current_user)):
    """One admin edit of the vocabulary: merge, move, nest, rename or delete_parent.
    Applied to the stored results at once (snapshot taken first) and recorded as
    a label rule that later runs re-apply."""
    _require_admin(user)
    if check_stalled().get("running"):
        raise HTTPException(status_code=409, detail="A metadata clustering job is in progress; edit the vocabulary when it is done.")
    with _write_lock:
        data = _read_json(_RESULT_PATH)
        if not data or not data.get("themes"):
            raise HTTPException(status_code=404, detail="No theme vocabulary yet.")
        snapshot = _snapshot_before_edit()
        try:
            rule = apply_vocab_edit(data, body or {})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _rebuild_table(data)
        _write_json(_RESULT_PATH, data)
        edits = _read_json(_EDITS_PATH)
        rules = edits.get("rules") or []
        rules.append({**rule, "at": datetime.now().isoformat(), "by": user["email"]})
        _write_json(_EDITS_PATH, {"rules": rules, "last_snapshot": snapshot, "updated_at": datetime.now().isoformat()})
    logging.info("[metadata_clustering] vocabulary edit by %s: %s", user["email"], rule)
    return vocabulary_view(data)


@router.post("/metadata-clustering/vocabulary/undo")
async def undo_vocabulary_edit(user: Any = Depends(get_current_user)):
    """Restore the results from before the last edit and drop its rule."""
    _require_admin(user)
    if check_stalled().get("running"):
        raise HTTPException(status_code=409, detail="A metadata clustering job is in progress.")
    with _write_lock:
        edits = _read_json(_EDITS_PATH)
        snapshot = str(edits.get("last_snapshot") or "")
        if not snapshot or not os.path.exists(snapshot):
            raise HTTPException(status_code=404, detail="Nothing to undo: only the last edit can be undone, once.")
        os.replace(snapshot, _RESULT_PATH)
        rules = (edits.get("rules") or [])[:-1]
        _write_json(_EDITS_PATH, {"rules": rules, "last_snapshot": None, "updated_at": datetime.now().isoformat()})
        data = _read_json(_RESULT_PATH)
    return vocabulary_view(data)


@router.get("/metadata-clustering/results")
async def results(user: Any = Depends(get_current_user)):
    _require_admin(user)
    data = _read_json(_RESULT_PATH)
    if not data:
        raise HTTPException(status_code=404, detail="No metadata clustering results yet.")
    return data
