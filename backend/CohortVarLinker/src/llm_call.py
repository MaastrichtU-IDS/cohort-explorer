from __future__ import annotations
import json, re, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_model import ContextMatchType, MappingType

import hashlib
from pathlib import Path

_LABELS = {
    "COMPLETE",
    "COMPATIBLE",
    "PARTIAL",
    "IMPOSSIBLE",
}

_LABEL_ALIASES = {
    "complete": "COMPLETE",
    "compatible": "COMPATIBLE",
    "partial": "PARTIAL",
    "impossible": "IMPOSSIBLE",
}

_CODE_TO_LABEL = {"1": "COMPLETE", "2": "COMPATIBLE", "3": "PARTIAL", "4": "IMPOSSIBLE"}
class LLMDiskCache:
    def __init__(self, cache_dir: str = settings.LLM_CACHE_DIR):
        self.root = Path(cache_dir)
        self._sys_hash = ""

    def set_system_prompt(self, sys_prompt: str):
        """Call once when system prompt changes to invalidate stale entries."""
        self._sys_hash = hashlib.sha256(sys_prompt.encode()).hexdigest()[:12]

   
    def _path(self, model: str, prompt: str, mode: str = "") -> Path:
        name = model.split("/")[-1]
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        key_material = f"{self._sys_hash}::{mode}::{model}::{prompt}"
        return d / f"{hashlib.sha256(key_material.encode()).hexdigest()}.json"

    def get(self, model: str, prompt: str, mode: str = "") -> str | None:
        p = self._path(model, prompt, mode)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["r"]
        except Exception:
            p.unlink(missing_ok=True)
            return None

    def get_record(self, model: str, prompt: str, mode: str = "") -> dict | None:
        p = self._path(model, prompt, mode)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())

            # Backward compatibility with old cache format:
            # {"prompt": ..., "r": "..."}
            if "r" in data:
                return data

            return None
        except Exception:
            p.unlink(missing_ok=True)
            return None
    def put_record(
        self,
        model: str,
        prompt: str,
        response: str,
        mode: str = "",
        logprob_dist: dict | None = None,
        raw_logprobs: dict | None = None,
    ):
        self._path(model, prompt, mode).write_text(
            json.dumps({
                "prompt": prompt,
                "r": response,
                "logprob_dist": logprob_dist,
                "raw_logprobs": raw_logprobs,
            }, ensure_ascii=False)
        )

    def put(self, model: str, 
          prompt: str, 
          response: str, 
          mode: str = ""):
        self.put_record(model, prompt, response, mode=mode)

    def delete(self, model: str, prompt: str, mode: str = ""):
        self._path(model, prompt, mode).unlink(missing_ok=True)

_INPUT_NE = """
# INPUT
Two variables are provided: Source and Target. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
 
Each variable has:
- description: variable label from study metadata
- unit: measurement unit, if available
- categories: allowed values; [] means continuous or free-text
"""

_BATCH_INPUT_NE = """
# INPUT
One Source variable and multiple target variables are provided. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
 
Each variable has:
- description: variable label from study metadata
- unit: measurement unit, if available
- categories: allowed values; [] means continuous or free-text
"""
 
_INPUT_EV = """
# INPUT
Two variables are provided: Source and Target. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
 
Each variable may include:
- Description: short variable label
- Concepts: ordered concepts separated by " | "
  - first concept = primary concept (authoritative for meaning)
  - remaining concepts = refinements that may narrow or qualify meaning
- Categories: allowed values, if available
- Unit: measurement unit, if available
- Graph evidence: optional semantic evidence from OMOP vocabularies
"""

_BATCH_INPUT_EV = """
# INPUT
One Source variable and multiple target variables are provided. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
 
Each variable may include:
- Description: short variable label
- Concepts: ordered concepts separated by " | "
  - first concept = primary concept (authoritative for meaning)
  - remaining concepts = refinements that may narrow or qualify meaning
- Categories: allowed values, if available
- Unit: measurement unit, if available
- Graph evidence: optional semantic evidence from OMOP vocabularies
"""

_RULES_EV_ONLY = """- The first concept is authoritative for variable meaning. Treat additional concepts as refinements only if they materially change clinical interpretation; otherwise treat them as annotation context. 
- If Description and Concepts conflict, use Concepts as the primary meaning. 
- When graph evidence conflicts with descriptions or concepts, prefer the explicit metadata. Use graph evidence to break ties or confirm relationships, not to override stated meaning."""
 
_OUTPUT_PAIR = """
# OUTPUT CONSTRAINTS
- status: COMPLETE | COMPATIBLE | PARTIAL | IMPOSSIBLE
- confidence: float 0.0-1.0
- reason: 50 words or fewer; prioritize clarity and explanation over word count. explain which rules you applied and why.
- transform: 40 words or fewer, or ""
- harmonized_variable: 12 words or fewer, snake_case, or ""
- alignment_direction: "bidirectional" | "source to target" | "target to source" | "both for derivation" | ""

# OUTPUT FORMAT
Return ONLY one valid JSON object:
{{
  "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
  "confidence": <float>,
  "reason": "<50 words or fewer>",
  "transform": "<40 words or fewer, or empty>",
  "harmonized_variable": "<snake_case or empty>",
  "alignment_direction": "<direction or empty>"
}}
"""

_OUTPUT_BATCH = """
# OUTPUT CONSTRAINTS
- Each verdict has the same fields and limits as the single-pair schema.
- Evaluate each (Source, Target i) pair independently. Targets must not influence one another.

# OUTPUT FORMAT
Return ONLY one valid JSON array, one object per target, in the same order as Target 1 .. Target N:
[
  {{
    "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
    "confidence": <float>,
    "reason": "<25 words or fewer>",
    "transform": "<40 words or fewer, or empty>",
    "harmonized_variable": "<snake_case or empty>",
    "alignment_direction": "<direction or empty>"
  }}
]
"""

_SHARED_BODY = """
# REASONING PROCEDURE
Before assigning a status, answer these questions internally:
1. What clinical entity does each variable measure? Separate the entity itself from how, where, or when it is observed. 
2. Are these the same entity, or genuinely different entities? Are the entities interchanble in medical and physiological standard?
3. If the same entity: can a single common variable be defined that both sides can populate — even if one side requires reduction, recoding, or approximation?
4. If a common variable exists: can at least one side fully populate it while the other contributes without fabricating values? Mapping a genuinely unknown value to missing/unknown is allowed and expected.

Use the answers to select exactly ONE status below.

# STATUS DEFINITIONS

- COMPLETE:
  Same clinical entity, same granularity, clinically interpretable threshold and same values representation. Values merge as-is with no transformation. Only Mathematically equivalent units with different notation (e.g., ug/L = ng/mL, {{counts}}/min = /min) should be considered of the same representation. 
  Examples:
  - systolic BP (mmHg) vs sitting systolic BP (mmHg)
  - history of myocardial infarction (yes/no) vs myocardial infarction using ecg (yes/no)
  - age (years) vs age at baseline (years)
  - aspartate aminotransferase [enzymatic activity/volume] in serum or plasma ([U]/L) vs aspartate aminotransferase measurement ([U]/L)
  - NT-proBNP (ug/L) vs NT-proBNP (ng/mL)
  - jugular vein elevated (yes/no) vs central venous pressure > 6 cm H₂O (yes/no)

- COMPATIBLE:
  Same clinical entity, same analysis-level granularity clinically interpretable threshold and different values representation. Values can be merged through deterministic, lossless recoding, unit conversion, or rescaling.
  Examples:
  - weight (kg) vs weight (lb)
  - history of myocardial infarction (yes/no) vs myocardial infarction using ecg (1/0)
  - blood glucose (mmol/L) vs blood glucose (mg/dL)
  - Gender (Female/Male) vs Gender (m/f)
  - NYHA class (I/II/III/IV) vs NYHA class (1/2/3/4)
  - jugular vein elevated (1/0) vs central venous pressure > 6 cm H₂O (yes/no)

- PARTIAL:
  A valid common harmonized variable can be created, but at least one side requires transformation that is lossy or requires manual review (e.g., reduction, collapsing, recoding, temporal approximation, or external interpretation).
  Use PARTIAL only when ALL of these hold:
  1. The variables measure the same clinical entity, or one variable is a clinical subclass of the other.
  2. The harmonized_variable must represent ONE clinical concept, not a union such as "A or B".
  3. If one variable can only provide positive evidence (“yes”) while its negative or missing values are genuinely unknown, classify as PARTIAL only when the other side can fully populate the harmonized variable and the positive-only evidence is clinically valid. Map unsupported negative or missing values to unknown, never to “no”.  
  4. Values may be collapsed or mapped to missing/unknown, but never to invented quantitative values, subtype labels, causes, procedures, or patient facts.
  5. If one side records only a parent class and the other records clinical subtype, the harmonized_variable must be the parent class unless the parent side explicitly encodes the subtype/detail.Do not create a subtype/detail harmonized_variable from a parent-class value.
  6. The harmonized variable is directly supported by the provided metadata — not a new concept invented by merging two different entities.

  Sub-patterns:
  (a) Granularity reduction — one side is finer-grained; collapse to the coarser level.
  (b) External-reference alignment — same or class-related entity in different scales/units, convertible via external references (target-dose tables, drug-equivalence tables, conversion factors). Member-drug vs class-level dose across scales (e.g., mg vs % of target) qualifies. Excluded: composites/sums of sibling classes (e.g., "A + B" vs "A") — those are IMPOSSIBLE.

  Examples:
  - smoking status (never/former/current) vs smoker yes/no -> harmonized: smoker_yes_no
  - month of coronary angioplasty vs coronary angioplasty yes/no -> coronary_angioplasty_yes_no
  - edema severity (mild/moderate/severe) vs left leg edema((mild/moderate/severe)) -> leg_edema_severity
  - type of cancer (breast|lung|colon|other) vs breast cancer (yes/no) -> breast_cancer_yes_no
  - metoprolol dose (mg) vs beta-blocker dose (% of target) -> harmonized: beta_blocker_dose_pct_target
  - Torsemide use yes/no vs loop diuretic dose (mg) -> harmonized: loop_diuretic_use_yes_no
  - hydrochlorothiazide use yes/no vs diuretics use (yes/no) -> harmonized: diuretic_use_yes_no

- IMPOSSIBLE:
  No valid common harmonized variable can be created without fabricating information.

  Use IMPOSSIBLE when ANY of these hold:
  1. The variables measure genuinely different clinical entities, even if superficially related.
  2. The variables are sibling concepts (e.g., two different drug classes, two different conditions) rather than a finer/coarser version of the same variable.
  3. Creating a common variable would require fabricating values (numeric or inferential).
  4. A composite variable on one side cannot be decomposed into the specific component on the other side.

  Note: different units or scales alone do NOT make a pair IMPOSSIBLE. If the underlying clinical entity is the same (or related by class membership) and a conversion path exists via external references, the pair is PARTIAL(b).
  Examples:
  - sitting SBP vs standing SBP (different physiological states)
  - ACE inhibitor use vs ARB use (sibling drug classes, not member/class)
  - ACEi+ARB total dose vs ACE inhibitor dose
  - hemoglobin concentration vs MCHC (different lab analytes despite shared terminology)
  - LVEF value (%) vs LVEF measurement method (different dimensions of same entity)
  - all-cause death (yes/no) vs HF hospitalization or death (yes/no) (composite cannot be decomposed from component)
  - cause of death (cancer|heart attack|accident) vs skin cancer death yes/no


# IMPORTANT RULES
- Harmonization can only reduce information, never invent it. Map unsupported values to missing/unknown rather than inventing "no". Absence is "no" only under a complete denominator; case-only or not-assessed fields stay missing.
- Source/Target order must not change the status. The same pair in reverse should receive the same verdict. Only the alignment_direction may differ based on which side is finer-grained.
- Causal or clinical association alone does not make variables harmonizable; classify as IMPOSSIBLE unless both measure the same entity or one value directly entails a broader class concept without information loss.
- Interpreted threshold variables follow the same granularity rule: same N-of-classes is COMPLETE/COMPATIBLE; different granularity (e.g., binary vs graded) is PARTIAL.
- Detection method, data source, or assessment setting (e.g., "on ECG", "during echo") does not change a diagnosis: classify as COMPLETE, or COMPATIBLE when only value representation differs. Laterality, anatomical site, time window, or granularity DO change the population — classify these as PARTIAL.
- Unit or scale differences (mg vs %, counts/min vs /min, kg vs lb) are about values representation, not about whether two variables measure the same entity. Do not treat unit match/mismatch alone as evidence of different clinical entities.
- Recognize mathematically equivalent units: ug/L = ng/mL, {{counts}}/min = /min = bpm, mg/dL = mg/100mL. Classify them as COMPLETE only in such cases. 
{extra_rules}
- member/class requires one concept to be an ancestor of the other — a shared parent does not
- Do not treat a compound/composite concept as equivalent to one of its components. A component can only support a broader harmonized variable if both variables explicitly represent that broader concept; otherwise classify as IMPOSSIBLE.
- Use only the provided metadata. Do not invent units, categories, or details that are not stated.
- When in doubt between two statuses, prefer the more conservative one.
- Before finalizing: if leaning IMPOSSIBLE, check whether a single-concept (non-composite) harmonized variable satisfies PARTIAL; if leaning PARTIAL, check whether your named harmonized variable is actually a composite/union — if so, return IMPOSSIBLE.

# CONFIDENCE
Confidence reflects how certain you are about the chosen status, regardless of which status it is:
- 0.95-1.00: unambiguous; metadata clearly supports the verdict
- 0.80-0.94: minor ambiguity (e.g., unit missing but inferable)
- 0.60-0.79: meaningful ambiguity; manual review recommended
- below 0.60: low certainty; reconsider whether the verdict is defensible

# TRANSFORM
The data operation needed to create the harmonized variable. Limitations belong in reason, not here.
- COMPLETE: ""
- COMPATIBLE: deterministic conversion (e.g., "kg = lb x 0.4536", "recode {{0->Male, 1->Female}}")
- PARTIAL: lossy reduction (e.g., "collapse categories to yes/no", "derive presence from date", "convert mg to % via target-dose reference", "member yes->class yes, member no->missing")
- IMPOSSIBLE: ""

# HARMONIZED VARIABLE
- COMPLETE/COMPATIBLE/PARTIAL: short snake_case name of the common analysis variable (e.g., "smoker_yes_no", "weight_kg", "beta_blocker_dose_pct_target")
- IMPOSSIBLE: ""

# ALIGNMENT DIRECTION
- COMPLETE: "bidirectional"
- COMPATIBLE: "bidirectional" if reversible; otherwise the valid one-way direction
- PARTIAL:
  -- "source to target" if source is finer-grained and target is the coarser harmonized meaning
  -- "target to source" if target is finer-grained and source is the coarser harmonized meaning
  -- "both for derivation" when both variables contribute partial positive evidence to a new broader harmonized variable
- IMPOSSIBLE: ""
"""
 
_PREAMBLE = """You are a clinical data harmonization expert assessing whether two variables can be aligned and merged into a common harmonized analysis variable for pooled statistical analysis.
"""

_BATCH_PREAMBLE = """You are a clinical data harmonization expert assessing whether a single Source variable can be aligned to each of several candidate Target variables and merged into a common harmonized analysis variable for pooled statistical analysis.

You will receive ONE Source and multiple Targets in a single request. Evaluate each (Source, Target i) pair independently and in isolation, using the same rules and definitions as a single-pair assessment. A target's verdict must depend only on that target and the Source — never on the presence, similarity, or verdict of any other target in the batch. Do not normalize, balance, or rank verdicts across targets. Two targets that would each receive COMPLETE in isolation must each receive COMPLETE here.
"""
SYSTEM_PROMPT_NE = (
    _PREAMBLE
    + _INPUT_NE
    + _SHARED_BODY.format(extra_rules="")
    + _OUTPUT_PAIR
)

SYSTEM_PROMPT_EV = (
    _PREAMBLE
    + _INPUT_EV
    + _SHARED_BODY.format(extra_rules=_RULES_EV_ONLY)
    + _OUTPUT_PAIR
)

SYSTEM_PROMPT_NE_BATCH = (
    _BATCH_PREAMBLE
    + _BATCH_INPUT_NE
    + _SHARED_BODY.format(extra_rules="")
    + _OUTPUT_BATCH
)

SYSTEM_PROMPT_EV_BATCH = (
    _BATCH_PREAMBLE
    + _BATCH_INPUT_EV
    + _SHARED_BODY.format(extra_rules=_RULES_EV_ONLY)
    + _OUTPUT_BATCH
)


def _build_pair_prompt(src_concepts:str = "", src_cats:str = "", tgt_concepts:str = "", tgt_cats:str = "",
                       src_desc:str = "", tgt_desc:str = "",
                       src_unit:str = "", tgt_unit:str = "",
                       evidence:str = "", mode:str = MappingType.OEH.value):
    if mode == MappingType.NE.value:
        src_line = f"Source: description: {src_desc}"
        tgt_line = f"Target: description: {tgt_desc}"
    else:
        src_line = f"Source: description: {src_desc}, concepts: {src_concepts}"
        tgt_line = f"Target: description: {tgt_desc}, concepts: {tgt_concepts}"
    if src_unit: src_line += f", unit: {src_unit}"
    if tgt_unit: tgt_line += f", unit: {tgt_unit}"
    src_line += f", categories: [{src_cats}]"
    tgt_line += f", categories: [{tgt_cats}]"
    prompt = f"## INPUT\n{src_line}\n{tgt_line}"
    if evidence:
        prompt += f"\ngraph_evidence: [{evidence}]"
    return prompt

def _build_batch_prompt(
    src_desc: str = "",
    src_concepts: str = "",
    src_cats: str = "",
    src_unit: str = "",
    targets: List[Dict[str, str]] = None,
    mode: str = MappingType.OEH.value,
) -> str:
    """Build a prompt with 1 Source + N Targets. Schema and independence rule
    live in the batch system prompt — this function only formats the data.
    """
    targets = targets or []

    if mode == MappingType.NE.value:
        src_line = f"Source: description: {src_desc}"
    else:
        src_line = f"Source: description: {src_desc}, concepts: {src_concepts}"
    if src_unit:
        src_line += f", unit: {src_unit}"
    src_line += f", categories: [{src_cats}]"

    target_lines = []
    for i, t in enumerate(targets, start=1):
        if mode == MappingType.NE.value:
            line = f"Target {i}: description: {t.get('desc', '')}"
        else:
            line = (
                f"Target {i}: description: {t.get('desc', '')}, "
                f"concepts: {t.get('tgt_concepts', '')}"
            )
        if t.get("tgt_unit"):
            line += f", unit: {t['tgt_unit']}"
        line += f", categories: [{t.get('tgt_cats', '')}]"
        if t.get("evidence"):
            line += f"\n  graph_evidence: [{t['evidence']}]"
        target_lines.append(line)

    return "## INPUT\n" + src_line + "\n" + "\n".join(target_lines)



# def _extract_logprob_dist(logprobs_obj) -> dict | None:
#     """Find the status_code token position and return {label: prob} or None."""
#     if not logprobs_obj or not logprobs_obj.content:
#         return None
#     for tok in logprobs_obj.content:
#         if tok.token.strip() in _CODE_TO_LABEL:
#             import math
#             dist = {}
#             for alt in tok.top_logprobs:
#                 label = _CODE_TO_LABEL.get(alt.token.strip())
#                 if label:
#                     dist[label] = math.exp(alt.logprob)
#             # include the sampled token itself if not already in alts
#             sampled_label = _CODE_TO_LABEL.get(tok.token.strip())
#             if sampled_label and sampled_label not in dist:
#                 dist[sampled_label] = math.exp(tok.logprob)
#             return dist if dist else None
#     return None

def _parse_batch(text: str, expected_n: int) -> List[Tuple[Optional[bool], str, float, str]]:
    """Parse a JSON array of verdicts. Returns exactly expected_n results,
    padding with PENDING entries if the model under-delivered.
    """
    _pending = (
        None,
        ContextMatchType.PENDING.value,
        0.0,
        json.dumps({"status": "PARSE_ERROR", "reason": "batch_missing_item"}),
    )

    if not text:
        return [_pending] * expected_n

    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    # Try full JSON array parse
    try:
        arr = json.loads(text)
        if isinstance(arr, dict):
            # Some models wrap in {"results": [...]}
            arr = arr.get("results") or arr.get("verdicts") or [arr]
        if not isinstance(arr, list):
            arr = [arr]
        results = [_parse_single(json.dumps(item)) for item in arr[:expected_n]]
        while len(results) < expected_n:
            results.append(_pending)
        return results
    except json.JSONDecodeError:
        pass

    # Fallback: extract balanced JSON objects from the text
    objects, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : i + 1])
                start = None

    if objects:
        results = [_parse_single(obj) for obj in objects[:expected_n]]
        while len(results) < expected_n:
            results.append(_pending)
        return results

    return [_pending] * expected_n


def _apply_logprob_confidence(parsed_tuple, logprob_dist: Dict[str, float] | None):
    """
    Replace the model-written JSON confidence with normalized logprob confidence.
    """
    matched, ctx_type, self_reported_conf, reason_json = parsed_tuple

    try:
        d = json.loads(reason_json) if isinstance(reason_json, str) else {}
    except Exception:
        d = {"reason": str(reason_json)}

    status = str(d.get("status", "")).upper().strip()
    logprob_dist = logprob_dist or {}
    
    logprob_conf = float(logprob_dist.get(status, 0.0)) if len(logprob_dist.keys()) > 0 else None

    confidence_source = (
        "normalized_status_token_logprob"
        if logprob_dist
        else "unavailable_logprob"
    )

    ranked = sorted(logprob_dist.items(), key=lambda x: x[1], reverse=True)

    top_label = ""
    top_prob = 0.0
    runner_up = ""
    margin = 0.0

    if ranked:
        top_label, top_prob = ranked[0]

    if len(ranked) >= 2:
        runner_up, second_prob = ranked[1]
        margin = float(top_prob - second_prob)

    d["llm_self_reported_confidence"] = float(self_reported_conf or 0.0)
    d["logprob_dist"] = logprob_dist
    d["logprob_confidence"] = logprob_conf
    d["logprob_top_label"] = top_label
    d["logprob_top_prob"] = float(top_prob)
    d["logprob_runner_up"] = runner_up
    d["logprob_margin"] = margin
    d["confidence_source"] = confidence_source
    conf = logprob_conf if logprob_conf else self_reported_conf
    return matched, ctx_type, conf, json.dumps(d, ensure_ascii=False)


def _clean_token(tok: str) -> str:
    return str(tok).strip().strip('"').strip("'").strip(":,{}[] \n\t").upper()


def _normalize_logprobs(label_logprobs: dict[str, float]) -> dict[str, float]:
    import math

    if not label_logprobs:
        return {}

    max_lp = max(label_logprobs.values())
    exp_vals = {
        label: math.exp(lp - max_lp)
        for label, lp in label_logprobs.items()
    }
    total = sum(exp_vals.values())

    if total <= 0:
        return {}

    return {
        label: val / total
        for label, val in exp_vals.items()
    }


def _extract_status_logprob_dist(logprobs_obj) -> dict[str, float]:
    """
    Extract normalized probabilities over COMPLETE / COMPATIBLE / PARTIAL / IMPOSSIBLE
    without changing the old prompt schema.

    Works when the status value appears as a single token in top_logprobs.
    """
    if not logprobs_obj or not getattr(logprobs_obj, "content", None):
        return {}

    for tok in logprobs_obj.content:
        sampled = _clean_token(getattr(tok, "token", ""))
        sampled_label = sampled if sampled in _LABELS else None

        if not sampled_label:
            continue

        label_logprobs = {}

        sampled_lp = getattr(tok, "logprob", None)
        if sampled_lp is not None:
            label_logprobs[sampled_label] = sampled_lp

        for alt in getattr(tok, "top_logprobs", []) or []:
            alt_token = _clean_token(getattr(alt, "token", ""))
            alt_label = alt_token if alt_token in _LABELS else None
            alt_lp = getattr(alt, "logprob", None)

            if alt_label and alt_lp is not None:
                label_logprobs[alt_label] = alt_lp

        return _normalize_logprobs(label_logprobs)

    return {}
    
# def _parse_batch(text: str, expected_n: int) -> List[Tuple[Optional[bool], int, float, str]]:
#     _fail = (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "empty")
#     if not text:
#         return [_fail] * expected_n
#     text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL).strip()
#     if text.startswith("```"):
#         text = re.sub(r"^```(?:json)?\s*", "", text)
#         text = re.sub(r"\s*```$", "", text)

#     # Try full JSON parse
#     try:
#         arr = json.loads(text)
#         if isinstance(arr, dict):
#             arr = arr.get("results", [arr])
#         results = [_parse_single(json.dumps(item)) for item in arr[:expected_n]]
#         while len(results) < expected_n:
#             results.append(_fail)
#         return results
#     except json.JSONDecodeError:
#         pass

#     # Fallback: extract individual JSON objects
#     objects = re.findall(r'\{[^{}]+\}', text)
#     if objects:
#         results = [_parse_single(obj) for obj in objects[:expected_n]]
#         while len(results) < expected_n:
#             results.append(_fail)
#         return results

#     # Last resort: single parse, pad rest
#     return [_parse_single(text)] + [_fail] * (expected_n - 1)

# def _parse_single(text: str) -> Tuple[Optional[bool], str, float, str]:
#     if not text:
#         return (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "empty_response")
#     text = text.strip()
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
#     if not text:
#         return (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "think_only_response")
#     if text.startswith("```"):
#         text = re.sub(r"^```(?:json)?\s*", "", text)
#         text = re.sub(r"\s*```$", "", text)

#     # Extract fields directly via regex — works on truncated JSON
#     status_m = re.search(r'"status"\s*:\s*"([^"]+)"', text)
#     conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
#     reason_m = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"?', text)
#     transform_m = re.search(r'"transform"\s*:\s*"((?:[^"\\]|\\.)*)"?', text)
#     transform = transform_m.group(1) if transform_m else ""
#     alignment_direction = re.search(r'"alignment_direction"\s*:\s*"([^"]+)"', text)
#     harmonized_variable = re.search(r'"harmonized_variable"\s*:\s*"([^"]+)"', text)
#     if status_m:

#         alignment_direction = alignment_direction.group(1) if alignment_direction else ""
#         harmonized_variable = harmonized_variable.group(1) if harmonized_variable else ""

#         status = status_m.group(1).upper()
#         conf = float(conf_m.group(1)) if conf_m else 0.8
#         reason = reason_m.group(1) if reason_m else "truncated"
#         reason = f"{reason}: harmonized_variable: {harmonized_variable}"
#         reason = json.dumps({
#             "status": status,
#             "llm_confidence": conf,
#             "reason": reason,
#             "transform": transform,
#             "alignment_direction": alignment_direction,
#             "harmonized_variable": harmonized_variable,
#         })        # print(f"status: {status}, conf: {conf}, reason: {reason}, transform: {transform}")
#         if status.startswith("IMPOSSIBLE"):
#             return (False, ContextMatchType.NOT_APPLICABLE.value, 0.0, reason)
#         if status.startswith("COMPLETE"):
#             return (True,ContextMatchType.EXACT.value,  max(conf, 0.9), reason)
#         if status.startswith("COMPATIBLE"):
#             return (True, ContextMatchType.COMPATIBLE.value , max(min(conf, 0.9), 0.8), reason)
#         if status.startswith("PARTIAL"):
#             return (True, ContextMatchType.PARTIAL.value, max(min(conf, 0.8), 0.75), reason)
#     return (None, ContextMatchType.PENDING.value, 0.0, "json_parse_fail")

def _parse_single(text: str) -> Tuple[Optional[bool], str, float, str]:
    if not text:
        return (None, ContextMatchType.PENDING.value, 0.0, "empty_response")

    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        return (None, ContextMatchType.PENDING.value, 0.0, "think_only_response")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return (None, ContextMatchType.PENDING.value, 0.0, "json_parse_fail")

    required = {
        "status",
        "confidence",
        "reason",
        "transform",
        "harmonized_variable",
        "alignment_direction",
    }

    missing = required - set(d.keys())
    if missing:
        return (
            None,
            ContextMatchType.PENDING.value,
            0.0,
            f"missing_required_fields:{sorted(missing)}",
        )

    status = str(d.get("status", "")).upper().strip()
    
    conf = float(d.get("confidence") or 0.0)

    reason_json = json.dumps({
        "status": status,
        "reason": d.get("reason", ""),
        "transform": d.get("transform", ""),
        "transform_direction": d.get("alignment_direction", ""),
        "alignment_direction": d.get("alignment_direction", ""),
        "harmonized_variable": d.get("harmonized_variable", ""),
    })

    if status == "IMPOSSIBLE":
        return (False, ContextMatchType.NOT_APPLICABLE.value, conf, reason_json)

    if status == "COMPLETE":
        return (True, ContextMatchType.EXACT.value, conf, reason_json)

    if status == "COMPATIBLE":
        return (True, ContextMatchType.COMPATIBLE.value, conf, reason_json)

    if status == "PARTIAL":
        return (True, ContextMatchType.PARTIAL.value, conf , reason_json)

    return (
        None,
        ContextMatchType.PENDING.value,
        0.0,
        f"invalid_status:{status}",
    )
@dataclass
class LLMConceptMatcher:
    models: List[str] = field(default_factory=list)
    max_retries: int = 5

    _study_context: str = field(default="", repr=False)

     # Deterministic generation parameters
    max_tokens: int = 2000
    temperature: float = 0.0  
    top_p: float = 0.95
    top_k: int = 0  # 0 = disabled/default for OpenRouter; omit for Gemini if 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1
    min_p: float = 0.0
    top_a: float = 0.0

    timeout: int = 2000
    mode: str = MappingType.OEH.value
    backend:str = "google"
    _clients: Dict[str, Any] = field(default_factory=dict, repr=False)
    batching:bool =False
    def __post_init__(self):
        self._cache = LLMDiskCache()
        if self.batching:
            sys_prompt = SYSTEM_PROMPT_NE_BATCH if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV_BATCH
        else: 

            sys_prompt = SYSTEM_PROMPT_NE if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV
        # batch_sys = SYSTEM_PROMPT_NE_BATCH if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV_BATCH
        self._cache.set_system_prompt(sys_prompt)
        model = self.models[0]

        self.backend = self._backend_for(model)
        self.temperature = 1.0 if self.backend == "google" else self.temperature
        if self.backend  == "ollama":
             
            from ollama import Client
            self._clients["ollama"] = Client(host=settings.OLLAMA_URL)
        # elif backend == "together":
        #     from together import Together
        #     self._clients["together"] = Together(api_key=settings.TOGETHER_API_KEY, timeout=self.timeout)
        # elif backend == "openai":
        #     from openai import OpenAI
        #     self._clients["openai"] = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=self.timeout)
        # elif backend == "anthropic":
        #     from anthropic import Anthropic
        #     self._clients["anthropic"] = Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=self.timeout)
        elif self.backend  == "google":
            from google.genai import Client
            self._clients["google"] = Client(api_key=settings.GEMINI_API_KEY)
        elif self.backend  == "openrouter":
            from openrouter import OpenRouter
            self._clients["openrouter"] = OpenRouter(api_key=settings.OPENROUTER_API_KEY)

    @staticmethod
    def _backend_for(model: str) -> str:
        m = model.lower()
        if "ollama/" in m: return "ollama"
        if "openrouter/" in m: return "openrouter"
        # if m.startswith("gpt-"): return "openai"
        # if m.startswith("claude-"): return "anthropic"
        if m.startswith("gemini-") or m.startswith("gemma-"): return "google"
        return "openrouter"

    def _openrouter_generation_params(self) -> Dict[str, Any]:
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "service_tier":"flex"
        }

        # Only send non-default optional controls.
        # top_k=0, min_p=0.0, top_a=0.0 mean disabled.
        if self.top_k and self.top_k > 0:
            params["top_k"] = self.top_k

        if self.repetition_penalty != 1.0:
            params["repetition_penalty"] = self.repetition_penalty

        if self.min_p and self.min_p > 0.0:
            params["min_p"] = self.min_p

        if self.top_a and self.top_a > 0.0:
            params["top_a"] = self.top_a

        return params

    def _gemini_generation_config(self, sys_prompt: str, force_freeform: bool = False):
        from google.genai import types

        kwargs = {
            "system_instruction": sys_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "thinking_config": types.ThinkingConfig(
                thinking_level="medium"
            ),
        }

        # Use JSON mode on first attempt, but allow free-form fallback on retries.
        if not force_freeform:
            kwargs["response_mime_type"] = "application/json"

        # For Gemini, do not send top_k=0. Treat 0 as disabled/default.
        if self.top_k and self.top_k > 0:
            kwargs["top_k"] = self.top_k

        return types.GenerateContentConfig(**kwargs)
        
    def _cache_mode(self) -> str:
        return f"{self.mode}::{self._generation_fingerprint()}"
    # def _generation_fingerprint(self) -> str:
    #     cfg = {
    #         "max_tokens": self.max_tokens,
    #         "temperature": self.temperature,
    #         "top_p": self.top_p,
    #         "top_k": self.top_k,
    #         "frequency_penalty": self.frequency_penalty,
    #         "presence_penalty": self.presence_penalty,
    #         "repetition_penalty": self.repetition_penalty,
    #         "min_p": self.min_p,
    #         "top_a": self.top_a,
    #     }
    #     return hashlib.sha256(
    #         json.dumps(cfg, sort_keys=True).encode()
    #     ).hexdigest()[:12]

    def _generation_fingerprint(self) -> str:
        cfg = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "min_p": self.min_p,
            "top_a": self.top_a,

            # logprob-aware cache separation
            "logprobs": self.backend == "openrouter",
            "top_logprobs": 20 if self.backend == "openrouter" else 0,
            "confidence_source": "normalized_status_token_logprob_v1",
        }
        return hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12]
    def _call_one(self, model: str, prompt: str, force_freeform: bool = False) ->  tuple[str, dict]:

        # output_token_limit = self.max_tokens

        output_token_limit = self.max_tokens
        api_model = model.replace("openrouter/", "") if self.backend == "openrouter" else model
        mname = model.split('/')[-1]
        client = self._clients[self.backend]

        if self.batching:
            sys_prompt = SYSTEM_PROMPT_NE_BATCH if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV_BATCH
        else:
            sys_prompt = SYSTEM_PROMPT_NE if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV
        api_model = model.replace("openrouter/", "") if self.backend == "openrouter" else model
        mname = model.split('/')[-1]
        client = self._clients[self.backend]
        logprob_dist = None
        try:
            if self.backend == "ollama":
                resp = client.chat(
                    model=api_model.replace("ollama/", ""),
                    messages=[{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}],
                    options={"temperature": self.temperature, "num_predict": output_token_limit},
                )
                result = resp["message"]["content"] or ""
                print(f"[ollama] got {len(result)} chars")

            # elif backend == "together":
            #     print(f"api_model: {api_model}")
            #     resp = client.chat.completions.create(
            #         model=api_model, temperature=self.temperature, max_tokens=token_limit,
            #         messages=[{"role": "system", "content": sys_prompt},
            #                 {"role": "user", "content": prompt}],
            #     )
            #     result = resp.choices[0].message.content or ""
            # elif backend == "openrouter":
            #     resp = client.chat.send(
            #         model=api_model, temperature=self.temperature, max_tokens=output_token_limit,
            #         stream=False,
            #         messages=[{"role": "system", "content": sys_prompt},
            #                 {"role": "user", "content": prompt}],
            #                  reasoning={
            #          "effort": "low"  # "low", "medium", or "high"
            #             },
            #     )
            #     result = resp.choices[0].message.content if resp.choices else ""

            elif self.backend == "openrouter":
                kwargs = {
                    "model": api_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "reasoning": {"effort": "medium"},
                  
                    # "reasoning": {
                    #     "effort": "medium"
                    # },
                    # "provider": {
                    #     "require_parameters": True
                    # },
                     "logprobs": True,
                    "top_logprobs": 20,
                    **self._openrouter_generation_params(),
                }

                # Use JSON mode on first attempt only.
                # If a routed provider rejects JSON mode, retry free-form and parse with regex.
                if not force_freeform:
                    kwargs["response_format"] = {
                        "type": "json_object"
                    }

                resp = client.chat.send(**kwargs)
                if not resp.choices:
                    return "", {}

                choice = resp.choices[0]
                text = choice.message.content or ""
                logprob_dist = _extract_status_logprob_dist(getattr(choice, "logprobs", None))
                print("DEBUG status text:", text[:500])
                logprobs_obj = getattr(choice, "logprobs", None)
                print("DEBUG logprob_dist:", _extract_status_logprob_dist(logprobs_obj))
                return text, logprob_dist
            # elif backend == "openai":
            #     msgs = [{"role": "system", "content": sys_prompt},
            #             {"role": "user", "content": prompt}]
            #     kwargs = dict(model=api_model, messages=msgs)
            #     kwargs["max_completion_tokens"] = token_limit
            #     kwargs["temperature"] = self.temperature
            #     kwargs["reasoning_effort"] = "low"
            #     if not force_freeform:
            #         kwargs["response_format"] = {"type": "json_object"}
            #     resp = client.chat.completions.create(**kwargs)
            #     result = resp.choices[0].message.content or ""

            # elif backend == "anthropic":
            #     resp = client.messages.create(
            #         model=api_model, max_tokens=token_limit, temperature=self.temperature,
            #         system=sys_prompt,
            #         messages=[{"role": "user", "content": prompt}],
            #     )
            #     result = resp.content[0].text if resp.content else ""
            elif self.backend == "google":
                resp = client.models.generate_content(
                    model=api_model,
                    contents=prompt,
                    config=self._gemini_generation_config(sys_prompt, force_freeform=force_freeform),
                )
                result = resp.text or ""
            else:
                result = ""
        except Exception as e:
            print(f"[{mname}] call failed: {e}")
            return "", {}
        return result, logprob_dist

    # def _eval_pair(self, model: str, prompt: str) -> Tuple[Optional[bool], str, float, str]:
    #     cache_mode = self._cache_mode()
    #     cached = self._cache.get(model, prompt, mode=cache_mode)
    #     if cached:
    #         v = _parse_single(cached)
    #         if v[0] is not None: 
    #             return v
    #     for attempt in range(self.max_retries):
    #         text, logprob_dist = self._call_one(model, prompt, force_freeform=(attempt > 1))
    #         v = _parse_single(text)
    #         if v[0] is not None:
    #             self._cache.put(model, prompt, text, mode=cache_mode)
    #             return v
    #         time.sleep(1 + attempt)
    #     return (None, ContextMatchType.PENDING.value, 0.0,
    #             json.dumps({"status": "IMPOSSIBLE", "reason": "parse_failed_after_retries"}))

    def _eval_pair(self, model: str, prompt: str) -> Tuple[Optional[bool], str, float, str]:
        cache_mode = self._cache_mode()

        cached = self._cache.get_record(model, prompt, mode=cache_mode)
        # cached = None
        if cached and cached.get("r"):
            v = _parse_single(cached["r"])
            if v[0] is not None:
                return _apply_logprob_confidence(
                    v,
                    cached.get("logprob_dist") or {},
                )

        for attempt in range(self.max_retries):
            text, logprob_dist = self._call_one(
                model,
                prompt,
                force_freeform=(attempt > 1),
            )

            v = _parse_single(text)

            if v[0] is not None:
                v = _apply_logprob_confidence(v, logprob_dist)

                self._cache.put_record(
                    model,
                    prompt,
                    text,
                    mode=cache_mode,
                    logprob_dist=logprob_dist,
                )

                return v

            time.sleep(1 + attempt)

        return (
            None,
            ContextMatchType.PENDING.value,
            0.0,
            json.dumps({
                "status": "IMPOSSIBLE",
                "reason": "parse_failed_after_retries",
                "logprob_dist": {},
                "logprob_confidence": 0.0,
                "confidence_source": "unavailable_logprob",
            }),
        )
        
    # def _eval_batch(self, model: str, prompt: str, expected_n: int) -> List[Tuple]:
    #     cached = self._cache.get(model, prompt, mode=self.mode, batch=True)
    #     if cached:
    #         v = _parse_batch(cached, expected_n)
    #         if sum(1 for x in v if x[0] is not None) == expected_n: return v
    #     for attempt in range(self.max_retries):
    #         text = self._call_one(model, prompt, force_freeform=(attempt > 0), batch=True)
    #         v = _parse_batch(text, expected_n)
    #         parsed = sum(1 for x in v if x[0] is not None)
    #         if parsed == expected_n:
    #             self._cache.put(model, prompt, text, mode=self.mode, batch=True)
    #             return v
    #         if parsed > 0 and attempt == self.max_retries - 1: return v
    #         time.sleep(1 + attempt)
    #     return [(None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "batch_parse_fail")] * expected_n

    def _is_pending_llm_result(self, result) -> bool:
        """
        result format:
            (matched, ctx_type, conf, reason_json)

        Pending means: technical failure / parse failure / unresolved LLM output.
        """
        if result is None:
            return True

        matched, ctx_type, conf, reason_json = result

        if matched is None:
            return True

        if ctx_type == ContextMatchType.PENDING.value:
            return True

        try:
            d = json.loads(reason_json) if isinstance(reason_json, str) else {}
            status = str(d.get("status", "")).upper()
            reason = str(d.get("reason", "")).lower()

            if status in {"PENDING", "PARSE_ERROR"}:
                return True

            if "parse_failed" in reason or "json_parse_fail" in reason:
                return True

        except Exception:
            pass

        return False
    # def assess(self, groups: List[Dict], case_ids: List[str] = None) -> Tuple[List[List[Tuple]], Dict]:
    #     if not groups: return [], {}
    #     model = self.models[0]
    #     flat_meta, prompts = [], []
    #     for g_idx, g in enumerate(groups):
    #         for t_idx, t in enumerate(g["targets"]):
    #             flat_meta.append((g_idx, t_idx))
    #             prompts.append(_build_pair_prompt(
    #                 g.get("src_concepts", ""), g.get("src_cats", ""),
    #                 t.get("tgt_concepts", ""), t.get("tgt_cats", ""),
    #                 src_desc=g.get("src_desc", ""), tgt_desc=t.get("desc", ""),
    #                 src_unit=g.get("src_unit", ""), tgt_unit=t.get("tgt_unit", ""),
    #                 evidence=t.get("evidence", ""), mode=self.mode))
    #     with ThreadPoolExecutor(max_workers=5) as pool:
    #         futures = {pool.submit(self._eval_pair, model, p): i for i, p in enumerate(prompts)}
    #         results = [None] * len(prompts)
    #         for fut in as_completed(futures):
    #             results[futures[fut]] = fut.result()
    #     grouped = [[] for _ in range(len(groups))]
    #     for fi, (g_idx, _) in enumerate(flat_meta):
    #         grouped[g_idx].append(results[fi])
    #     return grouped, {"total_targets": len(prompts), "model": model}

    def set_study_context(self, context_block: str) -> None:
        """Inject a cohort-pair context block to be prepended to every
        user prompt built during this run.
        """
        self._study_context = (context_block or "").strip()
    
    def assess(
            self,
            groups: List[Dict],
            case_ids: List[str] = None,
            max_pending_rounds: int = 2,
        ) -> Tuple[List[List[Tuple]], Dict]:

            if not groups:
                return [], {}

            model = self.models[0]

            flat_meta, prompts = [], []

            for g_idx, g in enumerate(groups):
                for t_idx, t in enumerate(g["targets"]):
                    flat_meta.append((g_idx, t_idx))
                   
                    pair_prompt =  _build_pair_prompt(
                            g.get("src_concepts", ""),
                            g.get("src_cats", ""),
                            t.get("tgt_concepts", ""),
                            t.get("tgt_cats", ""),
                            src_desc=g.get("src_desc", ""),
                            tgt_desc=t.get("desc", ""),
                            src_unit=g.get("src_unit", ""),
                            tgt_unit=t.get("tgt_unit", ""),
                            evidence=t.get("evidence", ""),
                            mode=self.mode,
                        )
                    
                    if self._study_context and self._study_context != "":
                        pair_prompt = f"{self._study_context}\n\n{pair_prompt}"
                    prompts.append(pair_prompt)

            results = [None] * len(prompts)

            # Initially all prompts are active.
            active_indices = list(range(len(prompts)))

            total_calls = 0
            pending_rounds_used = 0

           
            for round_idx in range(max_pending_rounds + 1):
                if not active_indices:
                    break

                pending_rounds_used = round_idx

                if len(active_indices) == 1:
                    # Single prompt — call directly, no pool overhead
                    i = active_indices[0]
                    try:
                        results[i] = self._eval_pair(model, prompts[i])
                    except Exception as e:
                        results[i] = (
                            None,
                            ContextMatchType.PENDING.value,
                            0.0,
                            json.dumps({
                                "status": "PARSE_ERROR",
                                "reason": f"eval_pair_exception: {type(e).__name__}: {e}",
                                "transform": "",
                                "transform_direction": "",
                            }),
                        )
                else:
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futures = {
                            pool.submit(self._eval_pair, model, prompts[i]): i
                            for i in active_indices
                        }
                        for fut in as_completed(futures):
                            i = futures[fut]
                            try:
                                results[i] = fut.result()
                            except Exception as e:
                                results[i] = (
                                    None,
                                    ContextMatchType.PENDING.value,
                                    0.0,
                                    json.dumps({
                                        "status": "PARSE_ERROR",
                                        "reason": f"eval_pair_exception: {type(e).__name__}: {e}",
                                        "transform": "",
                                        "transform_direction": "",
                                    }),
                                )

                total_calls += len(active_indices)
                active_indices = [
                    i for i in active_indices
                    if self._is_pending_llm_result(results[i])
                ]

            # After all retry rounds, unresolved items remain as technical failures.
            unresolved_indices = [
                i for i, r in enumerate(results)
                if self._is_pending_llm_result(r)
            ]

            for i in unresolved_indices:
                results[i] = (
                    None,
                    ContextMatchType.PENDING.value,
                    0.0,
                    json.dumps({
                        "status": "PARSE_ERROR",
                        "reason": "pending_after_assess_retries",
                        "transform": "",
                        "transform_direction": "",
                    }),
                )

            grouped = [[] for _ in range(len(groups))]

            for fi, (g_idx, _) in enumerate(flat_meta):
                grouped[g_idx].append(results[fi])

            return grouped, {
                "total_targets": len(prompts),
                "total_llm_calls": total_calls,
                "pending_after_retries": len(unresolved_indices),
                "pending_rounds_used": pending_rounds_used,
                "model": model,
            }

    # def assess_batch(self, groups: List[Dict], case_ids: List[str] = None) -> Tuple[List[List[Tuple]], Dict]:
    #     if not groups: return [], {}
    #     model = self.models[0]
    #     prompts, sizes = [], []
    #     for g in groups:
    #         prompts.append(_build_batch_prompt(
    #             src_desc=g.get("src_desc", ""), src_concepts=g.get("src_concepts", ""),
    #             src_cats=g.get("src_cats", ""), src_unit=g.get("src_unit", ""),
    #             targets=g["targets"], mode=self.mode))
    #         sizes.append(len(g["targets"]))
    #     with ThreadPoolExecutor(max_workers=5) as pool:
    #         futures = {pool.submit(self._eval_batch, model, prompts[i], sizes[i]): i
    #                    for i in range(len(prompts))}
    #         results = [None] * len(prompts)
    #         for fut in as_completed(futures):
    #             i = futures[fut]
    #             try: results[i] = fut.result()
    #             except Exception as e:
    #                 results[i] = [(None, ContextMatchType.NOT_APPLICABLE.value, 0.0, str(e))] * sizes[i]
    #     return results, {"total_groups": len(prompts), "total_targets": sum(sizes), "model": model}