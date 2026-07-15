from __future__ import annotations
import json, re, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .utils import setup_logger
from .config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_model import ContextMatchType, MappingType

import hashlib
from pathlib import Path

logger = setup_logger("llm_logs.log")
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
                # logger.info(f"cached = {data}")
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
- categories: allowed values in format [original value=readable label|original value=readable label]
"""

_BATCH_INPUT_NE = """
# INPUT
One Source variable and multiple target variables are provided. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
Each variable has:
- description: variable label from study metadata
- unit: measurement unit, if available
- categories: allowed values in format [original value=readable label|original value=readable label]
"""
 
_INPUT_EV = """
# INPUT
The source and target variables originate from separate studies. Harmonization pools these patients into a single patient-level analysis variable, one row per patient; the two sides are therefore never repeated measurements of the same individual.
Two variables are provided: Source and Target. Source/Target are positional labels only.
Each variable may include:
- Description: short variable label
- Concepts: ordered concepts separated by " | "
  - first concept = primary concept (authoritative for meaning)
  - remaining concepts = refinements that may narrow or qualify meaning
- categories: allowed values in format [original value=readable label|original value=readable label]
- Unit: measurement unit, if available
- Graph evidence: optional hierarchical structure of primary standard concept from controlled vocabularies
"""

_BATCH_INPUT_EV = """
# INPUT
One Source variable and multiple target variables are provided. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
Each variable may include:
- Description: short variable label
- Concepts: ordered concepts separated by " | "
  - first concept = primary concept (authoritative for meaning)
  - remaining concepts = refinements that may narrow or qualify meaning
- categories: allowed values in format [original value=readable label|original value=readable label]
- Unit: measurement unit, if available
- Graph evidence: hierarchical structure of primary standard concept from controlled vocabularies
"""

_OUTPUT_PAIR = """
# OUTPUT FORMAT
Return ONLY one valid JSON object:
{{
  "status_code": <1|2|3|4> (1=COMPLETE, 2=COMPATIBLE, 3=PARTIAL, 4=IMPOSSIBLE),
  "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
  "confidence": <float>,
  "reason": "<50 words or fewer; prioritize clarity and explanation over word count. explain which rules you applied and why.>",
  "transform": "<40 words or fewer, or empty>",
  "harmonized_variable": "<12 words or fewer; snake_case or empty>",
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
    "reason": "<50 words or fewer>",
    "transform": "<40 words or fewer, or empty>",
    "harmonized_variable": "<snake_case or empty>",
    "alignment_direction": "<direction or empty>"
  }}
]
"""

_STUDY_CONTEXT_RULES = """

# STUDY CONTEXT
Use study metadata only when relevant to judging whether the harmonized variable is meaningful for pooled analysis.
Use cohort population, inclusion criteria, and shared morbidities to identify variables that would be constant (zero variance), structurally non-informative , or non-comparable across cohorts. 
"""
_SHARED_BODY = """
{study_context_block}
### MANDATORY SEMANTIC GATES

Assess harmonizability for pooled patient-level analysis, not surface semantic similarity.

For each variable, where the metadata allows, identify:
1. Clinical entity — condition, measurement, medication, procedure, or event.
2. Information axis — presence, severity, amount, dose, date, frequency, cause, method, or other attribute.
3. Observation frame — history/ever, current state, during a specified test, incident event, cumulative period, or point-in-time assessment.
4. Anatomical scope — general, organ-specific, regional, unilateral, bilateral, or other site restriction.
5. Composition — single entity, parent class, subtype/member, union, sum, aggregate, or residual "other" field.
6. Value support — which clinical states are explicitly represented and what a missing value means.
7. Timepoint — baseline, follow-up, event date, or other study period.

Apply the following gates before assigning a status.
** Composite/component gate. On each side identify whether it is a single entity, a parent class, a member/subtype, or a pre-composed aggregate (sum or union), and whether the axis is dose or presence.
    - A pre-composed aggregate of members matched to the parent class they exhaust, with the axis-correct operator (sum for dose, logical OR for presence), equals the class total → valid. Classify by representation: COMPLETE if value and unit align, COMPATIBLE if a lossless reversible recode/conversion is needed, PARTIAL if lossy or external-reference dependent. If the members do not exhaust the class, PARTIAL, or IMPOSSIBLE where the shortfall cannot be quantified.
    - An aggregate or composite matched to a single member, or to a different aggregate, would require isolating a component from a combined figure → IMPOSSIBLE.
    - An operator that does not match the axis (sum on presence, OR on dose) constructs a different variable → IMPOSSIBLE.
    - Rolling members up to their class is permitted; pulling a component out of a composite is not.
** Residual-field gate. "Other X" or "remaining X" excludes the separately itemized members of X, so its membership is unknown from the pair alone. It cannot align to a single itemized member (disjoint by construction) or to total X (unknown membership) → IMPOSSIBLE. A true parent "any X" that excludes nothing is not a residual field; a specific member maps to it as subtype→parent (PARTIAL).
** Anatomical-scope gate. Laterality and site restriction are defining when one variable can be positive while the other is negative for the same patient. Identical category sets do not override a scope difference.
** Setting/default gate. A defining qualifier (posture, physiological state, specimen, provocation, assay) present on one side and omitted on the other is read as the conventional default for that measurement, but only where a recognized clinical default exists. COMPLETE/COMPATIBLE only if the specified value is that default; if it departs from the default, or if both sides specify conflicting values, no single pooled variable exists → IMPOSSIBLE. Where no recognized default exists (assay/method, device scale, anatomical site), an omitted qualifier stays unverifiable — do not upgrade to COMPLETE.
** Value-mapping gate. For every explicit observed value, determine whether it maps to a valid harmonized value or must remain unknown. Never map missing, not recorded, not assessed, or an unsupported negative to "No".

### STATUS BOUNDARIES

COMPLETE
Same clinical entity, information axis, observation frame, anatomical scope, granularity, and value meaning; values merge as-is, with no recoding, conversion, threshold reinterpretation, or category normalization. Mathematically equivalent unit notation is allowed only when the numeric values are unchanged.
Examples:
- systolic BP (mmHg) vs sitting systolic BP (mmHg)  [seated is the office-BP default]
- NT-proBNP (ug/L) vs NT-proBNP (ng/mL)  [1 ug/L = 1 ng/mL]
- central venous pressure > 6 cmH2O (1=yes|0=no) vs jugular vein elevated (1=yes|0=no)
- atrial fibrillation at baseline (t=yes|f=no) vs atrial fibrillation on ECG at baseline (t=yes|f=no)

COMPATIBLE
The same six attributes as COMPLETE, but value representations differ and a deterministic, lossless, reversible transformation aligns them — unit conversion, or bijective recoding (equal number of distinct clinical states). Clinical association or approximate interchangeability is not sufficient; a surrogate qualifies only where the metadata or adjudication policy establishes equivalence and a deterministic mapping.
Examples:
- weight (kg) vs weight (lb)
- myocardial infarction (yes|no) vs myocardial infarction (t=yes|f=no)
- aspartate aminotransferase [enzymatic activity/volume] in /L vs AST measurement in (U/L) 
- central venous pressure > 6 cmH2O (3=yes|1=no) vs jugular vein elevated (0=no|1=yes) 

PARTIAL
One clinically meaningful variable can be built through a lossy, directional, or externally supported transformation. All must hold:
1. Same entity, or a subtype/member/scope-restriction relationship identifiable from the provided metadata.
2. The harmonized variable is one clinical concept, not a union or sum.
3. Every explicit observed value is mapped validly or retained as unknown.
4. No unsupported negative or missing value becomes "No".
5. The transformation names the information loss: category collapse, positive-only derivation, observation-frame reduction, anatomical reduction, datetime approximation, or external-reference conversion.
Examples:
- Year/date of diabetes diagnosis vs diabetes history: recorded date→yes, missing→unknown (positive-only).
- Atrial fibrillation during ECG vs history of atrial fibrillation: different frames; ECG yes→history yes, ECG no→unknown.
- LVEF category (1:<40%, 2:40-49%, 3:>=50%) vs LVEF <40% (1=yes|0=no): collapse cat1→yes, cat2/3→no (target→source; not reversible).
- Ordinal pulmonary-rales extent (0=absent | 1= few basal | 2=less than lower third of thorax | 3= more than thord of thorax) vs basal rales (1=yes/0=no): maps positive — 1→yes; 0→no; 2,3→no (extent exceeds the basal zone, (category collapse).
- Left-leg edema vs general lower-limb edema: harmonize as presence — left-leg positive→general positive, left-leg negative→general unknown; severity not comparable across scopes.
- captopril dose (mg) vs ACE-inhibitor dose (% target): single member→class via external target-dose conversion.

IMPOSSIBLE
No single pooled variable can be built without ambiguous decomposition or unsupported inference:
- ARB-or-ACE use (yes/no) vs ACE-inhibitor use (yes/no)  [union → component].
- Sum of ACE-inhibitor and ARB dose vs ARB dose  [aggregate → component].
- "Other ARB" dose vs total ARB dose  [residual; membership unknown].
- captopril dose (mg) vs trandolapril dose (mg)  [sibling drugs; raw mg not comparable across agents].
- disease severity vs disease etiology  [different axes].
- sitting systolic BP (mmHg) vs standing systolic BP (mmHg)  [conflicting specified settings].

### FINAL VERIFICATION
1. Is the harmonized variable one single clinical concept?
2. Are information axis and observation frame preserved, or explicitly reduced with the loss named?
3. Is every explicit value mapped, or safely retained as unknown?
4. Did any missing or unsupported value become "No"?
5. Did a composite, residual field, sibling relationship, anatomical restriction, setting conflict, or history/current distinction invalidate the match?


# CONFIDENCE
Certainty in the chosen status, whatever it is:
- 0.95-1.00: unambiguous
- 0.80-0.94: minor ambiguity (e.g., unit missing but inferable)
- 0.60-0.79: meaningful ambiguity; manual review advised
- below 0.60: low certainty; reconsider the verdict

# TRANSFORM
The data operation that builds the harmonized variable; limitations go in reason.
- COMPLETE: ""
- COMPATIBLE: deterministic conversion ("kg = lb x 0.4536", "recode {{0 maps to no,1 maps to yes}}")
- PARTIAL: lossy reduction ("collapse to yes/no", "derive presence from date", "mg to % via target-dose external conversion", "specific subtype yes -> broader class yes; specific subtype no -> broader class unknown/missing")
- IMPOSSIBLE: ""

# HARMONIZED VARIABLE
- COMPLETE/COMPATIBLE/PARTIAL: short snake_case name ("smoker_yes_no", "weight_kg")
- IMPOSSIBLE: ""

# ALIGNMENT DIRECTION
- COMPLETE: "bidirectional"
- COMPATIBLE: "bidirectional" if reversible, else the valid one-way direction
- PARTIAL: "source to target" (source finer) | "target to source" (target finer) | "both for derivation" (both contribute positive evidence to a broader variable)
- IMPOSSIBLE: ""

"""
 
_PREAMBLE = """You are a clinical data harmonization expert assessing whether two variables from separate studies can be aligned into a common harmonized variable for pooled patient-level analysis. Determine whether the merge is clinically meaningful and whether any required transformation is supported by clinical guidelines or accepted domain knowledge.
The source and target variables come from different cohorts. Harmonization pools different patients into one dataset, with one row per patient; therefore, the two sides are never repeated measurements of the same individual."""

_BATCH_PREAMBLE = """You are a clinical data harmonization expert assessing whether a single Source variable can be aligned to each of several candidate Target variables and merged into a common harmonized analysis variable for pooled statistical analysis. The source and target variables originate from separate studies. Harmonization pools these patients into a single patient-level analysis variable, one row per patient; the two sides are therefore never repeated measurements of the same individual.
You will receive ONE Source and multiple Targets in a single request. Evaluate each (Source, Target i) pair independently and in isolation, using the same rules and definitions as a single-pair assessment. A target's verdict must depend only on that target and the Source — never on the presence, similarity, or verdict of any other target in the batch. Do not normalize, balance, or rank verdicts across targets. Two targets that would each receive COMPLETE in isolation must each receive COMPLETE here.
"""


VERDICT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status_code",
        "status",
        "confidence",
        "reason",
        "transform",
        "harmonized_variable",
        "alignment_direction",
    ],
    "properties": {
        "status_code": {"type": "integer", "enum": [1, 2, 3, 4]}, 

        "status": {
            "type": "string",
            "enum": ["COMPLETE", "COMPATIBLE", "PARTIAL", "IMPOSSIBLE"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "transform": {"type": "string"},
        "harmonized_variable": {"type": "string"},
        "alignment_direction": {
            "type": "string",
            "enum": [
                "bidirectional",
                "source to target",
                "target to source",
                "both for derivation",
                "",
            ],
        },
    }
}

def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_single_code(token: Any) -> str:
    s = str(token or "")
    m = re.fullmatch(
        r'[\s"\':,\{\}\[\]\n\r\t]*([1-4])[\s"\':,\{\}\[\]\n\r\t]*',
        s,
    )
    return m.group(1) if m else ""


def _reconstruct_token_text(tokens: list) -> str:
    return "".join(str(_get_field(t, "token", "")) for t in tokens)


def _find_token_at_char_offset(tokens: list, char_pos: int) -> int | None:
    best_i = None

    for i, tok in enumerate(tokens):
        off = _get_field(tok, "text_offset", None)
        if off is None:
            continue

        token_text = str(_get_field(tok, "token", ""))
        end = off + len(token_text)

        if off <= char_pos < end:
            return i

        if off <= char_pos:
            best_i = i

    return best_i

def _extract_status_code_logprob_evidence(logprobs_obj) -> dict:
    """
    Extract logprob evidence from the final JSON status_code decision point.

    Returns complete or truncated observed-alternative evidence.
    """
    tokens = _get_field(logprobs_obj, "content", None)

    if not tokens:
        return {
            "logprob_usable": False,
            "error": "missing_logprob_content",
            "dist": {},
        }

    full_text = _reconstruct_token_text(tokens)

    # matches = list(re.finditer(r'"status_code"\s*:\s*([1-4])', full_text))
    matches = list(re.finditer(r'"status_code"\s*:\s*"?([1-4])', full_text))

    if not matches:
        return {
            "logprob_usable": False,
            "error": "final_status_code_not_found",
            "dist": {},
        }

    m = matches[-1]
    emitted_code = m.group(1)
    digit_char_pos = m.start(1)

    token_index = _find_token_at_char_offset(tokens, digit_char_pos)

    if token_index is None:
        return {
            "logprob_usable": False,
            "error": "status_code_token_index_not_found",
            "emitted_code": emitted_code,
            "dist": {},
        }

    tok = tokens[token_index]
    sampled_code = _extract_single_code(_get_field(tok, "token", ""))

    # If landed on whitespace/punctuation, scan nearby tokens.
    if sampled_code != emitted_code:
        for j in range(token_index, min(token_index + 5, len(tokens))):
            candidate_code = _extract_single_code(_get_field(tokens[j], "token", ""))
            if candidate_code == emitted_code:
                token_index = j
                tok = tokens[j]
                sampled_code = candidate_code
                break

    if sampled_code != emitted_code:
        return {
            "logprob_usable": False,
            "error": "sampled_token_does_not_match_final_status_code",
            "emitted_code": emitted_code,
            "sampled_token": _get_field(tok, "token", ""),
            "token_index": token_index,
            "dist": {},
        }

    code_logprobs = {}

    sampled_lp = _get_field(tok, "logprob", None)
    if sampled_code in _CODE_TO_LABEL and sampled_lp is not None:
        code_logprobs[sampled_code] = float(sampled_lp)

    for alt in _get_field(tok, "top_logprobs", []) or []:
        alt_code = _extract_single_code(_get_field(alt, "token", ""))
        alt_lp = _get_field(alt, "logprob", None)

        if alt_code in _CODE_TO_LABEL and alt_lp is not None:
            code_logprobs[alt_code] = float(alt_lp)

    required_codes = {"1", "2", "3", "4"}
    observed_codes = set(code_logprobs)
    missing_codes = required_codes - observed_codes
    observability = len(observed_codes) / 4

    if len(observed_codes) < 2:
        return {
            "logprob_usable": False,
            "error": "insufficient_code_alternatives",
            "distribution_type": "unusable",
            "complete_distribution": False,
            "observability": observability,
            "token_index": token_index,
            "emitted_code": emitted_code,
            "sampled_code": sampled_code,
            "observed_codes": sorted(observed_codes),
            "missing_codes": sorted(missing_codes),
            "raw_code_logprobs": code_logprobs,
            "dist": {},
        }

    distribution_type = (
        "complete_four_class"
        if observed_codes == required_codes
        else "observed_alternatives"
    )

    label_logprobs = {
        _CODE_TO_LABEL[code]: lp
        for code, lp in code_logprobs.items()
    }

    probs = _normalize_logprobs(label_logprobs)
    dist = {
        k: float(f"{v:.8g}")
        for k, v in probs.items()
    }

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    top_label, top_prob = ranked[0]
    runner_up, runner_up_prob = ranked[1]

    label_to_code = {v: k for k, v in _CODE_TO_LABEL.items()}
    raw_margin = (
        code_logprobs[label_to_code[top_label]]
        - code_logprobs[label_to_code[runner_up]]
    )

    return {
        "logprob_usable": True,
        "error": "" if distribution_type == "complete_four_class" else "incomplete_code_alternatives",
        "distribution_type": distribution_type,
        "complete_distribution": distribution_type == "complete_four_class",
        "observability": observability,
        "token_index": token_index,
        "emitted_code": emitted_code,
        "sampled_code": sampled_code,
        "observed_codes": sorted(observed_codes),
        "missing_codes": sorted(missing_codes),
        "raw_code_logprobs": code_logprobs,
        "dist": dist,
        "top_label": top_label,
        "top_prob": round(float(top_prob), 6),
        "runner_up": runner_up,
        "runner_up_prob": round(float(runner_up_prob), 6),
        "margin": round(float(top_prob - runner_up_prob), 6),
        "raw_logprob_margin": round(float(raw_margin), 6),
    }
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
    # if evidence:
    #     prompt += f"\ngraph_evidence: [{evidence}]"

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
        # src_line = f"Source: description: {src_desc}"
        src_line = f"Source: description: {src_desc}, concepts: {src_concepts}"
    if src_unit:
        src_line += f", unit: {src_unit}"
    src_line += f", categories: [{src_cats}]"

    target_lines = []
    for i, t in enumerate(targets, start=1):
        if mode == MappingType.NE.value:
            line = f"Target {i}: description: {t.get('desc', '')}"
        else:
            # line = (
            #     f"Target {i}: description: {t.get('desc', '')}"
            # )
            line = (
                f"Target {i}: description: {t.get('desc', '')}, "
                f"concepts: {t.get('tgt_concepts', '')}"
            )
        if t.get("tgt_unit"):
            line += f", unit: {t['tgt_unit']}"
        line += f", categories: [{t.get('tgt_cats', '')}]"
        # if t.get("evidence"):
        #     line += f"\n  graph_evidence: [{t['evidence']}]"
        target_lines.append(line)

    return "## INPUT\n" + src_line + "\n" + "\n".join(target_lines)



# def _parse_batch(text: str, expected_n: int) -> List[Tuple[Optional[bool], str, float, str]]:
#     """Parse a JSON array of verdicts. Returns exactly expected_n results,
#     padding with PENDING entries if the model under-delivered.
#     """
#     _pending = (
#         None,
#         ContextMatchType.PENDING.value,
#         0.0,
#         json.dumps({"status": "PARSE_ERROR", "reason": "batch_missing_item"}),
#     )

#     if not text:
#         return [_pending] * expected_n

#     text = text.strip()
#     text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

#     if text.startswith("```"):
#         text = re.sub(r"^```(?:json)?\s*", "", text)
#         text = re.sub(r"\s*```$", "", text).strip()

#     # Try full JSON array parse
#     try:
#         arr = json.loads(text)
#         if isinstance(arr, dict):
#             # Some models wrap in {"results": [...]}
#             arr = arr.get("results") or arr.get("verdicts") or [arr]
#         if not isinstance(arr, list):
#             arr = [arr]
#         results = [_parse_single(json.dumps(item)) for item in arr[:expected_n]]
#         while len(results) < expected_n:
#             results.append(_pending)
#         return results
#     except json.JSONDecodeError:
#         pass

#     # Fallback: extract balanced JSON objects from the text
#     objects, depth, start = [], 0, None
#     for i, ch in enumerate(text):
#         if ch == "{":
#             if depth == 0:
#                 start = i
#             depth += 1
#         elif ch == "}":
#             depth -= 1
#             if depth == 0 and start is not None:
#                 objects.append(text[start : i + 1])
#                 start = None

#     if objects:
#         results = [_parse_single(obj) for obj in objects[:expected_n]]
#         while len(results) < expected_n:
#             results.append(_pending)
#         return results

#     return [_pending] * expected_n

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

    normalized_logprob =  {
        label: val / total
        for label, val in exp_vals.items()
    }

    return normalized_logprob


def _apply_logprob_confidence(parsed_tuple, logprob_evidence: Dict[str, Any] | None):
    """
    Preserve model-written confidence, and attach logprob evidence separately.

    Use logprob confidence only when logprob evidence is usable.
    """
    matched, ctx_type, self_reported_conf, reason_json = parsed_tuple

    try:
        d = json.loads(reason_json) if isinstance(reason_json, str) else {}
    except Exception:
        d = {"reason": str(reason_json)}

    status = str(d.get("status", "")).upper().strip()
    logprob_evidence = logprob_evidence or {
        "logprob_usable": False,
        "error": "unavailable_logprob",
        "dist": {},
    }

    dist = logprob_evidence.get("dist") or {}

    logprob_conf = None
    if logprob_evidence.get("logprob_usable") and dist:
        logprob_conf = float(dist.get(status, 0.0))

    d["llm_self_reported_confidence"] = float(self_reported_conf or 0.0)

    d["logprob_usable"] = bool(logprob_evidence.get("logprob_usable"))
    d["logprob_error"] = logprob_evidence.get("error", "")
    d["logprob_distribution_type"] = logprob_evidence.get("distribution_type", "")
    d["logprob_complete_distribution"] = logprob_evidence.get("complete_distribution", False)
    d["logprob_observability"] = logprob_evidence.get("observability", 0.0)

    d["observed_codes"] = logprob_evidence.get("observed_codes", [])
    d["missing_codes"] = logprob_evidence.get("missing_codes", [])
    d["raw_code_logprobs"] = logprob_evidence.get("raw_code_logprobs", {})

    d["logprob_dist"] = dist
    d["logprob_confidence"] = logprob_conf
    d["logprob_top_label"] = logprob_evidence.get("top_label", "")
    d["logprob_top_prob"] = logprob_evidence.get("top_prob", "")
    d["logprob_runner_up"] = logprob_evidence.get("runner_up", "")
    d["logprob_margin"] = logprob_evidence.get("margin", "")
    d["logprob_raw_margin"] = logprob_evidence.get("raw_logprob_margin", "")

    if logprob_evidence.get("logprob_usable"):
        d["confidence_source"] = (
            "logprob_complete_four_class"
            if logprob_evidence.get("complete_distribution")
            else "logprob_observed_alternatives"
        )
    else:
        d["confidence_source"] = "self_reported_confidence"

    # Important methodological choice:
    # Keep the model-written confidence as the operational confidence.
    # Store logprob evidence separately for uncertainty analysis.
    conf = self_reported_conf

    return matched, ctx_type, conf, json.dumps(d, ensure_ascii=False)

def _build_system_prompt(
    *,
    mode: str,
    batching: bool,
    study_context: str = "",
) -> str:
    preamble = _BATCH_PREAMBLE if batching else _PREAMBLE

    if batching:
        input_block = (
            _BATCH_INPUT_NE
            if mode == MappingType.NE.value
            else _BATCH_INPUT_EV
        )
        output_block = _OUTPUT_BATCH
    else:
        input_block = (
            _INPUT_NE
            if mode == MappingType.NE.value
            else _INPUT_EV
        )
        output_block = _OUTPUT_PAIR

    context_parts = [_STUDY_CONTEXT_RULES]

    if study_context:
        dynamic_context = study_context.strip()

        # format_study_context_block() already starts with "# STUDY CONTEXT";
        # remove it to avoid duplicate headers.
        dynamic_context = re.sub(
            r"^\s*#\s*STUDY CONTEXT\s*",
            "",
            dynamic_context,
            flags=re.IGNORECASE,
        ).strip()

        if dynamic_context:
            context_parts.append(dynamic_context)

    study_context_block = "\n".join(context_parts)

    final_prompt =(
        preamble
        + input_block
        + _SHARED_BODY.format(study_context_block=study_context_block)
        + output_block
    )
   
    return final_prompt


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
        "status_code",
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

    code = str(d.get("status_code", "")).strip()
    status = _CODE_TO_LABEL.get(code, str(d.get("status", "")).upper().strip())    
    conf = float(d.get("confidence") or 0.0)

    reason_json = json.dumps({
        "status": status,
        "status_code": code, 
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
    max_tokens: int = 4096
    temperature: float = 0.0 # 0.0
    top_p: float = 1.0  # 1.0
    top_k: int = 0  # 0 = disabled/default for OpenRouter; omit for Gemini if 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    top_a: float = 0.0

    timeout: int = 2000
    mode: str = MappingType.OEH.value
    backend:str = "google"
    _clients: Dict[str, Any] = field(default_factory=dict, repr=False)
    batching:bool =False
   
    def __post_init__(self):
        self._cache = LLMDiskCache()
        self._refresh_system_prompt()
     
        # self._cache.set_system_prompt(sys_prompt)
        model = self.models[0]

        self.backend = self._backend_for(model)
        self._apply_model_generation_defaults(model)
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
        elif self.backend == "fireworks":
            if not settings.FIREWORKS_API_KEY:
                raise ValueError(
                    "Fireworks backend selected but FIREWORKS_API_KEY is not set."
                )
            from openai import OpenAI
            self._clients["fireworks"] = OpenAI(
                api_key=settings.FIREWORKS_API_KEY,
                base_url=settings.FIREWORKS_BASE_URL,
                timeout=self.timeout,
            )
        elif self.backend == "litellm":
            from openai import OpenAI
            self._clients["litellm"] = OpenAI(
                api_key=settings.LITELLM_API_KEY,
                base_url=settings.LITELLM_BASE_URL,
                timeout=self.timeout,
            )

  

    def _refresh_system_prompt(self) -> None:
            self._system_prompt = _build_system_prompt(
                mode=self.mode,
                batching=self.batching,
                study_context=self._study_context,
            )
            self._cache.set_system_prompt(self._system_prompt)


   
    def _apply_model_generation_defaults(self, model: str) -> None:
        """Apply model-specific decoding defaults for LLM-as-classifier runs.

        top_k=0 means disabled in this code path; OpenRouter/Gemini params
        omit top_k unless the value is greater than zero.
        """
        m = model.lower()

        is_gpt_oss_120b = (
            ("gpt-oss" in m or "gpt_oss" in m or "gpt oss" in m or "gpt-120" in m)
            and ("120b" in m or "120-b" in m or "120" in m)
        )
        if is_gpt_oss_120b:
            self.max_tokens = 16384
        # logger.info (f"for LLM {m}, temperature = { self.temperature}, top_p = {self.top_p},top_k = {self.top_k} ")
    @staticmethod
    def _backend_for(model: str) -> str:
        m = model.lower()
        if "ollama/" in m: return "ollama"
        if m.startswith("fireworks/") or m.startswith("accounts/fireworks/") or m.startswith("accounts/komalsyeda29-qw87svj/") :
            return "fireworks"
        if "openrouter/" in m: return "openrouter"
        # if m.startswith("gpt-"): return "openai"
        # if m.startswith("claude-"): return "anthropic"
        if m.startswith("gemini-") or m.startswith("gemma-"): return "google"
        if "litellm/" in m: 
            return "litellm"
        return "openrouter"

    def _openrouter_generation_params(self) -> Dict[str, Any]:
        params = {
           
            # "reasoning": {"enabled": True},
            # "reasoning_effort":"high",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            # "logprobs": True,
            # Fireworks currently accepts top_logprobs in the range 0..5.
            # "top_logprobs": 10,
            # "service_tier":"flex"
           
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

    def _fireworks_model_name(self, model: str) -> str:
        """Return the Fireworks model id accepted by the OpenAI-compatible API.

        Supported inputs:
        - fireworks/deepseek-v4-flash
        - fireworks/accounts/fireworks/models/deepseek-v4-flash"
        - accounts/fireworks/models/deepseek-v4-flash
        """
        if model.startswith("fireworks/"):
            model = model.replace("fireworks/", "", 1)
        if model.startswith("accounts/fireworks/"):
            return model
        # Convenience alias for common serverless model ids.
        if "/" not in model:
            return f"accounts/fireworks/models/{model}"
        return model

    def _fireworks_generation_params(self) -> Dict[str, Any]:
        params = {
            "reasoning_effort":"high",
            # "reasoning_history": "preserved",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "logprobs": True,
            # Fireworks currently accepts top_logprobs in the range 0..5.
            "top_logprobs": 5,
        }
        if self.top_k and self.top_k > 0:
            params["top_k"] = self.top_k
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
                thinking_level="high"
            ),

            # "response_logprobs":True,
            # "logprobs":5,
        }

        # Use JSON mode on first attempt, but allow free-form fallback on retries.
        if not force_freeform:
            # kwargs["response_mime_type"] = "application/json"
            # kwargs["response_mime_type"] = {"type": "json_schema",
                        # "json_schema": {"name": "HarmonizationVerdict", "schema": VERDICT_JSON_SCHEMA}}
            kwargs["response_mime_type"] = "application/json"
            # kwargs["response_schema"] = VERDICT_JSON_SCHEMA

        # For Gemini, do not send top_k=0. Treat 0 as disabled/default.
        if self.top_k and self.top_k > 0:
            kwargs["top_k"] = self.top_k

        return types.GenerateContentConfig(**kwargs)
        
    def _cache_mode(self) -> str:
        return f"{self.mode}::{self._generation_fingerprint()}"
  

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
            "backend": self.backend,
            "logprobs": self.backend in {"openrouter", "fireworks"},
            "top_logprobs": 5 if self.backend == "fireworks" else (20 if self.backend == "openrouter" else 0),
            "confidence_source": "normalized_status_token_logprob_v1",
        }
        return hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12]
    def _call_one(self, model: str, prompt: str, force_freeform: bool = False) ->  tuple[str, dict]:

        # output_token_limit = self.max_tokens

        output_token_limit = self.max_tokens
        api_model = self._fireworks_model_name(model) if self.backend == "fireworks" else (model.replace("openrouter/", "") if self.backend == "openrouter" else model)
        mname = model.split('/')[-1]
        client = self._clients[self.backend]

      

        sys_prompt = self._system_prompt
        api_model = self._fireworks_model_name(model) if self.backend == "fireworks" else (model.replace("openrouter/", "") if self.backend == "openrouter" else model)
        mname = model.split('/')[-1]
        client = self._clients[self.backend]
        logprob_evidence = None

    
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

          

            elif self.backend == "openrouter":
                kwargs = {
                    "model": api_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "reasoning": {"effort": "high"},
                  
                    "provider": {
                        "quantizations": [
                        "bf16"
                        ]
                    },
                     "logprobs": True,
                    "top_logprobs": 10,
                    **self._openrouter_generation_params(),
                }

                # Use JSON mode on first attempt only.
                # If a routed provider rejects JSON mode, retry free-form and parse with regex.
                if not force_freeform:
                    # kwargs["response_format"] = {
                    #     "type": "json_object"
                    # }
                     kwargs["response_format"] = {"type": "json_schema",
                        "json_schema": {"name": "HarmonizationVerdict", "schema": VERDICT_JSON_SCHEMA}}
                   

                resp = client.chat.send(**kwargs)
                if not resp.choices:
                    return "", {}

                choice = resp.choices[0]
                text = choice.message.content or ""
                logprob_evidence = _extract_status_code_logprob_evidence( getattr(choice, "logprobs", None))

                print("DEBUG status text:", text[:500])
                # logprobs_obj = getattr(choice, "logprobs", None)
                # print("DEBUG logprob_dist:", _extract_status_logprob_dist(logprobs_obj))
                return text, logprob_evidence

            elif self.backend == "litellm":
                kwargs = {
                    "model": model.replace("litellm/", "", 1),
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                     "extra_body": {
                        "reasoning_effort": "high",
                        "allowed_openai_params": ["reasoning_effort"],
                        "drop_params": True,
                    },
                    **self._fireworks_generation_params(),
                }
                if not force_freeform:
                    kwargs["response_format"] = {"type": "json_schema",
                        "json_schema": {"name": "HarmonizationVerdict", "schema": VERDICT_JSON_SCHEMA}}

                resp = client.chat.completions.create(**kwargs)
                if not resp.choices:
                    return "", {}
                choice = resp.choices[0]
                text = choice.message.content or ""
                logprob_evidence = _extract_status_code_logprob_evidence(
                    getattr(choice, "logprobs", None)
                )
                return text, logprob_evidence
            elif self.backend == "fireworks":
                kwargs = {
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
           
                    **self._fireworks_generation_params(),
                }

                # Use JSON mode on early attempts; retry without it for models
                # that do not support response_format on the routed backend.
                if not force_freeform:
                    # kwargs["response_format"] = {"type": "json_object"}
                    kwargs["response_format"] = {"type": "json_schema",
                        "json_schema": {"name": "HarmonizationVerdict", "schema": VERDICT_JSON_SCHEMA}}

                resp = client.chat.completions.create(**kwargs)
                if not resp.choices:
                    return "", {}

                choice = resp.choices[0]
                text = choice.message.content or ""
                logprob_evidence = _extract_status_code_logprob_evidence(
                    getattr(choice, "logprobs", None)
                )
                return text, logprob_evidence
    
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
        
        return result, logprob_evidence

   

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
                force_freeform=(attempt > 3),
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
                        self._refresh_system_prompt()
                        # pair_prompt = f"{self._study_context}\n\n{pair_prompt}"
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

   