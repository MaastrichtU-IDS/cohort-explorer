from __future__ import annotations
import json, re, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_model import ContextMatchType, MappingType

import hashlib
from pathlib import Path


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

    def put(self, model: str, prompt: str, response: str, mode: str = ""):
        self._path(model, prompt, mode).write_text(
            json.dumps({"prompt": prompt, "r": response}))

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
- reason: 30 words or fewer; prioritize clarity and explanation over word count.
- transform: 40 words or fewer, or ""
- harmonized_variable: 12 words or fewer, snake_case, or ""
- alignment_direction: "bidirectional" | "source to target" | "target to source" | "both for derivation" | ""

# OUTPUT FORMAT
Return ONLY one valid JSON object:
{{
  "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
  "confidence": <float>,
  "reason": "<30 words or fewer>",
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
4. If a common variable exists: can at least one side fully populate it, while the other side contributes without fabricating values (mapping to missing/unknown is acceptable)?

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
  1. The variables measure the same clinical entity, or one variable is a subtype/member/detail of the other.
  2. The harmonized_variable must represent ONE clinical concept, not a union such as "A or B".
  3. Each side must be able to populate the harmonized variable without turning its negative values into unknown. If one side maps only "yes" and all "no" values become unknown, classify as IMPOSSIBLE for pooled yes/no harmonization.
  4. Values may be collapsed or mapped to missing/unknown, but never to invented quantitative values, subtype labels, causes, procedures, or patient facts.
  5. If one side records only a broad class and the other records a subtype/detail, the harmonized_variable must be the broad class unless the broad side explicitly encodes the subtype/detail.Do not create a subtype/detail harmonized_variable from a broad-class value.
  6. The harmonized variable is directly supported by the provided metadata — not a new concept invented by merging two different entities.

  Sub-patterns:
  (a) Granularity reduction — one side is finer-grained; collapse to the coarser level.
  (b) External-reference alignment — same or class-related entity in different scales/units, convertible via external references (target-dose tables, drug-equivalence tables, conversion factors). Member-drug vs class-level dose across scales (e.g., mg vs % of target) qualifies. Excluded: composites/sums of sibling classes (e.g., "A + B" vs "A") — those are IMPOSSIBLE.
  (c) Asymmetric member/class — a specific-drug indicator vs a generic class-level variable. The harmonized variable should at the broader class level. 

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
  4. On BOTH sides, only one polarity (e.g., only "yes") can be mapped and the opposite is uninformative — meaning neither side can fully populate any valid harmonized variable.
  5. A composite variable on one side cannot be decomposed into the specific component on the other side.
  6. The variables capture different clinical dimensions of the same entity (e.g., a quantitative value vs the method used to obtain it, disease presence vs intervention for the disease, cause of death vs death following a procedure).

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
- Harmonization can only reduce information, never invent it.  If unsupported values cannot be safely interpreted, map them to missing/unknown rather than inventing "no" or a more specific value.
- Source/Target order must not change the status. The same pair in reverse should receive the same verdict. Only the alignment_direction may differ based on which side is finer-grained.
- Causal or clinical association alone does not make variables harmonizable; classify as IMPOSSIBLE unless both measure the same entity or one value directly entails a broader class/member concept without information loss.
- Interpreted threshold variables follow the same granularity rule: same N-of-classes is COMPLETE/COMPATIBLE; different granularity (e.g., binary vs graded) is PARTIAL.
- Detection method, data source, or assessment setting (e.g., "on ECG", "during echo") does not change a diagnosis: classify as COMPLETE, or COMPATIBLE when only value representation differs. Laterality, anatomical site, time window, or granularity DO change the population — classify these as PARTIAL.
- Unit or scale differences (mg vs %, counts/min vs /min, kg vs lb) are about values representation, not about whether two variables measure the same entity. Do not treat unit match/mismatch alone as evidence of different clinical entities.
- Recognize mathematically equivalent units: ug/L = ng/mL, {{counts}}/min = /min = bpm, mg/dL = mg/100mL. Classify them as COMPLETE only in such cases. 
{extra_rules}
- Class/member relationships: member "yes" → class "yes"; member "no" → missing (absence of one member doesn't exclude others). If the class side has full data, this is PARTIAL(c).
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
        }
        return hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()
        ).hexdigest()[:12]
    def _call_one(self, model: str, prompt: str, force_freeform: bool = False) -> str:

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
        sys_prompt = SYSTEM_PROMPT_NE if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV
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
                    "reasoning": {
                        "effort": "medium"
                    },
                    **self._openrouter_generation_params(),
                }

                # Use JSON mode on first attempt only.
                # If a routed provider rejects JSON mode, retry free-form and parse with regex.
                if not force_freeform:
                    kwargs["response_format"] = {
                        "type": "json_object"
                    }

                resp = client.chat.send(**kwargs)
                result = resp.choices[0].message.content if resp.choices else ""
                
          
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
            return ""
        return result

    def _eval_pair(self, model: str, prompt: str) -> Tuple[Optional[bool], str, float, str]:
        cache_mode = self._cache_mode()
        cached = self._cache.get(model, prompt, mode=cache_mode)
        if cached:
            v = _parse_single(cached)
            if v[0] is not None: 
                return v
        for attempt in range(self.max_retries):
            text = self._call_one(model, prompt, force_freeform=(attempt > 1))
            v = _parse_single(text)
            if v[0] is not None:
                self._cache.put(model, prompt, text, mode=cache_mode)
                return v
            time.sleep(1 + attempt)
        return (None, ContextMatchType.PENDING.value, 0.0,
                json.dumps({"status": "IMPOSSIBLE", "reason": "parse_failed_after_retries"}))

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

    