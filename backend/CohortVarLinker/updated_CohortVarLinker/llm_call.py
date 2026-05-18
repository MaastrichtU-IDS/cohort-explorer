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

   
    def _path(self, model: str, prompt: str, mode: str = "", batch: bool = False) -> Path:
        name = model.split("/")[-1]
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        key_material = f"{self._sys_hash}::{mode}::{model}::{prompt}"
        return d / f"{hashlib.sha256(key_material.encode()).hexdigest()}.json"

    def get(self, model: str, prompt: str, mode: str = "", batch: bool = False) -> str | None:
        p = self._path(model, prompt, mode, batch)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["r"]
        except Exception:
            p.unlink(missing_ok=True)
            return None

    def put(self, model: str, prompt: str, response: str, mode: str = "", batch: bool = False):
        self._path(model, prompt, mode, batch).write_text(
            json.dumps({"prompt": prompt, "r": response}))

    def delete(self, model: str, prompt: str, mode: str = "", batch: bool = False):
        self._path(model, prompt, mode, batch).unlink(missing_ok=True)

_INPUT_NE = """
# INPUT
Two variables are provided: Source and Target. Source/Target are positional labels only — they do not imply which side is finer or coarser, nor which side is the reference.
 
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
 
_RULES_EV_ONLY = """- The first concept is authoritative for variable meaning. Treat additional concepts as refinements only if they materially change clinical interpretation; otherwise treat them as annotation context.
- When graph evidence conflicts with descriptions or concepts, prefer the explicit metadata. Use graph evidence to break ties or confirm relationships, not to override stated meaning."""
 
_SHARED_BODY = """
# REASONING PROCEDURE
Before assigning a status, answer these questions internally:
1. What clinical entity does each variable measure? Separate the entity itself from how, where, or when it is observed.
2. Are these the same entity, or genuinely different entities?
3. If the same entity: can a single common variable be defined that both sides can populate — even if one side requires reduction, recoding, or approximation?
4. If a common variable exists: can at least one side fully populate it, while the other side contributes without fabricating values (mapping to missing/unknown is acceptable)?

Use the answers to select exactly ONE status below.

# STATUS DEFINITIONS

- COMPLETE:
  Same clinical entity, same granularity, same data representation. Values merge as-is with no transformation.
  Mathematically equivalent units with different notation (e.g., ug/L = ng/mL, {{counts}}/min = /min) are the same representation — use COMPLETE, not COMPATIBLE.
  Examples:
  - systolic BP (mmHg) vs sitting systolic BP (mmHg)
  - history of myocardial infarction (yes/no) vs myocardial infarction (yes/no)
  - HbA1c (%) vs HbA1c (%)
  - age (years) vs age at baseline (years)
  - serum creatinine (mg/dL) vs creatinine (mg/dL)
  - NT-proBNP (ug/L) vs NT-proBNP (ng/mL)

- COMPATIBLE:
  Same clinical entity, same analysis-level granularity, different data representation. Values can be merged through deterministic, lossless recoding, unit conversion, or rescaling.
  Thresholds are COMPATIBLE only if both variables already represent an interpreted clinical state.
  Examples:
  - weight (kg) vs weight (lb)
  - blood glucose (mmol/L) vs blood glucose (mg/dL)
  - Gender (Female/Male) vs Gender (m/f)
  - NYHA class (I/II/III/IV) vs NYHA class (1/2/3/4)
  - jugular vein elevated (yes/no) vs central venous pressure > 6 cm H2O (yes/no)
  - creatinine (umol/L) vs creatinine (mg/dL)

- PARTIAL:
  A valid common harmonized variable can be created, but at least one side requires transformation that is lossy or requires manual review (e.g., reduction, collapsing, recoding, temporal approximation, or external interpretation).

  The harmonized variable should preserve the finest common granularity both sides support — do not collapse further than necessary. If both sides share identical categories, remap the codes 1:1 rather than collapsing to binary.

  Use PARTIAL only when ALL of these hold:
  1. Both variables describe the same clinical entity (or one is a recognized member/subtype of the other's class).
  2. A concrete harmonized variable can be named that represents a single clinical concept both variables independently measure. A composite union of two different entities ("A or B", "A and/or B") is not a valid harmonized variable.
  3. At least one side can fully populate the harmonized variable. The other side may have values that map to missing/unknown if the relationship is uninformative in one direction (e.g., member-drug "no" does not inform class-level use). However, no value on either side may map to a fabricated value. "Fabricating" includes: (a) assigning a quantitative value where none exists — e.g., mapping "yes" to a dose number; (b) inferring unmeasured patient-level facts — e.g., assuming a patient underwent a procedure because their cause of death is clinically related.
  4. The harmonized variable is directly supported by the provided metadata — not a new concept invented by merging two different entities.

  Sub-patterns:
  (a) Granularity reduction — one side is finer-grained; collapse to the coarser level.
  (b) External-reference alignment — the two variables represent the same (or class-related) clinical entity in different scales or units. Conversion is computable using external clinical knowledge such as target-dose tables, drug-equivalence tables, or standard conversion factors. This includes combinations: a member-drug dose in one scale (e.g., mg) vs a class-level dose in another scale (e.g., % of target) is PARTIAL, not IMPOSSIBLE, because both the class membership and the scale conversion are resolvable with external references.
  (c) Asymmetric member/class — a specific-drug indicator vs a class-level variable. The harmonized variable is at the class level. The specific-drug "yes" maps to class "yes"; the specific-drug "no" maps to missing (not class "no"). The class-level side fully populates the harmonized variable.

  Examples:
  - smoking status (never/former/current) vs smoker yes/no -> harmonized: smoker_yes_no
  - diagnosis date vs diagnosis present yes/no -> harmonized: diagnosis_present_yes_no
  - furosemide dose (mg) vs furosemide dose (% of target) -> harmonized: furosemide_dose_pct_target
  - metoprolol dose (mg) vs beta-blocker dose (% of target) -> harmonized: beta_blocker_dose_pct_target
  - furosemide use yes/no vs loop diuretic dose (mg) -> harmonized: loop_diuretic_use_yes_no

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
  - hemoglobin concentration vs MCHC (different lab analytes despite shared terminology)
  - LVEF value (%) vs LVEF measurement method (different dimensions of same entity)
  - all-cause death (yes/no) vs HF hospitalization or death (yes/no) (composite cannot be decomposed from component)
  - cause of death (categories) vs death following a procedure yes/no (cause does not encode procedural history)

# IMPORTANT RULES
- Harmonization can only reduce information, never invent it. Fabrication includes both numeric invention (mapping "yes" to a specific dose) and inferential invention (assuming a patient underwent a procedure because their diagnosis is related). If a mapping requires clinical inference beyond what the variable's values encode, it is IMPOSSIBLE. Mapping a value to missing/unknown is acceptable information loss, not fabrication.
- Causal or logical relationships between two variables do not make them the same clinical entity. A device and a physiological finding it may produce, a risk factor and an outcome it may cause, or a treatment and a condition it targets are different entities even if clinically associated. Harmonization requires the variables to measure the same thing (or a class/member version of the same thing).
- For disease yes/no variables, detection method or setting (ECG, echo, wearable, screening, hospital record, history) does not by itself change the diagnosis concept. Distinguish the clinical entity being measured from the measurement context (diagnostic modality, body position, procedure setting). Different context for the SAME entity is not automatically IMPOSSIBLE — evaluate whether a valid common variable still exists. If both sides measure the same entity at the same granularity (e.g., both binary yes/no, both continuous in same units) and differ only in measurement context or value encoding, this is COMPATIBLE — not PARTIAL. In data harmonization, each study's recorded value is taken at face value — do not reason about diagnostic sensitivity differences between modalities (e.g., "ECG has lower sensitivity for LVH than echo"). Both studies answered the same clinical question using their chosen method; their "no" means "not detected" in that study, which is valid for pooling. PARTIAL requires actual information loss through reduction or collapsing.
-   

- Unit or scale differences (mg vs %, counts/min vs /min, kg vs lb) are about data representation, not about whether two variables measure the same entity. Do not treat unit mismatch alone as evidence of different clinical entities.
- Recognize mathematically equivalent units: ug/L = ng/mL, {{counts}}/min = /min = bpm, mg/dL = mg/100mL. These are COMPLETE, not COMPATIBLE.
- The first concept is authoritative for variable meaning. Treat additional concepts as refinements only if they materially change clinical interpretation; otherwise treat them as annotation context.
- When graph evidence conflicts with descriptions or concepts, prefer the explicit metadata. Use graph evidence to break ties or confirm relationships, not to override stated meaning.
- For class/member relationships (not measurement-context differences): the specific-drug "yes" always maps to the class-level "yes". The specific-drug "no" maps to missing (not class "no"), because absence of one member does not exclude other members. If the class-level side provides full data, this is PARTIAL(c). Do not apply this asymmetric logic to measurement-context differences — those follow the measurement-context rule above.
- Source/Target order must not change the status. The same pair in reverse should receive the same verdict. Only the alignment_direction may differ based on which side is finer-grained.
- Use only the provided metadata. Do not invent units, categories, or details that are not stated.
- When in doubt between two statuses, prefer the more conservative one.
- Before returning IMPOSSIBLE: verify that no valid common harmonized variable — representing a single clinical concept, not a composite — can be named. If you can name one that satisfies the PARTIAL criteria above, return PARTIAL instead.

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
- PARTIAL: never "bidirectional"; direction from finer-grained to coarser, or from the side that can fully populate to the harmonized variable. If neither side can fully populate, return IMPOSSIBLE.
- IMPOSSIBLE: ""

# OUTPUT CONSTRAINTS
- status: COMPLETE | COMPATIBLE | PARTIAL | IMPOSSIBLE
- confidence: float 0.0-1.0
- reason: 25 words or fewer
- transform: 40 words or fewer, or ""
- harmonized_variable: 12 words or fewer, snake_case, or ""
- alignment_direction: "bidirectional" | "source to target" | "target to source" | ""

# OUTPUT FORMAT
Return ONLY one valid JSON object:
{{
  "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
  "confidence": <float>,
  "reason": "<25 words or fewer>",
  "transform": "<40 words or fewer, or empty>",
  "harmonized_variable": "<snake_case or empty>",
  "alignment_direction": "<direction or empty>"
}}
"""
 
_PREAMBLE = """You are a clinical data harmonization expert assessing whether two variables can be aligned and merged into a common harmonized analysis variable for pooled statistical analysis.
"""
 
SYSTEM_PROMPT_NE = (
    _PREAMBLE
    + _INPUT_NE
    + _SHARED_BODY.format(extra_rules="")
)
 
SYSTEM_PROMPT_EV = (
    _PREAMBLE
    + _INPUT_EV
    + _SHARED_BODY.format(extra_rules=_RULES_EV_ONLY)
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
        return (True, ContextMatchType.EXACT.value, max(conf, 0.9), reason_json)

    if status == "COMPATIBLE":
        return (True, ContextMatchType.COMPATIBLE.value, max(min(conf, 0.9), 0.8), reason_json)

    if status == "PARTIAL":
        return (True, ContextMatchType.PARTIAL.value, max(min(conf, 0.8), 0.75), reason_json)

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

   

     # Deterministic generation parameters
    max_tokens: int = 1000
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled/default for OpenRouter; omit for Gemini if 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    top_a: float = 0.0

    timeout: int = 2000
    mode: str = MappingType.OEH.value
    _clients: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._cache = LLMDiskCache()
        sys_prompt = SYSTEM_PROMPT_NE if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV

        self._cache.set_system_prompt(sys_prompt)
        model = self.models[0]
        backend = self._backend_for(model)
        if backend == "ollama":
             
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
        elif backend == "google":
            from google.genai import Client
            self._clients["google"] = Client(api_key=settings.GEMINI_API_KEY)
        elif backend == "openrouter":
            from openrouter import OpenRouter
            self._clients["openrouter"] = OpenRouter(api_key=settings.OPENROUTER_API_KEY)

    @staticmethod
    def _backend_for(model: str) -> str:
        m = model.lower()
        if "ollama/" in m: return "ollama"
        if "openrouter/" in m: return "openrouter"
        # if m.startswith("gpt-"): return "openai"
        # if m.startswith("claude-"): return "anthropic"
        if m.startswith("gemini-"): return "google"
        return "openrouter"

    def _openrouter_generation_params(self) -> Dict[str, Any]:
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
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
                thinking_level="LOW"
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
        output_token_limit = self.max_tokens

        backend = self._backend_for(model)
        api_model = model.replace("openrouter/", "") if backend == "openrouter" else model
        mname = model.split('/')[-1]
        client = self._clients[backend]
        sys_prompt = SYSTEM_PROMPT_NE if self.mode == MappingType.NE.value else SYSTEM_PROMPT_EV
        try:
            if backend == "ollama":
                resp = client.chat(
                    model=api_model.replace("ollama/", ""),
                    messages=[{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}],
                    options={"temperature": self.temperature, "num_predict": output_token_limit},
                )
                result = resp["message"]["content"] or ""
                print(f"[ollama] got {len(result)} chars")

           

            elif backend == "openrouter":
                kwargs = {
                    "model": api_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "reasoning": {
                        "effort": "low"
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
         
            elif backend == "google":
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
                    prompts.append(
                        _build_pair_prompt(
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
                    )

            results = [None] * len(prompts)

            # Initially all prompts are active.
            active_indices = list(range(len(prompts)))

            total_calls = 0
            pending_rounds_used = 0

            for round_idx in range(max_pending_rounds + 1):
                if not active_indices:
                    break

                pending_rounds_used = round_idx

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

                # Re-add only pending items to the next round.
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

  