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
        key_material = f"{self._sys_hash}::{mode}::{prompt}"
        return d / f"{hashlib.sha256(key_material.encode()).hexdigest()}.json"

    def get(self, model: str, prompt: str, mode: str = "", batch: bool = False) -> str | None:
        p = self._path(model, prompt, mode, batch)
        return json.loads(p.read_text())["r"] if p.exists() else None

    def put(self, model: str, prompt: str, response: str, mode: str = "", batch: bool = False):
        self._path(model, prompt, mode, batch).write_text(
            json.dumps({"prompt": prompt, "r": response}))

    def delete(self, model: str, prompt: str, mode: str = "", batch: bool = False):
        self._path(model, prompt, mode, batch).unlink(missing_ok=True)

# SYSTEM_PROMPT_NE = """
# You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# # INPUT
# Two variables are provided: Source and Target.

# Each variable has:
# - description: variable label from study metadata
# - unit: measurement unit, if available
# - categories: allowed values; [] means continuous or free-text

# # TASK
# Using only the provided metadata, assign exactly ONE of the following statuses.


# - COMPLETE:
#   Same clinical meaning or safely alignable to a broader clinically acceptable class with no data transformation needed.
#   Examples: sitting systolic blood pressure mmHg vs systolic blood pressure mmHg, furosemide(mg) vs loop diuretic (mg), myocardia infraction vs actue myocardia infraction etc.


# - COMPATIBLE:
#   Same clinical meaning, or safely alignable to a broader clinically acceptable class through possible or deterministic data transformation.
#   Examples: weight (kg) vs weight (lb), Blood glucose (mmol/L) vs Blood glucose(mg/dL), Gender: Female/Male vs Gender: 0/1, furosemide (%) vs loop diuretic (mg)

# - PARTIAL:
#   Both variables describe the same clinical entity but at different levels of granularity. A common (coarser) variable can be derived by reducing the finer-grained side to match the broader side — typically through category collapse, dichotomization, or aggregation.
#   The transformation is lossy and one-directional (specific → general only); the coarser side cannot recover the finer side.  Examples:
#   - multi-category smoking status vs smoker yes/no
#   - specific beta-blocker taken vs beta-blocker taken yes/no
#   - diagnosis date vs diagnosis yes/no
#   - heart failure hospitalization or death vs all-cause death

# - IMPOSSIBLE:
#   This includes variables that share a broad therapeutic area or clinical domain but measure DIFFERENT entities.
#   Examples:
#   - sitting systolic blood pressure vs standing systolic blood pressure (different)
#   - diabetes medication use vs diabetes diagnosis
#   - ACE inhibitor vs ARBs inhibitor 


# # IMPORTANT RULES
# - Judge based on clinical context and relevance, not surface word overlap.
# - Base the decision only on the provided metadata. Do not invent missing details, and do not over-classify alignment unless it is supported by the available evidence.
# - A transformation is considered safe only if it is a pure representation normalization and does not change granularity or meaning.

# # CONFIDENCE
# Confidence is your confidence that the assigned status is correct. Use these ranges:
# - COMPLETE: 1.0
# - COMPATIBLE: 0.85 to 0.95
# - PARTIAL: 0.75 to 0.85
# - IMPOSSIBLE: 0.0

# # TRANSFORM
# - COMPLETE: ""
# - IMPOSSIBLE: ""
# - COMPATIBLE: exact normalization
# - PARTIAL: required transformation or main limitation

# # TRANSFORM DIRECTION
# Use one of:
# - "source to target"
# - "target to source"
# - "bidirectional"
# - ""

# Use "bidirectional" only when both sides can be aligned without loss of meaning.
# Use "" for COMPLETE and IMPOSSIBLE.

# # OUTPUT CONSTRAINTS
# - status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
# - confidence must be a number between 0.0 and 1.0
# - reason must be 25 words or fewer
# - transform must be 35 words or fewer, or ""
# - transform_direction must be one of:
#   "source to target", "target to source", "bidirectional", ""

# # OUTPUT FORMAT
# Return ONLY one valid JSON object with this structure:

# {
#   "status": "COMPLETE",
#   "confidence": 1.0,
#   "reason": "Same variable meaning and representation.",
#   "transform": "",
#   "transform_direction": ""
# }
# """

# SYSTEM_PROMPT_EV = """
# You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# # INPUT
# Two variables are provided: Source and Target.

# Each variable may include:
# - Description: short variable label
# - Concepts: pipe-separated ordered concepts
#   - first concept = primary concept of the variable
#   - remaining concepts = additional concepts that may refine the meaning
# - Categories: allowed values, if available
# - Unit: measurement unit, if available
# - Graph evidence: optional semantic evidence from OMOP vocabularies

# # TASK
# Using only the provided metadata, assign exactly ONE of the following statuses.


# - COMPLETE:
#   Same clinical meaning or safely alignable to a broader clinically acceptable class with no data transformation needed.
#   Examples: sitting systolic blood pressure mmHg vs systolic blood pressure mmHg, furosemide(mg) vs loop diuretic (mg), myocardia infraction vs actue myocardia infraction etc.

# - COMPATIBLE:
#   Same clinical meaning, or safely alignable to a broader clinically acceptable class through possible or deterministic data transformation.
#   Examples: weight (kg) vs weight (lb), Blood glucose (mmol/L) vs Blood glucose(mg/dL), Gender: Female/Male vs Gender: 0/1, furosemide (%) vs loop diuretic (mg)

# - PARTIAL:
#   Both variables describe the same clinical entity but at different levels of granularity. A common (coarser) variable can be derived by reducing the finer-grained side to match the broader side — typically through category collapse, dichotomization, or aggregation.
#   The transformation is lossy and one-directional (specific → general only); the coarser side cannot recover the finer side.
#   Examples:
#   - multi-category smoking status vs smoker yes/no
#   - specific beta-blockers taken vs beta-blocker taken yes/no
#   - diagnosis date vs diagnosis yes/no
#   - heart failure hospitalization or death vs all-cause death

# - IMPOSSIBLE:
#   This includes variables that share a broad therapeutic area or clinical domain but measure DIFFERENT entities.
#   Examples:
#   - sitting systolic blood pressure vs standing systolic blood pressure (different)
#   - diabetes medication use vs diabetes diagnosis
#   - ACE inhibitor vs ARBs inhibitor 


# # IMPORTANT RULES
# - Judge based on clinical context and relevance, not surface word overlap.
# - Base the decision only on the provided metadata. Do not invent missing details, and do not over-classify alignment unless it is supported by the available evidence.
# - Treat an additional concept as important only if it changes the actual variable meaning; otherwise ignore it as annotation noise or redundant wording
# - Use graph evidence only as supporting evidence for semantic clarity.

# # CONFIDENCE
# Confidence means confidence that the assigned status is correct.
# - COMPLETE: 1.0
# - COMPATIBLE: 0.85 to 0.95
# - PARTIAL: 0.75 to 0.85
# - IMPOSSIBLE: 0.0

# # TRANSFORM
# - COMPLETE: ""
# - IMPOSSIBLE: ""
# - COMPATIBLE: exact normalization
# - PARTIAL: required transformation or main limitation

# # TRANSFORM DIRECTION
# Use one of:
# - "source to target"
# - "target to source"
# - "bidirectional"
# - ""

# Use "bidirectional" only when both sides can be aligned without loss of meaning.
# Use "" for COMPLETE and IMPOSSIBLE.

# # OUTPUT CONSTRAINTS
# - status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
# - confidence must be a number between 0.0 and 1.0
# - reason must be 25 words or fewer
# - transform must be 35 words or fewer, or ""
# - transform_direction must be one of:
#   "source to target", "target to source", "bidirectional", ""

# # OUTPUT FORMAT
# Return ONLY one valid JSON object with exactly these keys:
# {
#   "status": "COMPLETE",
#   "confidence": 1.0,
#   "reason": "Same variable meaning and representation.",
#   "transform": "",
#   "transform_direction": ""
# }
# """

# SYSTEM_PROMPT_NE = """
# You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# # INPUT
# Two variables are provided: Source and Target. The Source/Target labels are positional only — they do not imply which side is finer or coarser, nor which side is the reference.

# Each variable has:
# - description: variable label from study metadata
# - unit: measurement unit, if available
# - categories: allowed values; [] means continuous or free-text

# # TASK
# Using only the provided metadata, assign exactly ONE of the following statuses.

# - COMPLETE:
#   Identical clinical meaning AND identical representation (same units, same coding, same granularity). Values from one side can be used as-is for the other with no transformation.
#   Examples:
#   - systolic BP (mmHg) vs sitting systolic BP (mmHg)
#   - acute myocardial infarction vs myocardial infarction (when MI is the analysis target)
#   - HbA1c (%) vs HbA1c (%)

# - COMPATIBLE:
#   Same clinical meaning and same granularity, but different representation. Values can be losslessly converted via a deterministic transformation (unit conversion, recoding, rescaling).
#   Examples:
#   - weight (kg) vs weight (lb)
#   - blood glucose (mmol/L) vs blood glucose (mg/dL)
#   - Gender Female/Male vs Gender 0/1
#   - furosemide (mg) vs furosemide (%)

# - PARTIAL:
#   Both variables describe the same clinical entity but at different levels of granularity. A common (coarser) variable can be derived by reducing the finer-grained side to match the broader side — typically through category collapse, dichotomization, or aggregation. The transformation is lossy and one-directional (specific → general only); the coarser side cannot recover the finer side.
#   Examples:
#   - multi-category smoking status vs smoker yes/no → collapse to yes/no
#   - specific beta-blocker taken vs beta-blocker taken yes/no → dichotomize to any-use
#   - diagnosis date vs diagnosis yes/no → derive presence from date
#   - heart failure hospitalization or death vs all-cause death → reduce composite to component
#   - furosemide (mg) vs loop diuretic (any-use) → collapse specific drug to drug class indicator

# - IMPOSSIBLE:
#   The two variables measure different clinical entities, even if they share a therapeutic area, organ system, or surface vocabulary. No transformation can align them without fabricating data. Also use this when the variables have no clinical relationship at all.
#   Examples:
#   - sitting SBP vs standing SBP (different physiological states)
#   - diabetes medication use vs diabetes diagnosis (treatment ≠ condition)
#   - ACE inhibitor vs ARB (different drug classes)
#   - hemoglobin vs zip code (no clinical relationship)

# # IMPORTANT RULES
# - Judge based on clinical meaning, not surface word overlap.
# - Use only the provided metadata. Do not invent units, categories, or details that are not stated.
# - A transformation is considered safe only if it is a pure representation normalization and does not change granularity or meaning. If granularity changes, the status is PARTIAL, not COMPATIBLE.
# - The Source/Target labels are positional only — determine granularity and direction from the metadata itself, not from which side is labeled "Source."
# - When in doubt between two statuses, prefer the more conservative one: COMPATIBLE over COMPLETE; PARTIAL over COMPATIBLE; IMPOSSIBLE over PARTIAL.

# # CONFIDENCE
# Confidence reflects genuine uncertainty in the status assignment based on metadata completeness and ambiguity:
# - 0.95–1.00: metadata is unambiguous; status is clear
# - 0.80–0.94: minor ambiguity (e.g., unit missing but inferable from description)
# - 0.65–0.79: meaningful ambiguity (e.g., descriptions conflict, categories unclear)
# - below 0.65: do not assign a positive alignment — return IMPOSSIBLE instead

# For IMPOSSIBLE:
# - 0.0 when the variables have no clinical relationship
# - 0.70–0.95 when they share a domain but measure different entities

# # TRANSFORM
# Describes the data operation needed to align the two variables. Limitations and caveats belong in `reason`, not here.
# - COMPLETE: ""  (no transformation)
# - COMPATIBLE: the deterministic conversion (e.g., "kg = lb × 0.4536", "recode {0→Male, 1→Female}", "mmol/L × 18.0182 = mg/dL for glucose")
# - PARTIAL: the lossy reduction (e.g., "collapse smoking categories to ever/never", "dichotomize specific drug to any-use yes/no", "derive presence indicator from diagnosis date")
# - IMPOSSIBLE: ""

# # TRANSFORM DIRECTION
# Use one of:
# - "source to target"
# - "target to source"
# - "bidirectional"
# - ""

# Rules:
# - COMPLETE and IMPOSSIBLE: always "".
# - COMPATIBLE: typically "bidirectional" since deterministic conversions are reversible. Use a single direction only if the metadata indicates one side cannot be reconstructed (e.g., precision loss in rounding).
# - PARTIAL: never "bidirectional". The lossy reduction flows only from the finer-grained side to the coarser side. Use "source to target" if the source is finer; "target to source" if the target is finer.

# # OUTPUT CONSTRAINTS
# - status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
# - confidence must be a number between 0.0 and 1.0
# - reason must be 25 words or fewer; briefly justify the status with reference to the specific metadata that drove the decision
# - transform must be 35 words or fewer, or ""
# - transform_direction must be one of: "source to target", "target to source", "bidirectional", ""

# # OUTPUT FORMAT
# Return ONLY one valid JSON object with exactly these keys, in this order:
# {
#   "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
#   "confidence": <float 0.0–1.0>,
#   "reason": "<≤25 words>",
#   "transform": "<≤35 words, or empty string>",
#   "transform_direction": "<source to target|target to source|bidirectional|empty string>"
# }
# """

# SYSTEM_PROMPT_EV = """
# You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# # INPUT
# Two variables are provided: Source and Target. The Source/Target labels are positional only — they do not imply which side is finer or coarser, nor which side is the reference.

# Each variable may include:
# - Description: short variable label
# - Concepts: ordered concepts separated by " | "
#   - first concept = primary concept of the variable (authoritative for meaning)
#   - remaining concepts = refinements that may narrow or qualify the meaning
# - Categories: allowed values, if available
# - Unit: measurement unit, if available
# - Graph evidence: optional semantic evidence from OMOP vocabularies

# # TASK
# Using only the provided metadata, assign exactly ONE of the following statuses.

# - COMPLETE:
#   Identical clinical meaning AND identical representation (same units, same coding, same granularity). Values from one side can be used as-is for the other with no transformation.
#   Examples:
#   - systolic BP (mmHg) vs sitting systolic BP (mmHg)
#   - acute myocardial infarction vs myocardial infarction (when MI is the analysis target)
#   - HbA1c (%) vs HbA1c (%)


# - COMPATIBLE:
#   Same clinical meaning and same granularity, but different representation. Values can be losslessly converted via a deterministic transformation (unit conversion, recoding, rescaling).
#   Examples:
#   - weight (kg) vs weight (lb)
#   - blood glucose (mmol/L) vs blood glucose (mg/dL)
#   - Gender Female/Male vs Gender 0/1
#   - furosemide (mg) vs furosemide (%)

# - PARTIAL:
#   Both variables describe the same clinical entity but at different levels of granularity. A common (coarser) variable can be derived by reducing the finer-grained side to match the broader side — typically through category collapse, dichotomization, or aggregation. The transformation is lossy and one-directional (specific → general only); the coarser side cannot recover the finer side. - 

#   Examples:
#   - multi-category smoking status vs smoker yes/no → collapse to yes/no
#   - specific beta-blocker taken vs beta-blocker taken yes/no → dichotomize to any-use
#   - diagnosis date vs diagnosis yes/no → derive presence from date
#   - heart failure hospitalization or death vs all-cause death → reduce composite to component
#   - furosemide (mg) vs loop diuretic (any-use) → collapse specific drug to drug class indicator

# - IMPOSSIBLE:
#   The two variables measure different clinical entities, even if they share a therapeutic area, organ system, or surface vocabulary. No transformation can align them without fabricating data. Also use this when the variables have no clinical relationship at all.
#   Examples:
#   - sitting SBP vs standing SBP (different physiological states)
#   - diabetes medication use vs diabetes diagnosis (treatment ≠ condition)
#   - ACE inhibitor vs ARB (different drug classes)
#   - hemoglobin vs zip code (no clinical relationship)

# # IMPORTANT RULES
# - Judge based on clinical meaning, not surface word overlap.
# - Use only the provided metadata. Do not invent units, categories, or concepts.
# - Treat the first concept as authoritative for variable meaning; treat additional concepts as refinements only if they materially change clinical interpretation. Otherwise ignore them as annotation noise.
# - The Source/Target labels are positional only — determine granularity and direction from the metadata itself, not from which side is labeled "Source."
# - When graph evidence conflicts with description or concepts, prefer the explicit metadata. Use graph evidence to break ties or confirm semantic relationships, not to override stated meaning.
# - When in doubt between two statuses, prefer the more conservative one: COMPATIBLE over COMPLETE; PARTIAL over COMPATIBLE; IMPOSSIBLE over PARTIAL.

# # CONFIDENCE
# Confidence reflects genuine uncertainty in the status assignment based on metadata completeness and ambiguity:
# - 0.95–1.00: metadata is unambiguous; status is clear
# - 0.80–0.94: minor ambiguity (e.g., unit missing but inferable from description)
# - 0.65–0.79: meaningful ambiguity (e.g., concepts conflict, categories unclear)
# - below 0.65: do not assign a positive alignment — return IMPOSSIBLE instead

# For IMPOSSIBLE:
# - 0.0 when the variables have no clinical relationship
# - 0.70–0.95 when they share a domain but measure different entities

# # TRANSFORM
# Describes the data operation needed to align the two variables. Limitations and caveats belong in `reason`, not here.
# - COMPLETE: ""  (no transformation)
# - COMPATIBLE: the deterministic conversion (e.g., "kg = lb × 0.4536", "recode {0→Male, 1→Female}", "mmol/L × 18.0182 = mg/dL for glucose")
# - PARTIAL: the lossy reduction (e.g., "collapse smoking categories to ever/never", "dichotomize specific drug to any-use yes/no", "derive presence indicator from diagnosis date")
# - IMPOSSIBLE: ""

# # TRANSFORM DIRECTION
# Use one of:
# - "source to target"
# - "target to source"
# - "bidirectional"
# - ""

# Rules:
# - COMPLETE and IMPOSSIBLE: always "".
# - COMPATIBLE: typically "bidirectional" since deterministic conversions are reversible. Use a single direction only if the metadata indicates one side cannot be reconstructed (e.g., precision loss in rounding).
# - PARTIAL: never "bidirectional". The lossy reduction flows only from the finer-grained side to the coarser side. Use "source to target" if the source is finer; "target to source" if the target is finer.

# # OUTPUT CONSTRAINTS
# - status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
# - confidence must be a number between 0.0 and 1.0
# - reason must be 25 words or fewer; briefly justify the status with reference to the specific metadata that drove the decision
# - transform must be 35 words or fewer, or ""
# - transform_direction must be one of: "source to target", "target to source", "bidirectional", ""

# # OUTPUT FORMAT
# Return ONLY one valid JSON object with exactly these keys, in this order:
# {
#   "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
#   "confidence": <float 0.0–1.0>,
#   "reason": "<≤25 words>",
#   "transform": "<≤35 words, or empty string>",
#   "transform_direction": "<source to target|target to source|bidirectional|empty string>"
# }
# """

# SYSTEM_PROMPT_NE = """
# You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# # INPUT
# Two variables are provided: Source and Target. The Source/Target labels are positional only — they do not imply which side is finer or coarser, nor which side is the reference.

# Each variable has:
# - description: variable label from study metadata
# - unit: measurement unit, if available
# - categories: allowed values; [] means continuous or free-text

# # TASK
# Using only the provided metadata, assign exactly ONE of the following statuses.

# - COMPLETE:
#   Identical clinical meaning AND identical representation (same units, same coding, same granularity). Values from one side can be used as-is for the other with no transformation.
#   Examples:
#   - systolic BP (mmHg) vs sitting systolic BP (mmHg)
#   - acute myocardial infarction vs myocardial infarction (when MI is the analysis target)
#   - HbA1c (%) vs HbA1c (%)

# - COMPATIBLE:
#   Same clinical meaning and same granularity, but different representation. Values can be losslessly converted via a deterministic transformation (unit conversion, recoding, rescaling).
#   Examples:
#   - weight (kg) vs weight (lb)
#   - blood glucose (mmol/L) vs blood glucose (mg/dL)
#   - Gender Female/Male vs Gender 0/1
#   - furosemide (mg) vs furosemide (%)

# - PARTIAL:
#   Both variables describe the same clinical entity but at different levels of granularity. A common (coarser) variable can be derived by reducing the finer-grained side to match the broader side — typically through category collapse, dichotomization, or aggregation. The transformation is lossy and one-directional (specific → general only); the coarser side cannot recover the finer side.
#   Examples:
#   - multi-category smoking status vs smoker yes/no → collapse to yes/no
#   - specific beta-blocker taken vs beta-blocker taken yes/no → dichotomize to any-use
#   - diagnosis date vs diagnosis yes/no → derive presence from date
#   - heart failure hospitalization or death vs all-cause death → reduce composite to component
#   - furosemide (mg) vs loop diuretic (any-use) → collapse specific drug to drug class indicator

# - IMPOSSIBLE:
#   The two variables measure different clinical entities, even if they share a therapeutic area, organ system, or surface vocabulary. No transformation can align them without fabricating data. Also use this when the variables have no clinical relationship at all.
#   Examples:
#   - sitting SBP vs standing SBP (different physiological states)
#   - diabetes medication use vs diabetes diagnosis (treatment ≠ condition)
#   - ACE inhibitor vs ARB (different drug classes)
#   - hemoglobin vs zip code (no clinical relationship)

# # IMPORTANT RULES
# - Judge based on clinical meaning, not surface word overlap.
# - Use only the provided metadata. Do not invent units, categories, or details that are not stated.
# - A transformation is considered safe only if it is a pure representation normalization and does not change granularity or meaning. If granularity changes, the status is PARTIAL, not COMPATIBLE.
# - The Source/Target labels are positional only — determine granularity and direction from the metadata itself, not from which side is labeled "Source."
# - When in doubt between two statuses, prefer the more conservative one: COMPATIBLE over COMPLETE; PARTIAL over COMPATIBLE; IMPOSSIBLE over PARTIAL.

# # CONFIDENCE
# Confidence reflects genuine uncertainty in the status assignment based on metadata completeness and ambiguity:
# - 0.95–1.00: metadata is unambiguous; status is clear
# - 0.80–0.94: minor ambiguity (e.g., unit missing but inferable from description)
# - 0.65–0.79: meaningful ambiguity (e.g., descriptions conflict, categories unclear)
# - below 0.65: do not assign a positive alignment — return IMPOSSIBLE instead

# For IMPOSSIBLE:
# - 0.0 when the variables have no clinical relationship
# - 0.70–0.95 when they share a domain but measure different entities

# # TRANSFORM
# Describes the data operation needed to align the two variables. Limitations and caveats belong in `reason`, not here.
# - COMPLETE: ""  (no transformation)
# - COMPATIBLE: the deterministic conversion (e.g., "kg = lb × 0.4536", "recode {0→Male, 1→Female}", "mmol/L × 18.0182 = mg/dL for glucose")
# - PARTIAL: the lossy reduction (e.g., "collapse smoking categories to ever/never", "dichotomize specific drug to any-use yes/no", "derive presence indicator from diagnosis date")
# - IMPOSSIBLE: ""

# # TRANSFORM DIRECTION
# Use one of:
# - "source to target"
# - "target to source"
# - "bidirectional"
# - ""

# Rules:
# - COMPLETE and IMPOSSIBLE: always "".
# - COMPATIBLE: typically "bidirectional" since deterministic conversions are reversible. Use a single direction only if the metadata indicates one side cannot be reconstructed (e.g., precision loss in rounding).
# - PARTIAL: never "bidirectional". The lossy reduction flows only from the finer-grained side to the coarser side. Use "source to target" if the source is finer; "target to source" if the target is finer.

# # OUTPUT CONSTRAINTS
# - status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
# - confidence must be a number between 0.0 and 1.0
# - reason must be 25 words or fewer; briefly justify the status with reference to the specific metadata that drove the decision
# - transform must be 35 words or fewer, or ""
# - transform_direction must be one of: "source to target", "target to source", "bidirectional", ""

# # OUTPUT FORMAT
# Return ONLY one valid JSON object with exactly these keys, in this order:
# {
#   "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
#   "confidence": <float 0.0–1.0>,
#   "reason": "<≤25 words>",
#   "transform": "<≤35 words, or empty string>",
#   "transform_direction": "<source to target|target to source|bidirectional|empty string>"
# }
# """


SYSTEM_PROMPT_NE = """
You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# INPUT
Two variables are provided: Source and Target. The Source/Target labels are positional only — they do not imply which side is finer or coarser, nor which side is the reference.

Each variable has:
- description: variable label from study metadata
- unit: measurement unit, if available
- categories: allowed values; [] means continuous or free-text

# TASK
Using only the provided metadata, assign exactly ONE of the following statuses.

- COMPLETE:
  Identical clinical meaning AND identical representation (same units, same coding, same granularity). Values from one side can be used as-is for the other with no transformation.
  Examples:
  - systolic BP (mmHg) vs sitting systolic BP (mmHg)
  - acute myocardial infarction vs myocardial infarction (when MI is the analysis target)
  - HbA1c (%) vs HbA1c (%)

- COMPATIBLE:
  Same clinical meaning and same granularity, but different representation. Values can be losslessly converted via a deterministic transformation (unit conversion, recoding, rescaling).
  Examples:
  - weight (kg) vs weight (lb)
  - blood glucose (mmol/L) vs blood glucose (mg/dL)
  - Gender Female/Male vs Gender 0/1

- PARTIAL:
  Both variables describe the same clinical entity but at different levels of granularity. Either of the following situations qualifies as PARTIAL and requires manual review.

  (a) Granularity reduction. The two variables describe the same clinical entity
      at different levels of granularity. A coarser variable can be derived from
      the finer side via category collapse, dichotomization, or aggregation.
      Lossy and one-directional (specific → general only); the coarser side
      cannot recover the finer side.
      Examples:
      - multi-category smoking status vs smoker yes/no
      - specific beta-blocker taken vs beta-blocker taken yes/no
      - diagnosis date vs diagnosis yes/no
      - HF hospitalization or death vs all-cause death
      - furosemide (mg) vs loop diuretic (any-use)

  (b) External-reference alignment. The two variables represent the same (or
      class-related) clinical entity in different scales, and conversion is
      computable but requires a external knowledge and may involve clinical approximations.
      Examples:
      - spironolactone (mg) vs MRA class as % of target dose
      - furosemide (mg) vs furosemide (% of target dose)
      - bumetanide (mg) vs furosemide-equivalent (mg)


- IMPOSSIBLE:
  The two variables measure different clinical entities, even if they share a therapeutic area, organ system, or surface vocabulary. No transformation can align them without fabricating data. Also use this when the variables have no clinical relationship at all.
  Examples:
  - sitting SBP vs standing SBP (different physiological states)
  - diabetes medication use vs diabetes diagnosis (treatment ≠ condition)
  - ACE inhibitor vs ARB (different drug classes)
  - hemoglobin vs zip code (no clinical relationship)

# IMPORTANT RULES
- Judge based on clinical meaning, not surface word overlap.
- Use only the provided metadata. Do not invent units, categories, or details that are not stated.
- A transformation is considered safe only if it is a pure representation normalization and does not change granularity or meaning. If granularity changes, the status is PARTIAL, not COMPATIBLE.
- The Source/Target labels are positional only — determine granularity and direction from the metadata itself, not from which side is labeled "Source."
- When in doubt between two statuses, prefer the more conservative one: COMPATIBLE over COMPLETE; PARTIAL over COMPATIBLE; IMPOSSIBLE over PARTIAL.

# CONFIDENCE
Confidence reflects genuine uncertainty in the status assignment based on metadata completeness and ambiguity:
- 0.95–1.00: metadata is unambiguous; status is clear
- 0.80–0.94: minor ambiguity (e.g., unit missing but inferable from description)
- 0.65–0.79: meaningful ambiguity (e.g., descriptions conflict, categories unclear)
- below 0.65: do not assign a positive alignment — return IMPOSSIBLE instead

For IMPOSSIBLE:
- 0.0 when the variables have no clinical relationship
- 0.70–0.95 when they share a domain but measure different entities

# TRANSFORM
Describes the data operation needed to align the two variables. Limitations and caveats belong in `reason`, not here.
- COMPLETE: ""  (no transformation)
- COMPATIBLE: the deterministic conversion (e.g., "kg = lb × 0.4536", "recode {0→Male, 1→Female}", "mmol/L × 18.0182 = mg/dL for glucose")
- PARTIAL: the lossy reduction (e.g., "collapse smoking categories to ever/never", "dichotomize specific drug to any-use yes/no", "derive presence indicator from diagnosis date")
- IMPOSSIBLE: ""

# TRANSFORM DIRECTION
Use one of:
- "source to target"
- "target to source"
- "bidirectional"
- ""

Rules:
- COMPLETE and IMPOSSIBLE: always "".
- COMPATIBLE: typically "bidirectional" since deterministic conversions are reversible. Use a single direction only if the metadata indicates one side cannot be reconstructed (e.g., precision loss in rounding).
- PARTIAL: never "bidirectional". The lossy reduction flows only from the finer-grained side to the coarser side. Use "source to target" if the source is finer; "target to source" if the target is finer.

# OUTPUT CONSTRAINTS
- status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
- confidence must be a number between 0.0 and 1.0
- reason must be 25 words or fewer; briefly justify the status with reference to the specific metadata that drove the decision
- transform must be 35 words or fewer, or ""
- transform_direction must be one of: "source to target", "target to source", "bidirectional", ""

# OUTPUT FORMAT
Return ONLY one valid JSON object with exactly these keys, in this order:
{
  "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
  "confidence": <float 0.0–1.0>,
  "reason": "<≤25 words>",
  "transform": "<≤35 words, or empty string>",
  "transform_direction": "<source to target|target to source|bidirectional|empty string>"
}
"""


SYSTEM_PROMPT_EV = """
You are a clinical data harmonization expert assessing whether two variables have the same clinical meaning and whether they can be aligned for pooled statistical analysis.

# INPUT
Two variables are provided: Source and Target. The Source/Target labels are positional only — they do not imply which side is finer or coarser, nor which side is the reference.

Each variable may include:
- Description: short variable label
- Concepts: ordered concepts separated by " | "
  - first concept = primary concept of the variable (authoritative for meaning)
  - remaining concepts = refinements that may narrow or qualify the meaning
- Categories: allowed values, if available
- Unit: measurement unit, if available
- Graph evidence: optional semantic evidence from OMOP vocabularies

# TASK
Using only the provided metadata, assign exactly ONE of the following statuses.

- COMPLETE:
  Identical clinical meaning AND identical representation (same units, same coding, same granularity). Values from one side can be used as-is for the other with no transformation.
  Examples:
  - systolic BP (mmHg) vs sitting systolic BP (mmHg)
  - acute myocardial infarction vs myocardial infarction (when MI is the analysis target)
  - HbA1c (%) vs HbA1c (%)


- COMPATIBLE:
  Same clinical meaning and same granularity, but different representation. Values can be losslessly converted via a deterministic transformation (unit conversion, recoding, rescaling).
  Examples:
  - weight (kg) vs weight (lb)
  - blood glucose (mmol/L) vs blood glucose (mg/dL)
  - Gender Female/Male vs Gender 0/1


- PARTIAL:
  Both variables describe the same clinical entity but at different levels of granularity. Either of the following situations qualifies as PARTIAL and requires manual review.

  (a) Granularity reduction. The two variables describe the same clinical entity
      at different levels of granularity. A coarser variable can be derived from
      the finer side via category collapse, dichotomization, or aggregation.
      Lossy and one-directional (specific → general only); the coarser side
      cannot recover the finer side.
      Examples:
      - multi-category smoking status vs smoker yes/no
      - specific beta-blocker taken vs beta-blocker taken yes/no
      - diagnosis date vs diagnosis yes/no
      - HF hospitalization or death vs all-cause death
      - furosemide (mg) vs loop diuretic (any-use)

  (b) External-reference alignment. The two variables represent the same (or
      class-related) clinical entity in different scales, and conversion is
      computable but requires a external knowledge and may involve clinical approximations.
      Examples:
      - spironolactone (mg) vs MRA class as % of target dose
      - furosemide (mg) vs furosemide (% of target dose)
      - bumetanide (mg) vs furosemide-equivalent (mg)

- IMPOSSIBLE:
  The two variables measure different clinical entities, even if they share a therapeutic area, organ system, or surface vocabulary. No transformation can align them without fabricating data. Also use this when the variables have no clinical relationship at all.
  Examples:
  - sitting SBP vs standing SBP (different physiological states)
  - diabetes medication use vs diabetes diagnosis (treatment ≠ condition)
  - ACE inhibitor vs ARB (different drug classes)
  - hemoglobin vs zip code (no clinical relationship)

# IMPORTANT RULES
- Judge based on clinical meaning, not surface word overlap.
- Use only the provided metadata. Do not invent units, categories, or concepts.
- Treat the first concept as authoritative for variable meaning; treat additional concepts as refinements only if they materially change clinical interpretation. Otherwise ignore them as annotation noise.
- The Source/Target labels are positional only — determine granularity and direction from the metadata itself, not from which side is labeled "Source."
- When graph evidence conflicts with description or concepts, prefer the explicit metadata. Use graph evidence to break ties or confirm semantic relationships, not to override stated meaning.
- When in doubt between two statuses, prefer the more conservative one: COMPATIBLE over COMPLETE; PARTIAL over COMPATIBLE; IMPOSSIBLE over PARTIAL.

# CONFIDENCE
Confidence reflects genuine uncertainty in the status assignment based on metadata completeness and ambiguity:
- 0.95–1.00: metadata is unambiguous; status is clear
- 0.80–0.94: minor ambiguity (e.g., unit missing but inferable from description)
- 0.65–0.79: meaningful ambiguity (e.g., concepts conflict, categories unclear)
- below 0.65: do not assign a positive alignment — return IMPOSSIBLE instead

For IMPOSSIBLE:
- 0.0 when the variables have no clinical relationship
- 0.70–0.95 when they share a domain but measure different entities

# TRANSFORM
Describes the data operation needed to align the two variables. Limitations and caveats belong in `reason`, not here.
- COMPLETE: ""  (no transformation)
- COMPATIBLE: the deterministic conversion (e.g., "kg = lb × 0.4536", "recode {0→Male, 1→Female}", "mmol/L × 18.0182 = mg/dL for glucose")
- PARTIAL: the lossy reduction (e.g., "collapse smoking categories to ever/never", "dichotomize specific drug to any-use yes/no", "derive presence indicator from diagnosis date")
- IMPOSSIBLE: ""

# TRANSFORM DIRECTION
Use one of:
- "source to target"
- "target to source"
- "bidirectional"
- ""

Rules:
- COMPLETE and IMPOSSIBLE: always "".
- COMPATIBLE: typically "bidirectional" since deterministic conversions are reversible. Use a single direction only if the metadata indicates one side cannot be reconstructed (e.g., precision loss in rounding).
- PARTIAL: never "bidirectional". The lossy reduction flows only from the finer-grained side to the coarser side. Use "source to target" if the source is finer; "target to source" if the target is finer.

# OUTPUT CONSTRAINTS
- status must be one of: COMPLETE, COMPATIBLE, PARTIAL, IMPOSSIBLE
- confidence must be a number between 0.0 and 1.0
- reason must be 25 words or fewer; briefly justify the status with reference to the specific metadata that drove the decision
- transform must be 35 words or fewer, or ""
- transform_direction must be one of: "source to target", "target to source", "bidirectional", ""

# OUTPUT FORMAT
Return ONLY one valid JSON object with exactly these keys, in this order:
{
  "status": "<COMPLETE|COMPATIBLE|PARTIAL|IMPOSSIBLE>",
  "confidence": <float 0.0–1.0>,
  "reason": "<≤25 words>",
  "transform": "<≤35 words, or empty string>",
  "transform_direction": "<source to target|target to source|bidirectional|empty string>"
}
"""

def _truncate_cats(cats: str, max_items: int = 10) -> str:
    if not cats:
        return ""
    items = [c.strip() for c in cats.split("|") if c.strip()]
    if len(items) <= max_items:
        return " | ".join(items)
    return " | ".join(items[:max_items]) + f" | ... ({len(items)} total)"


# def _build_batch_prompt(src_desc: str, src_concepts: str, src_cats: str,
#                         src_unit: str, targets: List[Dict],
#                         mode: str = MappingType.OEH.value) -> str:
#     is_ne = mode == MappingType.NE.value
#     src = f"Source: description: {src_desc}"
#     if not is_ne and src_concepts:
#         src += f", concepts: {src_concepts}"
#     if src_unit:
#         src += f", unit: {src_unit}"
#     src += f", categories: [{_truncate_cats(src_cats)}]"

#     tgts = []
#     for i, t in enumerate(targets):
#         line = f"Target[{i}]: description: {t.get('desc', '')}"
#         if not is_ne and t.get('tgt_concepts'):
#             line += f", concepts: {t['tgt_concepts']}"
#         if t.get('tgt_unit'):
#             line += f", unit: {t['tgt_unit']}"
#         line += f", categories: [{_truncate_cats(t.get('tgt_cats', ''))}]"
#         tgts.append(line)

#     return f"## INPUT\n{src}\n" + "\n".join(tgts)


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

def _parse_single(text: str) -> Tuple[Optional[bool], str, float, str]:
    if not text:
        return (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "empty_response")
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        return (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "think_only_response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Extract fields directly via regex — works on truncated JSON
    status_m = re.search(r'"status"\s*:\s*"([^"]+)"', text)
    conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    reason_m = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"?', text)
    transform_m = re.search(r'"transform"\s*:\s*"((?:[^"\\]|\\.)*)"?', text)
    transform = transform_m.group(1) if transform_m else ""
    transform_direction_m = re.search(r'"transform_direction"\s*:\s*"([^"]+)"', text)
    
    if status_m:

        transform_direction = transform_direction_m.group(1) if transform_direction_m else "none"
        status = status_m.group(1).upper()
        conf = float(conf_m.group(1)) if conf_m else 0.8
        reason = reason_m.group(1) if reason_m else "truncated"
        reason = json.dumps({"status": status,"reason": reason, "transform": transform, "transform_direction": transform_direction})
        # print(f"status: {status}, conf: {conf}, reason: {reason}, transform: {transform}")
        if status.startswith("IMPOSSIBLE"):
            return (False, ContextMatchType.NOT_APPLICABLE.value, 0.0, reason)
        if status.startswith("COMPLETE"):
            return (True,ContextMatchType.EXACT.value,  max(conf, 0.9), reason)
        if status.startswith("COMPATIBLE"):
            return (True, ContextMatchType.COMPATIBLE.value , max(min(conf, 0.9), 0.8), reason)
        if status.startswith("PARTIAL"):
            return (True, ContextMatchType.PARTIAL.value, max(min(conf, 0.8), 0.75), reason)
    return (None, ContextMatchType.NOT_APPLICABLE.value, 0.0, "json_parse_fail")

@dataclass
class LLMConceptMatcher:
    models: List[str] = field(default_factory=list)
    max_retries: int = 5
    temperature: float = 0
    timeout: int = 1000
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
        elif backend == "together":
            from together import Together
            self._clients["together"] = Together(api_key=settings.TOGETHER_API_KEY, timeout=self.timeout)
        elif backend == "openai":
            from openai import OpenAI
            self._clients["openai"] = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=self.timeout)
        elif backend == "anthropic":
            from anthropic import Anthropic
            self._clients["anthropic"] = Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=self.timeout)
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
        if m.startswith("gpt-"): return "openai"
        if m.startswith("claude-"): return "anthropic"
        if m.startswith("gemini-"): return "google"
        return "together"

    def _call_one(self, model: str, prompt: str, force_freeform: bool = False) -> str:
        token_limit = 2000
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
                    options={"temperature": self.temperature, "num_predict": token_limit},
                )
                result = resp["message"]["content"] or ""
                print(f"[ollama] got {len(result)} chars")

            elif backend == "together":
                print(f"api_model: {api_model}")
                resp = client.chat.completions.create(
                    model=api_model, temperature=self.temperature, max_tokens=token_limit,
                    messages=[{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}],
                )
                result = resp.choices[0].message.content or ""
            elif backend == "openrouter":
                resp = client.chat.send(
                    model=api_model, temperature=self.temperature, max_tokens=token_limit,
                    stream=False,
                    messages=[{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}],
                )
                result = resp.choices[0].message.content if resp.choices else ""
            elif backend == "openai":
                msgs = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}]
                kwargs = dict(model=api_model, messages=msgs)
                kwargs["max_completion_tokens"] = token_limit
                kwargs["temperature"] = self.temperature
                kwargs["reasoning_effort"] = "low"
                if not force_freeform:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                result = resp.choices[0].message.content or ""

            elif backend == "anthropic":
                resp = client.messages.create(
                    model=api_model, max_tokens=token_limit, temperature=self.temperature,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = resp.content[0].text if resp.content else ""
            elif backend == "google":
                from google.genai import types
                resp = client.models.generate_content(
                    model=api_model, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt, temperature=self.temperature,
                        max_output_tokens=token_limit, response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                )
                result = resp.text or ""
            else:
                result = ""
        except Exception as e:
            print(f"[{mname}] call failed: {e}")
            return ""
        return result

    def _eval_pair(self, model: str, prompt: str) -> Tuple[Optional[bool], str, float, str]:
        cached = self._cache.get(model, prompt, mode=self.mode)
        if cached:
            v = _parse_single(cached)
            if v[0] is not None: 
                return v
        for attempt in range(self.max_retries):
            text = self._call_one(model, prompt, force_freeform=(attempt > 0))
            v = _parse_single(text)
            if v[0] is not None:
                self._cache.put(model, prompt, text, mode=self.mode)
                return v
            time.sleep(1 + attempt)
        return (None, ContextMatchType.NOT_APPLICABLE.value, 0.0,
                json.dumps({"status": "IMPOSSIBLE", "reason": "parse_failed_after_retries"}))

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

    def assess(self, groups: List[Dict], case_ids: List[str] = None) -> Tuple[List[List[Tuple]], Dict]:
        if not groups: return [], {}
        model = self.models[0]
        flat_meta, prompts = [], []
        for g_idx, g in enumerate(groups):
            for t_idx, t in enumerate(g["targets"]):
                flat_meta.append((g_idx, t_idx))
                prompts.append(_build_pair_prompt(
                    g.get("src_concepts", ""), g.get("src_cats", ""),
                    t.get("tgt_concepts", ""), t.get("tgt_cats", ""),
                    src_desc=g.get("src_desc", ""), tgt_desc=t.get("desc", ""),
                    src_unit=g.get("src_unit", ""), tgt_unit=t.get("tgt_unit", ""),
                    evidence=t.get("evidence", ""), mode=self.mode))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(self._eval_pair, model, p): i for i, p in enumerate(prompts)}
            results = [None] * len(prompts)
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        grouped = [[] for _ in range(len(groups))]
        for fi, (g_idx, _) in enumerate(flat_meta):
            grouped[g_idx].append(results[fi])
        return grouped, {"total_targets": len(prompts), "model": model}

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