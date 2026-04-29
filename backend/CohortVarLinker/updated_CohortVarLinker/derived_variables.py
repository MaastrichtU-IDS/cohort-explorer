"""Clinical derived-variable definitions used by the neuro-symbolic matcher.

These are not runtime configuration — they are curated clinical-knowledge
entries that let the matcher synthesise derived concepts (e.g. BMI from
weight+height) even when the cohort does not report them directly.

Each entry describes:
  - the target OMOP concept (id, code, label, unit, category, data_type)
  - the `required_omops` inputs (OMOP IDs the cohort must have to derive it)

If you want to add a new derived concept (e.g. an additional eGFR formula),
add a dict here rather than touching config or the matcher.
"""

from typing import Any, Dict, List


DERIVED_VARIABLES: List[Dict[str, Any]] = [
    {
        "name": "BMI",
        "omop_id": 3038553,
        "code": "loinc:39156-5",
        "label": "Body mass index (BMI) [Ratio]",
        "unit": "ucum:kg/m2",
        "required_omops": [3016723, 3025315],  # Weight, Height
        "category": "measurement",
        "data_type": "continuous_variable",
    },
    {
        "name": "eGFR_CG",
        "omop_id": 37169169,
        "code": "snomed:1556501000000100",
        "label": "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula",
        "unit": "ucum:ml/min",
        "required_omops": [3025315, 3016723, 3022304, 46235213],  # Height, Weight, Creatinine, Age/Gender proxy
        "category": "measurement",
        "data_type": "continuous_variable",
    },
    # CKD-EPI is prioritized over others (eGFR CKD-EPI and MDRD).
]
