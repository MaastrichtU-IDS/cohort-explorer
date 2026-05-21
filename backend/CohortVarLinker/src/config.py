"""Configuration for the CohortVarLinker mapping pipeline.

Follows the same .env-driven conventions as `backend/src/config.py` and the
legacy `backend/CohortVarLinker/src/config.py`, so folder paths (data_folder,
cohort_folder, output_dir) stay consistent across the whole app.

LLM adjudication is gated on MAPPING_LLM_MODEL being a non-empty string; with the
default (empty) it never calls an LLM, so missing API keys are harmless. Only one
LLM is used per run — see `llm_model` property below.
"""
import os
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any

from dotenv import load_dotenv

# Load root .env (same pattern as backend/src/config.py).
load_dotenv(".env")


def _default_data_folder() -> str:
    """Match legacy cvl_src/config.py: DATA_FOLDER env, else an abs path
    derived from this file's location."""
    return os.getenv(
        "DATA_FOLDER",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data")),
    )


def _get_output_dir() -> str:
    """Resolve the mapping-output directory using the same resolution order as
    the legacy cvl_src/config.py:

      1. MAPPING_OUTPUT_DIR env var (if set and exists)
      2. Docker production path /app/CohortVarLinker/data/mapping_output
      3. ../data/mapping_output relative to this file (local dev)
    """
    env_dir = os.getenv("MAPPING_OUTPUT_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir

    docker_path = "/app/CohortVarLinker/data/mapping_output"
    if os.path.exists(docker_path):
        return docker_path

    relative_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/mapping_output")
    )
    return relative_path


@dataclass
class Settings:
    # ------------------------------------------------------------------ #
    # Matching / thresholds
    # ------------------------------------------------------------------ #
    SIMILARITY_THRESHOLD: float = 0.8
    # Aligned with the new cross-mapping branch (0.55, was 0.45 in our copy
    # before integration). Tuned together with ADAPTIVE_ALPHA below — see
    # `vector_db.adaptive_retrieval` for how they interact.
    ADAPTIVE_THRESHOLD: float = 0.55
    # Upper-bound score gate used by `vector_db.adaptive_retrieval`. Carried
    # over from the branch's config; required by the function default.
    ADAPTIVE_ALPHA: float = 0.9
    LIMIT: int = 5
    DEFAULT_GRAPH_DEPTH: int = 1

    DATE_HINTS: List[str] = field(default_factory=lambda: [
        "visit date", "date of visit", "date of event", "event date",
    ])
    CROSS_CATS: Set[str] = field(default_factory=lambda: {
        "measurement", "observation", "condition_occurrence",
        "condition_era", "observation_period",
    })
    DATA_DOMAINS: List[str] = field(default_factory=lambda: [
        "drug_exposure", "condition_occurrence", "condition_era", "observation",
        "observation_era", "measurement", "visit_occurrence",
        "procedure_occurrence", "device_exposure", "person",
    ])

    # ------------------------------------------------------------------ #
    # Derived Variable Rules
    # Inline registry of computable variables (BMI, eGFR, MAP, BSA, etc.)
    # used by the constraint solver to recognise compatible variables
    # whose codes don't match exactly but can be derived from the same
    # underlying inputs.
    # ------------------------------------------------------------------ #
    DERIVED_VARIABLES_LIST: List[Dict[str, Any]] = field(default_factory=lambda: [
    {
        "name": "BMI",
        "omop_id": 3038553,
        "code": "loinc:39156-5",
        "label": "Body mass index (BMI) [Ratio]",
        "unit": "ucum:kg/m2",
        "required_omops": [3016723, 3025315],  # Weight, Height
        "category": "measurement",
        "data_type": "continuous_variable"
    },
    {
        "name": "eGFR_CG",
        "omop_id": 37169169,
        "code": "snomed:1556501000000100",
        "label": "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula",
        "unit": "ucum:ml/min",
        "required_omops": [3025315, 3016723, 3022304, 46235213],  # Height, Weight, Creatinine, Age/Gender proxy
        "category": "measurement",
        "data_type": "continuous_variable"
    },
    {
        "name": "eGFR_CKD_EPI", # 2021 CKD-EPI is race-free
        "omop_id": 3030354,
        "code": "loinc:33914-3",
        "label": "Glomerular filtration rate/1.73 sq M.predicted by Creatinine-based formula (CKD-EPI)",
        "unit": "ucum:mL/min/{1.73_m2}",
        "required_omops": [4324383, 4265453, 46235213],       # Creatinine, Age, Sex assigned at birth
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "CKD-EPI 2021 race-free: 142*min(Scr/k,1)^a*max(Scr/k,1)^-1.200*0.9938^Age*1.012[F] ; k=0.7[F]/0.9[M], a=-0.241[F]/-0.302[M]"
    },
    {
        "name": "eGFR_MDRD",
        "omop_id": 3053283,
        "code": "loinc:48643-1",
        "label": "Glomerular filtration rate/1.73 sq M predicted by Creatinine-based formula (MDRD)",
        "unit": "ucum:mL/min/{1.73_m2}",
        "required_omops": [4324383, 4265453, 46235213],        # Creatinine, Age, Sex
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "175 * Scr^-1.154 * Age^-0.203 * 0.742[F] * 1.212[Black]"
    },
    {
        "name": "MAP",
        "omop_id": 4239021,                                  # verify
        "code": "snomed:6797001",
        "label": "Mean blood pressure",
        "unit": "ucum:mm[Hg]",
        "required_omops": [4152194, 4154790],                 # SBP, DBP
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "MAP = DBP + (SBP - DBP)/3"
    },
    {
        "name": "Pulse_Pressure",
        "omop_id": 3559113,                                   # SNOMED
        "code": "snomed:811751000000106",
        "label": "Pulse pressure",
        "unit": "ucum:mm[Hg]",
        "required_omops": [4152194, 4154790],                 # SBP, DBP
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "PP = SBP - DBP"
    },
    {
        "name": "BSA_DuBois",
        "omop_id": 4201235,
        "code": "snomed:301898006",
        "label": "Body surface area",
        "unit": "ucum:m2",
        "required_omops": [3025315, 3036277],                 # Body Weight, Body Height
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "DuBois: BSA = 0.007184 * W(kg)^0.425 * H(cm)^0.725"
    },
    {
        "name": "Waist_Hip_Ratio",
        "omop_id": 4087501,                                   # LOINC 9844-2 — verify
        "code": "snomed:248367009",
        "label": "Ratio of waist circumference to hip circumference",
        "unit": "ucum:1",
        "required_omops": [40329251, 4111665],                 # Waist, Hip — verify
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "WHR = waist / hip"
    },
    {
        "name": "LDL_Friedewald", # invalid when TG >= 400 mg/dL or in non-fasting samples
        "omop_id": 3028288,
        "code": "loinc:13457-7",
        "label": "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation",
        "unit": "ucum:mg/dL",
        "required_omops": [4008265, 3007070, 3022192],        # TC, HDL, TG
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "LDL = TC - HDL - TG/5 ; invalid if TG >= 400 mg/dL"
    },
    {
        "name": "Non_HDL_Cholesterol",
        "omop_id": 3044491,                                  # verify
        "code": "loinc:43396-1",
        "label": "Cholesterol non HDL [Mass/volume] in Serum or Plasma",
        "unit": "ucum:mg/dL",
        "required_omops": [4008265, 3007070],                 # TC, HDL
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "Non-HDL = TC - HDL"
    },
    {
        "name": "QTc_Bazett",
        "omop_id": 46235174,
        "code": "loinc:76635-2",
        "label": "Q-T interval corrected based on Bazett formula",
        "unit": "ucum:ms",
        "required_omops": [4216826, 3027018],                 # QT interval feature, heart rate  — verify
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "QTc = QT_ms / sqrt(RR_s) ; RR_s = 60/HR"
    }
    ])

    # ------------------------------------------------------------------ #
    # Embedding model (neural matcher)
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL_NAME: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sapbert")
    )
    FULL_EMBEDDING_MODEL_NAME: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    MODEL_CACHE_DIR: str = field(
        default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "../data/models")
    )

    # ------------------------------------------------------------------ #
    # LLM adjudication (optional).
    #
    # All API keys default to empty strings. The LLM path is only invoked
    # when MAPPING_LLM_MODELS is non-empty (see `llm_models` property), so
    # with no configuration the pipeline runs purely embedding + graph +
    # constraint-solver and never needs a valid key.
    # ------------------------------------------------------------------ #
    TOGETHER_API_KEY: str = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY", ""))
    OPENROUTER_API_KEY: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    OLLAMA_URL: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "localhost:11434"))
    LLM_CACHE_DIR: str = field(default_factory=lambda: os.getenv("LLM_CACHE_DIR", ".llm_cache"))

    # Reference catalogues of known models (author-curated). Not env-driven
    # because they are just "menus"; runtime selection is via MAPPING_LLM_MODELS.
    LOCAL_LLM_MODELS: List[str] = field(default_factory=lambda: [
        "mistral:latest", "llama3.3:70b", "qwen2.5:72b",
    ])
    TOGETHER_LLM_MODELS: List[str] = field(default_factory=lambda: [
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "openai/gpt-oss-120b",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    ])
    LARGE_LMs: List[str] = field(default_factory=lambda: ["gemini-2.5-flash"])

    # ------------------------------------------------------------------ #
    # Graph / ontology resources
    # ------------------------------------------------------------------ #
    GRAPH_REPO: str = "https://w3id.org/CMEO/graph"

    # ------------------------------------------------------------------ #
    # Auth / admins (mirrors legacy cvl_src/config.py)
    # ------------------------------------------------------------------ #
    admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))
    scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email"))

    # ------------------------------------------------------------------ #
    # Folders & endpoints — resolved identically to legacy cvl_src/config.py
    # so that backend/src/mapping.py and the new pipeline read/write the
    # same directories.
    # ------------------------------------------------------------------ #
    data_folder: str = field(default_factory=_default_data_folder)
    cohort_folder: str = field(
        default_factory=lambda: os.path.join(_default_data_folder(), "cohorts")
    )
    output_dir: str = field(default_factory=_get_output_dir)

    sparql_endpoint: str = field(
        default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7878")
    )

    @property
    def auth_audience(self) -> str:
        return "https://explorer.icare4cvd.eu"

    @property
    def query_endpoint(self) -> str:
        return f"{self.sparql_endpoint}/query"

    @property
    def update_endpoint(self) -> str:
        return f"{self.sparql_endpoint}/update"

    @property
    def admins_list(self) -> list[str]:
        return [e.strip() for e in self.admins.split(",") if e.strip()]

    @property
    def logs_filepath(self) -> str:
        return os.path.join(self.data_folder, "logs.log")

    @property
    def sqlite_db_filepath(self) -> str:
        return "vocab.db"

    @property
    def vector_db_path(self) -> str:
        """Qdrant host. Defaults to 'localhost' for dev; set VECTOR_DB_HOST=qdrant
        (the docker-compose service name) in production."""
        return os.getenv("VECTOR_DB_HOST", "localhost")

    @property
    def omop_graph_pickle_path(self) -> str:
        """Path to the cached OmopGraphNX pickle (gzipped).

        Stored alongside the mapping_output directory (i.e. its parent), so
        the cache lives under the volume-mounted CohortVarLinker/data
        directory and survives container rebuilds. Overridable via the
        OMOP_GRAPH_PICKLE_PATH env var.
        """
        explicit = os.getenv("OMOP_GRAPH_PICKLE_PATH")
        if explicit:
            return explicit
        return os.path.join(os.path.dirname(self.output_dir), "graph_nx.pkl.gz")

    @property
    def concepts_file_path(self) -> str:
        """Path to the OMOP concept_relationship CSV used by OmopGraphNX.

        Overridable via CONCEPTS_FILE_PATH env var; otherwise defaults to
        {data_folder}/concept_relationship_enriched.csv so it colocates with
        cohort data under the standard /data mount.
        """
        explicit = os.getenv("CONCEPTS_FILE_PATH")
        if explicit:
            return explicit
        return os.path.join(self.data_folder, "concept_relationship_enriched.csv")

    @property
    def llm_model(self) -> str | None:
        """Single LLM identifier from MAPPING_LLM_MODEL (empty/unset = disabled).

        Passed as the `llm_model` argument to `StudyMapper`. The branch's
        `NeuroSymbolicMatcher` accepts a single model string at a time; an
        empty value disables LLM adjudication entirely (the pipeline runs
        purely embedding + graph + constraint-solver, no API keys read).

        For backward compat we still accept the historical plural
        `MAPPING_LLM_MODELS` env var and pick the first entry from its
        comma-separated list.
        """
        raw = os.getenv("MAPPING_LLM_MODEL", "").strip()
        if raw:
            return raw
        legacy = os.getenv("MAPPING_LLM_MODELS", "")
        first = next((m.strip() for m in legacy.split(",") if m.strip()), "")
        return first or None


settings = Settings()