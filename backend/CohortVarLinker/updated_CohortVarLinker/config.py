"""Configuration for the CohortVarLinker mapping pipeline.

Follows the same .env-driven conventions as `backend/src/config.py` and the
legacy `backend/CohortVarLinker/src/config.py`, so folder paths (data_folder,
cohort_folder, output_dir) stay consistent across the whole app.

All LLM adjudication is gated on MAPPING_LLM_MODELS being non-empty; with the
default (empty) it never calls an LLM, so missing API keys are harmless.
"""
import os
from dataclasses import dataclass, field
from typing import List, Set

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
    ADAPTIVE_THRESHOLD: float = 0.45
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
    def llm_models(self) -> list[str]:
        """Comma-separated list from MAPPING_LLM_MODELS (empty = LLM disabled).

        Passed as the `llm_models` argument to `StudyMapper`. When empty the
        new pipeline runs purely embedding + graph + constraint-solver with
        no LLM calls, so API keys are never read.
        """
        raw = os.getenv("MAPPING_LLM_MODELS", "")
        return [m.strip() for m in raw.split(",") if m.strip()]


settings = Settings()