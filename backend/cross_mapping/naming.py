"""Filename conventions for cross-study mapping outputs.

Keep this module free of heavy imports — backend/src/mapping.py imports it on
the cache-check path, which must answer immediately.
"""

from typing import Optional

from .src.data_model import EmbeddingType, MappingType

DEFAULT_EMBED_MODEL = "biolord"
DEFAULT_EMBEDDING_MODE = EmbeddingType.EH.value
DEFAULT_MAPPING_MODE = MappingType.OEH.value
DEFAULT_LLM_MODEL = "litellm/gpt-oss:120b"

# "{embed_model}+{llm_tag}" and "{mapping_mode}"
_CONFIG_FIELDS = 2

_CSV_SUFFIX = ".csv"


def llm_tag(llm_model: Optional[str], mapping_mode: str) -> str:
    """Filename-safe form of an LLM model id.
    """
    if not llm_model or mapping_mode == MappingType.OO.value:
        return "no-llm"
    return llm_model.split("/")[-1].replace(":", "-")


def config_tag(embed_model: str = DEFAULT_EMBED_MODEL,
               mapping_mode: str = DEFAULT_MAPPING_MODE,
               llm_model: Optional[str] = DEFAULT_LLM_MODEL) -> str:
    """The configuration fingerprint embedded in every output filename."""
    return f"{embed_model}+{llm_tag(llm_model, mapping_mode)}_{mapping_mode}"


def json_name(source_study: str, target_studies: list, cfg: str = None) -> str:
    """Name of the combined JSON the frontend downloads. target_studies are the studies the caller asked for, 
    in the caller's order — not the expanded families, so the name stays predictable.
    """
    cfg = cfg if cfg is not None else config_tag()
    targets = "_".join(t.lower() for t in target_studies)
    return f"{source_study.lower()}_{targets}_{cfg}.json"


def csv_name(source_study: str, target_study: str, cfg: str = None) -> str:
    """Name of the per-pair mapping CSV."""
    cfg = cfg if cfg is not None else config_tag()
    return f"{source_study.lower()}_{target_study.lower()}_{cfg}{_CSV_SUFFIX}"


def parse_cohorts(filename: str) -> Optional[list]:
    """Cohort names from a mapping JSON filename, or None if it is not one.
    Inverse of json_name(). Returns None rather than raising so a caller
    scanning a directory can skip unrelated files.
    """
    if not filename.endswith(".json"):
        return None
    parts = filename[: -len(".json")].split("_")
    # Need at least two cohorts plus the config fields to be a mapping file.
    if len(parts) < _CONFIG_FIELDS + 2:
        return None
    cohorts = [p.lower() for p in parts[:-_CONFIG_FIELDS]]
    # The config always carries the "+" joining embed model and LLM tag; without
    # it this is some other .json that happens to have enough underscores.
    if "+" not in parts[-_CONFIG_FIELDS]:
        return None
    return cohorts
