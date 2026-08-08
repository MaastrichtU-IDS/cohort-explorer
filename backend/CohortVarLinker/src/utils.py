from collections.abc import Iterable
from rdflib import Dataset, Namespace,Graph, RDF, RDFS, URIRef, DC, Literal
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import quote
import pandas as pd
from .config import settings
import requests
from thefuzz import fuzz
import os
import re
import urllib.parse
from enum import Enum
from typing import Dict, Any, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import unicodedata
from pathlib import Path
# import re

_VISIT_MONTH_RE = re.compile(r'(\d+)\s*months?', re.IGNORECASE)
_VISIT_BASELINE_RE = re.compile(r'baseline|month\s*0|randomization', re.IGNORECASE)
_PRE_BASELINE_MONTH_RE = re.compile(
    r'(\d+)\s*months?\s*(?:prior to|before)\s*baseline',
    re.IGNORECASE,
)
_BASELINE_PRIOR_MONTH_RE = re.compile(
    r'baseline.*?(\d+)\s*months?\s*(?:prior|before)',
    re.IGNORECASE,
)
_BETWEEN_BASELINE_VISIT_RE = re.compile(
    r"\b(?:between|from)\s+(?:baseline|bl)\s+(?:and|to|-)\s+(?:visit\s*)?(\d+)\b",
    re.IGNORECASE,
)
_BASELINE_TO_VISIT_RE = re.compile(
    r"\b(?:baseline|bl)\s*(?:to|[-–—]|through|until)\s*"
    r"(?:(?:visit|month|week|day)s?\s*)?(\d+)\b",
    re.IGNORECASE,
)

_TEMPORAL_CONTEXT_RE = re.compile(
    r'(?:'
      # Pattern A: 'at <temporal-expression>' (with optional double-stamp)
      r'\s*(?:at\s+)?at\s+(?:'
        r'baseline\s*(?:visit)?'
        r'|randomization'
        r'|end\s+of\s+study'
        r'|\d+\s*(?:months?|years?|weeks?|days?)'
        r'|(?:visit\s+)?(?:month\s*)?\d+'
        r'|(?:visit\s*|v)\d+'

        # NEW: follow-up expressions after "at"
        r'|follow[-\s]*up'
        r'|follow[-\s]*up\s+\d+\s*(?:months?|years?|weeks?|days?)'
        r'|\d+\s*(?:months?|years?|weeks?|days?)\s+follow[-\s]*up'
      r')'

    #   # Pattern B: trailing bare 'Month12' (no 'at' prefix)
    #   r'|\s+Month\s*\d+\s*$'

      # Pattern C: '[N months] prior to randomization' (no 'at' prefix)
      r'|\s+(?:\d+\s*months?\s+)?prior\s+to\s+randomization'

      # NEW Pattern D: bare '{X} month follow-up' without "at"
      r'|\s+\d+\s*(?:months?|years?|weeks?|days?)\s+follow[-\s]*up\b'
    r')',
    re.IGNORECASE
)

def is_interval_period(period: str) -> bool:
    if not period:
        return False

    p = str(period).lower().strip()
    return (
        p.startswith("interval ")
        or p.startswith("pre-baseline ")
        or "between" in p
        or " to " in p
    )
def has_real_value(x):
    if x is None or pd.isna(x):
        return False
    s = str(x).strip()
    return bool(s) and s.lower() not in {"na", "n/a", "nan", "none", "null"}

def clean_label_remove_temporal_context(label: str) -> str:
    if not has_real_value(label):
        return label
    
    # Apply repeatedly for double-stamped labels like
    # "Pulmonary valve velocity at visit month 18 at visit month 18"
    cleaned = label
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _TEMPORAL_CONTEXT_RE.sub(' ', cleaned)
    
    # Normalize whitespace and strip punctuation artifacts
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip(' ,;-')
    
    # Safety: never return empty string
    return cleaned if cleaned else label

# check if variable is identifier-code like to determine its statistical type

STRONG_ID_PHRASES = [
    "id",
    "identifier",
    "identification",
    "patient code",
    "patient id",
    "patient number",
    "subject code",
    "subject id",
    "subject number",
    "participant code",
    "participant id",
    "participant number",
    "hospital code",
    "hospital id",
    "hospital number",
    "record code",
    "record id",
    "record number",
    "registry code",
    "registry id",
    "registry number",
    "screening code",
    "screening id",
    "screening number",
    "randomization code",
    "randomization id",
    "randomization number",
    "randomisation code",
    "randomisation id",
    "randomisation number",
    "case report form",
    "code",
    "crf",
]

CLINICAL_CODE_EXCLUSIONS = [
    "code",
    "diagnosis code",
    "procedure code",
    "medication code",
    "device code",
]


_STRONG_ID_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(p) for p in STRONG_ID_PHRASES) + r')\b'
)
_EXCLUSION_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(p) for p in CLINICAL_CODE_EXCLUSIONS) + r')\b'
)
def is_identifier_like_variable(row: pd.Series) -> tuple[bool, list[str]]:
    def _clean_text(x):
        if x is None or pd.isna(x):
            return ""
        return str(x).lower().replace("_", " ").replace("-", " ").strip()

    fields = {
        "variablename":          _clean_text(row.get("variablename", "")),
        "variablelabel":         _clean_text(row.get("variablelabel", "")),
        "variable concept name": _clean_text(row.get("variable concept name", "")),
    }

    reasons = []
    for field, text in fields.items():
        if not text or _EXCLUSION_RE.search(text):
            continue
        m = _STRONG_ID_RE.search(text)
        if m:
            reasons.append(f"{field} contains '{m.group(0)}'")
    return bool(reasons), reasons



_INSTANCE_PREFIX_RE = re.compile(r"^\s*\d+\s*\.\s*")  # leading event index: "1.", "10."

def canonical_var_key(x, strip_instance_prefix: bool = True) -> str:
    """Robust key for comparing variable names across CSVs.

    Encoding-invariant (mojibake repaired, diacritics folded) and
    separator-invariant ('.', '_', '-', whitespace all unified), so names
    differing only in punctuation or accent encoding collapse to one key.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()

    # 1) Repair mojibake deterministically: ftfy if available, else fixed table.
    try:
        from ftfy import fix_text
        s = fix_text(s)
    except Exception:
        for bad, good in {
            "Ã¤": "ä", "Ã¶": "ö", "Ã¼": "ü", "ÃŸ": "ß",
            "Ã„": "Ä", "Ã–": "Ö", "Ãœ": "Ü",
            "√§": "ä", "√∂": "ö", "√º": "ü", "â¤": "ä",
        }.items():
            s = s.replace(bad, good)

    # 2) Normalise + casefold + strip diacritics (ä->a, é->e): key no longer
    #    depends on how the accent was encoded on disk.
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(ch))

    # 3) Optional: drop a leading repeated-event index so per-instance source
    #    variables align with a single base dictionary entry.
    if strip_instance_prefix:
        s = _INSTANCE_PREFIX_RE.sub("", s)

    # 4) Unify ALL separators so 'hf.first.diagnosed' == 'hf_first_diagnosed'.
    s = re.sub(r"[\s._\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s.strip("_").lower()


def is_absolute_vs_percent_dose(src_unit: Optional[str], tgt_unit: Optional[str]) -> bool:
    """
    Detect whether a unit pair represents an absolute medication dose
    versus a percent/target-dose-normalised value.

    Examples:
        mg vs %          -> True
        mg/day vs %      -> True
        ucum:mg vs percent -> True
        kg vs %          -> False
        mg/dl vs %       -> False
        mg vs mg         -> False
    """

    if not src_unit or not tgt_unit:
        return False

    s = str(src_unit).lower().strip()
    t = str(tgt_unit).lower().strip()

    # Light normalization
    s = (
        s.replace("ucum:", "")
         .replace("milligram", "mg")
         .replace("microgram", "mcg")
         .replace("µg", "mcg")
         .replace("μg", "mcg")
         .replace("percentage", "percent")
    )

    t = (
        t.replace("ucum:", "")
         .replace("milligram", "mg")
         .replace("microgram", "mcg")
         .replace("µg", "mcg")
         .replace("μg", "mcg")
         .replace("percentage", "percent")
    )

    percent_units = {"%", "percent"}

    absolute_dose_units = {
        "mg", "mg/day", "mg/d", "mg/24h",
        "g", "g/day", "g/d", "g/24h",
        "mcg", "mcg/day", "mcg/d", "mcg/24h",
        "ug", "ug/day", "ug/d", "ug/24h"
    }

    return (
        (s in absolute_dose_units and t in percent_units)
        or
        (t in absolute_dose_units and s in percent_units)
    )
# Add near the top of graph_similarity.py, after imports

def split_categories(categories: str | None) -> tuple[List[str], List[str]]:
    if not categories or not isinstance(categories, str):
        return [], []
    parts = [c.strip().lower() for c in categories.split("|") if c.strip()]
    if not parts:
        return [], []
    if "=" not in categories:
        return parts, parts

    original_categories: List[str] = []
    category_labels: List[str] = []
    for part in parts:
        if "=" in part:
            code, label = part.split("=", 1)
            code = code.strip().strip('"')
            original_categories.append(code)
            category_labels.append(label.strip())
        else:
            # Truncated or legacy token without code=label (e.g. "contraindica")
            token = part.strip().strip('"')
            original_categories.append(token)
            category_labels.append(token)
    return original_categories, category_labels


def day_month_year(date_str: str) -> tuple:
    formats = [
        "%d-%m-%Y", "%Y-%m-%d", "%m-%Y", "%Y/%m/%d", "%m/%Y","%Y/%m", "%d/%m/%Y", "%m/%d/%Y", "%B %Y", "%Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return (str(dt.day).zfill(2), str(dt.month).zfill(2), str(dt.year))
        except Exception as e:
            pass
    return None

class OntologyNamespaces(Enum):
    CMEO = Namespace("https://w3id.org/CMEO/")
    OMOP = Namespace("https://ohdsi.org/")
    ATC = Namespace("http://purl.bioontology.org/ontology/ATC/")
    RXNORM = Namespace("http://purl.bioontology.org/ontology/RXNORM/")
    UCUM = Namespace("http://unitsofmeasure.org/")
    OMOP_EXT = Namespace("http://omop.org/omopextension/")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")
    OBI = Namespace("http://purl.obolibrary.org/obo/obi.owl/")
    OBCS = Namespace("http://purl.obolibrary.org/obo/obcs.owl/")
    BFO = Namespace("http://purl.obolibrary.org/obo/bfo.owl/")
    STATO = Namespace("http://purl.obolibrary.org/obo/stato.owl/")
    DEFAULT_VALUE = 'Unmapped'
    SNOMEDCT = Namespace("http://purl.bioontology.org/ontology/SNOMEDCT/")
    LOINC = Namespace("http://purl.bioontology.org/ontology/LNC/") 
    RO = Namespace("http://purl.obolibrary.org/obo/ro.owl/")
    IAO = Namespace("http://purl.obolibrary.org/obo/iao.owl/")
    TIME = Namespace("http://www.w3.org/2006/time#")
    SIO = Namespace("http://semanticscience.org/ontology/sio.owl/")
    ICD10 = Namespace("http://purl.bioontology.org/ontology/ICD10/")
    ICD9 = Namespace("http://purl.bioontology.org/ontology/ICD9CM/")
    DUO = Namespace("http://purl.obolibrary.org/obo/duo.owl/")
    NCBI = Namespace("http://purl.bioontology.org/ontology/NCBITAXON/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    CPT4 = Namespace("http://purl.bioontology.org/ontology/CPT4/")
    MESH = Namespace("http://purl.bioontology.org/ontology/MESH/")
    NCI = Namespace("http://purl.bioontology.org/ontology/NCI/")
    # OMOP = Namespace("http://purl.bioontology.org/ontology/OMOP/")
    ICARE = Namespace("https://icare4cvd.eu/")
    UKBiobank = Namespace("https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=")
    CDISC = Namespace("http://rdf.cdisc.org/mms#")
    # UCUM = Namespace("http://purl.bioontology.org/ontology/UCUM/")
    # RXNORM = Namespace("http://purl.bioontology.org/ontology/RXNORM/")



STUDY_TYPES = {
    "observational study": URIRef(f"{OntologyNamespaces.CMEO.value}observational_study_design"),
    "randomized controlled trial": URIRef(f"{OntologyNamespaces.CMEO.value}randomized_controlled_trial_design"),
    "RCT": URIRef(f"{OntologyNamespaces.CMEO.value}randomized_controlled_trial_design"),
    "federated database": URIRef(f"{OntologyNamespaces.CMEO.value}federated_database"),
    "single-arm cross-over randomized intervention": URIRef(f"{OntologyNamespaces.CMEO.value}single_arm_cross_over_design"),
}



def normalize_text(text: str) -> str:
    if text is None or text == "nan" or text == "":
        return None
    text =str(text).lower().strip().replace(" ", "_").replace("/", "_").replace(":", "_").replace('[','').replace(']','')
    return urllib.parse.quote(text, safe='_-')


# def publish_graph_in_chunks(g: Graph, graph_uri: str | None = None, chunk_size: int = 50000) -> bool:
#     """
#     Insert the graph into the triplestore endpoint in chunks.
    
#     :param g: RDF Graph (rdflib.Graph)
#     :param graph_uri: The named graph URI (optional)
#     :param chunk_size: Number of triples per chunk
#     :return: True if all chunks are uploaded successfully, False otherwise
#     """
#     url = f"{settings.sparql_endpoint}/store"
#     if graph_uri:
#         url += f"?graph={graph_uri}"
#         print(f"URL: {url}")

#     headers = {"Content-Type": "application/trig"}
#     total_triples = len(g)
#     print(f"Total triples: {total_triples}")

#     success = True
#     chunk_graph = Graph()
    
#     for i, triple in enumerate(g):
#         chunk_graph.add(triple)

#         # Upload when chunk reaches chunk_size or at the last iteration
#         if len(chunk_graph) >= chunk_size or i == total_triples - 1:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".trig") as tmp_file:
#                 chunk_graph.serialize(tmp_file.name, format="trig")
#                 with open(tmp_file.name, "rb") as file:
#                     response = requests.post(url, headers=headers, data=file, timeout=300)
#                     print(f"Chunk {i//chunk_size + 1}: Response {response.status_code}")
#                     if not response.ok:
#                         print(f"Failed to upload chunk: {response.status_code}, {response.text}")
#                         success = False
            
#             # Clear the chunk_graph for the next batch
#             chunk_graph = Graph()

#     return success

def init_graph(default_graph_identifier: str | None = "https://w3id.org/CMEO/graph/studies_metadata") -> Dataset:
    """Initialize a new RDF graph for nquads with the voc namespace bindings."""
    g = Dataset(store="Oxigraph")
    g.bind("cmeo", OntologyNamespaces.CMEO.value)
    g.bind("bfo", OntologyNamespaces.BFO.value)
    g.bind("obi", OntologyNamespaces.OBI.value)
    g.bind("stato", OntologyNamespaces.STATO.value)
    g.bind("obcs", OntologyNamespaces.OBCS.value)
    g.bind("rdf", RDF)
    g.bind("iao", OntologyNamespaces.IAO.value)
    g.bind("ro", OntologyNamespaces.RO.value)
    g.bind("time", OntologyNamespaces.TIME.value)
    g.bind("sio", OntologyNamespaces.SIO.value)
    g.bind("duo", OntologyNamespaces.DUO.value)
    g.bind("rdfs", RDFS)
    g.bind("ncbi", OntologyNamespaces.NCBI.value)   

    g.bind("dc", DC)
   
    g.graph(identifier=URIRef(default_graph_identifier))
    return g


def get_study_uri(study_id: str) -> URIRef:
    study_uri = URIRef(OntologyNamespaces.CMEO.value + study_id)
    return study_uri
def get_cohort_uri(cohort_id: str) -> URIRef:
    safe_cohort_id = normalize_text(cohort_id)
    if safe_cohort_id is None:
       raise ValueError("Cohort ID is empty")
    return OntologyNamespaces.CMEO.value[f"{safe_cohort_id}"]




def get_cohort_mapping_uri(cohort_id: str) -> URIRef:
    print(f"cohort_id: {cohort_id}")
    safe_cohort_mapping_id = normalize_text(cohort_id)
    if safe_cohort_mapping_id == "":
        print("Cohort ID is empty")
    return OntologyNamespaces.CMEO.value[f"graph/{safe_cohort_mapping_id}"]

def get_var_uri(cohort_id: str | URIRef, var_id: str) -> URIRef:
    safe_var_id = normalize_text(var_id)
    if safe_var_id == "":
        print("Variable ID is empty")
    safe_cohort_id = normalize_text(cohort_id)
    return OntologyNamespaces.CMEO.value[f"{safe_cohort_id}/{safe_var_id}"]





def extract_age_range(text):
    # Normalize Unicode comparison symbols
    text = text.strip().replace("≥", ">=").replace("≤", "<=")

    # Patterns for extracting min and max age
    age_conditions = re.findall(r'(?:age\s*)?(>=|<=|>|<)\s*(\d+(?:\.\d+)?)\s*(?:years\s*old|years)?', text, flags=re.IGNORECASE)

    min_age = None
    max_age = None

    for operator, value in age_conditions:
        value = float(value)
        if operator in ('>=', '>'):
            if min_age is None or value > min_age:
                min_age = value if operator == '>' else value  # can adjust to value + epsilon if needed
        elif operator in ('<=', '<'):
            if max_age is None or value < max_age:
                max_age = value if operator == '<' else value  # can adjust to value - epsilon if needed

    # Also handle "between X and Y years" separately
    match = re.search(r'between\s+(\d+(?:\.\d+)?)\s*(?:and|[-–])\s*(\d+(?:\.\d+)?)\s*years?', text, flags=re.IGNORECASE)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        if min_age is None or min_val > min_age:
            min_age = min_val
        if max_age is None or max_val < max_age:
            max_age = max_val

    if min_age is not None or max_age is not None:
        return min_age, max_age

    return None

# def determine_var_uri(cohort_id: str | URIRef, var_name: str,multi_class_categorical: list[str], binary_categorical: list[str], data_type: str = None, unit:str=None) -> tuple[URIRef, str]:
#     print(f"data_type: {data_type}")
#     # cohort_uri = get_cohort_uri(cohort_id)
#     var_uri = get_var_uri(cohort_id, var_name)
#     if var_name in binary_categorical:
#         statistical_type_uri =  URIRef(var_uri + "/binary_class_variable")
#         statistical_type = "binary_class_variable"
        
#     elif var_name in multi_class_categorical:
#         statistical_type_uri =  URIRef(var_uri + "/multi_class_variable")
#         statistical_type = "multi_class_variable"
#     elif data_type  and data_type in  ["str"] and unit is None:
#         statistical_type_uri =  URIRef(var_uri + "/qualitative_variable")
#         statistical_type = "qualitative_variable"
#     else:
#         # date/time --- dosage/measurement variables variables
#         statistical_type_uri =  URIRef(var_uri + "/continuous_variable")
#         statistical_type = "continuous_variable"
#     return statistical_type_uri,statistical_type

# def  _uri(var_uri,str_label):
#     return URIRef(var_uri + str_label)
def determine_var_uri(cohort_id, var_name, multi_class_categorical, binary_categorical,
                     data_type=None, unit=None, var_label=None):
    var_uri = get_var_uri(cohort_id, var_name)

    if var_name in binary_categorical:
        return URIRef(var_uri + "/binary_class_variable"), "binary_class_variable"
    if var_name in multi_class_categorical:
        return URIRef(var_uri + "/multi_class_variable"), "multi_class_variable"
    # Only now does str + no unit mean free-text
    if data_type in ["str", "string"]:
        return URIRef(var_uri + "/qualitative_variable"), "qualitative_variable"
    # NEW: date/time variables → continuous (or a new temporal type)
    label = (var_label or "").lower()
    name  = (var_name or "").lower()
    is_temporal = (
        data_type == "datetime"
        or any(tok in name  for tok in ("date", "dat", "dt_", "_dt", "time", "month", "year", "visit"))
        or any(tok in label for tok in ("date", "time", "month", "year"))
    )
    if is_temporal:
        return URIRef(var_uri + "/continuous_variable"), "continuous_variable"
    numeric_types = {"int", "integer", "float", "double", "numeric", "real", "decimal"}
    if data_type in numeric_types or unit is not None:
        return URIRef(var_uri + "/continuous_variable"), "continuous_variable"
    return URIRef(var_uri + "/qualitative_variable"), "qualitative_variable"
    

def parse_post_cordinating_concepts_ids(pipe_str) -> List[int]:
    if pd.isna(pipe_str) or not pipe_str: return []
    out = []
    for p in str(pipe_str).replace("||","|").split("|"):
        try: out.append(int(p.strip()))
        except (ValueError, TypeError): pass
    return out

def parse_post_cordinating_concepts_labels(pipe_str) -> List[str]:
    if pd.isna(pipe_str) or not pipe_str: return []
    pipe_str = pipe_str.replace("||", "|")
    return [p.strip().lower() for p in str(pipe_str).split("|") if p.strip()]

def build_concept_parts(main_label, composite_labels_str) -> List[str]:
    """Merge main label with composite labels into an ordered, deduped list.
    """
    parts = []
    if main_label is not None and not pd.isna(main_label):
        m = str(main_label).strip().lower()
        if m:
            parts.append(m)
    for l in parse_post_cordinating_concepts_labels(composite_labels_str):
        if l not in parts:
            parts.append(l)
    return parts

def build_concept_text(main_label, composite_labels_str) -> str:
    """String form of the concept signature (kept for callers that expect a string)."""
    return " ".join(build_concept_parts(main_label, composite_labels_str)).strip()

def is_interval_period(period: str) -> bool:
    if not period:
        return False

    p = str(period).lower().strip()
    return (
        p.startswith("interval ")
        or p.startswith("pre-baseline ")
        or "between" in p
        or " to " in p
    )
def extract_visit_period(visit: str) -> str:
    """Normalize visit string to comparable period label."""
    if not visit:
        return ""

    v = str(visit).lower().strip()
    m = _BETWEEN_BASELINE_VISIT_RE.search(v) or _BASELINE_TO_VISIT_RE.search(v)
    if m:
        return f"interval baseline to visit {m.group(1)}"
    # Must come before generic baseline detection.
    m = _PRE_BASELINE_MONTH_RE.search(v) or _BASELINE_PRIOR_MONTH_RE.search(v)
    if m:
        return f"pre-baseline {m.group(1)} months"

    if _VISIT_BASELINE_RE.search(v):
        return "baseline time"

    m = _VISIT_MONTH_RE.search(v)
    if m:
        return f"follow-up {m.group(1)} months"

    m2 = re.search(r'month\s*(\d+)', v, re.IGNORECASE)
    if m2:
        return f"follow-up {m2.group(1)} months"

    return v

def is_determinate_period(visit: str) -> bool:
    """Return True iff the visit string resolves to a recognised discrete or
    interval period.

    A visit that does not resolve — an undetermined label, a date field, or a
    study-period axis such as 'study days'
    """
    if not visit:
        return False
    v = str(visit).lower().strip()
    return bool(
        _BETWEEN_BASELINE_VISIT_RE.search(v)
        or _BASELINE_TO_VISIT_RE.search(v)
        or _PRE_BASELINE_MONTH_RE.search(v)
        or _BASELINE_PRIOR_MONTH_RE.search(v)
        or _VISIT_BASELINE_RE.search(v)
        or _VISIT_MONTH_RE.search(v)
        or re.search(r'month\s*(\d+)', v, re.IGNORECASE)
    )

# =============================================================================
# UNDETERMINED-VISIT EXPANSION
# =============================================================================
# A study such as Aachen-HF records every variable against a single generic
# axis ("visit date"): the value was taken at *a* visit, but the dictionary
# never says which protocol timepoint that visit is. A study such as TIME-CHF
# names its schedule explicitly (baseline, follow-up 1/3/6/18 months, 1 year,
# end of study). When those two are cross-mapped, an undetermined visit does
# not "differ" from baseline — it is unresolved, and it is a legitimate
# candidate for *every* timepoint the counterpart study exposes. The helpers
# below enumerate that candidate set exhaustively instead of silently
# collapsing it onto baseline.

_VISIT_YEAR_RE = re.compile(r'(\d+)\s*years?', re.IGNORECASE)
_VISIT_WEEK_RE = re.compile(r'(\d+)\s*weeks?', re.IGNORECASE)
_PRE_BASELINE_PERIOD_RE = re.compile(r'pre-baseline\s+(\d+)\s*months?', re.IGNORECASE)
# Unquantified pre-baseline windows: TIME-CHF's "prior to baseline visit".
# extract_visit_period() folds these into "baseline time" because it only
# looks for a month count; that would make a pre-baseline variable align with
# a baseline one, so canonical_visit_period() keeps them apart.
_BARE_PRE_BASELINE_RE = re.compile(
    r'\b(?:prior\s+to|before|pre[-\s]?)\s*baseline', re.IGNORECASE
)

# Labels that place a record in time without naming a protocol timepoint.
# Matched on a space-normalised label, so "visit_date" and "visit date" both hit.
_UNDETERMINED_VISIT_HINTS = (
    "visit date", "date of visit", "visit day",
    "event date", "date of event",
    "index date", "date of assessment", "assessment date",
    "undetermined", "unknown", "not specified", "unspecified",
    "study day", "study days",
)


def is_undetermined_visit(visit: Optional[str]) -> bool:
    """True when a visit label carries no protocol-timepoint information.

    A determinate label always wins: "date of baseline visit" names baseline,
    so it is *not* undetermined even though it contains "date".
    """
    if visit is None:
        return True
    v = str(visit).strip().lower()
    if not v or v in {"nan", "none", "null"}:
        return True
    if is_determinate_period(v):
        return False
    v_norm = re.sub(r'[_\-]+', ' ', v)
    v_norm = re.sub(r'\s+', ' ', v_norm)
    return any(hint in v_norm for hint in _UNDETERMINED_VISIT_HINTS)


def canonical_visit_period(visit: Optional[str]) -> str:
    """extract_visit_period() plus year/week folding.
    """
    if visit is None:
        return ""
    v = str(visit).strip().lower()
    if not v:
        return ""
    if (_BARE_PRE_BASELINE_RE.search(v)
            and not _PRE_BASELINE_MONTH_RE.search(v)
            and not _BASELINE_PRIOR_MONTH_RE.search(v)):
        return "pre-baseline"
    period = extract_visit_period(v)
    if period != v:
        return period
    m = _VISIT_YEAR_RE.search(v)
    if m:
        return f"follow-up {int(m.group(1)) * 12} months"
    m = _VISIT_WEEK_RE.search(v)
    if m:
        return f"follow-up {int(m.group(1))} weeks"
    return period


def is_determinate_canonical_period(visit: Optional[str]) -> bool:
    """is_determinate_period() over the units canonical_visit_period() reads.
    """
    if not visit:
        return False
    v = str(visit).strip().lower()
    if is_determinate_period(v):
        return True
    return bool(_VISIT_YEAR_RE.search(v) or _VISIT_WEEK_RE.search(v))


def visit_sort_key(period: str):
    """Chronological ordering for a canonical period label."""
    p = (period or "").lower()
    m = _PRE_BASELINE_PERIOD_RE.search(p)
    if m:
        return (0, -int(m.group(1)), p)
    if p.startswith("pre-baseline"):
        return (0, 0, p)
    if p.startswith("interval ") or is_interval_period(p):
        return (2, 0, p)
    if "baseline" in p or "randomization" in p:
        return (1, 0, p)
    m = re.search(r'(\d+)\s*months?', p)
    if m:
        return (1, int(m.group(1)), p)
    m = re.search(r'(\d+)\s*weeks?', p)
    if m:
        return (1, int(m.group(1)) / 4.345, p)
    if "end of study" in p or "final" in p or "last" in p:
        return (4, 0, p)
    return (3, 0, p)


def is_pre_baseline_period(period: Optional[str]) -> bool:
    """True for a retrospective window that precedes the baseline visit."""
    p = (period or "").lower()
    return p.startswith("pre-baseline") or bool(_BARE_PRE_BASELINE_RE.search(p))


def expand_undetermined_visit(
    undetermined_visit: Optional[str],
    counterpart_visits: Optional[Iterable] = None,
    include_pre_baseline: bool = False,
) -> List[str]:
    """Enumerate the timepoints an undetermined visit could resolve to.

    `counterpart_visits` is the visit vocabulary of the *other* study — the one
    that names its schedule. Only determinate labels are kept: another
    undetermined label on that side adds no information. The result is
    canonicalised, de-duplicated and ordered chronologically.

    Pre-baseline windows are excluded by default. A pre-baseline period is not
    a scheduled study visit — it is a retrospective window (history taken
    before enrolment), so a generic 'visit date' is never evidence of one.
    Pre-baseline is therefore matched only against pre-baseline, which the
    determinate-vs-determinate branch of resolve_visit_pair() handles by
    canonical equality. Pass include_pre_baseline=True to override.

    Returns [] when the visit is not undetermined, or when the counterpart
    study has no explicit timepoints to expand onto.
    """
    if not is_undetermined_visit(undetermined_visit):
        return []
    if not counterpart_visits:
        return []
    periods: Dict[str, None] = {}
    for cv in counterpart_visits:
        if cv is None:
            continue
        raw = str(cv).strip()
        if not raw or is_undetermined_visit(raw):
            continue
        period = canonical_visit_period(raw)
        if not period:
            continue
        if not include_pre_baseline and is_pre_baseline_period(period):
            continue
        periods.setdefault(period, None)
    return sorted(periods.keys(), key=visit_sort_key)


def resolve_visit_pair(
    source_visit: Optional[str],
    target_visit: Optional[str],
    source_visit_universe: Optional[Iterable] = None,
    target_visit_universe: Optional[Iterable] = None,
) -> Dict[str, Any]:
    """Classify a candidate pair's temporal relation into three states.

    'aligned'      both sides name the same protocol timepoint
    'mismatch'     both sides name a timepoint and they disagree
    'undetermined' at least one side has no protocol timepoint; the pair is one
                   of `candidate_timepoints` possible resolutions rather than a
                   mismatch

    'undetermined' means the dictionaries alone cannot say which visit a
    generic date column refers to, so a consumer must resolve it against the
    actual visit records before harmonising. It is an open question, not a
    finding. No separate verification flag is returned: the status and
    `undetermined_side` already carry it.

    The universes are the visit vocabularies of each study (or of the matched
    concept within each study). The undetermined side is expanded against the
    *opposite* side's universe, falling back to that side's own visit so the
    candidate list is never empty when the counterpart is explicit.
    """
    s_raw = "" if source_visit is None else str(source_visit).strip()
    t_raw = "" if target_visit is None else str(target_visit).strip()
    s_period = canonical_visit_period(s_raw)
    t_period = canonical_visit_period(t_raw)

    s_undet = is_undetermined_visit(s_raw)
    t_undet = is_undetermined_visit(t_raw)

    result: Dict[str, Any] = {
        "source_visit": s_raw,
        "target_visit": t_raw,
        "source_period": s_period,
        "target_period": t_period,
        "undetermined_side": None,
        "candidate_timepoints": [],
        "resolved_timepoint": "",
        "note": "",
    }

    if not s_undet and not t_undet:
        # Canonical equality already keeps pre-baseline apart from baseline,
        # so a pre-baseline window aligns only with another pre-baseline one.
        aligned = s_period == t_period
        result["status"] = "aligned" if aligned else "mismatch"
        result["note"] = (
            "" if aligned
            else f"Timepoints differ ({s_period or s_raw} vs {t_period or t_raw})."
        )
        return result

    result["status"] = "undetermined"
    if s_undet and t_undet:
        result["undetermined_side"] = "both"
        result["note"] = (
            f"Neither timepoint is protocol-resolved ({s_raw or 'unspecified'} vs "
            f"{t_raw or 'unspecified'}); temporal alignment cannot be established "
            f"from the dictionaries and must be verified against the actual visit "
            f"records before harmonisation."
        )
        return result

    if t_undet:
        result["undetermined_side"] = "target"
        universe = list(source_visit_universe) if source_visit_universe else [s_raw]
        candidates = expand_undetermined_visit(t_raw, universe)
        undet_label, other_label, other_period = t_raw, "source", s_period or s_raw
    else:
        result["undetermined_side"] = "source"
        universe = list(target_visit_universe) if target_visit_universe else [t_raw]
        candidates = expand_undetermined_visit(s_raw, universe)
        undet_label, other_label, other_period = s_raw, "target", t_period or t_raw

    result["candidate_timepoints"] = candidates
    result["resolved_timepoint"] = other_period

    # A pre-baseline window is not a scheduled visit, so a generic date column
    # is no evidence that the record falls in one. Do not offer it as a
    # resolution — say plainly that it needs the data to settle.
    if is_pre_baseline_period(other_period):
        result["candidate_timepoints"] = []
        result["note"] = (
            f"The {other_label} timepoint is a pre-baseline window "
            f"('{other_period}') while the {result['undetermined_side']} timepoint is "
            f"undetermined ('{undet_label}'). A pre-baseline window aligns only with "
            f"another pre-baseline window, and a generic visit/date column is not "
            f"evidence of one. Verify against the actual visit records before "
            f"harmonisation."
        )
        return result

    if candidates:
        result["note"] = (
            f"{result['undetermined_side'].capitalize()} timepoint is undetermined "
            f"('{undet_label}'); it may correspond to any of the {len(candidates)} "
            f"{other_label} timepoints [{', '.join(candidates)}]. This pair records "
            f"the '{other_period}' resolution, which is a candidate only — confirm "
            f"against the actual visit records before harmonisation."
        )
    else:
        result["note"] = (
            f"{result['undetermined_side'].capitalize()} timepoint is undetermined "
            f"('{undet_label}') and the {other_label} study exposes no explicit "
            f"timepoints; temporal alignment is unresolved and must be verified "
            f"against the actual visit records."
        )
    return result


def build_visit_universe(rows: Iterable, visit_key: str = "visit") -> List[str]:
    """Collect the distinct raw visit labels present in a study's rows.

    Accepts dicts, pandas rows, or objects with a `.visit` attribute.
    """
    seen: Dict[str, None] = {}
    for row in rows or []:
        if row is None:
            continue
        if isinstance(row, dict):
            value = row.get(visit_key)
        else:
            value = getattr(row, visit_key, None)
        if value is None:
            continue
        raw = str(value).strip()
        if raw and raw.lower() not in {"nan", "none", "null"}:
            seen.setdefault(raw, None)
    return list(seen.keys())


# =============================================================================
# TIMEPOINT INSTANCE EXPANSION
# =============================================================================
# When one study records a variable against a generic 'visit date' and the other
# holds one column per protocol visit, a single asserted mapping stands for
# several concrete column correspondences. Aachen-HF's `Orthopnea` is one column
# repeated per visit row; TIME-CHF spreads the same measurement across
# Orthopnea, Orthopnea1, Orthopnea3, Orthopnea6, Orthopnea12, Orthopnea18. A
# consumer reshaping wide<->long needs those column names, which the mapping row
# does not carry — it names timepoints, not columns.
#
# This produces CANDIDATES, never assertions. Which Aachen visit-date row is
# month 3 is a fact about the data, not the dictionary, so downstream accepts or
# discards each instance once it can see the visit dates. Because a discarded
# candidate is cheap and a missing one is unrecoverable, expansion deliberately
# over-generates: every family member at every timepoint is emitted, and
# lossy pairings (a 0/1/2/3 severity against a yes/no flag) are emitted too,
# flagged rather than filtered.
#
# Nothing here writes files or mutates its inputs. The evaluated mapping CSV is
# untouched, so retrieval counts and ground-truth scoring are unaffected.


@dataclass(frozen=True)
class TimepointExpansionRow:
    """One concrete (source column, target column, timepoint) correspondence.

    Distinct from verdict.TimepointInfo, which annotates a *mapping row* with
    how its two timepoints relate. This is a *row of the expansion output*: one
    candidate column pair a consumer can accept or discard against the data.
    """

    source_var: str
    target_var: str
    visit_period: str
    """Canonical period this instance sits at, e.g. 'follow-up 3 months'."""
    source_visit: str
    target_visit: str
    concept_id: Optional[int] = None
    undetermined_side: str = ""
    anchor_source: str = ""
    anchor_target: str = ""
    """The mapping row this instance was derived from. Lets a caller clone that
    row so the expansion keeps the mapping file's own schema instead of
    inventing a second one."""
    origin: str = "expanded"
    """'asserted' when the cross-mapping itself produced this pair, 'expanded'
    when it was derived from the counterpart study's visit schedule. Lets a
    consumer weight the two differently, or ignore expansions entirely."""
    parameter_columns: str = ""
    """For a derived variable, the actual columns to feed the formula at this
    timepoint — 'Weight3|Crea3|Age|Gender' — rather than the concept ids the
    mapping carries, which do not identify a column when a concept spans six of
    them."""
    broadcast_parameters: str = ""
    """Parameters taken as available here despite not being re-measured at this
    visit (age, sex, height). Recorded so a consumer can check the assumption:
    a lab run only at enrolment would look the same to the availability rule."""
    harmonization_status: str = ""
    """No separate verification flag: a non-empty `undetermined_side` already
    means the dictionaries cannot say which visit this row belongs to, so it
    must be checked against the data. Only the *timepoint* is in question —
    the concept/unit/category verdict in `harmonization_status` stands."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _record_get(record: Any, *names: str) -> Any:
    """Read a field from a VariableNode, dict, or pandas row alike."""
    for name in names:
        if isinstance(record, dict):
            if name in record:
                return record[name]
        else:
            value = getattr(record, name, None)
            if value is not None:
                return value
    return None


def _context_signature(value: Any) -> frozenset:
    """Normalise a context id field (list, pipe string, or empty) to a set.

    Context is part of the family key because two variables can share a concept
    id and still be different variables — TIME-CHF's ALT measurement versus
    GISSI-HF's ALT reference-range bounds being the standing example.
    """
    if value is None:
        return frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(v).strip() for v in value]
    else:
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none", "null"):
            return frozenset()
        parts = [p.strip() for p in re.split(r"[|;]+", text)]
    return frozenset(p for p in parts if p and p.lower() not in ("nan", "none", "null"))


def _normalise_unit(value: Any) -> str:
    """UCUM code, lowercased, with any ``ucum:`` prefix dropped."""
    text = "" if value is None else str(value).strip().lower()
    if text in ("nan", "none", "null"):
        return ""
    return text[5:].strip() if text.startswith("ucum:") else text


def _family_key(record: Any) -> Optional[Tuple[Any, frozenset, frozenset, str, str]]:
    """The identity a timepoint family shares.

    ``(concept, context, category concepts, unit, statistical type)``.

    Concept and context alone are not enough: they say what a variable is
    *about*, not how it is recorded, so `Edema` (0/1/2/3 severity) and
    `Edema_01` (yes/no) key identically and become each other's timepoint
    siblings. Expansion then manufactures pairs no matcher judged, and a
    verdict can transfer between genuinely different variables.

    Category CONCEPTS rather than labels, and a set rather than a sequence,
    because the label list is unstable in all three ways that matter: order
    (`no||yes` at baseline, `yes||no` at 12 months for one variable), spelling
    (`sinus rhythm` and `sinus rythm` both resolve to 4091898), and language
    across the study dictionaries. Keying on the rendered labels splits real
    families apart; keying on the concepts does not.

    Statistical type here is the analytic classification (binary / multi-class
    / continuous), NOT a storage type — `INT` covers binary and multi-class
    alike and would separate almost nothing.
    """
    concept = _record_get(record, "main_id", "omop_id", "variable_omop_id")
    if concept in (None, "", 0):
        return None
    try:
        concept = int(concept)
    except (TypeError, ValueError):
        return None

    context = _context_signature(
        _record_get(record, "context_ids", "composite_code_omop_ids",
                    "additional_context_omop_id")
    )
    categories = _context_signature(
        _record_get(record, "category_ids", "categorical_value_omop_id",
                    "categories_omop_ids")
    )
    unit = _normalise_unit(_record_get(record, "unit", "units"))
    stat_type = _record_get(record, "statistical_type", "vartype", "var_type")
    stat_type = "" if stat_type is None else str(stat_type).strip().lower()
    if stat_type in ("nan", "none", "null"):
        stat_type = ""
    return (concept, context, categories, unit, stat_type)


_PARAM_IDS_RE = re.compile(r"parameter columns?\s*\[([^\]]*)\]", re.IGNORECASE)


def parse_parameter_concepts(transformation_rule: Any) -> List[int]:
    """Pull the parameter concept ids out of a derived-variable rule string.

    The rule reads "Derived variable eGFR_CKD_EPI using parameter columns
    [3025315, 3016723, 3022304, 46235213]." — concept ids, which do not name a
    column: 3025315 (body weight) is six columns in TIME-CHF.
    """
    if transformation_rule is None:
        return []
    match = _PARAM_IDS_RE.search(str(transformation_rule))
    if not match:
        return []
    return [int(n) for n in re.findall(r"\d+", match.group(1))]


def derive_parameter_timepoints(
    parameter_concepts: Sequence[int],
    variables: Sequence[Any],
    broadcast_time_invariant: bool = True,
) -> Tuple[Dict[str, Dict[int, str]], Set[int]]:
    """Work out at which timepoints a derived variable can actually be computed.

    A derived variable exists wherever *all* its parameters are available, so
    the timepoints are the intersection of the parameters' visits.

    Availability is not the same as being re-measured. Age, sex and height are
    recorded once and still apply at month 18; requiring them to repeat
    collapses every derivation onto baseline, which is what the plain
    intersection in neuro_matcher does. A parameter occurring exactly once
    across the study is therefore treated as time-invariant and broadcast to
    every timepoint — and returned in the second element so the caller can say
    which parameters that assumption was made for. A lab run only at enrolment
    is indistinguishable to this rule, which is why it is reported rather than
    hidden.

    Returns ({period: {concept_id: variable name}}, broadcast concept ids).
    """
    by_concept: Dict[int, List[Any]] = {}
    for record in variables or []:
        concept = _record_get(record, "main_id", "omop_id", "variable_omop_id")
        try:
            concept = int(concept)
        except (TypeError, ValueError):
            continue
        by_concept.setdefault(concept, []).append(record)

    varying: List[Set[str]] = []
    per_concept_periods: Dict[int, Dict[str, str]] = {}
    broadcast: Set[int] = set()

    for concept in parameter_concepts:
        records = by_concept.get(concept, [])
        if not records:
            return {}, set()          # a parameter absent entirely: nothing to derive
        periods: Dict[str, str] = {}
        for record in records:
            period = canonical_visit_period(_record_get(record, "visit") or "")
            name = str(_record_get(record, "name", "variablename") or "").strip()
            if period and name:
                periods.setdefault(period, name)
        per_concept_periods[concept] = periods
        if broadcast_time_invariant and len(records) == 1:
            broadcast.add(concept)
        else:
            varying.append(set(periods))

    if varying:
        shared = set.intersection(*varying)
    else:
        # Every parameter is time-invariant, so the derivation holds wherever
        # any of them is recorded — in practice, baseline.
        shared = {p for periods in per_concept_periods.values() for p in periods}

    resolved: Dict[str, Dict[int, str]] = {}
    for period in shared:
        columns: Dict[int, str] = {}
        for concept in parameter_concepts:
            periods = per_concept_periods[concept]
            name = periods.get(period)
            if name is None and concept in broadcast:
                name = next(iter(periods.values()), "")
            if not name:
                columns = {}
                break
            columns[concept] = name
        if columns:
            resolved[period] = columns
    return resolved, broadcast


def _index_families(variables: Sequence[Any]) -> Dict[Tuple[Any, frozenset], List[Any]]:
    """Group a study's variables by (concept, context)."""
    families: Dict[Tuple[Any, frozenset], List[Any]] = {}
    for record in variables or []:
        key = _family_key(record)
        if key is None:
            continue
        families.setdefault(key, []).append(record)
    return families


def expand_derived_variable_timepoints(mappings, source_variables):
    """Emit one row per timepoint at which a derived variable can be computed.

    Independent of timepoint expansion, and deliberately so. An ordinary
    repeated variable is a real dictionary column at every visit -- TIME-CHF
    names Edema, Edema1, Edema3, Edema6 -- so candidate generation already
    produces a row per timepoint and expansion adds nothing. A derived variable
    is different: `derived_BSA_DuBois` is synthesised by compute_derived_variables
    and exists wherever ALL its parameters do, so nothing names it per visit and
    the matcher can only ever produce the single row it was built from.

    Returns (rows, diagnostics). Each output row is the judged row with the
    visit swapped for the resolved period, plus `parameter_columns` naming the
    actual source columns for that period and `broadcast_parameters` listing the
    parameters assumed time-invariant (age, sex and height are recorded once and
    still apply at month 18; requiring them to repeat collapses every derivation
    onto baseline). Rows whose parameters never co-occur are returned unchanged
    with a diagnostic rather than dropped.
    """
    # Check for a DataFrame BEFORE list(): list(df) yields column names, not
    # rows, so testing the converted value never fires.
    if mappings is None:
        rows = []
    elif hasattr(mappings, "to_dict"):
        rows = mappings.to_dict("records")
    else:
        rows = list(mappings)

    out, diagnostics = [], []
    for row in rows:
        parameter_concepts = parse_parameter_concepts(_record_get(row, "transformation_rule"))
        if not parameter_concepts:
            out.append(dict(row))
            continue

        s_name = str(_record_get(row, "source", "source_var", "src_var") or "").strip()
        t_name = str(_record_get(row, "target", "target_var", "tgt_var") or "").strip()
        s_visit = str(_record_get(row, "source_visit") or "").strip()

        resolved, broadcast = derive_parameter_timepoints(parameter_concepts, source_variables)
        if not resolved:
            kept = dict(row)
            kept.setdefault("parameter_columns", "")
            kept.setdefault("broadcast_parameters", "")
            out.append(kept)
            diagnostics.append({
                "issue": "derived_parameters_unavailable",
                "source": s_name, "target": t_name,
                "detail": f"parameters {parameter_concepts} are not all available at "
                          f"any single timepoint; kept the judged row only",
            })
            continue

        for period, columns in sorted(resolved.items()):
            instance = dict(row)
            instance["source_visit"] = period or s_visit
            instance["parameter_columns"] = "|".join(columns[c] for c in parameter_concepts)
            instance["broadcast_parameters"] = "|".join(
                str(c) for c in parameter_concepts if c in broadcast)
            out.append(instance)
    return out, diagnostics


def expand_timepoint_instances(
    mappings: Iterable[Any],
    source_variables: Sequence[Any] = (),
    target_variables: Sequence[Any] = (),
    include_pre_baseline: bool = False,
) -> Tuple[List[TimepointExpansionRow], List[Dict[str, Any]]]:
    """Expand undetermined-timepoint mappings into concrete column instances.

    `mappings` are cross-mapping rows (dicts or a DataFrame's ``to_dict("records")``)
    carrying at least source, target, source_visit, target_visit; the concept and
    context columns are used when present. `source_variables` / `target_variables`
    are the two studies' full variable lists — full, not scoped, since the point
    is to name every real column a consumer will meet in the data.

    Returns (instances, diagnostics).

    Expansion only fires where exactly one side is undetermined. A pair whose
    timepoints are both known needs no expansion, and a pair with both sides
    undetermined has nothing to anchor on; each is emitted once, as asserted.

    `diagnostics` records what could not be expanded and why — a family that
    could not be located, a concept whose context diverges across timepoints —
    so gaps surface instead of silently shrinking the candidate set.
    """
    rows = list(mappings) if mappings is not None else []
    if hasattr(rows, "to_dict"):  # a DataFrame slipped through
        rows = rows.to_dict("records")

    src_families = _index_families(source_variables)
    tgt_families = _index_families(target_variables)

    # Name -> record, so a mapping row's anchor can be resolved to the record
    # its family key was built from.
    def _by_name(variables: Sequence[Any]) -> Dict[str, Any]:
        index: Dict[str, Any] = {}
        for record in variables or []:
            name = str(_record_get(record, "name", "variablename") or "").strip().lower()
            if name:
                index.setdefault(name, record)
        return index

    src_by_name = _by_name(source_variables)
    tgt_by_name = _by_name(target_variables)

    # pair -> verdict, for every pair the matcher actually decided. Consulted in
    # _emit rather than threaded through the loop: one family is reached from
    # several mapping rows, and whichever arrives first must not determine
    # whether the others keep their verdict.
    asserted: Dict[Tuple[str, str], str] = {}
    for row in rows:
        s_name = str(_record_get(row, "source", "source_var", "src_var") or "").strip()
        t_name = str(_record_get(row, "target", "target_var", "tgt_var") or "").strip()
        if s_name and t_name:
            asserted[(s_name.lower(), t_name.lower())] = str(
                _record_get(row, "harmonization_status") or "").strip()

    instances: Dict[Tuple[str, str, str], TimepointExpansionRow] = {}
    diagnostics: List[Dict[str, Any]] = []

    def _emit(source_var: str, target_var: str, period: str,
              s_visit: str, t_visit: str, concept: Optional[int],
              side: str, anchor: Tuple[str, str],
              parameter_columns: str = "", broadcast: str = "",
              origin: Optional[str] = None) -> None:
        pair = (source_var.lower(), target_var.lower())
        key = (pair[0], pair[1], period)
        if key in instances:
            return
        instances[key] = TimepointExpansionRow(
            source_var=source_var, target_var=target_var, visit_period=period,
            source_visit=s_visit, target_visit=t_visit, concept_id=concept,
            undetermined_side=side,
            anchor_source=anchor[0], anchor_target=anchor[1],
            # `origin` is normally decided by the pair name, but a derived
            # variable keeps the same name at every timepoint, so the caller
            # states which period was the judged one.
            origin=origin or ("asserted" if pair in asserted else "expanded"),
            harmonization_status=asserted.get(pair, ""),
            parameter_columns=parameter_columns, broadcast_parameters=broadcast,
        )

    for row in rows:
        s_name = str(_record_get(row, "source", "source_var", "src_var") or "").strip()
        t_name = str(_record_get(row, "target", "target_var", "tgt_var") or "").strip()
        if not s_name or not t_name:
            continue
        s_visit = str(_record_get(row, "source_visit") or "").strip()
        t_visit = str(_record_get(row, "target_visit") or "").strip()
        resolution = resolve_visit_pair(s_visit, t_visit)
        side = resolution["undetermined_side"] or ""

        # A derived variable is not a column, so it has no family to expand
        # against — it exists wherever all its parameters do. Resolve those
        # parameters to real column names per timepoint instead.
        parameter_concepts = parse_parameter_concepts(
            _record_get(row, "transformation_rule"))
        if parameter_concepts:
            resolved, broadcast = derive_parameter_timepoints(
                parameter_concepts, source_variables)
            if resolved:
                judged_period = canonical_visit_period(s_visit)
                for period, columns in resolved.items():
                    _emit(s_name, t_name, period,
                          period if period else s_visit, t_visit,
                          _family_key(row)[0] if _family_key(row) else None,
                          side, (s_name, t_name),
                          parameter_columns="|".join(columns[c] for c in parameter_concepts),
                          broadcast="|".join(str(c) for c in parameter_concepts
                                             if c in broadcast),
                          origin="asserted" if period == judged_period else "expanded")
                continue
            diagnostics.append({
                "issue": "derived_parameters_unavailable",
                "source": s_name, "target": t_name,
                "concept_id": _family_key(row)[0] if _family_key(row) else None,
                "context": [],
                "detail": f"parameters {parameter_concepts} are not all available "
                          f"at any single timepoint; kept the asserted pair only",
            })

        # Both timepoints known, or neither anchorable: one row, no expansion.
        if resolution["status"] != "undetermined" or side == "both":
            period = (canonical_visit_period(s_visit)
                      or canonical_visit_period(t_visit) or "")
            _emit(s_name, t_name, period, s_visit, t_visit,
                  _family_key(row)[0] if _family_key(row) else None, side,
                  (s_name, t_name))
            continue

        # One side is undetermined: expand the determinate side's family.
        if side == "target":
            families, by_name, anchor_name = src_families, src_by_name, s_name
        else:
            families, by_name, anchor_name = tgt_families, tgt_by_name, t_name

        # Key off the anchor's own variable record rather than rebuilding the
        # key from the mapping row's flattened columns. The row and the record
        # spell the same facts differently — pipe-joined strings against typed
        # lists, `ucum:mg` against `mg` — so two constructions have to be kept
        # byte-identical or every lookup silently misses and expansion stops
        # happening at all. One builder, one input type, no drift.
        anchor = by_name.get(anchor_name.lower())
        key = _family_key(anchor) if anchor is not None else None
        family = families.get(key, []) if key else []

        if not family:
            # Nothing to expand onto — emit the asserted pair and say so.
            period = canonical_visit_period(s_visit if side == "target" else t_visit)
            _emit(s_name, t_name, period, s_visit, t_visit,
                  key[0] if key else None, side, (s_name, t_name))
            diagnostics.append({
                "issue": "family_not_found",
                "source": s_name, "target": t_name,
                "concept_id": key[0] if key else None,
                "context": sorted(key[1]) if key else [],
                "detail": ("no variable record found for the anchor"
                           if anchor is None else
                           "no variables share this (concept, context, categories, "
                           "unit, statistical type); expansion limited to the asserted pair"),
            })
            continue

        # A timepoint family holds one member per visit. Two members claiming a
        # period is a contradiction — whatever they are, they are not the same
        # variable measured twice — so refuse rather than pick. This is the only
        # guard that catches variables sharing concept, categories, unit AND
        # type, such as `edema_01` (current) against `edemahistory_01` (history)
        # or `fu__v6` (all-cause) against `fu_hf_v6` (HF-specific).
        by_period: Dict[str, List[str]] = {}
        for member in family:
            member_name = str(_record_get(member, "name", "variablename") or "").strip()
            if member_name:
                by_period.setdefault(
                    canonical_visit_period(str(_record_get(member, "visit") or "").strip()) or "",
                    []).append(member_name)
        clashes = {p: sorted(set(n)) for p, n in by_period.items() if len(set(n)) > 1}
        if clashes:
            detail = "; ".join(
                "{}: {}".format(period_name or "unknown", ", ".join(names))
                for period_name, names in sorted(clashes.items())
            )
            period = canonical_visit_period(s_visit if side == "target" else t_visit)
            _emit(s_name, t_name, period, s_visit, t_visit,
                  key[0], side, (s_name, t_name))
            diagnostics.append({
                "issue": "family_not_a_timepoint_series",
                "source": s_name, "target": t_name,
                "concept_id": key[0],
                "context": sorted(key[1]),
                "detail": f"two variables claim the same visit period ({detail}) "
                          f"— not a timepoint series, so not expanded",
            })
            continue

        for member in family:
            member_name = str(_record_get(member, "name", "variablename") or "").strip()
            member_visit = str(_record_get(member, "visit") or "").strip()
            if not member_name:
                continue
            period = canonical_visit_period(member_visit)
            if not include_pre_baseline and is_pre_baseline_period(period):
                continue
            if side == "target":
                _emit(member_name, t_name, period, member_visit, t_visit,
                      key[0], side, (s_name, t_name))
            else:
                _emit(s_name, member_name, period, s_visit, member_visit,
                      key[0], side, (s_name, t_name))

    ordered = sorted(
        instances.values(),
        key=lambda i: (i.source_var.lower(), i.target_var.lower(),
                       visit_sort_key(i.visit_period)),
    )
    return ordered, diagnostics


def extract_tick_values(texts: str) -> List[float]:
    """Extract numeric tick labels from a matplotlib Text() list‑string.

    Example input (single string):
        "Text(-2.5, 0, '−2.5') - Text(0.0, 0, '0.0') - Text(2.5, 0, '2.5')"

    Returns:
        [-2.5, 0.0, 2.5]
    """
    ticks = []
    # Split the string at the separators used by the user (" - ")
    for token in texts.split(" - "):
        # Regex captures the *label* part (text between the final pair of quotes)
        m = re.search(r"Text\([^,]+,\s*[^,]+,\s*'([^']+)'\)", token)
        if m:
            val_str = m.group(1).replace('−', '-')  # normalise Unicode minus
            try:
                ticks.append(float(val_str))
            except ValueError:
                # Skip if the captured label is not a number
                pass
    return ticks
def is_categorical_variable(df):
    binary_categorical = []
    multi_class_categorical = []
    # create dict using variable name and CATREGORICAL
    column_dict = dict(zip(df['variablename'], df['categorical']))
    for key, value in column_dict.items():
        if pd.notna(value) and value and value != "":
                if len(value.split("|")) == 2:
                    binary_categorical.append(normalize_text(key))
                else:
                    multi_class_categorical.append(normalize_text(key))
    return binary_categorical, multi_class_categorical

def safe_int(value):
    """Safely convert a value to an integer, returning None if the value is invalid."""
    try:
        return int(float(value)) if value else None
    except ValueError:
        print(f"Invalid integer value: {value}")
        return None


def build_label_mapping(src_labels_str,src_codes_str,  tgt_labels_str, tgt_codes_str,):

    """_summary_
       detect semantic similarities between categories (value set) of two variables and design the overlap for possible joining in actual datasets
    """

    def parse_list(v):
        if v in (None, "") or (isinstance(v, float) and pd.isna(v)):
            return []
        return [str(x).strip() for x in str(v).split(";") if str(x).strip()]

    # code -> label
    def c2l(codes, labels):
        return {c:l for c,l in zip(parse_list(codes), parse_list(labels)) if c and l}
   
    src_c2l = c2l(src_codes_str, src_labels_str)
    tgt_c2l = c2l(tgt_codes_str, tgt_labels_str)

    # label -> [codes] (case-insensitive key, but keep a pretty label)
    def l2codes(c2l_):
        lab2codes, pretty = {}, {}
        for c, l in c2l_.items():
            k = l.lower()
            lab2codes.setdefault(k, []).append(c)
            pretty.setdefault(k, l)
        for k in lab2codes:
            lab2codes[k].sort()
        return lab2codes, pretty

    src_lab2codes, src_pretty = l2codes(src_c2l)
    tgt_lab2codes, tgt_pretty = l2codes(tgt_c2l)

    overlap_keys = sorted(set(src_lab2codes) & set(tgt_lab2codes))

    # Build readable mapping and a deterministic code→code map (choose first target code per label)
    items, code_map = [], {}
    for k in overlap_keys:
        label = src_pretty.get(k, tgt_pretty.get(k, k))
        s_codes = src_lab2codes[k]
        t_codes = tgt_lab2codes[k]
        items.append(f"{label}: {', '.join(s_codes)}<->{', '.join(t_codes)}")
        # choose first source code and first target code for a compact map
        code_map[s_codes[0]] = t_codes[0]

    unmapped_src_labels = sorted(src_pretty[k] for k in src_lab2codes.keys() - set(overlap_keys))
    unmapped_tgt_labels = sorted(tgt_pretty[k] for k in tgt_lab2codes.keys() - set(overlap_keys))

    identical = (set(src_lab2codes) == set(tgt_lab2codes)) and all(
        len(src_lab2codes[k]) == len(tgt_lab2codes[k]) for k in overlap_keys
    )

    return {
        "mapping_str": "; ".join(items) if items else None,
        "code_map": code_map,  # e.g., {"2": "3", "5": "6"}
        "overlap_labels": [src_pretty[k] for k in overlap_keys],
        "unmapped_source_labels": "; ".join(unmapped_src_labels),
        "unmapped_target_labels": "; ".join(unmapped_tgt_labels),
        "has_overlap": bool(overlap_keys),
        "identical": identical,
    }

def adjust_for_additional_context(result_dict, status, src_info, tgt_info, mapping_relation):
    """ __summary__ 
        Post-process the transformation description, harmonization status and skos mapping relation for a pair of variables to account for additional context and timepoint differences.
    """
    def lower_stat_by_1(status: str) -> str:
        hierarchy_ascending = [
            "Identical Match",
            "Compatible Match",
            "Partial Match (Proximate)",
            "Partial Match (Tentative)",
            "Not Applicable",
        ]
        # skos_hierarchy = ["skos:exactMatch", "skos:closeMatch", "relatedMatch"]
        if status in hierarchy_ascending:
            idx = hierarchy_ascending.index(status)
            if idx + 1 < len(hierarchy_ascending):
                return hierarchy_ascending[idx + 1]
        return status
    
    src_codes = src_info.get("composite_code", None)
    tgt_codes = tgt_info.get("composite_code", None)
    src_visit = src_info.get("visit", None)
    tgt_visit = tgt_info.get("visit", None)
    desc = result_dict.get("description", "").rstrip(".") + "."
    # No context on either side -> nothing to adjust
    if not src_codes and not tgt_codes:
        if src_visit == tgt_visit:
            return result_dict, mapping_relation, status
        else:
            result_dict["description"] = desc + (
                "Temporal context differs between source and target at metadata level.")
            if (('event' in src_visit.lower() and  'baseline' in tgt_visit.lower()) or ('baseline' in src_visit.lower() and  'event' in tgt_visit.lower())):
                status = lower_stat_by_1(status)
            return result_dict, mapping_relation, status

    # Exact match
    elif src_codes == tgt_codes:
        if src_visit == tgt_visit:
            return result_dict, mapping_relation, status
        else:
            # status = lower_stat_by_1(status)
            if (('event' in src_visit.lower() and  'baseline' in tgt_visit.lower()) or ('baseline' in src_visit.lower() and  'event' in tgt_visit.lower())):
                status = lower_stat_by_1(status)
            result_dict["description"] = desc + (
                "Temporal context differs between source and target at metadata level.")
            return result_dict, "skos:relatedMatch", status
    else:
        # print(f"Adjusting for additional context: src_codes={src_codes}: {src_info}, tgt_codes={tgt_codes}: {tgt_info}")
        src_codes_lst = src_codes.split("|") if src_codes else []
        
        tgt_codes_lst = tgt_codes.split("|") if tgt_codes else []
        mapping_relation = "skos:relatedMatch"
        if set(src_codes_lst) & set(tgt_codes_lst):
            # Partial overlap
            extra_note = (
                f"Clinical context partially overlaps between source ({src_codes}) "
                f"and target ({tgt_codes})."
            )
            
            # Adjust status one step down if it was a "complete" match
            status = lower_stat_by_1(status)
        else:
            # Disjoint context or context only on one side
            extra_note = (
                f"Clinical context differs between source ({src_codes}) "
                f"and target ({tgt_codes})."
            )
            # Downgrade to at most Partial
            status = "Partial Match (Tentative)"

        
        result_dict["description"] =  desc + extra_note
       
        if src_visit != tgt_visit:
            result_dict["description"] += " Temporal context also differs between source and target at metadata level."
            # status = lower_stat_by_1(status)

        return result_dict,  mapping_relation, status


def execute_query(query: str) -> Iterable[Dict[str, Any]]:
    sparql = SPARQLWrapper(settings.query_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_embedding_model(model_name="biolord"):
    from .embed_model import get_model
    return get_model(backend=model_name)
   
def apply_rules(domain, mapping_relation, src_info, tgt_info):
    
    """ __summary:
        Apply transformation rules to determine mapping between source and target variables.
        Returns a tuple of (details_dict, status_str).
        Status can be one of:
        - "Identical Match" --- perfect match at both semantic/statistical level (unit/categories, timepoint), no transformation needed
        - "Compatible Match" --- perfect match at both semantic but differ in statistical representation  (e.g., unit conversion, time-point difference, derived variables), but overall possible to transform
        - "Partial Match (Proximate)" --- partial overlap in semantic/statistical representation (e.g., some categories match, some differ; composite variable with overlapping context, timepoint difference, statistical datatype different but convertable), manual review needed
        - "Partial Match (Tentative)" --- minimal overlap in semantics and no overlap in statistical representation (e.g., no categories match; composite variable with disjoint context; statistical datatype very different), manual review needed
        - "Not Applicable" --- transformation not applicable as variables are fundamentally different (e.g., different statistical types) 
    """

    def norm_or_none(x):
        return str(x).strip().lower() if (x not in (None, "", "nan") and not pd.isna(x)) else None
    
    def finalize(details, status, src_ctxt=None, tgt_ctxt=None, mapping_relation=None):
        """Apply additional-context logic uniformly before returning."""
        return adjust_for_additional_context(details, status, src_ctxt, tgt_ctxt, mapping_relation)
    
    src_type = norm_or_none(src_info.get('stats_type'))
    tgt_type = norm_or_none(tgt_info.get('stats_type'))
    src_unit = norm_or_none(src_info.get('unit'))
    tgt_unit = norm_or_none(tgt_info.get('unit'))
    src_data_type = norm_or_none(src_info.get('data_type'))
    tgt_data_type = norm_or_none(tgt_info.get('data_type'))
    src_composite_codes = norm_or_none(src_info.get('composite_code')) # all codes for composite variables
    tgt_composite_codes = norm_or_none(tgt_info.get('composite_code')) # all codes for composite variables
    # src_visit = norm_or_none(src_info.get('visit'))
    # tgt_visit = norm_or_none(tgt_info.get('visit'))
    src_var_name = str(src_info.get('var_name', '')).lower()
    tgt_var_name = str(tgt_info.get('var_name', '')).lower()
    
    valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable", "qualitative_variable"}
    if (src_type not in valid_types or tgt_type not in valid_types) or (src_type is None or tgt_type is None):
        if 'derived' not in src_var_name and 'derived' not in tgt_var_name:
            details = {"description": "Transformation not applicable (invalid or missing statistical type)."}
            return finalize(details, "Not Applicable", src_info, tgt_info, mapping_relation)
        else:
            details = {"description": "Use one or more variables to derive this variable."}
            return finalize(details, "Compatible Match", src_info, tgt_info, mapping_relation)

    label_mapping = build_label_mapping(
            src_info.get('categories_labels', ''),
            src_info.get('original_categories', ''),
            tgt_info.get('categories_labels', ''),
            tgt_info.get('original_categories', '')
        )
    # --- same type 
 
    if src_type == tgt_type:
        if src_type == "continuous_variable":
            
            if src_composite_codes != tgt_composite_codes:
                details = {
                    "description": "Source and target variable have different semantic context; manual review required for transformation."
                    
                }
                return finalize(details, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)

            if src_unit and tgt_unit and src_unit != tgt_unit:
                details = {
                    "description": (
                        f"Unit conversion in dataset required from {src_unit} to {tgt_unit} (or vice versa)."
                    )
                }
                return finalize(details, "Compatible Match", src_info, tgt_info, mapping_relation)

            # Same type and compatible units (or units missing on one side)
            if mapping_relation not in {"skos:exactMatch", "skos:closeMatch"}:
                details = {
                "description": "Variables are broadly similar continuous types; manual review required to confirm exact transformation."
                
                }
                return finalize(details, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)
           
            details = {
                "description": "No transformation required. Continuous types and units match."
            }
            return finalize(details, "Identical Match", src_info, tgt_info, mapping_relation)

        # categorical/qualitative — align by labels
        elif src_type == "qualitative_variable":
            if src_data_type == tgt_data_type:
                if mapping_relation not in {"skos:exactMatch", "skos:closeMatch"}:
                    details = {
                    "description": "Variables are broadly similar qualitative types; manual review required to confirm exact transformation."
                    
                    }
                    return finalize(details, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)
            
                else:
                    return finalize(
                        {"description": "The qualitative variables share semantics and statistics."},
                        "Identical Match",
                        src_info,
                        tgt_info,
                        mapping_relation,
                    )
            else:
                details = {
                    "description": "Qualitative variables share semantics and statistics but differ in data type."
                }
                return finalize(details, "Compatible Match", src_info, tgt_info, mapping_relation)
            
        else:
            
            if label_mapping["has_overlap"]:
                if label_mapping["identical"]:
                    if mapping_relation not in {"skos:exactMatch", "skos:closeMatch"}:
                        details = {
                        "description": "Variables are broadly similar categorical types; manual review required to confirm exact transformation."
                        
                        }
                        return finalize(details, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)
                
                    else:
                        details = {
                        "description": "Categorical values are identical and aligned by standard labels.",
                        "categorical_mapping": label_mapping["mapping_str"],
                        "standard_labels": "; ".join(sorted(label_mapping["overlap_labels"])),
                    }
                        return finalize(details, "Identical Match", src_info, tgt_info, mapping_relation)
                else:
                    details = {
                        "description": "Original categorical values differ but overlap on standard labels.",
                        "categorical_mapping": label_mapping["mapping_str"],
                        "unmapped_source_labels": label_mapping["unmapped_source_labels"],
                        "unmapped_target_labels": label_mapping["unmapped_target_labels"],
                    }
                    return finalize(details, "Compatible Match", src_info, tgt_info, mapping_relation)
            else:
                src_labels_raw = (src_info.get('original_categories', '') or '')
                tgt_labels_raw = (tgt_info.get('original_categories', '') or '')
                src_labels = "; ".join(
                    sorted(
                        {v.strip() for v in src_labels_raw.split(';') if v.strip()}
                    )
                )
                tgt_labels = "; ".join(
                    sorted(
                        {v.strip() for v in tgt_labels_raw.split(';') if v.strip()}
                    )
                )
                details = {
                    "description": "No overlap in standard labels between categorical values; mapping/review required.",
                    "source_labels": src_labels,
                    "target_labels": tgt_labels,
                }
                return finalize(details, "Partial Match (Tentative)",src_info, tgt_info, mapping_relation)

         
    # --- binary vs multi-class
    elif ((src_type == "binary_class_variable" and tgt_type == "multi_class_variable") or
        (src_type == "multi_class_variable" and tgt_type == "binary_class_variable")):
       
        if domain in ["drug_exposure", "drug_era", "medication"]:
            msg = (
                "Multi-class <-> binary conversion requires justification of information loss. "
                "For drug-related variables, consider therapy details and clinical context."
            )
        else:
            msg = (
                "Multi-class ↔ binary conversion requires justification of information loss "
                "for the specific research question."
            )
       
        overlap_labels = set(l.lower() for l in label_mapping["overlap_labels"])

        src_labels = overlap_labels | {
            x.strip().lower()
            for x in (src_info.get('categories_codes', '') or '').split(';')
            if x.strip()
        }
        tgt_labels = overlap_labels | {
            x.strip().lower()
            for x in (tgt_info.get('categories_codes', '') or '').split(';')
            if x.strip()
        }

       # Binary set ⊆ multi-class set
        source_is_binary_subset = (
            src_type == "binary_class_variable" and
            len(src_labels) > 0 and
            src_labels.issubset(tgt_labels)
        )
        target_is_binary_subset = (
            tgt_type == "binary_class_variable" and
            len(tgt_labels) > 0 and
            tgt_labels.issubset(src_labels)
        )
        if source_is_binary_subset or target_is_binary_subset:
            categorical_mapping = (
                label_mapping["mapping_str"]
                if label_mapping["has_overlap"] and label_mapping["mapping_str"]
                else None
            )
            if categorical_mapping:
                details = {
                    "description": (
                        "Binary categories are a subset of multi-class categories; "
                        "expansion/aggregation of categories is possible."
                    ),
                    "categorical_mapping": categorical_mapping,
                }
                return finalize(details, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)
            else:
                details = {
                    "description": msg,
                    "categorical_mapping": None,
                    "unmapped_source_labels": label_mapping["unmapped_source_labels"],
                    "unmapped_target_labels": label_mapping["unmapped_target_labels"],
                }
                return finalize(details, "Not Applicable",  src_info, tgt_info, mapping_relation)

        # no subset relation at all
        details = {
            "description": msg,
            "categorical_mapping": label_mapping.get("mapping_str"),
            "unmapped_source_labels": label_mapping.get("unmapped_source_labels"),
            "unmapped_target_labels": label_mapping.get("unmapped_target_labels"),
        }
        return finalize(details, "Not Applicable", src_info, tgt_info, mapping_relation)
     

    # CASE 3: continuous vs categorical
    
    
    elif ((src_type == "continuous_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}) or 
        (tgt_type == "continuous_variable" and src_type in {"binary_class_variable", "multi_class_variable"})):
        
        if src_data_type == "datetime" or tgt_data_type == "datetime":
            return finalize({"description": "Unable to align datetime to binary/multi-class indicator."}, "Not Applicable", src_info, tgt_info, mapping_relation)
        else:
            if domain not in ["drug_exposure", "drug_era"]:
                msg = ("Discretize continuous variable to categories only if information loss is minimal (e.g., classification).") 
            else:
                msg = "Avoid continuous→categorical harmonization for drug-related variables unless strongly justified."
            status = "Not Applicable" if domain in ["drug_exposure", "drug_era"] else "Partial Match (Tentative)"
            return finalize({"description": msg}, status, src_info, tgt_info,mapping_relation)
    
    # --- qualitative vs categorical/continuous
    elif (src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "qualitative_variable"):
        return finalize({"description": "Map structured categorical codes to consistent text labels; normalize values."}, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)
    
    elif (src_type == "qualitative_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}):
        return finalize({"description": "Normalize qualitative text to standard categories; encode to labels/codes."}, "Partial Match (Proximate)", src_info, tgt_info, mapping_relation)
    
    elif ((src_type == "qualitative_variable" and tgt_type == "continuous_variable") or (src_type == "continuous_variable" and tgt_type == "qualitative_variable")):
        if domain in {"person"} or (not src_unit and not tgt_unit):
            return finalize({"description": "A qualitative variable and a continuous variable can be merged if underlying semantics align."}, "Partial Match (Proximate)", src_info, tgt_info,mapping_relation)
        else:
            return finalize({"description": "Merging qualitative and continuous variables (e.g. with units) requires strong justification of information loss."}, "Partial Match (Tentative)", src_info, tgt_info, mapping_relation)

    return finalize({"description": "No specific transformation rule available."}, "Not Applicable", src_info, tgt_info, mapping_relation)


def get_member_studies(study_name: str) -> List[str]:
    query = f"""PREFIX dc:   <http://purl.org/dc/elements/1.1/>
                    PREFIX obi:  <http://purl.obolibrary.org/obo/obi.owl/>
                    PREFIX ro:   <http://purl.obolibrary.org/obo/ro.owl/>
                    PREFIX iao:  <http://purl.obolibrary.org/obo/iao.owl/>

                    SELECT DISTINCT ?related_study
                    WHERE {{
                    GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {{
                        # anchor the index study
                        ?study_design  dc:identifier ?study_name.
                        VALUES (?study_name) {{ ("{study_name}") }} 
                        # membership in BOTH directions
                        {{
                        ?study_design obi:has_member ?related_study .
                        }} UNION {{
                        ?related_study obi:has_member ?study_design .
                        }} UNION {{
                         ?study_design obi:member_of ?related_study .
                        }} UNION {{
                        ?related_study obi:member_of ?study_design .
                        }}
                        # ensure the target is a study and not the same as the anchor
                        FILTER(?related_study != ?study_design)
                    }}
                    }}
            """
            
    query_endpoint = SPARQLWrapper(settings.query_endpoint)
    query_endpoint.setReturnFormat(JSON)
    query_endpoint.setQuery(query)
    results = query_endpoint.query().convert()
    studies_uris = []
    if results["results"]["bindings"]:
        for result in results["results"]["bindings"]:
            related_study_uri = result["related_study"]["value"].split("/")[-2]
            studies_uris.append(related_study_uri)
            
    return studies_uris
    
    


def parse_joined_string(input_str: str) -> List[str]:
    """
    Parses a string that may be either:
    - a key-value categorical string like '1=No|2=Yes' or '1="mmol|l"|2="g|dl"'
    - a plain joined string like '"mg|dl"|mmol'
    
    Returns a list of extracted values, handling quoted values and internal pipes correctly.
    """
    if not has_real_value(input_str) or not isinstance(input_str, str):
        return []

    # Case 1: If the string has key=value pattern
    if re.search(r'\d+\s*=', input_str):
        # Match key=value pairs with quoted or unquoted values
        pattern = r'\d+\s*=\s*"[^"]*"|\d+\s*=\s*[^|]+'
        matches = re.findall(pattern, input_str)
        values = [
            re.sub(r'^\d+\s*=\s*', '', match).strip().strip('"')
            for match in matches if match.strip()
        ]
    else:
        # Case 2: Just split by top-level pipes, respecting quotes
        pattern = r'"[^"]*"|[^|"]+'
        matches = re.findall(pattern, input_str)
        values = [match.strip().strip('"') for match in matches if match.strip()]

    return values



def compare_with_fuzz(text1: str, text2: str):
    similarity = fuzz.ratio(text1, text2) / 100

    return similarity

def delete_existing_triples(graph_uri: str | URIRef, subject="?s", predicate="?p"):
    print(f"deleting existing triples from the graph={graph_uri}")
    if graph_exists(graph_uri):
        
        print(f"Graph exists: {graph_uri}")
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        DELETE WHERE {{
            GRAPH <{graph_uri!s}> {{ {subject} {predicate} ?o . }}
        }}
        """
        # print(f"Query = {query}")
        
        query_endpoint = SPARQLWrapper(settings.update_endpoint)
        query_endpoint.setMethod("POST")
        query_endpoint.setRequestMethod("urlencoded")
        query_endpoint.setQuery(query)
        results =query_endpoint.query()
        response_status = results.response.status
        # response_content = results.response.read()
        print(f"graph deletion status code: {response_status}")
    else:
        print(f"Graph does not exist: {graph_uri}")
        
def graph_exists(graph_uri: str | URIRef):
    query = f"""
    ASK WHERE {{
        GRAPH <{graph_uri!s}> {{ ?s ?p ?o }}
    }}
    """
    # print(f"Checking if graph exists: {query}")
    query_endpoint = SPARQLWrapper(settings.query_endpoint)
    query_endpoint.setReturnFormat(JSON)
    query_endpoint.setQuery(query)
    results = query_endpoint.query().convert()
    # print(f"Graph exists: {results['boolean']}")
    return results['boolean']


def check_triple_exists(graph_uri: str | URIRef, subject: URIRef, predicate: URIRef, obj: URIRef | Literal):
    query = f"""
    ASK WHERE {{
        GRAPH <{graph_uri!s}> {{ <{subject}> <{predicate}> {f'<{obj}>' if isinstance(obj, URIRef) else f'"{obj}"'} }}
    }}
    """
    # print(f"Checking if triple exists: {query}")
    query_endpoint = SPARQLWrapper(settings.query_endpoint)
    query_endpoint.setReturnFormat(JSON)
    query_endpoint.setQuery(query)
    results = query_endpoint.query().convert()
    # print(f"Triple exists: {results['boolean']}")
    return results['boolean']
    


def add_triples_to_graph(graph: Graph, triples: list, graph_context: URIRef = None) -> None:
    """
    Adds a list of triples to the graph, optionally under a specific graph context.

    :param graph: RDF Graph
    :param triples: List of triples (subject, predicate, object)
    :param graph_context: Specific graph/context to add the triples to
    """
    for subj, pred, obj in triples:
        if graph_context:
            graph.add((subj, pred, obj, graph_context))
            print(f"Added triple: {subj} {pred} {obj} in graph {graph_context}")
        else:
            graph.add((subj, pred, obj))
            print(f"Added triple: {subj} {pred} {obj}")
    return graph



def save_graph_to_trig_file(graph_data, file_path):
    """
    Save RDFLib Graph data to a TRiG file under a specific named graph.
    
    :param graph_data: An RDFLib Graph containing the query results
    :param file_path: Path to the TRiG file to save data
    :param graph_uri: The named graph URI to wrap the triples under
    """
    try:
        # Serialize the graph into TriG format, placing data inside the specified named graph block
        trig_data = graph_data.serialize(format='trig')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Write the TRiG data to a file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(trig_data)
        print(f"Graph data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving graph to TRiG file: {e}")



# its for graphDB
# def publish_graph_to_endpoint(g: Graph, graph_uri: str | None = None) -> bool:
#     """Insert the graph into the triplestore endpoint."""
#     # url = f"{settings.sparql_endpoint}/store?{graph_uri}"
#     url = f"{settings.sparql_endpoint}/rdf-graphs/{graph_uri}"
#     print(f"URL: {url}")
#     headers = {"Content-Type": "application/trig"}
#     g.serialize("/tmp/upload-data.trig", format="trig")
#     with open("/tmp/upload-data.trig", "rb") as file:
#         response = requests.post(url, headers=headers, data=file, timeout=300)
#         print(f"Response: {response}")
#     # NOTE: Fails when we pass RDF as string directly
#     # response = requests.post(url, headers=headers, data=graph_data)
#     # Check response status and print result
#     if not response.ok:
#         print(f"Failed to upload data: {response.status_code}, {response.text}")
#     return response.ok

# for oxigraph
def publish_graph_to_endpoint(g: Graph, graph_uri: str | None = None) -> bool:
    """Insert the graph into the triplestore endpoint."""
    # url = f"{settings.sparql_endpoint}/store?{graph_uri}"
    url = f"{settings.sparql_endpoint}/store"
    if graph_uri:
        url += f"?graph={graph_uri}"
        print(f"URL: {url}")
    headers = {"Content-Type": "application/trig"}
    g.serialize("/tmp/upload-data.trig", format="trig")
    with open("/tmp/upload-data.trig", "rb") as file:
        response = requests.post(url, headers=headers, data=file, timeout=300)
        print(f"Response: {response}")
    # NOTE: Fails when we pass RDF as string directly
    # response = requests.post(url, headers=headers, data=graph_data)
    # Check response status and print result
    if not response.ok:
        print(f"Failed to upload data: {response.status_code}, {response.text}")
    return response.ok

def find_related_studies(study_name:str) -> list[str]:
    query = f"""

    PREFIX dc:  <http://purl.org/dc/elements/1.1/>
    PREFIX ro:  <http://purl.obolibrary.org/obo/ro.owl/>

    SELECT DISTINCT ?parent_name
    WHERE {{
    GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {{
        VALUES (?q) { (study_name) }
        ?design dc:identifier ?study_name .
        FILTER(LCASE(STR(?study_name)) = LCASE(?q))

        # Only true parents of the design (protocol is not linked this way)
        ?design (ro:has_part|ro:part_of) ?parent_design .
        ?parent_design dc:identifier ?parent_name .
    }}
    }}
    """
    sparql = SPARQLWrapper(settings.query_endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    results = sparql.query().convert()
    
    related_studies = []
    if results['results']['bindings']:
        for binding in results['results']['bindings']:
            related_studies.append(binding['parent_name']['value'])
    return related_studies
def load_dictionary( filepath=None) -> pd.DataFrame:
        """Loads the input dataset."""
        if filepath.endswith('.sav'):
            df_input = pd.read_spss(filepath)
            # Optionally save to Excel if needed
         
        elif filepath.endswith('.csv'):
            try:
                df_input = pd.read_csv(filepath, dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                df_input = pd.read_csv(filepath, dtype=str, keep_default_na=False, encoding="latin-1")
        elif filepath.endswith('.xlsx'):
            df_input = pd.read_excel(filepath, sheet_name=0)
        else:
            raise ValueError("Unsupported file format.")
        if not df_input.empty:
            return df_input
        else:
            return None
   
   
def load_file( filepath=None) -> pd.DataFrame:
        """Loads the input dataset."""
        if filepath.endswith('.sav'):
            df_input = pd.read_spss(filepath)
            # Optionally save to Excel if needed
         
        elif filepath.endswith('.csv'):
            df_input = pd.read_csv(filepath, low_memory=False)
        elif filepath.endswith('.xlsx'):
            df_input = pd.read_excel(filepath, sheet_name=0)
        else:
            raise ValueError("Unsupported file format.")
        if not df_input.empty:
            return df_input
        else:
            return None
   
         
def export_hierarchy_to_excel(hierarchy: dict, label_map: dict, output_file: str):
 
 rows = []
 for child_id, relations in hierarchy.items():
    child_label = label_map.get(child_id, f"OMOP ID={child_id}")
    for parent_id in relations["parents"]:
        parent_label = label_map.get(parent_id, f"OMOP ID={parent_id}")
        rows.append({
            "parent_id": parent_id,
            "parent_label": parent_label,
            "child_id": child_id,
            "child_label": child_label,
        })
 df = pd.DataFrame(rows, columns=["parent_id", "parent_label", "child_id", "child_label"])
 df = df.groupby(["parent_id", "parent_label"], dropna=False, as_index=False).agg({
            "child_id": lambda x: list(x),
            "child_label": lambda x: list(x),
        })
 df.to_excel(output_file, index=False)
 print(f"[INFO] Hierarchy exported to {output_file}")
 
 
 

def create_code_uri(code:str, cohort_uri: URIRef) -> URIRef:
    code_only = code.split(":")[-1]
    code_only_encoded = quote(code_only, safe='')
    if 'snomed' in code or 'snomedct' in code:
        code_uri = URIRef(f"{OntologyNamespaces.SNOMEDCT.value}{code_only_encoded}")
    elif 'icd9' in code:
        code_uri = URIRef(f"{OntologyNamespaces.ICD9.value}{code_only_encoded}")
    elif 'icd10' in code:
        code_uri = URIRef(f"{OntologyNamespaces.ICD10.value}{code_only_encoded}")
    elif 'loinc' in code:
        code_uri = URIRef(f"{OntologyNamespaces.LOINC.value}{code_only_encoded}")
    elif 'ucum' in code:
        code_uri = URIRef(f"{OntologyNamespaces.UCUM.value}{code_only_encoded}")
    elif 'rxnorm' in code:
        code_uri = URIRef(f"{OntologyNamespaces.RXNORM.value}{code_only_encoded}")
    elif 'atc' in code:
        code_uri = URIRef(f"{OntologyNamespaces.ATC.value}{code_only_encoded}")
    elif 'omop' in code:
        code_uri = URIRef(f"{OntologyNamespaces.OMOP.value}{code_only_encoded}")
    else:
        code_uri = URIRef(f"{cohort_uri}/{code_only_encoded}")
    return code_uri
    
def insert_graph_into_named_graph(g_new: Graph, graph_uri: str, chunk_size: int = 500) -> None:
    """
    Append triples from g_new into an existing named graph using SPARQL UPDATE INSERT DATA.
    Does NOT delete/replace existing data.

    :param g_new: rdflib.Graph containing only the new triples to insert
    :param graph_uri: target named graph URI (string)
    :param chunk_size: number of triples per INSERT batch (avoid huge updates)
    """
    # Convert the new triples to N-Triples lines (safe to embed in SPARQL)
    nt_bytes = g_new.serialize(format="nt")
    nt_str = nt_bytes.decode("utf-8") if isinstance(nt_bytes, (bytes, bytearray)) else nt_bytes

    lines = [ln for ln in nt_str.splitlines() if ln.strip()]
    if not lines:
        print("No new triples to insert.")
        return

    sparql = SPARQLWrapper(settings.update_endpoint)
    sparql.setMethod("POST")
    sparql.setRequestMethod("urlencoded")

    # Chunk the payload into multiple INSERT DATA blocks
    for i in range(0, len(lines), chunk_size):
        block = "\n".join(lines[i:i+chunk_size])
        query = f"""
        INSERT DATA {{
          GRAPH <{graph_uri}> {{
            {block}
          }}
        }}
        """
        sparql.setQuery(query)
        res = sparql.query()
        print(f"Inserted {min(i+chunk_size, len(lines))}/{len(lines)} triples; HTTP {res.response.status}")


def setup_logger(log_file: str):
    logger = logging.getLogger(f"cohortvarlinker.{Path(log_file).stem}")
    if logger.handlers:                      # already configured this run
        return logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False                 # do not re-emit via root
    return logger