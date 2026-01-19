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
from typing import Dict, Any




from datetime import datetime

def day_month_year(date_str: str) -> tuple:
    formats = [
        "%d-%m-%Y", "%Y-%m-%d", "%m-%Y", "%Y/%m/%d", "%m/%Y","%Y/%m", "%d/%m/%Y", "%m/%d/%Y", "%B %Y", "%Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return (str(dt.day).zfill(2), str(dt.month).zfill(2), str(dt.year))
        except:
            continue
    return None
class OntologyNamespaces(Enum):
    CMEO = Namespace("https://w3id.org/CMEO/")
    OMOP = Namespace("http://omop.org/OMOP/")
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
def determine_var_uri(g, cohort_id, var_name,multi_class_categorical, binary_categorical, data_type=None):
    # cohort_uri = get_cohort_uri(cohort_id)
    var_uri = get_var_uri(cohort_id, var_name)
    if var_name in binary_categorical:
        statistical_type_uri =  URIRef(var_uri + "/binary_class_variable")
        statistical_type = "binary_class_variable"
        
    elif var_name in multi_class_categorical:
        statistical_type_uri =  URIRef(var_uri + "/multi_class_variable")
        statistical_type = "multi_class_variable"
    elif data_type  and data_type in  ["str", "datetime"]:
        statistical_type_uri =  URIRef(var_uri + "/qualitative_variable")
        statistical_type = "qualitative_variable"
    else:
        statistical_type_uri =  URIRef(var_uri + "/continuous_variable")
        statistical_type = "continuous_variable"
    return statistical_type_uri,statistical_type


def extract_tick_values(texts: str) -> list[float]:
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
        # if pd.notna(value) and value:
            if pd.notna(value) and value != "":
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
            if (('event' in src_visit.lower() and  'baseline' in tgt_visit.lower()) or ('baseline' in src_visit.lower() and  'event' in tgt_visit.lower())):
                status = lower_stat_by_1(status)
            result_dict["description"] = desc + (
                "Temporal context differs between source and target at metadata level.")
            # status = lower_stat_by_1(status)
            return result_dict, mapping_relation, status

    # Exact match
    elif src_codes == tgt_codes:
        # Optionally, store what matched (can help debugging)
        # if src_codes:
            # result_dict.setdefault("additional_context_comparison", "exact")
            # result_dict.setdefault("additional_context_source",src_codes)
            # result_dict.setdefault("additional_context_target",tgt_codes)
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
        print(f"Adjusting for additional context: src_codes={src_codes}: {src_info}, tgt_codes={tgt_codes}: {tgt_info}")
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
        # result_dict["additional_context_comparison"] = comparison_kind
        # result_dict["additional_context_source"] = src_codes
        # result_dict["additional_context_target"] = tgt_codes
        if src_visit != tgt_visit:
            result_dict["description"] += " Temporal context also differs between source and target at metadata level."
            # status = lower_stat_by_1(status)

        return result_dict,  mapping_relation, status


def execute_query(query: str) -> Iterable[Dict[str, Any]]:
    sparql = SPARQLWrapper(settings.query_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()



   
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
            details = {
                "description": "No transformation required. Continuous types and units match."
            }
            return finalize(details, "Identical Match", src_info, tgt_info, mapping_relation)

        # categorical/qualitative — align by labels
        elif src_type == "qualitative_variable":
            if src_data_type == tgt_data_type:
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
     
      # -------------------------------------------------------------------------
    # CASE 3: continuous vs categorical
    # -------------------------------------------------------------------------
    
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


def get_member_studies(study_name: str) -> URIRef | None:
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
    
    
    

# def apply_rules(domain, src_info, tgt_info):
#     def parse_categories(cat_str):
#         if pd.notna(cat_str) and cat_str not in [None, '']:
#             return [c.strip().lower() for c in str(cat_str).split(";")]
#         return []

#     def map_category_to_code(code_str:list, label_str:list):
#         codes = [c.strip() for c in code_str]
#         labels = [l.strip().lower() for l in label_str]
#         # Returns a dict: label → code
#         return {code: label for code, label in zip(codes, labels) if code and label}
    
#     print(f"src_info: {src_info}  tgt_info: {tgt_info}")
#     src_var_name = src_info.get('var_name', '').lower()
#     tgt_var_name = tgt_info.get('var_name', '').lower()
#     src_type = str(src_info.get('stats_type')).lower() if pd.notna(src_info.get('stats_type')) and src_info.get('stats_type') not in [None, ''] else None
#     tgt_type = str(tgt_info.get('stats_type')).lower() if pd.notna(tgt_info.get('stats_type')) and tgt_info.get('stats_type') not in [None, ''] else None

#     src_unit = str(src_info.get('unit', '').lower() if pd.notna(src_info.get('unit', '')) else None)
#     tgt_unit = str(tgt_info.get('unit', '').lower() if pd.notna(tgt_info.get('unit', '')) else None)
#     src_data_type = str(src_info.get('data_type', '').lower() if pd.notna(src_info.get('data_type', '')) else None)
#     tgt_data_type = str(tgt_info.get('data_type', '').lower() if pd.notna(tgt_info.get('data_type', '')) else None)
#     src_categories = parse_categories(src_info.get('categories_codes', ''))
#     tgt_categories = parse_categories(tgt_info.get('categories_codes', ''))
#     original_src_categories = parse_categories(src_info.get('original_categories', ''))
#     original_tgt_categories = parse_categories(tgt_info.get('original_categories', ''))
 
#     valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable", "qualitative_variable"}
#     if (src_type not in valid_types or tgt_type not in valid_types) or (src_type is None or tgt_type is None):
#         if "derived" not in src_var_name and "derived" not in tgt_var_name:
#             return {
#                 "description": "Transformation not applicable (invalid or missing statistical type)."
#             }, "Not Applicable"
#         else:
#             return {
#                 "description": "Derived variable - Transformation depends on derivation logic."
#             }, "Compatible Match"
#     elif src_type == tgt_type:
#         if src_type == "continuous_variable":
#             if src_unit and tgt_unit and src_unit != tgt_unit:
#                 # if (src_unit in ["mg", "milligram"] and tgt_unit in ["%", "percent"]) or \
#                 #    (src_unit in ["%", "percent"] and tgt_unit in ["mg", "milligram"]):
#                 return {
#                         "description": "Unit conversion in dataset required from {src_unit} to {tgt_unit} or vice versa.",
#                     }, "Compatible Match"
#                 # return {
#                 #     "description": "Unit conversion required. Evaluate based on research question."
#                 # }
#             return {
#                 "description": "No transformation required. Continuous types and units match."
#             }, "Identical Match"
        
#         else:
            
#             src_pairs = map_category_to_code(src_categories, original_src_categories)
#             tgt_pairs = map_category_to_code(tgt_categories, original_tgt_categories)

#             print(f"src_label_to_code: {src_pairs}")
#             print(f"tgt_label_to_code: {tgt_pairs}")
#             # Try to match on label (case-insensitive)
#             common_codes = set(src_pairs) & set(tgt_pairs)

#             if common_codes:
#                 if set(original_src_categories) == set(original_tgt_categories):
#                     mapping_str = [f"{sl} ↔ {tl}" for sl, tl in zip(original_src_categories, original_tgt_categories)]
#                     print(f"mapping_str: {mapping_str}")
#                     return {
#                         "description": "Categorical values are identical and aligned by standard codes.",
#                         "categorical_mapping": "; ".join(mapping_str),
#                         "standard_codes": "; ".join(common_codes) if common_codes else "No common codes found",
#                     }, "Identical Match"
#                 else:
#                     mapping_str = [f"{sl} -> {tl}" for sl, tl in zip(src_pairs.values(), tgt_pairs.values())]
#                     print(f"mapping_str: {mapping_str}")
#                     return {
#                         "description": f"The original categorical values are not similar however aligned by standard codes.",
#                         "categorical_mapping": "; ".join(mapping_str),
#                         "standard_codes": "; ".join(common_codes) if common_codes else "No common codes found",
#                     }, "Compatible Match"
#             else:
                
#                 return {
#                     "description": "Found no matching standard labels for categories values. Mapping/review is required for harmonization.",
#                     "source_categories": "; ".join(src_categories),
#                     "target_categories": "; ".join(tgt_categories)
#                 }, "Partial Match (Tentative)"
           
#     elif (
#         (src_type == "binary_class_variable" and tgt_type == "multi_class_variable") or
#         (src_type == "multi_class_variable" and tgt_type == "binary_class_variable")
#     ):
#         print(f"src_categories: {src_categories} and tgt_categories: {tgt_categories} for vars {src_var_name} and {tgt_var_name}")
#         msg = (
#             "multi-class to binary class requires justification of information loss for specific research question. For drug-related variables, consider therapy details and surrounding context."
#             if domain in ["drug_exposure", "drug_era"]
#             else "Both variables don't share similar categories. The conversion of multi-class to binary class (e.g. yes/no) requires justification of information loss for specific research question."
#         )
        
#         # src_codes = map_category_to_code(src_categories, original_src_categories)
#         # tgt_codes = map_category_to_code(tgt_categories, original_tgt_categories)
        
#         # check if src code exists in tgt codes
        
#          # Check if all source categories exist in target categories or vice versa
#         if (src_type == "multi_class_variable" and tgt_type == "binary_class_variable"):
#             proximate_ok = set(tgt_categories).issubset(set(src_categories))
#         elif (src_type == "binary_class_variable" and tgt_type == "multi_class_variable"):
#             proximate_ok = set(src_categories).issubset(set(tgt_categories))
#         else:
#             proximate_ok = False
#         if proximate_ok:
#             return {
#                 "description": f"Both variables share some categories. Expand binary categories in one variable by extending with additional categories from other variables",
#                 "source_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(src_categories, original_src_categories)]),
#                 "target_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(tgt_categories, original_tgt_categories)])
#             }, "Partial Match (Proximate)"
#         else:   
#             return {
#                 "description": msg,
#                 "source_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(src_categories, original_src_categories)]),
#                 "target_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(tgt_categories, original_tgt_categories)])
#             }, "Partial Match (Tentative)"

#     elif (
#         (src_type == "continuous_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}) or
#         (tgt_type == "continuous_variable" and src_type in {"binary_class_variable", "multi_class_variable"})
#     ):
#         if src_data_type == "datetime" or tgt_data_type == "datetime":
#             return {
#                 "description": "Unable to align datetime to binary indicator (presence/absence)",
#                 "source_categories": "; ".join(src_categories),
#                 "target_categories": "; ".join(tgt_categories),
#             }, "Not Applicable"
#         msg = (
#             "Discretize continuous variable to categories. Acceptable only if information loss is minimal. Represent as: (1) binary flag for event presence, (2) category of event type."
#             if domain not in ["drug_exposure", "drug_era"]
#             else "Harmonization may not be possible for drug-related continuous to categorical mappings. Review medication normalization depending on research question."
#         )
#         match = "Not Applicable" if domain in ["drug_exposure", "drug_era"] else "Partial Match (Tentative)"
#         return {
#             "description": msg,
#             "source_categories": "; ".join(src_categories),
#             "target_categories": "; ".join(tgt_categories)
#         }, match

#     elif src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "qualitative_variable":
#         return {
#             "description": (
#                 "Map structured categorical codes to consistent/unique text labels. Requires normalization."
#                 "Only suitable for qualitative fields with finite, structured values."
#             ),
#             "source_categories": "; ".join(src_categories),
#             "target_categories": ""
#         }, "Partial Match (Tentative)"

#     elif src_type == "qualitative_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}:
#         return {
#             "description": (
#                 "Map qualitative text to standard categories. Normalize and encode."
#                 "Applicable only if text values are consistently structured."
#             ),
#             "source_categories": "",
#             "target_categories": "; ".join(tgt_categories)
#         }, "Partial Match (Tentative)"
#     elif src_type == "qualitative_variable" and tgt_type == "continuous_variable" or src_type == "continuous_variable" and tgt_type == "qualitative_variable":
#         return {
#             "description": (
#                 "Map qualitative text to binary indicators."
#                 "Applicable only if text values are consistently structured."
#             ),
#             "source_categories": "",
#             "target_categories": "; ".join(tgt_categories)
#         }, "Partial Match (Tentative)"
#     return {
#         "description": "No specific transformation rule available."
#     }, "Not Applicable"


# def apply_rules(domain, src_info, tgt_info):
#     def parse_categories(cat_str):
#         if pd.notna(cat_str) and cat_str not in [None, '']:
#             return [c.strip().lower() for c in str(cat_str).split(";")]
#         return []

#     def map_category_to_code(code_str:list, label_str:list):
#         codes = [c.strip() for c in code_str]
#         labels = [l.strip().lower() for l in label_str]
#         # Returns a dict: label → code
#         return {code: label for code, label in zip(codes, labels) if code and label}
    
#     print(f"src_info: {src_info}  tgt_info: {tgt_info}")
#     src_var_name = src_info.get('var_name', '').lower()
#     tgt_var_name = tgt_info.get('var_name', '').lower()
#     src_type = str(src_info.get('stats_type')).lower() if pd.notna(src_info.get('stats_type')) and src_info.get('stats_type') not in [None, ''] else None
#     tgt_type = str(tgt_info.get('stats_type')).lower() if pd.notna(tgt_info.get('stats_type')) and tgt_info.get('stats_type') not in [None, ''] else None

#     src_unit = str(src_info.get('unit', '').lower() if pd.notna(src_info.get('unit', '')) else None)
#     tgt_unit = str(tgt_info.get('unit', '').lower() if pd.notna(tgt_info.get('unit', '')) else None)
#     src_data_type = str(src_info.get('data_type', '').lower() if pd.notna(src_info.get('data_type', '')) else None)
#     tgt_data_type = str(tgt_info.get('data_type', '').lower() if pd.notna(tgt_info.get('data_type', '')) else None)
#     src_categories = parse_categories(src_info.get('categories_codes', ''))
#     tgt_categories = parse_categories(tgt_info.get('categories_codes', ''))
#     original_src_categories = parse_categories(src_info.get('original_categories', ''))
#     original_tgt_categories = parse_categories(tgt_info.get('original_categories', ''))
 
#     valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable", "qualitative_variable"}
#     if src_type not in valid_types or tgt_type not in valid_types:
#         if "derived" not in src_var_name and "derived" not in tgt_var_name:
#             return {
#                 "description": "Transformation not applicable (invalid or missing statistical type)."
#             }, "Not Applicable"

#     elif src_type == tgt_type:
#         if src_type == "continuous_variable":
#             if src_unit and tgt_unit and src_unit != tgt_unit:
#                 # if (src_unit in ["mg", "milligram"] and tgt_unit in ["%", "percent"]) or \
#                 #    (src_unit in ["%", "percent"] and tgt_unit in ["mg", "milligram"]):
#                 return {
#                         "description": "Unit conversion in dataset required from {src_unit} to {tgt_unit} or vice versa.",
#                     }, "Compatible Match"
#                 # return {
#                 #     "description": "Unit conversion required. Evaluate based on research question."
#                 # }
#             return {
#                 "description": "No transformation required. Continuous types and units match."
#             }, "Identical Match"
#         elif set(src_categories) == set(tgt_categories):
#             src_pairs = map_category_to_code(src_categories, original_src_categories)
#             tgt_pairs = map_category_to_code(tgt_categories, original_tgt_categories)

#             print(f"src_label_to_code: {src_pairs}")
#             print(f"tgt_label_to_code: {tgt_pairs}")
#             # Try to match on label (case-insensitive)
#             common_codes = set(src_pairs) & set(tgt_pairs)

#             if common_codes:
#                 if set(original_src_categories) == set(original_tgt_categories):
#                     mapping_str = [f"{sl} ↔ {tl}" for sl, tl in zip(original_src_categories, original_tgt_categories)]
#                     print(f"mapping_str: {mapping_str}")
#                     return {
#                         "description": "Categorical values are identical and aligned by standard codes.",
#                         "categorical_mapping": "; ".join(mapping_str),
#                         "standard_codes": "; ".join(common_codes) if common_codes else "No common codes found",
#                     }, "Identical Match"
#                 else:
#                     mapping_str = [f"{sl} -> {tl}" for sl, tl in zip(src_pairs.values(), tgt_pairs.values())]
#                     print(f"mapping_str: {mapping_str}")
#                     return {
#                         "description": f"The original categorical values are not similar however aligned by standard codes.",
#                         "categorical_mapping": "; ".join(mapping_str),
#                         "standard_codes": "; ".join(common_codes) if common_codes else "No common codes found",
#                     }, "Compatible Match"
#             else:
                
#                 return {
#                     "description": "Found no matching standard labels for categories values. Mapping/review is required for harmonization.",
#                     "source_categories": "; ".join(src_categories),
#                     "target_categories": "; ".join(tgt_categories)
#                 }, "Partial Match (Tentative)"
           
#     elif (
#         (src_type == "binary_class_variable" and tgt_type == "multi_class_variable") or
#         (src_type == "multi_class_variable" and tgt_type == "binary_class_variable")
#     ):
#         print(f"src_categories: {src_categories} and tgt_categories: {tgt_categories} for vars {src_var_name} and {tgt_var_name}")
#         msg = (
#             "multi-class to binary class requires justification of information loss for specific research question. For drug-related variables, consider therapy details and surrounding context."
#             if domain in ["drug_exposure", "drug_era"]
#             else "Both variables don't share similar categories. The conversion of multi-class to binary class (e.g. yes/no) requires justification of information loss for specific research question."
#         )
        
#         # src_codes = map_category_to_code(src_categories, original_src_categories)
#         # tgt_codes = map_category_to_code(tgt_categories, original_tgt_categories)
        
#         # check if src code exists in tgt codes
        
#          # Check if all source categories exist in target categories or vice versa
#         if (src_type == "multi_class_variable" and tgt_type == "binary_class_variable"):
#             proximate_ok = set(tgt_categories).issubset(set(src_categories))
#         elif (src_type == "binary_class_variable" and tgt_type == "multi_class_variable"):
#             proximate_ok = set(src_categories).issubset(set(tgt_categories))
#         else:
#             proximate_ok = False
#         if proximate_ok:
#             return {
#                 "description": f"Both variables share some categories. Expand binary categories in one variable by extending with additional categories from other variables",
#                 "source_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(src_categories, original_src_categories)]),
#                 "target_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(tgt_categories, original_tgt_categories)])
#             }, "Partial Match (Proximate)"
#         else:   
#             return {
#                 "description": msg,
#                 "source_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(src_categories, original_src_categories)]),
#                 "target_categories": "; ".join([f"{s} <-> {t}" for s, t in zip(tgt_categories, original_tgt_categories)])
#             }, "Partial Match (Tentative)"

#     elif (
#         (src_type == "continuous_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}) or
#         (tgt_type == "continuous_variable" and src_type in {"binary_class_variable", "multi_class_variable"})
#     ):
#         if src_data_type == "datetime" or tgt_data_type == "datetime":
#             return {
#                 "description": "Convert datetime to binary indicator (presence/absence) if needed.",
#                 "source_categories": "; ".join(src_categories),
#                 "target_categories": "; ".join(tgt_categories)
#             }, "Partial Match (Tentative)"
#         msg = (
#             "Discretize continuous variable to categories. Acceptable only if information loss is minimal. Represent as: (1) binary flag for event presence, (2) category of event type."
#             if domain not in ["drug_exposure", "drug_era"]
#             else "Harmonization may not be possible for drug-related continuous to categorical mappings. Review medication normalization depending on research question."
#         )
#         return {
#             "description": msg,
#             "source_categories": "; ".join(src_categories),
#             "target_categories": "; ".join(tgt_categories)
#         }, "Partial Match (Tentative)"

#     elif src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "qualitative_variable":
#         return {
#             "description": (
#                 "Map structured categorical codes to consistent/unique text labels. Requires normalization."
#                 "Only suitable for qualitative fields with finite, structured values."
#             ),
#             "source_categories": "; ".join(src_categories),
#             "target_categories": ""
#         }, "Partial Match (Tentative)"

#     elif src_type == "qualitative_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}:
#         return {
#             "description": (
#                 "Map qualitative text to standard categories. Normalize and encode."
#                 "Applicable only if text values are consistently structured."
#             ),
#             "source_categories": "",
#             "target_categories": "; ".join(tgt_categories)
#         }, "Partial Match (Tentative)"
#     elif src_type == "qualitative_variable" and tgt_type == "continuous_variable" or src_type == "continuous_variable" and tgt_type == "qualitative_variable":
#         return {
#             "description": (
#                 "Map qualitative text to binary indicators."
#                 "Applicable only if text values are consistently structured."
#             ),
#             "source_categories": "",
#             "target_categories": "; ".join(tgt_categories)
#         }, "Partial Match (Tentative)"
#     return {
#         "description": "No specific transformation rule available."
#     }, "Not Applicable"


# def apply_rules_v0(domain, src_info, tgt_info):
#     # print(f"src_info: {src_info}  tgt_info: {tgt_info}")
#     src_var_name = src_info.get('var_name', '').lower()
#     tgt_var_name = tgt_info.get('var_name', '').lower()
#     src_type = src_info.get('stats_type', '').lower()
#     tgt_type = tgt_info.get('stats_type', '').lower()
#     src_unit = src_info.get('unit', '').lower()
#     tgt_unit = tgt_info.get('unit', '').lower()
#     src_data_type = src_info.get('data_type', '').lower()
#     tgt_data_type = tgt_info.get('data_type', '').lower()
#     src_categories = src_info.get('categories', '').lower().split("|") if src_info.get('categories') else []
#     tgt_categories = tgt_info.get('categories', '').lower().split("|") if tgt_info.get('categories') else []
#     domains_list = ["observation", "drug_exposure", "device_exposure", "condition_era", "condition_occurrence","measurement", "procedure_occurrence", "observation_period", "demographic", "person"]
#   #  print(f"src_type: {src_type} tgt_type: {tgt_type} src_unit: {src_unit} tgt_unit: {tgt_unit}")
#     valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable","qualitative_variable"}
#     if src_type not in valid_types or tgt_type not in valid_types and ("derived" not in src_var_name or "derived" not in tgt_var_name):
#         return "Transformation Not applicable (invalid statistical type)"

    
#     if '|' in domain:
#         domains = domain.split("|")[0].strip()
#         for d in domains:
#             if d not in domains:
#                 return "Transformation not applicable for given domain(s)"
#     # Case 1: Same type
#     if src_type == tgt_type:
#         # Check if units differ for continuous variables
#         if src_type == "continuous_variable":
#             if src_unit and tgt_unit and src_unit != tgt_unit:
#                 if (src_unit in ["mg", "milligram"] and tgt_unit in ["%", "percent"]) or \
#                     (src_unit in ["%", "percent"] and tgt_unit in ["mg", "milligram"]):
#                     return "Unit conversion required (e.g., mg to %)"
#                 return "Unit conversion required (Research Question dependent)"
#             return "For harmonization, no transformation required."
#         else:
#             # check if categorical values are the same
#             if set(src_categories) == set(tgt_categories):
#                 return f"For harmonization, no transformation required if categorical values of the source variable: {src_categories} and target variable: {tgt_categories} are the same."
#             else:
#                 return f"For harmonization, alignment of categorical values is needed. Ensure that the categorical values of  the source variable : {src_categories} and target: {tgt_categories} variables are consistent. If they differ, transformation is needed to align them. This may involve semantical alignment of categories to ensure they match across datasets."

#     # Case 2: Binary ↔ Multiclass
#     if (
#         (src_type == "binary_class_variable" and tgt_type == "multi_class_variable") or
#         (src_type == "multi_class_variable" and tgt_type == "binary_class_variable")
#     ):
#         if domain in [ "drug_exposure", "drug_era"]:
#             return "For harmonization convert multi-class variables to binary classes, but accept only the degree of information loss justified by the research question. For drug-related variables, scrutinize the surrounding categorical context—e.g., therapy adjustments or supplemental medication descriptors—before deciding on the optimal harmonization, because these details may not map cleanly onto a binary split."
#         else:
#             return "For harmonization, convert multi-class variables to binary classes, but accept only the degree of information loss justified by the research question"

#     # Case 3: Continuous → Categorical
#     if (src_type == "continuous_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}) or src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "continuous_variable":
#         if src_data_type == "datetime" or tgt_data_type == "datetime":
#             return "When harmonizing a datetime variable with a binary or multi class variable, transform the datetime into a presence/absence indicator. Any non-missing datetime value indicates 'presence'; a missing or null datetime indicates 'absence'."
#         if domain in [ "drug_exposure", "drug_era"]:
#             return "For drug-related variables in harmonization, first examine any accompanying categorical context—such as therapy adjustments or descriptive qualifiers—because these details may not align neatly with the drug-dosage columns and harmonization may not be possible."
#         return "For harmonization, you may discretize continuous variables into categorical classes: {tgt_categories}, but only when the resulting information loss is acceptable for the research question. Represent each clinical domain with two elements: 1Presence/absence flag:a binary indicator showing whether an event exists, 2) Event category: a categorical field specifying which event occurred (e.g., which condition, which procedure, which device etc)."
#     if src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "qualitative_variable":
#         return "This variable pair involves categorical variable with values: {src_categories} and a qualitative variable (string/text-based). Harmonization is conditionally possible if the qualitative variable contains a finite and consistently used set of values that can be reliably mapped to the categorical codes. Transformation requires: (1) value normalization (e.g., spelling, casing), and (2) manual or automated mapping to standardized categories. This process may incur minor information loss and should be justified based on the harmonization goal. Applicability is limited to cases where the qualitative variable represents discrete categories, not unstructured narrative text."
#     if src_type == "qualitative_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}:
#         return "This variable pair involves a qualitative variable (string/text-based) and a categorical variable with values: {tgt_categories}. Harmonization is conditionally possible if the qualitative variable contains a finite and consistently used set of values that can be reliably mapped to the categorical codes. Transformation requires: (1) value normalization (e.g., spelling, casing), and (2) manual or automated mapping to standardized categories. This process may incur minor information loss and should be justified based on the harmonization goal. Applicability is limited to cases where the qualitative variable represents discrete categories, not unstructured narrative text."
#     # # Case 4: Categorical → Continuous (rare)
#     # if src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "continuous_variable":
#     #     if src_data_type == "datetime" or src_data_type == "datetime":
#     #         return "transformation Not applicable"
#     #     return "Transform categorical to continuous (acceptable loss, RQ dependent)"

#     return "Transformation rule not defined"



def parse_joined_string(input_str: str) -> list:
    """
    Parses a string that may be either:
    - a key-value categorical string like '1=No|2=Yes' or '1="mmol|l"|2="g|dl"'
    - a plain joined string like '"mg|dl"|mmol'
    
    Returns a list of extracted values, handling quoted values and internal pipes correctly.
    """
    if not input_str or not isinstance(input_str, str):
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
        print(f"Query = {query}")
        
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
    print(f"Checking if graph exists: {query}")
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
    print(f"Checking if triple exists: {query}")
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

# def chunks(iterable, batch_size):
#     batch = []
#     for triple in iterable:
#         batch.append(triple)
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#     if batch:
#         yield batch

# def publish_graph_in_batches(g: Graph, graph_uri: str, batch_size: int = 10000) -> bool:
#     url = f"{settings.sparql_endpoint}/store?graph={graph_uri}"
#     headers = {"Content-Type": "application/trig"}
#     success = True

#     for batch in chunks(g, batch_size):
#         batch_graph = Graph()
#         for triple in batch:
#             print(f"Triple: {triple}")
#             batch_graph.add(triple)
#         batch_data = batch_graph.serialize(format="trig")
#         response = requests.post(url, headers=headers, data=batch_data, timeout=300)
#         if not response.ok:
#             print(f"Failed to upload a batch: {response.status_code}, {response.text}")
#             success = False
#             # Optionally break or continue based on your error strategy
#         else:
#             print("Batch uploaded successfully")
#     return success



# def find_study_by_design(design_str: str, study_name: str = None) -> str | None:
#     sparql = SPARQLWrapper(settings.query_endpoint)
#     sparql.setReturnFormat(JSON)
#     study_name = normalize_text(study_name) if study_name else None
#     study_design_uri = f"obi:{design_str}"
#     query = f"""
#     PREFIX dc: <http://purl.org/dc/elements/1.1/>
#     PREFIX ro: <http://purl.obolibrary.org/obo/ro.owl/>
#     PREFIX obi: <http://purl.obolibrary.org/obo/obi.owl/>

#     SELECT ?sde
#     WHERE {{
#     GRAPH <https://w3id.org/CMEO/graph/studies_metadata> {{
#         {{
#         ?sde a {study_design_uri} ;
#          dc:identifier ?id .
#         FILTER(LCASE(STR(?id)) = LCASE("{study_name}"))
#         }}
#     }}
#     }}
#     LIMIT 1
#     """
#     print(f"SPARQL Query: {query}")
#     sparql.setQuery(query)
#     results = sparql.query().convert()
#     print(f"Results: {results}")
#     if results['results']['bindings']:
#         study_uri = results['results']['bindings'][0]['sde']['value']
#         print(f"Study found: {study_uri}")
#         return study_uri
#     else:
#         print("No study found with the given design.")
#         return None

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
