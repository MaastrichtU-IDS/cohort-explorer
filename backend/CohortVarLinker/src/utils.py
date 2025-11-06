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
    g.bind("ncbi", OntologyNamespaces.NCBI.value)  
    g.bind("rdfs", RDFS)
    # g.bind("omop", OMOP)
    g.bind("dc", DC)
    # g.bind("snomed", SNOMED)
    # do we need individual bindings for each ontology?
         # g.bind("snomed", SNOMED)
        # g.bind("loinc", LOINC)
        # g.bind("atc", ATC)
        # g.bind("rxnorm", RXNORM)
        # g.bind("ucum", UCUM)
        # g.bind("mesh", MESH)
        # g.bind("omop_ext", OMOP_EXT)
    g.graph(identifier=URIRef(default_graph_identifier))
    return g

# create log file and add log function

# def log(message: str) -> None:
#     """Append a log message to the log file with a timestamp."""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_message = f"{timestamp} - {message}\n"
#     with open(settings.logs_filepath, "a") as log_file:
#         log_file.write(log_message)
#         log_file.flush()    
    
# def sanitize(value: str) -> str:
#     """Sanitizes the input value for URI safety."""
#     if value is None:
#         return ""
#     return quote(value.replace(' ', '_'))

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
            
    # print(f"categorical columns: ({len(binary_categorical)})")
    # print(f"categorical columns: ({len(multi_class_categorical)})")
    return binary_categorical, multi_class_categorical

# def get_standard_label_uri(var_uri: URIRef, base_entity_label: str) -> URIRef:
#     safe_base_entity_label = normalize_text(base_entity_label)
#     return URIRef(f"{var_uri}/standard_label/{safe_base_entity_label}")


# def get_code_uri(var_uri: URIRef, code: str) -> URIRef:
#     # safe_base_entity_label = sanitize(base_entity_label)
#     return URIRef(f"{var_uri}/code/{code}")

# def get_omop_id_uri(var_uri: URIRef, omop_id: str) -> URIRef:
#     # safe_base_entity_label = sanitize(base_entity_label)
#     if omop_id == "" or omop_id is None:
#         print("OMOP ID is empty")
#     return URIRef(f"{var_uri}/omop_id/{omop_id}")

# def get_category_uri(var_uri: URIRef, category_value: str) -> URIRef:
#     """Generates a unique and URI-safe URI for each category based on the variable URI and category value."""
#     # Encode the category value to make it URI-safe
#     safe_category_value = normalize_text(category_value) # This will replace unsafe characters with % encoding
#     return URIRef(f"{var_uri}/category/{safe_category_value}")


# def get_context_uri(var_uri: str | URIRef, context_id: str) -> URIRef: 
#     safe_context_id = normalize_text(context_id)
#     return URIRef(f"{var_uri!s}/context/{safe_context_id}")

# def get_temporal_context_uri(var_uri: str | URIRef, temporal_context_id: str) -> URIRef:
#     safe_temporal_context_id = normalize_text(temporal_context_id)
#     return URIRef(f"{var_uri!s}/visit/{safe_temporal_context_id}")

# def get_measurement_unit_uri(var_uri: str | URIRef, unit_label: str) -> URIRef:
#     if unit_label is None:
#         print("Unit label is None")
#     safe_unit_label = normalize_text(unit_label)
#     return URIRef(f"{var_uri!s}/unit/{safe_unit_label}")


def safe_int(value):
    """Safely convert a value to an integer, returning None if the value is invalid."""
    try:
        return int(float(value)) if value else None
    except ValueError:
        print(f"Invalid integer value: {value}")
        return None



# Read a small part of the file to detect encoding

# def detect_code(file_path: str) -> str:
#     with open(file_path, 'rb') as f:
#         # Read the first 10,000 bytes of the file for detection
#         rawdata = f.read(10000)
#     result = chardet.detect(rawdata)
#     print(result)
#     if 'encoding' in result and result['encoding']:
#         return result
#     else:
#         raise ValueError("Encoding detection failed, no valid encoding found.")
    
    


# def add_optional_literal(
#     graph: Graph, 
#     uri: URIRef, 
#     predicate: URIRef, 
#     row: pd.Series, 
#     column: str, 
#     datatype=XSD.string, 
#     graph_context: URIRef = None
# ) -> None:
#     """
#     Adds a literal to the graph if the value exists in the row.

#     :param graph: RDF Graph
#     :param uri: URIRef to which the literal is attached
#     :param predicate: RDF predicate
#     :param row: Pandas Series representing the row
#     :param column: Column name in the row
#     :param datatype: XSD datatype for the literal
#     :param graph_context: Specific graph/context to add the triple to
#     """
#     value = row.get(column)
#     if pd.notna(value) and value != "":
#         if datatype == XSD.boolean:
#             if isinstance(value, str):
#                 value = value.strip().lower()
#                 if value in ['yes', 'true', '1']:
#                     literal_value = True
#                 elif value in ['no', 'false', '0']:
#                     literal_value = False
#                 else:
#                     print(f"Warning: Unrecognized boolean value '{value}' in column '{column}'. Skipping.")
#                     return
#             else:
#                 literal_value = bool(value)
#         elif datatype == XSD.integer:
#             try:
#                 literal_value = safe_int(value)
#             except ValueError:
#                 print(f"Warning: Cannot convert value '{value}' to integer for column '{column}'. Skipping.")
#                 return
#         elif datatype == XSD.decimal:
#             try:
#                 literal_value = float(value)
#             except ValueError:
#                 print(f"Warning: Cannot convert value '{value}' to decimal for column '{column}'. Skipping.")
#                 return
#         else:
#             literal_value = normalize_text(value)
        
#         if literal_value is not None:
#             if graph_context:
#                 graph.add((uri, predicate, Literal(literal_value, datatype=datatype), graph_context))
#                 print(f"Added triple: {uri} {predicate} {literal_value} in graph {graph_context}")
#             else:
#                 graph.add((uri, predicate, Literal(literal_value, datatype=datatype)))
#                 print(f"Added triple: {uri} {predicate} {literal_value}")
#     return  graph



# def apply_rules(domain, src_info, tgt_info):
#     def parse_categories(cat_str):
#         if pd.notna(cat_str) and cat_str not in [None, '']:
#             return [c.strip().lower() for c in str(cat_str).split(";")]
#         return []

#     print(f"src_info: {src_info}  tgt_info: {tgt_info}")
#     src_var_name = src_info.get('var_name', '').lower()
#     tgt_var_name = tgt_info.get('var_name', '').lower()
#     src_type = str(src_info.get('stats_type')).lower() if pd.notna(src_info.get('stats_type')) and src_info.get('stats_type') not in [None, ''] else None
#     tgt_type = str(tgt_info.get('stats_type')).lower() if pd.notna(tgt_info.get('stats_type')) and tgt_info.get('stats_type') not in [None, ''] else None

#     src_unit = str(src_info.get('unit', '').lower() if pd.notna(src_info.get('unit', '')) else None)
#     tgt_unit = str(tgt_info.get('unit', '').lower() if pd.notna(tgt_info.get('unit', '')) else None)
#     src_data_type = str(src_info.get('data_type', '').lower() if pd.notna(src_info.get('data_type', '')) else None)
#     tgt_data_type = str(tgt_info.get('data_type', '').lower() if pd.notna(tgt_info.get('data_type', '')) else None)
#     src_categories = parse_categories(src_info.get('categories', ''))
#     tgt_categories = parse_categories(tgt_info.get('categories', ''))

#     valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable", "qualitative_variable"}
#     if src_type not in valid_types or tgt_type not in valid_types:
#         if "derived" not in src_var_name and "derived" not in tgt_var_name:
#             return {
#                 "rule": "Transformation not applicable (invalid or missing statistical type).",
#                 "source_categories": "; ".join(src_categories),
#                 "target_categories": "; ".join(tgt_categories)
#             }

#     # Handle domains
#     if "|" in domain:
#         domains = domain.split("|")
#         for d in domains:
#             if d.strip() not in domains:
#                 return {
#                     "rule": "Transformation not applicable for given domain(s).",
#                     "source_categories": "; ".join(src_categories),
#                     "target_categories": "; ".join(tgt_categories)
#                 }

#     if src_type == tgt_type:
#         if src_type == "continuous_variable":
#             if src_unit and tgt_unit and src_unit != tgt_unit:
#                 if (src_unit in ["mg", "milligram"] and tgt_unit in ["%", "percent"]) or \
#                    (src_unit in ["%", "percent"] and tgt_unit in ["mg", "milligram"]):
#                     return {
#                         "rule": "Unit conversion required (e.g., mg to %).",
#                         "source_categories": "",
#                         "target_categories": ""
#                     }
#                 return {
#                     "rule": "Unit conversion required. Evaluate based on research question.",
#                     "source_categories": "",
#                     "target_categories": ""
#                 }
#             return {
#                 "rule": "No transformation required. Continuous types and units match.",
#                 "source_categories": "",
#                 "target_categories": ""
#             }
#         elif set(src_categories) == set(tgt_categories):
#             return {
#                 "rule": "No transformation required. Source and target categorical values match.",
#                 "source_categories": "; ".join(src_categories),
#                 "target_categories": "; ".join(tgt_categories)
#             }
#         else:
#             return {
#                 "rule": (
#                     "Alignment of categorical values required. Source and target differ. "
#                     "Map categories semantically across datasets."
#                 ),
#                 "source_categories": "; ".join(src_categories),
#                 "target_categories": "; ".join(tgt_categories)
#             }

#     if (
#         (src_type == "binary_class_variable" and tgt_type == "multi_class_variable") or
#         (src_type == "multi_class_variable" and tgt_type == "binary_class_variable")
#     ):
#         msg = (
#             "Convert multi-class to binary class. Accept only justified information loss. "
#             "For drug-related variables, consider therapy details and surrounding context."
#             if domain in ["drug_exposure", "drug_era"]
#             else "Convert multi-class to binary class. Information loss must be justified."
#         )
#         return {
#             "rule": msg,
#             "source_categories": "; ".join(src_categories),
#             "target_categories": "; ".join(tgt_categories)
#         }

#     if (
#         (src_type == "continuous_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}) or
#         (tgt_type == "continuous_variable" and src_type in {"binary_class_variable", "multi_class_variable"})
#     ):
#         if src_data_type == "datetime" or tgt_data_type == "datetime":
#             return {
#                 "rule": "Convert datetime to binary indicator (presence/absence).",
#                 "source_categories": "; ".join(src_categories),
#                 "target_categories": "; ".join(tgt_categories)
#             }
#         msg = (
#             "Discretize continuous variable to categories. Acceptable only if information loss is minimal. "
#             "Represent as: (1) binary flag for event presence, (2) category of event type."
#             if domain not in ["drug_exposure", "drug_era"]
#             else "Harmonization may not be possible for drug-related continuous ↔ categorical mappings. Review therapy context."
#         )
#         return {
#             "rule": msg,
#             "source_categories": "; ".join(src_categories),
#             "target_categories": "; ".join(tgt_categories)
#         }

#     if src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "qualitative_variable":
#         return {
#             "rule": (
#                 "Map structured categorical codes to consistent text labels. Requires normalization. "
#                 "Only suitable for qualitative fields with finite, structured values."
#             ),
#             "source_categories": "; ".join(src_categories),
#             "target_categories": ""
#         }

#     if src_type == "qualitative_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}:
#         return {
#             "rule": (
#                 "Map qualitative text to standard categories. Normalize and encode. "
#                 "Applicable only if text values are consistently structured."
#             ),
#             "source_categories": "",
#             "target_categories": "; ".join(tgt_categories)
#         }

#     return {
#         "rule": "No specific transformation rule matched.",
#         "source_categories": "; ".join(src_categories),
#         "target_categories": "; ".join(tgt_categories)
#     }



def build_label_mapping(src_labels_str,src_codes_str,  tgt_labels_str, tgt_codes_str,):


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


def apply_rules(domain, src_info, tgt_info):
    import pandas as pd

    def norm_or_none(x):
        return str(x).strip().lower() if (x not in (None, "") and not pd.isna(x)) else None

    src_type = norm_or_none(src_info.get('stats_type'))
    tgt_type = norm_or_none(tgt_info.get('stats_type'))
    src_unit = norm_or_none(src_info.get('unit'))
    tgt_unit = norm_or_none(tgt_info.get('unit'))
    src_data_type = norm_or_none(src_info.get('data_type'))
    tgt_data_type = norm_or_none(tgt_info.get('data_type'))
    
    valid_types = {"continuous_variable", "binary_class_variable", "multi_class_variable", "qualitative_variable"}
    if (src_type not in valid_types or tgt_type not in valid_types) or (src_type is None or tgt_type is None):
        return {"description": "Transformation not applicable (invalid or missing statistical type)."}, "Not Applicable"

    # --- same type
    if src_type == tgt_type:
        if src_type == "continuous_variable":
            if src_unit and tgt_unit and src_unit != tgt_unit:
                return {
                    "description": f"Unit conversion in dataset required from {src_unit} to {tgt_unit} (or vice versa)."
                }, "Complete Match (Compatible)"
            return {"description": "No transformation required. Continuous types and units match."}, "Complete Match (Identical)"

        # categorical/qualitative — align by labels
        m = build_label_mapping(
            src_info.get('categories_labels', ''),
            src_info.get('original_categories', ''),
            tgt_info.get('categories_labels', ''),
            tgt_info.get('original_categories', '')
        )

        if m["has_overlap"]:
            if m["identical"]:
                return {
                    "description": "Categorical values are identical and aligned by standard labels.",
                    "categorical_mapping": m["mapping_str"],
                    "standard_labels": "; ".join(sorted(m["overlap_labels"]))
                }, "Complete Match (Identical)"
            else:
                return {
                    "description": "Original categorical values differ but overlap on standard labels.",
                    "categorical_mapping": m["mapping_str"],
                    "unmapped_source_labels": m["unmapped_source_labels"],
                    "unmapped_target_labels": m["unmapped_target_labels"]
                }, "Complete Match (Compatible)"
        else:
            return {
                "description": "No overlap in standard labels between categorical values; mapping/review required.",
                "source_labels": "; ".join(sorted(set([v for v in (src_info.get('original_categories','') or '').split(';') if v.strip()]))),
                "target_labels": "; ".join(sorted(set([v for v in (tgt_info.get('original_categories','') or '').split(';') if v.strip()])))
            }, "Partial Match (Tentative)"

    # --- binary vs multi-class
    if ((src_type == "binary_class_variable" and tgt_type == "multi_class_variable") or
        (src_type == "multi_class_variable" and tgt_type == "binary_class_variable")):
        m = build_label_mapping(
            src_info.get('categories_codes', ''),
            src_info.get('original_categories', ''),
            tgt_info.get('categories_codes', ''),
            tgt_info.get('original_categories', '')
        )
        msg = ("Multi-class <-> binary conversion requires justification of information loss. "
               "For drug-related variables, consider therapy details and context."
               if domain in ["drug_exposure", "drug_era"]
               else "Multi-class ↔ binary conversion requires justification of information loss for the specific research question.")

        # proximate if the smaller label set ⊆ larger label set
        # (use labels for subset check; overlap already built)
        src_labels = set(l.lower() for l in m["overlap_labels"]) | set(
            x.strip().lower() for x in (src_info.get('categories_codes','') or '').split(';') if x.strip()
        )
        tgt_labels = set(l.lower() for l in m["overlap_labels"]) | set(
            x.strip().lower() for x in (tgt_info.get('categories_codes','') or '').split(';') if x.strip()
        )

        if (src_type == "binary_class_variable" and src_labels.issubset(tgt_labels) and len(src_labels) > 0) or \
           (tgt_type == "binary_class_variable" and tgt_labels.issubset(src_labels) and len(tgt_labels) > 0):
            categorical_mapping = m["mapping_str"] if m["has_overlap"] and m["mapping_str"] else None
            if categorical_mapping: 
                return {
                    "description": "Binary categories are a subset of multi-class categories; expansion/aggregation is possible.",
                    "categorical_mapping": categorical_mapping
                }, "Partial Match (Proximate)"
            else:
                return {
                    "description": msg,
                    "categorical_mapping": None,
                    "unmapped_source_labels": m["unmapped_source_labels"],
                    "unmapped_target_labels": m["unmapped_target_labels"]
                }, "Not Applicable"

    # --- continuous vs categorical
    if ((src_type == "continuous_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}) or
        (tgt_type == "continuous_variable" and src_type in {"binary_class_variable", "multi_class_variable"})):
        if src_data_type == "datetime" or tgt_data_type == "datetime":
            return {"description": "Unable to align datetime to binary/multi-class indicator."}, "Not Applicable"
        msg = ("Discretize continuous variable to categories only if information loss is minimal (e.g., event flag or type)."
               if domain not in ["drug_exposure", "drug_era"]
               else "Avoid continuous→categorical harmonization for drug-related variables unless strongly justified.")
        status = "Not Applicable" if domain in ["drug_exposure", "drug_era"] else "Partial Match (Tentative)"
        return {"description": msg}, status

    # --- qualitative vs categorical/continuous
    if (src_type in {"binary_class_variable", "multi_class_variable"} and tgt_type == "qualitative_variable"):
        return {"description": "Map structured categorical codes to consistent text labels; normalize values."}, "Partial Match (Tentative)"
    if (src_type == "qualitative_variable" and tgt_type in {"binary_class_variable", "multi_class_variable"}):
        return {"description": "Normalize qualitative text to standard categories; encode to labels/codes."}, "Partial Match (Tentative)"
    if ((src_type == "qualitative_variable" and tgt_type == "continuous_variable") or
        (src_type == "continuous_variable" and tgt_type == "qualitative_variable")):
        return {"description": "Map qualitative text to binary indicators only if values are consistently structured."}, "Partial Match (Tentative)"

    return {"description": "No specific transformation rule available."}, "Not Applicable"




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



# added from an older commit of the branch "cross_mapping"
def variable_exists(cohort_uri, variable_name) -> bool:
    sparql = SPARQLWrapper(settings.sparql_endpoint)
    variable_name = normalize_text(variable_name)
    sparql.setReturnFormat(JSON)

    query = f"""
            PREFIX cmeo: <https://w3id.org/CMEO/>
            PREFIX bfo: <http://purl.obolibrary.org/obo/bfo.owl/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX dc: <http://purl.org/dc/elements/1.1/>
            ASK WHERE {{
                GRAPH <{cohort_uri}> {{
                    ?variable rdf:type cmeo:data_element ;
                            dc:identifier "{variable_name}" ;
                            bfo:is_part_of ?variable_spec_uri .
                }}
            }}
    """
    # print(f"Query: {query}")        
        # print(f"SPARQL Query: {query}")
    sparql.setQuery(query)
    # print(f"Query: {query}")
    results = sparql.query().convert()
    
   # print(f"Results: {results}")
    if results['boolean'] == True:
        print(f"Variable {variable_name} exists in the graph.")
    return results['boolean']