"""The explorer's SPARQL queries over the knowledge graph, written against
CohortVarLinker's vocabulary.

CohortVarLinker/src/utils.py owns the vocabulary: `term_iri(name)` maps a
friendly term name ("has_value", "data_element", "protocol", ...) to the
ontology IRI the graph writers use (OBO classes and properties from STATO,
OBI, IAO, RO, BFO, PATO, CMEO, ...; SIO terms; KG-local names under
https://w3id.org/CMEO/). The queries here are written with the same friendly
names, as `cmeo:has_value`, `obi:protocol`, `ro:has_part`, ..., and
`resolve_terms` rewrites each one into `<iri>` through `term_iri` before the
query is sent. The prefix in front of a name is documentation only: the IRI
comes from the term table, so a query and the graph cannot drift apart. A name
the table does not know is an error here, not a silent fall-back to a KG-local
IRI (the writers' behaviour), because in a query that would just match nothing.

KG-local terms the explorer itself writes on the cohort node (email,
institution, studyType, ...) are spelled with the `cmeokg:` prefix, which is
left to SPARQL (bound to the KG base).

Two queries are built here:
  cohorts_metadata_query()   - one row per cohort from the studies_metadata graph
                               (upload.cohorts_metadata_file_to_graph writes it)
  variables_metadata_query() - one row per variable from every cohort graph
                               (CohortVarLinker.src.variables_kg writes them)
Both replace the queries that used to be sliced by line number out of
CohortVarLinker/queries/sparql_queries.txt (written for the previous IRIs).
"""

import re

# The term table and its KG-local list are private to CohortVarLinker's module;
# they are read here only to refuse unknown names (see resolve_terms).
from CohortVarLinker.src.utils import CMEO_KG, _LOCAL_TERMS, _ONTOLOGY_TERM_IRIS, term_iri

STUDIES_METADATA_GRAPH = f"{CMEO_KG}graph/studies_metadata"
COHORT_GRAPH_BASE = f"{CMEO_KG}graph/"

# Prefixes whose names are resolved through the term table. `cmeokg:` is not
# among them on purpose: it is a real SPARQL prefix for KG-local names.
_TERM_PREFIXES = ("cmeo", "obi", "iao", "ro", "sio", "bfo", "stato", "ncbi", "pato")
_TERM_RE = re.compile(r"\b(" + "|".join(_TERM_PREFIXES) + r"):([A-Za-z_]\w*)")

# Record and field separators of the per-category records in the variables
# query (ASCII record / unit separator: never part of a label).
CATEGORY_RECORD_SEP = "\u001e"
CATEGORY_FIELD_SEP = "\u001f"

_PLAIN_PREFIXES = f"""PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX dc:     <http://purl.org/dc/elements/1.1/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
PREFIX cmeokg: <{CMEO_KG}>
"""


def resolve_terms(sparql: str) -> str:
    """Rewrite every `prefix:name` term of the query into `<iri>` via term_iri.

    Raises KeyError for a name that is neither an ontology term nor a declared
    KG-local term of CohortVarLinker's table, so a typo cannot become a query
    that silently matches nothing.
    """
    unknown: list[str] = []

    def _sub(m: re.Match) -> str:
        name = m.group(2)
        if name not in _ONTOLOGY_TERM_IRIS and name not in _LOCAL_TERMS:
            unknown.append(f"{m.group(1)}:{name}")
            return m.group(0)
        return f"<{term_iri(name)}>"

    out = _TERM_RE.sub(_sub, sparql)
    if unknown:
        raise KeyError(f"terms unknown to CohortVarLinker's vocabulary: {sorted(set(unknown))}")
    return out


def cohorts_metadata_query() -> str:
    """One row per cohort (and study type) from the studies_metadata graph.

    Anchored on the study design node: dc:identifier (the spreadsheet spelling
    of the study name, the explorer's cohort id), iao:is_about its descriptor
    (study type), ro:has_part its protocol. The shape is the one
    upload.cohorts_metadata_file_to_graph writes, which mirrors
    CohortVarLinker.src.study_kg.generate_studies_kg.
    """
    graph = STUDIES_METADATA_GRAPH
    query = f"""{_PLAIN_PREFIXES}
SELECT
  ?cohortId
  ?study_type
  (SAMPLE(?study_participants) AS ?study_participants)
  (SAMPLE(?study_objective) AS ?study_objective)
  (SAMPLE(?study_start) AS ?study_start)
  (SAMPLE(?study_end) AS ?study_end)
  (SAMPLE(?study_ongoing) AS ?study_ongoing)
  (SAMPLE(?cohortInstitution) AS ?cohortInstitution)
  (SAMPLE(?administrator) AS ?administrator)
  (SAMPLE(?administrator_email) AS ?administrator_email)
  (SAMPLE(?study_contact_person) AS ?study_contact_person)
  (SAMPLE(?study_contact_person_email) AS ?study_contact_person_email)
  (SAMPLE(?population_location) AS ?population_location)
  (SAMPLE(?language) AS ?language)
  (SAMPLE(?duration) AS ?duration)
  (SAMPLE(?data_collection_frequency) AS ?data_collection_frequency)
  (SAMPLE(?interventions) AS ?interventions)
  (SAMPLE(?inclusion_labels) AS ?inclusion_labels)
  (SAMPLE(?inclusion_values) AS ?inclusion_values)
  (SAMPLE(?exclusion_labels) AS ?exclusion_labels)
  (SAMPLE(?exclusion_values) AS ?exclusion_values)
  (SAMPLE(?variable_uris) AS ?variable_uris)
  (SAMPLE(?primary_outcome_spec) AS ?primary_outcome_spec)
  (SAMPLE(?secondary_outcome_spec) AS ?secondary_outcome_spec)
  (SAMPLE(?morbidity) AS ?morbidity)
WHERE {{
  GRAPH <{graph}> {{

    # --- Core study & protocol (keeps the rest anchored) ---
    ?study dc:identifier ?cohortId ;
           iao:is_about  ?descriptor ;
           ro:has_part   ?protocol .
    ?descriptor a sio:descriptor ;
                rdfs:label ?study_type .
    ?protocol a obi:protocol .

    # --- Participants ---
    OPTIONAL {{
      ?protocol ro:has_part ?participants_spec .
      ?participants_spec a cmeo:number_of_participants ;
                         cmeo:has_value ?study_participants .
    }}

    # --- Objective ---
    OPTIONAL {{
      ?protocol ro:has_part ?objective_spec .
      ?objective_spec a obi:objective_specification ;
                      cmeo:has_value ?study_objective .
    }}

    OPTIONAL {{
      ?protocol ro:has_part ?dcfrequency .
      ?dcfrequency a cmeo:timeline_specification ;
                   cmeo:has_value ?data_collection_frequency .
    }}

    OPTIONAL {{
      ?protocol ro:has_part ?intervention_spec .
      ?intervention_spec a cmeo:intervention_specification ;
                         cmeo:has_value ?interventions .
    }}

    # --- Study execution / time / language / status / organization ---
    OPTIONAL {{
      ?sde a obi:study_design_execution ;
           ro:concretizes ?study ;
           dc:language ?language .
      OPTIONAL {{
        ?durationclass a cmeo:duration_of_observation ;
                       iao:is_about ?sde ;
                       cmeo:has_value ?duration .
      }}
      OPTIONAL {{
        ?sde iao:has_time_stamp ?start_time .
        ?start_time a cmeo:start_time ;
                    cmeo:has_value ?study_start .
      }}
      OPTIONAL {{
        ?sde iao:has_time_stamp ?end_time .
        ?end_time a cmeo:end_time ;
                  cmeo:has_value ?study_end .
      }}
      OPTIONAL {{
        ?completion_status a sio:ongoing ;
                           iao:is_about ?sde ;
                           cmeo:has_value ?study_ongoing .
      }}
      OPTIONAL {{
        ?sde ro:has_participant ?organization .
        ?organization a obi:organization ;
                      cmeo:has_value ?cohortInstitution .
        OPTIONAL {{
          ?admin a ncbi:homo_sapiens ;
                 obi:member_of ?organization ;
                 cmeo:has_value ?administrator .
          OPTIONAL {{
            ?admin_email a obi:email_address ;
                         iao:is_about ?admin ;
                         cmeo:has_value ?administrator_email .
          }}
        }}
        OPTIONAL {{
          ?contactperson a ncbi:homo_sapiens ;
                         obi:member_of ?organization ;
                         cmeo:has_value ?study_contact_person .
          ?contact_person_email a obi:email_address ;
                                iao:is_about ?contactperson ;
                                cmeo:has_value ?study_contact_person_email .
        }}
      }}
    }}

    # --- PRIMARY outcomes (pre-aggregated) ---
    OPTIONAL {{
      {{
        SELECT ?study (GROUP_CONCAT(DISTINCT ?v; separator="||") AS ?primary_outcome_spec)
        WHERE {{
          GRAPH <{graph}> {{
            ?study ro:has_part ?p .
            ?p a obi:protocol ;
               ro:has_part ?outcome_spec .
            ?outcome_spec a cmeo:outcome_specification ;
                          ro:has_part ?prim_spec .
            ?prim_spec a cmeo:primary_outcome_specification ;
                       cmeo:has_value ?raw .
            BIND(STR(?raw) AS ?v)
          }}
        }}
        GROUP BY ?study
      }}
    }}

    # --- SECONDARY outcomes (pre-aggregated) ---
    OPTIONAL {{
      {{
        SELECT ?study (GROUP_CONCAT(DISTINCT ?v; separator="||") AS ?secondary_outcome_spec)
        WHERE {{
          GRAPH <{graph}> {{
            ?study ro:has_part ?p .
            ?p a obi:protocol ;
               ro:has_part ?outcome_spec .
            ?outcome_spec a cmeo:outcome_specification ;
                          ro:has_part ?sec_spec .
            ?sec_spec a cmeo:secondary_outcome_specification ;
                      cmeo:has_value ?raw .
            BIND(STR(?raw) AS ?v)
          }}
        }}
        GROUP BY ?study
      }}
    }}

    # --- VARIABLE URIs (pre-aggregated) ---
    OPTIONAL {{
      {{
        SELECT ?study (GROUP_CONCAT(DISTINCT ?v; separator="||") AS ?variable_uris)
        WHERE {{
          GRAPH <{graph}> {{
            ?study ro:has_part ?p .
            ?p a obi:protocol ;
               ro:has_part ?variable_spec .
            OPTIONAL {{
              ?variable_spec a cmeo:study_design_variable_specification ;
                             ro:has_part ?variable_uri .
              BIND(STR(?variable_uri) AS ?v)
            }}
          }}
        }}
        GROUP BY ?study
      }}
    }}

    {{
      SELECT ?study ?population_location
             (GROUP_CONCAT(?lbl; SEPARATOR=" || ") AS ?inclusion_labels)
             (GROUP_CONCAT(?val; SEPARATOR=" || ") AS ?inclusion_values)
             (GROUP_CONCAT(?ex_lbl; SEPARATOR=" || ") AS ?exclusion_labels)
             (GROUP_CONCAT(?ex_val; SEPARATOR=" || ") AS ?exclusion_values)
             (GROUP_CONCAT(DISTINCT ?m; separator="; ") AS ?morbidity)
      WHERE {{
        {{
          # Order the rows by a stable key (?k) so labels and values line up positionally
          SELECT ?study ?k ?lbl ?val ?ex_lbl ?ex_val ?population_location ?m
          WHERE {{
            GRAPH <{graph}> {{
              ?study ro:has_part ?protocol .
              ?protocol a obi:protocol ;
                        ro:has_part ?selection_criterion .
              ?selection_criterion a obi:eligibility_criterion ;
                                   ro:has_part ?criterion .

              OPTIONAL {{
                ?selection_criterion ro:is_concretized_by ?enrollment .
                ?enrollment a cmeo:human_subject_enrollment ;
                            ro:has_output ?output_population .
                ?site iao:is_about ?output_population ;
                      cmeo:has_value ?population_location .
                ?morbidity_spec a obi:morbidity ;
                                ro:is_characteristic_of ?output_population ;
                                cmeo:has_value ?mv .
                BIND(STR(?mv) AS ?m)
              }}
              OPTIONAL {{
                ?criterion a obi:inclusion_criterion ;
                           ro:has_part ?inclusion_criterion_sub .
                BIND(STR(?inclusion_criterion_sub) AS ?k)
                OPTIONAL {{ ?inclusion_criterion_sub rdfs:label ?inc_label . }}
                BIND(STR(?inc_label) AS ?lbl)
                OPTIONAL {{ ?inclusion_criterion_sub cmeo:has_value ?val_raw . }}
                BIND(STR(?val_raw) AS ?val)
                FILTER(BOUND(?val) && ?val != "")
              }}
              OPTIONAL {{
                ?criterion a obi:exclusion_criterion ;
                           ro:has_part ?exclusion_criterion_sub .
                BIND(STR(?exclusion_criterion_sub) AS ?exc_k)
                OPTIONAL {{ ?exclusion_criterion_sub rdfs:label ?exc_label . }}
                BIND(STR(?exc_label) AS ?ex_lbl)
                OPTIONAL {{ ?exclusion_criterion_sub cmeo:has_value ?exc_val_raw . }}
                BIND(STR(?exc_val_raw) AS ?ex_val)
                FILTER(BOUND(?ex_val) && ?ex_val != "")
              }}
            }}
          }}
          ORDER BY ?study ?population_location ?k ?exc_k ?m
        }}
      }}
      GROUP BY ?study ?population_location
    }}
  }}
}}
GROUP BY ?study ?cohortId ?study_type
"""
    return resolve_terms(query)


def variables_metadata_query() -> str:
    """One row per variable from every cohort graph (not studies_metadata).

    Reads the shape CohortVarLinker.src.variables_kg.process_variables_metadata_file
    writes:
      variable      a cmeo:data_element ; dc:identifier ; rdfs:label
      data type     variable sio:has_attribute ?dt . ?dt a cmeo:data_type ; cmeo:has_value
      formula       variable sio:has_attribute ?f . ?f a cmeo:formula ; cmeo:has_value
      domain        variable sio:has_annotation ?c . ?c a skos:Concept ; cmeo:has_value
      concepts      variable skos:closeMatch ?code_set . ?code_set a iao:code_set ;
                    rdf:_1/_2/... ?code . ?code a skos:Concept ; rdfs:label ; cmeo:has_value ;
                    iao:denotes ?o . ?o a cmeo:omop_id ; cmeo:has_value
      categories    ?cv a obi:categorical_value_specification ; obi:specifies_value_of variable ;
                    cmeo:has_value ; rdfs:label ; optionally skos:closeMatch ?code (as above)
      unit          variable obi:has_measurement_unit_label ?u . ?u a iao:measurement_unit_label ;
                    cmeo:has_value ; optionally skos:closeMatch ?code
      visit         variable sio:has_attribute ?v . ?v a cmeo:visit_type ; cmeo:has_value ;
                    optionally skos:closeMatch ?code
      statistics    variable iao:is_denoted_by ?stype . ?dataset iao:is_about ?stype .
                    ?stat a stato:statistic ; iao:is_about ?dataset ; ro:has_part ?x .
                    ?x a <stat class> ; rdfs:label ; cmeo:has_value
                    (several counts share the STATO 'count' class; the label tells them apart)

    Categories come back as one record per category in ?categorical_values:
    value, label, concept code, concept label, OMOP id joined by
    CATEGORY_FIELD_SEP, records joined by CATEGORY_RECORD_SEP (see
    parse_category_records). The variable's concepts come back as "|"-joined
    lists in code-set order (?concept_codes, ?concept_labels, ?concept_omop_id).
    """
    fs, rs = CATEGORY_FIELD_SEP, CATEGORY_RECORD_SEP
    query = f"""{_PLAIN_PREFIXES}
SELECT
  ?study_graph
  ?study_name
  ?var_name
  ?var_label
  ?varType
  ?categorical_values
  ?concept_labels
  ?concept_codes
  ?concept_omop_id
  ?unit_value
  ?unit_label
  ?visit
  ?visit_concept_name
  ?omopDomain
  ?count_value
  ?formula_value
  ?minimum_value
  ?maximum_value
WHERE {{
  # -------- base: all variables from each cohort graph (skip studies_metadata) --------
  GRAPH ?g {{
    ?var a cmeo:data_element ;
         dc:identifier ?var_name .
    OPTIONAL {{ ?var rdfs:label ?var_label . }}
    OPTIONAL {{ ?study a cmeokg:study ; rdfs:label ?_study_label . }}
  }}
  FILTER( STRSTARTS(STR(?g), "{COHORT_GRAPH_BASE}") )
  FILTER( STRAFTER(STR(?g), "{COHORT_GRAPH_BASE}") != "studies_metadata" )
  BIND(?g AS ?study_graph)
  BIND(COALESCE(?_study_label, STRAFTER(STR(?g), "/graph/")) AS ?study_name)

  # ---------------- data type ----------------
  OPTIONAL {{
    SELECT ?g ?var (SAMPLE(?vt) AS ?varType)
    WHERE {{
      GRAPH ?g {{
        ?var sio:has_attribute ?dt .
        ?dt a cmeo:data_type ;
            cmeo:has_value ?vt .
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- formula ----------------
  OPTIONAL {{
    SELECT ?g ?var (GROUP_CONCAT(DISTINCT ?f_val; SEPARATOR=" || ") AS ?formula_value)
    WHERE {{
      GRAPH ?g {{
        ?var sio:has_attribute ?formula .
        ?formula a cmeo:formula ;
                 cmeo:has_value ?f_val .
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- categories: one record per category, with its concept ----------------
  OPTIONAL {{
    SELECT ?g ?var (GROUP_CONCAT(?rec; SEPARATOR="{rs}") AS ?categorical_values)
    WHERE {{
      {{
        SELECT ?g ?var ?cv (SAMPLE(?rec0) AS ?rec)
        WHERE {{
          GRAPH ?g {{
            ?cv a obi:categorical_value_specification ;
                obi:specifies_value_of ?var ;
                cmeo:has_value ?c_val ;
                rdfs:label ?c_label .
            OPTIONAL {{
              ?cv skos:closeMatch ?c_code .
              ?c_code a skos:Concept ;
                      cmeo:has_value ?c_code_val .
              OPTIONAL {{ ?c_code rdfs:label ?c_code_lbl . }}
              OPTIONAL {{ ?c_code iao:denotes ?c_omop . ?c_omop a cmeo:omop_id ; cmeo:has_value ?c_omop_val . }}
            }}
            BIND(CONCAT(STR(?c_val), "{fs}", STR(?c_label), "{fs}",
                        COALESCE(STR(?c_code_val), ""), "{fs}",
                        COALESCE(STR(?c_code_lbl), ""), "{fs}",
                        COALESCE(STR(?c_omop_val), "")) AS ?rec0)
          }}
        }} GROUP BY ?g ?var ?cv
        ORDER BY ?g ?var ?cv
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- OMOP domain (annotation of the variable) ----------------
  OPTIONAL {{
    SELECT ?g ?var (SAMPLE(?domain_val) AS ?omopDomain)
    WHERE {{
      GRAPH ?g {{
        ?var sio:has_annotation ?category .
        ?category a skos:Concept ;
                  cmeo:has_value ?domain_val .
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- variable concepts (code set, in rdf:_n order) ----------------
  OPTIONAL {{
    SELECT ?g ?var
           (GROUP_CONCAT(?code_val; SEPARATOR="|") AS ?concept_codes)
           (GROUP_CONCAT(?code_lbl; SEPARATOR="|") AS ?concept_labels)
           (GROUP_CONCAT(?omop_str; SEPARATOR="|") AS ?concept_omop_id)
    WHERE {{
      {{
        SELECT ?g ?var ?seq (SAMPLE(?code_val0) AS ?code_val) (SAMPLE(?code_lbl0) AS ?code_lbl) (SAMPLE(?omop_str0) AS ?omop_str)
        WHERE {{
          GRAPH ?g {{
            ?var skos:closeMatch ?code_set .
            ?code_set a iao:code_set ;
                      ?seq ?code .
            FILTER(STRSTARTS(STR(?seq), "http://www.w3.org/1999/02/22-rdf-syntax-ns#_"))
            ?code a skos:Concept ;
                  cmeo:has_value ?code_val0 .
            OPTIONAL {{ ?code rdfs:label ?code_lbl0 . }}
            OPTIONAL {{ ?code iao:denotes ?omop_class . ?omop_class a cmeo:omop_id ; cmeo:has_value ?omop_val . }}
            BIND(COALESCE(STR(?omop_val), "") AS ?omop_str0)
          }}
        }} GROUP BY ?g ?var ?seq
        ORDER BY ?g ?var ?seq
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- unit ----------------
  OPTIONAL {{
    SELECT ?g ?var (SAMPLE(?u_val) AS ?unit_value) (SAMPLE(?u_lbl) AS ?unit_label)
    WHERE {{
      GRAPH ?g {{
        ?var obi:has_measurement_unit_label ?unit .
        ?unit a iao:measurement_unit_label ;
              cmeo:has_value ?u_val .
        OPTIONAL {{ ?unit skos:closeMatch ?u_code . ?u_code rdfs:label ?u_lbl . }}
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- visit ----------------
  OPTIONAL {{
    SELECT ?g ?var (GROUP_CONCAT(DISTINCT ?visit_val; SEPARATOR="|") AS ?visit)
                   (GROUP_CONCAT(DISTINCT ?visit_lbl; SEPARATOR="|") AS ?visit_concept_name)
    WHERE {{
      GRAPH ?g {{
        ?var sio:has_attribute ?visit_node .
        ?visit_node a cmeo:visit_type ;
                    cmeo:has_value ?visit_val .
        OPTIONAL {{ ?visit_node skos:closeMatch ?v_code . ?v_code rdfs:label ?visit_lbl . }}
      }}
    }} GROUP BY ?g ?var
  }}

  # ---------------- statistics (count, min, max) ----------------
  OPTIONAL {{
    SELECT ?g ?var
           (SAMPLE(?cnt) AS ?count_value)
           (SAMPLE(?min_val) AS ?minimum_value)
           (SAMPLE(?max_val) AS ?maximum_value)
    WHERE {{
      GRAPH ?g {{
        ?var iao:is_denoted_by ?stype .
        ?dataset iao:is_about ?stype .
        ?statistic a stato:statistic ;
                   iao:is_about ?dataset .
        OPTIONAL {{
          ?statistic ro:has_part ?count_node .
          ?count_node a stato:count_ ;
                      rdfs:label ?count_label ;
                      cmeo:has_value ?cnt .
          FILTER(STR(?count_label) = "count")
        }}
        OPTIONAL {{
          ?statistic ro:has_part ?min_node .
          ?min_node a stato:minimum_value ;
                    cmeo:has_value ?min_val .
        }}
        OPTIONAL {{
          ?statistic ro:has_part ?max_node .
          ?max_node a stato:maximum_value ;
                    cmeo:has_value ?max_val .
        }}
      }}
    }} GROUP BY ?g ?var
  }}
}}
"""
    return resolve_terms(query)


def parse_category_records(value: str) -> list[dict[str, str]]:
    """Split ?categorical_values of variables_metadata_query into dicts with the
    keys value, label, code, code_label, omop_id (empty strings when absent)."""
    out: list[dict[str, str]] = []
    if not value:
        return out
    for rec in value.split(CATEGORY_RECORD_SEP):
        if not rec:
            continue
        parts = rec.split(CATEGORY_FIELD_SEP)
        parts += [""] * (5 - len(parts))
        out.append({
            "value": parts[0].strip(),
            "label": parts[1].strip(),
            "code": parts[2].strip(),
            "code_label": parts[3].strip(),
            "omop_id": parts[4].strip(),
        })
    return out
