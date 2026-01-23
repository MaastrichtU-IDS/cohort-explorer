import glob
import logging
import math
import os
import re
import shutil
import csv
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from re import sub
import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from rdflib import Graph, Literal, URIRef, Dataset
from rdflib.namespace import DC, RDF, RDFS, XSD
from SPARQLWrapper import SPARQLWrapper

from src.config import settings
from src.auth import get_current_user
from src.utils import (
    ICARE,
    curie_converter,
    extract_age_range,
    get_cohorts_metadata_query,
    init_graph,
    OntologyNamespaces,
    normalize_text,
    retrieve_cohorts_metadata,
    run_query
)
from src.cohort_cache import add_cohort_to_cache, clear_cache, create_cohort_from_dict_file, create_cohort_from_metadata_graph
from src.decentriq import create_provision_dcr, metadatadict_cols_schema1
from src.mapping_generation.retriever import map_csv_to_standard_codes

router = APIRouter()


def update_upload_log(upload_data: dict):
    """Update the upload log with structured data similar to provision_dcr logging"""
    try:
        upload_log_file = os.path.join(settings.data_folder, "upload_dictionary_log.jsonl")
        with open(upload_log_file, "r") as f:
            log = json.load(f)
            log.append(upload_data)
    except (FileNotFoundError, json.JSONDecodeError):
        log = [upload_data]
    
    with open(upload_log_file, "w") as f:
        json.dump(log, f, indent=2, default=str)


def log_upload_event_csv(upload_data: dict):
    """Log upload events to CSV file for easy analysis"""
    upload_events_log = os.path.join(settings.data_folder, "upload_dictionary_events.csv")
    
    # Define the fieldnames for the CSV
    fieldnames = [
        "timestamp", "cohort_id", "user_email", "filename", "file_size_bytes", 
        "total_rows", "total_variables", "errors_count", "success", "processing_time_seconds",
        "source", "graph_triples_count"
    ]
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(upload_events_log)
    
    with open(upload_events_log, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(upload_data)


def publish_graph_to_endpoint(g: Graph, graph_uri: str | None = None) -> bool:
    """Insert the graph into the triplestore endpoint."""
    # url = f"{settings.sparql_endpoint}/store?{graph_uri}"
    url = f"{settings.sparql_endpoint}/store"
    if graph_uri:
        url += f"?graph={graph_uri}"
    headers = {"Content-Type": "application/trig"}
    g.serialize("/tmp/upload-data.trig", format="trig")
    with open("/tmp/upload-data.trig", "rb") as file:
        response = requests.post(url, headers=headers, data=file, timeout=120)

    # NOTE: Fails when we pass RDF as string directly
    # response = requests.post(url, headers=headers, data=graph_data)
    # Check response status and print result
    if not response.ok:
        logging.warning(f"Failed to upload data: {response.status_code}, {response.text}")
    return response.ok


def delete_existing_triples(graph_uri: str | URIRef, subject="?s", predicate="?p"):
    """Function to delete existing triples in a cohort's graph"""
    query_endpoint = SPARQLWrapper(settings.update_endpoint)
    query_endpoint.setMethod("POST")
    query_endpoint.setRequestMethod("urlencoded")
    query_endpoint.setTimeout(300)  # 5 minutes timeout
    
    try:
        # Count triples before deletion (for logging)
        count_query = f"""
            SELECT (COUNT(*) as ?count)
            WHERE {{
                GRAPH <{graph_uri!s}> {{ {subject} {predicate} ?o . }}
            }}
        """
        count_endpoint = SPARQLWrapper(settings.query_endpoint)
        count_endpoint.setReturnFormat("json")
        count_endpoint.setTimeout(300)  # 5 minutes timeout
        count_endpoint.setQuery(count_query)
        count_results = count_endpoint.query().convert()
        triples_before = count_results["results"]["bindings"][0]["count"]["value"]
        
        # Execute the delete
        delete_query = f"""
            PREFIX icare: <https://w3id.org/icare4cvd/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            DELETE WHERE {{
                GRAPH <{graph_uri!s}> {{ {subject} {predicate} ?o . }}
            }}
        """
        query_endpoint.setQuery(delete_query)
        query_endpoint.query()
        
        # Count triples after deletion (for verification)
        count_endpoint.setQuery(count_query)
        count_results = count_endpoint.query().convert()
        triples_after = count_results["results"]["bindings"][0]["count"]["value"]
        
        logging.info(f"Deleted {int(triples_before) - int(triples_after)} triples from graph {graph_uri}")
        
        return True
    except Exception as e:
        logging.error(f"Error deleting triples: {e}")
        return False

def get_cohort_uri(cohort_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}"]


def get_cohort_graph_uri(cohort_id: str) -> URIRef:
    """Get the graph URI for a cohort in CMEO namespace (matches query expectations)"""
    return URIRef(OntologyNamespaces.CMEO.value + f"graph/{cohort_id}")


def get_cohort_mapping_uri(cohort_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}/mappings"]


def get_var_uri(cohort_id: str | URIRef, var_id: str) -> URIRef:
    # Strip and validate inputs to prevent Invalid IRI errors
    cohort_id_clean = str(cohort_id).strip().replace(' ', '_')
    var_id_clean = str(var_id).strip().replace(' ', '_')
    if not var_id_clean:
        raise ValueError(f"Variable ID cannot be empty or whitespace-only for cohort {cohort_id}")
    return ICARE[f"cohort/{cohort_id_clean}/{var_id_clean}"]


def get_category_uri(var_uri: str | URIRef, category_id: str) -> URIRef:
    return URIRef(f"{var_uri!s}/category/{category_id}")


def get_latest_datadictionary(cohort_folder_path: str) -> str | None:
    """
    Get the latest datadictionary file from a cohort folder.
    
    Args:
        cohort_folder_path: Path to the cohort folder
        
    Returns:
        Path to the latest datadictionary file or None if no datadictionary file is found
    """
    if not os.path.isdir(cohort_folder_path):
        return None
        
    # Select most recent CSV file with 'datadictionary' in the name
    csv_candidates = [
        f for f in glob.glob(os.path.join(cohort_folder_path, "*.csv"))
        if ("datadictionary" in os.path.basename(f).lower()
        and "noheader" not in os.path.basename(f).lower())
    ]
    
    if not csv_candidates:
        return None
        
    # Pick the most recently modified file
    return max(csv_candidates, key=os.path.getmtime)


# TODO
@router.post(
    "/insert-triples",
    name="Insert triples about a cohort variable in the triplestore",
    response_description="Upload result",
)
def insert_triples(
    cohort_id: str = Form(...),
    var_id: str = Form(...),
    predicate: str = Form(...),
    value: str = Form(...),
    label: str | None = Form(None),
    category_id: str | None = Form(None),
    user: Any = Depends(get_current_user),
) -> None:
    """Insert triples about mappings for cohorts variables or variables categories into the triplestore"""
    # Use cache instead of SPARQL query for better performance
    from src.cohort_cache import get_cohorts_from_cache
    cohorts = get_cohorts_from_cache(user["email"])
    cohort_info = cohorts.get(cohort_id)
    if not cohort_info:
        raise HTTPException(
            status_code=403,
            detail=f"Cohort ID {cohort_id} does not exists",
        )
    if not cohort_info.can_edit:
        raise HTTPException(
            status_code=403,
            detail=f"User {user['email']} cannot edit cohort {cohort_id}",
        )
    graph_uri = get_cohort_mapping_uri(cohort_id)
    subject_uri = get_var_uri(cohort_id, var_id)
    if category_id:
        subject_uri = get_category_uri(subject_uri, category_id)
        # TODO: handle when a category is provided (we add triple to the category instead of the variable)
    delete_existing_triples(graph_uri, f"<{subject_uri!s}>", predicate)
    label_part = ""
    object_uri = f"<{curie_converter.expand(value)}>"
    if label:
        delete_existing_triples(graph_uri, f"{object_uri}", "rdfs:label")
        label_part = f'{object_uri} rdfs:label "{label}" .'
    # TODO: some namespaces like Gender are not in the bioregistry
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    INSERT DATA {{
        GRAPH <{graph_uri!s}> {{ <{subject_uri!s}> {predicate} {object_uri} . {label_part} }}
    }}
    """
    # print(query)
    query_endpoint = SPARQLWrapper(f"{settings.sparql_endpoint}/update")
    query_endpoint.setMethod("POST")
    query_endpoint.setRequestMethod("urlencoded")
    query_endpoint.setTimeout(300)  # 5 minutes timeout
    query_endpoint.setQuery(query)
    query_endpoint.query()


def parse_categorical_string(s: str) -> list[dict[str, str]]:
    """Categorical string format: "value1=label1, value2=label2" or "value1=label1 | value2=label2"""
    # Split the string into items
    split_char = "," if "|" not in s else "|"
    items = s.split(split_char)
    result = []
    for item in items:
        if item:
            try:
                key, value = map(str.strip, item.split("="))
            except Exception:
                key, value = item.strip(), item.strip()
            result.append({"value": key, "label": value})
    if s and len(result) < 1:
        raise HTTPException(
            status_code=422,
            detail="Error parsing categorical string",
        )
    return result

cols_normalized = {
    "variable name": "VARIABLENAME",
    "variable label": "VARIABLELABEL", 
    "var type": "VARTYPE",
    "units": "UNITS",
    "categorical": "CATEGORICAL",
    "missing": "MISSING",
    "count": "COUNT",
    "na": "NA",
    "min": "MIN",
    "max": "MAX",
    "formula": "Formula",
    "categoricalvalueconceptcode": "Categorical Value Concept Code",
    "categorical value concept code": "Categorical Value Concept Code",
    "categoricalvaluename": "Categorical Value Name",
    "categorical value name": "Categorical Value Name",
    "categoricalvalueomopid": "Categorical Value OMOP ID",
    "categorical value omop id": "Categorical Value OMOP ID",
    "variableconceptcode": "Variable Concept Code",
    "variable concept code": "Variable Concept Code",
    "variableconceptname": "Variable Concept Name",
    "variable concept name": "Variable Concept Name",
    "variableomopid": "Variable OMOP ID",
    "variable omop id": "Variable OMOP ID",
    "additionalcontextconceptname": "Additional Context Concept Name",
    "additional context concept name": "Additional Context Concept Name",
    "additionalcontextconceptcode": "Additional Context Concept Code",
    "additional context concept code": "Additional Context Concept Code",
    "additionalcontextomopid": "Additional Context OMOP ID",
    "additional context omop id": "Additional Context OMOP ID",
    "unitconceptname": "Unit Concept Name",
    "unit concept name": "Unit Concept Name",
    "unitconceptcode": "Unit Concept Code",
    "unit concept code": "Unit Concept Code",
    "unitomopid": "Unit OMOP ID",
    "unit omop id": "Unit OMOP ID",
    "domain": "Domain",
    "visits": "Visits",
    "visitomopid": "Visit OMOP ID",
    "visit omop id": "Visit OMOP ID",
    "visitconceptname": "Visit Concept Name",
    "visit concept name": "Visit Concept Name",
    "visitconceptcode": "Visit Concept Code",
    "visit concept code": "Visit Concept Code"
}

ACCEPTED_DATATYPES = ["STR", "FLOAT", "INT", "DATETIME"]

def normalize_csv_header(file_content: str) -> str:
    """Normalize CSV header column names using the cols_normalized mapping.
    
    Args:
        file_content: Raw CSV file content as string
        
    Returns:
        CSV content with normalized header line
    """
    lines = file_content.splitlines()
    
    if not lines:
        return file_content
    
    # Normalize column names in header (first line)
    header_columns = lines[0].split(',')
    normalized_columns = []
    
    for col in header_columns:
        # Strip whitespace and convert to lowercase for lookup
        col_clean = col.strip()
        col_lower = col_clean.lower()
        # Look up in normalization dictionary, fallback to original
        normalized_col = cols_normalized.get(col_lower, col_clean)
        normalized_columns.append(normalized_col)
    
    # Replace header with normalized version
    lines[0] = ','.join(normalized_columns)
    
    return '\n'.join(lines)

def validate_metadata_dataframe(df: pd.DataFrame, cohort_id: str) -> list[str]:
    """Validate a metadata dictionary DataFrame and return a list of error messages.
    
    This function contains all validation logic and can be used during upload or report generation.
    
    Args:
        df: DataFrame with normalized column names
        cohort_id: Cohort identifier for error messages
        
    Returns:
        List of error message strings
    """
    errors = []
    
    # Check for duplicate variables
    duplicate_variables = df[df.duplicated(subset=["VARIABLENAME"], keep=False)]
    if not duplicate_variables.empty:
        errors.append(f"Duplicate VARIABLENAME found: {', '.join(duplicate_variables['VARIABLENAME'].unique())}")
    
    # Row-level validation
    for i, row in df.iterrows():
        var_name_for_error = row.get("VARIABLENAME", f"UNKNOWN_VAR_ROW_{i+2}")
        
        # Check if required values are present in rows
        req_fields = ["VARIABLENAME", "VARIABLELABEL", "VARTYPE", "DOMAIN"]
        for rf in req_fields:
            if not str(row.get(rf, "")).strip():
                errors.append(f"Row {i+2} (Variable: '{var_name_for_error}') is missing value for the required field: '{rf}'.")
        
        # Validate VARTYPE
        if row.get("VARTYPE") and str(row["VARTYPE"]).upper() not in ACCEPTED_DATATYPES:
            errors.append(
                f"Row {i+2} (Variable: '{var_name_for_error}') has an invalid data type: '{row['VARTYPE']}'. Accepted types: {', '.join(ACCEPTED_DATATYPES)}."
            )
        
        # Validate DOMAIN
        acc_domains = ["condition_occurrence", "visit_occurrence", "procedure_occurrence", "measurement", "drug_exposure", "device_exposure", "person", "observation", "observation_period", "death", "specimen", "condition_era"]
        if row.get('DOMAIN') and str(row['DOMAIN']).strip().lower() not in acc_domains:
            errors.append(
                f'Row {i+2} (Variable: "{var_name_for_error}") has an invalid domain: "{row["DOMAIN"]}". Accepted domains: {", ".join(acc_domains)}.'
            )
        
        # Variable Concept Code Validation
        if "VARIABLE CONCEPT CODE" in df.columns:
            var_concept_code_str = str(row.get("VARIABLE CONCEPT CODE", "")).strip()
            if var_concept_code_str and var_concept_code_str.lower() != "na":
                # Check for multiple codes separated by "|"
                if "|" in var_concept_code_str:
                    errors.append(
                        f"Row {i+2} (Variable: '{var_name_for_error}'): Multiple concept codes are not allowed in VARIABLE CONCEPT CODE. Found: '{var_concept_code_str}'. Please provide only one concept code."
                    )
                else:
                    # Validate the prefix (normalize to lowercase)
                    try:
                        normalized_code = var_concept_code_str.lower()
                        expanded_uri = curie_converter.expand(normalized_code)
                        if not expanded_uri:
                            errors.append(
                                f"Row {i+2} (Variable: '{var_name_for_error}'): The variable concept code '{var_concept_code_str}' is not valid or its prefix is not recognized. Valid prefixes: {', '.join([record['prefix'] for record in prefix_map if record.get('prefix')])}."
                            )
                    except Exception as curie_exc:
                        error_msg = str(curie_exc)
                        # Check if it's a missing delimiter error
                        if "missing a delimiter" in error_msg.lower():
                            if ":" not in var_concept_code_str:
                                errors.append(
                                    f"Row {i+2} (Variable: '{var_name_for_error}'): The variable concept code '{var_concept_code_str}' is missing a colon (:) separator. Expected format: 'prefix:code' (e.g., 'snomed:12345')."
                                )
                            else:
                                errors.append(
                                    f"Row {i+2} (Variable: '{var_name_for_error}'): The variable concept code '{var_concept_code_str}' is missing a valid prefix before the colon. Expected format: 'prefix:code' (e.g., 'snomed:12345')."
                                )
                        else:
                            errors.append(
                                f"Row {i+2} (Variable: '{var_name_for_error}'): Error expanding CURIE '{var_concept_code_str}': {curie_exc}."
                            )
        
        # Variable Concept OMOP ID - must have at most one value (no '|')
        if "VARIABLE CONCEPT OMOP ID" in df.columns:
            var_omop_id_str = str(row.get("VARIABLE CONCEPT OMOP ID", "")).strip()
            if var_omop_id_str and var_omop_id_str.lower() != "na":
                if "|" in var_omop_id_str:
                    errors.append(
                        f"Row {i+2} (Variable: '{var_name_for_error}'): Multiple OMOP IDs are not allowed in VARIABLE CONCEPT OMOP ID. Found: '{var_omop_id_str}'. Please provide only one OMOP ID."
                    )
        
        # Additional Context Concept Name requires Variable Concept Name
        if "ADDITIONAL CONTEXT CONCEPT NAME" in df.columns and "VARIABLE CONCEPT NAME" in df.columns:
            additional_context_str = str(row.get("ADDITIONAL CONTEXT CONCEPT NAME", "")).strip()
            var_concept_name_str = str(row.get("VARIABLE CONCEPT NAME", "")).strip()
            if additional_context_str and additional_context_str.lower() != "na":
                if not var_concept_name_str or var_concept_name_str.lower() == "na":
                    errors.append(
                        f"Row {i+2} (Variable: '{var_name_for_error}'): ADDITIONAL CONTEXT CONCEPT NAME is provided ('{additional_context_str}'), but VARIABLE CONCEPT NAME is missing."
                    )
        
        # Additional Context - count matching for names, codes, and OMOP IDs
        # All three fields must have matching counts when provided
        additional_names = str(row.get("ADDITIONAL CONTEXT CONCEPT NAME", "")).strip() if "ADDITIONAL CONTEXT CONCEPT NAME" in df.columns else ""
        additional_codes = str(row.get("ADDITIONAL CONTEXT CONCEPT CODE", "")).strip() if "ADDITIONAL CONTEXT CONCEPT CODE" in df.columns else ""
        additional_omop_ids = str(row.get("ADDITIONAL CONTEXT CONCEPT OMOP ID", "")).strip() if "ADDITIONAL CONTEXT CONCEPT OMOP ID" in df.columns else ""
        
        # Count non-empty values in each field
        names_count = len([n for n in additional_names.split("|") if n.strip()]) if additional_names and additional_names.lower() != "na" else 0
        codes_count = len([c for c in additional_codes.split("|") if c.strip()]) if additional_codes and additional_codes.lower() != "na" else 0
        omop_ids_count = len([o for o in additional_omop_ids.split("|") if o.strip()]) if additional_omop_ids and additional_omop_ids.lower() != "na" else 0
        
        # Check if any of the fields are provided
        if names_count > 0 or codes_count > 0 or omop_ids_count > 0:
            # All provided fields must have the same count
            counts = [c for c in [names_count, codes_count, omop_ids_count] if c > 0]
            if len(set(counts)) > 1:
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): The number of ADDITIONAL CONTEXT CONCEPT NAMEs ({names_count}), CODEs ({codes_count}), and OMOP IDs ({omop_ids_count}) must all match."
                )
        
        # Unit concepts validation
        units_value = str(row.get("UNITS", "")).strip() if "UNITS" in df.columns else ""
        
        # Check if any unit concept fields are provided
        unit_code_str = str(row.get("UNIT CONCEPT CODE", "")).strip() if "UNIT CONCEPT CODE" in df.columns else ""
        unit_omop_id_str = str(row.get("UNIT CONCEPT OMOP ID", "")).strip() if "UNIT CONCEPT OMOP ID" in df.columns else ""
        unit_name_str = str(row.get("UNIT CONCEPT NAME", "")).strip() if "UNIT CONCEPT NAME" in df.columns else ""
        
        has_unit_code = unit_code_str and unit_code_str.lower() != "na"
        has_unit_omop_id = unit_omop_id_str and unit_omop_id_str.lower() != "na"
        has_unit_name = unit_name_str and unit_name_str.lower() != "na"
        
        # Unit Concept Code - single value only (no '|')
        if has_unit_code and "|" in unit_code_str:
            errors.append(
                f"Row {i+2} (Variable: '{var_name_for_error}'): Multiple concept codes are not allowed in UNIT CONCEPT CODE. Found: '{unit_code_str}'. Please provide only one concept code."
            )
        
        # Unit Concept OMOP ID - single value only (no '|')
        if has_unit_omop_id and "|" in unit_omop_id_str:
            errors.append(
                f"Row {i+2} (Variable: '{var_name_for_error}'): Multiple OMOP IDs are not allowed in UNIT CONCEPT OMOP ID. Found: '{unit_omop_id_str}'. Please provide only one OMOP ID."
            )
        
        # Check if any unit concept is provided but UNITS field is empty (only if UNITS column exists)
        if (has_unit_code or has_unit_omop_id or has_unit_name):
            if "UNITS" in df.columns and (not units_value or units_value.lower() == "na"):
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): Unit concept fields are provided, but UNITS field is empty."
                )
        
        # Visit concepts validation
        visits_value = str(row.get("VISITS", "")).strip() if "VISITS" in df.columns else ""
        
        # Check if any visit concept fields are provided
        visit_code_str = str(row.get("VISIT CONCEPT CODE", "")).strip() if "VISIT CONCEPT CODE" in df.columns else ""
        visit_omop_id_str = str(row.get("VISIT CONCEPT OMOP ID", "")).strip() if "VISIT CONCEPT OMOP ID" in df.columns else ""
        visit_name_str = str(row.get("VISIT CONCEPT NAME", "")).strip() if "VISIT CONCEPT NAME" in df.columns else ""
        
        has_visit_code = visit_code_str and visit_code_str.lower() != "na"
        has_visit_omop_id = visit_omop_id_str and visit_omop_id_str.lower() != "na"
        has_visit_name = visit_name_str and visit_name_str.lower() != "na"
        
        # Visit Concept Code - single value only (no '|')
        if has_visit_code and "|" in visit_code_str:
            errors.append(
                f"Row {i+2} (Variable: '{var_name_for_error}'): Multiple concept codes are not allowed in VISIT CONCEPT CODE. Found: '{visit_code_str}'. Please provide only one concept code."
            )
        
        # Visit Concept OMOP ID - single value only (no '|')
        if has_visit_omop_id and "|" in visit_omop_id_str:
            errors.append(
                f"Row {i+2} (Variable: '{var_name_for_error}'): Multiple OMOP IDs are not allowed in VISIT CONCEPT OMOP ID. Found: '{visit_omop_id_str}'. Please provide only one OMOP ID."
            )
        
        # Check if any visit concept is provided but VISITS field is empty (only if VISITS column exists)
        if (has_visit_code or has_visit_omop_id or has_visit_name):
            if "VISITS" in df.columns and (not visits_value or visits_value.lower() == "na"):
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): Visit concept fields are provided, but VISITS field is empty."
                )
        
        # Category Concept Validation
        current_categories = row.get("categories")
        
        # Check if any categorical concept field is provided (can have multiple pipe-separated values)
        categories_names_str = str(row.get("CATEGORICAL VALUE CONCEPT NAME", "")).strip() if "CATEGORICAL VALUE CONCEPT NAME" in df.columns else ""
        categories_codes_str = str(row.get("CATEGORICAL VALUE CONCEPT CODE", "")).strip() if "CATEGORICAL VALUE CONCEPT CODE" in df.columns else ""
        categories_omop_ids_str = str(row.get("CATEGORICAL VALUE OMOP ID", "")).strip() if "CATEGORICAL VALUE OMOP ID" in df.columns else ""
        
        has_cat_name = categories_names_str and categories_names_str.lower() != "na"
        has_cat_code = categories_codes_str and categories_codes_str.lower() != "na"
        has_cat_omop_id = categories_omop_ids_str and categories_omop_ids_str.lower() != "na"
        
        # Check if any categorical concept is provided but CATEGORICAL field is empty or invalid
        # Note: Unlike UNITS/VISITS, categorical allows multiple values, so we need to ensure
        # the CATEGORICAL column exists and is properly parsed when concept fields are provided
        if (has_cat_name or has_cat_code or has_cat_omop_id):
            if "CATEGORICAL" in df.columns and (not current_categories or not isinstance(current_categories, list)):
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): Categorical concept fields are provided, but CATEGORICAL field is empty or invalid."
                )
        
        if isinstance(current_categories, list) and current_categories:
            # Get all three categorical concept fields
            categories_names_str = str(row.get("CATEGORICAL VALUE CONCEPT NAME", "")).strip() if "CATEGORICAL VALUE CONCEPT NAME" in df.columns else ""
            categories_codes_str = str(row.get("CATEGORICAL VALUE CONCEPT CODE", "")).strip() if "CATEGORICAL VALUE CONCEPT CODE" in df.columns else ""
            categories_omop_ids_str = str(row.get("CATEGORICAL VALUE OMOP ID", "")).strip() if "CATEGORICAL VALUE OMOP ID" in df.columns else ""
            
            # Parse each field into lists
            categories_names = categories_names_str.split("|") if categories_names_str else []
            categories_codes = categories_codes_str.split("|") if categories_codes_str else []
            categories_omop_ids = categories_omop_ids_str.split("|") if categories_omop_ids_str else []
            
            # Count matching validation
            num_categories = len(current_categories)
            
            if categories_names and len(categories_names) != num_categories:
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): The number of CATEGORICAL VALUE CONCEPT NAMEs ({len(categories_names)}) does not match the number of parsed categories ({num_categories})."
                )
            
            if categories_codes and len(categories_codes) != num_categories:
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): The number of CATEGORICAL VALUE CONCEPT CODEs ({len(categories_codes)}) does not match the number of parsed categories ({num_categories})."
                )
            
            if categories_omop_ids and len(categories_omop_ids) != num_categories:
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}'): The number of CATEGORICAL VALUE OMOP IDs ({len(categories_omop_ids)}) does not match the number of parsed categories ({num_categories})."
                )
            
            # Per-category validation - only validate CURIE format if concept codes are provided
            for idx, category_data in enumerate(current_categories):
                category_value = category_data.get('value', f'Category_{idx}')
                
                # Get the concept code for this category (if provided)
                cat_code = categories_codes[idx].strip() if idx < len(categories_codes) else ""
                has_code = cat_code and cat_code.lower() != "na"
                
                # Validate concept code prefix if provided
                if has_code:
                    try:
                        # Normalize to lowercase before validation
                        normalized_code = cat_code.lower()
                        expanded_uri = curie_converter.expand(normalized_code)
                        if not expanded_uri:
                            errors.append(
                                f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_value}'): The category concept code '{cat_code}' is not valid or its prefix is not recognized. Valid prefixes: {', '.join([record['prefix'] for record in prefix_map if record.get('prefix')])}."
                            )
                    except Exception as curie_exc:
                        error_msg = str(curie_exc)
                        # Check if it's a missing delimiter error
                        if "missing a delimiter" in error_msg.lower():
                            if ":" not in cat_code:
                                errors.append(
                                    f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_value}'): The category concept code '{cat_code}' is missing a colon (:) separator. Expected format: 'prefix:code' (e.g., 'snomed:12345')."
                                )
                            else:
                                errors.append(
                                    f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_value}'): The category concept code '{cat_code}' is missing a valid prefix before the colon. Expected format: 'prefix:code' (e.g., 'snomed:12345')."
                                )
                        else:
                            errors.append(
                                f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_value}'): Error expanding CURIE '{cat_code}': {curie_exc}."
                            )
        elif row.get("CATEGORICAL") and not isinstance(current_categories, list):
            errors.append(
                f"Row {i+2} (Variable: '{var_name_for_error}') has an invalid category: '{row['CATEGORICAL']}'."
            )
    
    return errors


def to_camelcase(s: str) -> str:
    # Special case mappings for variable concept fields
    special_mappings = {
        "VARIABLE CONCEPT CODE": "conceptCode",
        "VARIABLE CONCEPT NAME": "conceptName",
        "VARIABLE OMOP ID": "omopId",
        "VARIABLENAME": "variableName",
        "VARIABLELABEL": "variableLabel",
        "VARTYPE": "varType",
    }
    
    # Check if the uppercase string is in our special mappings
    if s.upper() in special_mappings:
        return special_mappings[s.upper()]
    
    # Otherwise, use the standard camelCase conversion
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])



def load_cohort_dict_file(dict_path: str, cohort_id: str, source: str = "", user_email: str = None, filename: str = None) -> Dataset:
    """Parse the cohort dictionary uploaded as excel or CSV spreadsheet, and load it to the triplestore
    
    Also updates the cohort cache with the new data.
    """
    # Initialize logging variables
    start_time = datetime.now()
    file_size_bytes = 0
    total_rows = 0
    total_variables = 0
    errors = []
    success = False
    graph_triples_count = 0
    g = None  # Initialize graph variable to avoid UnboundLocalError
    
    # Get file information for logging
    if os.path.exists(dict_path):
        file_size_bytes = os.path.getsize(dict_path)
    
    print(f"NOW PROCESSING DICTIONARY FILE FOR COHORT: {cohort_id} \nFile path: {dict_path} - source: {source}")
    
    # Log the start of processing for upload dictionary calls
    if source == "upload_dict":
        logging.info(f"Starting dictionary upload processing for cohort {cohort_id} by user {user_email}. File: {filename}, Size: {file_size_bytes} bytes")
    
    if not dict_path.endswith(".csv"):
        error_msg = "Only CSV files are supported. Please convert your file to CSV and try again."
        if source == "upload_dict":
            # Log the failure
            log_data = {
                "timestamp": start_time,
                "cohort_id": cohort_id,
                "user_email": user_email or "system",
                "filename": filename or os.path.basename(dict_path),
                "file_size_bytes": file_size_bytes,
                "total_rows": 0,
                "total_variables": 0,
                "errors_count": 1,
                "success": False,
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "source": source,
                "graph_triples_count": 0
            }
            log_upload_event_csv(log_data)
            update_upload_log({**log_data, "errors": [error_msg]})
            raise HTTPException(status_code=422, detail=error_msg)
        else:
            # During init_triplestore, log and return empty graph
            print(f"⚠️  SKIPPING cohort {cohort_id}: {error_msg}")
            logging.warning(f"Skipping cohort {cohort_id}: {error_msg}")
            return init_graph()  # Return empty graph
    
    try:
        df = pd.read_csv(dict_path, na_values=[""], keep_default_na=False)
        df = df.dropna(how="all") # Drop rows where all cells are NA
        df = df.fillna("") # Fill remaining NA with empty string
        
        # Capture data metrics for logging
        total_rows = len(df)
        total_variables = len(df['VARIABLENAME'].unique()) if 'VARIABLENAME' in df.columns else 0
        
        # Normalize column names (lowercase for lookup, then use normalized value or uppercase original)
        df.columns = [cols_normalized.get(c.lower().strip(), c.upper().strip()) for c in df.columns]
        # print(f"POST NORMALIZATION -- COHORT {cohort_id} -- Columns: {df.columns}")
        # --- Structural Validation: Check for required columns ---
        # Define columns absolutely essential for the row-processing logic to run without KeyErrors
        # Use actual schema column names (not uppercased) since cols_normalized maps to mixed case
        critical_column_names_for_processing = [c.name for c in metadatadict_cols_schema1]
        # Create lowercase lookup sets for case-insensitive comparison
        critical_cols_lower = {c.lower() for c in critical_column_names_for_processing}
        df_cols_lower = {c.lower() for c in df.columns}
        
        missing_columns = []
        for required_col_name in critical_column_names_for_processing:
            if required_col_name.lower() not in df_cols_lower:
                missing_columns.append(required_col_name)
        
        # Check for extra columns not in the approved list
        extra_columns = []
        for col_name in df.columns:
            if col_name.lower() not in critical_cols_lower:
                extra_columns.append(col_name)
        
        # If critical columns are missing or extra columns exist, reject the upload
        if len(missing_columns) > 0:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        if len(extra_columns) > 0:
            errors.append(f"Unexpected columns found (not in approved list): {', '.join(extra_columns)}")
        
        if len(missing_columns) > 0 or len(extra_columns) > 0:
            if source == "upload_dict":
                raise HTTPException(status_code=422, detail="\n\n".join(errors))

        # --- Content Pre-processing (assuming critical columns are present) ---
        try:
            df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)
        except HTTPException as e: # Catch error from parse_categorical_string
            errors.append(f"Error processing 'CATEGORICAL' column: {e.detail}")
            # If this parsing fails, we might still be able to report other errors, so don't raise immediately
            # unless it's the only error. The final check `if len(errors) > 0` will handle it.

        df["VARTYPE"] = df.apply(lambda row: str(row["VARTYPE"]).upper(), axis=1)

        # --- Content Validation: DataFrame-level and Row-level ---
        # Use centralized validation function
        validation_errors = validate_metadata_dataframe(df, cohort_id)
        errors.extend(validation_errors)


        # --- Final Error Check & Graph Generation ---
        if len(errors) > 0 and source == "upload_dict":
            raise HTTPException(
                status_code=422,
                detail="\n\n".join(errors),
            )

        # If no errors, proceed to create the graph (RDF triples)
        # CRITICAL: Use CMEO graph namespace to match query expectations
        cohort_uri = get_cohort_uri(cohort_id)
        cohort_graph_uri = get_cohort_graph_uri(cohort_id)
        g = init_graph()
        # Use CMEO ontology: cmeo:study instead of icare:Cohort
        g.add((cohort_uri, RDF.type, OntologyNamespaces.CMEO.value.study, cohort_graph_uri))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_graph_uri))
        g.add((cohort_uri, RDFS.label, Literal(cohort_id), cohort_graph_uri))

        i = 0
        for i, row in df.iterrows():
            try:
                # Skip rows with empty/whitespace variable names
                var_name = str(row["VARIABLENAME"]).strip() if pd.notna(row["VARIABLENAME"]) else ""
                if not var_name:
                    logging.warning(f"Skipping row {i+2} in cohort {cohort_id}: empty or whitespace-only variable name")
                    continue
                
                variable_uri = get_var_uri(cohort_id, var_name)
                # Use CMEO ontology: cmeo:data_element instead of icare:Variable
                # Note: CMEO model doesn't use hasVariable relationship, variables are just in the same graph
                g.add((variable_uri, RDF.type, OntologyNamespaces.CMEO.value.data_element, cohort_graph_uri))
                g.add((variable_uri, DC.identifier, Literal(var_name), cohort_graph_uri))
                var_label = str(row["VARIABLELABEL"]).strip() if pd.notna(row["VARIABLELABEL"]) else var_name
                g.add((variable_uri, RDFS.label, Literal(var_label), cohort_graph_uri))

                # Get categories code if provided (re-fetch for graph generation phase)
                categories_codes = []
                if "CATEGORICAL VALUE CONCEPT CODE" in df.columns and str(row.get("CATEGORICAL VALUE CONCEPT CODE","")).strip():
                     categories_codes = str(row["CATEGORICAL VALUE CONCEPT CODE"]).split("|")

                for column_name_from_df, col_value in row.items():
                    #print("in load_cohort_dict_file -- column_name_from_df value:", column_name_from_df)
                    # Use the already normalized column_name_from_df
                    if column_name_from_df not in ["categories"] and col_value: # Exclude our temporary 'categories' column
                        # Use CMEO namespace for properties
                        property_uri = OntologyNamespaces.CMEO.value[to_camelcase(column_name_from_df)]
                        if (
                            isinstance(col_value, str)
                            and (col_value.startswith("http://") or col_value.startswith("https://"))
                            and " " not in col_value
                        ):
                            g.add((variable_uri, property_uri, URIRef(col_value), cohort_graph_uri))
                        else:
                            g.add((variable_uri, property_uri, Literal(col_value), cohort_graph_uri))
                    
                    if column_name_from_df == "categories" and isinstance(col_value, list): # 'categories' is our parsed list
                        for index, category in enumerate(col_value):
                            # Skip categories with empty labels to prevent Invalid IRI errors
                            normalized_label = normalize_text(category['label'])
                            if not normalized_label:
                                continue
                            # Use CMEO model: obi:categorical_value_specification
                            cat_uri = URIRef(f"{variable_uri}/categorical_value_specification/{normalized_label}")
                            g.add((cat_uri, RDF.type, OntologyNamespaces.OBI.value.categorical_value_specification, cohort_graph_uri))
                            g.add((cat_uri, OntologyNamespaces.OBI.value.specifies_value_of, variable_uri, cohort_graph_uri))
                            g.add((cat_uri, OntologyNamespaces.CMEO.value.has_value, Literal(category["value"]), cohort_graph_uri))
                            g.add((cat_uri, RDFS.label, Literal(category["label"]), cohort_graph_uri))
                            
                            if index < len(categories_codes):
                                code_to_check = categories_codes[index].strip()
                                if code_to_check and code_to_check.lower() != "na":
                                    #code_to_check = code_to_check.lower().replace("ucum:%", "ucum:percent").replace("[", "").replace("]", "")
                                    try:
                                        cat_code_uri = curie_converter.expand(code_to_check)
                                        if cat_code_uri: # Only add if valid and expanded
                                            #print(f"Adding category code {cat_code_uri} for category {category['value']} in cohort {cohort_id}, line {i}")
                                            # Another temp fix just for TIM-HF!!
                                            # cat_code_uri = cat_code_uri.lower().replace("ucum/%", "ucum/percent").replace("[", "").replace("]", "")
                                            #print(f"Adding category code {cat_code_uri} for category {category['value']} in cohort {cohort_id}, line {i}, cat_uri: {cat_uri}")
                                            # Store concept code using CMEO model (will be added via standardization process)
                                            g.add((cat_uri, OntologyNamespaces.CMEO.value.conceptId, URIRef(cat_code_uri), cohort_graph_uri))
                                    except Exception as curie_exc:
                                        error_msg = str(curie_exc)
                                        var_name = row.get("VARIABLENAME", f"UNKNOWN_VAR_ROW_{i+2}")
                                        # Check if it's a missing delimiter error
                                        if "missing a delimiter" in error_msg.lower():
                                            if ":" not in code_to_check:
                                                errors.append(
                                                    f"Row {i+2} (Variable: '{var_name}', Category: '{category['value']}'): The category concept code '{code_to_check}' is missing a colon (:) separator. Expected format: 'prefix:code' (e.g., 'snomed:12345')."
                                                )
                                            else:
                                                errors.append(
                                                    f"Row {i+2} (Variable: '{var_name}', Category: '{category['value']}'): The category concept code '{code_to_check}' is missing a valid prefix before the colon. Expected format: 'prefix:code' (e.g., 'snomed:12345')."
                                                )
                                        else:
                                            errors.append(
                                                f"Row {i+2} (Variable: '{var_name}', Category: '{category['value']}'): Error expanding CURIE '{code_to_check}': {str(curie_exc)}."
                                            )
            
            except Exception as row_exc:
                # Catch any error during row processing and provide detailed context
                var_name = row.get("VARIABLENAME", f"UNKNOWN_VAR_ROW_{i+2}")
                error_msg = str(row_exc)
                
                # Try to identify which column/value caused the issue
                detailed_msg = f"Row {i+2} (Variable: '{var_name}'): Error processing row - {error_msg}"
                
                # If it's an IRI error, try to find the problematic value
                if "Invalid IRI" in error_msg or "code point" in error_msg:
                    # Log all non-empty values in the row to help identify the issue
                    problematic_values = []
                    for col_name, col_val in row.items():
                        if col_val and str(col_val).strip():
                            val_str = str(col_val)
                            # Check if value contains problematic characters
                            if any(ord(c) < 32 or ord(c) == 127 for c in val_str):
                                problematic_values.append(f"{col_name}='{val_str}' (contains invalid characters)")
                    
                    if problematic_values:
                        detailed_msg += f"\nProblematic values found: {'; '.join(problematic_values)}"
                
                errors.append(detailed_msg)
                logging.error(f"Error processing row {i+2} for cohort {cohort_id}: {error_msg}", exc_info=True)

        print(f"Finished processing cohort dictionary: {cohort_id}")
        

    except HTTPException as http_exc: # Re-raise specific HTTPExceptions (ours or from parse_categorical_string)
        # Log the collected errors that led to this for server-side records
        logging.warning(f"Validation errors for cohort {cohort_id}:\n{http_exc.detail}")
        if source == "upload_dict":
            raise http_exc
        else:
            # During init_triplestore, log prominently and return empty graph
            print(f"⚠️  SKIPPING cohort {cohort_id} due to validation errors:")
            print(f"    {http_exc.detail}")
            return init_graph()  # Return empty graph
    except pd.errors.EmptyDataError:
        logging.warning(f"Uploaded CSV for cohort {cohort_id} is empty or unreadable.")
        if source == "upload_dict":
            raise HTTPException(status_code=422, detail="The uploaded CSV file is empty or could not be read.")
        else:
            print(f"⚠️  SKIPPING cohort {cohort_id}: CSV file is empty or unreadable")
            return init_graph()  # Return empty graph
    except Exception as e:
        logging.error(f"Unexpected error during dictionary processing for {cohort_id}: {str(e)}", exc_info=True)
        errors.append(f"An unexpected error occurred during file processing: {str(e)}")
        if source == "upload_dict":
            raise HTTPException(
                status_code=500, # Use 500 for truly unexpected server-side issues
                detail="\n\n".join(errors),
            )
        else:
            print(f"⚠️  SKIPPING cohort {cohort_id} due to unexpected error: {str(e)}")
            return init_graph()  # Return empty graph
    
    # Calculate final metrics
    processing_time = (datetime.now() - start_time).total_seconds()
    success = len(errors) == 0
    graph_triples_count = len(g) if g is not None else 0
    
    # Enhanced structured logging for upload dictionary calls
    if source == "upload_dict":
        log_data = {
            "timestamp": start_time,
            "cohort_id": cohort_id,
            "user_email": user_email or "system",
            "filename": filename or os.path.basename(dict_path),
            "file_size_bytes": file_size_bytes,
            "total_rows": total_rows,
            "total_variables": total_variables,
            "errors_count": len(errors),
            "success": success,
            "processing_time_seconds": processing_time,
            "source": source,
            "graph_triples_count": graph_triples_count
        }
        
        # Log to CSV for easy analysis
        log_upload_event_csv(log_data)
        
        # Log to JSON with detailed error information
        detailed_log_data = {**log_data, "errors": errors}
        update_upload_log(detailed_log_data)
        
        # Log completion status
        if success:
            logging.info(f"Successfully processed dictionary upload for cohort {cohort_id}. Processed {total_variables} variables from {total_rows} rows in {processing_time:.2f}s. Generated {graph_triples_count} triples.")
        else:
            logging.warning(f"Dictionary upload for cohort {cohort_id} completed with {len(errors)} errors. Processing time: {processing_time:.2f}s")


    # Update the cohort cache directly from the graph data
    # This is more efficient than retrieving it from the triplestore later
    # Only update cache if this is called from upload API, not from init_triplestore
    if source == "upload_dict" and g is not None:
        from src.cohort_cache import create_cohort_from_dict_file
        cohort_uri = get_cohort_uri(cohort_id)
        create_cohort_from_dict_file(cohort_id, cohort_uri, g)
        logging.info(f"Added cohort {cohort_id} to cache directly from dictionary file")
    else:
        logging.info(f"Skipping cache update for cohort {cohort_id} - cache managed by {source}")

    return g





@router.post(
    "/normalize-all-dictionary-headers",
    name="Normalize all cohort dictionary headers",
    response_description="Normalization report file",
)
async def normalize_all_dictionary_headers(
    user: Any = Depends(get_current_user),
):
    """Normalize column headers for all existing cohort dictionaries.
    
    This endpoint:
    - Finds the latest dictionary file for each cohort
    - Records the original header
    - Applies column name normalization
    - Saves the normalized CSV
    - Generates a report of changes made
    
    Admins only.
    """
    user_email = user["email"]
    if user_email not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    
    from fastapi.responses import FileResponse
    from src.cohort_cache import get_cohorts_from_cache
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(settings.data_folder, f"header_normalization_report_{timestamp}.txt")
    
    # Initialize the report file
    with open(report_file, "w") as f:
        f.write(f"Dictionary Header Normalization Report - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Performed by: {user_email}\n")
        f.write("=" * 80 + "\n\n")
    
    # Get all cohorts
    all_cohorts = get_cohorts_from_cache(user_email)
    
    total_cohorts = 0
    cohorts_normalized = 0
    cohorts_no_change = 0
    cohorts_without_dict = 0
    
    for cohort_id in sorted(all_cohorts.keys()):
        total_cohorts += 1
        cohort_folder_path = os.path.join(settings.cohort_folder, cohort_id)
        
        if not os.path.exists(cohort_folder_path):
            cohorts_without_dict += 1
            continue
        
        # Get the latest metadata dictionary file
        latest_dict_file = get_latest_datadictionary(cohort_folder_path)
        
        if not latest_dict_file:
            cohorts_without_dict += 1
            with open(report_file, "a") as f:
                f.write(f"COHORT: {cohort_id}\n")
                f.write(f"Status: No metadata dictionary file found\n")
                f.write(f"Folder: {cohort_folder_path}\n")
                f.write("-" * 80 + "\n\n")
            continue
        
        # Read the file
        try:
            with open(latest_dict_file, "r", encoding='utf-8') as f:
                file_content = f.read()
            
            # Record original header
            original_header = file_content.splitlines()[0] if file_content.splitlines() else ""
            
            # Apply normalization
            normalized_content = normalize_csv_header(file_content)
            
            # Get normalized header
            normalized_header = normalized_content.splitlines()[0] if normalized_content.splitlines() else ""
            
            # Check if anything changed
            if original_header == normalized_header:
                cohorts_no_change += 1
                with open(report_file, "a") as f:
                    f.write(f"COHORT: {cohort_id}\n")
                    f.write(f"File: {os.path.basename(latest_dict_file)}\n")
                    f.write(f"Status: No changes needed\n")
                    f.write(f"Header: {original_header}\n")
                    f.write("-" * 80 + "\n\n")
            else:
                cohorts_normalized += 1
                
                # Save the normalized file
                with open(latest_dict_file, "w", encoding='utf-8') as f:
                    f.write(normalized_content)
                
                # Record changes in report
                with open(report_file, "a") as f:
                    f.write(f"COHORT: {cohort_id}\n")
                    f.write(f"File: {os.path.basename(latest_dict_file)}\n")
                    f.write(f"Status: Headers normalized\n")
                    f.write(f"Original header:\n  {original_header}\n")
                    f.write(f"Normalized header:\n  {normalized_header}\n")
                    
                    # Show column-by-column changes
                    original_cols = [c.strip() for c in original_header.split(',')]
                    normalized_cols = [c.strip() for c in normalized_header.split(',')]
                    
                    changes = []
                    for i, (orig, norm) in enumerate(zip(original_cols, normalized_cols)):
                        if orig != norm:
                            changes.append(f"  Column {i+1}: '{orig}' → '{norm}'")
                    
                    if changes:
                        f.write(f"Changes:\n")
                        f.write('\n'.join(changes) + '\n')
                    
                    f.write("-" * 80 + "\n\n")
                
                logging.info(f"Normalized headers for cohort {cohort_id}: {os.path.basename(latest_dict_file)}")
        
        except Exception as e:
            logging.error(f"Error normalizing headers for cohort {cohort_id}: {str(e)}", exc_info=True)
            with open(report_file, "a") as f:
                f.write(f"COHORT: {cohort_id}\n")
                f.write(f"File: {os.path.basename(latest_dict_file)}\n")
                f.write(f"Status: ERROR - {str(e)}\n")
                f.write("-" * 80 + "\n\n")
    
    # Write summary
    with open(report_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total cohorts processed: {total_cohorts}\n")
        f.write(f"Cohorts with normalized headers: {cohorts_normalized}\n")
        f.write(f"Cohorts with no changes needed: {cohorts_no_change}\n")
        f.write(f"Cohorts without dictionary files: {cohorts_without_dict}\n")
    
    logging.info(f"Header normalization completed. Report: {report_file}")
    
    return FileResponse(
        path=report_file,
        filename=f"header_normalization_report_{timestamp}.txt",
        media_type="text/plain"
    )


@router.post(
    "/generate-metadata-issues-report",
    name="Generate metadata issues report",
    response_description="Metadata issues report file",
)
async def generate_metadata_issues_report():
    """Generate a report of all metadata dictionary validation issues across all cohorts.
    
    This endpoint can be called without authentication as it's typically used during startup.
    Returns the report as a downloadable text file.
    """
    from fastapi.responses import FileResponse
    from src.cohort_cache import get_cohorts_from_cache
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports_folder = os.path.join(settings.data_folder, "DICTIONARY_ISSUES_REPORTS")
    errors_file = os.path.join(reports_folder, f"metadata_files_issues_{timestamp}.txt")
    
    # Ensure directory exists
    os.makedirs(reports_folder, exist_ok=True)
    
    # Initialize the report file
    with open(errors_file, "w") as f:
        f.write(f"Metadata Issues Report - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    # Get all cohorts and process their dictionaries
    admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
    all_cohorts = get_cohorts_from_cache(admin_email)
    
    total_cohorts = 0
    cohorts_with_errors = 0
    cohorts_without_dict = 0
    total_errors = 0
    
    for cohort_id in sorted(all_cohorts.keys()):
        total_cohorts += 1
        cohort_folder_path = os.path.join(settings.cohort_folder, cohort_id)
        
        if not os.path.exists(cohort_folder_path):
            cohorts_without_dict += 1
            continue
        
        # Get the latest metadata dictionary file
        latest_dict_file = get_latest_datadictionary(cohort_folder_path)
        
        if not latest_dict_file:
            cohorts_without_dict += 1
            with open(errors_file, "a") as f:
                f.write(f"COHORT: {cohort_id}\n")
                f.write(f"Status: No metadata dictionary file found\n")
                f.write(f"Folder: {cohort_folder_path}\n")
                f.write("-" * 80 + "\n\n")
            continue
        
        # Try to validate the dictionary file
        try:
            # Read and validate the CSV
            df = pd.read_csv(latest_dict_file, na_values=[""], keep_default_na=False)
            df = df.dropna(how="all")
            df = df.fillna("")
            
            # Normalize column names (lowercase for lookup, then use normalized value or uppercase original)
            from src.upload import cols_normalized
            df.columns = [cols_normalized.get(c.lower().strip(), c.upper().strip()) for c in df.columns]
            
            # Check for required columns (case-insensitive)
            from src.upload import metadatadict_cols_schema1
            critical_column_names = [c.name for c in metadatadict_cols_schema1]
            critical_cols_lower = {c.lower() for c in critical_column_names}
            df_cols_lower = {c.lower() for c in df.columns}
            missing_columns = [col for col in critical_column_names if col.lower() not in df_cols_lower]
            
            # Check for extra columns not in the approved list (case-insensitive)
            extra_columns = [col for col in df.columns if col.lower() not in critical_cols_lower]
            
            # Collect validation errors
            errors = []
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            if extra_columns:
                errors.append(f"Unexpected columns found (not in approved list): {', '.join(extra_columns)}")
            
            # Parse categories for validation
            try:
                df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)
            except:
                pass  # If parsing fails, validation will catch it
            
            # Uppercase VARTYPE for validation
            df["VARTYPE"] = df.apply(lambda row: str(row.get("VARTYPE", "")).upper(), axis=1)
            
            # Use centralized validation function
            validation_errors = validate_metadata_dataframe(df, cohort_id)
            errors.extend(validation_errors)
            
            # Write results
            if errors:
                cohorts_with_errors += 1
                total_errors += len(errors)
                with open(errors_file, "a") as f:
                    f.write(f"COHORT: {cohort_id}\n")
                    f.write(f"File: {os.path.basename(latest_dict_file)}\n")
                    f.write(f"Modified: {datetime.fromtimestamp(os.path.getmtime(latest_dict_file)).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Errors found: {len(errors)}\n\n")
                    for idx, error in enumerate(errors, 1):
                        f.write(f"  {idx}. {error}\n")
                    f.write("-" * 80 + "\n\n")
            
        except Exception as e:
            cohorts_with_errors += 1
            total_errors += 1
            with open(errors_file, "a") as f:
                f.write(f"COHORT: {cohort_id}\n")
                f.write(f"File: {os.path.basename(latest_dict_file)}\n")
                f.write(f"Error: Failed to process file - {str(e)}\n")
                f.write("-" * 80 + "\n\n")
    
    # Write summary
    with open(errors_file, "a") as f:
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total cohorts: {total_cohorts}\n")
        f.write(f"Cohorts with errors: {cohorts_with_errors}\n")
        f.write(f"Cohorts without dictionary: {cohorts_without_dict}\n")
        f.write(f"Total errors: {total_errors}\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logging.info(f"Generated metadata issues report: {errors_file}")
    logging.info(f"Report summary - Total: {total_cohorts}, Errors: {cohorts_with_errors}, No dict: {cohorts_without_dict}")
    
    # Return the file as a download
    return FileResponse(
        path=errors_file,
        media_type="text/plain",
        filename=os.path.basename(errors_file),
        headers={
            "X-Total-Cohorts": str(total_cohorts),
            "X-Cohorts-With-Errors": str(cohorts_with_errors),
            "X-Cohorts-Without-Dict": str(cohorts_without_dict),
            "X-Total-Errors": str(total_errors)
        }
    )


@router.post(
    "/get-logs",
    name="Get logs",
    response_description="Logs",
)
async def get_logs(
    user: Any = Depends(get_current_user),
) -> list[str]:
    """Get server logs (admins only)."""
    user_email = user["email"]
    if user_email not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    with open(settings.logs_filepath) as log_file:
        logs = log_file.read()
    return logs.split("\n")
    # return {
    #     "message": f"Cohort {cohort_id} has been successfully deleted.",
    # }


@router.post(
    "/clear-cache",
    name="Clear the cohort cache",
    response_description="Cache clear result",
)
async def clear_cohort_cache(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Clear the cohort cache (in-memory, disk files, and timestamps). Admins only."""
    user_email = user["email"]
    if user_email not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    
    from src.cohort_cache import clear_cache, get_cache_file_path, get_cache_timestamp_file
    
    # Clear in-memory cache and remove cache file
    clear_cache()
    
    # Remove timestamp file if it exists
    timestamp_file = get_cache_timestamp_file()
    if timestamp_file.exists():
        os.remove(timestamp_file)
        logging.info(f"Removed cache timestamp file {timestamp_file}")
    
    # Remove any stale lock files
    lock_file_path = os.path.join(settings.data_folder, ".cache_init.lock")
    write_lock_path = os.path.join(settings.data_folder, ".cache_write.lock")
    
    for lock_path in [lock_file_path, write_lock_path]:
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
                logging.info(f"Removed lock file {lock_path}")
            except Exception as e:
                logging.warning(f"Could not remove lock file {lock_path}: {e}")
    
    logging.info(f"Cache cleared by admin user {user_email}")
    return {
        "message": "Cache has been successfully cleared. It will be re-initialized on the next API request.",
        "cleared_by": user_email
    }


@router.post(
    "/refresh-cache",
    name="Refresh the cohort cache from triplestore",
    response_description="Cache refresh result",
)
async def refresh_cohort_cache(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Regenerate the cohort cache from the triplestore. Admins only.
    
    This endpoint forces a complete refresh of the cache by re-reading all cohort
    metadata from the triplestore. Useful when the cache gets out of sync or after
    manual changes to the triplestore.
    """
    user_email = user["email"]
    if user_email not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    
    from src.cohort_cache import initialize_cache_from_triplestore
    
    try:
        # Force refresh the cache from triplestore
        initialize_cache_from_triplestore(user_email, force_refresh=True)
        
        logging.info(f"Cache refreshed from triplestore by admin user {user_email}")
        return {
            "message": "Cache has been successfully refreshed from the triplestore.",
            "refreshed_by": user_email
        }
    except Exception as e:
        logging.error(f"Error refreshing cache: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh cache: {str(e)}"
        )


@router.post(
    "/delete-cohort",
    name="Delete a cohort from the database",
    response_description="Delete result",
)
async def delete_cohort(
    user: Any = Depends(get_current_user),
    cohort_id: str = Form(...),
) -> dict[str, Any]:
    """Delete a cohort from the triplestore and delete its metadata file from the server."""
    user_email = user["email"]
    if user_email not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    delete_existing_triples(
        get_cohort_mapping_uri(cohort_id), f"<{get_cohort_uri(cohort_id)!s}>", "icare:previewEnabled"
    )
    delete_existing_triples(get_cohort_graph_uri(cohort_id))
    # Delete folder
    cohort_folder_path = os.path.join(settings.data_folder, "cohorts", cohort_id)
    if os.path.exists(cohort_folder_path) and os.path.isdir(cohort_folder_path):
        shutil.rmtree(cohort_folder_path)
    
    # Remove the cohort from the cache
    from src.cohort_cache import remove_cohort_from_cache
    remove_cohort_from_cache(cohort_id)
    logging.info(f"Removed cohort {cohort_id} from cache after deletion")
    
    return {
        "message": f"Cohort {cohort_id} has been successfully deleted.",
    }



@router.post(
    "/upload-cohort",
    name="Upload cohort metadata file",
    response_description="Upload result",
)
async def upload_cohort(
    background_tasks: BackgroundTasks,
    user: Any = Depends(get_current_user),
    # cohort_id: str = Form(..., pattern="^[a-zA-Z0-9-_\w]+$"),
    cohort_id: str = Form(...),
    cohort_dictionary: UploadFile = File(...),
    cohort_data: UploadFile | None = None,
) -> dict[str, Any]:
    """Upload a cohort metadata file to the server and add its variables to the triplestore."""
    user_email = user["email"]
    # Use cache instead of SPARQL query for better performance
    from src.cohort_cache import get_cohorts_from_cache
    cohorts = get_cohorts_from_cache(user_email)
    cohort_info = cohorts.get(cohort_id)
    if not cohort_info:
        raise HTTPException(
            status_code=403,
            detail=f"Cohort ID {cohort_id} does not exists",
        )
    if not cohort_info.can_edit:
        raise HTTPException(
            status_code=403,
            detail=f"User {user_email} cannot edit cohort {cohort_id}",
        )

    # Create directory named after cohort_id
    os.makedirs(cohort_info.folder_path, exist_ok=True)
    # Check if cohort already uploaded
    if cohort_info and len(cohort_info.variables) > 0:
        # Check for existing data dictionary file and back it up
        for file_name in os.listdir(cohort_info.folder_path):
            if file_name.endswith("_datadictionary.csv"):
                # Construct the backup file name with the current date
                backup_file_name = f"{file_name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backup_file_path = os.path.join(cohort_info.folder_path, backup_file_name)
                existing_file_path = os.path.join(cohort_info.folder_path, file_name)
                # Rename (backup) the existing file
                os.rename(existing_file_path, backup_file_path)
                break  # Assuming there's only one data dictionary file per cohort

    # Make sure metadata file ends with _datadictionary
    metadata_filename = cohort_dictionary.filename
    filename, ext = os.path.splitext(metadata_filename)
    if not filename.endswith("_datadictionary"):
        filename += "_datadictionary"

    # Store metadata file on disk in the cohorts folder with normalized headers
    metadata_path = os.path.join(cohort_info.folder_path, filename + ext)
    
    # Read the uploaded file content and normalize just the header line
    cohort_dictionary.file.seek(0)
    file_content = cohort_dictionary.file.read().decode('utf-8')
    
    # Normalize the header using the isolated function
    normalized_content = normalize_csv_header(file_content)
    
    # Write the file with normalized header
    with open(metadata_path, "w", encoding='utf-8') as f:
        f.write(normalized_content)

    try:
        g = load_cohort_dict_file(metadata_path, cohort_id, source="upload_dict", user_email=user_email, filename=metadata_filename)
        # Airlock preview setting goes to mapping graph because it is defined in the explorer UI
        # g.add(
        #     (
        #         get_cohort_uri(cohort_id),
        #         ICARE.previewEnabled,
        #         Literal(str(airlock).lower(), datatype=XSD.boolean),
        #         get_cohort_mapping_uri(cohort_id),
        #     )
        # )

        # TODO: waiting for more tests before sending to production
        # if settings.dev_mode:
        if False:
            background_tasks.add_task(generate_mappings, cohort_id, metadata_path, g)
        else:
            # Delete previous graph for this file from triplestore
            # TODO: remove these lines once we move to generating mapping through the background task
            # delete_existing_triples(
            #     get_cohort_mapping_uri(cohort_id), f"<{get_cohort_uri(cohort_id)!s}>", "icare:previewEnabled"
            # )
            delete_existing_triples(get_cohort_graph_uri(cohort_id))
            if publish_graph_to_endpoint(g):
                # Cache was already updated directly from the graph in load_cohort_dict_file
                logging.info(f"Cohort {cohort_id} published to triplestore and cache updated")
    except Exception as e:
        os.remove(cohort_info.metadata_filepath)
        raise e

    # return {
    #     "message": f"Metadata for cohort {cohort_id} have been successfully uploaded. The variables are being mapped to standard codes and will be available in the Cohort Explorer in a few minutes.",
    #     "identifier": cohort_id,
    #     # **cohort.dict(),
    # }
    return {
        "message": f"Metadata for cohort {cohort_id} have been successfully uploaded.",
        "identifier": cohort_id,
    }



def generate_mappings(cohort_id: str, metadata_path: str, g: Graph) -> None:
    """Function to generate mappings for a cohort and publish them to the triplestore running as background task"""
    print(f"Generating mappings for cohort {cohort_id}")
    map_csv_to_standard_codes(metadata_path)
    # delete_existing_triples(
    #     get_cohort_mapping_uri(cohort_id), f"<{get_cohort_uri(cohort_id)!s}>", "icare:previewEnabled"
    # )
    delete_existing_triples(get_cohort_graph_uri(cohort_id))
    if publish_graph_to_endpoint(g):
        # Update the cache with the new cohort data
        # Use admin email to ensure we can access all cohorts
        cohorts = retrieve_cohorts_metadata("admin@example.com")
        if cohort_id in cohorts:
            add_cohort_to_cache(cohorts[cohort_id])
            logging.info(f"Added cohort {cohort_id} to cache after generating mappings")


@router.post(
    "/create-provision-dcr",
    name="Create Data Clean Room to provision the dataset",
    response_description="Creation result",
)
async def post_create_provision_dcr(
    user: Any = Depends(get_current_user),
    # cohort_id: str = Form(..., pattern="^[a-zA-Z0-9-_\w]+$"),
    cohort_id: str = Form(...),
) -> dict[str, Any]:
    import time
    t0 = time.time()
    # Use cache instead of SPARQL query for better performance
    from src.cohort_cache import get_cohorts_from_cache
    cohorts = get_cohorts_from_cache(user["email"])
    logging.info(f"[TIMING] Retrieved cohorts from cache in {time.time() - t0:.3f}s")
    
    cohort_info = cohorts.get(cohort_id)
    if not cohort_info:
        raise HTTPException(
            status_code=403,
            detail=f"Cohort ID {cohort_id} does not exists",
        )
    if not cohort_info.can_edit:
        raise HTTPException(
            status_code=403,
            detail=f"User {user['email']} cannot publish cohort {cohort_id}",
        )
    try:
        dcr_data = create_provision_dcr(user, cohort_info)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"There was an issue when uploading the cohort {cohort_id} to Decentriq: {e}",
        )
    return dcr_data


COHORTS_METADATA_FILEPATH = os.path.join(settings.data_folder, "iCARE4CVD_Cohorts.xlsx")


@router.post(
    "/upload-cohorts-metadata",
    name="Upload metadata file for all cohorts",
    response_description="Upload result",
)
async def upload_cohorts_metadata(
    background_tasks: BackgroundTasks,
    user: Any = Depends(get_current_user),
    cohorts_metadata: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload the file with all cohorts metadata to the server and triplestore."""
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    with open(COHORTS_METADATA_FILEPATH, "wb") as buffer:
        shutil.copyfileobj(cohorts_metadata.file, buffer)
    g = cohorts_metadata_file_to_graph(COHORTS_METADATA_FILEPATH)
    if len(g) > 0:
        delete_existing_triples(ICARE["graph/metadata"])
        publish_graph_to_endpoint(g)
        
        # Refresh the cache in background to avoid blocking the response
        from src.cohort_cache import initialize_cache_from_triplestore
        background_tasks.add_task(initialize_cache_from_triplestore, user["email"], True)
        logging.info("Cache refresh scheduled after cohorts metadata update")
    
    return {
        "message": "Cohorts metadata file has been successfully uploaded and processed. The cache will be refreshed in the background and changes will be visible in a few minutes.",
        "triples_count": len(g)
    }


def is_valid_value(value: Any) -> bool:
    """
    Check if a value is valid (not empty and not 'Not Applicable')
    Handles various data types, not just strings.
    """
    # Handle None, empty values, NaN
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return False
    
    # Convert to string for text-based checks
    try:
        # Handle numeric values that should be considered valid
        if isinstance(value, (int, float)):
            # Zero is considered valid
            return True
        # Convert to string for text comparison
        str_value = str(value).strip().lower()
        if str_value == "" or str_value == "not applicable" or str_value == "nan":
            return False
        return True
    except:
        # If any error occurs during conversion, consider it invalid
        return False

def cohorts_metadata_file_to_graph(filepath: str) -> Dataset:
    df = pd.read_excel(filepath, sheet_name="Descriptions")
    df = df.fillna("")
    # Convert column names to lowercase for consistency
    df.columns = df.columns.str.lower()
    g = init_graph()
    metadata_graph = URIRef(OntologyNamespaces.CMEO.value + "graph/studies_metadata")
    
    for _i, row in df.iterrows():
        print("now processing cohorts' metadata row: ", _i, row)
        cohort_id = str(row["study name"]).strip()
        # print(cohort_id)
        cohort_uri = get_cohort_uri(cohort_id)
        # Use CMEO ontology: cmeo:study instead of icare:Cohort
        g.add((cohort_uri, RDF.type, OntologyNamespaces.CMEO.value.study, metadata_graph))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), metadata_graph))
        g.add((cohort_uri, RDFS.label, Literal(cohort_id), metadata_graph))
        # Study design execution - needed for SPARQL query to find organization
        study_design_execution_uri = URIRef(cohort_uri + "/study_design_execution")
        g.add((study_design_execution_uri, RDF.type, OntologyNamespaces.OBI.value.study_design_execution, metadata_graph))
        
        # Institute/Organization - create OBI structure matching study_kg.py
        if is_valid_value(row.get("institute", "")):
            organization_uri = URIRef(cohort_uri + "/institute")
            g.add((organization_uri, RDF.type, OntologyNamespaces.OBI.value.organization, metadata_graph))
            g.add((study_design_execution_uri, OntologyNamespaces.RO.value.has_participant, organization_uri, metadata_graph))
            g.add((organization_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["institute"], datatype=XSD.string), metadata_graph))
            # Also keep the simple predicate for backward compatibility
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.institution, Literal(row["institute"]), metadata_graph))
            
            # Study contact person - create OBI structure matching study_kg.py
            if is_valid_value(row.get("study contact person", "")):
                contact_person_name = row["study contact person"]
                contact_uri = URIRef(cohort_uri + "/" + normalize_text(contact_person_name))
                study_contact_person_role_uri = URIRef(cohort_uri + "/study_contact_person_role")
                
                # Person entity
                g.add((contact_uri, RDF.type, OntologyNamespaces.NCBI.value.homo_sapiens, metadata_graph))
                g.add((organization_uri, OntologyNamespaces.OBI.value.has_member, contact_uri, metadata_graph))
                g.add((contact_uri, OntologyNamespaces.OBI.value.member_of, organization_uri, metadata_graph))
                g.add((contact_uri, OntologyNamespaces.CMEO.value.has_value, Literal(contact_person_name, datatype=XSD.string), metadata_graph))
                
                # Role entity
                g.add((study_contact_person_role_uri, RDF.type, OntologyNamespaces.CMEO.value.study_contact_person_role, metadata_graph))
                g.add((contact_uri, OntologyNamespaces.RO.value.has_role, study_contact_person_role_uri, metadata_graph))
                g.add((study_contact_person_role_uri, OntologyNamespaces.RO.value.role_of, contact_uri, metadata_graph))
                g.add((study_contact_person_role_uri, OntologyNamespaces.CMEO.value.has_value, Literal(contact_person_name, datatype=XSD.string), metadata_graph))
                
                # Email entity for contact person
                if is_valid_value(row.get("study contact person email address", "")):
                    contact_email = row["study contact person email address"].strip()
                    email_uri = URIRef(study_contact_person_role_uri + "/email_address")
                    g.add((email_uri, RDF.type, OntologyNamespaces.OBI.value.email_address, metadata_graph))
                    g.add((email_uri, OntologyNamespaces.IAO.value.is_about, contact_uri, metadata_graph))
                    g.add((email_uri, OntologyNamespaces.CMEO.value.has_value, Literal(contact_email, datatype=XSD.string), metadata_graph))
                    # Also add to cmeo:email for permissions
                    for email in contact_email.split(";"):
                        g.add((cohort_uri, OntologyNamespaces.CMEO.value.email, Literal(email.strip().lower()), metadata_graph))
            
            # Administrator - create OBI structure matching study_kg.py
            if is_valid_value(row.get("administrator", "")):
                admin_name = row["administrator"]
                administrator_person_uri = URIRef(cohort_uri + "/" + normalize_text(admin_name))
                
                # Person entity
                g.add((administrator_person_uri, RDF.type, OntologyNamespaces.NCBI.value.homo_sapiens, metadata_graph))
                g.add((administrator_person_uri, OntologyNamespaces.OBI.value.member_of, organization_uri, metadata_graph))
                g.add((administrator_person_uri, OntologyNamespaces.CMEO.value.has_value, Literal(admin_name, datatype=XSD.string), metadata_graph))
                
                # Role entity
                administrator_role_uri = URIRef(cohort_uri + "/administrator_role")
                g.add((administrator_role_uri, RDF.type, OntologyNamespaces.CMEO.value.administrator_role, metadata_graph))
                g.add((administrator_person_uri, OntologyNamespaces.RO.value.has_role, administrator_role_uri, metadata_graph))
                g.add((administrator_role_uri, OntologyNamespaces.RO.value.role_of, administrator_person_uri, metadata_graph))
                g.add((administrator_role_uri, OntologyNamespaces.CMEO.value.has_value, Literal(admin_name, datatype=XSD.string), metadata_graph))
                
                # Email entity for administrator
                if is_valid_value(row.get("administrator email address", "")):
                    admin_email = row["administrator email address"].strip()
                    email_uri = URIRef(administrator_person_uri + "/email_address")
                    g.add((email_uri, RDF.type, OntologyNamespaces.OBI.value.email_address, metadata_graph))
                    g.add((email_uri, OntologyNamespaces.IAO.value.is_about, administrator_person_uri, metadata_graph))
                    g.add((email_uri, OntologyNamespaces.CMEO.value.has_value, Literal(admin_email, datatype=XSD.string), metadata_graph))
                    # Also add to cmeo:email for permissions
                    for email in admin_email.split(";"):
                        g.add((cohort_uri, OntologyNamespaces.CMEO.value.email, Literal(email.strip().lower()), metadata_graph))
        # References
        if is_valid_value(row.get("references", "")):
            for reference in row["references"].split(";"):
                g.add((cohort_uri, OntologyNamespaces.CMEO.value.references, Literal(reference.strip()), metadata_graph))
                
        # Additional metadata fields
        if is_valid_value(row.get("population location", "")):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.populationLocation, Literal(row["population location"]), metadata_graph))
        if is_valid_value(row.get("language", "")):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.language, Literal(row["language"]), metadata_graph))
        if is_valid_value(row.get("frequency of data collection", "")):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.dataCollectionFrequency, Literal(row["frequency of data collection"]), metadata_graph))
        if is_valid_value(row.get("interventions", "")):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.interventions, Literal(row["interventions"]), metadata_graph))
        if is_valid_value(row["study type"]):
            # Split study types on '/' and add each as a separate triple
            study_types = [st.strip() for st in row["study type"].split("/")]
            for study_type in study_types:
                g.add((cohort_uri, OntologyNamespaces.CMEO.value.cohortType, Literal(study_type), metadata_graph))
        if is_valid_value(row["study design"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyType, Literal(row["study design"]), metadata_graph))
        #if is_valid_value(row["Study duration"]):
        #    g.add((cohort_uri, ICARE.studyDuration, Literal(row["Study duration"]), cohorts_graph))
        if is_valid_value(row["start date"]) and is_valid_value(row["end date"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyStart, Literal(row["start date"]), metadata_graph))
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyEnd, Literal(row["end date"]), metadata_graph))
        if is_valid_value(row["number of participants"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyParticipants, Literal(row["number of participants"]), metadata_graph))
        if is_valid_value(row["ongoing"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyOngoing, Literal(row["ongoing"]), metadata_graph))
        #if is_valid_value(row["Patient population"]):
        #    g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyPopulation, Literal(row["Patient population"]), metadata_graph))
        if is_valid_value(row["study objective"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.studyObjective, Literal(row["study objective"]), metadata_graph))
            
        # Handle primary outcome specification
        if "primary outcome specification" in row and is_valid_value(row["primary outcome specification"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.primaryOutcomeSpec, Literal(row["primary outcome specification"]), metadata_graph))
            
        # Handle secondary outcome specification
        if "secondary outcome specification" in row and is_valid_value(row["secondary outcome specification"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.secondaryOutcomeSpec, Literal(row["secondary outcome specification"]), metadata_graph))
            
        # Handle morbidity
        if "morbidity" in row and is_valid_value(row["morbidity"]):
            g.add((cohort_uri, OntologyNamespaces.CMEO.value.morbidity, Literal(row["morbidity"]), metadata_graph))
            
        # Keep original study name for identifier, normalized for URI
        original_study_name = str(row["study name"]).strip()
        study_name = normalize_text(original_study_name)
        study_uri = URIRef(OntologyNamespaces.CMEO.value + study_name)
        study_design_execution_uri = URIRef(study_uri + "/study_design_execution")
        
        # Create study design execution entity (without dc:identifier - it goes on study_design per query expectations)
        g.add((study_design_execution_uri, RDF.type, OntologyNamespaces.OBI.value.study_design_execution, metadata_graph))
        
        # Create study design and protocol structure
        study_design_value = row.get("study design", "").lower().strip() if pd.notna(row.get("study design", "")) else None
        if study_design_value:
            study_design_value = normalize_text(study_design_value)
            study_design_uri = URIRef(study_uri + "/" + study_design_value)
            g.add((study_design_uri, OntologyNamespaces.CMEO.value.has_value, Literal(study_design_value, datatype=XSD.string), metadata_graph))
            g.add((study_design_uri, DC.identifier, Literal(original_study_name, datatype=XSD.string), metadata_graph))
            dynamic_class_uri = URIRef(OntologyNamespaces.OBI.value + study_design_value)
            g.add((study_design_uri, RDF.type, dynamic_class_uri, metadata_graph))
            protocol_uri = URIRef(study_uri + "/" + study_design_value + "/protocol")
            g.add((protocol_uri, RDF.type, OntologyNamespaces.OBI.value.protocol, metadata_graph))
            g.add((study_design_uri, OntologyNamespaces.RO.value.has_part, protocol_uri, metadata_graph))
        else:
            study_design_uri = URIRef(study_uri + "/study_design")
            protocol_uri = URIRef(study_uri + "/protocol")
            g.add((protocol_uri, RDF.type, OntologyNamespaces.OBI.value.protocol, metadata_graph))
            g.add((study_design_uri, RDF.type, OntologyNamespaces.OBI.value.study_design, metadata_graph))
            g.add((study_design_uri, OntologyNamespaces.RO.value.has_part, protocol_uri, metadata_graph))
            g.add((study_design_uri, DC.identifier, Literal(original_study_name, datatype=XSD.string), metadata_graph))
            
        g.add((study_design_execution_uri, OntologyNamespaces.RO.value.concretizes, study_design_uri, metadata_graph))
        g.add((study_design_uri, OntologyNamespaces.RO.value.is_concretized_by, study_design_execution_uri, metadata_graph))
        
        # Create study design variable specification
        study_design_variable_specification_uri = URIRef(study_uri + "/study_design_variable_specification")
        g.add((study_design_variable_specification_uri, RDFS.label, Literal("study design variable specification", datatype=XSD.string), metadata_graph))
        g.add((study_design_variable_specification_uri, RDF.type, OntologyNamespaces.CMEO.value.study_design_variable_specification, metadata_graph))
        g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, study_design_variable_specification_uri, metadata_graph))
        
        # Process all metadata fields using unified mapping
        g = process_all_metadata_fields(g, row, study_design_execution_uri, study_design_uri, study_uri, protocol_uri, metadata_graph)
        
    print(f"✅ Cohorts metadata graph created with {len(g)} triples")
    return g


def process_all_metadata_fields(g: Graph, row: pd.Series, study_design_execution_uri: URIRef, 
                               study_design_uri: URIRef, study_uri: URIRef, protocol_uri: URIRef, 
                               metadata_graph: URIRef) -> Graph:
    """Process ALL metadata fields using unified mapping to eliminate code repetition"""
    
    # Simplified field mapping - only essential info, rest is derived
    # Using OntologyNamespaces enum consistently with study_kg.py
    field_config = {
        # Direct properties (no entity creation)
        "language": "direct_property",
        
        # Entity fields - minimal config, smart defaults
        "study type": {"ns": "SIO", "type": "descriptor", "target": "study_design", "rel": "is_about", "rel_ns": "IAO", "use_rdfs_label": True},
        "study objective": {"ns": "OBI", "type": "objective_specification", "target": "protocol"},
        "number of participants": {"ns": "CMEO", "type": "number_of_participants", "target": "protocol", "label": True},
        "start date": {"ns": "CMEO", "type": "start_time", "target": "execution", "rel": "has_time_stamp", "rel_ns": "IAO"},
        "end date": {"ns": "CMEO", "type": "end_time", "target": "execution", "rel": "has_time_stamp", "rel_ns": "IAO"},
        "ongoing": {"ns": "SIO", "type": "ongoing", "target": "execution", "rel": "is_about", "rel_ns": "IAO", "datatype": "boolean"},
        # NOTE: age distribution, population location, and morbidity handled in handle_special_fields (need output_population structure)
        # "age distribution": {"ns": "OBI", "type": "age_distribution", "target": "execution", "label": True},
        # "population location": {"ns": "BFO", "type": "site", "target": "execution"},
        "institute": {"ns": "OBI", "type": "organization", "target": "execution", "rel": "has_participant"},
        # Note: primary/secondary outcome specifications, morbidity, age distribution, population location handled in handle_special_fields (need nested structure)
        # "morbidity": {"ns": "OBI", "type": "morbidity", "target": "protocol"},
        "administrator": {"ns": "NCBI", "type": "homo_sapiens", "target": "execution", "rel": "has_participant"},
        "study contact person": {"ns": "NCBI", "type": "homo_sapiens", "target": "execution", "rel": "has_participant"},
        # Additional fields to match CohortVarLinker
        "frequency of data collection": {"ns": "CMEO", "type": "timeline_specification", "target": "protocol"},
        "interventions": {"ns": "CMEO", "type": "intervention_specification", "target": "protocol", "label": True},
    }
    
    # Smart defaults and derivation logic
    def get_namespace(ns_key):
        ns_map = {"DC": DC, "SIO": OntologyNamespaces.SIO.value, "OBI": OntologyNamespaces.OBI.value, 
                  "CMEO": OntologyNamespaces.CMEO.value, "BFO": OntologyNamespaces.BFO.value, 
                  "IAO": OntologyNamespaces.IAO.value, "NCBI": OntologyNamespaces.NCBI.value}
        return ns_map[ns_key]
    
    def get_subject_uri(target):
        targets = {"execution": study_design_execution_uri, "protocol": protocol_uri, "study_design": study_design_uri}
        return targets[target]
    
    def get_uri_suffix(field_name):
        # Special case mappings to match CohortVarLinker URIs
        uri_mappings = {
            "study type": "/descriptor",  # Must match CohortVarLinker
            "frequency of data collection": "/timeline_specification",  # Must match CohortVarLinker
            "interventions": "/intervention",  # Must match CohortVarLinker (singular!)
        }
        if field_name in uri_mappings:
            return uri_mappings[field_name]
        return "/" + field_name.replace(" ", "_")
    
    def get_default_relationship(rdf_type):
        return "has_part"  # Most common default
    
    # Process all fields using simplified mapping with smart defaults
    for excel_field, config in field_config.items():
        if pd.notna(row.get(excel_field, "")):
            
            # Handle direct properties
            if config == "direct_property":
                if excel_field == "language":
                    value = str(row[excel_field]).lower().strip()
                    g.add((study_design_execution_uri, DC.language, Literal(value, datatype=XSD.string), metadata_graph))
                continue
            
            # Process value with smart defaults
            if config.get("datatype") == "boolean":
                value = True if str(row[excel_field]).lower().strip() == "yes" else False
                datatype = XSD.boolean
            else:
                value = str(row[excel_field]).strip()
                datatype = XSD.string
            
            # Create entity with derived properties
            namespace = get_namespace(config["ns"])
            rdf_type = config["type"]
            subject_uri = get_subject_uri(config["target"])
            uri_suffix = get_uri_suffix(excel_field)
            relationship = config.get("rel", get_default_relationship(rdf_type))
            rel_namespace = get_namespace(config.get("rel_ns", "RO")) if config.get("rel_ns") else OntologyNamespaces.RO.value
            
            # Add RDF triples
            entity_uri = URIRef(study_uri + uri_suffix)
            rdf_type_uri = URIRef(namespace + rdf_type)
            g.add((entity_uri, RDF.type, rdf_type_uri, metadata_graph))
            
            # Add relationship
            relationship_uri = URIRef(rel_namespace + relationship)
            g.add((subject_uri, relationship_uri, entity_uri, metadata_graph))
            
            # Add label if specified
            if config.get("label"):
                g.add((entity_uri, RDFS.label, Literal(excel_field, datatype=XSD.string), metadata_graph))
            
            # Add rdfs:label with the value itself (for query compatibility)
            if config.get("use_rdfs_label"):
                g.add((entity_uri, RDFS.label, Literal(value, datatype=XSD.string), metadata_graph))
            
            # Add value
            g.add((entity_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=datatype), metadata_graph))
    
    # Handle special cases that need custom logic
    g = handle_special_fields(g, row, study_design_execution_uri, study_uri, protocol_uri, metadata_graph)
    return g


def handle_special_fields(g: Graph, row: pd.Series, study_design_execution_uri: URIRef,
                         study_uri: URIRef, protocol_uri: URIRef, metadata_graph: URIRef) -> Graph:
    """Handle only the truly special cases that can't be mapped generically"""
    
    # Create eligibility criterion container (matches study_kg.py structure)
    eligibility_criterion_uri = URIRef(study_uri + "/eligibility_criterion")
    g.add((eligibility_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.eligibility_criterion, metadata_graph))
    g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, eligibility_criterion_uri, metadata_graph))
    
    # Create enrollment and population entities (matches study_kg.py and query expectations)
    human_subject_enrollment = URIRef(study_uri + "/human_subject_enrollment")
    g.add((human_subject_enrollment, RDF.type, OntologyNamespaces.CMEO.value.human_subject_enrollment, metadata_graph))
    g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.is_concretized_by, human_subject_enrollment, metadata_graph))
    
    input_population_uri = URIRef(study_uri + "/input_population")
    g.add((input_population_uri, RDF.type, OntologyNamespaces.OBI.value.population, metadata_graph))
    g.add((human_subject_enrollment, OntologyNamespaces.RO.value.has_input, input_population_uri, metadata_graph))
    
    output_population_uri = URIRef(study_uri + "/output_population")
    g.add((output_population_uri, RDF.type, OntologyNamespaces.OBI.value.population, metadata_graph))
    g.add((human_subject_enrollment, OntologyNamespaces.RO.value.has_output, output_population_uri, metadata_graph))
    
    # Handle Mixed Sex field with male/female percentage parsing and as characteristic of output population
    if pd.notna(row.get("mixed sex", "")):
        mixed_sex_value = str(row["mixed sex"])
        male_percentage = None
        female_percentage = None
        
        # Add mixed sex quality of output population (matches study_kg.py)
        mixed_sex_quality = URIRef(study_uri + "/mixed_sex")
        g.add((mixed_sex_quality, RDF.type, OntologyNamespaces.OBI.value.mixed_sex, metadata_graph))
        g.add((output_population_uri, OntologyNamespaces.RO.value.has_characteristic, mixed_sex_quality, metadata_graph))
        g.add((mixed_sex_quality, RDFS.label, Literal("mixed sex", datatype=XSD.string), metadata_graph))
        g.add((mixed_sex_quality, OntologyNamespaces.CMEO.value.has_value, Literal(mixed_sex_value, datatype=XSD.string), metadata_graph))
        
        # Parse percentages from mixed sex field
        parts = mixed_sex_value.split(";") if ";" in mixed_sex_value else mixed_sex_value.split("and") if "and" in mixed_sex_value else [mixed_sex_value]
        
        for part in parts:
            part = part.strip().lower().replace(",", ".")
            digits_only = ''.join(c for c in part if c.isdigit() or c == '.')
            if digits_only:
                try:
                    percentage = float(digits_only)
                    if "male" in part and "female" not in part:
                        male_percentage = percentage
                    elif "female" in part:
                        female_percentage = percentage
                except ValueError:
                    continue
        
        # Add parsed percentages to graph (for frontend display)
        if male_percentage is not None:
            male_percentage_uri = URIRef(study_uri + "/male_percentage")
            g.add((male_percentage_uri, RDF.type, OntologyNamespaces.CMEO.value.male_percentage, metadata_graph))
            g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.has_part, male_percentage_uri, metadata_graph))
            g.add((male_percentage_uri, OntologyNamespaces.CMEO.value.has_value, Literal(male_percentage, datatype=XSD.float), metadata_graph))
            
        if female_percentage is not None:
            female_percentage_uri = URIRef(study_uri + "/female_percentage")
            g.add((female_percentage_uri, RDF.type, OntologyNamespaces.CMEO.value.female_percentage, metadata_graph))
            g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.has_part, female_percentage_uri, metadata_graph))
            g.add((female_percentage_uri, OntologyNamespaces.CMEO.value.has_value, Literal(female_percentage, datatype=XSD.float), metadata_graph))
        
        print(f"Mixed Sex parsing for {row.get('study name', 'unknown')}: '{mixed_sex_value}' → Male: {male_percentage}, Female: {female_percentage}")
    
    # Handle morbidity as characteristic of output population (matches query expectations)
    if pd.notna(row.get("morbidity", "")):
        morbidities = str(row["morbidity"]).lower().split(";") if ";" in str(row["morbidity"]) else [str(row["morbidity"]).lower()]
        for morbidity in morbidities:
            morbidity = morbidity.strip()
            if morbidity:
                dynamic_morbidity_uri = URIRef(OntologyNamespaces.OBI.value + normalize_text(morbidity))
                g.add((output_population_uri, OntologyNamespaces.RO.value.has_characteristic, dynamic_morbidity_uri, metadata_graph))
                g.add((dynamic_morbidity_uri, OntologyNamespaces.RO.value.is_characteristic_of, output_population_uri, metadata_graph))
                g.add((dynamic_morbidity_uri, RDF.type, OntologyNamespaces.OBI.value.morbidity, metadata_graph))
                g.add((dynamic_morbidity_uri, RDFS.label, Literal(morbidity, datatype=XSD.string), metadata_graph))
                g.add((dynamic_morbidity_uri, OntologyNamespaces.CMEO.value.has_value, Literal(morbidity, datatype=XSD.string), metadata_graph))
    
    # Handle age distribution as characteristic of output population (matches query expectations)
    if pd.notna(row.get("age distribution", "")):
        age_quality = URIRef(study_uri + "/age_distribution")
        g.add((age_quality, RDF.type, OntologyNamespaces.OBI.value.age_distribution, metadata_graph))
        g.add((output_population_uri, OntologyNamespaces.RO.value.has_characteristic, age_quality, metadata_graph))
        g.add((age_quality, RDFS.label, Literal("age distribution", datatype=XSD.string), metadata_graph))
        g.add((age_quality, OntologyNamespaces.CMEO.value.has_value, Literal(str(row["age distribution"]), datatype=XSD.string), metadata_graph))
    
    # Handle population location with site pointing to output population (matches query expectations)
    if pd.notna(row.get("population location", "")):
        site_uri = URIRef(output_population_uri + "/site")
        g.add((site_uri, RDF.type, OntologyNamespaces.BFO.value.site, metadata_graph))
        g.add((site_uri, OntologyNamespaces.IAO.value.is_about, output_population_uri, metadata_graph))
        g.add((site_uri, OntologyNamespaces.CMEO.value.has_value, Literal(str(row["population location"]), datatype=XSD.string), metadata_graph))
    
    # Handle outcome specifications with nested structure required by SPARQL query
    # Query expects: protocol -> outcome_specification -> primary/secondary_outcome_specification
    # URIs must match CohortVarLinker: /primary_outcome_specification (not /primary_outcome_spec)
    has_primary = pd.notna(row.get("primary outcome specification", ""))
    has_secondary = pd.notna(row.get("secondary outcome specification", ""))
    
    if has_primary or has_secondary:
        # Create the parent outcome_specification entity
        outcome_spec_uri = URIRef(study_uri + "/outcome_specification")
        g.add((outcome_spec_uri, RDF.type, OntologyNamespaces.CMEO.value.outcome_specification, metadata_graph))
        g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, outcome_spec_uri, metadata_graph))
        g.add((outcome_spec_uri, RDFS.label, Literal("outcome specification", datatype=XSD.string), metadata_graph))
        
        # Add primary outcome specification - split by semicolon like CohortVarLinker
        if has_primary:
            primary_values = str(row["primary outcome specification"]).lower().split(';') if pd.notna(row["primary outcome specification"]) else []
            for primary_value in primary_values:
                primary_value = primary_value.strip()
                if primary_value:  # Skip empty values
                    # URI matches CohortVarLinker: /primary_outcome_specification
                    primary_uri = URIRef(study_uri + "/primary_outcome_specification")
                    g.add((primary_uri, RDF.type, OntologyNamespaces.CMEO.value.primary_outcome_specification, metadata_graph))
                    g.add((primary_uri, RDFS.label, Literal("primary outcome specification", datatype=XSD.string), metadata_graph))
                    g.add((outcome_spec_uri, OntologyNamespaces.RO.value.has_part, primary_uri, metadata_graph))
                    g.add((primary_uri, OntologyNamespaces.CMEO.value.has_value, Literal(primary_value, datatype=XSD.string), metadata_graph))
        
        # Add secondary outcome specification - split by semicolon like CohortVarLinker
        if has_secondary:
            secondary_values = str(row["secondary outcome specification"]).lower().split(';') if pd.notna(row["secondary outcome specification"]) else []
            for secondary_value in secondary_values:
                secondary_value = secondary_value.strip()
                if secondary_value:  # Skip empty values
                    # URI matches CohortVarLinker: /secondary_outcome_specification
                    secondary_uri = URIRef(study_uri + "/secondary_outcome_specification")
                    g.add((secondary_uri, RDF.type, OntologyNamespaces.CMEO.value.secondary_outcome_specification, metadata_graph))
                    g.add((secondary_uri, RDFS.label, Literal("secondary outcome specification", datatype=XSD.string), metadata_graph))
                    g.add((outcome_spec_uri, OntologyNamespaces.RO.value.has_part, secondary_uri, metadata_graph))
                    g.add((secondary_uri, OntologyNamespaces.CMEO.value.has_value, Literal(secondary_value, datatype=XSD.string), metadata_graph))
    
    # Handle inclusion criteria - matches CohortVarLinker implementation
    inclusion_criteria_columns = [col for col in row.index if "inclusion criterion" in col.lower()]
    if inclusion_criteria_columns:
        # Create inclusion criterion container
        inclusion_criterion_uri = URIRef(study_uri + "/inclusion_criterion")
        g.add((inclusion_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.inclusion_criterion, metadata_graph))
        g.add((inclusion_criterion_uri, OntologyNamespaces.RO.value.part_of, eligibility_criterion_uri, metadata_graph))
        g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.has_part, inclusion_criterion_uri, metadata_graph))
        
        for col in inclusion_criteria_columns:
            inclusion_criterion_name = normalize_text(col)
            row_value = str(row[col]).lower().strip() if pd.notna(row[col]) else ""
            if row_value == "not applicable" or row_value == "" or row_value is None:
                continue
            
            # Special handling for age-related inclusion criteria (matches study_kg.py)
            if "age" in inclusion_criterion_name:
                g = add_age_group_inclusion_criterion(g, study_uri, inclusion_criterion_uri, metadata_graph, row[col])
            else:
                # Create dynamic type for this specific inclusion criterion
                dynamic_inclusion_criterion_type = URIRef(OntologyNamespaces.CMEO.value + inclusion_criterion_name)
                
                # Split by semicolon for multiple values
                inc_all_values = row_value.split(";") if row_value else []
                for inclusion_criteria_value in inc_all_values:
                    inclusion_criteria_value = inclusion_criteria_value.strip()
                    if inclusion_criteria_value:
                        col_inclusion_criteria_uri = URIRef(study_uri + "/" + inclusion_criterion_name)
                        g.add((col_inclusion_criteria_uri, RDF.type, dynamic_inclusion_criterion_type, metadata_graph))
                        g.add((col_inclusion_criteria_uri, OntologyNamespaces.RO.value.part_of, inclusion_criterion_uri, metadata_graph))
                        g.add((inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, col_inclusion_criteria_uri, metadata_graph))
                        g.add((col_inclusion_criteria_uri, RDFS.label, Literal(col, datatype=XSD.string), metadata_graph))
                        g.add((col_inclusion_criteria_uri, OntologyNamespaces.CMEO.value.has_value, Literal(inclusion_criteria_value, datatype=XSD.string), metadata_graph))
    
    # Handle exclusion criteria - matches CohortVarLinker implementation
    exclusion_criteria_columns = [col for col in row.index if "exclusion criterion" in col.lower()]
    if exclusion_criteria_columns:
        # Create exclusion criterion container
        exclusion_criterion_uri = URIRef(study_uri + "/exclusion_criterion")
        g.add((exclusion_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.exclusion_criterion, metadata_graph))
        g.add((exclusion_criterion_uri, OntologyNamespaces.RO.value.part_of, eligibility_criterion_uri, metadata_graph))
        g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.has_part, exclusion_criterion_uri, metadata_graph))
        
        for col in exclusion_criteria_columns:
            exclusion_criterion_name = normalize_text(col)
            row_value = str(row[col]).lower().strip() if pd.notna(row[col]) else ""
            if row_value == "not applicable" or row_value == "" or row_value is None:
                continue
            
            # Create dynamic type for this specific exclusion criterion
            dynamic_exclusion_criterion_type = URIRef(OntologyNamespaces.CMEO.value + exclusion_criterion_name)
            
            # Split by semicolon for multiple values
            exc_all_values = row_value.split(";") if row_value else []
            for exclusion_criteria_value in exc_all_values:
                exclusion_criteria_value = exclusion_criteria_value.strip()
                if exclusion_criteria_value:
                    col_exclusion_criteria_uri = URIRef(study_uri + "/" + exclusion_criterion_name)
                    g.add((col_exclusion_criteria_uri, RDF.type, dynamic_exclusion_criterion_type, metadata_graph))
                    g.add((col_exclusion_criteria_uri, OntologyNamespaces.RO.value.part_of, exclusion_criterion_uri, metadata_graph))
                    g.add((exclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, col_exclusion_criteria_uri, metadata_graph))
                    g.add((col_exclusion_criteria_uri, RDFS.label, Literal(col, datatype=XSD.string), metadata_graph))
                    g.add((col_exclusion_criteria_uri, OntologyNamespaces.CMEO.value.has_value, Literal(exclusion_criteria_value, datatype=XSD.string), metadata_graph))
    
    return g


def add_age_group_inclusion_criterion(g: Graph, study_uri: URIRef, inclusion_criterion_uri: URIRef, 
                                      metadata_graph: URIRef, inclusion_criteria_value: str) -> Graph:
    """
    Add age group inclusion criterion with min/max age value specifications.
    Matches study_kg.py implementation.
    """
    if pd.isna(inclusion_criteria_value):
        return g
    
    age_group_inclusion_criterion_uri = URIRef(study_uri + "/age_group_inclusion_criterion")
    g.add((age_group_inclusion_criterion_uri, RDF.type, OntologyNamespaces.CMEO.value.age_group_inclusion_criterion, metadata_graph))
    g.add((inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, age_group_inclusion_criterion_uri, metadata_graph))
    g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.RO.value.part_of, inclusion_criterion_uri, metadata_graph))
    g.add((age_group_inclusion_criterion_uri, RDFS.label, Literal("age group inclusion criterion", datatype=XSD.string), metadata_graph))
    g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.CMEO.value.has_value, Literal(inclusion_criteria_value, datatype=XSD.string), metadata_graph))
    
    # Extract age range using the utility function
    agic_value_ranges = extract_age_range(inclusion_criteria_value)
    if agic_value_ranges:
        min_age, max_age = agic_value_ranges
        
        if min_age is not None:
            min_age = float(min_age)
            min_age_value_specification = URIRef(age_group_inclusion_criterion_uri + "/minimum_age_value_specification")
            g.add((min_age_value_specification, RDF.type, OntologyNamespaces.OBI.value.minimum_age_value_specification, metadata_graph))
            g.add((min_age_value_specification, OntologyNamespaces.CMEO.value.has_value, Literal(min_age, datatype=XSD.float), metadata_graph))
            g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, min_age_value_specification, metadata_graph))
        
        if max_age is not None:
            max_age = float(max_age)
            max_age_value_specification = URIRef(age_group_inclusion_criterion_uri + "/maximum_age_value_specification")
            g.add((max_age_value_specification, RDF.type, OntologyNamespaces.OBI.value.maximum_age_value_specification, metadata_graph))
            g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, max_age_value_specification, metadata_graph))
            g.add((max_age_value_specification, OntologyNamespaces.CMEO.value.has_value, Literal(max_age, datatype=XSD.float), metadata_graph))
    
    return g


def init_triplestore():
    """Initialize triplestore with the OMOP CDM ontology and the iCARE4CVD cohorts metadata."""
    import fcntl
    import time
    
    # Use file-based locking to ensure only one worker initializes the triplestore
    # Ensure data folder exists first
    os.makedirs(settings.data_folder, exist_ok=True)
    lock_file_path = os.path.join(settings.data_folder, "triplestore_init.lock")
    
    try:
        with open(lock_file_path, "w") as lock_file:
            # Try to acquire exclusive lock (non-blocking)
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"🔒 Worker {os.getpid()} acquired initialization lock")
            except BlockingIOError:
                print(f"⏳ Worker {os.getpid()} waiting for initialization to complete by another worker...")
                # Another worker is initializing, wait for it to complete
                time.sleep(5)
                # Initialize cache from triplestore after other worker completes
                from src.cohort_cache import initialize_cache_from_triplestore
                admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
                initialize_cache_from_triplestore(admin_email)
                print(f"✅ Worker {os.getpid()} cache initialized from triplestore")
                return
            
            # If we reach here, we have the lock and should proceed with initialization
            print("Clearing cohort cache before initialization...")
            clear_cache()
            
            # Continue with the rest of the initialization logic...
            _perform_triplestore_initialization()
            
    except Exception as e:
        print(f"❌ Error during triplestore initialization: {e}")
        raise
    finally:
        # Clean up lock file
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)


def _perform_triplestore_initialization():
    """Perform the actual triplestore initialization logic."""
    import time
    
    # Check multiple times if triples exist to ensure we don't have race conditions
    triplestore_initialized = False
    for _ in range(3):
        if run_query("ASK WHERE { GRAPH ?g {?s ?p ?o .} }")["boolean"]:
            print("⏩ Triplestore already contains data. Skipping triplestore initialization.")
            triplestore_initialized = True
            break
        time.sleep(0.5)  # Small delay between checks
    
    # Generate metadata issues report (runs on every startup)
    print("Generating metadata issues report...")
    try:
        import httpx
        import asyncio
        
        # Call the endpoint to generate the report
        async def call_report_endpoint():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:3000/upload/generate-metadata-issues-report",
                    timeout=60.0
                )
                return response
        
        response = asyncio.run(call_report_endpoint())
        
        # Extract summary info from response headers
        filename = response.headers.get('content-disposition', '').split('filename=')[-1].strip('"')
        total_cohorts = response.headers.get('X-Total-Cohorts', 'N/A')
        cohorts_with_errors = response.headers.get('X-Cohorts-With-Errors', 'N/A')
        cohorts_without_dict = response.headers.get('X-Cohorts-Without-Dict', 'N/A')
        
        print(f"✅ Metadata issues report generated: {filename}")
        print(f"   Total cohorts: {total_cohorts}")
        print(f"   Cohorts with errors: {cohorts_with_errors}")
        print(f"   Cohorts without dictionary: {cohorts_without_dict}")
    except Exception as e:
        print(f"⚠️  Failed to generate metadata issues report: {e}")
    
    # If triplestore is already initialized, initialize cache from it and return
    if triplestore_initialized:
        from src.cohort_cache import initialize_cache_from_triplestore
        print("Initializing cache from existing triplestore...")
        admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
        initialize_cache_from_triplestore(admin_email)
        print("✅ Cohort cache initialization complete.")
        return
    
    # Otherwise, continue with triplestore initialization
    # First, load cohorts metadata to establish basic cohort entities
    print("Loading cohorts metadata file to triplestore...")
    g = cohorts_metadata_file_to_graph(COHORTS_METADATA_FILEPATH)
    
    # Delete existing triples for cohorts metadata before publishing new ones
    cohorts_graph = URIRef("https://w3id.org/icare4cvd/cohorts")
    delete_existing_triples(cohorts_graph)

    if publish_graph_to_endpoint(g):
        print(f"🪪 Triplestore initialization: added {len(g)} triples for the cohorts metadata.")
    else:
        print("❌ Failed to publish cohort metadata to triplestore.")
        return
    
    # Then, load cohorts data dictionaries to add variables to the established cohorts
    print("Now loading cohort data dictionaries to add variables to the established cohorts...")
    for folder in os.listdir(os.path.join(settings.data_folder, "cohorts")):
        folder_path = os.path.join(settings.data_folder, "cohorts", folder)
        if os.path.isdir(folder_path):
            try:
                #Note (August 2025): we now find the latest version of a data dictionary instead of processing all
                #for file in glob.glob(os.path.join(folder_path, "*_datadictionary.*")):
                latest_dict_file = get_latest_datadictionary(folder_path)
                if latest_dict_file:
                    print(f"Using latest datadictionary file for {folder}: {os.path.basename(latest_dict_file)}, date: {os.path.getmtime(latest_dict_file)}")
                    g = load_cohort_dict_file(latest_dict_file, folder, source="init_triplestore")
                    
                    # Check if graph is empty (indicates processing failure)
                    if len(g) == 0:
                        print(f"❌ Failed to process dictionary for cohort {folder} - graph is empty. Check logs above for details.")
                        continue
                    
                    # Delete existing triples for this cohort before publishing new ones
                    # This ensures we don't have duplicate or conflicting triples
                    delete_existing_triples(get_cohort_uri(folder))
                    # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.trig", format="trig")
                    if publish_graph_to_endpoint(g):
                        print(f"💾 Triplestore initialization: added {len(g)} triples for cohort {folder}.")
                    else:
                        print(f"❌ Failed to publish graph to triplestore for cohort {folder}")
                else:
                    print(f"No datadictionary file found for cohort {folder}.")
            except HTTPException as http_exc:
                print(f"❌ SKIPPING cohort {folder} - HTTPException: {http_exc.detail}")
                logging.error(f"Failed to process cohort {folder} during init: {http_exc.detail}")
            except Exception as e:
                print(f"❌ SKIPPING cohort {folder} - Unexpected error: {str(e)}")
                logging.error(f"Failed to process cohort {folder} during init: {str(e)}", exc_info=True)
        else:
            print(f"No datadictionary file found for cohort {folder}.")
    
    print("✅ Triplestore initialization complete!")
    
    # Initialize the cache from the now-populated triplestore
    # This must happen AFTER the triplestore is populated, not before
    from src.cohort_cache import initialize_cache_from_triplestore
    print("Initializing cache from triplestore...")
    admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
    initialize_cache_from_triplestore(admin_email)
    print("✅ Cohort cache initialization complete.")