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
from rdflib import XSD, Dataset, Graph, Literal, URIRef
from rdflib.namespace import DC, RDF, RDFS
from SPARQLWrapper import SPARQLWrapper

from src.config import settings
from src.auth import get_current_user
from src.utils import run_query, retrieve_cohorts_metadata
from src.cohort_cache import add_cohort_to_cache, clear_cache, create_cohort_from_dict_file, create_cohort_from_metadata_graph
from src.decentriq import create_provision_dcr, metadatadict_cols_schema1
from src.mapping_generation.retriever import map_csv_to_standard_codes
from src.utils import ICARE, curie_converter, init_graph, prefix_map, retrieve_cohorts_metadata, run_query

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


def get_cohort_mapping_uri(cohort_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}/mappings"]


def get_var_uri(cohort_id: str | URIRef, var_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}/{var_id.replace(' ', '_')}"]


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

cols_normalized = {"VARIABLE NAME": "VARIABLENAME", 
                   "VARIABLE LABEL": "VARIABLELABEL",
                   "VAR TYPE": "VARTYPE",
                   "CATEGORICALVALUECONCEPTCODE": "CATEGORICAL VALUE CONCEPT CODE"}

ACCEPTED_DATATYPES = ["STR", "FLOAT", "INT", "DATETIME"]

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
    
    try:
        df = pd.read_csv(dict_path, na_values=[""], keep_default_na=False)
        df = df.dropna(how="all") # Drop rows where all cells are NA
        df = df.fillna("") # Fill remaining NA with empty string
        
        # Capture data metrics for logging
        total_rows = len(df)
        total_variables = len(df['VARIABLENAME'].unique()) if 'VARIABLENAME' in df.columns else 0
        
        # Normalize column names (uppercase, specific substitutions)
        #df.columns = [cols_normalized.get(c.upper().strip(), c.upper().strip().replace("VALUES", "VALUE")) for c in df.columns]
        df.columns = [cols_normalized.get(c.upper().strip(), c.upper().strip()) for c in df.columns]
        # print(f"POST NORMALIZATION -- COHORT {cohort_id} -- Columns: {df.columns}")
        # --- Structural Validation: Check for required columns ---
        # Define columns absolutely essential for the row-processing logic to run without KeyErrors
        #critical_column_names_for_processing = [c.name.upper().strip() for c in metadatadict_cols_schema1 if c.name.upper().strip() != "VISITS"]
        critical_column_names_for_processing = [c.name.upper().strip() for c in metadatadict_cols_schema1]
        missing_columns = []
        for required_col_name in critical_column_names_for_processing:
            if required_col_name not in df.columns:
                missing_columns.append(required_col_name)
        
        # If critical columns are missing, further processing is unreliable or will cause crashes.
        # Report all errors found so far (which will include all missing column messages) and stop.
        if len(missing_columns) > 0:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
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
        duplicate_variables = df[df.duplicated(subset=["VARIABLENAME"], keep=False)]
        if not duplicate_variables.empty:
            errors.append(f"Duplicate VARIABLENAME found: {', '.join(duplicate_variables['VARIABLENAME'].unique())}")

        for i, row in df.iterrows():
            var_name_for_error = row.get("VARIABLENAME", f"UNKNOWN_VAR_ROW_{i+2}")

            # Check if required values are present in rows (for critical columns)
            req_fields = ["VARIABLENAME", "VARIABLELABEL", "VARTYPE", "DOMAIN"]
            for rf in req_fields:
                if not row[rf].strip():
                    errors.append(f"Row {i+2} (Variable: '{var_name_for_error}') is missing value for the required field: '{rf}'.")
            
            if row["VARTYPE"] not in ACCEPTED_DATATYPES:
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}') has an invalid data type: '{row['VARTYPE']}'. Accepted types: {', '.join(ACCEPTED_DATATYPES)}."
                )

            acc_domains = ["condition_occurrence", "visit_occurrence", "procedure_occurrence", "measurement", "drug_exposure", "device_exposure", "person", "observation", "observation_period", "death", "specimen", "condition_era"]
            if row['DOMAIN'].strip().lower() not in acc_domains:
                errors.append(
                    f'Row {i+2} (Variable: "{var_name_for_error}") has an invalid domain: "{row["DOMAIN"]}". Accepted domains: {", ".join(acc_domains)}.'
                )

            # Handle "codes" columns validation (from 'categories' column created by parse_categorical_string)
            # Ensure 'categories' column exists and is a list before checking its length or content
            current_categories = row.get("categories")
            if isinstance(current_categories, list):
                #if len(current_categories) == 1:
                #    errors.append(
                #        f"Row {i+2} (Variable: '{var_name_for_error}') has only one category defined: '{current_categories[0]['value']}'. Categorical variables should have at least two distinct categories or be left blank if not applicable."
                #    )
                
                # Category Concept Code Validation (if 'Categorical Value Concept Code' column exists)
                # This column is not in COLUMNS_LIST, so it's optional.
                if "CATEGORICAL VALUE CONCEPT CODE" in df.columns: # Check against normalized column name
                    categories_codes_str = str(row.get("CATEGORICAL VALUE CONCEPT CODE", "")).strip()
                    if categories_codes_str: # Only process if there's content
                        categories_codes = categories_codes_str.split("|")
                        if len(categories_codes) != len(current_categories) and current_categories: # check if categories were successfully parsed
                             errors.append(
                                 f"Row {i+2} (Variable: '{var_name_for_error}'): The number of category concept codes ({len(categories_codes)}) does not match the number of parsed categories ({len(current_categories)})."
                             )
                        else: 
                            for idx, category_data in enumerate(current_categories):
                                if idx < len(categories_codes):
                                    code_to_check = categories_codes[idx].strip()
                                    if code_to_check and code_to_check.lower() != "na":
                                        try:
                                            # Another temp fix just for TIM-HF!!
                                            #if code_to_check.find(":") == -1:
                                            #    code_to_check = code_to_check.replace("OMOP", "OMOP:")
                                            expanded_uri = curie_converter.expand(code_to_check)
                                            if not expanded_uri:
                                                errors.append(
                                                    f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_data['value']}'): The category concept code '{code_to_check}' is not valid or its prefix is not recognized. Valid prefixes: {', '.join([record['prefix'] + ':' for record in prefix_map if record.get('prefix')])}."
                                                )
                                        except Exception as curie_exc:
                                            errors.append(
                                                f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_data['value']}'): Error expanding CURIE '{code_to_check}': {curie_exc}."
                                            )
            elif row.get("CATEGORICAL") and not isinstance(current_categories, list): # If original CATEGORICAL had content but parsing failed (already logged by parse_categorical_string's own exception if it was fatal)
                 # This case might be covered if parse_categorical_string added its own error to the 'errors' list already
                 errors.append(
                     f"Row {i+2} (Variable: '{var_name_for_error}') has an invalid category: '{row['CATEGORICAL']}'."
                 )


        # --- Final Error Check & Graph Generation ---
        if len(errors) > 0 and source == "upload_dict":
            raise HTTPException(
                status_code=422,
                detail="\n\n".join(errors),
            )

        # If no errors, proceed to create the graph (RDF triples)
        cohort_uri = get_cohort_uri(cohort_id)
        g = init_graph()
        g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_uri))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_uri))

        i = 0
        for i, row in df.iterrows():
            variable_uri = get_var_uri(cohort_id, row["VARIABLENAME"])
            g.add((cohort_uri, ICARE.hasVariable, variable_uri, cohort_uri))
            g.add((variable_uri, RDF.type, ICARE.Variable, cohort_uri))
            g.add((variable_uri, DC.identifier, Literal(row["VARIABLENAME"]), cohort_uri))
            g.add((variable_uri, RDFS.label, Literal(row["VARIABLELABEL"]), cohort_uri))
            g.add((variable_uri, ICARE["index"], Literal(i, datatype=XSD.integer), cohort_uri))

            # Get categories code if provided (re-fetch for graph generation phase)
            categories_codes = []
            if "CATEGORICAL VALUE CONCEPT CODE" in df.columns and str(row.get("CATEGORICAL VALUE CONCEPT CODE","")).strip():
                 categories_codes = str(row["CATEGORICAL VALUE CONCEPT CODE"]).split("|")

            for column_name_from_df, col_value in row.items():
                #print("in load_cohort_dict_file -- column_name_from_df value:", column_name_from_df)
                # Use the already normalized column_name_from_df
                if column_name_from_df not in ["categories"] and col_value: # Exclude our temporary 'categories' column
                    property_uri = ICARE[to_camelcase(column_name_from_df)] # to_camelcase expects original-like names
                    if (
                        isinstance(col_value, str)
                        and (col_value.startswith("http://") or col_value.startswith("https://"))
                        and " " not in col_value
                    ):
                        g.add((variable_uri, property_uri, URIRef(col_value), cohort_uri))
                    else:
                        g.add((variable_uri, property_uri, Literal(col_value), cohort_uri))
                
                if column_name_from_df == "categories" and isinstance(col_value, list): # 'categories' is our parsed list
                    for index, category in enumerate(col_value):
                        cat_uri = get_category_uri(variable_uri, index) # Use index for unique category URI part
                        g.add((variable_uri, ICARE.categories, cat_uri, cohort_uri))
                        g.add((cat_uri, RDF.type, ICARE.VariableCategory, cohort_uri))
                        g.add((cat_uri, RDF.value, Literal(category["value"]), cohort_uri))
                        g.add((cat_uri, RDFS.label, Literal(category["label"]), cohort_uri))
                        
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
                                        #print(f"Adding category code {cat_code_uri} for category {category['value']} in cohort {cohort_id}, line {i}, cat_uri: {cat_uri}, conceptId: {ICARE.conceptId}")
                                        g.add((cat_uri, ICARE.conceptId, URIRef(cat_code_uri), cohort_uri))
                                except Exception as curie_exc:
                                    errors.append(
                                        f"Row {i+2} (Variable: '{var_name_for_error}', Category: '{category_data['value']}'): Error expanding CURIE '{code_to_check}': {curie_exc}."
                                    )

        print(f"Finished processing cohort dictionary: {cohort_id}")
        

    except HTTPException as http_exc: # Re-raise specific HTTPExceptions (ours or from parse_categorical_string)
        # Log the collected errors that led to this for server-side records
        logging.warning(f"Validation errors for cohort {cohort_id}:\n{http_exc.detail}")
        if source == "upload_dict":
            raise http_exc 
    except pd.errors.EmptyDataError:
        logging.warning(f"Uploaded CSV for cohort {cohort_id} is empty or unreadable.")
        if source == "upload_dict":
            raise HTTPException(status_code=422, detail="The uploaded CSV file is empty or could not be read.")
    except Exception as e:
        logging.error(f"Unexpected error during dictionary processing for {cohort_id}: {str(e)}", exc_info=True)
        errors.append(f"An unexpected error occurred during file processing: {str(e)}")
        if source == "upload_dict":
            raise HTTPException(
                status_code=500, # Use 500 for truly unexpected server-side issues
                detail="\n\n".join(errors),
            )
    
    # Calculate final metrics
    processing_time = (datetime.now() - start_time).total_seconds()
    success = len(errors) == 0
    graph_triples_count = len(g) if 'g' in locals() else 0
    
    # Log to existing text file (maintain backward compatibility)
    if len(errors) > 0:
        errors_file = os.path.join(settings.data_folder, f"metadata_files_issues.txt")
        with open(errors_file, "a") as f:
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(f"Errors for cohort {cohort_id}:\n")
            f.write("\n".join(errors))
            f.write("\n\n\n")
    
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
    if source == "upload_dict":
        from src.cohort_cache import create_cohort_from_dict_file
        cohort_uri = get_cohort_uri(cohort_id)
        create_cohort_from_dict_file(cohort_id, cohort_uri, g)
        logging.info(f"Added cohort {cohort_id} to cache directly from dictionary file")
    else:
        logging.info(f"Skipping cache update for cohort {cohort_id} - cache managed by {source}")

    return g





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
    delete_existing_triples(get_cohort_uri(cohort_id))
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

    # Store metadata file on disk in the cohorts folder
    metadata_path = os.path.join(cohort_info.folder_path, filename + ext)
    with open(metadata_path, "wb") as buffer:
        shutil.copyfileobj(cohort_dictionary.file, buffer)

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
            delete_existing_triples(get_cohort_uri(cohort_id))
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
    delete_existing_triples(get_cohort_uri(cohort_id))
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
    g = init_graph()
    for _i, row in df.iterrows():
        print("now processing cohorts' metadata row: ", _i, row)
        cohort_id = str(row["Study Name"]).strip()
        # print(cohort_id)
        cohort_uri = get_cohort_uri(cohort_id)
        cohorts_graph = ICARE["graph/metadata"]

        g.add((cohort_uri, RDF.type, ICARE.Cohort, cohorts_graph))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohorts_graph))
        g.add((cohort_uri, ICARE.institution, Literal(row["Institute"]), cohorts_graph))
        # Administrator information
        if is_valid_value(row.get("Administrator", "")):
            g.add((cohort_uri, ICARE.administrator, Literal(row["Administrator"]), cohorts_graph))
        if is_valid_value(row.get("Administrator Email Address", "")):
            # Store as administratorEmail for backward compatibility
            g.add((cohort_uri, ICARE.administratorEmail, Literal(row["Administrator Email Address"].lower()), cohorts_graph))
            # Also add to icare:email predicate (split by semicolon) for cohort ownership permissions
            for email in row["Administrator Email Address"].split(";"):
                g.add((cohort_uri, ICARE.email, Literal(email.strip().lower()), cohorts_graph))
        # Study contact person information
        if is_valid_value(row["Study Contact Person"]):
            g.add((cohort_uri, DC.creator, Literal(row["Study Contact Person"]), cohorts_graph))
        if is_valid_value(row["Study Contact Person Email Address"]):
            for email in row["Study Contact Person Email Address"].split(";"):
                g.add((cohort_uri, ICARE.email, Literal(email.strip().lower()), cohorts_graph))
        # References
        if is_valid_value(row.get("References", "")):
            for reference in row["References"].split(";"):
                g.add((cohort_uri, ICARE.references, Literal(reference.strip()), cohorts_graph))
                
        # Additional metadata fields
        if is_valid_value(row.get("Population Location", "")):
            g.add((cohort_uri, ICARE.populationLocation, Literal(row["Population Location"]), cohorts_graph))
        if is_valid_value(row.get("Language", "")):
            g.add((cohort_uri, ICARE.language, Literal(row["Language"]), cohorts_graph))
        if is_valid_value(row.get("Frequency of data collection", "")):
            g.add((cohort_uri, ICARE.dataCollectionFrequency, Literal(row["Frequency of data collection"]), cohorts_graph))
        if is_valid_value(row.get("Interventions", "")):
            g.add((cohort_uri, ICARE.interventions, Literal(row["Interventions"]), cohorts_graph))
        if is_valid_value(row["Study Type"]):
            # Split study types on '/' and add each as a separate triple
            study_types = [st.strip() for st in row["Study Type"].split("/")]
            for study_type in study_types:
                g.add((cohort_uri, ICARE.cohortType, Literal(study_type), cohorts_graph))
        if is_valid_value(row["Study Design"]):
            g.add((cohort_uri, ICARE.studyType, Literal(row["Study Design"]), cohorts_graph))
        #if is_valid_value(row["Study duration"]):
        #    g.add((cohort_uri, ICARE.studyDuration, Literal(row["Study duration"]), cohorts_graph))
        if is_valid_value(row["Start date"]) and is_valid_value(row["End date"]):
            g.add((cohort_uri, ICARE.studyStart, Literal(row["Start date"]), cohorts_graph))
            g.add((cohort_uri, ICARE.studyEnd, Literal(row["End date"]), cohorts_graph))
        if is_valid_value(row["Number of Participants"]):
            g.add((cohort_uri, ICARE.studyParticipants, Literal(row["Number of Participants"]), cohorts_graph))
        if is_valid_value(row["Ongoing"]):
            g.add((cohort_uri, ICARE.studyOngoing, Literal(row["Ongoing"]), cohorts_graph))
        #if is_valid_value(row["Patient population"]):
        #    g.add((cohort_uri, ICARE.studyPopulation, Literal(row["Patient population"]), cohorts_graph))
        if is_valid_value(row["Study Objective"]):
            g.add((cohort_uri, ICARE.studyObjective, Literal(row["Study Objective"]), cohorts_graph))
            
        # Handle primary outcome specification
        if "primary outcome specification" in row and is_valid_value(row["primary outcome specification"]):
            g.add((cohort_uri, ICARE.primaryOutcomeSpec, Literal(row["primary outcome specification"]), cohorts_graph))
            
        # Handle secondary outcome specification
        if "secondary outcome specification" in row and is_valid_value(row["secondary outcome specification"]):
            g.add((cohort_uri, ICARE.secondaryOutcomeSpec, Literal(row["secondary outcome specification"]), cohorts_graph))
            
        # Handle morbidity
        if "Morbidity" in row and is_valid_value(row["Morbidity"]):
            g.add((cohort_uri, ICARE.morbidity, Literal(row["Morbidity"]), cohorts_graph))
            
        # Handle inclusion criteria fields using exact field names
        # Sex inclusion
        if "Sex inclusion criterion" in row and is_valid_value(row["Sex inclusion criterion"]):
            g.add((cohort_uri, ICARE["sexInclusion"], Literal(row["Sex inclusion criterion"]), cohorts_graph))
        
        # Health status inclusion
        if "Health status inclusion criterion" in row and is_valid_value(row["Health status inclusion criterion"]):
            g.add((cohort_uri, ICARE["healthStatusInclusion"], Literal(row["Health status inclusion criterion"]), cohorts_graph))
        
        # Clinically relevant exposure inclusion
        if "clinically relevant exposure inclusion criterion" in row and is_valid_value(row["clinically relevant exposure inclusion criterion"]):
            g.add((cohort_uri, ICARE["clinicallyRelevantExposureInclusion"], Literal(row["clinically relevant exposure inclusion criterion"]), cohorts_graph))
        
        # Age group inclusion
        if "age group inclusion criterion" in row and is_valid_value(row["age group inclusion criterion"]):
            g.add((cohort_uri, ICARE["ageGroupInclusion"], Literal(row["age group inclusion criterion"]), cohorts_graph))
        
        # BMI range inclusion
        if "BMI range inclusion criterion" in row and is_valid_value(row["BMI range inclusion criterion"]):
            g.add((cohort_uri, ICARE["bmiRangeInclusion"], Literal(row["BMI range inclusion criterion"]), cohorts_graph))
        
        # Ethnicity inclusion
        if "ethnicity inclusion criterion" in row and is_valid_value(row["ethnicity inclusion criterion"]):
            g.add((cohort_uri, ICARE["ethnicityInclusion"], Literal(row["ethnicity inclusion criterion"]), cohorts_graph))
        
        # Family status inclusion
        if "family status inclusion criterion" in row and is_valid_value(row["family status inclusion criterion"]):
            g.add((cohort_uri, ICARE["familyStatusInclusion"], Literal(row["family status inclusion criterion"]), cohorts_graph))
        
        # Hospital patient inclusion
        if "hospital patient inclusion criterion" in row and is_valid_value(row["hospital patient inclusion criterion"]):
            g.add((cohort_uri, ICARE["hospitalPatientInclusion"], Literal(row["hospital patient inclusion criterion"]), cohorts_graph))
        
        # Use of medication inclusion
        if "use of medication inclusion criterion" in row and is_valid_value(row["use of medication inclusion criterion"]):
            g.add((cohort_uri, ICARE["useOfMedicationInclusion"], Literal(row["use of medication inclusion criterion"]), cohorts_graph))
        
        # Handle exclusion criteria fields using exact field names
        # Health status exclusion
        if "health status exclusion criterion" in row and is_valid_value(row["health status exclusion criterion"]):
            g.add((cohort_uri, ICARE["healthStatusExclusion"], Literal(row["health status exclusion criterion"]), cohorts_graph))
        
        # BMI range exclusion
        if "bmi range exclusion criterion" in row and is_valid_value(row["bmi range exclusion criterion"]):
            g.add((cohort_uri, ICARE["bmiRangeExclusion"], Literal(row["bmi range exclusion criterion"]), cohorts_graph))
        
        # Limited life expectancy exclusion
        if "limited life expectancy exclusion criterion" in row and is_valid_value(row["limited life expectancy exclusion criterion"]):
            g.add((cohort_uri, ICARE["limitedLifeExpectancyExclusion"], Literal(row["limited life expectancy exclusion criterion"]), cohorts_graph))
        
        # Need for surgery exclusion
        if "need for surgery exclusion criterion" in row and is_valid_value(row["need for surgery exclusion criterion"]):
            g.add((cohort_uri, ICARE["needForSurgeryExclusion"], Literal(row["need for surgery exclusion criterion"]), cohorts_graph))
        
        # Surgical procedure history exclusion
        if "surgical procedure history exclusion criterion" in row and is_valid_value(row["surgical procedure history exclusion criterion"]):
            g.add((cohort_uri, ICARE["surgicalProcedureHistoryExclusion"], Literal(row["surgical procedure history exclusion criterion"]), cohorts_graph))
        
        # Clinically relevant exposure exclusion
        if "clinically relevant exposure exclusion criterion" in row and is_valid_value(row["clinically relevant exposure exclusion criterion"]):
            g.add((cohort_uri, ICARE["clinicallyRelevantExposureExclusion"], Literal(row["clinically relevant exposure exclusion criterion"]), cohorts_graph))

        # Handle Mixed Sex field
        if "Mixed Sex" in row and is_valid_value(row["Mixed Sex"]):
            mixed_sex_value = row["Mixed Sex"]
            male_percentage = None
            female_percentage = None
            
            # Split the string by common separators
            parts = []
            if ";" in mixed_sex_value:
                parts = mixed_sex_value.split(";")
            elif "and" in mixed_sex_value:
                parts = mixed_sex_value.split("and")
            else:
                parts = [mixed_sex_value]
            
            # Process each part to find male and female percentages
            for part in parts:
                part = part.strip().lower().replace(",", ".")
                if "male" in part and "female" not in part:  # Ensure we're not catching 'female' in 'male'
                    # Extract only digits and period for the percentage
                    digits_only = ''.join(c for c in part if c.isdigit() or c == '.')
                    if digits_only:
                        try:
                            male_percentage = float(digits_only)
                            g.add((cohort_uri, ICARE.malePercentage, Literal(male_percentage), cohorts_graph))
                        except ValueError:
                            print(f"Could not convert '{digits_only}' to float for male percentage")
                
                if "female" in part:
                    # Extract only digits and period for the percentage
                    digits_only = ''.join(c for c in part if c.isdigit() or c == '.')
                    if digits_only:
                        try:
                            female_percentage = float(digits_only)
                            g.add((cohort_uri, ICARE.femalePercentage, Literal(female_percentage), cohorts_graph))
                        except ValueError:
                            print(f"Could not convert '{digits_only}' to float for female percentage")
                
            # Debug output to help diagnose parsing issues
            print(f"Mixed Sex parsing for {cohort_id}: '{mixed_sex_value}' → Male: {male_percentage}, Female: {female_percentage}")
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
            
            # Create/clear the metadata issues file at the start of initialization
            errors_file = os.path.join(settings.data_folder, "metadata_files_issues.txt")
            os.makedirs(os.path.dirname(errors_file), exist_ok=True)
            with open(errors_file, "w") as f:
                f.write(f"Metadata Issues Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
            
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
    
    # Initialize the cache from triplestore
    # This works whether the triplestore is initialized or not
    from src.cohort_cache import initialize_cache_from_triplestore
    print("Initializing cache from triplestore...")
    # Use the first admin email to ensure we get all cohorts
    admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
    initialize_cache_from_triplestore(admin_email)
    print("✅ Cohort cache initialization complete.")
    
    # If triplestore is already initialized, we're done
    if triplestore_initialized:
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
        
        # Add cohort metadata to the cache
        print("Adding cohort metadata to the cache...")
        # Extract cohort IDs and URIs from the graph
        cohort_uris = set()
        for s, p, o, _ in g.quads((None, RDF.type, ICARE.Cohort, None)):
            cohort_uris.add(s)
        
        # Create cohort objects from metadata and add them to the cache
        for cohort_uri in cohort_uris:
            # Extract cohort ID from URI
            cohort_id = None
            for _, _, o, _ in g.quads((cohort_uri, DC.identifier, None, None)):
                cohort_id = str(o)
                break
            
            if cohort_id:
                from src.cohort_cache import create_cohort_from_metadata_graph
                create_cohort_from_metadata_graph(cohort_id, cohort_uri, g)
        
        print("✅ Cohort metadata added to cache.")
    else:
        print("❌ Failed to publish cohort metadata to triplestore.")
        return
    
    # Then, load cohorts data dictionaries to add variables to the established cohorts
    print("Now loading cohort data dictionaries to add variables to the established cohorts...")
    for folder in os.listdir(os.path.join(settings.data_folder, "cohorts")):
        folder_path = os.path.join(settings.data_folder, "cohorts", folder)
        if os.path.isdir(folder_path):
            #Note (August 2025): we now find the latest version of a data dictionary instead of processing all
            #for file in glob.glob(os.path.join(folder_path, "*_datadictionary.*")):
            latest_dict_file = get_latest_datadictionary(folder_path)
            if latest_dict_file:
                print(f"Using latest datadictionary file for {folder}: {os.path.basename(latest_dict_file)}, date: {os.path.getmtime(latest_dict_file)}")
                g = load_cohort_dict_file(latest_dict_file, folder, source="init_triplestore")
                # Delete existing triples for this cohort before publishing new ones
                # This ensures we don't have duplicate or conflicting triples
                delete_existing_triples(get_cohort_uri(folder))
                # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.trig", format="trig")
                if publish_graph_to_endpoint(g):
                    print(f"💾 Triplestore initialization: added {len(g)} triples for cohort {folder}.")
                    # Note: Variables are added to cache via create_cohort_from_dict_file in init_triplestore
                    from src.cohort_cache import create_cohort_from_dict_file
                    cohort_uri = get_cohort_uri(folder)
                    create_cohort_from_dict_file(folder, cohort_uri, g)
            else:
                print(f"No datadictionary file found for cohort {folder}.")
                # Ensure cohorts without dictionaries are still properly cached
                # Check if this cohort exists in the metadata and add it to cache if missing
                cohort_uri = get_cohort_uri(folder)
                from src.cohort_cache import get_cohorts_from_cache, create_cohort_from_metadata_graph
                admin_email = settings.admins_list[0] if settings.admins_list else "admin@example.com"
                current_cache = get_cohorts_from_cache(admin_email)
                if folder not in current_cache:
                    print(f"Adding cohort {folder} to cache (metadata only, no dictionary)")
                    # Get the metadata graph to extract cohort info
                    metadata_graph = cohorts_metadata_file_to_graph(COHORTS_METADATA_FILEPATH)
                    create_cohort_from_metadata_graph(folder, cohort_uri, metadata_graph)
        else:
            print(f"No datadictionary file found for cohort {folder}.")
    
    print("✅ Triplestore initialization complete!")
    
    print("✅ Cohort cache initialization complete.")