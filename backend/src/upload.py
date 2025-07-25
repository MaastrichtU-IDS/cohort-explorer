import glob
import logging
import os
import shutil
from datetime import datetime
from re import sub
from typing import Any

import pandas as pd
import requests
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from rdflib import XSD, Dataset, Graph, Literal, URIRef
from rdflib.namespace import DC, RDF, RDFS
from SPARQLWrapper import SPARQLWrapper

from src.auth import get_current_user
from src.config import settings
from src.decentriq import create_provision_dcr
from src.mapping_generation.retriever import map_csv_to_standard_codes
from src.utils import ICARE, curie_converter, init_graph, prefix_map, retrieve_cohorts_metadata, run_query

router = APIRouter()


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
    cohort_info = retrieve_cohorts_metadata(user["email"]).get(cohort_id)
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

cols_normalized = {"VARIABLENAME": "VARIABLE NAME", 
                   "VARIABLELABEL": "VARIABLE LABEL",
                   "VARTYPE": "VAR TYPE"
                   }

COLUMNS_LIST = [
    "VARIABLE NAME",
    "VARIABLE LABEL",
    "VAR TYPE",
    "UNITS",
    "CATEGORICAL",
    "COUNT",
#    "NA",
#    "MIN",
#    "MAX",
#    "Definition",
#    "Formula",
    "DOMAIN",
#    "Visits",
]

ACCEPTED_DATATYPES = ["STR", "FLOAT", "INT", "DATETIME"]

def to_camelcase(s: str) -> str:
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])


def load_cohort_dict_file(dict_path: str, cohort_id: str) -> Dataset:
    """Parse the cohort dictionary uploaded as excel or CSV spreadsheet, and load it to the triplestore"""
    print(f"NOW PROCESSING DICTIONARY FILE FOR COHORT: {cohort_id} \nFile path: {dict_path}")
    if not dict_path.endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail="Only CSV files are supported. Please convert your file to CSV and try again.",
        )
    errors: list[str] = []
    warnings: list[str] = []
    
    try:
        df = pd.read_csv(dict_path, na_values=[""], keep_default_na=False)
        df = df.dropna(how="all") # Drop rows where all cells are NA
        df = df.fillna("") # Fill remaining NA with empty string
        
        # Normalize column names (uppercase, specific substitutions)
        df.columns = [cols_normalized.get(c.upper().strip(), c.upper().strip()) for c in df.columns]

        # --- Structural Validation: Check for required columns ---
        missing_critical_columns = False
        # Define columns absolutely essential for the row-processing logic to run without KeyErrors
        critical_column_names_for_processing = {"VARIABLE NAME", "VARIABLE LABEL", "VAR TYPE", "CATEGORICAL"} 
        
        for required_col_name in COLUMNS_LIST:
            if required_col_name not in df.columns:
                errors.append(f"Missing required column: '{required_col_name}'")
                if required_col_name in critical_column_names_for_processing:
                    missing_critical_columns = True
        
        # If critical columns are missing, further processing is unreliable or will cause crashes.
        # Report all errors found so far (which will include all missing column messages) and stop.
        if missing_critical_columns:
            errors.append("Critical columns are missing. Further detailed validation of rows cannot proceed.")
            raise HTTPException(status_code=422, detail="\n\n".join(errors))

        # --- Content Pre-processing (assuming critical columns are present) ---
        try:
            df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)
        except HTTPException as e: # Catch error from parse_categorical_string
            errors.append(f"Error processing 'CATEGORICAL' column: {e.detail}")
            # If this parsing fails, we might still be able to report other errors, so don't raise immediately
            # unless it's the only error. The final check `if len(errors) > 0` will handle it.

        df["VAR TYPE"] = df.apply(lambda row: str(row["VAR TYPE"]).upper(), axis=1)

        # --- Content Validation: DataFrame-level and Row-level ---
        duplicate_variables = df[df.duplicated(subset=["VARIABLE NAME"], keep=False)]
        if not duplicate_variables.empty:
            errors.append(f"Duplicate VARIABLE NAME found: {', '.join(duplicate_variables['VARIABLE NAME'].unique())}")

        for i, row in df.iterrows():
            var_name_for_error = row.get("VARIABLE NAME", f"UNKNOWN_VAR_ROW_{i+2}")

            # Check if required values are present in rows (for critical columns)
            req_fields = ["VARIABLE NAME", "VARIABLE LABEL", "VAR TYPE", "DOMAIN"]
            for rf in req_fields:
                if not row[rf].strip():
                    errors.append(f"Row {i+2} (Variable: '{var_name_for_error}') is missing value for the required field: '{rf}'.")
            
            if row["VAR TYPE"] not in ACCEPTED_DATATYPES:
                errors.append(
                    f"Row {i+2} (Variable: '{var_name_for_error}') has an invalid data type: '{row['VAR TYPE']}'. Accepted types: {', '.join(ACCEPTED_DATATYPES)}."
                )

            # Handle Category validation (from 'categories' column created by parse_categorical_string)
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
                             warnings.append(
                                 f"Row {i+2} (Variable: '{var_name_for_error}'): The number of category concept codes ({len(categories_codes)}) does not match the number of parsed categories ({len(current_categories)})."
                             )
                        else:
                            for idx, category_data in enumerate(current_categories):
                                if idx < len(categories_codes):
                                    code_to_check = categories_codes[idx].strip()
                                    if code_to_check and code_to_check.lower() != "na":
                                        try:
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
                 pass


        # --- Final Error Check & Graph Generation ---
        if len(errors) > 0:
            raise HTTPException(
                status_code=422,
                detail="\n\n".join(errors),
            )

        # If no errors, proceed to create the graph (RDF triples)
        cohort_uri = get_cohort_uri(cohort_id)
        g = init_graph()
        g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_uri))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_uri))

        for i, row in df.iterrows():
            variable_uri = get_var_uri(cohort_id, row["VARIABLE NAME"])
            g.add((cohort_uri, ICARE.hasVariable, variable_uri, cohort_uri))
            g.add((variable_uri, RDF.type, ICARE.Variable, cohort_uri))
            g.add((variable_uri, DC.identifier, Literal(row["VARIABLE NAME"]), cohort_uri))
            g.add((variable_uri, RDFS.label, Literal(row["VARIABLE LABEL"]), cohort_uri))
            g.add((variable_uri, ICARE["index"], Literal(i, datatype=XSD.integer), cohort_uri))

            # Get categories code if provided (re-fetch for graph generation phase)
            categories_codes = []
            if "CATEGORICAL VALUE CONCEPT CODE" in df.columns and str(row.get("CATEGORICAL VALUE CONCEPT CODE","")).strip():
                 categories_codes = str(row["CATEGORICAL VALUE CONCEPT CODE"]).split("|")

            for column_name_from_df, col_value in row.items():
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
                                cat_code_uri = curie_converter.expand(code_to_check)
                                if cat_code_uri: # Only add if valid and expanded
                                    g.add((cat_uri, ICARE.conceptId, URIRef(cat_code_uri), cohort_uri))
        
        if len(warnings) > 0: # Log warnings even if processing succeeds
            logging.warning(f"Warnings uploading {cohort_id}: \n" + "\n".join(warnings))
        return g

    except HTTPException as http_exc: # Re-raise specific HTTPExceptions (ours or from parse_categorical_string)
        # Log the collected errors that led to this for server-side records
        logging.warning(f"Validation errors for cohort {cohort_id}:\n{http_exc.detail}")
        raise http_exc 
    except pd.errors.EmptyDataError:
        logging.warning(f"Uploaded CSV for cohort {cohort_id} is empty or unreadable.")
        raise HTTPException(status_code=422, detail="The uploaded CSV file is empty or could not be read.")
    except Exception as e:
        logging.error(f"Unexpected error during dictionary processing for {cohort_id}: {str(e)}", exc_info=True)
        # Combine any validation errors found before the crash with the unexpected error message
        final_error_detail = "\n\n".join(errors) if errors else "An unexpected error occurred."
        if errors: # if validation errors were already collected, add the unexpected error to them
            final_error_detail += f"\n\nAdditionally, an unexpected processing error occurred: {str(e)}"
        else: # if no prior validation errors, just report the unexpected one
            final_error_detail = f"An unexpected error occurred during file processing: {str(e)}"
        
        raise HTTPException(
            status_code=500, # Use 500 for truly unexpected server-side issues
            detail=final_error_detail,
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
    cohort_info = retrieve_cohorts_metadata(user_email).get(cohort_id)
    # cohorts = retrieve_cohorts_metadata(user_email)
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
        g = load_cohort_dict_file(metadata_path, cohort_id)
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
            publish_graph_to_endpoint(g)
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
    publish_graph_to_endpoint(g)

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
    cohort_info = retrieve_cohorts_metadata(user["email"]).get(cohort_id)
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


def cohorts_metadata_file_to_graph(filepath: str) -> Dataset:
    df = pd.read_excel(filepath, sheet_name="Descriptions")
    df = df.fillna("")
    g = init_graph()
    for _i, row in df.iterrows():
        cohort_id = str(row["Name of Study"]).strip()
        # print(cohort_id)
        cohort_uri = get_cohort_uri(cohort_id)
        cohorts_graph = ICARE["graph/metadata"]

        g.add((cohort_uri, RDF.type, ICARE.Cohort, cohorts_graph))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohorts_graph))
        g.add((cohort_uri, ICARE.institution, Literal(row["Institution"]), cohorts_graph))
        if row["Contact partner"]:
            g.add((cohort_uri, DC.creator, Literal(row["Contact partner"]), cohorts_graph))
        if row["Email"]:
            for email in row["Email"].split(";"):
                g.add((cohort_uri, ICARE.email, Literal(email.strip()), cohorts_graph))
        if row["Type"]:
            g.add((cohort_uri, ICARE.cohortType, Literal(row["Type"]), cohorts_graph))
        if row["Study Design"]:
            g.add((cohort_uri, ICARE.studyType, Literal(row["Study Design"]), cohorts_graph))
        if row["N"]:
            g.add((cohort_uri, ICARE.studyParticipants, Literal(row["N"]), cohorts_graph))
        if row["Study duration"]:
            g.add((cohort_uri, ICARE.studyDuration, Literal(row["Study duration"]), cohorts_graph))
        if row["Ongoing"]:
            g.add((cohort_uri, ICARE.studyOngoing, Literal(row["Ongoing"]), cohorts_graph))
        if row["Patient population"]:
            g.add((cohort_uri, ICARE.studyPopulation, Literal(row["Patient population"]), cohorts_graph))
        if row["Primary objective"]:
            g.add((cohort_uri, ICARE.studyObjective, Literal(row["Primary objective"]), cohorts_graph))
    return g


def init_triplestore() -> None:
    """Initialize triplestore with the OMOP CDM ontology and the iCARE4CVD cohorts metadata."""
    # If triples exist, skip initialization
    if run_query("ASK WHERE { GRAPH ?g {?s ?p ?o .} }")["boolean"]:
        print("⏩ Triplestore already contains data. Skipping initialization.")
        return
    # Load OMOP CDM ontology
    onto_graph_uri = URIRef("https://w3id.org/icare4cvd/omop-cdm-v6")
    g = init_graph(onto_graph_uri)
    ntriple_g = init_graph()
    # ntriple_g.parse("https://raw.githubusercontent.com/vemonet/omop-cdm-owl/main/omop_cdm_v6.ttl", format="turtle")
    ntriple_g.parse(
        "https://raw.githubusercontent.com/MaastrichtU-IDS/cohort-explorer/main/cohort-explorer-ontology.ttl",
        format="turtle",
    )
    # Trick to convert ntriples to nquads with a given graph
    for s, p, o in ntriple_g.triples((None, None, None)):
        g.add((s, p, o, onto_graph_uri))
    # print(g.serialize(format="trig"))
    if publish_graph_to_endpoint(g):
        print(f"🦉 Triplestore initialization: added {len(g)} triples for the iCARE4CVD Cohort Explorer OWL ontology.")

    os.makedirs(os.path.join(settings.data_folder, "cohorts"), exist_ok=True)
    # Load cohorts data dictionaries already present in data/cohorts/
    for folder in os.listdir(os.path.join(settings.data_folder, "cohorts")):
        folder_path = os.path.join(settings.data_folder, "cohorts", folder)
        if os.path.isdir(folder_path):
            for file in glob.glob(os.path.join(folder_path, "*_datadictionary.*")):
                # NOTE: default airlock preview to false if we ever need to reset cohorts,
                # admins can easily ddl and reupload the cohorts with the correct airlock value
                g = load_cohort_dict_file(file, folder)
                # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.trig", format="trig")
                if publish_graph_to_endpoint(g):
                    print(f"💾 Triplestore initialization: added {len(g)} triples for cohorts {file}.")

    # Load cohorts metadata
    g = cohorts_metadata_file_to_graph(COHORTS_METADATA_FILEPATH)

    # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.ttl", format="turtle")
    if publish_graph_to_endpoint(g):
        print(f"🪪 Triplestore initialization: added {len(g)} triples for the cohorts metadata.")


@router.get("/test-sparql")
def test_sparql(query: str = None, user: Any = Depends(get_current_user)) -> dict:
    """Run test SPARQL queries"""
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # If no query is provided, run the test queries
    if not query:
        results = {}
        
        # Query 1: Count triples in metadata graph
        query1 = """
        SELECT (COUNT(*) as ?count)
        WHERE {
          GRAPH <https://w3id.org/icare4cvd/graph/metadata> { ?s ?p ?o }
        }
        """
        results["metadata_triples_count"] = run_query(query1)
        
        # Query 2: Check what cohorts exist
        query2 = """
        SELECT ?cohort ?cohortId
        WHERE {
          GRAPH <https://w3id.org/icare4cvd/graph/metadata> {
            ?cohort a <https://w3id.org/icare4cvd/Cohort> ;
                    <http://purl.org/dc/elements/1.1/identifier> ?cohortId .
          }
        }
        """
        results["cohorts_in_metadata"] = run_query(query2)
        
        # Query 3: Compare with cohorts that have variables
        query3 = """
        SELECT DISTINCT ?cohort
        WHERE {
          GRAPH ?varGraph {
            ?cohort <https://w3id.org/icare4cvd/hasVariable> ?var .
          }
          FILTER NOT EXISTS {
            GRAPH <https://w3id.org/icare4cvd/graph/metadata> {
              ?cohort a <https://w3id.org/icare4cvd/Cohort> .
            }
          }
        }
        """
        results["orphaned_cohorts"] = run_query(query3)
        
        return results
    
    # If a query is provided, run it
    return run_query(query)