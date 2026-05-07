import csv
from copy import deepcopy
from typing import Any
import os, json
import logging # Add logging import
import threading
import zipfile
import tempfile
from importlib.metadata import version, PackageNotFoundError

import decentriq_platform as dq
from decentriq_platform.analytics import (
    AnalyticsDcrBuilder,
    Column,
    FormatType,
    PreviewComputeNodeDefinition,
    PythonComputeNodeDefinition,
    TableDataNodeDefinition,
    RawDataNodeDefinition
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import get_current_user
from src.config import settings
from src.analysis_dcr_logging import log_dcr_event, read_events
from src.eda_scripts import c1_data_dict_check, c2_save_to_json, c3_eda_data_profiling, shuffle_data
from src.analysisDCR_scripts import data_fragment_script, visualization_script, exploration_script
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata
from datetime import datetime

router = APIRouter()

logger = logging.getLogger(__name__)
try:
    logger.info("decentriq-platform version: %s", version("decentriq-platform"))
except PackageNotFoundError:
    logger.warning("decentriq-platform is not installed")


def get_cohort_schema(cohort_dict: Cohort) -> list[Column]:
    """Convert cohort variables to Decentriq schema"""
    schema = []
    for variable_id, variable_info in cohort_dict.variables.items():
        prim_type = FormatType.STRING
        if variable_info.var_type == "FLOAT":
            prim_type = FormatType.FLOAT
        if variable_info.var_type == "INT":
            prim_type = FormatType.INTEGER
        # If we want to get na from data dictionary (removed for demo)
        # nullable = bool(variable_info.na != 0)

        schema.append(Column(name=variable_id, format_type=prim_type, is_nullable=True))
    return schema

metadatadict_cols_schema2 = [
    Column(name="VARIABLENAME", format_type=FormatType.STRING, is_nullable=True),
    Column(name="VARIABLELABEL", format_type=FormatType.STRING, is_nullable=True),
    Column(name="VARTYPE" , format_type=FormatType.STRING, is_nullable=True),
    Column(name="UNITS", format_type=FormatType.STRING, is_nullable=True),
    Column(name="CATEGORICAL", format_type=FormatType.STRING, is_nullable=True),
    Column(name="MISSING", format_type=FormatType.STRING, is_nullable=True),
    Column(name="COUNT", format_type=FormatType.INTEGER, is_nullable=True),
    Column(name="NA", format_type=FormatType.INTEGER, is_nullable=True),
    Column(name="MIN", format_type=FormatType.STRING, is_nullable=True),
    Column(name="MAX", format_type=FormatType.STRING, is_nullable=True),
    #Column(name="Definition", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Formula", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Categorical Value Concept Code", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Categorical Value Concept Name", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Categorical Value OMOP ID", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Variable Concept Code", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Variable Concept Name", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Variable OMOP ID", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Additional Context Concept Name", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Additional Context Concept Code", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Additional Context OMOP ID", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Unit Concept Name", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Unit Concept Code", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Unit OMOP ID", format_type=FormatType.INTEGER, is_nullable=True),
    Column(name="Domain", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Visits", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Visit OMOP ID", format_type=FormatType.INTEGER, is_nullable=True),
    Column(name="Visit Concept Name", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Visit Concept Code", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Device", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Sensor", format_type=FormatType.STRING, is_nullable=True),
    Column(name="Wearer Location", format_type=FormatType.STRING, is_nullable=True)
]

metadatadict_cols_schema1 = metadatadict_cols_schema2[0:-3]

def identify_cohort_meta_schema(cohort):
    """
    Helper function to identify the appropriate metadata schema for a cohort
    by reading the first line of the metadata dictionary file and counting columns.
    
    Args:
        cohort: Cohort object with metadata_filepath property
        
    Returns:
        The appropriate metadata dictionary columns schema (schema1 or schema2)
    """
    try:
        metadata_file_path = cohort.metadata_filepath
        if not metadata_file_path or not os.path.exists(metadata_file_path):
            print(f"Warning: Metadata file not found for cohort {cohort.cohort_id}. Using schema1 as default.")
            return metadatadict_cols_schema1
            
        with open(metadata_file_path, "rb") as data:
            header = data.readline().decode('utf-8')
            column_count = len(header.split(","))
            print(f"Metadata file for cohort {cohort.cohort_id} has {column_count} columns")
            
            # If the header has at least as many columns as the second schema, use the second schema
            if column_count >= len(metadatadict_cols_schema2):
                return metadatadict_cols_schema2
            else:
                return metadatadict_cols_schema1
    except FileNotFoundError:
        print(f"Warning: Could not find metadata file for cohort {cohort.cohort_id}. Using schema1 as default.")
        return metadatadict_cols_schema1
    except Exception as e:
        print(f"Error identifying metadata schema for cohort {cohort.cohort_id}: {str(e)}. Using schema1 as default.")
        return metadatadict_cols_schema1

# https://docs.decentriq.com/sdk/python-getting-started
def create_provision_dcr(user: Any, cohort: Cohort) -> dict[str, Any]:
    """Initialize a Data Clean Room in Decentriq when a new cohort is uploaded"""
    import time
    start_time = time.time()
    logging.info(f"[TIMING] Starting DCR provisioning for {cohort.cohort_id}")
    
    # Establish connection to Decentriq
    t0 = time.time()
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    logging.info(f"[TIMING] Client creation took {time.time() - t0:.2f}s")

    # Creation of a Data Clean Room (DCR)
    dcr_title = f"iCARE4CVD DCR provision {cohort.cohort_id}"
    builder = (
        AnalyticsDcrBuilder(client=client)
            .with_name(dcr_title)
            .with_owner(user["email"])
            .with_description(f"A data clean room to provision the data for the {cohort.cohort_id} cohort")
    )

    # Create data node for Cohort Data
    data_node_id = cohort.cohort_id.replace(" ", "-")
    #print("\n\nIn create_provision_dcr - columns from get_cohort_schema: ", get_cohort_schema(cohort))
    builder.add_node_definition(
        #TableDataNodeDefinition(name=data_node_id, columns=get_cohort_schema(cohort), is_required=False)
        #https://docs.decentriq.com/sdk/guides/advanced-analytics-dcr/create_dcr
        RawDataNodeDefinition(name=data_node_id, is_required=True)
    )

    #looking at the uploaded file's header to decide which schema to use:
    metadata_file_to_upload = cohort.metadata_filepath 
    if not metadata_file_to_upload or not os.path.exists(metadata_file_to_upload):
        raise FileNotFoundError(f"Physical metadata CSV file for cohort {cohort.cohort_id} not found at expected path: {metadata_file_to_upload or '[No path determined]'}")

    with open(metadata_file_to_upload, "rb") as data:
        header = data.readline()
        header = header.decode('utf-8')
        print("header removed from the file: ", header, "number of columns: ", len(header.split(",")))

    # if the header has at least as many columns as the second schema, use the second schema
    if len(header.split(",")) >= len(metadatadict_cols_schema2):
        metadatadict_cols = metadatadict_cols_schema2
    else:
        metadatadict_cols = metadatadict_cols_schema1


    # Create data node for metadata dictionary file
    metadata_node_id = f"{data_node_id}-metadata"
    builder.add_node_definition(
        TableDataNodeDefinition(name=metadata_node_id, columns=metadatadict_cols, is_required=True)
    )

    # Add scripts that perform EDA to get info about the dataset as a PNG image
    t0 = time.time()
    builder.add_node_definition(
        PythonComputeNodeDefinition(name="c1_data_dict_check", script=c1_data_dict_check(cohort.cohort_id), dependencies=[metadata_node_id, data_node_id])
    )
    builder.add_node_definition(
        PythonComputeNodeDefinition(name="c2_save_to_json", script=c2_save_to_json(cohort.cohort_id), dependencies=[metadata_node_id, data_node_id])
    )
    # This one is auto run by the last script c3_eda_data_profiling:
    #builder.add_node_definition(
    #    PythonComputeNodeDefinition(name="c3_map_missing_do_not_run", script=c3_map_missing_do_not_run(cohort.cohort_id), dependencies=[metadata_node_id, data_node_id, "c2_save_to_json"])
    #)
    builder.add_node_definition(
        PythonComputeNodeDefinition(name="c3_eda_data_profiling", script=c3_eda_data_profiling(cohort.cohort_id), dependencies=["c1_data_dict_check", "c2_save_to_json", metadata_node_id, data_node_id])
    )
    builder.add_node_definition(
        PythonComputeNodeDefinition(name="shuffle_data", script=shuffle_data(cohort.cohort_id), dependencies=[metadata_node_id, data_node_id])
    )
    logging.info(f"[TIMING] Adding 4 EDA script nodes took {time.time() - t0:.2f}s")

    # Add permissions for data owners
    all_participants = set(cohort.cohort_email)
    # Also include administrator_email if it exists
    if cohort.administrator_email:
        all_participants.add(cohort.administrator_email)
    if settings.dev_mode:
        all_participants = set()
        print(f"Dev mode, only adding {user['email']} as data owner")
    #Adding the user whose email & secret were used to create the client above
    # too broad!: all_participants.add(settings.decentriq_email)
    all_participants.add(user["email"])
    for participant in all_participants:
        print(f"Adding {participant} as data owner and analyst")
        builder.add_participant(
            participant,
            data_owner_of=[data_node_id, metadata_node_id],
            # Permission to run scripts:
            analyst_of=["c1_data_dict_check", "c2_save_to_json", "c3_eda_data_profiling", "shuffle_data"],
        )

    if settings.decentriq_email not in all_participants:
        builder.add_participant(settings.decentriq_email, 
                                analyst_of=["c1_data_dict_check", "c2_save_to_json", "c3_eda_data_profiling", "shuffle_data"],
                                data_owner_of=[metadata_node_id])

    # Build and publish DCR
    t0 = time.time()
    dcr_definition = builder.build()
    logging.info(f"[TIMING] Building DCR definition took {time.time() - t0:.2f}s")

    #for debugging:
    print("NOW INSIDE THE provision function!!!", datetime.now())
    print("User ", user)
    with open(f"dcr_{data_node_id}_representation.json", "w") as f:
        json.dump({ "dataScienceDataRoom": dcr_definition._get_high_level_representation() }, f)
    
    t0 = time.time()
    dcr = client.publish_analytics_dcr(dcr_definition)
    logging.info(f"[TIMING] Publishing DCR to Decentriq took {time.time() - t0:.2f}s")
    dcr_url = f"https://platform.decentriq.com/datarooms/p/{dcr.id}"

    # Now the DCR has been created we can upload the metadata file and run computations
    try:
        t0 = time.time()
        key = dq.Key()  # generate an encryption key with which to encrypt the dataset
        metadata_node = dcr.get_node(metadata_node_id)
        
        # The cohort.metadata_filepath property might raise FileNotFoundError
        # This ensures we attempt to get the path within the try-except block
        metadata_file_to_upload = cohort.metadata_filepath 
        
        # Double check existence, though property should raise if not found by its criteria
        if not metadata_file_to_upload or not os.path.exists(metadata_file_to_upload):
             raise FileNotFoundError(f"Physical metadata CSV file for cohort {cohort.cohort_id} not found at expected path: {metadata_file_to_upload or '[No path determined]'}")

        t1 = time.time()
        metadata_noheader_filepath = metadata_file_to_upload.split(".")[0] + "_noHeader.csv"
        with open(metadata_file_to_upload, "rb") as data:
            header = data.readline()
            print("header removed from the file: ", header.decode('utf-8'))
            restfile = data.read()
        with open( metadata_noheader_filepath, "wb") as data_noheader:
            data_noheader.write(restfile)
        os.sync()
        logging.info(f"[TIMING] Preparing metadata file (removing header) took {time.time() - t1:.2f}s")
        
        t1 = time.time()
        with open(metadata_noheader_filepath, "rb") as data_noheader:
            metadata_node.upload_and_publish_dataset(data_noheader, key, f"{metadata_node_id}.csv")
        logging.info(f"[TIMING] Uploading and publishing metadata to DCR took {time.time() - t1:.2f}s")
        logging.info(f"[TIMING] Total metadata upload process took {time.time() - t0:.2f}s")

    except FileNotFoundError as e:
        logging.error(f"Decentriq DCR provisioning: Metadata file not found for cohort {cohort.cohort_id}. Detail: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot create DCR: The metadata dictionary file for cohort '{cohort.cohort_id}' could not be found on the server. Please upload it first via Step 1."
        )
    except Exception as e: 
        logging.error(f"Decentriq DCR provisioning: Error processing metadata for {cohort.cohort_id}. Detail: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while preparing metadata for Decentriq for cohort '{cohort.cohort_id}': {str(e)}"
        )

    #print("columns of the metadata node: ", metadata_node.columns)

    #data_node = dcr.get_node(data_node_id)
    #print("columns of the data node: ", data_node.columns)

    #logging the creation of the DCR:
    
    update_provision_log({"DCR_id": dcr.id, "DCR_url": dcr_url,
                        "cohort_id": cohort.cohort_id,
                      "User": user["email"], "Prov Time":str(datetime.now()),
                      "Metadata file path":cohort.metadata_filepath})

    total_time = time.time() - start_time
    logging.info(f"[TIMING] *** TOTAL DCR provisioning for {cohort.cohort_id} took {total_time:.2f}s ***")

    return {
        "message": f"Data Clean Room for {cohort.cohort_id} provisioned at {dcr_url}",
        "identifier": cohort.cohort_id,
        "dcr_url": dcr_url,
        "dcr_title": dcr_title,
        **cohort.dict(),
    }


def pandas_script_filter_cohort_vars(cohort: Cohort, requested_vars: list[str], df_var: str) -> str:
    """Generate pandas script for filtering variables"""
    if len(requested_vars) <= len(cohort.variables):
        filter_columns_script = f"{df_var} = {df_var}[{requested_vars}]\n"
    return filter_columns_script


def pandas_script_merge_cohorts(merged_cohorts: dict[str, list[str]], all_cohorts: dict[str, Cohort]) -> str:
    """Generate pandas script for merging cohorts on variables mapped_id"""
    # TODO: to be fixed, just here as a starter example
    merge_script = ""
    dfs_to_merge = []
    for cohort_id, vars_requested in merged_cohorts.items():
        if cohort_id not in all_cohorts:
            raise ValueError(f"Cohort {cohort_id} does not exist.")
        df_name = f"df_{cohort_id}"
        vars_mapped = [f"'{var}'" for var in vars_requested]
        dfs_to_merge.append(df_name)
        merge_script += f"{df_name} = pd.DataFrame({cohort_id})[{vars_mapped}]\n"
    # Assuming all dataframes have a common column for merging
    merge_script += f"merged_df = pd.concat([{', '.join(dfs_to_merge)}], ignore_index=True)\n"
    return merge_script


def find_patient_id_variable(cohort_id: str) -> str | None:
    """Find the patient ID variable in a cohort using SNOMED code (preferred) or OMOP ID (fallback).
    
    Searches for:
    1. SNOMED code snomed:184107009 (preferred) - checks concept_code field
    2. OMOP ID 4086934 (fallback) - checks omop_id field
    
    Args:
        cohort_id: The ID of the cohort to search in
        
    Returns:
        The variable name if found, None otherwise
    """
    import time
    start_time = time.time()
    
    # Patient ID identifiers (in order of preference)
    SNOMED_PATIENT_ID = "snomed:184107009"
    OMOP_PATIENT_ID = "4086934"
    
    try:
        from src.cohort_cache import get_cohorts_from_cache
        from src.config import settings
        
        admin_email = settings.admins_list[0] if settings.admins_list else None
        cached_cohorts = get_cohorts_from_cache(admin_email)
        cache_time = time.time()
        
        if cohort_id not in cached_cohorts:
            elapsed = time.time() - start_time
            logging.warning(f"Cohort {cohort_id} not found in cache (took {elapsed:.3f}s)")
            return None
            
        cohort = cached_cohorts[cohort_id]
        
        if not hasattr(cohort, 'variables') or not cohort.variables:
            elapsed = time.time() - start_time
            logging.info(f"No variables in cohort {cohort_id} (took {elapsed:.3f}s)")
            return None
        
        # Log sample of concept_codes and omop_ids for debugging
        sample_codes = []
        for var_name, variable in list(cohort.variables.items())[:5]:
            code = getattr(variable, 'concept_code', None)
            omop = getattr(variable, 'omop_id', None)
            sample_codes.append(f"{var_name}=(code={code}, omop={omop})")
        logging.info(f"Searching for patient ID in cohort {cohort_id}. Sample codes: {sample_codes}")
        
        # First pass: look for SNOMED code (preferred)
        for var_name, variable in cohort.variables.items():
            concept_code = getattr(variable, 'concept_code', None)
            if concept_code:
                concept_code_str = str(concept_code).strip().lower()
                if SNOMED_PATIENT_ID.lower() in concept_code_str or "184107009" in concept_code_str:
                    elapsed = time.time() - start_time
                    logging.info(f"Found patient ID variable '{var_name}' with SNOMED code {concept_code} in cohort {cohort_id} (took {elapsed:.3f}s)")
                    return var_name
        
        # Second pass: look for OMOP ID (fallback)
        for var_name, variable in cohort.variables.items():
            omop_id = getattr(variable, 'omop_id', None)
            if omop_id:
                omop_id_str = str(omop_id).strip()
                if omop_id_str == OMOP_PATIENT_ID or OMOP_PATIENT_ID in omop_id_str:
                    elapsed = time.time() - start_time
                    logging.info(f"Found patient ID variable '{var_name}' with OMOP ID {omop_id} in cohort {cohort_id} (took {elapsed:.3f}s)")
                    return var_name
        
        elapsed = time.time() - start_time
        logging.info(f"No patient ID variable found in cohort {cohort_id} (took {elapsed:.3f}s)")
        return None
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Error finding patient ID variable in cohort {cohort_id} (took {elapsed:.3f}s): {e}")
        return None


def find_variable_by_omop_id(cohort_id: str, omop_id: str) -> str | None:
    """Find a variable in a cohort by its OMOP ID using the cache.
    
    Note: For patient ID specifically, prefer using find_patient_id_variable() which
    also checks SNOMED codes.
    
    Args:
        cohort_id: The ID of the cohort to search in
        omop_id: The OMOP ID to search for (e.g., "4086934", the patient ID variable)
        
    Returns:
        The variable name if found, None otherwise
    """
    import time
    start_time = time.time()
    
    try:
        from src.cohort_cache import get_cohorts_from_cache
        from src.config import settings
        
        admin_email = settings.admins_list[0] if settings.admins_list else None
        cached_cohorts = get_cohorts_from_cache(admin_email)
        cache_time = time.time()
        
        if cohort_id not in cached_cohorts:
            elapsed = time.time() - start_time
            logging.warning(f"Cohort {cohort_id} not found in cache (took {elapsed:.3f}s)")
            return None
            
        cohort = cached_cohorts[cohort_id]
        
        if hasattr(cohort, 'variables') and cohort.variables:
            for var_name, variable in cohort.variables.items():
                if hasattr(variable, 'omop_id') and variable.omop_id:
                    var_omop_id = str(variable.omop_id).strip()
                    if var_omop_id == omop_id or omop_id in var_omop_id:
                        elapsed = time.time() - start_time
                        logging.info(f"Found variable '{var_name}' with OMOP ID {var_omop_id} in cohort {cohort_id} (took {elapsed:.3f}s)")
                        return var_name
        
        elapsed = time.time() - start_time
        logging.info(f"No variable with OMOP ID {omop_id} found in cohort {cohort_id} (took {elapsed:.3f}s)")
        return None
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Error finding variable with OMOP ID {omop_id} in cohort {cohort_id} (took {elapsed:.3f}s): {e}")
        return None


def remove_id_column_from_shuffled_csv(csv_file_path: str, cohort_id: str) -> bool:
    """Post-process shuffled CSV file by removing the ID variable column.
    
    Uses find_patient_id_variable to identify the patient ID variable (SNOMED or OMOP code)
    and removes that column from the CSV file. Does NOT use pandas to preserve data types.
    
    Args:
        csv_file_path: Path to the shuffled CSV file
        cohort_id: The cohort ID to look up the ID variable
        
    Returns:
        True if column was removed, False otherwise
    """
    try:
        # Find the ID variable name using SNOMED code (preferred) or OMOP ID (fallback)
        id_variable_name = find_patient_id_variable(cohort_id)
        
        if not id_variable_name:
            logging.info(f"No patient ID variable found for cohort {cohort_id}, CSV not modified")
            return False
        
        if not os.path.exists(csv_file_path):
            logging.warning(f"CSV file not found: {csv_file_path}")
            return False
        
        # Read the CSV file without pandas
        rows = []
        id_column_index = None
        
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Find the ID column index (case-insensitive match)
            id_variable_lower = id_variable_name.lower()
            for idx, col_name in enumerate(header):
                if col_name.lower() == id_variable_lower:
                    id_column_index = idx
                    logging.info(f"Found ID column '{col_name}' at index {idx} in shuffled CSV")
                    break
            
            if id_column_index is None:
                logging.info(f"ID column '{id_variable_name}' not found in CSV header, CSV not modified")
                return False
            
            # Remove the ID column from header
            new_header = [col for idx, col in enumerate(header) if idx != id_column_index]
            rows.append(new_header)
            
            # Read and process all data rows, removing the ID column
            for row in reader:
                new_row = [val for idx, val in enumerate(row) if idx != id_column_index]
                rows.append(new_row)
        
        # Write the modified CSV back to the same file
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        logging.info(f"Successfully removed ID column '{id_variable_name}' from {csv_file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error removing ID column from shuffled CSV {csv_file_path}: {e}")
        return False


async def get_compute_dcr_definition(
    cohorts_request: dict[str, Any],
    user: Any,
    client: Any,
    include_shuffled_samples: bool | dict[str, bool] = True,
    additional_analysts: list[str] = None,
    airlock_settings: dict[str, int] = None,
    dcr_name: str = None,
    excluded_data_owners: list[str] = None,
    selected_mapping_files: list[dict] = None,
    include_mapping_upload_slot: bool = False,
    research_question: str = None,
) -> Any:
    start_time = datetime.now()
    logging.info(f"Starting DCR definition creation for user {user['email']} at {start_time}")
    
    # users = [user["email"]]
    # TODO: cohorts_request could also be a dict of union of cohorts to merge
    # {"cohorts": {"cohort_id": ["var1", "var2"], "merged_cohort3": {"cohort1": ["weight", "sex"], "cohort2": ["gender", "patient weight"]}}}
    # We automatically merge the cohorts, figuring out which variables are the same thanks to mappings
    metadata_start = datetime.now()
    # Use cache for consistency and performance (same as find_variable_by_omop_id)
    from src.cohort_cache import get_cohorts_from_cache
    from src.config import settings
    admin_email = settings.admins_list[0] if settings.admins_list else None
    all_cohorts = get_cohorts_from_cache(admin_email)
    metadata_time = datetime.now() - metadata_start
    logging.info(f"Retrieved cohorts metadata from cache in {metadata_time.total_seconds():.3f}s")

    # Pre-compute mapping file info for visualization scripts
    # This needs to be done before the cohort loop so we can pass it to visualization_script
    mapping_files_for_viz = []
    selected_mapping_files = selected_mapping_files or []
    for mapping_file in selected_mapping_files:
        cohorts = mapping_file.get('cohorts', [])
        if len(cohorts) >= 2:
            node_name = f"{'_'.join(cohorts)}_mapping"
        else:
            base_name = mapping_file['filename'].replace('.json', '').replace('.csv', '')
            base_name = base_name.replace(' ', '-').replace('(', '').replace(')', '').replace('+', '_')
            node_name = f"mapping_{base_name}"
        mapping_files_for_viz.append({'node_name': node_name})

    # Get metadata for selected cohorts and variables
    selected_cohorts = {}
    for cohort_id, requested_vars in cohorts_request["cohorts"].items():
        cohort_meta = deepcopy(all_cohorts[cohort_id])
        if isinstance(requested_vars, list):
            # Direct cohort variables list
            # NOTE: this block would filter variables only selected by user.
            # We don't want this anymore.
            # Get all cohort and variables metadata for selected variables
            # for var in all_cohorts[cohort_id].variables:
            #     if var not in requested_vars:
            #         del cohort_meta.variables[var]
            selected_cohorts[cohort_id] = cohort_meta
        # elif isinstance(requested_vars, dict):
        #     # Merge operation, need to be implemented on the frontend
        #     pandas_script += pandas_script_merge_cohorts(requested_vars, all_cohorts)
        #     # TODO: add merged cohorts schema to selected_cohorts
        else:
            raise HTTPException(status_code=400, detail=f"Invalid structure for cohort {cohort_id}")


    # Creation of a Data Clean Room (DCR)
    dcr_start = datetime.now()
    data_nodes = []
    metadata_nodes = []
    dcr_count = len(client.get_data_room_descriptions())
    # Use custom DCR name if provided, otherwise use default naming
    if dcr_name and dcr_name.strip():
        dcr_title = dcr_name.strip()
    else:
        dcr_title = f"iCARE4CVD DCR compute {dcr_count+5}" #to avoid issues with duplicate names
    # Append username to DCR title for clarity
    dcr_title = f"{dcr_title} - created by {user['email']}"
    builder = (
        AnalyticsDcrBuilder(client=client)
        .with_name(dcr_title)
        # .with_owner(settings.decentriq_email)
        .with_owner(user["email"])
        .with_description(
            build_dcr_description(
                research_question,
                list(cohorts_request.get("cohorts", {}).keys()),
                user["email"],
            )
        )
    )
    logging.info(f"DCR builder initialized for {len(cohorts_request['cohorts'])} cohorts")

    # Build participants dictionary using shared function
    participants = build_dcr_participants(
        cohorts_request,
        user["email"],
        all_cohorts,
        additional_analysts,
        excluded_data_owners
    )
    preview_nodes = []
    # Convert cohort variables to decentriq schema
    for cohort_id, cohort in selected_cohorts.items():
        # Create data node for cohort (using RawDataNodeDefinition instead of TableDataNodeDefinition)
        data_node_id = cohort_id.replace(" ", "-")
        builder.add_node_definition(
            RawDataNodeDefinition(name=data_node_id, is_required=False)
        )
        data_nodes.append(data_node_id)

        # Add a node for the cohort's metadata dictionary
        metadata_node_id = f"{cohort_id.replace(' ', '-')}_metadata_dictionary"
        
        # Use the helper function to identify the appropriate metadata schema for this cohort
        metadata_cols = identify_cohort_meta_schema(cohort)
        
        builder.add_node_definition(
            TableDataNodeDefinition(name=metadata_node_id, columns=metadata_cols, is_required=False)
        )
        metadata_nodes.append(metadata_node_id)
        
        # Add the requester as data owner of the metadata dictionary node
        participants[user["email"]]["data_owner_of"].add(metadata_node_id)
        
        # Add service account as data owner of metadata node so it can upload files
        if settings.decentriq_email and settings.decentriq_email in participants:
            participants[settings.decentriq_email]["data_owner_of"].add(metadata_node_id)

        # Add data node for shuffled sample if it exists and is requested
        # include_shuffled_samples can be a boolean (legacy) or a dict of cohort_id -> boolean
        should_include_shuffled = False
        if isinstance(include_shuffled_samples, dict):
            should_include_shuffled = include_shuffled_samples.get(cohort_id, False)
        elif include_shuffled_samples:
            should_include_shuffled = True

        # Track whether a shuffled sample data node was actually created for this cohort.
        # Used later to decide whether to add the "test-visualize-shuffled-sample-of-..." compute node.
        shuffled_node_added = False
        if should_include_shuffled:
            storage_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
            shuffled_csv = os.path.join(storage_dir, "shuffled_sample.csv")
            
            if os.path.exists(shuffled_csv):
                shuffled_node_id = f"{cohort_id.replace(' ', '-')}_shuffled_sample"
                logging.info(f"Found shuffled sample for {cohort_id}, adding data node: {shuffled_node_id}")
                
                # Use RawDataNodeDefinition for shuffled samples (no schema needed)
                builder.add_node_definition(
                    RawDataNodeDefinition(name=shuffled_node_id, is_required=False)
                )
                shuffled_node_added = True
                
                # Add data owners for shuffled sample node (participants already have base nodes from build_dcr_participants)
                # Add requester as data owner
                participants[user["email"]]["data_owner_of"].add(shuffled_node_id)
                
                # Add service account as data owner of shuffled sample node so it can upload files
                if settings.decentriq_email and settings.decentriq_email in participants:
                    participants[settings.decentriq_email]["data_owner_of"].add(shuffled_node_id)
                
                # Add existing data owners to shuffled sample
                for email, roles in participants.items():
                    if data_node_id in roles["data_owner_of"]:
                        # If they own the data node, they should also own the shuffled sample
                        roles["data_owner_of"].add(shuffled_node_id)
            else:
                logging.info(f"No shuffled sample found for {cohort_id} at {shuffled_csv}")
        else:
            logging.info(f"Shuffled samples not requested for {cohort_id}, skipping node creation")
        
        # Get variable names from the cohort for documentation in the visualization scripts
        cohort_var_names = list(cohort.variables.keys()) if hasattr(cohort, 'variables') and cohort.variables else None

        # If a shuffled sample data node was provisioned for this cohort, add a
        # dedicated visualization script that defaults to the shuffled sample.
        # This is added FIRST (before airlock and full-dataset scripts) so it appears
        # at the top of the cohort's compute scripts list.
        if shuffled_node_added:
            shuffled_viz_node_name = f"test-visualize-shuffled-sample-of-{cohort_id}"
            shuffled_viz_dependencies = [shuffled_node_id, metadata_node_id]
            # Note: fragment_node_name is not yet defined here, so we pass None.
            # The shuffled-sample viz script doesn't need airlock info anyway.
            shuffled_viz_script = visualization_script(
                None,  # No airlock dependency for shuffled sample viz
                cohort_id,
                cohort_var_names,
                mapping_files_for_viz,
                include_mapping_upload_slot,
                data_source="shuffled",
            )
            builder.add_node_definition(
                PythonComputeNodeDefinition(
                    name=shuffled_viz_node_name,
                    script=shuffled_viz_script,
                    dependencies=shuffled_viz_dependencies,
                )
            )
            # All participants can run the shuffled-sample visualization
            for p_email in participants:
                participants[p_email]["analyst_of"].add(shuffled_viz_node_name)
            logging.info(f"Added shuffled-sample visualization node: {shuffled_viz_node_name}")

        # # Add pandas preparation script - COMMENTED OUT Nov 2025
        # pandas_script = "import pandas as pd\nimport decentriq_util\n\n"
        # df_var = f"df_{cohort_id.replace('-', '_')}"
        # requested_vars = cohorts_request["cohorts"][cohort_id].copy() if isinstance(cohorts_request["cohorts"][cohort_id], list) else cohorts_request["cohorts"][cohort_id]

        # cohort_id_var = find_variable_by_omop_id(cohort_id, "4086934")

        # if cohort_id_var is None:
        #     pandas_script += f"#No cohort ID variable (i.e. no variable with OMOP ID 4086934) found for cohort {cohort_id}\n"
        #     pandas_script += f"#No modifications will be done on the dataframe\n"
        
        # elif cohort_id_var not in requested_vars:
        #     pandas_script += f"#Cohort ID variable ({cohort_id_var}) not in requested variables list for cohort {cohort_id}\n"
        #     pandas_script += f"#No modifications will be done on the dataframe\n"
        # else:
        #     pandas_script += f"#Cohort ID variable ({cohort_id_var}) in among requested variables list for cohort {cohort_id}\n"
        #     pandas_script += f"#The Cohort ID variable will be dropped from the dataframe\n"
        #     requested_vars.remove(cohort_id_var)
        
        # # Direct cohort variables list
        # if isinstance(requested_vars, list):
        #     pandas_script += f'{df_var} = decentriq_util.read_tabular_data("/input/{cohort_id}")\n'
        #     if len(requested_vars) <= len(cohort.variables):
        #         # Add filter variables to pandas script
        #         pandas_script += f"{df_var} = {df_var}[{requested_vars}]\n"

        # # TODO: Merge operation, need to be implemented on the frontend?
        # elif isinstance(requested_vars, dict):
        #     pandas_script += pandas_script_merge_cohorts(requested_vars, all_cohorts)
        #     # TODO: add merged cohorts schema to selected_cohorts
        # else:
        #     raise HTTPException(status_code=400, detail=f"Invalid structure for cohort {cohort_id}")
        # pandas_script += f'\n#The following line commented out - Sept 2025\n'
        # pandas_script += f'#{df_var}.to_csv("/output/{cohort_id}.csv", index=False, header=True)\n\n'

        # # Add python data preparation script
        # builder.add_node_definition(
        #     PythonComputeNodeDefinition(name=f"prepare-{cohort_id}", script=pandas_script, dependencies=[data_node_id])
        # )
        # # Add the requester as analyst of prepare script
        # participants[user["email"]]["analyst_of"].add(f"prepare-{cohort_id}")

        # Get the airlock percentage for this cohort (default to 0 if not specified)
        # A value of 0 means no airlock is requested for this cohort
        airlock_percentage = 0
        if airlock_settings and cohort_id in airlock_settings:
            airlock_percentage = airlock_settings[cohort_id]
        
        # Only create airlock/fragment nodes if airlock is enabled (percentage > 0)
        fragment_node_name = None
        if airlock_percentage > 0:
            # Add data fragment script for this cohort (Dec 2025)
            # This script excludes the ID column and splits the data based on airlock settings
            id_variable_name = find_patient_id_variable(cohort_id)
            
            # Get the data fragment script from the analysisDCR_scripts module
            fragment_script = data_fragment_script(cohort_id, id_variable_name, airlock_percentage)
            fragment_node_name = f"create-airlock-without-outliers-{cohort_id}"
            
            builder.add_node_definition(
                PythonComputeNodeDefinition(
                    name=fragment_node_name,
                    script=fragment_script,
                    dependencies=[data_node_id, metadata_node_id]
                )
            )
            # Add data owners of this cohort's data node as analysts of the fragmentation script
            # (not the requester or other analysts - only data owners can run the fragmentation)
            for p_email, p_roles in participants.items():
                if data_node_id in p_roles["data_owner_of"]:
                    participants[p_email]["analyst_of"].add(fragment_node_name)
            
            # Add a preview (airlock) node for the data fragment
            preview_node_name = f"preview-airlock-{cohort_id}"
            builder.add_node_definition(
                PreviewComputeNodeDefinition(
                    name=preview_node_name,
                    dependency=f"create-airlock-without-outliers-{cohort_id}",
                    quota_bytes=10000000  # 10 MB quota
                )
            )
            # Add the requester as analyst of the preview node
            participants[user["email"]]["analyst_of"].add(preview_node_name)
            # Track the preview node for additional analysts
            preview_nodes.append(preview_node_name)
            logging.info(f"Created airlock nodes for {cohort_id} with {airlock_percentage}% airlock")
        else:
            logging.info(f"Skipping airlock nodes for {cohort_id} (airlock disabled)")

        # Add a visualization script that reads the FULL cohort dataset (the original
        # unprocessed data). The node name makes the data source explicit.
        visualization_node_name = f"visualize-full-dataset-of-{cohort_id}"

        # Visualization script dependencies depend on whether airlock is enabled
        if fragment_node_name:
            viz_dependencies = [data_node_id, metadata_node_id, fragment_node_name]
        else:
            viz_dependencies = [data_node_id, metadata_node_id]

        viz_script = visualization_script(
            fragment_node_name,
            cohort_id,
            cohort_var_names,
            mapping_files_for_viz,
            include_mapping_upload_slot,
            data_source="full",
        )
        builder.add_node_definition(
            PythonComputeNodeDefinition(
                name=visualization_node_name,
                script=viz_script,
                dependencies=viz_dependencies
            )
        )
        # Add all participants as analysts of the visualization script
        for p_email in participants:
            participants[p_email]["analyst_of"].add(visualization_node_name)



    # Define exploration script (will be added to builder first, after cohort loop)
    # Get the exploration script from the analysisDCR_scripts module
    explore_script = exploration_script()
    
    # Add the exploration script FIRST (before other nodes) with dependencies on all metadata nodes
    # COMMENTED OUT - Keeping script definition but not adding to DCR
    # builder.add_node_definition(
    #     PythonComputeNodeDefinition(
    #         name="optional-basic-data-exploration",
    #         script=explore_script,
    #         dependencies=metadata_nodes
    #     )
    # )
    # # Add the requester as analyst of the exploration script
    # participants[user["email"]]["analyst_of"].add("optional-basic-data-exploration")
    # 
    # # Add service account as analyst of the exploration script
    # if settings.decentriq_email and settings.decentriq_email in participants:
    #     participants[settings.decentriq_email]["analyst_of"].add("optional-basic-data-exploration")

    # Add mapping file nodes if any are selected
    mapping_nodes = []
    selected_mapping_files = selected_mapping_files or []
    
    if selected_mapping_files:
        logging.info(f"Adding {len(selected_mapping_files)} mapping file nodes to DCR")
        for mapping_file in selected_mapping_files:
            # Create node name from cohort IDs: cohortID1_cohortID2_mapping
            cohorts = mapping_file.get('cohorts', [])
            if len(cohorts) >= 2:
                node_name = f"{'_'.join(cohorts)}_mapping"
            else:
                # Fallback to old naming if cohorts not available
                base_name = mapping_file['filename'].replace('.json', '').replace('.csv', '')
                base_name = base_name.replace(' ', '-').replace('(', '').replace(')', '').replace('+', '_')
                node_name = f"mapping_{base_name}"
            
            builder.add_node_definition(
                RawDataNodeDefinition(name=node_name, is_required=False)
            )
            mapping_nodes.append({
                'node_name': node_name,
                'filepath': mapping_file['filepath'],
                'display_name': mapping_file.get('display_name', mapping_file['filename']),
                'cohorts': mapping_file.get('cohorts', [])
            })
            
            # Add requester as data owner
            participants[user["email"]]["data_owner_of"].add(node_name)
            
            # Add service account as data owner so it can upload files
            if settings.decentriq_email and settings.decentriq_email in participants:
                participants[settings.decentriq_email]["data_owner_of"].add(node_name)
            
            # Add all participants (data owners and analysts) as owners of mapping nodes
            for email, roles in participants.items():
                roles["data_owner_of"].add(node_name)
            
            logging.info(f"Added mapping node: {node_name} for file: {mapping_file['filepath']}")
    
    # Add CrossStudyMappings upload slot only if explicitly requested
    if include_mapping_upload_slot:
        logging.info("Adding CrossStudyMappings upload slot (explicitly requested)")
        builder.add_node_definition(
            RawDataNodeDefinition(name="CrossStudyMappings", is_required=False)
        )
        participants[user["email"]]["data_owner_of"].add("CrossStudyMappings")
        
        # Add service account as data owner of CrossStudyMappings so it can upload files
        if settings.decentriq_email and settings.decentriq_email in participants:
            participants[settings.decentriq_email]["data_owner_of"].add("CrossStudyMappings")
        
        # Add all participants (data owners and analysts) as owners of CrossStudyMappings
        for email, roles in participants.items():
            roles["data_owner_of"].add("CrossStudyMappings")
    
    # Add users permissions for previews
    # for prev_node in preview_nodes:
    #     participants[user["email"]]["analyst_of"].add(prev_node)

    # Add analyst_of roles for additional analysts (they're already in participants from build_dcr_participants)
    # COMMENTED OUT - exploration script permissions
    # if additional_analysts:
    #     for analyst_email in additional_analysts:
    #         if analyst_email and analyst_email != user["email"] and analyst_email in participants:
    #             # Add as analyst of the exploration script
    #             participants[analyst_email]["analyst_of"].add("optional-basic-data-exploration")
    #             logging.info(f"Added {analyst_email} as additional analyst")

    # Grant all participants (data owners and analysts) access to preview fragment nodes
    for p_email in participants.keys():
        for preview_node in preview_nodes:
            participants[p_email]["analyst_of"].add(preview_node)
    
    logging.info(f"Granted all {len(participants)} participants access to {len(preview_nodes)} preview fragment nodes")

    # Log detailed participant permissions before adding to builder
    logging.info("=" * 80)
    logging.info("PARTICIPANT PERMISSIONS SUMMARY")
    logging.info("=" * 80)
    for p_email, p_perm in participants.items():
        logging.info(f"\nParticipant: {p_email}")
        logging.info(f"  Data Owner Of ({len(p_perm['data_owner_of'])} nodes): {sorted(list(p_perm['data_owner_of']))}")
        logging.info(f"  Analyst Of ({len(p_perm['analyst_of'])} nodes): {sorted(list(p_perm['analyst_of']))}")
        builder.add_participant(
            p_email,
            data_owner_of=list(p_perm["data_owner_of"]),
            analyst_of=list(p_perm["analyst_of"]),
        )
    logging.info("=" * 80)

    # Build and publish DCR
    build_start = datetime.now()
    dcr_definition = builder.build()
    build_time = datetime.now() - build_start
    
    total_time = datetime.now() - start_time
    logging.info(f"DCR build completed in {build_time.total_seconds():.3f}s")
    logging.info(f"Total DCR definition creation completed in {total_time.total_seconds():.3f}s for {len(cohorts_request['cohorts'])} cohorts")
    
    return dcr_definition, dcr_title, participants, mapping_nodes



async def create_live_compute_dcr(
    cohorts_request: dict[str, Any],
    user: Any,
    client: Any,
    include_shuffled_samples: bool | dict[str, bool] = True,
    additional_analysts: list[str] = None,
    airlock_settings: dict[str, int] = None,
    dcr_name: str = None,
    excluded_data_owners: list[str] = None,
    selected_mapping_files: list[dict] = None,
    include_mapping_upload_slot: bool = False,
    research_question: str = None,
) -> dict[str, Any]:
    """Create and publish a live compute DCR that is immediately available for use.
    
    This function combines the DCR definition creation from get_compute_dcr_definition
    with the actual room creation and publishing from create_provision_dcr.
    
    Args:
        cohorts_request: Dictionary with cohort IDs and requested variables
        user: User information dictionary with email
        client: Decentriq client instance
        include_shuffled_samples: Whether to include shuffled sample data nodes (bool or dict of cohort_id -> bool)
        additional_analysts: List of email addresses to add as analysts
        excluded_data_owners: List of data owner emails to exclude from the DCR
        selected_mapping_files: List of mapping files to include in the DCR
        include_mapping_upload_slot: Whether to include a CrossStudyMappings upload slot
        
    Returns:
        Dictionary with DCR information including ID, URL, and title
    """
    start_time = datetime.now()
    logging.info(f"Starting live compute DCR creation for user {user['email']} at {start_time}")
    
    # Step 1: Create the DCR definition (reuse existing logic)
    dcr_definition, dcr_title, participants, mapping_nodes = await get_compute_dcr_definition(cohorts_request, user, client, include_shuffled_samples, additional_analysts, airlock_settings, dcr_name, excluded_data_owners, selected_mapping_files, include_mapping_upload_slot, research_question)
    
    # Step 2: Publish the DCR to Decentriq with retry logic for race conditions
    import time
    logging.info(f"Publishing live compute DCR: {dcr_title}")
    publish_start = datetime.now()
    
    dcr = None
    max_retries = 4
    retry_delay_seconds = 3  # Start with a 6-second delay
    error_messages = {}
    for attempt in range(max_retries):
        time.sleep(retry_delay_seconds)
        try:
            # Attempt to publish the DCR
            dcr = client.publish_analytics_dcr(dcr_definition)
            # If successful, break the loop
            logging.info(f"DCR published successfully on attempt {attempt + 1}")
            break
        except Exception as e:
            error_messages[attempt+1] = e
            logging.error(f"Error occurred during DCR publication attempt {attempt + 1}: {e}")
            
            retry_delay_seconds *= 1.5
            
    
    # If the loop completes without success, raise a final error
    if dcr is None:
        raise Exception(f"Failed to publish DCR '{dcr_title}' after {max_retries} attempts.\nErrors for each attempt: {error_messages}")
    
    publish_time = datetime.now() - publish_start
    logging.info(f"DCR published successfully in {publish_time.total_seconds():.3f}s")
    
    try:
        dcr_url = f"https://platform.decentriq.com/datarooms/p/{dcr.id}"
        
        # Step 3: Upload metadata dictionaries for each cohort
        from src.cohort_cache import get_cohorts_from_cache
        from src.config import settings
        from src.upload import get_latest_datadictionary
        admin_email = settings.admins_list[0] if settings.admins_list else None
        all_cohorts = get_cohorts_from_cache(admin_email)
        
        logging.info(f"Starting metadata upload for {len(cohorts_request['cohorts'])} cohorts")
        logging.info(f"Cohort folder path: {settings.cohort_folder}")
        logging.info(f"Cohorts to process: {list(cohorts_request['cohorts'].keys())}")
        
        upload_start = datetime.now()
        metadata_upload_results = {}
        
        for cohort_id in cohorts_request["cohorts"].keys():
            try:
                cohort = all_cohorts.get(cohort_id)
                if not cohort:
                    logging.warning(f"Cohort {cohort_id} not found in cache, skipping metadata upload")
                    metadata_upload_results[cohort_id] = "cohort_not_found"
                    continue
                
                metadata_node_id = f"{cohort_id.replace(' ', '-')}_metadata_dictionary"
                
                # Get the latest metadata dictionary file using the utility function
                cohort_folder_path = os.path.join(settings.cohort_folder, cohort_id)
                logging.info(f"Looking for metadata dictionary in: {cohort_folder_path}")
                
                # Check if folder exists
                if not os.path.exists(cohort_folder_path):
                    logging.warning(f"Cohort folder does not exist: {cohort_folder_path}")
                    metadata_upload_results[cohort_id] = "folder_not_found"
                    continue
                
                # List files in the folder
                try:
                    files_in_folder = os.listdir(cohort_folder_path)
                    logging.info(f"Files in {cohort_id} folder: {files_in_folder}")
                except Exception as e:
                    logging.error(f"Error listing files in {cohort_folder_path}: {e}")
                
                metadata_file_to_upload = get_latest_datadictionary(cohort_folder_path)
                
                if not metadata_file_to_upload:
                    logging.warning(f"No metadata dictionary file found for cohort {cohort_id} in {cohort_folder_path}")
                    metadata_upload_results[cohort_id] = "file_not_found"
                    continue
                
                logging.info(f"Found metadata file for {cohort_id}: {metadata_file_to_upload}")
                
                if not os.path.exists(metadata_file_to_upload):
                    logging.warning(f"Metadata file does not exist for cohort {cohort_id} at {metadata_file_to_upload}")
                    metadata_upload_results[cohort_id] = "file_not_exists"
                    continue
                
                # Get the metadata node from the DCR
                metadata_node = dcr.get_node(metadata_node_id)
                
                # Generate encryption key
                key = dq.Key()
                
                # Remove header from metadata file (same as provision DCR)
                metadata_noheader_filepath = metadata_file_to_upload.split(".")[0] + "_noHeader.csv"
                with open(metadata_file_to_upload, "rb") as data:
                    header = data.readline()
                    logging.info(f"Removed header from {cohort_id} metadata: {header.decode('utf-8').strip()}")
                    restfile = data.read()
                
                with open(metadata_noheader_filepath, "wb") as data_noheader:
                    data_noheader.write(restfile)
                os.sync()
                
                # Upload and publish the metadata
                with open(metadata_noheader_filepath, "rb") as data_noheader:
                    metadata_node.upload_and_publish_dataset(data_noheader, key, f"{metadata_node_id}.csv")
                
                logging.info(f"Successfully uploaded metadata for cohort {cohort_id}")
                metadata_upload_results[cohort_id] = "success"
                
            except Exception as e:
                logging.error(f"Failed to upload metadata for cohort {cohort_id}: {e}", exc_info=True)
                metadata_upload_results[cohort_id] = f"error: {str(e)}"
        
        upload_time = datetime.now() - upload_start
        logging.info(f"Metadata upload completed in {upload_time.total_seconds():.3f}s. Results: {metadata_upload_results}")
        
        # Step 4: Upload shuffled sample files if they exist and are requested
        shuffled_upload_start = datetime.now()
        shuffled_upload_results = {}
        
        for cohort_id in cohorts_request["cohorts"].keys():
            # Check if shuffled samples should be included for this cohort
            # include_shuffled_samples can be a boolean (legacy) or a dict of cohort_id -> boolean
            should_include_shuffled = False
            if isinstance(include_shuffled_samples, dict):
                should_include_shuffled = include_shuffled_samples.get(cohort_id, False)
            elif include_shuffled_samples:
                should_include_shuffled = True
            
            if not should_include_shuffled:
                shuffled_upload_results[cohort_id] = "not_requested"
                continue
                
            try:
                storage_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
                shuffled_csv = os.path.join(storage_dir, "shuffled_sample.csv")
                
                if not os.path.exists(shuffled_csv):
                    logging.info(f"No shuffled sample file for cohort {cohort_id}, skipping")
                    shuffled_upload_results[cohort_id] = "no_file"
                    continue
                
                shuffled_node_id = f"{cohort_id.replace(' ', '-')}_shuffled_sample"
                
                # Get the shuffled sample node from the DCR
                try:
                    shuffled_node = dcr.get_node(shuffled_node_id)
                except Exception as node_error:
                    logging.warning(f"Shuffled sample node not found for {cohort_id}: {node_error}")
                    shuffled_upload_results[cohort_id] = "node_not_found"
                    continue
                
                # Generate encryption key
                key = dq.Key()
                
                # Upload and publish the shuffled sample
                with open(shuffled_csv, "rb") as shuffled_data:
                    shuffled_node.upload_and_publish_dataset(shuffled_data, key, f"{shuffled_node_id}.csv")
                
                logging.info(f"Successfully uploaded shuffled sample for cohort {cohort_id}")
                shuffled_upload_results[cohort_id] = "success"
                
            except Exception as e:
                logging.error(f"Failed to upload shuffled sample for cohort {cohort_id}: {e}", exc_info=True)
                shuffled_upload_results[cohort_id] = f"error: {str(e)}"
        
        shuffled_upload_time = datetime.now() - shuffled_upload_start
        logging.info(f"Shuffled sample upload completed in {shuffled_upload_time.total_seconds():.3f}s. Results: {shuffled_upload_results}")
        
        # Step 5: Upload mapping files
        mapping_upload_results = {}
        if mapping_nodes:
            logging.info(f"Starting mapping file upload for {len(mapping_nodes)} files")
            mapping_upload_start = datetime.now()
            
            for mapping_info in mapping_nodes:
                node_name = mapping_info['node_name']
                filepath = mapping_info['filepath']
                display_name = mapping_info.get('display_name', node_name)
                
                try:
                    if not os.path.exists(filepath):
                        logging.warning(f"Mapping file not found: {filepath}")
                        mapping_upload_results[node_name] = "file_not_found"
                        continue
                    
                    # Get the mapping node from the DCR
                    try:
                        mapping_node = dcr.get_node(node_name)
                    except Exception as node_error:
                        logging.warning(f"Mapping node not found for {node_name}: {node_error}")
                        mapping_upload_results[node_name] = "node_not_found"
                        continue
                    
                    # Generate encryption key
                    key = dq.Key()
                    
                    # Generate upload filename in format: mapping__cohortID1__to__cohortID2__cohortID3.json
                    cohorts = mapping_info.get('cohorts', [])
                    if len(cohorts) >= 2:
                        # First cohort is source, rest are targets joined with __
                        upload_filename = f"mapping__{cohorts[0]}__to__{'__'.join(cohorts[1:])}.json"
                    else:
                        # Fallback to original filename if cohorts not available
                        upload_filename = os.path.basename(filepath)
                    
                    # Upload and publish the mapping file
                    with open(filepath, "rb") as mapping_data:
                        mapping_node.upload_and_publish_dataset(mapping_data, key, upload_filename)
                    
                    logging.info(f"Uploaded mapping file as: {upload_filename}")
                    
                    logging.info(f"Successfully uploaded mapping file: {display_name}")
                    mapping_upload_results[node_name] = "success"
                    
                except Exception as e:
                    logging.error(f"Failed to upload mapping file {node_name}: {e}", exc_info=True)
                    mapping_upload_results[node_name] = f"error: {str(e)}"
            
            mapping_upload_time = datetime.now() - mapping_upload_start
            logging.info(f"Mapping file upload completed in {mapping_upload_time.total_seconds():.3f}s. Results: {mapping_upload_results}")
        
        # Step 6: Log the DCR creation
        cohort_ids = list(cohorts_request["cohorts"].keys())
        log_data = {
            "DCR_id": dcr.id,
            "DCR_url": dcr_url,
            "DCR_type": "live_compute",
            "cohort_ids": cohort_ids,
            "user": user["email"],
            "creation_time": str(datetime.now()),
            "num_cohorts": len(cohort_ids)
        }
        
        # Log to compute DCR log file
        compute_dcr_log = "/data/live_compute_dcrs.jsonl"
        try:
            with open(compute_dcr_log, "a") as f:
                f.write(json.dumps(log_data) + "\n")
            logging.info(f"Logged live compute DCR creation to {compute_dcr_log}")
        except Exception as log_error:
            logging.warning(f"Failed to log DCR creation: {log_error}")
        
        total_time = datetime.now() - start_time
        logging.info(f"Live compute DCR creation completed in {total_time.total_seconds():.3f}s")
        
        # Count successful uploads
        successful_metadata_uploads = sum(1 for status in metadata_upload_results.values() if status == "success")
        successful_shuffled_uploads = sum(1 for status in shuffled_upload_results.values() if status == "success")
        successful_mapping_uploads = sum(1 for status in mapping_upload_results.values() if status == "success")
        
        # Convert participants dictionary for JSON serialization
        participants_json = {}
        for email, roles in participants.items():
            participants_json[email] = {
                "data_owner_of": list(roles["data_owner_of"]),
                "analyst_of": list(roles["analyst_of"])
            }
        
        mapping_msg = f" Mapping files uploaded: {successful_mapping_uploads}/{len(mapping_nodes)}." if mapping_nodes else ""
        return {
            "message": f"Live compute DCR created successfully for {len(cohort_ids)} cohort(s). Metadata uploaded for {successful_metadata_uploads}/{len(cohort_ids)} cohorts. Shuffled samples uploaded for {successful_shuffled_uploads}/{len(cohort_ids)} cohorts.{mapping_msg}",
            "dcr_id": dcr.id,
            "dcr_url": dcr_url,
            "dcr_title": dcr_title,
            "cohort_ids": cohort_ids,
            "num_cohorts": len(cohort_ids),
            "metadata_upload_results": metadata_upload_results,
            "metadata_uploads_successful": successful_metadata_uploads,
            "shuffled_upload_results": shuffled_upload_results,
            "shuffled_uploads_successful": successful_shuffled_uploads,
            "mapping_upload_results": mapping_upload_results,
            "mapping_uploads_successful": successful_mapping_uploads,
            "participants": participants_json
        }
        
    except Exception as e:
        logging.error(f"Failed to publish live compute DCR: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create live compute DCR: {str(e)}"
        )


@router.post(
    "/create-live-compute-dcr",
    name="Create and publish a live compute DCR",
    response_description="DCR information with ID and URL",
)
async def api_create_live_compute_dcr(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Create and publish a live compute Data Clean Room that is immediately available.
    
    This endpoint creates a DCR with the requested cohorts and publishes it to Decentriq,
    making it immediately available for data owners to provision data and analysts to run computations.
    
    Unlike /get-compute-dcr-definition which only returns the configuration,
    this endpoint actually creates the room on the Decentriq platform.
    
    Args:
        cohorts_request: Dictionary with structure:
            {
                "cohorts": {"cohort_id": ["var1", "var2", ...], ...},
                "include_shuffled_samples": true/false (optional, defaults to true)
            }
        user: Current authenticated user
        
    Returns:
        Dictionary with DCR ID, URL, title, and cohort information
    """
    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    
    # Extract include_shuffled_samples from request, default to True
    include_shuffled_samples = cohorts_request.get("include_shuffled_samples", True)
    
    # Extract additional_analysts from request, default to empty list
    additional_analysts = cohorts_request.get("additional_analysts", [])
    
    # Extract airlock_settings from request, default to empty dict
    airlock_settings = cohorts_request.get("airlock_settings", {})
    
    # Extract dcr_name from request, default to None
    dcr_name = cohorts_request.get("dcr_name", None)

    # Extract research_question from request, used as DCR description
    research_question = cohorts_request.get("research_question", None)

    # Extract excluded_data_owners from request, default to empty list
    excluded_data_owners = cohorts_request.get("excluded_data_owners", [])
    
    # Extract selected_mapping_files from request, default to empty list
    selected_mapping_files = cohorts_request.get("selected_mapping_files", [])
    
    # Extract include_mapping_upload_slot from request, default to False
    include_mapping_upload_slot = cohorts_request.get("include_mapping_upload_slot", False)

    # Extract session_id from request (frontend-generated UUID correlating wizard events)
    session_id = cohorts_request.get("session_id")

    # Resolve the full participants list (cohort data owners + additional analysts
    # + service account, minus excluded owners) so the activity log captures who
    # the DCR is being created for, not just the requester's inputs.
    try:
        from src.cohort_cache import get_cohorts_from_cache
        resolved_participants = build_dcr_participants(
            cohorts_request,
            user["email"],
            get_cohorts_from_cache(user.get("email")),
            additional_analysts,
            excluded_data_owners,
        )
        participants_for_log = {
            email: {
                "data_owner_of": sorted(list(roles.get("data_owner_of", set()))),
                "analyst_of": sorted(list(roles.get("analyst_of", set()))),
            }
            for email, roles in resolved_participants.items()
        }
    except Exception as exc:
        logging.warning("Failed to resolve participants for logging: %s", exc)
        participants_for_log = None

    publish_started_at = datetime.now()
    log_dcr_event(
        "dcr_publish_started",
        user_email=user.get("email"),
        session_id=session_id,
        dcr_name=dcr_name,
        cohorts=list(cohorts_request.get("cohorts", {}).keys()),
        research_question=research_question,
        additional_analysts=additional_analysts,
        excluded_data_owners=excluded_data_owners,
        airlock_settings=airlock_settings,
        include_shuffled_samples=include_shuffled_samples,
        selected_mapping_files=[m.get("filename") for m in (selected_mapping_files or []) if isinstance(m, dict)],
        include_mapping_upload_slot=include_mapping_upload_slot,
        participants=participants_for_log,
    )

    # Create and publish the live compute DCR
    try:
        result = await create_live_compute_dcr(cohorts_request, user, client, include_shuffled_samples, additional_analysts, airlock_settings, dcr_name, excluded_data_owners, selected_mapping_files, include_mapping_upload_slot, research_question)
        duration_ms = int((datetime.now() - publish_started_at).total_seconds() * 1000)
        log_dcr_event(
            "dcr_publish_succeeded",
            user_email=user.get("email"),
            session_id=session_id,
            duration_ms=duration_ms,
            dcr_id=result.get("dcr_id") if isinstance(result, dict) else None,
            dcr_url=result.get("dcr_url") if isinstance(result, dict) else None,
            dcr_title=result.get("dcr_title") if isinstance(result, dict) else None,
            dcr_name=dcr_name,
            cohorts=list(cohorts_request.get("cohorts", {}).keys()),
            research_question=research_question,
            additional_analysts=additional_analysts,
            excluded_data_owners=excluded_data_owners,
            airlock_settings=airlock_settings,
            include_shuffled_samples=include_shuffled_samples,
            participants=participants_for_log,
        )
        return result
    except Exception as e:
        duration_ms = int((datetime.now() - publish_started_at).total_seconds() * 1000)
        log_dcr_event(
            "dcr_publish_failed",
            user_email=user.get("email"),
            session_id=session_id,
            duration_ms=duration_ms,
            error_type=type(e).__name__,
            error_message=str(e),
            cohorts=list(cohorts_request.get("cohorts", {}).keys()),
        )
        error_msg = f"Failed to create live compute DCR: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Full error details: {repr(e)}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post(
    "/get-compute-dcr-definition",
    name="Get the Data Clean Room definition for computing as JSON or ZIP with shuffled samples",
    response_description="DCR definition JSON or ZIP file with config and shuffled samples",
)
async def api_get_compute_dcr_definition(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> Any:
    """Create a Data Clean Room for computing with the cohorts requested using Decentriq SDK.
    
    If shuffled sample files exist for any of the selected cohorts, returns a ZIP file containing:
    - dcr_config.json: The DCR configuration
    - {cohort_id}_shuffled_sample.csv: Shuffled sample data for each cohort (if available)
    - {cohort_id}_shuffle_summary.txt: Shuffle summary for each cohort (if available)
    
    If no shuffled samples exist, returns just the JSON configuration.
    
    Args:
        cohorts_request: Dictionary with structure:
            {
                "cohorts": {"cohort_id": ["var1", "var2", ...], ...},
                "include_shuffled_samples": true/false (optional, defaults to true)
            }
    """
    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)

    # Extract include_shuffled_samples from request, default to True
    include_shuffled_samples = cohorts_request.get("include_shuffled_samples", True)
    # Extract dcr_name from request, default to None
    dcr_name = cohorts_request.get("dcr_name", None)

    # Extract selected_mapping_files from request
    selected_mapping_files = cohorts_request.get("selected_mapping_files", [])

    session_id = cohorts_request.get("session_id")
    log_dcr_event(
        "dcr_preview_requested",
        user_email=user.get("email"),
        session_id=session_id,
        dcr_name=dcr_name,
        cohorts=list(cohorts_request.get("cohorts", {}).keys()),
        include_shuffled_samples=include_shuffled_samples,
    )

    dcr_definition, _dcr_title, _participants, _mapping_nodes = await get_compute_dcr_definition(cohorts_request, user, client, include_shuffled_samples, dcr_name=dcr_name)

    # Generate DCR config JSON
    dcr_config_json = { "dataScienceDataRoom": dcr_definition.high_level }
    
    # Collect all ancillary files to include in ZIP
    shuffled_files = {}  # cohort_id -> {csv: path, summary: path}
    metadata_files = {}  # cohort_id -> path
    mapping_files_to_include = []  # list of {filepath: path, filename: name}
    
    # Get cohort metadata from cache for metadata dictionary paths
    from src.cohort_cache import get_cohorts_from_cache
    all_cohorts = get_cohorts_from_cache(user.get("email"))
    
    for cohort_id in cohorts_request["cohorts"].keys():
        # Check for shuffled samples
        should_include_shuffled = False
        if isinstance(include_shuffled_samples, dict):
            should_include_shuffled = include_shuffled_samples.get(cohort_id, False)
        elif include_shuffled_samples:
            should_include_shuffled = True
        
        if should_include_shuffled:
            storage_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
            shuffled_csv = os.path.join(storage_dir, "shuffled_sample.csv")
            shuffled_summary = os.path.join(storage_dir, "shuffle_summary.txt")
            
            cohort_files = {}
            if os.path.exists(shuffled_csv):
                cohort_files["csv"] = shuffled_csv
                logging.info(f"Found shuffled sample CSV for cohort {cohort_id}")
            if os.path.exists(shuffled_summary):
                cohort_files["summary"] = shuffled_summary
                logging.info(f"Found shuffle summary for cohort {cohort_id}")
            
            if cohort_files:
                shuffled_files[cohort_id] = cohort_files
        
        # Check for metadata dictionary
        if cohort_id in all_cohorts:
            try:
                cohort = all_cohorts[cohort_id]
                metadata_path = cohort.metadata_filepath
                if os.path.exists(metadata_path):
                    metadata_files[cohort_id] = metadata_path
                    logging.info(f"Found metadata dictionary for cohort {cohort_id}")
            except (FileNotFoundError, AttributeError) as e:
                logging.info(f"No metadata dictionary found for cohort {cohort_id}: {e}")
    
    # Collect mapping files
    for mapping_file in selected_mapping_files:
        filepath = mapping_file.get('filepath')
        if filepath and os.path.exists(filepath):
            mapping_files_to_include.append({
                'filepath': filepath,
                'filename': os.path.basename(filepath)
            })
            logging.info(f"Found mapping file: {filepath}")
    
    # Always create ZIP with ancillary files organized in subfolders
    logging.info(f"Creating ZIP with DCR config and ancillary files")
    logging.info(f"  - Shuffled samples: {len(shuffled_files)} cohorts")
    logging.info(f"  - Metadata dictionaries: {len(metadata_files)} cohorts")
    logging.info(f"  - Mapping files: {len(mapping_files_to_include)} files")
    
    # Create temp file that won't be auto-deleted (we'll handle cleanup)
    temp_zip = tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False)
    temp_zip_path = temp_zip.name
    temp_zip.close()
    
    try:
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add DCR config JSON at root level
            config_json_str = json.dumps(dcr_config_json, indent=2)
            zipf.writestr("dcr_config.json", config_json_str)
            logging.info("Added dcr_config.json to ZIP")
            
            # Add shuffled sample files in shuffled_samples/ subfolder
            for cohort_id, files in shuffled_files.items():
                if "csv" in files:
                    zipf.write(files["csv"], f"shuffled_samples/{cohort_id}_shuffled_sample.csv")
                    logging.info(f"Added shuffled_samples/{cohort_id}_shuffled_sample.csv to ZIP")
                
                if "summary" in files:
                    zipf.write(files["summary"], f"shuffled_samples/{cohort_id}_shuffle_summary.txt")
                    logging.info(f"Added shuffled_samples/{cohort_id}_shuffle_summary.txt to ZIP")
            
            # Add metadata dictionaries in metadata_dictionaries/ subfolder
            for cohort_id, metadata_path in metadata_files.items():
                filename = os.path.basename(metadata_path)
                zipf.write(metadata_path, f"metadata_dictionaries/{filename}")
                logging.info(f"Added metadata_dictionaries/{filename} to ZIP")
            
            # Add mapping files in mapping_files/ subfolder
            for mapping_file in mapping_files_to_include:
                zipf.write(mapping_file['filepath'], f"mapping_files/{mapping_file['filename']}")
                logging.info(f"Added mapping_files/{mapping_file['filename']} to ZIP")
        
        # Return the ZIP file for download
        return FileResponse(
            path=temp_zip_path,
            filename="dcr_config_package.zip",
            media_type="application/zip",
            background=None  # File will be cleaned up by FastAPI after sending
        )
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        logging.error(f"Error creating ZIP file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP file: {str(e)}")


@router.get("/dcr-log/{dcr_id}", 
            name = "display the log file of the specified DCR")
def get_dcr_log(dcr_id: str,  user: Any = Depends(get_current_user)):
    print("now in get-dcr-log function")
    #id = "d2b060860906f94bce726a6cba3d948e236386359956c47cdc2dc477bbe199ee"
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    dcr = client.retrieve_analytics_dcr(dcr_id)
    log = dcr.retrieve_audit_log()
    events = [x for x in log.split("\n") if x.strip() != ""]
    #print(events)
    events_j = [{"timestamp": datetime.fromtimestamp(int(x.split(",")[0])/1000),
                  "user": x.split(",")[1], 
                  "desc": " - ".join(x.split(",")[2:])} for x in events]
    #return log.replace('\\n', '\n\n\n')
    #formatted_log = pformat(events_j, indent=2, width=80)
    #return formatted_log
    return events_j



@router.get("/dcr-log-main/{dcr_id}", 
            name = "display the main events in the log file of the specified DCR (excludes log fetching events)")
def get_dcr_log_main(dcr_id: str,  user: Any = Depends(get_current_user)):
    all_events = get_dcr_log(dcr_id, user)
    main_events = [e for e in all_events if e['desc'].find("log has been retrieved") == -1]
    return main_events


@router.get("/compute-get-output/{dcr_id}", 
            name = "run the scripts for a given DCR and download the output")
def run_computation_get_output(dcr_id: str,  user: Any = Depends(get_current_user)):
    """Run the scripts for a given DCR and download the output. Admins only."""
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    #example id = "9e2715f4b32a646d2da3d8952b7fa7ca48537ee6731627417f735d15fa17d4f6"
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    dcr = client.retrieve_analytics_dcr(dcr_id)
    cohort_id = dcr.node_definitions[0].name.strip()
    
    #SINCE C3 depends on c1 and c2, they will run automatically in the background!
    #c1_node = dcr.get_node("c1_data_dict_check") 
    #c1_node.run_computation()
    #c2_node = dcr.get_node("c2_save_to_json") 
    c3_node = dcr.get_node("c3_eda_data_profiling")
    result = c3_node.run_computation_and_get_results_as_zip()
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_dir = f"/data/dcr_output_{cohort_id}"

    data_to_log = {"DCR_id": dcr_id, 
                   "user": settings.decentriq_email,
                   "cohort_id": cohort_id,  "datetime": datetime.now(),
                   "storage_directory": os.path.abspath(storage_dir)}
    print("Compute evet: ", data_to_log)

    compute_events_log = "/data/dcr_computations.csv"
    if not os.path.exists(compute_events_log):
        print(f"Note: The file '{compute_events_log}' does not exist.")
    else:
        # Append to the existing file
        with open(compute_events_log, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_to_log.keys())
            writer.writerow(data_to_log)
            

    #print("Full path of storage directory: ", storage_dir.resolve())
    os.makedirs(storage_dir, mode=0o777, exist_ok=True)
    result.extractall(str(storage_dir))
    os.sync()
        
    return {"status": "success", "saved_path": str(storage_dir)}


@router.get("/shuffle-get-output/{dcr_id}",
            name = "run the C4 shuffle script for a given DCR and download the output")
def run_shuffle_get_output(dcr_id: str, user: Any = Depends(get_current_user)):
    """Run the C4 shuffle_data script and save output to the same folder as C3 (EDA) output.
    
    Accessible to all authenticated users.
    """
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    dcr = client.retrieve_analytics_dcr(dcr_id)
    cohort_id = dcr.node_definitions[0].name.strip()
    
    # Run the shuffle_data node (C4)
    shuffle_node = dcr.get_node("shuffle_data")
    result = shuffle_node.run_computation_and_get_results_as_zip()
    
    # Save to the same directory as C3 output
    storage_dir = f"/data/dcr_output_{cohort_id}"
    
    data_to_log = {
        "DCR_id": dcr_id,
        "script": "shuffle_data",
        "user": settings.decentriq_email,
        "cohort_id": cohort_id,
        "datetime": datetime.now(),
        "storage_directory": os.path.abspath(storage_dir)
    }
    logging.info(f"Shuffle computation event: {data_to_log}")
    
    # Log to the same CSV file
    compute_events_log = "/data/dcr_computations.csv"
    if os.path.exists(compute_events_log):
        with open(compute_events_log, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_to_log.keys())
            writer.writerow(data_to_log)
    else:
        # Create file with header if it doesn't exist
        with open(compute_events_log, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_to_log.keys())
            writer.writeheader()
            writer.writerow(data_to_log)
    
    # Extract results to the same folder as C3 output
    os.makedirs(storage_dir, mode=0o777, exist_ok=True)
    result.extractall(str(storage_dir))
    os.sync()
    
    return {
        "status": "success",
        "saved_path": str(storage_dir),
        "script": "shuffle_data",
        "files": ["shuffled_sample.csv", "shuffle_summary.txt"]
    }


def build_dcr_description(
    research_question: str | None,
    cohort_ids: list[str],
    creator_email: str,
) -> str:
    """Build a uniform DCR description string used for all compute DCRs.

    Decentriq's UI renders DCR descriptions but it is not clear whether it
    treats them as plain text, Markdown, or HTML. To find out, this function
    deliberately emits three different line-break styles between the
    description sections -- a single real newline, a literal backslash-n
    sequence, and an HTML <br> tag. Whichever one actually shows up as a line
    break in the DCR page tells us the correct format to keep; once confirmed,
    the other two variants can be removed.
    """
    rq_text = research_question.strip() if research_question and research_question.strip() else "no research question specified"
    cohorts_text = ", ".join(cohort_ids) if cohort_ids else "none"
    return (
        "This Data Clean Room was created via iCARE4CVD Cohort Explorer to "
        f"investigate the following research question: {rq_text}."
        "\n"  # variant 1: real newline
        f"Cohorts involved: {cohorts_text}"
        "\\n"  # variant 2: literal backslash-n
        f"Created by: {creator_email}"
        "<br>"  # variant 3: HTML line break
        "(testing line-break rendering)"
    )


def build_dcr_participants(
    cohorts_request: dict[str, Any],
    user_email: str,
    all_cohorts: dict[str, Any],
    additional_analysts: list[str] = None,
    excluded_data_owners: list[str] = None
) -> dict[str, dict[str, set]]:
    """Build the participants dictionary for a DCR.
    
    This function encapsulates the logic for determining who should be a data owner
    and/or analyst in the DCR. It can be called before DCR creation to preview
    participants, or during DCR creation to configure them.
    
    Args:
        cohorts_request: Dictionary with cohort IDs as keys
        user_email: Email of the user creating the DCR
        all_cohorts: Dictionary of all available cohorts with metadata
        additional_analysts: Optional list of additional analyst emails to add
        excluded_data_owners: Optional list of data owner emails to exclude from the DCR
        
    Returns:
        Dictionary mapping email addresses to their roles:
        {
            "email@example.com": {
                "data_owner_of": set([node_ids]),
                "analyst_of": set([node_ids])
            }
        }
    """
    excluded_data_owners = excluded_data_owners or []
    participants = {}
    participants[user_email] = {"data_owner_of": set(), "analyst_of": set()}
    
    # Always add service account as analyst (will get analyst_of roles later)
    if settings.decentriq_email and settings.decentriq_email != user_email:
        participants[settings.decentriq_email] = {"data_owner_of": set(), "analyst_of": set()}
    
    # Process each cohort to determine data owners
    for cohort_id in cohorts_request.get('cohorts', {}).keys():
        if cohort_id not in all_cohorts:
            continue
            
        cohort = all_cohorts[cohort_id]
        data_node_id = cohort_id.replace(" ", "-")
        metadata_node_id = f"{cohort_id.replace(' ', '-')}_metadata_dictionary"
        
        # Add data owners (in non-dev mode)
        if not settings.dev_mode:
            # Add all cohort_email owners (unless excluded)
            for owner in cohort.cohort_email:
                if owner in excluded_data_owners:
                    continue
                if owner not in participants:
                    participants[owner] = {"data_owner_of": set(), "analyst_of": set()}
                participants[owner]["data_owner_of"].add(data_node_id)
                participants[owner]["data_owner_of"].add(metadata_node_id)
            
            # Also add administrator_email if it exists (unless excluded)
            if cohort.administrator_email and cohort.administrator_email not in excluded_data_owners:
                if cohort.administrator_email not in participants:
                    participants[cohort.administrator_email] = {"data_owner_of": set(), "analyst_of": set()}
                participants[cohort.administrator_email]["data_owner_of"].add(data_node_id)
                participants[cohort.administrator_email]["data_owner_of"].add(metadata_node_id)
        else:
            # In dev_mode the requester is added as data owner
            participants[user_email]["data_owner_of"].add(data_node_id)
            participants[user_email]["data_owner_of"].add(metadata_node_id)
    
    # Add additional analysts if provided
    # They get the same privileges as the requester (data_owner_of and analyst_of)
    if additional_analysts:
        for analyst_email in additional_analysts:
            if analyst_email and analyst_email != user_email:
                if analyst_email not in participants:
                    participants[analyst_email] = {"data_owner_of": set(), "analyst_of": set()}
                # Give them the same data_owner_of permissions as the requester
                participants[analyst_email]["data_owner_of"].update(participants[user_email]["data_owner_of"])
    
    return participants


@router.post(
    "/preview-dcr-participants",
    name="Preview DCR participants",
    response_description="Participants that would be configured for the DCR",
)
async def api_preview_dcr_participants(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Preview what participants would be configured for a DCR without creating it.
    
    Accessible to all authenticated users.
    """
    try:
        # Get cohort metadata
        from src.cohort_cache import get_cohorts_from_cache
        all_cohorts = get_cohorts_from_cache(user["email"])
        
        # Build participants using the shared function
        additional_analysts = cohorts_request.get('additional_analysts', [])
        excluded_data_owners = cohorts_request.get('excluded_data_owners', [])
        participants = build_dcr_participants(
            cohorts_request,
            user["email"],
            all_cohorts,
            additional_analysts,
            excluded_data_owners
        )
        
        # Convert sets to lists for JSON serialization
        participants_json = {}
        for email, roles in participants.items():
            participants_json[email] = {
                "data_owner_of": list(roles["data_owner_of"]),
                "analyst_of": list(roles["analyst_of"]),
                "is_current_user": email == user["email"]
            }
        
        return {"participants": participants_json}
        
    except Exception as e:
        logging.error(f"Failed to preview DCR participants: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to preview DCR participants: {str(e)}"
        )


# Set of event names the frontend is allowed to emit via /dcr-wizard-event.
# Anything outside this set is dropped so stray/malformed traffic cannot
# pollute the activity log.
_ALLOWED_FRONTEND_EVENTS = {
    "wizard_opened",
    "wizard_step_advanced",
    "wizard_step_left",
    "wizard_closed",
    "wizard_abandoned",
    "dcr_download_config_clicked",
}


@router.post(
    "/dcr-wizard-event",
    name="Log a DCR wizard UI event",
    response_description="Acknowledgement",
)
async def api_dcr_wizard_event(
    event_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> dict[str, str]:
    """Append a frontend-reported wizard event to the activity log.

    Expected payload:
        {
          "event": "<one of _ALLOWED_FRONTEND_EVENTS>",
          "session_id": "<uuid>",
          "step": <int|optional>,
          "step_name": "<str|optional>",
          "time_on_step_seconds": <number|optional>,
          "details": { ...optional extra fields... }
        }

    The endpoint always returns ``{"status": "ok"}``; logging failures are
    swallowed so the frontend's fire-and-forget calls never error out the UI.
    """
    event_name = event_request.get("event")
    if event_name not in _ALLOWED_FRONTEND_EVENTS:
        raise HTTPException(status_code=400, detail=f"Unknown event: {event_name!r}")

    session_id = event_request.get("session_id")
    step = event_request.get("step")
    step_name = event_request.get("step_name")
    time_on_step_seconds = event_request.get("time_on_step_seconds")
    details = event_request.get("details") or {}

    log_dcr_event(
        event_name,
        user_email=user.get("email"),
        session_id=session_id,
        step=step,
        step_name=step_name,
        time_on_step_seconds=time_on_step_seconds,
        **{k: v for k, v in details.items() if k not in {"event", "session_id", "step", "step_name", "time_on_step_seconds"}},
    )
    return {"status": "ok"}


@router.get(
    "/dcr-events",
    name="List DCR activity events",
    response_description="All logged DCR events, newest first",
)
async def api_list_dcr_events(
    limit: int | None = None,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the full DCR activity history (newest first).

    Admin-only. Restricts access to users listed in ``settings.admins_list``.
    """
    user_email = user.get("email") if isinstance(user, dict) else None
    admins = getattr(settings, "admins_list", []) or []
    if not user_email or user_email.strip().lower() not in admins:
        raise HTTPException(status_code=403, detail="Admins only")

    events = read_events(limit=limit)
    return {"events": events, "count": len(events)}


@router.get(
    "/dcr-events/successful",
    name="List successfully created DCRs",
    response_description="Events belonging to sessions that successfully created a DCR, newest first",
)
async def api_list_successful_dcr_events(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Return only the events from sessions that resulted in a live DCR.

    Available to any authenticated user. Sessions are identified by
    ``session_id``: any session containing a ``dcr_publish_succeeded`` event
    is considered successful and all of its events are returned. Sessions
    that were only previewed / downloaded / abandoned / failed are omitted.
    """
    all_events = read_events()
    successful_sessions: set[str] = {
        evt.get("session_id")
        for evt in all_events
        if evt.get("event") == "dcr_publish_succeeded" and evt.get("session_id")
    }
    filtered = [
        evt for evt in all_events
        if evt.get("session_id") in successful_sessions
    ]
    return {"events": filtered, "count": len(filtered)}


def _require_admin(user: Any) -> None:
    """Raise 403 unless the authenticated user is in ``settings.admins_list``."""
    user_email = user.get("email") if isinstance(user, dict) else None
    admins = getattr(settings, "admins_list", []) or []
    if not user_email or user_email.strip().lower() not in admins:
        raise HTTPException(status_code=403, detail="Admins only")


def _decentriq_history_dir() -> str:
    """Return (and create) the directory used for Decentriq history exports."""
    history_dir = os.path.join(settings.data_folder, "logs")
    os.makedirs(history_dir, exist_ok=True)
    return history_dir


def _dcr_history_path() -> str:
    """Return the path of the DCR history JSONL file."""
    return os.path.join(_decentriq_history_dir(), "decentriq_dcrs_history.jsonl")


# In-memory DCR history state. Populated by ``refresh_all_dcrs_via_decentriq_api``
# at startup and lazily on first access if the JSONL already exists on disk.
# ``_dcr_history_mtime_ns`` tracks the JSONL file's mtime at the moment the
# in-memory state was last synced, so each accessor can detect whether another
# worker rewrote the file and trigger a cheap reload.
_dcr_history_lock = threading.RLock()
_dcr_history_records: list[dict[str, Any]] = []
_dcr_history_by_participant: dict[str, list[dict[str, Any]]] = {}
_dcr_history_loaded: bool = False
_dcr_history_mtime_ns: int | None = None


def _extract_participants(dcr: Any) -> list[dict[str, Any]]:
    """Pull detailed participant info from ``dcr.high_level``.

    Returns a list of ``{"email", "roles", "data_owner_of", "analyst_of"}``
    dicts. Returns ``[]`` (and never raises) if the structure is missing /
    different than expected.
    """
    try:
        config = getattr(dcr, "high_level", None)
        if not isinstance(config, dict) or not config:
            return []
        # Top-level key is the DCR version (e.g. "v15"); take the first.
        version_key = next(iter(config))
        participants_data = (
            config.get(version_key, {})
            .get("interactive", {})
            .get("initialConfiguration", {})
            .get("participants", [])
            or []
        )
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for participant in participants_data:
        if not isinstance(participant, dict):
            continue
        email = participant.get("user")
        permissions = participant.get("permissions") or []
        roles: list[str] = []
        data_owner_of: list[str] = []
        analyst_of: list[str] = []
        for perm in permissions:
            if not isinstance(perm, dict):
                continue
            if "manager" in perm:
                roles.append("Owner")
            if "analyst" in perm:
                node_id = (perm.get("analyst") or {}).get("nodeId")
                if node_id:
                    analyst_of.append(node_id)
            if "dataOwner" in perm:
                node_id = (perm.get("dataOwner") or {}).get("nodeId")
                if node_id:
                    data_owner_of.append(node_id)
        out.append({
            "email": email,
            "roles": roles,
            "data_owner_of": data_owner_of,
            "analyst_of": analyst_of,
        })
    return out


def _build_participant_index(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return a mapping ``email -> [dcr_record, ...]`` covering every email
    referenced in either the DCR's ``owner`` or its ``participants`` list.
    """
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        emails: set[str] = set()
        owner = record.get("owner") or {}
        if isinstance(owner, dict):
            owner_email = owner.get("email")
            if owner_email:
                emails.add(owner_email)
        for p in record.get("participants") or []:
            email = (p or {}).get("email") if isinstance(p, dict) else None
            if email:
                emails.add(email)
        for email in emails:
            index.setdefault(email, []).append(record)
    return index


def _file_mtime_ns(path: str) -> int | None:
    """Return ``st_mtime_ns`` for ``path`` or ``None`` if it can't be read."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _set_dcr_history_state(records: list[dict[str, Any]], source_mtime_ns: int | None) -> None:
    """Atomically replace the in-memory DCR history state and record the mtime
    of the JSONL the state was synced from (so future accessors can detect a
    fresher file written by another worker).
    """
    global _dcr_history_records, _dcr_history_by_participant
    global _dcr_history_loaded, _dcr_history_mtime_ns
    index = _build_participant_index(records)
    with _dcr_history_lock:
        _dcr_history_records = records
        _dcr_history_by_participant = index
        _dcr_history_loaded = True
        _dcr_history_mtime_ns = source_mtime_ns


def load_dcr_history_from_disk() -> int:
    """Populate the in-memory DCR history from the JSONL file on disk.

    Used as a lazy fallback so other request handlers don't have to wait for
    the background refresh task to finish before they can query the index.
    Returns the number of records loaded (0 if the file does not exist).
    """
    path = _dcr_history_path()
    if not os.path.isfile(path):
        return 0
    records: list[dict[str, Any]] = []
    try:
        # Snapshot the mtime *before* reading so a concurrent rewrite mid-read
        # is detected next time (rather than us claiming we have the latest).
        mtime_ns = _file_mtime_ns(path)
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logging.warning("Failed to read DCR history file %s: %s", path, exc)
        return 0
    _set_dcr_history_state(records, mtime_ns)
    logging.info("Loaded %d DCR records into memory from %s", len(records), path)
    return len(records)


def _reload_if_stale() -> None:
    """Reload the in-memory state if (a) it was never loaded, or (b) the
    JSONL on disk has a newer mtime than what we last synced from. Cheap:
    one ``os.stat`` per call. No-op if the file is missing.
    """
    path = _dcr_history_path()
    disk_mtime = _file_mtime_ns(path)
    if disk_mtime is None:
        return
    with _dcr_history_lock:
        loaded = _dcr_history_loaded
        last_mtime = _dcr_history_mtime_ns
    if not loaded or last_mtime is None or disk_mtime != last_mtime:
        load_dcr_history_from_disk()


def get_all_dcrs() -> list[dict[str, Any]]:
    """Return a snapshot of the full in-memory DCR history list.

    Reloads from disk transparently if another worker has rewritten the JSONL
    (detected via mtime) or the index hasn't been populated yet in this worker.
    """
    _reload_if_stale()
    with _dcr_history_lock:
        return list(_dcr_history_records)


def get_dcrs_for_participant(email: str) -> list[dict[str, Any]]:
    """Return DCR records the given email participates in (owner or participant)."""
    if not email:
        return []
    target = email.strip().lower()
    _reload_if_stale()
    with _dcr_history_lock:
        index = _dcr_history_by_participant
        # Case-insensitive match on email; copy into a list while holding the
        # lock so we don't expose the live dict to the caller.
        return [
            record
            for stored_email, records in index.items()
            if stored_email and stored_email.strip().lower() == target
            for record in records
        ]


def refresh_all_dcrs_via_decentriq_api() -> dict[str, Any]:
    """Fetch every DCR the cohort-explorer service account is a member of,
    enrich each with its node titles/types and detailed participant list,
    persist them to a single JSONL file at
    ``<data_folder>/logs/decentriq_dcrs_history.jsonl``, and rebuild the
    in-memory ``email -> [dcr_records]`` index used by ``get_dcrs_for_participant``.

    Only fetches detailed info for new DCRs (not already in the JSONL file).
    Appends new DCRs to the JSONL file instead of rewriting it.

    Returns a summary dict (count, earliest / latest ``createdAt``, processed /
    failures / total_nodes / total_participants, output path). Synchronous;
    safe to run inside ``asyncio.to_thread`` from a non-blocking startup task.
    """
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    logging.info("Fetching all DCR descriptions from Decentriq (incremental startup refresh)...")
    descriptions = client.get_data_room_descriptions()

    output_path = _dcr_history_path()

    # Load existing DCRs from JSONL file
    existing_dcrs_by_id: dict[str, dict[str, Any]] = {}
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                    dcr_id = record.get("id")
                    if dcr_id:
                        existing_dcrs_by_id[dcr_id] = record
                except json.JSONDecodeError:
                    continue

    logging.info("Found %d existing DCRs in JSONL file", len(existing_dcrs_by_id))

    earliest: str | None = None
    latest: str | None = None
    count = 0
    processed = 0
    failures = 0
    total_nodes = 0
    total_participants = 0
    new_dcrs_added = 0
    new_records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    for desc in descriptions:
        count += 1
        record: dict[str, Any] = dict(desc) if isinstance(desc, dict) else {"raw": desc}

        created_at = record.get("createdAt")
        if created_at:
            if earliest is None or created_at < earliest:
                earliest = created_at
            if latest is None or created_at > latest:
                latest = created_at

        dcr_id = record.get("id")
        is_new = dcr_id and dcr_id not in existing_dcrs_by_id

        record["nodes"] = []
        record["participants"] = []
        record["cohorts"] = []

        # Only fetch detailed info for new DCRs
        if is_new and dcr_id:
            try:
                dcr = client.retrieve_analytics_dcr(dcr_id=dcr_id)
                for node_def in getattr(dcr, "node_definitions", []) or []:
                    node_name = getattr(node_def, "name", None)
                    node_info = {
                        "name": node_name,
                        "type": type(node_def).__name__,
                    }
                    # Capture script name for compute nodes
                    if type(node_def).__name__ in ("PreviewComputeNodeDefinition", "PythonComputeNodeDefinition"):
                        script = getattr(node_def, "script", None)
                        if script:
                            node_info["script"] = str(script).split("\n")[0] if isinstance(script, str) else str(script)
                    record["nodes"].append(node_info)
                # Extract cohort names from metadata nodes (nodes with "metadata" in the name)
                # The cohort name is at the start of the node name before "metadata"
                cohort_names = set()
                for node in record["nodes"]:
                    if node["name"] and "metadata" in node["name"].lower():
                        # Extract cohort name from node name (e.g., "cohort1-metadata" -> "cohort1")
                        # Handle both "-metadata" and "_metadata" suffixes
                        name_lower = node["name"].lower()
                        for suffix in ["-metadata", "_metadata", "-metadata_dictionary", "_metadata_dictionary"]:
                            if name_lower.endswith(suffix):
                                cohort_name = node["name"][:-len(suffix)]
                                if cohort_name:
                                    cohort_names.add(cohort_name)
                                break
                record["cohorts"] = sorted(list(cohort_names))
                total_nodes += len(record["nodes"])
                record["participants"] = _extract_participants(dcr)
                total_participants += len(record["participants"])
                processed += 1
                new_dcrs_added += 1
                new_records.append(record)
            except Exception as exc:
                # Non-analytics DCRs (e.g. MEDIA) or transient errors land here.
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
                new_records.append(record)
        elif dcr_id:
            # For existing DCRs, use the record from the JSONL file
            record = existing_dcrs_by_id[dcr_id]
        else:
            record["error"] = "missing_dcr_id"
            failures += 1

        all_records.append(record)

        if count % 25 == 0:
            logging.info(
                "DCR refresh progress (startup): count=%d processed=%d failures=%d new=%d",
                count, processed, failures, new_dcrs_added,
            )

    # Append new DCRs to JSONL file if any were found
    if new_records:
        logging.info("Appending %d new DCRs to JSONL file", len(new_records))
        with open(output_path, "a", encoding="utf-8") as fh:
            for record in new_records:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    # Update in-memory state with all records
    _set_dcr_history_state(all_records, _file_mtime_ns(output_path) if os.path.exists(output_path) else None)

    logging.info(
        "DCR refresh done (startup): %d total records (processed=%d, failures=%d, "
        "new_dcrs_added=%d, total_nodes=%d, total_participants=%d, indexed_emails=%d, earliest=%s, "
        "latest=%s)",
        count, processed, failures, new_dcrs_added, total_nodes,
        total_participants, len(_dcr_history_by_participant), earliest, latest,
    )
    return {
        "count": count,
        "processed": processed,
        "failures": failures,
        "total_nodes": total_nodes,
        "total_participants": total_participants,
        "indexed_emails": len(_dcr_history_by_participant),
        "new_dcrs_added": new_dcrs_added,
        "earliest_created_at": earliest,
        "latest_created_at": latest,
        "output_path": output_path,
    }


def refresh_dcrs_in_memory_only() -> dict[str, Any]:
    """Fetch every DCR from the Decentriq API and update in-memory state.
    Only fetches detailed info for new DCRs (not already in memory).
    Appends new DCRs to the JSONL file if any are found.

    Returns a summary dict (count, earliest / latest ``createdAt``, processed /
    failures / total_nodes / total_participants, new_dcrs_added).
    """
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    logging.info("Fetching all DCR descriptions from Decentriq (incremental refresh)...")
    descriptions = client.get_data_room_descriptions()

    # Get existing DCR IDs from in-memory state
    existing_dcr_ids = set()
    with _dcr_history_lock:
        existing_dcr_ids = {record.get("id") for record in _dcr_history_records if record.get("id")}

    earliest: str | None = None
    latest: str | None = None
    count = 0
    processed = 0
    failures = 0
    total_nodes = 0
    total_participants = 0
    new_dcrs_added = 0
    new_records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    for desc in descriptions:
        count += 1
        record: dict[str, Any] = dict(desc) if isinstance(desc, dict) else {"raw": desc}

        created_at = record.get("createdAt")
        if created_at:
            if earliest is None or created_at < earliest:
                earliest = created_at
            if latest is None or created_at > latest:
                latest = created_at

        dcr_id = record.get("id")
        is_new = dcr_id and dcr_id not in existing_dcr_ids

        record["nodes"] = []
        record["participants"] = []
        record["cohorts"] = []

        # Only fetch detailed info for new DCRs
        if is_new and dcr_id:
            try:
                dcr = client.retrieve_analytics_dcr(dcr_id=dcr_id)
                for node_def in getattr(dcr, "node_definitions", []) or []:
                    node_name = getattr(node_def, "name", None)
                    node_info = {
                        "name": node_name,
                        "type": type(node_def).__name__,
                    }
                    # Capture script name for compute nodes
                    if type(node_def).__name__ in ("PreviewComputeNodeDefinition", "PythonComputeNodeDefinition"):
                        script = getattr(node_def, "script", None)
                        if script:
                            node_info["script"] = str(script).split("\n")[0] if isinstance(script, str) else str(script)
                    record["nodes"].append(node_info)
                # Extract cohort names from metadata nodes
                cohort_names = set()
                for node in record["nodes"]:
                    if node["name"] and "metadata" in node["name"].lower():
                        name_lower = node["name"].lower()
                        for suffix in ["-metadata", "_metadata", "-metadata_dictionary", "_metadata_dictionary"]:
                            if name_lower.endswith(suffix):
                                cohort_name = node["name"][:-len(suffix)]
                                if cohort_name:
                                    cohort_names.add(cohort_name)
                                break
                record["cohorts"] = sorted(list(cohort_names))
                total_nodes += len(record["nodes"])
                record["participants"] = _extract_participants(dcr)
                total_participants += len(record["participants"])
                processed += 1
                new_dcrs_added += 1
                new_records.append(record)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
                new_records.append(record)
        elif dcr_id:
            # For existing DCRs, use the in-memory record
            with _dcr_history_lock:
                existing_record = next((r for r in _dcr_history_records if r.get("id") == dcr_id), None)
                if existing_record:
                    record = existing_record
        else:
            record["error"] = "missing_dcr_id"
            failures += 1

        all_records.append(record)

        if count % 25 == 0:
            logging.info(
                "DCR refresh progress (incremental): count=%d processed=%d failures=%d new=%d",
                count, processed, failures, new_dcrs_added,
            )

    # Append new DCRs to JSONL file if any were found
    output_path = _dcr_history_path()
    if new_records:
        logging.info("Appending %d new DCRs to JSONL file", len(new_records))
        with open(output_path, "a", encoding="utf-8") as fh:
            for record in new_records:
                fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    # Update in-memory state with all records
    _set_dcr_history_state(all_records, _file_mtime_ns(output_path) if os.path.exists(output_path) else None)

    logging.info(
        "DCR refresh done (incremental): %d total records (processed=%d, failures=%d, "
        "new_dcrs_added=%d, total_nodes=%d, total_participants=%d, indexed_emails=%d, earliest=%s, "
        "latest=%s)",
        count, processed, failures, new_dcrs_added, total_nodes,
        total_participants, len(_dcr_history_by_participant), earliest, latest,
    )
    return {
        "count": count,
        "processed": processed,
        "failures": failures,
        "total_nodes": total_nodes,
        "total_participants": total_participants,
        "indexed_emails": len(_dcr_history_by_participant),
        "new_dcrs_added": new_dcrs_added,
        "earliest_created_at": earliest,
        "latest_created_at": latest,
    }


@router.post(
    "/refresh-all-dcrs-via-decentriq-api",
    name="Refresh the local DCR history JSONL via the Decentriq API",
    response_description="Summary stats and path to the JSONL file",
)
async def api_refresh_all_dcrs_via_decentriq_api(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Admin-only. Fetch all DCRs from the Decentriq account, enrich each with
    its node titles + types and detailed participants, persist the result to
    one JSONL file, and rebuild the in-memory participant index.
    """
    _require_admin(user)
    # Run the (blocking) SDK calls off the event loop.
    import asyncio
    return await asyncio.to_thread(refresh_all_dcrs_via_decentriq_api)


@router.get(
    "/my-dcrs",
    name="List DCRs for the current user",
    response_description="DCR records where the current user is a participant",
)
async def api_my_dcrs(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Return all DCR records (from the Decentriq API history) where the
    authenticated user appears as owner or participant.
    """
    user_email = user.get("email") if isinstance(user, dict) else None
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    records = get_dcrs_for_participant(user_email)
    return {"dcrs": records, "count": len(records), "email": user_email}


@router.post(
    "/my-dcrs/refresh",
    name="Refresh DCR history and return current user's DCRs",
    response_description="Summary of the refresh plus the user's DCR records",
)
async def api_refresh_my_dcrs(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Trigger a refresh of the DCR history from the Decentriq API (in-memory only),
    then return the records relevant to the authenticated user.
    """
    user_email = user.get("email") if isinstance(user, dict) else None
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    import asyncio
    summary = await asyncio.to_thread(refresh_dcrs_in_memory_only)
    records = get_dcrs_for_participant(user_email)
    return {"dcrs": records, "count": len(records), "email": user_email, "refresh_summary": summary}


@router.get(
    "/my-dcrs/last-modified",
    name="Get last modified timestamp of DCR history",
    response_description="ISO timestamp of when the DCR history JSONL file was last modified",
)
async def api_dcr_history_last_modified(
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the last modified timestamp of the DCR history JSONL file."""
    user_email = user.get("email") if isinstance(user, dict) else None
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    history_path = _dcr_history_path()
    try:
        mtime_ns = os.path.getmtime(history_path)
        mtime = datetime.fromtimestamp(mtime_ns)
        return {"last_modified": mtime.isoformat()}
    except FileNotFoundError:
        return {"last_modified": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get last modified: {exc}")


@router.post("/check-shuffled-samples")
async def check_shuffled_samples(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Check which cohorts have shuffled sample files available.
    
    Accessible to all authenticated users.
    """
    try:
        cohorts_with_samples = []
        cohorts_without_samples = []
        
        for cohort_id in cohorts_request.get("cohorts", {}).keys():
            storage_dir = os.path.join(settings.data_folder, f"dcr_output_{cohort_id}")
            shuffled_csv = os.path.join(storage_dir, "shuffled_sample.csv")
            
            if os.path.exists(shuffled_csv):
                cohorts_with_samples.append(cohort_id)
            else:
                cohorts_without_samples.append(cohort_id)
        
        return {
            "cohorts_with_samples": cohorts_with_samples,
            "cohorts_without_samples": cohorts_without_samples
        }
        
    except Exception as e:
        logging.error(f"Failed to check shuffled samples: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check shuffled samples: {str(e)}"
        )


def update_provision_log(new_dcr):
    try:
        with open("/data/provisions_log.jsonl") as f:
            log = json.load(f)
            log.append(new_dcr)
    except:
        log = [new_dcr]
    with open("/data/provisions_log.jsonl", "w") as f:
        json.dump(log, f)
