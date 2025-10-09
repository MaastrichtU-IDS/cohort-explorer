import csv
from copy import deepcopy
from typing import Any
import os, json
import logging # Add logging import

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

from src.auth import get_current_user
from src.config import settings
from src.eda_scripts import c1_data_dict_check, c2_save_to_json, c3_eda_data_profiling, shuffle_data
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata
from datetime import datetime

router = APIRouter()


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


def find_variable_by_omop_id(cohort_id: str, omop_id: str) -> str | None:
    """Find a variable in a cohort by its OMOP ID using the cache.
    
    Args:
        cohort_id: The ID of the cohort to search in
        omop_id: The OMOP ID to search for (e.g., "4086934", the patient ID variable)
        
    Returns:
        The variable name if found, None otherwise
    """
    import time
    start_time = time.time()
    
    try:
        # Use cache-based approach which is more reliable
        from src.cohort_cache import get_cohorts_from_cache
        
        # Get cohorts from cache (use admin email for full access)
        from src.config import settings
        admin_email = settings.admins_list[0] if settings.admins_list else None
        cached_cohorts = get_cohorts_from_cache(admin_email)
        cache_time = time.time()
        
        # Check if the cohort exists in cache
        if cohort_id not in cached_cohorts:
            elapsed = time.time() - start_time
            logging.warning(f"Cohort {cohort_id} not found in cache (took {elapsed:.3f}s)")
            return None
            
        cohort = cached_cohorts[cohort_id]
        
        # Search through all variables in the cohort
        if hasattr(cohort, 'variables') and cohort.variables:
            for var_name, variable in cohort.variables.items():
                if hasattr(variable, 'omop_id') and variable.omop_id == omop_id:
                    elapsed = time.time() - start_time
                    cache_elapsed = cache_time - start_time
                    search_elapsed = elapsed - cache_elapsed
                    logging.info(f"Found variable '{var_name}' with OMOP ID {omop_id} in cohort {cohort_id} (total: {elapsed:.3f}s, cache: {cache_elapsed:.3f}s, search: {search_elapsed:.3f}s)")
                    return var_name
        
        elapsed = time.time() - start_time
        logging.info(f"No variable with OMOP ID {omop_id} found in cohort {cohort_id} (took {elapsed:.3f}s)")
        return None
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Error finding variable with OMOP ID {omop_id} in cohort {cohort_id} (took {elapsed:.3f}s): {e}")
        return None


async def get_compute_dcr_definition(
    cohorts_request: dict[str, Any],
    user: Any,
    client: Any,
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
    dcr_count = len(client.get_data_room_descriptions())
    dcr_title = f"iCARE4CVD DCR compute {dcr_count}"
    builder = (
        AnalyticsDcrBuilder(client=client)
        .with_name(dcr_title)
        # .with_owner(settings.decentriq_email)
        .with_owner(user["email"])
        .with_description("A data clean room to run analyses on cohorts for the iCARE4CVD project")
    )
    logging.info(f"DCR builder initialized for {len(cohorts_request['cohorts'])} cohorts")

    participants = {}
    participants[user["email"]] = {"data_owner_of": set(), "analyst_of": set()}
    preview_nodes = []
    # Convert cohort variables to decentriq schema
    for cohort_id, cohort in selected_cohorts.items():
        # Create data node for cohort
        data_node_id = cohort_id.replace(" ", "-")
        builder.add_node_definition(
            TableDataNodeDefinition(name=data_node_id, columns=get_cohort_schema(cohort), is_required=True)
        )
        data_nodes.append(data_node_id)

        # Add a node for the cohort's metadata dictionary
        metadata_node_id = f"{cohort_id.replace(' ', '-')}_metadata_dictionary"
        
        # Use the helper function to identify the appropriate metadata schema for this cohort
        metadata_cols = identify_cohort_meta_schema(cohort)
        
        builder.add_node_definition(
            TableDataNodeDefinition(name=metadata_node_id, columns=metadata_cols, is_required=False)
        )

        # Add data owners to provision the data (in dev we dont add them to avoid unnecessary emails)
        if not settings.dev_mode:
            for owner in cohort.cohort_email:
                if owner not in participants:
                    participants[owner] = {"data_owner_of": set(), "analyst_of": set()}
                participants[owner]["data_owner_of"].add(data_node_id)
                participants[owner]["data_owner_of"].add(metadata_node_id)
            #participants[user["email"]]["analyst_of"].add(metadata_node_id)
        else:
            # In dev_mode the requester is added as data owner instead
            participants[user["email"]]["data_owner_of"].add(data_node_id)
            participants[user["email"]]["data_owner_of"].add(metadata_node_id)
        

        # Add pandas preparation script
        pandas_script = "import pandas as pd\nimport decentriq_util\n\n"
        df_var = f"df_{cohort_id.replace('-', '_')}"
        requested_vars = cohorts_request["cohorts"][cohort_id].copy() if isinstance(cohorts_request["cohorts"][cohort_id], list) else cohorts_request["cohorts"][cohort_id]

        cohort_id_var = find_variable_by_omop_id(cohort_id, "4086934")

        if cohort_id_var is None:
            pandas_script += f"#No cohort ID variable (i.e. no variable with OMOP ID 4086934) found for cohort {cohort_id}\n"
            pandas_script += f"#No modifications will be done on the dataframe\n"
        
        elif cohort_id_var not in requested_vars:
            pandas_script += f"#Cohort ID variable ({cohort_id_var}) not in requested variables list for cohort {cohort_id}\n"
            pandas_script += f"#No modifications will be done on the dataframe\n"
        else:
            pandas_script += f"#Cohort ID variable ({cohort_id_var}) in among requested variables list for cohort {cohort_id}\n"
            pandas_script += f"#The Cohort ID variable will be dropped from the dataframe\n"
            requested_vars.remove(cohort_id_var)
        
        # Direct cohort variables list
        if isinstance(requested_vars, list):
            pandas_script += f'{df_var} = decentriq_util.read_tabular_data("/input/{cohort_id}")\n'
            if len(requested_vars) <= len(cohort.variables):
                # Add filter variables to pandas script
                pandas_script += f"{df_var} = {df_var}[{requested_vars}]\n"

        # TODO: Merge operation, need to be implemented on the frontend?
        elif isinstance(requested_vars, dict):
            pandas_script += pandas_script_merge_cohorts(requested_vars, all_cohorts)
            # TODO: add merged cohorts schema to selected_cohorts
        else:
            raise HTTPException(status_code=400, detail=f"Invalid structure for cohort {cohort_id}")
        pandas_script += f'\n#The following line commented out - Sept 2025\n'
        pandas_script += f'#{df_var}.to_csv("/output/{cohort_id}.csv", index=False, header=True)\n\n'

        # Add python data preparation script
        builder.add_node_definition(
            PythonComputeNodeDefinition(name=f"prepare-{cohort_id}", script=pandas_script, dependencies=[data_node_id])
        )
        # Add the requester as analyst of prepare script
        participants[user["email"]]["analyst_of"].add(f"prepare-{cohort_id}")



    builder.add_node_definition(
        RawDataNodeDefinition(name="CrossStudyMappings", is_required=False)
    )
    participants[user["email"]]["data_owner_of"].add("CrossStudyMappings")
    # Add users permissions for previews
    # for prev_node in preview_nodes:
    #     participants[user["email"]]["analyst_of"].add(prev_node)

    for p_email, p_perm in participants.items():
        builder.add_participant(
            p_email,
            data_owner_of=list(p_perm["data_owner_of"]),
            analyst_of=list(p_perm["analyst_of"]),
        )

    # Build and publish DCR
    build_start = datetime.now()
    dcr_definition = builder.build()
    build_time = datetime.now() - build_start
    
    total_time = datetime.now() - start_time
    logging.info(f"DCR build completed in {build_time.total_seconds():.3f}s")
    logging.info(f"Total DCR definition creation completed in {total_time.total_seconds():.3f}s for {len(cohorts_request['cohorts'])} cohorts")
    
    return dcr_definition, dcr_title



@router.post(
    "/get-compute-dcr-definition",
    name="Get the Data Clean Room definition for computing as JSON",
    response_description="Upload result",
)
async def api_get_compute_dcr_definition(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> Any:
    """Create a Data Clean Room for computing with the cohorts requested using Decentriq SDK"""
    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)

    dcr_definition, _dcr_title = await get_compute_dcr_definition(cohorts_request, user, client)

    # return dcr_definition.model_dump_json(by_alias=True)
    # return json.dumps(dcr_definition.high_level)
    return { "dataScienceDataRoom": dcr_definition.high_level }


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
    #example id = "9e2715f4b32a646d2da3d8952b7fa7ca48537ee6731627417f735d15fa17d4f6"
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    dcr = client.retrieve_analytics_dcr(dcr_id)
    cohort_id = dcr.node_definitions[0].name.strip()
    
    #SINCE C3 depends on c1 and c2, they will run automatically in the background!
    #c1_node = dcr.get_node("c1_data_dict_check") 
    #c1_node.run_computation()
    #c2_node = dcr.get_node("c2_save_to_json") 
    #c2_node.run_computation()
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


def update_provision_log(new_dcr):
    try:
        with open("/data/provisions_log.jsonl") as f:
            log = json.load(f)
            log.append(new_dcr)
    except:
        log = [new_dcr]
    with open("/data/provisions_log.jsonl", "w") as f:
        json.dump(log, f)
