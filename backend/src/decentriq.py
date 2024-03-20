from copy import deepcopy
from typing import Any

import decentriq_platform as dq
import decentriq_platform.sql as dqsql
from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata

router = APIRouter()


def get_cohort_schema(cohort_dict: Cohort) -> list[tuple[str, dqsql.PrimitiveType, bool]]:
    """Convert cohort variables to Decentriq schema"""
    schema = []
    for variable_id, variable_info in cohort_dict.variables.items():
        prim_type = dqsql.PrimitiveType.STRING
        if variable_info.var_type == "FLOAT":
            prim_type = dqsql.PrimitiveType.FLOAT64
        if variable_info.var_type == "INT":
            prim_type = dqsql.PrimitiveType.INT64
        nullable = bool(variable_info.na != 0)
        schema.append((variable_id, prim_type, nullable))
    return schema


# https://docs.decentriq.com/sdk/python-getting-started
def create_provision_dcr(user: Any, cohort: Cohort) -> dict[str, Any]:
    """Initialize a Data Clean Room in Decentriq when a new cohort is uploaded"""
    users = [user["email"]]

    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    enclave_specs = dq.enclave_specifications.versions(
        [
            "decentriq.driver:v20",
            "decentriq.sql-worker:v12",
        ]
    )
    auth, _ = client.create_auth_using_decentriq_pki(enclave_specs)
    session = client.create_session(auth, enclave_specs)

    # Creation of a Data Clean Room (DCR)
    builder = dq.DataRoomBuilder(f"iCare4CVD DCR provision {cohort.cohort_id}", enclave_specs=enclave_specs)

    # Create data node for cohort
    data_node_builder = dqsql.TabularDataNodeBuilder(cohort.cohort_id, schema=get_cohort_schema(cohort))
    data_node_builder.add_to_builder(builder, authentication=client.decentriq_pki_authentication, users=users)

    builder.add_user_permission(
        email=user["email"],
        authentication_method=client.decentriq_pki_authentication,
        permissions=[dq.Permissions.update_data_room_status()],  # To delete the DCR
    )

    # Build and publish DCR
    data_room = builder.build()
    data_room_id = session.publish_data_room(data_room)

    dcr_desc = client.get_data_room_description(data_room_id, enclave_specs)
    dcr_url = f"https://platform.decentriq.com/datarooms/p/{data_room_id}"
    return {
        "message": f"Data Clean Room for {cohort.cohort_id} provisioned at {dcr_url}",
        "identifier": cohort.cohort_id,
        "dcr_url": dcr_url,
        "dcr_title": dcr_desc["title"],
        "dcr": dcr_desc,
        **cohort.dict(),
    }


def pandas_script_filter_cohort_vars(cohort: Cohort, requested_vars: list[str], df_var: str) -> str:
    """Generate pandas script for filtering variables"""
    if len(requested_vars) <= len(cohort.variables):
        filter_columns_script = f"{df_var} = {df_var}[{requested_vars}]\n"
    return filter_columns_script


def pandas_script_merge_cohorts(merged_cohorts: dict[str, list[str]], all_cohorts: dict[str, Cohort]) -> str:
    """Generate pandas script for merging cohorts on variables mapped_id"""
    # TODO: to be fixed
    merge_script = ""
    dfs_to_merge = []
    for cohort_id, vars_requested in merged_cohorts.items():
        if cohort_id not in all_cohorts:
            raise ValueError(f"Cohort {cohort_id} does not exist.")
        # Assuming you have a way to get dataframe variable names (mapped_id) from vars_requested
        df_name = f"df_{cohort_id}"
        vars_mapped = [f"'{var}'" for var in vars_requested]  # Example to generate a list of variable names
        dfs_to_merge.append(df_name)
        merge_script += (
            f"{df_name} = pd.DataFrame({cohort_id})[{vars_mapped}]\n"  # Placeholder for actual data retrieval
        )

    # Assuming all dataframes have a common column for merging
    merge_script += f"merged_df = pd.concat([{', '.join(dfs_to_merge)}], ignore_index=True)\n"
    return merge_script


@router.post(
    "/create-dcr",
    name="Create Data Clean Room for computing",
    response_description="Upload result",
)
async def create_compute_dcr(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a Data Clean Room for computing with the cohorts requested using Decentriq SDK"""
    users = [user["email"]]
    # TODO: cohorts_request could also be a dict of union of cohorts to merge
    # {"cohorts": {"cohort_id": ["var1", "var2"], "merged_cohort3": {"cohort1": ["weight", "sex"], "cohort2": ["gender", "patient weight"]}}}
    # We automatically merge the cohorts, figuring out which variables are the same thanks to mappings
    all_cohorts = retrieve_cohorts_metadata(user["email"])

    # Get metadata for selected cohorts and variables
    selected_cohorts = {}
    # We generate a pandas script to automatically prepare the data from the cohort based on known metadata
    pandas_script = "import pandas as pd\n\n"

    for cohort_id, requested_vars in cohorts_request["cohorts"].items():
        cohort_meta = deepcopy(all_cohorts[cohort_id])
        df_var = f"df_{cohort_id.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}"
        if isinstance(requested_vars, list):
            # Direct cohort variables list
            pandas_script += f"{df_var} = pd.read_csv('{cohort_id}.csv')\n"

            if len(requested_vars) <= len(cohort_meta.variables):
                # Add filter variables to pandas script
                pandas_script += f"{df_var} = {df_var}[{requested_vars}]\n"
            # Get all cohort and variables metadata for selected variables
            for var in all_cohorts[cohort_id].variables:
                if var not in requested_vars:
                    del cohort_meta.variables[var]
            selected_cohorts[cohort_id] = cohort_meta
        elif isinstance(requested_vars, dict):
            # Merge operation, need to be implemented on the frontend
            pandas_script += pandas_script_merge_cohorts(requested_vars, all_cohorts)
            # TODO: add merged cohorts schema to selected_cohorts
        else:
            raise HTTPException(status_code=400, detail=f"Invalid structure for cohort {cohort_id}")

    # TODO: Add pandas_script to the DCR?
    # print(pandas_script)

    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
    enclave_specs = dq.enclave_specifications.versions(
        [
            "decentriq.driver:v20",
            "decentriq.sql-worker:v12",
            # "decentriq.python-ml-worker-32-64:v21",
            # "decentriq.r-latex-worker-32-32:v16",
        ]
    )
    auth, _ = client.create_auth_using_decentriq_pki(enclave_specs)
    session = client.create_session(auth, enclave_specs)

    # Creation of a Data Clean Room (DCR)
    dcr_count = len(client.get_data_room_descriptions())
    builder = dq.DataRoomBuilder(f"iCare4CVD DCR compute {dcr_count}", enclave_specs=enclave_specs)

    # Convert cohort variables to decentriq schema
    for cohort_id, cohort in selected_cohorts.items():
        # Create data node for cohort
        data_node_builder = dqsql.TabularDataNodeBuilder(cohort_id, schema=get_cohort_schema(cohort))
        data_node_builder.add_to_builder(builder, authentication=client.decentriq_pki_authentication, users=users)

    # Add empty list of permissions
    builder.add_user_permission(
        email=user["email"],
        authentication_method=client.decentriq_pki_authentication,
        permissions=[
            dq.Permissions.update_data_room_status(),  # To delete the DCR
            # dq.Permissions.leaf_crud(data_node_id),
            # dq.Permissions.execute_compute(uppercase_text_node_id),
            # dq.Permissions.retrieve_compute_result(uppercase_text_node_id),
        ],
    )

    # Build and publish DCR
    data_room = builder.build()
    data_room_id = session.publish_data_room(data_room)

    dcr_desc = client.get_data_room_description(data_room_id, enclave_specs)
    dcr_url = f"https://platform.decentriq.com/datarooms/p/{data_room_id}"
    return {
        "message": f"Data Clean Room available for compute at {dcr_url}",
        "dcr_url": dcr_url,
        "dcr_title": dcr_desc["title"],
        "dcr": dcr_desc,
        "merge_script": pandas_script,
        **cohorts_request,
    }
