from copy import deepcopy
import json
from typing import Any

import decentriq_platform as dq

# import decentriq_platform.sql as dqsql
from decentriq_platform.analytics import (
    AnalyticsDcrBuilder,
    Column,
    FormatType,
    PreviewComputeNodeDefinition,
    PythonComputeNodeDefinition,
    TableDataNodeDefinition,
)
from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.config import settings
from src.models import Cohort
from src.utils import retrieve_cohorts_metadata

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


# https://docs.decentriq.com/sdk/python-getting-started
def create_provision_dcr(user: Any, cohort: Cohort) -> dict[str, Any]:
    """Initialize a Data Clean Room in Decentriq when a new cohort is uploaded"""
    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)

    # Creation of a Data Clean Room (DCR)
    dcr_title = f"iCARE4CVD DCR provision {cohort.cohort_id}"
    builder = (
        AnalyticsDcrBuilder(client=client)
            .with_name(dcr_title)
            .with_owner(user["email"])
            # .with_owner(settings.decentriq_email)
            .with_description(f"A data clean room to provision the data for the {cohort.cohort_id} cohort")
    )

    # Create data node for Cohort Data
    data_node_id = cohort.cohort_id.replace(" ", "-")
    # builder.add_node_definition(RawDataNodeDefinition(name=data_node_id, is_required=True))
    # TODO: providing schema is broken in new SDK
    builder.add_node_definition(
        TableDataNodeDefinition(name=data_node_id, columns=get_cohort_schema(cohort), is_required=True)
    )

    builder.add_participant(
        user["email"],
        data_owner_of=[data_node_id],
        # Permission to run stuff:
        # analyst_of=[data_node_id],
    )

    # TODO: add Cohort metadatadictionary to the DCR

    # # TODO: Add script that perform EDA to get info about the dataset as a PNG image
    # builder.add_node_definition(
    #     PythonComputeNodeDefinition(name=f"eda-{data_node_id}", script=eda_script, dependencies=[data_node_id])
    # )

    # builder.add_participant(
    #     settings.decentriq_email,
    #     data_owner_of=[data_node_id],
    # )
    # Build and publish DCR
    dcr_definition = builder.build()
    dcr = client.publish_analytics_dcr(dcr_definition)
    dcr_url = f"https://platform.decentriq.com/datarooms/p/{dcr.id}"
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


async def get_compute_dcr_definition(
    cohorts_request: dict[str, Any],
    user: Any,
    client: Any,
) -> Any:
    # users = [user["email"]]
    # TODO: cohorts_request could also be a dict of union of cohorts to merge
    # {"cohorts": {"cohort_id": ["var1", "var2"], "merged_cohort3": {"cohort1": ["weight", "sex"], "cohort2": ["gender", "patient weight"]}}}
    # We automatically merge the cohorts, figuring out which variables are the same thanks to mappings
    all_cohorts = retrieve_cohorts_metadata(user["email"])

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

        if cohort.airlock:
            # Add airlock node to make it easy to access small part of the dataset
            preview_node_id = f"preview-{data_node_id}"
            builder.add_node_definition(
                PreviewComputeNodeDefinition(
                    name=preview_node_id,
                    dependency=data_node_id,
                    quota_bytes=1048576,  # 10MB
                )
            )
            preview_nodes.append(preview_node_id)

        # Add data owners to provision the data (in dev we dont add them to avoid unnecessary emails)
        if not settings.dev_mode:
            for owner in cohort.cohort_email:
                if owner not in participants:
                    participants[owner] = {"data_owner_of": set(), "analyst_of": set()}
                participants[owner]["data_owner_of"].add(data_node_id)
        else:
            # In dev_mode the requester is added as data owner instead
            participants[user["email"]]["data_owner_of"].add(data_node_id)

        # Add pandas preparation script
        pandas_script = "import pandas as pd\nimport decentriq_util\n\n"
        df_var = f"df_{cohort_id.replace('-', '_')}"
        requested_vars = cohorts_request["cohorts"][cohort_id]

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
        pandas_script += f'{df_var}.to_csv("/output/{cohort_id}.csv", index=False, header=True)\n\n'

        # Add python data preparation script
        builder.add_node_definition(
            PythonComputeNodeDefinition(name=f"prepare-{cohort_id}", script=pandas_script, dependencies=[data_node_id])
        )
        # Add the requester as analyst of prepare script
        participants[user["email"]]["analyst_of"].add(f"prepare-{cohort_id}")

    # Add users permissions for previews
    for prev_node in preview_nodes:
        participants[user["email"]]["analyst_of"].add(prev_node)

    for p_email, p_perm in participants.items():
        builder.add_participant(
            p_email,
            data_owner_of=list(p_perm["data_owner_of"]),
            analyst_of=list(p_perm["analyst_of"]),
        )

    # Build and publish DCR
    return builder.build(), dcr_title



@router.post(
    "/create-compute-dcr",
    name="Create Data Clean Room for computing",
    response_description="Upload result",
)
async def create_compute_dcr(
    cohorts_request: dict[str, Any],
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    """Create a Data Clean Room for computing with the cohorts requested using Decentriq SDK"""
    # Establish connection to Decentriq
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)

    dcr_definition, dcr_title = await get_compute_dcr_definition(cohorts_request, user, client)

    dcr = client.publish_analytics_dcr(dcr_definition)
    dcr_url = f"https://platform.decentriq.com/datarooms/p/{dcr.id}"
    return {
        "message": f"Data Clean Room available for compute at {dcr_url}",
        "dcr_url": dcr_url,
        "dcr_title": dcr_title,
        # "merge_script": pandas_script,
        **cohorts_request,
    }


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
    return dcr_definition.high_level
