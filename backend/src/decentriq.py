from typing import Any

import decentriq_platform as dq
import decentriq_platform.sql as dqsql
from fastapi import APIRouter, Depends

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
    # Get metadata for selected cohorts
    cohorts = {cohort_id: metadata for cohort_id, metadata in retrieve_cohorts_metadata(user["email"]).items() if cohort_id in cohorts_request["cohorts"]}

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
    for cohort_id, cohort in cohorts.items():
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
        **cohorts_request,
    }
