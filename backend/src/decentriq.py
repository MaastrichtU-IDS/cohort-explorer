from typing import Any
import decentriq_platform as dq
import decentriq_platform.sql as dqsql
import decentriq_platform.container as dqc
from fastapi import APIRouter, HTTPException, Request

from src.config import settings

router = APIRouter()

# https://docs.decentriq.com/sdk/python-getting-started
@router.post(
    "/create-dcr",
    name="Create Data Clean Room",
    response_description="Upload result",
    response_model={},
)
async def create_dcr(
    request: Request,
    cohort_request: dict[str, Any],
) -> dict[str, str]:
    """Create a Data Clean Room using Decentriq SDK with the requested cohorts"""
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_email = settings.decentriq_email

    # TODO: Get user email, asked to Gaetan
    # Add field for user to provide their DQ API token


    # Establish connection to an enclave
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)

    # TODO: getting KeyError: 'decentriq.driver:v20'
    enclave_specs = dq.enclave_specifications.versions([
        "decentriq.driver:v20",
        "decentriq.sql-worker:v12",
        # "decentriq.driver:v21",
        # "decentriq.python-ml-worker-32-64:v21",
        # "decentriq.r-latex-worker-32-32:v16",
    ])
    auth, _ = client.create_auth_using_decentriq_pki(enclave_specs)
    session = client.create_session(auth, enclave_specs)

    # Creation of a Data Clean Room (DCR)
    dcr_count = len(client.get_data_room_descriptions())
    builder = dq.DataRoomBuilder(f"iCare4CVD DCR {dcr_count}", enclave_specs=enclave_specs)

    # Convert cohort variables to decentriq schema
    for cohort_id, cohort_info in cohort_request["cohorts"].items():
        schema = []
        for variable_id, variable_info in cohort_info["variables"].items():
            prim_type = dqsql.PrimitiveType.STRING
            if variable_info["VAR TYPE"] == "FLOAT":
                prim_type = dqsql.PrimitiveType.FLOAT64
            if variable_info["VAR TYPE"] == "INT":
                prim_type = dqsql.PrimitiveType.INT64
            print(variable_info["NA"])
            nullable = True
            # if not variable_info["NA"] or variable_info["NA"] and int(variable_info["NA"]) == 0:
            if variable_info["NA"] == 0:
                nullable = False
                # nullable = variable_info["NA"] and int(variable_info["NA"]) != 0
            schema.append((variable_id, prim_type, nullable))

        # Create data node for cohort
        data_node_builder = dqsql.TabularDataNodeBuilder(
            cohort_id,
            schema=schema
        )
        data_node_builder.add_to_builder(
            builder,
            authentication=client.decentriq_pki_authentication,
            users=[user_email]
        )

    data_room = builder.build()
    data_room_id = session.publish_data_room(data_room)
    # data_room_id = "TEST"
    print(data_room_id)
    print(client.get_data_room_descriptions())

    dcr_desc = client.get_data_room_description(data_room_id, enclave_specs)

    # TODO: not clear how we can import cohorts already in Decentriq

    print(session)
    print(builder)
    dcr_url = f"https://platform.decentriq.com/datarooms/p/{data_room_id}"
    return {
        "message": f"Data Clean Room available at {dcr_url}",
        "dcr_url": dcr_url,
        "dcr_title": dcr_desc["title"],
        "dcr": dcr_desc,
        **cohort_request,
    }

# TODO: We need to install the Decentriq SDK 0.24.2

# I would like to use the Decentriq SDK to create a Data Clean Room (DCR) on user request
# With just the data node provisioned, the user will then define compute on Decentriq platform
# Selecting the cohorts from a list of already uploaded datasets
# I expect to just have to pass the list of cohort names to the Decentriq SDK


# Create DCR without data, add the user to it using email they logged in
# The user, in 3 to 5 clicks add the data

