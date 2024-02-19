import decentriq_platform as dq
import decentriq_platform.container as dqc
from fastapi import APIRouter, HTTPException, Request

from src.config import settings

router = APIRouter()


@router.post(
    "/create-dcr",
    name="Create Data Clean Room",
    response_description="Upload result",
    response_model={},
)
async def create_dcr(
    request: Request,
    cohort_request: dict[str, list[str]],
) -> dict[str, str]:
    """Create a Data Clean Room in Decentriq with the requested cohorts"""
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # TODO: Get user email, asked to Gaetan
    # Add field for user to provide their DQ API token


    # Use decentriq python SDK to create the data clean room
    # https://docs.decentriq.com/sdk/python-getting-started

    # Establish connection to an enclave
    client = dq.create_client(settings.decentriq_email, settings.decentriq_token)

    # TODO: getting KeyError: 'decentriq.driver:v20'
    enclave_specs = dq.enclave_specifications.versions([
        "decentriq.driver:v20",
        # "decentriq.driver:v21",
        # "decentriq.python-ml-worker-32-64:v21",
        # "decentriq.r-latex-worker-32-32:v16",
    ])
    auth, _ = client.create_auth_using_decentriq_pki(enclave_specs)
    session = client.create_session(auth, enclave_specs)

    # Creation of a Data Clean Room (DCR)
    builder = dq.DataRoomBuilder("iCare4CVD DCR", enclave_specs=enclave_specs)

    # TODO: not clear how we can import cohorts already in Decentriq

    print(session)
    print(builder)

    return {
        "message": "Data Clean Room requested at https://platform.decentriq.com",
        "dcr_url": "https://platform.decentriq.com",
        **cohort_request,
    }

# TODO: We need to install the Decentriq SDK 0.24.2

# I would like to use the Decentriq SDK to create a Data Clean Room (DCR) on user request
# With just the data node provisioned, the user will then define compute on Decentriq platform
# Selecting the cohorts from a list of already uploaded datasets
# I expect to just have to pass the list of cohort names to the Decentriq SDK


# Create DCR without data, add the user to it using email they logged in
# The user, in 3 to 5 clicks add the data

