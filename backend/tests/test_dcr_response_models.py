from src.dcr_backends.models import (
    DcrCapabilities,
    DcrListResult,
    DcrRoom,
    LiveCreateResult,
    ProviderError,
)


def aadcr_capabilities() -> DcrCapabilities:
    return DcrCapabilities(
        supports_provisioning=True,
        supports_definition_preview=True,
        supports_live_creation=True,
        supports_room_refresh=True,
        supports_audit_log=True,
        supports_computation_output=True,
        supports_shuffle_output=False,
        synthetic_data_only=True,
        local_simulation=True,
    )


def test_live_create_result_preserves_existing_wizard_fields():
    capabilities = aadcr_capabilities()
    result = LiveCreateResult(
        message="Local room created",
        dcr_id="room-123",
        dcr_url="http://localhost:8001/api/dcr/room-123",
        dcr_title="TIME-CHF and GISSI-HF",
        cohort_ids=["TIME-CHF", "GISSI-HF"],
        num_cohorts=2,
        metadata_upload_results={"TIME-CHF": "success", "GISSI-HF": "success"},
        metadata_uploads_successful=2,
        shuffled_upload_results={"TIME-CHF": "success", "GISSI-HF": "success"},
        shuffled_uploads_successful=2,
        mapping_upload_results={"time-chf_gissi-hf_full.csv": "success"},
        mapping_uploads_successful=1,
        participants={
            "nikolas.molyndris@decentriq.ch": {
                "data_owner_of": [],
                "analyst_of": ["TIME-CHF", "GISSI-HF"],
            }
        },
        provider="aadcrv2",
        capabilities=capabilities,
    ).to_dict()

    assert result == {
        "message": "Local room created",
        "dcr_id": "room-123",
        "dcr_url": "http://localhost:8001/api/dcr/room-123",
        "dcr_title": "TIME-CHF and GISSI-HF",
        "cohort_ids": ["TIME-CHF", "GISSI-HF"],
        "num_cohorts": 2,
        "metadata_upload_results": {"TIME-CHF": "success", "GISSI-HF": "success"},
        "metadata_uploads_successful": 2,
        "shuffled_upload_results": {"TIME-CHF": "success", "GISSI-HF": "success"},
        "shuffled_uploads_successful": 2,
        "mapping_upload_results": {"time-chf_gissi-hf_full.csv": "success"},
        "mapping_uploads_successful": 1,
        "participants": {
            "nikolas.molyndris@decentriq.ch": {
                "data_owner_of": [],
                "analyst_of": ["TIME-CHF", "GISSI-HF"],
            }
        },
        "provider": "aadcrv2",
        "capabilities": capabilities.to_dict(),
    }


def test_room_and_list_result_preserve_my_dcr_contract():
    capabilities = aadcr_capabilities()
    room = DcrRoom(
        id="room-123",
        title="TIME-CHF local analysis",
        description="Synthetic local demonstration",
        createdAt="2026-07-15T12:00:00Z",
        owner={"email": "nikolas.molyndris@decentriq.ch"},
        participants=[
            {
                "email": "nikolas.molyndris@decentriq.ch",
                "roles": ["owner", "analyst"],
                "data_owner_of": ["TIME-CHF"],
                "analyst_of": ["TIME-CHF", "GISSI-HF"],
            }
        ],
        nodes=[
            {"name": "TIME-CHF", "type": "TableDataNodeDefinition"},
            {"name": "aggregate_summary", "type": "PythonComputeNodeDefinition"},
        ],
        cohorts=["TIME-CHF", "GISSI-HF"],
        provisioned_datasets=[
            {
                "dataset_name": "time-chf-synthetic.csv",
                "node_name": "TIME-CHF",
                "status": "PROVISIONED",
            }
        ],
        dcr_url="http://localhost:8001/api/dcr/room-123",
        provider="aadcrv2",
        capabilities=capabilities,
    )
    result = DcrListResult(
        dcrs=[room],
        count=1,
        email="nikolas.molyndris@decentriq.ch",
        provider="aadcrv2",
        capabilities=capabilities,
    ).to_dict()

    assert result["count"] == 1
    assert result["email"] == "nikolas.molyndris@decentriq.ch"
    assert result["provider"] == "aadcrv2"
    assert result["capabilities"]["supports_computation_output"] is True
    assert result["dcrs"][0] == {
        "id": "room-123",
        "title": "TIME-CHF local analysis",
        "description": "Synthetic local demonstration",
        "createdAt": "2026-07-15T12:00:00Z",
        "owner": {"email": "nikolas.molyndris@decentriq.ch"},
        "participants": [
            {
                "email": "nikolas.molyndris@decentriq.ch",
                "roles": ["owner", "analyst"],
                "data_owner_of": ["TIME-CHF"],
                "analyst_of": ["TIME-CHF", "GISSI-HF"],
            }
        ],
        "nodes": [
            {"name": "TIME-CHF", "type": "TableDataNodeDefinition"},
            {"name": "aggregate_summary", "type": "PythonComputeNodeDefinition"},
        ],
        "cohorts": ["TIME-CHF", "GISSI-HF"],
        "provisioned_datasets": [
            {
                "dataset_name": "time-chf-synthetic.csv",
                "node_name": "TIME-CHF",
                "status": "PROVISIONED",
            }
        ],
        "dcr_url": "http://localhost:8001/api/dcr/room-123",
        "provider": "aadcrv2",
        "capabilities": capabilities.to_dict(),
    }


def test_room_can_carry_a_normalized_provider_error():
    error = ProviderError(
        detail="Upload failed",
        provider="aadcrv2",
        failed_step="provision TIME-CHF",
        dcr_id="room-123",
        retryable=True,
    )
    room = DcrRoom(
        id="room-123",
        provider="aadcrv2",
        capabilities=aadcr_capabilities(),
        error=error,
    ).to_dict()

    assert room["error"] == {
        "detail": "Upload failed",
        "provider": "aadcrv2",
        "failed_step": "provision TIME-CHF",
        "dcr_id": "room-123",
        "retryable": True,
    }
