from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, EmailStr, field_validator

VALID_PERMISSIONS = {"NRES", "GRU", "HMB", "DS", "POA"}
VALID_MODIFIERS = {"NPU", "NCU", "GSO", "NPOA", "PUB", "IRB", "COL", "GS", "IS", "PS", "MOR", "TS", "US", "CC", "RTN"}

def _validate_duo_permission(v: str) -> str:
    upper = v.upper()
    if upper not in VALID_PERMISSIONS:
        raise ValueError(
            f"Invalid DUO permission '{v}'. Must be one of: {', '.join(sorted(VALID_PERMISSIONS))}"
        )
    return upper

def _validate_duo_modifiers(v: list[str]) -> list[str]:
    result = []
    for mod in v:
        upper = mod.upper()
        if upper not in VALID_MODIFIERS:
            raise ValueError(
                f"Invalid DUO modifier '{mod}'. Must be one of: {', '.join(sorted(VALID_MODIFIERS))}"
            )
        result.append(upper)
    return result

class ConsentPrepareRequest(BaseModel):
    owner_email: EmailStr = Field(..., description="Email of the data owner")
    cohort_id: str = Field(..., description="Unique cohort identifier")
    duo_permission: str = Field(..., description="Primary DUO permission code (NRES, GRU, HMB, DS, POA)")
    duo_modifiers: list[str] = Field(default_factory=list, description="List of DUO modifier codes")
    disease_code: str | None = Field(None, description="ICD-10 disease code (required if DS)")
    allowed_countries: list[str] = Field(default_factory=list, description="ISO country codes (if GS)")
    allowed_institutions: list[str] = Field(default_factory=list, description="Institution IDs (if IS)")
    expiration_days: int = Field(0, description="Days until expiration (0 = no expiration)")
    metadata_uri: str = Field("", description="IPFS hash of additional metadata")

    @field_validator("duo_permission")
    @classmethod
    def validate_permission(cls, v: str) -> str:
        return _validate_duo_permission(v)

    @field_validator("duo_modifiers")
    @classmethod
    def validate_modifiers(cls, v: list[str]) -> list[str]:
        return _validate_duo_modifiers(v)

class EIP712TypedData(BaseModel):
    domain: dict
    types: dict
    message: dict
    primaryType: str

class ConsentPrepareResponse(BaseModel):
    typed_data: EIP712TypedData
    message_hash: str
    cohort_hash: str
    owner_address: str
    nonce: int

class ConsentRecordRequest(BaseModel):
    owner_email: EmailStr = Field(..., description="Email of the data owner")
    cohort_id: str = Field(..., description="Unique cohort identifier")
    duo_permission: str = Field(..., description="Primary DUO permission code (NRES, GRU, HMB, DS, POA)")
    duo_modifiers: list[str] = Field(default_factory=list, description="List of DUO modifier codes")
    disease_code: str | None = Field(None, description="ICD-10 disease code (required if DS)")
    allowed_countries: list[str] = Field(default_factory=list, description="ISO country codes (if GS)")
    allowed_institutions: list[str] = Field(default_factory=list, description="Institution IDs (if IS)")
    expiration_days: int = Field(0, description="Days until expiration (0 = no expiration)")
    metadata_uri: str = Field("", description="IPFS hash of additional metadata")
    signature: str | None = Field(None, description="EIP-712 signature (for meta-transaction)")

    @field_validator("duo_permission")
    @classmethod
    def validate_permission(cls, v: str) -> str:
        return _validate_duo_permission(v)

    @field_validator("duo_modifiers")
    @classmethod
    def validate_modifiers(cls, v: list[str]) -> list[str]:
        return _validate_duo_modifiers(v)

class ConsentSummary(BaseModel):
    permission: str
    permission_label: str
    modifiers: list[str]
    modifier_details: list[dict]
    expires_at: datetime | None

class ConsentRecordResponse(BaseModel):
    success: bool
    tx_hash: str
    cohort_hash: str
    owner_address: str
    consent_summary: ConsentSummary
    via_signature: bool = False

class ConsentStatusResponse(BaseModel):
    active: bool
    owners: list[str]
    duo_permission: str
    duo_permission_label: str
    duo_modifiers: list[str]
    modifier_details: list[dict]
    disease_code: str | None
    expires_at: datetime | None
    recorded_at: datetime

class ConsentRevokeRequest(BaseModel):
    owner_email: EmailStr
    cohort_id: str

class ConsentRevokeResponse(BaseModel):
    success: bool
    tx_hash: str
    revoked_access_count: int

class ConsentUpdateRequest(BaseModel):
    owner_email: EmailStr
    cohort_id: str
    duo_permission: str
    duo_modifiers: list[str]
    disease_code: str | None = None

    @field_validator("duo_permission")
    @classmethod
    def validate_permission(cls, v: str) -> str:
        return _validate_duo_permission(v)

    @field_validator("duo_modifiers")
    @classmethod
    def validate_modifiers(cls, v: list[str]) -> list[str]:
        return _validate_duo_modifiers(v)

class RevalidationResult(BaseModel):
    total_access_grants: int
    still_valid: int
    revoked: int
    revoked_requesters: list[str]

class ConsentUpdateResponse(BaseModel):
    success: bool
    tx_hash: str
    revalidation: RevalidationResult

class AddOwnerRequest(BaseModel):
    owner_email: EmailStr = Field(..., description="Existing owner email (for authorization)")
    cohort_id: str
    new_owner_email: EmailStr

class AddOwnerResponse(BaseModel):
    success: bool
    tx_hash: str
    new_owner_address: str

class AccessRequestInput(BaseModel):
    requester_email: EmailStr
    cohort_id: str
    intended_use: str = Field(..., description="DUO permission code for intended use (NRES, GRU, HMB, DS, POA)")
    requester_type: str | None = Field(None, description="Override for requester type")
    research_purpose: str = Field("general", description="Research purpose type")
    disease_code: str | None = None
    institution_id: str | None = Field(None, description="Optional institution ID override")
    commitments: dict = Field(default_factory=dict, description="Commitments (publication, irb_hash, etc.)")

    @field_validator("intended_use")
    @classmethod
    def validate_intended_use(cls, v: str) -> str:
        return _validate_duo_permission(v)

class PassedCheck(BaseModel):
    check: str

class FailedCheck(BaseModel):
    check: str

class AccessRequestResponse(BaseModel):
    compliant: bool
    auto_approved: bool
    request_id: str | None
    tx_hash: str | None
    passed_checks: list[PassedCheck]
    failed_checks: list[FailedCheck]
    missing_attestations: list[str] = Field(default_factory=list)
    remediation_steps: list[str] = Field(default_factory=list)

class AccessCheckResponse(BaseModel):
    has_access: bool
    access_type: str | None = None
    granted_at: datetime | None = None
    reason: str | None = None

class CompliancePreviewRequest(BaseModel):
    requester_email: EmailStr
    cohort_id: str
    intended_use: str
    requester_type: str | None = None
    research_purpose: str = "general"
    disease_code: str | None = None
    institution_id: str | None = None

    @field_validator("intended_use")
    @classmethod
    def validate_intended_use(cls, v: str) -> str:
        return _validate_duo_permission(v)

class CompliancePreviewResponse(BaseModel):
    would_be_compliant: bool
    passed_checks: list[PassedCheck]
    failed_checks: list[FailedCheck]
    missing_attestations: list[str] = Field(default_factory=list)
    remediation_steps: list[str] = Field(default_factory=list)

class IRBAttestationRequest(BaseModel):
    requester_email: EmailStr
    cohort_id: str
    irb_document_hash: str = Field(..., description="IPFS/hash of IRB document")
    irb_approval_number: str | None = None
    institution_name: str | None = None
    valid_days: int = 365

class IRBAttestationResponse(BaseModel):
    success: bool
    tx_hash: str
    attestation_id: str | None
    valid_until: datetime | None

class AttestationResponse(BaseModel):
    success: bool
    tx_hash: str
    attestation_id: str

class CommitmentRequest(BaseModel):
    requester_email: EmailStr
    cohort_id: str
    commitment_type: str = Field(..., description="Type: PUBLICATION, RETURN_DATA, DATA_DESTRUCTION")
    details: str = ""
    timeline: str | None = None
    valid_days: int = 365

class CommitmentResponse(BaseModel):
    success: bool
    tx_hash: str
    commitment_type: str
    recorded_at: datetime

class AttestationStatusResponse(BaseModel):
    exists: bool
    valid: bool
    attestation_type: str
    valid_until: datetime | None = None
    submitted_at: datetime | None = None
    details: str | None = None

class CollaborationEstablishRequest(BaseModel):
    owner_email: EmailStr
    cohort_id: str
    requester_email: EmailStr
    terms_hash: str | None = None
    valid_days: int = 365

class CollaborationEstablishResponse(BaseModel):
    success: bool
    tx_hash: str
    collaboration_id: str | None
    valid_until: datetime | None

class CollaborationStatusResponse(BaseModel):
    exists: bool
    active: bool
    terms_hash: str | None = None
    valid_until: datetime | None = None
    established_at: datetime | None = None

class CollaborationRequest(BaseModel):
    owner_email: EmailStr
    cohort_id: str
    requester_email: EmailStr
    terms_hash: str | None = None
    valid_days: int = 365

class CollaborationResponse(BaseModel):
    success: bool
    tx_hash: str
    collaboration_id: str

class CollaborationCheckResponse(BaseModel):
    has_collaboration: bool
    established_at: datetime | None = None
    valid_until: datetime | None = None

class PermissionInfo(BaseModel):
    code: str
    label: str
    description: str
    parent: str | None
    allows: list[str]

class ModifierInfo(BaseModel):
    code: str
    label: str
    verification_method: str

class OntologyResponse(BaseModel):
    permissions: list[PermissionInfo]
    modifiers: list[ModifierInfo]

class HealthResponse(BaseModel):
    status: str
    blockchain_connected: bool
    chain_id: int | None
    latest_block: int | None
    cache_synced: bool
    cache_consents: int

class CreateCommitmentRequest(BaseModel):
    researcher_email: EmailStr
    cohort_id: str
    commitment_type: Literal["PUBLICATION", "DATA_RETURN", "COLLABORATION", "MORATORIUM"]
    description: str = ""
    deadline_days: int = Field(0, description="Days until deadline (0 = use default)")

class CreateCommitmentResponse(BaseModel):
    success: bool
    tx_hash: str | None
    commitment_id: str
    deadline: datetime
    commitment_type: str

class FulfillCommitmentRequest(BaseModel):
    commitment_id: str
    researcher_email: EmailStr | None = Field(None, description="Researcher email (optional, for audit)")
    evidence_hash: str | None = Field(None, description="IPFS hash of evidence")
    evidence_uri: str = Field(..., description="Human-readable evidence (DOI, etc.)")

class FulfillCommitmentResponse(BaseModel):
    success: bool
    tx_hash: str | None
    fulfilled_at: datetime
    new_reputation_score: int | None = None

class CommitmentDetailResponse(BaseModel):
    commitment_id: str
    cohort_id: str
    researcher_hash: str = Field("", description="Pseudonymised researcher identifier")
    commitment_type: str
    description: str
    created_at: datetime
    deadline: datetime
    status: Literal["ACTIVE", "FULFILLED", "EXPIRED", "CANCELLED"]
    days_remaining: int | None
    evidence_uri: str | None = None
    fulfilled_at: datetime | None = None

class CommitmentSummaryResponse(BaseModel):
    researcher_hash: str = Field("", description="Pseudonymised researcher identifier")
    total: int
    active: int
    fulfilled: int
    expired: int
    cancelled: int
    fulfillment_rate: float = Field(..., ge=0, le=100)
    upcoming_deadlines: list[CommitmentDetailResponse] = Field(default_factory=list)

class MarkExpiredRequest(BaseModel):
    commitment_ids: list[str] = Field(default_factory=list, description="Specific IDs or empty for all")

class MarkExpiredResponse(BaseModel):
    success: bool
    marked_expired_count: int
    commitment_ids: list[str]

class TransparencyRequirementResponse(BaseModel):
    modifier: str
    label: str
    description: str
    tier: Literal["BLOCKING", "ATTESTATION", "ACKNOWLEDGMENT"]
    status: Literal["VERIFIED", "DECLARED", "ACKNOWLEDGED", "ACTION_REQUIRED", "FAILED", "PENDING"]
    verification_method: str
    verification_detail: str
    blocking: bool
    message: str
    action: str | None = None
    applies_to: list[str] = Field(default_factory=list)

class TransparencyReportResponse(BaseModel):
    overall_status: Literal["READY", "NEEDS_ACTION", "BLOCKED"]
    status_message: str
    summary: dict = Field(default_factory=dict)
    blocking_requirements: list[TransparencyRequirementResponse] = Field(default_factory=list)
    attestation_requirements: list[TransparencyRequirementResponse] = Field(default_factory=list)
    acknowledgment_requirements: list[TransparencyRequirementResponse] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    can_create_dcr: bool

class GenerateTransparencyReportRequest(BaseModel):
    requester_email: EmailStr
    cohort_ids: list[str]
    intended_use: str

    @field_validator("intended_use")
    @classmethod
    def validate_intended_use(cls, v: str) -> str:
        return _validate_duo_permission(v)
