from enum import IntEnum, IntFlag
from typing import TypedDict

class DUOPermission:
    NRES = 0x00000004
    GRU = 0x00000042
    HMB = 0x00000006
    DS = 0x00000007
    POA = 0x00000011

    @classmethod
    def from_code(cls, code: str) -> int:
        return PERMISSION_VALUES.get(code.upper(), 0)

    @classmethod
    def to_code(cls, value: int) -> str:
        return PERMISSION_CODES.get(value, "UNKNOWN")

PERMISSION_VALUES = {
    "NRES": DUOPermission.NRES,
    "GRU": DUOPermission.GRU,
    "HMB": DUOPermission.HMB,
    "DS": DUOPermission.DS,
    "POA": DUOPermission.POA,
}

PERMISSION_CODES = {v: k for k, v in PERMISSION_VALUES.items()}

PERMISSION_LABELS = {
    "NRES": "No Restriction",
    "GRU": "General Research Use",
    "HMB": "Health/Medical/Biomedical Research",
    "DS": "Disease Specific Research",
    "POA": "Population Origins/Ancestry Research",
}

PERMISSION_HIERARCHY = {
    "NRES": ["GRU", "HMB", "DS", "POA"],
    "GRU": ["HMB", "DS"],
    "HMB": ["DS"],
    "DS": [],
    "POA": [],
}

def is_permission_compatible(consented: str, requested: str) -> bool:
    consented = consented.upper()
    requested = requested.upper()

    if consented == requested:
        return True

    return requested in PERMISSION_HIERARCHY.get(consented, [])

class DUOModifier(IntFlag):
    NPU    = 1 << 0
    NCU    = 1 << 1
    NPUNCU = 1 << 2
    PUB    = 1 << 3
    COL    = 1 << 4
    IRB    = 1 << 5
    GS     = 1 << 6
    MOR    = 1 << 7
    TS     = 1 << 8
    US     = 1 << 9
    PS     = 1 << 10
    IS     = 1 << 11
    RTN    = 1 << 12
    CC     = 1 << 13
    NPOA   = 1 << 14
    GSO    = 1 << 15
    RS     = 1 << 16
    NMDS   = 1 << 17

MODIFIER_VALUES = {
    "NPU": DUOModifier.NPU,
    "NCU": DUOModifier.NCU,
    "NPUNCU": DUOModifier.NPUNCU,
    "PUB": DUOModifier.PUB,
    "COL": DUOModifier.COL,
    "IRB": DUOModifier.IRB,
    "GS": DUOModifier.GS,
    "MOR": DUOModifier.MOR,
    "TS": DUOModifier.TS,
    "US": DUOModifier.US,
    "PS": DUOModifier.PS,
    "IS": DUOModifier.IS,
    "RTN": DUOModifier.RTN,
    "CC": DUOModifier.CC,
    "NPOA": DUOModifier.NPOA,
    "GSO": DUOModifier.GSO,
    "RS": DUOModifier.RS,
    "NMDS": DUOModifier.NMDS,
}

MODIFIER_LABELS = {
    "NPU": "Not-for-profit organisations only (DUO:0000045)",
    "NCU": "Non-commercial use only (DUO:0000046)",
    "NPUNCU": "Not-for-profit, non-commercial only (DUO:0000018)",
    "PUB": "Publication required (DUO:0000019)",
    "COL": "Collaboration required (DUO:0000020)",
    "IRB": "Ethics approval required (DUO:0000021)",
    "GS": "Geographic restriction (DUO:0000022)",
    "MOR": "Publication moratorium (DUO:0000024)",
    "TS": "Time limit on use (DUO:0000025)",
    "US": "User-specific restriction (DUO:0000026)",
    "PS": "Project-specific restriction (DUO:0000027)",
    "IS": "Institution-specific restriction (DUO:0000028)",
    "RTN": "Return to database or resource (DUO:0000029)",
    "CC": "Clinical care use (DUO:0000043)",
    "NPOA": "Population origins/ancestry research prohibited (DUO:0000044)",
    "GSO": "Genetic studies only (DUO:0000016)",
    "RS": "Research-specific restrictions (DUO:0000012)",
    "NMDS": "No general methods research (DUO:0000015)",
}

def get_modifiers_bitmask(modifiers: list[str]) -> int:
    bitmask = 0
    for mod in modifiers:
        if mod.upper() in MODIFIER_VALUES:
            bitmask |= MODIFIER_VALUES[mod.upper()]
    return bitmask

def bitmask_to_modifiers(bitmask: int) -> list[str]:
    modifiers = []
    for name, value in MODIFIER_VALUES.items():
        if bitmask & value:
            modifiers.append(name)
    return modifiers

def get_modifier_details(modifiers: list[str]) -> list[dict]:
    return [
        {"code": mod, "label": MODIFIER_LABELS.get(mod, mod)}
        for mod in modifiers
    ]

class RequesterType(IntEnum):
    UNKNOWN = 0
    ACADEMIC = 1
    NONPROFIT = 2
    GOVERNMENT = 3
    COMMERCIAL = 4

REQUESTER_TYPE_VALUES = {
    "unknown": RequesterType.UNKNOWN,
    "academic": RequesterType.ACADEMIC,
    "nonprofit": RequesterType.NONPROFIT,
    "government": RequesterType.GOVERNMENT,
    "commercial": RequesterType.COMMERCIAL,
}

NPU_ALLOWED_TYPES = {
    RequesterType.ACADEMIC,
    RequesterType.NONPROFIT,
    RequesterType.GOVERNMENT,
}

NCU_ALLOWED_TYPES = {
    RequesterType.ACADEMIC,
    RequesterType.NONPROFIT,
    RequesterType.GOVERNMENT,
}

def check_requester_type_constraint(modifiers_bitmask: int, requester_type: str) -> tuple[bool, str]:
    req_type = REQUESTER_TYPE_VALUES.get(requester_type.lower(), RequesterType.UNKNOWN)

    if modifiers_bitmask & DUOModifier.NPU:
        if req_type not in NPU_ALLOWED_TYPES:
            return False, f"NPU modifier: {requester_type} organizations not allowed"

    if modifiers_bitmask & DUOModifier.NCU:
        if req_type not in NCU_ALLOWED_TYPES:
            return False, f"NCU modifier: commercial use not allowed"

    return True, "Requester type constraints satisfied"

class ResearchPurpose(IntEnum):
    GENERAL = 0
    HEALTH_RESEARCH = 1
    DISEASE_RESEARCH = 2
    POPULATION_RESEARCH = 3
    METHODS_DEVELOPMENT = 4
    GENETIC_STUDIES = 5
    CLINICAL_CARE = 6

RESEARCH_PURPOSE_VALUES = {
    "general": ResearchPurpose.GENERAL,
    "health_research": ResearchPurpose.HEALTH_RESEARCH,
    "disease_research": ResearchPurpose.DISEASE_RESEARCH,
    "population_research": ResearchPurpose.POPULATION_RESEARCH,
    "methods_development": ResearchPurpose.METHODS_DEVELOPMENT,
    "genetic_studies": ResearchPurpose.GENETIC_STUDIES,
    "clinical_care": ResearchPurpose.CLINICAL_CARE,
}

def check_purpose_constraint(modifiers_bitmask: int, purpose: str) -> tuple[bool, str]:
    purpose_type = RESEARCH_PURPOSE_VALUES.get(purpose.lower(), ResearchPurpose.GENERAL)

    if modifiers_bitmask & DUOModifier.GSO:
        if purpose_type != ResearchPurpose.GENETIC_STUDIES:
            return False, "GSO modifier: only genetic studies allowed"

    if modifiers_bitmask & DUOModifier.NPOA:
        if purpose_type == ResearchPurpose.POPULATION_RESEARCH:
            return False, "NPOA modifier: population/ancestry research prohibited"

    return True, "Research purpose constraints satisfied"

class ConsentData(TypedDict, total=False):
    cohort_hash: str
    owners: list[str]
    permission: str
    modifiers: list[str]
    disease_code: str | None
    allowed_countries: list[str]
    allowed_institutions: list[str]
    valid_from: int
    valid_until: int
    active: bool
    metadata_uri: str

class AccessRequestData(TypedDict, total=False):
    request_id: str
    cohort_hash: str
    requester: str
    intended_use: str
    purpose: str
    disease_code: str | None
    project_id: str | None
    requested_at: int
    decided: bool
    approved: bool
    reasons: list[str]

class ComplianceResult(TypedDict):
    compliant: bool
    passed_checks: list[str]
    failed_checks: list[str]
    missing_attestations: list[str]
