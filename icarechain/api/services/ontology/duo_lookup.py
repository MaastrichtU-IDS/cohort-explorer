import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

@dataclass
class DUOTerm:
    code: str
    duo_id: str
    iri: str
    label: str
    definition: str
    is_permission: bool
    is_modifier: bool
    parent: Optional[str] = None
    synonyms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "duo_id": self.duo_id,
            "iri": self.iri,
            "label": self.label,
            "definition": self.definition,
            "is_permission": self.is_permission,
            "is_modifier": self.is_modifier,
            "parent": self.parent,
            "synonyms": self.synonyms,
        }

DUO_PERMISSIONS = {
    "NRES": DUOTerm(
        code="NRES",
        duo_id="DUO:0000004",
        iri="http://purl.obolibrary.org/obo/DUO_0000004",
        label="no restriction",
        definition="This data use permission indicates there are no restrictions on use.",
        is_permission=True,
        is_modifier=False,
        parent=None,
    ),
    "GRU": DUOTerm(
        code="GRU",
        duo_id="DUO:0000042",
        iri="http://purl.obolibrary.org/obo/DUO_0000042",
        label="general research use",
        definition="This data use permission indicates that use is allowed for general research use for any research purpose.",
        is_permission=True,
        is_modifier=False,
        parent="NRES",
    ),
    "HMB": DUOTerm(
        code="HMB",
        duo_id="DUO:0000006",
        iri="http://purl.obolibrary.org/obo/DUO_0000006",
        label="health or medical or biomedical research",
        definition="This data use permission indicates that use is allowed for health/medical/biomedical purposes; does not include the study of population origins or ancestry.",
        is_permission=True,
        is_modifier=False,
        parent="GRU",
    ),
    "DS": DUOTerm(
        code="DS",
        duo_id="DUO:0000007",
        iri="http://purl.obolibrary.org/obo/DUO_0000007",
        label="disease-specific research",
        definition="This data use permission indicates that use is allowed provided it is related to the specified disease.",
        is_permission=True,
        is_modifier=False,
        parent="HMB",
        synonyms=["disease specific"],
    ),
    "POA": DUOTerm(
        code="POA",
        duo_id="DUO:0000011",
        iri="http://purl.obolibrary.org/obo/DUO_0000011",
        label="population origins or ancestry research only",
        definition="This data use permission indicates that use of the data is limited to the study of population origins or ancestry.",
        is_permission=True,
        is_modifier=False,
        parent=None,
    ),
}

DUO_MODIFIERS = {
    "NPU": DUOTerm(
        code="NPU",
        duo_id="DUO:0000045",
        iri="http://purl.obolibrary.org/obo/DUO_0000045",
        label="not-for-profit use only",
        definition="This data use modifier indicates that use of the data is limited to not-for-profit organizations.",
        is_permission=False,
        is_modifier=True,
    ),
    "NCU": DUOTerm(
        code="NCU",
        duo_id="DUO:0000046",
        iri="http://purl.obolibrary.org/obo/DUO_0000046",
        label="non-commercial use only",
        definition="This data use modifier indicates that use of the data is limited to not-for-profit use.",
        is_permission=False,
        is_modifier=True,
        synonyms=["no commercial use"],
    ),
    "GSO": DUOTerm(
        code="GSO",
        duo_id="DUO:0000016",
        iri="http://purl.obolibrary.org/obo/DUO_0000016",
        label="genetic studies only",
        definition="This data use modifier indicates that use is limited to genetic studies only.",
        is_permission=False,
        is_modifier=True,
    ),
    "NPOA": DUOTerm(
        code="NPOA",
        duo_id="DUO:0000018",
        iri="http://purl.obolibrary.org/obo/DUO_0000018",
        label="not for use in research involving population origins or ancestry",
        definition="This data use modifier indicates that the data cannot be used for research concerning ancestry or population origins.",
        is_permission=False,
        is_modifier=True,
        synonyms=["no population research"],
    ),
    "PUB": DUOTerm(
        code="PUB",
        duo_id="DUO:0000019",
        iri="http://purl.obolibrary.org/obo/DUO_0000019",
        label="publication required",
        definition="This data use modifier indicates that requestor agrees to make results of studies using the data available to the larger scientific community.",
        is_permission=False,
        is_modifier=True,
    ),
    "COL": DUOTerm(
        code="COL",
        duo_id="DUO:0000020",
        iri="http://purl.obolibrary.org/obo/DUO_0000020",
        label="collaboration required",
        definition="This data use modifier indicates that the requestor must agree to collaboration with the primary study investigator(s).",
        is_permission=False,
        is_modifier=True,
    ),
    "IRB": DUOTerm(
        code="IRB",
        duo_id="DUO:0000021",
        iri="http://purl.obolibrary.org/obo/DUO_0000021",
        label="ethics approval required",
        definition="This data use modifier indicates that the requestor must provide documentation of local IRB/ERB approval.",
        is_permission=False,
        is_modifier=True,
        synonyms=["ethics committee approval", "ERB approval"],
    ),
    "GS": DUOTerm(
        code="GS",
        duo_id="DUO:0000022",
        iri="http://purl.obolibrary.org/obo/DUO_0000022",
        label="geographical restriction",
        definition="This data use modifier indicates that use is limited to within a specific geographic region.",
        is_permission=False,
        is_modifier=True,
        synonyms=["geographic restriction"],
    ),
    "TS": DUOTerm(
        code="TS",
        duo_id="DUO:0000024",
        iri="http://purl.obolibrary.org/obo/DUO_0000024",
        label="time limit on use",
        definition="This data use modifier indicates that use is approved for a specific time period.",
        is_permission=False,
        is_modifier=True,
        synonyms=["time limited"],
    ),
    "MOR": DUOTerm(
        code="MOR",
        duo_id="DUO:0000025",
        iri="http://purl.obolibrary.org/obo/DUO_0000025",
        label="publication moratorium",
        definition="This data use modifier indicates that requestor agrees not to publish results of studies until a specific date.",
        is_permission=False,
        is_modifier=True,
    ),
    "RTN": DUOTerm(
        code="RTN",
        duo_id="DUO:0000029",
        iri="http://purl.obolibrary.org/obo/DUO_0000029",
        label="return to database or resource",
        definition="This data use modifier indicates that the requestor must return derived/enriched data to the database/resource.",
        is_permission=False,
        is_modifier=True,
        synonyms=["return data required"],
    ),
    "IS": DUOTerm(
        code="IS",
        duo_id="DUO:0000028",
        iri="http://purl.obolibrary.org/obo/DUO_0000028",
        label="institution-specific restriction",
        definition="This data use modifier indicates that use is limited to use within an approved institution.",
        is_permission=False,
        is_modifier=True,
    ),
    "CC": DUOTerm(
        code="CC",
        duo_id="DUO:0000043",
        iri="http://purl.obolibrary.org/obo/DUO_0000043",
        label="clinical care use",
        definition="This data use modifier indicates that use is allowed for clinical care purposes.",
        is_permission=False,
        is_modifier=True,
    ),
    "PS": DUOTerm(
        code="PS",
        duo_id="DUO:0000027",
        iri="http://purl.obolibrary.org/obo/DUO_0000027",
        label="project-specific restriction",
        definition="This data use modifier indicates that use is limited to use within an approved project.",
        is_permission=False,
        is_modifier=True,
    ),
    "RS": DUOTerm(
        code="RS",
        duo_id="DUO:0000012",
        iri="http://purl.obolibrary.org/obo/DUO_0000012",
        label="research-specific restriction",
        definition="This data use modifier indicates that use is limited to studies with a particular research type.",
        is_permission=False,
        is_modifier=True,
    ),
}

ALL_DUO_TERMS = {**DUO_PERMISSIONS, **DUO_MODIFIERS}

class DUOLookup:
    OLS_BASE_URL = "https://www.ebi.ac.uk/ols4/api"

    def __init__(self, use_ols_fallback: bool = True):
        self.use_ols_fallback = use_ols_fallback
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.OLS_BASE_URL,
                timeout=10.0,
                headers={"Accept": "application/json"}
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_permission(self, code: str) -> Optional[DUOTerm]:
        return DUO_PERMISSIONS.get(code.upper())

    def get_modifier(self, code: str) -> Optional[DUOTerm]:
        return DUO_MODIFIERS.get(code.upper())

    def get_term(self, code: str) -> Optional[DUOTerm]:
        return ALL_DUO_TERMS.get(code.upper())

    def get_all_permissions(self) -> list[DUOTerm]:
        return list(DUO_PERMISSIONS.values())

    def get_all_modifiers(self) -> list[DUOTerm]:
        return list(DUO_MODIFIERS.values())

    def search(self, query: str) -> list[DUOTerm]:
        query_lower = query.lower()
        results = []

        for term in ALL_DUO_TERMS.values():
            score = 0

            if term.code.lower() == query_lower:
                score = 100

            elif query_lower in term.label.lower():
                score = 80

            elif query_lower in term.definition.lower():
                score = 50

            elif any(query_lower in syn.lower() for syn in term.synonyms):
                score = 60

            if score > 0:
                results.append((score, term))

        results.sort(key=lambda x: x[0], reverse=True)
        return [term for _, term in results]

    def get_permission_hierarchy(self) -> dict:
        return {
            "NRES": {
                "label": DUO_PERMISSIONS["NRES"].label,
                "children": {
                    "GRU": {
                        "label": DUO_PERMISSIONS["GRU"].label,
                        "children": {
                            "HMB": {
                                "label": DUO_PERMISSIONS["HMB"].label,
                                "children": {
                                    "DS": {
                                        "label": DUO_PERMISSIONS["DS"].label,
                                        "children": {}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "POA": {
                "label": DUO_PERMISSIONS["POA"].label,
                "children": {}
            }
        }

    def is_permission_compatible(self, consent_perm: str, request_perm: str) -> bool:
        consent_upper = consent_perm.upper()
        request_upper = request_perm.upper()

        if consent_upper == request_upper:
            return True

        if consent_upper == "NRES":
            return True

        if consent_upper == "GRU" and request_upper in ["HMB", "DS"]:
            return True

        if consent_upper == "HMB" and request_upper == "DS":
            return True

        if consent_upper == "POA":
            return request_upper == "POA"

        return False

    def get_modifier_requirements(self, modifier_code: str) -> dict:
        requirements = {
            "NPU": {
                "type": "requester_type",
                "allowed": ["academic", "nonprofit", "government"],
                "blocked": ["commercial"],
                "description": "Requester must be a non-profit organization"
            },
            "NCU": {
                "type": "requester_type",
                "allowed": ["academic", "nonprofit", "government"],
                "blocked": ["commercial"],
                "description": "Use must be non-commercial"
            },
            "IRB": {
                "type": "attestation",
                "attestation_type": "IRB_APPROVAL",
                "description": "Requires ethics committee (IRB) approval documentation"
            },
            "COL": {
                "type": "collaboration",
                "description": "Requires collaboration agreement with data owner"
            },
            "PUB": {
                "type": "attestation",
                "attestation_type": "PUBLICATION",
                "description": "Requires commitment to publish results"
            },
            "GS": {
                "type": "geographic",
                "description": "Requester must be in an allowed geographic region"
            },
            "IS": {
                "type": "institution",
                "description": "Requester must be from an allowed institution"
            },
            "TS": {
                "type": "time_check",
                "description": "Access has a time limit"
            },
            "RTN": {
                "type": "attestation",
                "attestation_type": "RETURN_DATA",
                "description": "Requires commitment to return derived data"
            },
            "GSO": {
                "type": "research_purpose",
                "allowed": ["genetic_studies", "methods_development"],
                "description": "Use is limited to genetic studies"
            },
            "NPOA": {
                "type": "research_purpose",
                "blocked": ["population_research"],
                "description": "Cannot be used for population/ancestry research"
            },
        }

        return requirements.get(modifier_code.upper(), {
            "type": "unknown",
            "description": f"Unknown modifier: {modifier_code}"
        })

_duo_lookup: Optional[DUOLookup] = None

def get_duo_lookup() -> DUOLookup:
    global _duo_lookup
    if _duo_lookup is None:
        _duo_lookup = DUOLookup()
    return _duo_lookup
