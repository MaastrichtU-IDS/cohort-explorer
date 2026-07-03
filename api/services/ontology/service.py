import logging
from dataclasses import dataclass
from typing import Optional

from api.services.ontology import icd10
from api.services.ontology.ror import RORClient, get_ror_client, InstitutionResult
from api.services.ontology.duo_lookup import DUOLookup, get_duo_lookup, DUOTerm

logger = logging.getLogger(__name__)

@dataclass
class UsageTypeMapping:
    ui_value: str
    duo_code: str
    label: str
    description: str

USAGE_TYPES = {
    "any": UsageTypeMapping(
        ui_value="any",
        duo_code="NRES",
        label="Any purpose (including commercial)",
        description="No restrictions on how the data can be used"
    ),
    "research": UsageTypeMapping(
        ui_value="research",
        duo_code="GRU",
        label="Research only",
        description="General research use, but not for commercial products"
    ),
    "health_research": UsageTypeMapping(
        ui_value="health_research",
        duo_code="HMB",
        label="Health/medical research only",
        description="Limited to health, medical, or biomedical research"
    ),
    "disease_specific": UsageTypeMapping(
        ui_value="disease_specific",
        duo_code="DS",
        label="Specific disease research only",
        description="Limited to research on a specific disease"
    ),
    "ancestry": UsageTypeMapping(
        ui_value="ancestry",
        duo_code="POA",
        label="Population/ancestry research only",
        description="Limited to population origins or ancestry studies"
    ),
}

RESTRICTION_OPTIONS = {
    "nonprofit_only": {
        "modifier": "NPU",
        "label": "Non-profit organizations only",
        "description": "Only non-profit organizations can access this data"
    },
    "noncommercial": {
        "modifier": "NCU",
        "label": "Non-commercial use only",
        "description": "Data cannot be used for commercial purposes"
    },
    "ethics_required": {
        "modifier": "IRB",
        "label": "Ethics committee approval required",
        "description": "Requesters must provide IRB/ethics approval"
    },
    "collaboration_required": {
        "modifier": "COL",
        "label": "Collaboration required",
        "description": "Requesters must collaborate with data owner"
    },
    "publication_required": {
        "modifier": "PUB",
        "label": "Publication required",
        "description": "Results must be published in a peer-reviewed journal"
    },
    "return_data": {
        "modifier": "RTN",
        "label": "Return derived data",
        "description": "Derived data must be returned to the original database"
    },
    "geographic": {
        "modifier": "GS",
        "label": "Geographic restriction",
        "description": "Limited to specific countries/regions"
    },
    "institution_specific": {
        "modifier": "IS",
        "label": "Institution restriction",
        "description": "Limited to specific institutions"
    },
    "time_limited": {
        "modifier": "TS",
        "label": "Time-limited access",
        "description": "Access expires after a specified period"
    },
}

class OntologyService:
    def __init__(self):
        self.ror = get_ror_client()
        self.duo = get_duo_lookup()

    async def close(self):
        await self.ror.close()
        await self.duo.close()

    async def search_diseases(
        self,
        query: str,
        limit: int = 10
    ) -> list[dict]:
        return icd10.search(query, limit=limit)

    async def get_disease(self, code: str) -> Optional[dict]:
        result = icd10.describe(code)
        return result if result.get("valid") else None

    async def validate_disease(self, code: str) -> dict:
        return icd10.describe(code)

    async def check_disease_compatibility(
        self,
        consent_disease: str,
        request_disease: str
    ) -> dict:
        if consent_disease == request_disease:
            return {
                "compatible": True,
                "reason": "Exact match"
            }

        if icd10.is_compatible(consent_disease, request_disease):
            return {
                "compatible": True,
                "reason": f"{request_disease} is a descendant of {consent_disease}"
            }

        return {
            "compatible": False,
            "reason": f"{request_disease} is not {consent_disease} nor one of its ICD-10 descendants"
        }

    async def search_institutions(
        self,
        query: str,
        country: Optional[str] = None,
        limit: int = 10
    ) -> list[dict]:
        results = await self.ror.search(query, country=country, limit=limit)
        return [
            {
                "id": r.ror_id,
                "name": r.name,
                "country": r.country_code,
                "country_name": r.country_name,
                "types": r.types,
                "city": r.city,
            }
            for r in results
        ]

    async def get_institution(self, ror_id: str) -> Optional[dict]:
        result = await self.ror.get_institution(ror_id)
        return result.to_dict() if result else None

    async def validate_institution(self, ror_id: str) -> dict:
        institution = await self.ror.get_institution(ror_id)
        if institution:
            return {
                "valid": True,
                "id": institution.ror_id,
                "name": institution.name,
                "country": institution.country_code,
                "types": institution.types,
            }
        return {
            "valid": False,
            "error": f"Invalid ROR ID: {ror_id}"
        }

    async def infer_institution_from_email(self, email: str) -> list[dict]:
        results = await self.ror.search_by_email_domain(email)
        return [
            {
                "id": r.ror_id,
                "name": r.name,
                "country": r.country_code,
                "confidence": "inferred"
            }
            for r in results
        ]

    def get_duo_permission(self, code: str) -> Optional[dict]:
        term = self.duo.get_permission(code)
        return term.to_dict() if term else None

    def get_duo_modifier(self, code: str) -> Optional[dict]:
        term = self.duo.get_modifier(code)
        return term.to_dict() if term else None

    def get_all_permissions(self) -> list[dict]:
        return [t.to_dict() for t in self.duo.get_all_permissions()]

    def get_all_modifiers(self) -> list[dict]:
        return [t.to_dict() for t in self.duo.get_all_modifiers()]

    def search_duo_terms(self, query: str) -> list[dict]:
        results = self.duo.search(query)
        return [t.to_dict() for t in results]

    def check_permission_compatibility(
        self,
        consent_perm: str,
        request_perm: str
    ) -> dict:
        is_compatible = self.duo.is_permission_compatible(consent_perm, request_perm)
        consent_term = self.duo.get_term(consent_perm)
        request_term = self.duo.get_term(request_perm)

        return {
            "compatible": is_compatible,
            "consent": {
                "code": consent_perm,
                "label": consent_term.label if consent_term else consent_perm
            },
            "request": {
                "code": request_perm,
                "label": request_term.label if request_term else request_perm
            },
            "explanation": self._explain_compatibility(consent_perm, request_perm, is_compatible)
        }

    def _explain_compatibility(self, consent: str, request: str, compatible: bool) -> str:
        if consent == request:
            return f"Exact match: both specify {consent}"

        if compatible:
            return f"{consent} allows {request} because it is more restrictive"

        return f"{consent} does not allow {request}. The requested use is broader than permitted."

    def get_usage_types(self) -> list[dict]:
        return [
            {
                "value": m.ui_value,
                "duo_code": m.duo_code,
                "label": m.label,
                "description": m.description
            }
            for m in USAGE_TYPES.values()
        ]

    def get_restriction_options(self) -> list[dict]:
        return [
            {
                "key": key,
                "modifier": opt["modifier"],
                "label": opt["label"],
                "description": opt["description"]
            }
            for key, opt in RESTRICTION_OPTIONS.items()
        ]

    def translate_to_duo(
        self,
        usage_type: str,
        restrictions: dict,
        disease_id: Optional[str] = None,
        countries: Optional[list[str]] = None,
        institutions: Optional[list[str]] = None,
        expires_at: Optional[str] = None
    ) -> dict:

        usage_mapping = USAGE_TYPES.get(usage_type, USAGE_TYPES["health_research"])
        permission_code = usage_mapping.duo_code
        permission_term = self.duo.get_permission(permission_code)

        modifiers = []
        modifier_terms = []

        for key, enabled in restrictions.items():
            if enabled and key in RESTRICTION_OPTIONS:
                mod_code = RESTRICTION_OPTIONS[key]["modifier"]
                modifiers.append(mod_code)
                term = self.duo.get_modifier(mod_code)
                if term:
                    modifier_terms.append(term)

        summary_parts = [f"Data can be used for {usage_mapping.label.lower()}"]

        if modifiers:
            mod_descriptions = [RESTRICTION_OPTIONS[k]["label"].lower()
                              for k, v in restrictions.items()
                              if v and k in RESTRICTION_OPTIONS]
            if mod_descriptions:
                summary_parts.append("Requires: " + ", ".join(mod_descriptions))

        if disease_id:
            summary_parts.append(f"Limited to disease: {disease_id}")

        if countries:
            summary_parts.append(f"Geographic restriction: {', '.join(countries)}")

        if institutions:
            summary_parts.append(f"Institution restriction: {len(institutions)} institution(s)")

        if expires_at:
            summary_parts.append(f"Access expires: {expires_at}")

        return {
            "permission": {
                "code": permission_code,
                "iri": permission_term.iri if permission_term else None,
                "label": permission_term.label if permission_term else permission_code,
                "definition": permission_term.definition if permission_term else None
            },
            "modifiers": [
                {
                    "code": t.code,
                    "iri": t.iri,
                    "label": t.label
                }
                for t in modifier_terms
            ],
            "modifier_codes": modifiers,
            "disease_code": disease_id,
            "allowed_countries": countries or [],
            "allowed_institutions": institutions or [],
            "expires_at": expires_at,
            "summary": {
                "plain_english": ". ".join(summary_parts) + ".",
                "restrictions_count": len(modifiers)
            }
        }

    def translate_from_duo(
        self,
        permission: str,
        modifiers: list[str],
        disease_code: Optional[str] = None
    ) -> dict:
        permission_term = self.duo.get_permission(permission)

        usage_type = None
        for key, mapping in USAGE_TYPES.items():
            if mapping.duo_code == permission.upper():
                usage_type = key
                break

        active_restrictions = {}
        for key, opt in RESTRICTION_OPTIONS.items():
            active_restrictions[key] = opt["modifier"] in [m.upper() for m in modifiers]

        modifier_descriptions = []
        for mod in modifiers:
            term = self.duo.get_modifier(mod)
            if term:
                modifier_descriptions.append(term.label)

        return {
            "usage_type": usage_type,
            "usage_label": permission_term.label if permission_term else permission,
            "restrictions": active_restrictions,
            "modifier_descriptions": modifier_descriptions,
            "disease_code": disease_code,
            "formatted": {
                "permission": permission_term.label if permission_term else permission,
                "modifiers": modifier_descriptions
            }
        }

_ontology_service: Optional[OntologyService] = None

def get_ontology_service() -> OntologyService:
    global _ontology_service
    if _ontology_service is None:
        _ontology_service = OntologyService()
    return _ontology_service

async def close_ontology_service():
    global _ontology_service
    if _ontology_service:
        await _ontology_service.close()
        _ontology_service = None
