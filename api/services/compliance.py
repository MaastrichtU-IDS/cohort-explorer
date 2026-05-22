import logging
from datetime import datetime
from typing import Any

from api.services.cache import get_cache
from api.services.requester_type import infer_requester_type, is_email_from_allowed_country
from api.models.duo import (
    DUOModifier,
    is_permission_compatible,
    check_requester_type_constraint,
    check_purpose_constraint,
    PERMISSION_LABELS,
    MODIFIER_LABELS,
    get_modifiers_bitmask,
    bitmask_to_modifiers,
)

logger = logging.getLogger(__name__)

class ComplianceResult:
    \
\
    def __init__(self):
        self.compliant = True
        self.passed_checks: list[str] = []
        self.failed_checks: list[str] = []
        self.missing_attestations: list[str] = []
        self.remediation_steps: list[str] = []

    def add_pass(self, message: str):
        self.passed_checks.append(message)

    def add_fail(self, message: str, remediation: str | None = None):
        self.failed_checks.append(message)
        self.compliant = False
        if remediation:
            self.remediation_steps.append(remediation)

    def add_missing(self, attestation_type: str, remediation: str):
        self.missing_attestations.append(attestation_type)
        self.remediation_steps.append(remediation)

    def to_dict(self) -> dict:
        return {
            "compliant": self.compliant,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "missing_attestations": self.missing_attestations,
            "remediation_steps": self.remediation_steps
        }

class ComplianceService:
    \
\
    def __init__(self):
        self.cache = get_cache()

    async def check_compliance(
        self,
        cohort_id: str,
        cohort_hash: str,
        requester_email: str,
        requester_address: str,
        intended_use: str,
        requester_type: str | None = None,
        research_purpose: str = "general",
        disease_code: str | None = None,
        institution_id: str | None = None
    ) -> ComplianceResult:
        \
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
        result = ComplianceResult()

        consent = await self.cache.get_consent(cohort_hash)
        if not consent:
            result.add_fail(
                "Consent not found in cache",
                "Ensure consent has been recorded on blockchain"
            )
            return result

        if not consent.get("active"):
            result.add_fail("Consent has been revoked")
            return result

        result.add_pass("Consent is active")

        valid_until = consent.get("valid_until")
        if valid_until:
            if isinstance(valid_until, str):
                valid_until = datetime.fromisoformat(valid_until).timestamp()
            if valid_until < datetime.utcnow().timestamp():
                result.add_fail("Consent has expired")
                return result

        result.add_pass("Consent has not expired")

        consent_permission = consent.get("permission", "").upper()
        intended_upper = intended_use.upper()

        if not is_permission_compatible(consent_permission, intended_upper):
            result.add_fail(
                f"Permission incompatible: consent is {consent_permission} "
                f"({PERMISSION_LABELS.get(consent_permission, consent_permission)}), "
                f"requested {intended_upper} ({PERMISSION_LABELS.get(intended_upper, intended_upper)})"
            )
        else:
            result.add_pass(
                f"Permission compatible: {consent_permission} allows {intended_upper}"
            )

        modifiers = consent.get("modifiers", [])
        if isinstance(modifiers, int):
            modifiers = bitmask_to_modifiers(modifiers)
        modifiers_bitmask = get_modifiers_bitmask(modifiers)

        inferred_type, type_source = infer_requester_type(requester_email, requester_type)

        type_ok, type_reason = check_requester_type_constraint(modifiers_bitmask, inferred_type)
        if not type_ok:
            result.add_fail(
                f"{type_reason} (detected from {type_source})",
                "Contact data owner to request exception"
            )
        else:
            result.add_pass(f"Requester type '{inferred_type}' satisfies constraints")

        purpose_ok, purpose_reason = check_purpose_constraint(modifiers_bitmask, research_purpose)
        if not purpose_ok:
            result.add_fail(purpose_reason)
        else:
            result.add_pass(f"Research purpose '{research_purpose}' satisfies constraints")

        consent_disease = consent.get("disease_code")
        if consent_permission == "DS" or intended_upper == "DS":
            if consent_disease and disease_code:

                if consent_disease != disease_code:
                    result.add_fail(
                        f"Disease code mismatch: consent is for {consent_disease}, "
                        f"requested {disease_code}"
                    )
                else:
                    result.add_pass("Disease code matches consent")
            elif consent_disease and not disease_code:
                result.add_fail(
                    f"Disease code required: consent is for {consent_disease}",
                    "Specify disease_code in your request"
                )

        if "GS" in modifiers:
            allowed_countries = consent.get("allowed_countries", [])
            if allowed_countries:
                if institution_id:

                    result.add_pass("Geographic restriction: institution-based check pending")
                else:

                    if is_email_from_allowed_country(requester_email, allowed_countries):
                        result.add_pass("Geographic restriction satisfied (email TLD)")
                    else:
                        result.add_fail(
                            f"Geographic restriction: your location not in allowed countries",
                            "Register with an institution in an allowed country"
                        )
            else:
                result.add_pass("Geographic restriction: no countries specified")

        if "IS" in modifiers:
            allowed_institutions = consent.get("allowed_institutions", [])
            if allowed_institutions:
                if institution_id and institution_id in allowed_institutions:
                    result.add_pass("Institution restriction satisfied")
                else:
                    result.add_fail(
                        "Institution restriction: your institution not in allowed list",
                        "Contact data owner to add your institution"
                    )
            else:
                result.add_pass("Institution restriction: no institutions specified")

        if "IRB" in modifiers:
            has_irb = await self.cache.has_valid_attestation(
                requester_address, "IRB_APPROVAL", cohort_hash
            )
            if not has_irb:
                result.add_fail("IRB approval required but not found")
                result.add_missing("IRB_APPROVAL", "Submit IRB approval via POST /api/attestation/irb")
            else:
                result.add_pass("IRB approval verified")

        if "COL" in modifiers:
            has_col = await self.cache.has_collaboration(cohort_hash, requester_address)
            if not has_col:
                result.add_fail("Collaboration agreement required but not established")
                result.add_missing("COLLABORATION", "Contact data owner to establish collaboration")
            else:
                result.add_pass("Collaboration agreement verified")

        if "PUB" in modifiers:
            has_pub = await self.cache.has_valid_attestation(
                requester_address, "PUBLICATION", cohort_hash
            )
            if not has_pub:
                result.add_fail("Publication commitment required")
                result.add_missing("PUBLICATION", "Submit publication commitment via POST /api/attestation/commitment")
            else:
                result.add_pass("Publication commitment recorded")

        if "RTN" in modifiers:
            has_rtn = await self.cache.has_valid_attestation(
                requester_address, "RETURN_DATA", cohort_hash
            )
            if not has_rtn:
                result.add_fail("Return-to-database commitment required")
                result.add_missing("RETURN_DATA", "Submit return data commitment via POST /api/attestation/commitment")
            else:
                result.add_pass("Return-to-database commitment recorded")

        return result

    async def quick_access_check(
        self,
        cohort_hash: str,
        requester_address: str
    ) -> dict[str, Any]:
\
\
\
\
\

        consent = await self.cache.get_consent(cohort_hash)
        if not consent or not consent.get("active"):
            return {"has_access": False, "reason": "Consent not active"}

        valid_until = consent.get("valid_until")
        if valid_until:
            if isinstance(valid_until, str):
                valid_until = datetime.fromisoformat(valid_until).timestamp()
            if valid_until < datetime.utcnow().timestamp():
                return {"has_access": False, "reason": "Consent expired"}

        has_access = await self.cache.has_access(cohort_hash, requester_address)
        if has_access:
            access_data = await self.cache.get_access(cohort_hash, requester_address)
            return {
                "has_access": True,
                "access_type": "auto_approved",
                "granted_at": access_data.get("granted_at") if access_data else None
            }

        return {"has_access": False, "reason": "No access grant found"}

_compliance_service: ComplianceService | None = None

def get_compliance_service() -> ComplianceService:
    \
    global _compliance_service
    if _compliance_service is None:
        _compliance_service = ComplianceService()
    return _compliance_service
