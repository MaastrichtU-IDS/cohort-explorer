import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from api.services.ontology import icd10

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Condition:
    modifier: str
    requirement: str
    attestation_type: Optional[str] = None
    deadline: Optional[int] = None
    severity: str = "required"

@dataclass
class ScoreBreakdown:
    component: str
    max_score: int
    achieved_score: int
    percentage: float
    details: list[str] = field(default_factory=list)

@dataclass
class ComplianceScore:
    total_score: int

    permission_score: int
    modifier_score: int
    attestation_score: int
    trust_score: int

    risk_level: RiskLevel

    conditions: list[Condition] = field(default_factory=list)

    recommendations: list[str] = field(default_factory=list)

    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    partial_checks: list[tuple[str, int]] = field(default_factory=list)

    auto_approve: bool = False
    requires_review: bool = False
    requires_conditions: bool = False

    @property
    def decision(self) -> str:
        if self.risk_level == RiskLevel.CRITICAL:
            return "DENIED"
        if self.conditions and any(c.severity == "required" for c in self.conditions):
            return "CONDITIONAL_APPROVAL"
        if self.risk_level == RiskLevel.LOW:
            return "APPROVED"
        if self.risk_level == RiskLevel.MEDIUM:
            return "REVIEW_RECOMMENDED"
        return "MANUAL_REVIEW_REQUIRED"

PERMISSION_HIERARCHY = {
    "NRES": {"level": 0, "children": ["GRU"]},
    "GRU": {"level": 1, "children": ["HMB"], "parent": "NRES"},
    "HMB": {"level": 2, "children": ["DS"], "parent": "GRU"},
    "DS": {"level": 3, "children": [], "parent": "HMB"},
    "POA": {"level": 0, "children": [], "branch": "ancestry"},
}

MODIFIER_IMPACTS = {
    "NPU": {
        "weight": 8,
        "critical": True,
        "bonus_possible": False,
        "description": "Non-profit organizations only",
        "check_type": "requester_type",
        "allowed_types": ["academic", "nonprofit", "government"],
    },
    "NCU": {
        "weight": 7,
        "critical": True,
        "bonus_possible": False,
        "description": "Non-commercial use only",
        "check_type": "requester_type",
        "allowed_types": ["academic", "nonprofit", "government"],
    },
    "IRB": {
        "weight": 10,
        "critical": True,
        "bonus_possible": True,
        "description": "Ethics committee approval required",
        "check_type": "attestation",
        "attestation_type": "IRB_APPROVAL",
    },
    "COL": {
        "weight": 9,
        "critical": True,
        "bonus_possible": True,
        "description": "Collaboration with data owner required",
        "check_type": "attestation",
        "attestation_type": "COLLABORATION",
    },
    "PUB": {
        "weight": 5,
        "critical": False,
        "bonus_possible": True,
        "description": "Publication of results required",
        "check_type": "attestation",
        "attestation_type": "PUBLICATION",
    },
    "RTN": {
        "weight": 5,
        "critical": False,
        "bonus_possible": True,
        "description": "Return derived data required",
        "check_type": "attestation",
        "attestation_type": "RETURN_DATA",
    },
    "GS": {
        "weight": 8,
        "critical": True,
        "bonus_possible": False,
        "description": "Geographic restriction",
        "check_type": "geographic",
    },
    "IS": {
        "weight": 8,
        "critical": True,
        "bonus_possible": False,
        "description": "Institution restriction",
        "check_type": "institution",
    },
    "TS": {
        "weight": 6,
        "critical": True,
        "bonus_possible": False,
        "description": "Time-limited access",
        "check_type": "time",
    },
    "GSO": {
        "weight": 4,
        "critical": False,
        "bonus_possible": False,
        "description": "Genetic studies only",
        "check_type": "purpose",
        "allowed_purposes": ["genetic_studies", "methods_development"],
    },
    "NPOA": {
        "weight": 4,
        "critical": True,
        "bonus_possible": False,
        "description": "Population research prohibited",
        "check_type": "purpose",
        "blocked_purposes": ["population_research"],
    },
}

class ComplianceScoringEngine:
    def calculate_score(
        self,
        consent: dict,
        request: dict,
        requester_profile: dict,
        attestations: list[dict],
        geographic_proof: Optional[dict] = None,
        institution_proof: Optional[dict] = None
    ) -> ComplianceScore:

        perm_result = self._score_permission(
            consent.get("permission", ""),
            request.get("intended_use", ""),
            consent.get("disease_code"),
            request.get("disease_code")
        )

        mod_result = self._score_modifiers(
            consent.get("modifiers", []),
            requester_profile,
            attestations,
            request.get("purpose"),
            geographic_proof,
            institution_proof
        )

        att_result = self._score_attestations(
            consent.get("modifiers", []),
            attestations
        )

        trust_result = self._score_trust(requester_profile)

        total = (
            perm_result["score"] +
            mod_result["score"] +
            att_result["score"] +
            trust_result["score"]
        )

        if total >= 800:
            risk_level = RiskLevel.LOW
        elif total >= 600:
            risk_level = RiskLevel.MEDIUM
        elif total >= 400:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        if perm_result["score"] == 0:
            risk_level = RiskLevel.CRITICAL

        passed = (
            perm_result["passed"] +
            mod_result["passed"] +
            att_result["passed"] +
            trust_result["passed"]
        )
        failed = (
            perm_result["failed"] +
            mod_result["failed"] +
            att_result["failed"] +
            trust_result["failed"]
        )
        partial = perm_result.get("partial", []) + mod_result.get("partial", [])

        recommendations = self._generate_recommendations(
            total, mod_result["conditions"], requester_profile
        )

        return ComplianceScore(
            total_score=total,
            permission_score=perm_result["score"],
            modifier_score=mod_result["score"],
            attestation_score=att_result["score"],
            trust_score=trust_result["score"],
            risk_level=risk_level,
            conditions=mod_result["conditions"],
            recommendations=recommendations,
            passed_checks=passed,
            failed_checks=failed,
            partial_checks=partial,
            auto_approve=(risk_level == RiskLevel.LOW and not mod_result["conditions"]),
            requires_review=(risk_level == RiskLevel.MEDIUM),
            requires_conditions=bool(mod_result["conditions"])
        )

    def _score_permission(
        self,
        consent_perm: str,
        request_perm: str,
        consent_disease: Optional[str] = None,
        request_disease: Optional[str] = None
    ) -> dict:
        result = {
            "score": 0,
            "passed": [],
            "failed": [],
            "partial": []
        }

        consent_perm = consent_perm.upper()
        request_perm = request_perm.upper()

        if not consent_perm or consent_perm not in PERMISSION_HIERARCHY:
            result["failed"].append(f"Invalid consent permission: {consent_perm}")
            return result

        if not request_perm or request_perm not in PERMISSION_HIERARCHY:
            result["failed"].append(f"Invalid request permission: {request_perm}")
            return result

        if consent_perm == request_perm:
            result["score"] = 300
            result["passed"].append(f"Exact permission match: {consent_perm}")

            if consent_perm == "DS" and consent_disease and request_disease:
                if icd10.is_compatible(consent_disease, request_disease):
                    result["passed"].append(f"Disease match: {request_disease} within {consent_disease}")
                else:

                    result["partial"].append(
                        (f"Disease mismatch: {consent_disease} vs {request_disease}", 80)
                    )
                    result["score"] = 240
            return result

        if consent_perm == "NRES":
            result["score"] = 285
            result["passed"].append(f"No restriction consent allows {request_perm}")
            return result

        consent_info = PERMISSION_HIERARCHY[consent_perm]
        request_info = PERMISSION_HIERARCHY[request_perm]

        if consent_info.get("branch") == "ancestry" or request_info.get("branch") == "ancestry":
            if consent_perm != request_perm:
                result["score"] = 0
                result["failed"].append(
                    f"POA incompatible with other permission types"
                )
                return result

        if self._is_descendant(request_perm, consent_perm):
            depth = request_info["level"] - consent_info["level"]
            if depth == 1:
                result["score"] = 270
                result["passed"].append(
                    f"Compatible: {consent_perm} allows {request_perm} (direct child)"
                )
            else:
                result["score"] = 240
                result["passed"].append(
                    f"Compatible: {consent_perm} allows {request_perm} (descendant)"
                )
            return result

        if self._is_descendant(consent_perm, request_perm):
            result["score"] = 60
            result["failed"].append(
                f"Request {request_perm} is broader than consent {consent_perm}"
            )
            result["partial"].append(
                (f"Same branch but incompatible direction", 20)
            )
            return result

        result["score"] = 0
        result["failed"].append(
            f"Permission incompatible: {consent_perm} does not allow {request_perm}"
        )
        return result

    def _is_descendant(self, child: str, parent: str) -> bool:
        current = child
        while current in PERMISSION_HIERARCHY:
            current_parent = PERMISSION_HIERARCHY[current].get("parent")
            if current_parent == parent:
                return True
            if current_parent is None:
                break
            current = current_parent
        return False

    def _score_modifiers(
        self,
        consent_modifiers: list[str],
        requester_profile: dict,
        attestations: list[dict],
        purpose: Optional[str] = None,
        geographic_proof: Optional[dict] = None,
        institution_proof: Optional[dict] = None
    ) -> dict:
        result = {
            "score": 400,
            "passed": [],
            "failed": [],
            "partial": [],
            "conditions": []
        }

        if not consent_modifiers:
            result["passed"].append("No modifier restrictions")
            return result

        total_weight = 0
        achieved_weight = 0

        for mod in consent_modifiers:
            mod = mod.upper()
            if mod not in MODIFIER_IMPACTS:
                continue

            impact = MODIFIER_IMPACTS[mod]
            total_weight += impact["weight"]

            satisfied, reason = self._check_modifier(
                mod,
                impact,
                requester_profile,
                attestations,
                purpose,
                geographic_proof,
                institution_proof
            )

            if satisfied:
                achieved_weight += impact["weight"]
                result["passed"].append(f"{mod}: {reason}")

                if impact["bonus_possible"]:
                    bonus = self._calculate_modifier_bonus(mod, attestations)
                    if bonus > 0:
                        achieved_weight += min(bonus, 2)
                        result["passed"].append(f"{mod} quality bonus: +{bonus}")
            else:
                result["failed"].append(f"{mod}: {reason}")

                if impact["critical"]:
                    result["conditions"].append(Condition(
                        modifier=mod,
                        requirement=reason,
                        attestation_type=impact.get("attestation_type"),
                        severity="required"
                    ))
                else:
                    result["conditions"].append(Condition(
                        modifier=mod,
                        requirement=reason,
                        attestation_type=impact.get("attestation_type"),
                        severity="recommended"
                    ))

        if total_weight == 0:
            return result

        result["score"] = int((achieved_weight / total_weight) * 400)
        return result

    def _check_modifier(
        self,
        modifier: str,
        impact: dict,
        requester_profile: dict,
        attestations: list[dict],
        purpose: Optional[str],
        geographic_proof: Optional[dict],
        institution_proof: Optional[dict]
    ) -> tuple[bool, str]:
        check_type = impact.get("check_type")

        if check_type == "requester_type":
            req_type = requester_profile.get("type", "unknown")
            allowed = impact.get("allowed_types", [])
            if req_type in allowed:
                return True, f"Requester type '{req_type}' is allowed"
            return False, f"Requester type '{req_type}' not in allowed list"

        elif check_type == "attestation":
            att_type = impact.get("attestation_type")
            att = next(
                (a for a in attestations if a.get("type") == att_type),
                None
            )
            if att:

                if att.get("valid_until") and att["valid_until"] < time.time():
                    return False, f"{att_type} attestation expired"
                return True, f"{att_type} attestation found and valid"
            return False, f"{att_type} attestation required but not found"

        elif check_type == "purpose":
            allowed = impact.get("allowed_purposes", [])
            blocked = impact.get("blocked_purposes", [])

            if allowed and purpose not in allowed:
                return False, f"Purpose '{purpose}' not in allowed list: {allowed}"
            if blocked and purpose in blocked:
                return False, f"Purpose '{purpose}' is blocked"
            return True, f"Purpose '{purpose}' is acceptable"

        elif check_type == "geographic":
            if geographic_proof and geographic_proof.get("verified"):
                return True, f"Geographic location verified: {geographic_proof.get('country')}"
            return False, "Geographic verification required"

        elif check_type == "institution":
            if institution_proof and institution_proof.get("verified"):
                return True, f"Institution verified: {institution_proof.get('institution')}"
            return False, "Institution verification required"

        elif check_type == "time":

            return True, "Time restriction will be enforced"

        return True, "No specific check required"

    def _calculate_modifier_bonus(
        self,
        modifier: str,
        attestations: list[dict]
    ) -> int:
        bonus = 0

        for att in attestations:
            if att.get("type") == MODIFIER_IMPACTS.get(modifier, {}).get("attestation_type"):

                if att.get("issuer_verified"):
                    bonus += 1

                if att.get("document_hash"):
                    bonus += 1

                if att.get("eas_uid"):
                    bonus += 1

        return bonus

    def _score_attestations(
        self,
        required_modifiers: list[str],
        attestations: list[dict]
    ) -> dict:
        result = {
            "score": 0,
            "passed": [],
            "failed": []
        }

        if not attestations:
            if not required_modifiers:
                result["score"] = 200
                result["passed"].append("No attestations required")
            else:
                result["score"] = 50
            return result

        score = 0

        for att in attestations:

            if att.get("valid"):
                score += 30
                result["passed"].append(f"Valid attestation: {att.get('type')}")
            else:
                result["failed"].append(f"Invalid attestation: {att.get('type')}")
                continue

            if att.get("issuer_verified"):
                score += 20
                result["passed"].append(f"Trusted issuer: {att.get('issuer')}")

            valid_until = att.get("valid_until")
            if valid_until is None or valid_until > time.time():
                score += 10
            else:
                result["failed"].append(f"Expired attestation: {att.get('type')}")

            if att.get("document_hash"):
                score += 15
                result["passed"].append(f"Verifiable document: {att.get('type')}")

            if att.get("eas_uid"):
                score += 15
                result["passed"].append(f"EAS attestation: {att.get('eas_uid')[:16]}...")

        result["score"] = min(score, 200)
        return result

    def _score_trust(self, requester_profile: dict) -> dict:
        result = {
            "score": 0,
            "passed": [],
            "failed": []
        }

        if requester_profile.get("institution_verified"):
            result["score"] += 30
            result["passed"].append(
                f"Verified institution: {requester_profile.get('institution')}"
            )
        else:
            result["failed"].append("Institution not verified")

        history = requester_profile.get("request_history", {})
        approved = history.get("approved", 0)
        if approved >= 5:
            result["score"] += 20
            result["passed"].append(f"Excellent history: {approved} approved requests")
        elif approved >= 2:
            result["score"] += 10
            result["passed"].append(f"Good history: {approved} approved requests")

        violations = history.get("violations", 0)
        if violations == 0:
            result["score"] += 30
            result["passed"].append("No compliance violations")
        else:
            result["failed"].append(f"Has {violations} violation(s)")

        if requester_profile.get("email_domain_trusted"):
            result["score"] += 20
            result["passed"].append("Trusted email domain")

        result["score"] = min(result["score"], 100)
        return result

    def _generate_recommendations(
        self,
        total_score: int,
        conditions: list[Condition],
        requester_profile: dict
    ) -> list[str]:
        recs = []

        if total_score < 600:
            if not requester_profile.get("institution_verified"):
                recs.append(
                    "Verify your institution via ROR to gain +30 points"
                )

        for cond in conditions:
            if cond.attestation_type == "IRB_APPROVAL":
                recs.append(
                    "Submit IRB approval document to satisfy ethics requirement (+~100 points)"
                )
            elif cond.attestation_type == "COLLABORATION":
                recs.append(
                    "Contact data owner to establish collaboration agreement"
                )
            elif cond.attestation_type == "PUBLICATION":
                recs.append(
                    "Make a publication commitment attestation (+~50 points)"
                )

        if total_score >= 600 and total_score < 800:
            recs.append(
                "Consider using EAS (Ethereum Attestation Service) for higher trust score"
            )

        return recs

_scoring_engine: Optional[ComplianceScoringEngine] = None

def get_scoring_engine() -> ComplianceScoringEngine:
    global _scoring_engine
    if _scoring_engine is None:
        _scoring_engine = ComplianceScoringEngine()
    return _scoring_engine
