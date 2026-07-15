import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.dictionary_validation import normalize_dictionary_headers, validate_dictionary_schema

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "demo" / "metadata-fixtures" / "concepts.json"
DEFAULT_RELATIONSHIPS_PATH = (
    Path(__file__).resolve().parents[2] / "demo" / "metadata-fixtures" / "concept_relationship_enriched.csv"
)
REPORT_COLUMNS = (
    "variable name",
    "variable label",
    "domain",
    "vartype_status",
    "vartype_reason",
    "categorical_value_status",
    "categorical_value_reason",
    "variable_status",
    "variable_reason",
    "additional_context_status",
    "unit_status",
    "unit_reason",
    "visit_status",
    "visit_reason",
    "overall_status",
)


@dataclass(frozen=True)
class FixtureConcept:
    code: str
    omop_id: str
    name: str


@dataclass(frozen=True)
class ComponentResult:
    status: str
    reason: str


def _normalized(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _present(value: object) -> bool:
    return bool(_normalized(value) and _normalized(value) != "na")


def _parts(value: object) -> list[str]:
    if not _present(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _row_value(row: dict[str, str], name: str) -> str:
    aliases = {
        "variable name": ("variable name", "variablename"),
        "variable label": ("variable label", "variablelabel"),
    }
    for candidate in aliases.get(name, (name,)):
        if candidate in row:
            return row[candidate]
    return ""


class FixtureConceptValidationProvider:
    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        relationships_path: Path = DEFAULT_RELATIONSHIPS_PATH,
    ) -> None:
        self._concepts = self._load_concepts(catalog_path, relationships_path)

    @staticmethod
    def _load_concepts(catalog_path: Path, relationships_path: Path) -> dict[tuple[str, str], FixtureConcept]:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        records = payload["concepts"] if isinstance(payload, dict) else payload
        catalog = {
            (_normalized(record["id"]), str(record["omop_id"]).strip()): FixtureConcept(
                code=str(record["id"]).strip(),
                omop_id=str(record["omop_id"]).strip(),
                name=str(record["label"]).strip(),
            )
            for record in records
        }
        with relationships_path.open(encoding="utf-8", newline="") as handle:
            relationship_pairs = {
                (_normalized(row["concept_code"]), str(row["omop_id"]).strip()) for row in csv.DictReader(handle)
            }
        return {key: concept for key, concept in catalog.items() if key in relationship_pairs}

    def _component(
        self,
        names: object,
        codes: object,
        omop_ids: object,
        *,
        required: bool = False,
    ) -> ComponentResult:
        name_parts = _parts(names)
        code_parts = _parts(codes)
        omop_parts = _parts(omop_ids)
        counts = (len(name_parts), len(code_parts), len(omop_parts))
        if not any(counts):
            if required:
                return ComponentResult("FAIL", "Concept name, code, and OMOP ID are required.")
            return ComponentResult("N/A", "Concept not provided.")
        if len(set(counts)) != 1:
            return ComponentResult("FAIL", "Concept name, code, and OMOP ID counts do not match.")

        failures: list[str] = []
        for name, code, omop_id in zip(name_parts, code_parts, omop_parts):
            concept = self._concepts.get((_normalized(code), omop_id.strip()))
            if concept is None:
                failures.append(f"Concept pair {code!r}/{omop_id!r} is not present in the offline fixture.")
            elif _normalized(name) != _normalized(concept.name):
                failures.append(f"Concept name {name!r} does not match fixture label {concept.name!r}.")
        if failures:
            return ComponentResult("FAIL", " ".join(failures))
        return ComponentResult("PASS", "All concept pairs match the offline fixture.")

    def _validate_row(self, row: dict[str, str]) -> dict[str, str]:
        variable = self._component(
            row.get("variable concept name"),
            row.get("variable concept code"),
            row.get("variable omop id"),
            required=True,
        )
        additional = self._component(
            row.get("additional context concept name"),
            row.get("additional context concept code"),
            row.get("additional context omop id"),
        )
        categorical = self._component(
            row.get("categorical value concept name"),
            row.get("categorical value concept code"),
            row.get("categorical value omop id"),
        )
        unit = self._component(
            row.get("unit concept name"),
            row.get("unit concept code"),
            row.get("unit omop id"),
        )
        visit = self._component(
            row.get("visit concept name"),
            row.get("visit concept code"),
            row.get("visit omop id"),
        )

        if categorical.status != "N/A" and len(_parts(row.get("categorical"))) != len(
            _parts(row.get("categorical value concept name"))
        ):
            categorical = ComponentResult("FAIL", "Categorical values and concept counts do not match.")
        if unit.status != "N/A" and not _present(row.get("units")):
            unit = ComponentResult("FAIL", "Unit concept provided while UNITS is empty.")
        if visit.status != "N/A" and not _present(row.get("visits")):
            visit = ComponentResult("FAIL", "Visit concept provided while VISITS is empty.")
        if additional.status != "N/A" and variable.status != "PASS":
            additional = ComponentResult("FAIL", "Additional context requires a valid variable concept.")

        components = (categorical, variable, additional, unit, visit)
        overall = "FAIL" if any(component.status == "FAIL" for component in components) else "PASS"
        vartype = _row_value(row, "vartype").strip().upper()
        vartype_status = "PASS" if vartype in {"STR", "FLOAT", "INT", "DATETIME"} else "FAIL"
        if vartype_status == "FAIL":
            overall = "FAIL"
        return {
            "variable name": _row_value(row, "variable name"),
            "variable label": _row_value(row, "variable label"),
            "domain": _row_value(row, "domain"),
            "vartype_status": vartype_status,
            "vartype_reason": "Valid vartype." if vartype_status == "PASS" else "Invalid vartype.",
            "categorical_value_status": categorical.status,
            "categorical_value_reason": categorical.reason,
            "variable_status": variable.status,
            "variable_reason": variable.reason,
            "additional_context_status": (
                f"ValidationLog(status='{additional.status}', "
                f"description='{additional.reason}')"
            ),
            "unit_status": unit.status,
            "unit_reason": unit.reason,
            "visit_status": visit.status,
            "visit_reason": visit.reason,
            "overall_status": overall,
        }

    @staticmethod
    def _write_report(report_path: Path, rows: list[dict[str, str]]) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def validate(self, dictionary_path: Path, report_path: Path) -> bool:
        normalized = normalize_dictionary_headers(dictionary_path.read_text(encoding="utf-8-sig"))
        syntax_issues = validate_dictionary_schema(normalized)
        if syntax_issues:
            self._write_report(
                report_path,
                [
                    {
                        "variable_status": "FAIL",
                        "variable_reason": " ".join(syntax_issues),
                        "overall_status": "FAIL",
                    }
                ],
            )
            return False

        reader = csv.DictReader(normalized.splitlines())
        rows = [
            self._validate_row({str(key).strip().casefold(): value for key, value in row.items()}) for row in reader
        ]
        self._write_report(report_path, rows)
        return bool(rows) and all(row["overall_status"] == "PASS" for row in rows)
