import csv
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.metadata_providers.contracts import ConceptValidationProvider

REQUIRED_DICTIONARY_COLUMNS = (
    "VARIABLENAME",
    "VARIABLELABEL",
    "VARTYPE",
    "UNITS",
    "CATEGORICAL",
    "MISSING",
    "COUNT",
    "NA",
    "MIN",
    "MAX",
    "FORMULA",
    "CATEGORICAL VALUE CONCEPT CODE",
    "CATEGORICAL VALUE CONCEPT NAME",
    "CATEGORICAL VALUE OMOP ID",
    "VARIABLE CONCEPT CODE",
    "VARIABLE CONCEPT NAME",
    "VARIABLE OMOP ID",
    "ADDITIONAL CONTEXT CONCEPT NAME",
    "ADDITIONAL CONTEXT CONCEPT CODE",
    "ADDITIONAL CONTEXT OMOP ID",
    "UNIT CONCEPT NAME",
    "UNIT CONCEPT CODE",
    "UNIT OMOP ID",
    "DOMAIN",
    "VISITS",
    "VISIT OMOP ID",
    "VISIT CONCEPT NAME",
    "VISIT CONCEPT CODE",
)
REQUIRED_ROW_VALUES = ("VARIABLENAME", "VARIABLELABEL", "VARTYPE", "DOMAIN", "VARIABLE OMOP ID")
ACCEPTED_DATATYPES = frozenset({"STR", "FLOAT", "INT", "DATETIME"})
ACCEPTED_DOMAINS = frozenset(
    {
        "condition_occurrence",
        "visit_occurrence",
        "procedure_occurrence",
        "measurement",
        "drug_exposure",
        "device_exposure",
        "person",
        "observation",
        "observation_period",
        "death",
        "specimen",
        "condition_era",
        "drug_era",
        "dose_era",
    }
)
HEADER_ALIASES = {
    "variable name": "VARIABLENAME",
    "variable label": "VARIABLELABEL",
    "var type": "VARTYPE",
}


class InvalidDictionary(ValueError):  # noqa: N818 - public contract from the implementation plan
    def __init__(self, issues: tuple[str, ...] | list[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n\n".join(self.issues))


@dataclass(frozen=True)
class DictionaryValidationResult:
    cohort_id: str
    normalized_csv: bytes
    syntax_issues: tuple[str, ...]
    concepts_valid: bool


def normalize_dictionary_header(header: str) -> str:
    cleaned = " ".join(header.strip().split())
    return HEADER_ALIASES.get(cleaned.casefold(), cleaned.upper())


def normalize_dictionary_headers(text: str) -> str:
    """Normalize dictionary headers without changing any data-cell bytes semantically."""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return text
    rows[0] = [normalize_dictionary_header(column) for column in rows[0]]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def _present(value: str | None) -> bool:
    return bool(value and value.strip() and value.strip().casefold() != "na")


def _split_present(value: str | None) -> list[str]:
    if not _present(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def validate_dictionary_schema(normalized_csv: str) -> list[str]:
    """Return deterministic local syntax issues, without writing canonical state."""
    try:
        reader = csv.DictReader(io.StringIO(normalized_csv))
        headers = reader.fieldnames or []
        rows = list(reader)
    except csv.Error as error:
        return [f"Could not parse CSV: {error}"]

    if not headers:
        return ["The uploaded CSV is empty or has no header row."]

    duplicate_headers = sorted({header for header in headers if headers.count(header) > 1})
    issues: list[str] = []
    if duplicate_headers:
        issues.append(f"Duplicate columns: {', '.join(duplicate_headers)}")

    missing_columns = [column for column in REQUIRED_DICTIONARY_COLUMNS if column not in headers]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        return issues
    if not rows:
        return ["The uploaded CSV contains no metadata rows."]

    seen_variables: set[str] = set()
    duplicate_variables: set[str] = set()
    for index, row in enumerate(rows, start=2):
        variable = str(row.get("VARIABLENAME", "")).strip()
        if variable and variable in seen_variables:
            duplicate_variables.add(variable)
        seen_variables.add(variable)

        for field in REQUIRED_ROW_VALUES:
            if not _present(row.get(field)):
                issues.append(f"Row {index} (Variable: '{variable or 'unknown'}') is missing value for {field}.")

        vartype = str(row.get("VARTYPE", "")).strip().upper()
        if vartype and vartype not in ACCEPTED_DATATYPES:
            issues.append(f"Row {index} (Variable: '{variable or 'unknown'}') has invalid VARTYPE '{vartype}'.")
        domain = str(row.get("DOMAIN", "")).strip().casefold()
        if domain and domain not in ACCEPTED_DOMAINS:
            issues.append(f"Row {index} (Variable: '{variable or 'unknown'}') has invalid DOMAIN '{domain}'.")

        for column in (header for header in headers if "OMOP ID" in header):
            for value in _split_present(row.get(column)):
                try:
                    int(value)
                except ValueError:
                    issues.append(
                        f"Row {index} (Variable: '{variable or 'unknown'}'): "
                        f"value '{value}' in {column} must be an integer."
                    )

        for single_column in (
            "VARIABLE CONCEPT CODE",
            "VARIABLE OMOP ID",
            "UNIT CONCEPT CODE",
            "UNIT OMOP ID",
            "VISIT CONCEPT CODE",
            "VISIT OMOP ID",
        ):
            if "|" in str(row.get(single_column, "")):
                issues.append(
                    f"Row {index} (Variable: '{variable or 'unknown'}'): "
                    f"multiple values are not allowed in {single_column}."
                )

        for prefix in ("ADDITIONAL CONTEXT", "CATEGORICAL VALUE"):
            counts = [
                len(_split_present(row.get(f"{prefix} CONCEPT NAME"))),
                len(_split_present(row.get(f"{prefix} CONCEPT CODE"))),
                len(_split_present(row.get(f"{prefix} OMOP ID"))),
            ]
            if any(counts) and len(set(counts)) != 1:
                issues.append(
                    f"Row {index} (Variable: '{variable or 'unknown'}'): "
                    f"{prefix} concept name, code, and OMOP ID counts must match."
                )

    if duplicate_variables:
        issues.insert(0, f"Duplicate VARIABLENAME found: {', '.join(sorted(duplicate_variables))}")
    return issues


def validate_dictionary_upload(
    cohort_id: str,
    content: bytes,
    concept_provider: ConceptValidationProvider,
    work_dir: Path,
) -> DictionaryValidationResult:
    """Validate an upload entirely in temporary state and return normalized bytes."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise InvalidDictionary(("The uploaded dictionary must be UTF-8 encoded.",)) from error

    normalized = normalize_dictionary_headers(text)
    issues = tuple(validate_dictionary_schema(normalized))
    if issues:
        raise InvalidDictionary(issues)

    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dictionary-validation-", dir=work_dir) as temporary:
        temporary_dir = Path(temporary)
        dictionary_path = temporary_dir / f"{cohort_id}_datadictionary.csv"
        report_path = temporary_dir / "concept-validation.csv"
        normalized_bytes = normalized.encode("utf-8")
        dictionary_path.write_bytes(normalized_bytes)
        try:
            concepts_valid = bool(concept_provider.validate(dictionary_path, report_path))
        except Exception as error:
            raise InvalidDictionary((f"Concept validation failed: {error}",)) from error
        if not concepts_valid:
            raise InvalidDictionary(("Concept validation failed",))

    return DictionaryValidationResult(
        cohort_id=cohort_id,
        normalized_csv=normalized_bytes,
        syntax_issues=(),
        concepts_valid=True,
    )
