import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.cohort_cache import get_cohorts_from_cache
from src.config import settings
from src.dictionary_validation import normalize_dictionary_headers, validate_dictionary_schema
from src.metadata_paths import canonical_dictionary_path


@dataclass(frozen=True)
class SyntaxReport:
    path: Path
    total_cohorts: int
    cohorts_with_errors: int
    cohorts_without_dictionary: int
    total_errors: int


def build_syntax_report() -> SyntaxReport:
    """Build the metadata syntax report directly from canonical source files."""
    generated_at = datetime.now()
    reports_folder = Path(settings.data_folder) / "DICTIONARY_ISSUES_REPORTS"
    reports_folder.mkdir(parents=True, exist_ok=True)
    report_path = reports_folder / f"metadata_syntax_issues_{generated_at:%Y%m%d_%H%M%S}.txt"
    admin_email = settings.admins_list[0] if settings.admins_list else settings.local_auth_email
    cohorts = get_cohorts_from_cache(admin_email)

    total_cohorts = len(cohorts)
    cohorts_with_errors = 0
    cohorts_without_dictionary = 0
    total_errors = 0
    sections: list[str] = [
        f"Metadata Syntax Issues Report - Generated: {generated_at:%Y-%m-%d %H:%M:%S}",
        "=" * 80,
        "",
    ]

    for cohort_id in sorted(cohorts):
        try:
            dictionary_path = canonical_dictionary_path(settings.cohort_folder, cohort_id)
        except ValueError as error:
            cohorts_with_errors += 1
            total_errors += 1
            sections.extend([f"COHORT: {cohort_id}", f"Error: {error}", "-" * 80, ""])
            continue
        if not dictionary_path.is_file():
            cohorts_without_dictionary += 1
            sections.extend(
                [
                    f"COHORT: {cohort_id}",
                    "Status: No canonical metadata dictionary file found",
                    f"Expected: {dictionary_path}",
                    "-" * 80,
                    "",
                ]
            )
            continue

        try:
            normalized = normalize_dictionary_headers(dictionary_path.read_text(encoding="utf-8-sig"))
            issues = validate_dictionary_schema(normalized)
        except Exception as error:
            issues = [f"Failed to process canonical dictionary: {error}"]
        modified_at = datetime.fromtimestamp(dictionary_path.stat().st_mtime)
        sections.extend(
            [
                f"COHORT: {cohort_id}",
                f"File: {dictionary_path.name}",
                f"Dictionary date: {modified_at:%Y-%m-%d %H:%M:%S}",
            ]
        )
        if issues:
            cohorts_with_errors += 1
            total_errors += len(issues)
            sections.extend([f"Errors found: {len(issues)}", ""])
            sections.extend(f"  {index}. {issue}" for index, issue in enumerate(issues, start=1))
        else:
            sections.append("Status: OK")
        sections.extend(["-" * 80, ""])

    sections.extend(
        [
            "=" * 80,
            "SUMMARY",
            "=" * 80,
            f"Total cohorts: {total_cohorts}",
            f"Cohorts with errors: {cohorts_with_errors}",
            f"Cohorts without dictionary: {cohorts_without_dictionary}",
            f"Total errors: {total_errors}",
            f"Report generated: {generated_at:%Y-%m-%d %H:%M:%S}",
            "",
        ]
    )
    report_path.write_text("\n".join(sections), encoding="utf-8")
    logging.info(
        "Generated metadata syntax report %s (cohorts=%d, errors=%d, missing=%d)",
        report_path,
        total_cohorts,
        cohorts_with_errors,
        cohorts_without_dictionary,
    )
    return SyntaxReport(
        path=report_path,
        total_cohorts=total_cohorts,
        cohorts_with_errors=cohorts_with_errors,
        cohorts_without_dictionary=cohorts_without_dictionary,
        total_errors=total_errors,
    )
