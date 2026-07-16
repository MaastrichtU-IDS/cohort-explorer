"""Deterministic synthetic TIME-CHF/GISSI-HF demo pack generation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook

from src.demo.eda import write_eda_assets
from src.demo.manifest import (
    CohortPackPaths,
    DemoManifest,
    DemoPackError,
    FileDigest,
    manifest_bytes,
    validate_demo_pack,
)
from src.demo.profiles import (
    CATEGORICAL_ENCODINGS,
    COHORT_PROFILES,
    SELECTED_BINDINGS,
    CohortProfile,
)
from src.dictionary_validation import REQUIRED_DICTIONARY_COLUMNS

GENERATOR_VERSION = "1.1.0"
MAX_ROWS_PER_COHORT = 100_000
PACK_MARKER_NAME = ".cohort-explorer-synthetic-demo-pack"
PACK_MARKER_BYTES = b"cohort-explorer-synthetic-demo-pack-v1\n"
BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND_ROOT.parent
MAPPING_SOURCE = BACKEND_ROOT / "CohortVarLinker" / "mapping_output" / "time-chf_gissi-hf_full.csv"
MAPPING_PROFILE = BACKEND_ROOT / "demo" / "metadata-fixtures" / "mapping-profile.json"
MAPPING_REPOSITORY_PATH = "backend/CohortVarLinker/mapping_output/time-chf_gissi-hf_full.csv"
FIXED_WORKBOOK_TIME = datetime(2000, 1, 1, 0, 0, 0)
VARIABLE_CONCEPT_LABEL_OVERRIDES = {
    ("loinc:8867-4", "3027018"): "Heart rate measurement",
}
SOURCE_LABELS = {
    "EHR": "Electronic Health Record",
    "CRF": "Case Report Form",
    "REGISTRY": "Outcome Registry",
}
CRF_SEMANTICS = {
    "nyha_class",
    "furosemide_exposed",
    "furosemide_dose",
    "spironolactone_exposed",
    "spironolactone_dose",
}
SHARED_DEMOGRAPHIC_SEMANTICS = {"patient_id", "age", "sex"}
DEMO_DICTIONARY_COLUMNS = (
    *REQUIRED_DICTIONARY_COLUMNS,
    "SOURCENAME",
    "SOURCE LABEL",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _subseed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _logistic(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def _mask(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.where(observed, values.astype(float), np.nan)


def _git_directory(repository_root: Path) -> Path | None:
    marker = repository_root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        content = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if content.startswith(prefix):
            path = Path(content[len(prefix) :])
            return path if path.is_absolute() else (repository_root / path).resolve()
    return None


def _source_commit(repository_root: Path = REPOSITORY_ROOT) -> str:
    override = os.getenv("DEMO_SOURCE_COMMIT", "").strip()
    if override:
        return override
    git_directory = _git_directory(repository_root)
    if git_directory is None:
        return "unknown"
    try:
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if not head.startswith("ref: "):
        return head or "unknown"
    reference = head[len("ref: ") :]
    search_directories = [git_directory]
    try:
        common_reference = (git_directory / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        common_reference = ""
    if common_reference:
        common_directory = Path(common_reference)
        if not common_directory.is_absolute():
            common_directory = (git_directory / common_directory).resolve()
        search_directories.append(common_directory)

    for directory in search_directories:
        try:
            return (directory / reference).read_text(encoding="utf-8").strip()
        except OSError:
            continue
    for directory in search_directories:
        try:
            packed = (directory / "packed-refs").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in packed:
            commit, separator, name = line.partition(" ")
            if separator and name == reference:
                return commit
    return "unknown"


def _semantic_rows(seed: int, cohort_id: str, row_count: int) -> dict[tuple[str, str], np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(_subseed(seed, cohort_id)))
    severity = rng.normal(0.0, 1.0, row_count)
    age = np.clip(rng.normal(72.0 + 2.3 * severity, 8.5, row_count), 40, 95)
    sex = rng.binomial(1, 0.62, row_count)
    diabetes = rng.binomial(1, _logistic(-0.45 + 0.45 * severity + 0.025 * (age - 70)))
    hypertension = rng.binomial(1, _logistic(0.55 + 0.30 * severity + 0.025 * (age - 70)))
    smoking = np.select(
        [rng.random(row_count) < 0.20, rng.random(row_count) < 0.48],
        [2, 1],
        default=0,
    ).astype(int)

    height = np.clip(rng.normal(163.0 + 12.0 * sex, 7.0, row_count), 145, 198)
    bmi = np.clip(rng.normal(27.0 + 0.9 * severity + 0.7 * diabetes, 3.4, row_count), 17, 48)
    weight = np.clip(bmi * np.square(height / 100.0), 35, 220)
    pressure_component = rng.normal(0, 8, row_count) + 3.0 * hypertension
    systolic = np.clip(125 + pressure_component - 2.0 * severity + rng.normal(0, 8, row_count), 80, 210)
    diastolic = np.clip(72 + 0.45 * pressure_component - 1.0 * severity + rng.normal(0, 5, row_count), 45, 125)
    heart_rate = np.clip(70 + 7.5 * severity + rng.normal(0, 8, row_count), 40, 135)

    nyha_latent = severity + rng.normal(0, 0.38, row_count)
    nyha = np.digitize(nyha_latent, (-0.75, 0.05, 0.85)) + 1
    ejection_fraction = np.clip(
        49 - 8.5 * severity - 1.5 * (nyha - 2) + rng.normal(0, 4.0, row_count),
        15,
        75,
    )
    nt_pro_bnp = np.clip(
        np.exp(6.25 + 0.72 * severity + 0.15 * (nyha - 2) + rng.normal(0, 0.34, row_count)),
        40,
        30000,
    )
    creatinine = np.clip(1.0 + 0.22 * severity + 0.012 * (age - 70) + rng.normal(0, 0.18, row_count), 0.45, 4.5)
    hemoglobin = np.clip(13.5 - 0.45 * severity - 0.45 * (1 - sex) + rng.normal(0, 1.0, row_count), 7.0, 18.5)

    furosemide_exposed = rng.binomial(1, _logistic(0.15 + 1.25 * severity + 0.35 * (nyha - 2)))
    furosemide_dose = np.where(
        furosemide_exposed,
        np.clip(
            np.round((45 + 32 * (severity + 1.0) + 15 * (nyha - 2) + rng.normal(0, 18, row_count)) / 20) * 20,
            20,
            240,
        ),
        0,
    )
    spironolactone_exposed = rng.binomial(1, _logistic(-0.45 + 0.85 * severity))
    spironolactone_dose = np.where(
        spironolactone_exposed,
        np.where(rng.random(row_count) < 0.72, 25, 50),
        0,
    )
    hospitalization = rng.binomial(
        1,
        _logistic(-1.25 + 1.15 * severity + 0.45 * (nyha - 2)),
    )

    dropout_3m = rng.random(row_count) < _logistic(-2.1 + 0.45 * severity)
    observed_3m = ~dropout_3m
    dropout_after_3m = rng.random(row_count) < _logistic(-1.55 + 0.50 * severity)
    observed_1y = observed_3m & ~dropout_after_3m
    response = 0.18 * furosemide_exposed + rng.normal(0, 0.32, row_count)

    weight_3m = np.clip(weight - 0.8 * response + rng.normal(0, 1.8, row_count), 35, 220)
    weight_1y = np.clip(weight_3m - 0.5 * response + rng.normal(0, 2.2, row_count), 35, 220)
    systolic_3m = np.clip(0.78 * systolic + 0.22 * (122 - 2 * severity) + rng.normal(0, 6, row_count), 80, 210)
    systolic_1y = np.clip(0.72 * systolic_3m + 0.28 * (121 - 2 * severity) + rng.normal(0, 6, row_count), 80, 210)
    diastolic_3m = np.clip(0.80 * diastolic + 0.20 * (71 - severity) + rng.normal(0, 4, row_count), 45, 125)
    diastolic_1y = np.clip(0.75 * diastolic_3m + 0.25 * (70 - severity) + rng.normal(0, 4, row_count), 45, 125)
    creatinine_3m = np.clip(0.82 * creatinine + 0.18 * (1.0 + 0.2 * severity) + rng.normal(0, 0.12, row_count), 0.45, 4.5)
    creatinine_1y = np.clip(0.80 * creatinine_3m + 0.20 * (1.0 + 0.22 * severity) + rng.normal(0, 0.14, row_count), 0.45, 4.5)
    nyha_3m = np.clip(np.rint(nyha - response), 1, 4)
    nyha_1y = np.clip(np.rint(nyha_3m - 0.55 * response + rng.normal(0, 0.28, row_count)), 1, 4)

    furosemide_exposed_3m = (
        (furosemide_exposed == 1) | (rng.random(row_count) < _logistic(-0.8 + severity))
    ).astype(int)
    furosemide_dose_3m = np.where(
        furosemide_exposed_3m,
        np.clip(furosemide_dose + np.round(rng.normal(0, 20, row_count) / 20) * 20, 20, 240),
        0,
    )
    furosemide_exposed_1y = (
        (furosemide_exposed_3m == 1) | (rng.random(row_count) < _logistic(-1.0 + severity))
    ).astype(int)
    furosemide_dose_1y = np.where(
        furosemide_exposed_1y,
        np.clip(furosemide_dose_3m + np.round(rng.normal(0, 20, row_count) / 20) * 20, 20, 240),
        0,
    )

    return {
        ("patient_id", "baseline"): np.arange(1, row_count + 1),
        ("age", "baseline"): age,
        ("sex", "baseline"): sex,
        ("diabetes", "baseline"): diabetes,
        ("hypertension", "baseline"): hypertension,
        ("smoking", "baseline"): smoking,
        ("systolic_pressure", "baseline"): systolic,
        ("systolic_pressure", "3m"): _mask(systolic_3m, observed_3m),
        ("systolic_pressure", "1y"): _mask(systolic_1y, observed_1y),
        ("diastolic_pressure", "baseline"): diastolic,
        ("diastolic_pressure", "3m"): _mask(diastolic_3m, observed_3m),
        ("diastolic_pressure", "1y"): _mask(diastolic_1y, observed_1y),
        ("heart_rate", "baseline"): heart_rate,
        ("weight", "baseline"): weight,
        ("weight", "3m"): _mask(weight_3m, observed_3m),
        ("weight", "1y"): _mask(weight_1y, observed_1y),
        ("height", "baseline"): height,
        ("ejection_fraction", "baseline"): ejection_fraction,
        ("nt_pro_bnp", "baseline"): nt_pro_bnp,
        ("creatinine", "baseline"): creatinine,
        ("creatinine", "3m"): _mask(creatinine_3m, observed_3m),
        ("creatinine", "1y"): _mask(creatinine_1y, observed_1y),
        ("hemoglobin", "baseline"): hemoglobin,
        ("nyha_class", "baseline"): nyha,
        ("nyha_class", "3m"): _mask(nyha_3m, observed_3m),
        ("nyha_class", "1y"): _mask(nyha_1y, observed_1y),
        ("furosemide_exposed", "baseline"): furosemide_exposed,
        ("furosemide_dose", "baseline"): furosemide_dose,
        ("furosemide_exposed", "3m"): _mask(furosemide_exposed_3m, observed_3m),
        ("furosemide_dose", "3m"): _mask(furosemide_dose_3m, observed_3m),
        ("furosemide_exposed", "1y"): _mask(furosemide_exposed_1y, observed_1y),
        ("furosemide_dose", "1y"): _mask(furosemide_dose_1y, observed_1y),
        ("spironolactone_exposed", "baseline"): spironolactone_exposed,
        ("spironolactone_dose", "baseline"): spironolactone_dose,
        ("heart_failure_hospitalization", "baseline"): hospitalization,
    }


def _load_mapping_rows() -> tuple[list[dict[str, str]], bytes, dict[str, Any]]:
    try:
        content = MAPPING_SOURCE.read_bytes()
        profile = json.loads(MAPPING_PROFILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DemoPackError(f"Could not load tracked mapping provenance: {error}") from error
    if _sha256(content) != profile.get("source_sha256"):
        raise DemoPackError("Tracked mapping hash does not match mapping-profile.json")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    lookup = {(row["source"], row["target"]): row for row in rows}
    selected: list[dict[str, str]] = []
    for binding in SELECTED_BINDINGS:
        row = lookup.get((binding.source, binding.target))
        if row is None:
            raise DemoPackError(
                f"Tracked mapping row is missing: {binding.source} -> {binding.target}"
            )
        selected.append(row)
    return selected, content, profile


def _project_rows(
    profile: CohortProfile,
    semantic_rows: dict[tuple[str, str], np.ndarray],
    mapping_rows: list[dict[str, str]],
) -> pd.DataFrame:
    columns: dict[str, Any] = {}
    prefix = "source" if profile.side == "source" else "target"
    for binding, mapping_row in zip(profile.bindings, mapping_rows):
        values = semantic_rows[(binding.semantic, binding.visit)]
        column_name = binding.source if profile.side == "source" else binding.target
        data_type = mapping_row.get(f"{prefix}_data_type", "").casefold()
        if binding.semantic == "patient_id" and profile.side == "source":
            columns[column_name] = [f"TIMECHF-{int(value):06d}" for value in values]
        elif data_type == "int":
            numeric = pd.Series(values).round()
            columns[column_name] = numeric.astype("Int64") if numeric.isna().any() else numeric.astype(int)
        else:
            columns[column_name] = values.astype(float)
    return pd.DataFrame(columns, columns=profile.columns)


def _value_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    return str(value)


def _source_metadata(semantic: str) -> tuple[str, str]:
    if semantic == "heart_failure_hospitalization":
        sources = ("EHR", "REGISTRY")
    elif semantic in SHARED_DEMOGRAPHIC_SEMANTICS:
        sources = ("EHR", "CRF")
    elif semantic in CRF_SEMANTICS:
        sources = ("CRF",)
    else:
        sources = ("EHR",)
    return " | ".join(sources), " | ".join(SOURCE_LABELS[source] for source in sources)


def _dictionary_frame(
    profile: CohortProfile,
    rows: pd.DataFrame,
    mapping_rows: list[dict[str, str]],
) -> pd.DataFrame:
    prefix = "s" if profile.side == "source" else "t"
    type_prefix = "source" if profile.side == "source" else "target"
    dictionary_rows: list[dict[str, Any]] = []
    for binding, mapping_row in zip(profile.bindings, mapping_rows):
        variable_name = binding.source if profile.side == "source" else binding.target
        series = rows[variable_name]
        observed = series.dropna()
        missing_count = int(series.isna().sum())
        source_data_type = mapping_row.get(f"{type_prefix}_data_type", "").upper()
        vartype = source_data_type if source_data_type in {"STR", "FLOAT", "INT", "DATETIME"} else "FLOAT"
        domain = mapping_row.get("category", "").split("|", 1)[0] or "observation"
        concept_code = mapping_row[f"{prefix}code"]
        concept_omop_id = mapping_row[f"{prefix}omop_id"]
        concept_name = VARIABLE_CONCEPT_LABEL_OVERRIDES.get(
            (concept_code, concept_omop_id),
            mapping_row[f"{prefix}label"],
        )
        source_name, source_label = _source_metadata(binding.semantic)
        dictionary_rows.append(
            {
                "VARIABLENAME": variable_name,
                "VARIABLELABEL": mapping_row[f"{prefix}label"],
                "VARTYPE": vartype,
                "UNITS": mapping_row.get(f"{type_prefix}_unit", ""),
                "CATEGORICAL": CATEGORICAL_ENCODINGS.get(binding.semantic, ""),
                "MISSING": f"{100 * missing_count / len(series):.2f}%",
                "COUNT": int(series.notna().sum()),
                "NA": missing_count,
                "MIN": _value_text(observed.min()) if not observed.empty else "",
                "MAX": _value_text(observed.max()) if not observed.empty else "",
                "FORMULA": "",
                "CATEGORICAL VALUE CONCEPT CODE": "",
                "CATEGORICAL VALUE CONCEPT NAME": "",
                "CATEGORICAL VALUE OMOP ID": "",
                "VARIABLE CONCEPT CODE": concept_code,
                "VARIABLE CONCEPT NAME": concept_name,
                "VARIABLE OMOP ID": concept_omop_id,
                "ADDITIONAL CONTEXT CONCEPT NAME": "",
                "ADDITIONAL CONTEXT CONCEPT CODE": "",
                "ADDITIONAL CONTEXT OMOP ID": "",
                "UNIT CONCEPT NAME": "",
                "UNIT CONCEPT CODE": "",
                "UNIT OMOP ID": "",
                "DOMAIN": domain,
                "VISITS": mapping_row[f"{type_prefix}_visit"],
                "VISIT OMOP ID": "",
                "VISIT CONCEPT NAME": "",
                "VISIT CONCEPT CODE": "",
                "SOURCENAME": source_name,
                "SOURCE LABEL": source_label,
            }
        )
    return pd.DataFrame(dictionary_rows, columns=DEMO_DICTIONARY_COLUMNS)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    output = io.StringIO(newline="")
    frame.to_csv(
        output,
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.6f",
    )
    return output.getvalue().encode("utf-8")


def _normalized_workbook_bytes(row_count: int) -> bytes:
    headers = (
        "Study name",
        "Institute",
        "Study type",
        "Study design",
        "Number of participants",
        "Study population",
        "Study duration",
        "Ongoing",
        "Study objective",
        "Primary outcome specification",
        "Secondary outcome specification",
        "Morbidity",
        "Start date",
        "End date",
        "Mixed Sex",
        "Age Distribution",
        "Administrator",
        "Administrator email address",
        "Study contact person",
        "Study contact person email address",
        "References",
        "Population location",
        "Language",
        "Frequency of data collection",
        "Interventions",
        "Anonymisation/Pseudonymization technique",
        "Dataset format",
        "Coding system",
        "Comparator",
        "Race/Ethnicity",
        "Enrolled with diabetes",
        "Enrolled with CVD (%)",
        "Part of Study",
    )
    common = {
        "Institute": "Synthetic iCARE4CVD Demo Consortium",
        "Study type": "Observational",
        "Study design": "Cohort study",
        "Number of participants": str(row_count),
        "Study population": "Synthetic adults with chronic heart failure",
        "Study duration": "Synthetic longitudinal follow-up to one year",
        "Ongoing": "false",
        "Study objective": "Exercise the local metadata and DCR demonstration",
        "Primary outcome specification": "Heart-failure hospitalization",
        "Secondary outcome specification": "NYHA class and biomarker trajectories",
        "Morbidity": "Heart failure",
        "Start date": "2020-01-01",
        "End date": "2021-12-31",
        "Mixed Sex": "62% male; 38% female",
        "Age Distribution": "40-64: 25%; 65-79: 50%; 80+: 25%",
        "Administrator": "Nikolas Molyndris",
        "Administrator email address": "nikolas.molyndris@decentriq.ch",
        "Study contact person": "Nikolas Molyndris",
        "Study contact person email address": "nikolas.molyndris@decentriq.ch",
        "References": "Synthetic demo data; no patient records",
        "Population location": "Europe",
        "Language": "English",
        "Frequency of data collection": "Baseline, three months, one year",
        "Interventions": "Routine heart-failure treatment (synthetic)",
        "Anonymisation/Pseudonymization technique": "Fully synthetic",
        "Dataset format": "CSV",
        "Coding system": "OMOP CDM vocabulary identifiers",
        "Comparator": "Not applicable",
        "Race/Ethnicity": "Synthetic mixed population",
        "Enrolled with diabetes": "Mixed",
        "Enrolled with CVD (%)": "100%",
        "Part of Study": "Local AADCR v2 demonstration",
    }
    cohort_metadata = {
        "TIME-CHF": {
            "Institute": "Synthetic iCARE4CVD Demo Consortium - TIME-CHF Site",
            "Study design": "Prospective synthetic cohort study",
        },
        "GISSI-HF": {
            "Institute": "Synthetic iCARE4CVD Demo Consortium - GISSI-HF Site",
            "Study design": "Synthetic registry cohort study",
        },
    }

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Descriptions"
    sheet.append(headers)
    for cohort_id in ("TIME-CHF", "GISSI-HF"):
        values = {**common, **cohort_metadata[cohort_id], "Study name": cohort_id}
        sheet.append([values[header] for header in headers])
    workbook.properties.creator = "Cohort Explorer deterministic demo generator"
    workbook.properties.lastModifiedBy = "Cohort Explorer deterministic demo generator"
    workbook.properties.created = FIXED_WORKBOOK_TIME
    workbook.properties.modified = FIXED_WORKBOOK_TIME
    workbook.properties.title = "Synthetic iCARE4CVD cohort descriptions"

    raw = io.BytesIO()
    workbook.save(raw)
    normalized = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(raw.getvalue()), "r") as source,
        zipfile.ZipFile(
            normalized,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as target,
    ):
        for name in sorted(source.namelist()):
            content = source.read(name)
            if name == "docProps/core.xml":
                content = re.sub(
                    rb"<dcterms:modified[^>]*>[^<]*</dcterms:modified>",
                    (
                        b'<dcterms:modified xsi:type="dcterms:W3CDTF">'
                        b"2000-01-01T00:00:00Z</dcterms:modified>"
                    ),
                    content,
                )
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            target.writestr(info, content)
    return normalized.getvalue()


def _write_shuffled_sample(
    rows: pd.DataFrame,
    profile: CohortProfile,
    seed: int,
    output_dir: Path,
) -> tuple[str, str]:
    patient_id = profile.column("patient_id")
    shuffled = rows.drop(columns=[patient_id]).copy()
    for column in shuffled.columns:
        present = shuffled[column].notna()
        values = shuffled.loc[present, column].to_numpy(copy=True)
        rng = np.random.Generator(np.random.PCG64(_subseed(seed, f"{profile.cohort_id}:{column}:shuffle")))
        rng.shuffle(values)
        shuffled.loc[present, column] = values
    sample_size = min(500, max(1, int(len(shuffled) * 0.20)))
    sample_rng = np.random.Generator(
        np.random.PCG64(_subseed(seed, f"{profile.cohort_id}:sample"))
    )
    selected = sample_rng.permutation(len(shuffled))[:sample_size]
    sample = shuffled.iloc[selected].reset_index(drop=True)
    sample.insert(0, "Synthetic_ID", [f"SYNTH_{index:05d}" for index in range(1, sample_size + 1)])
    sample_relative = f"dcr_output_{profile.cohort_id}/shuffled_sample.csv"
    summary_relative = f"dcr_output_{profile.cohort_id}/shuffle_summary.txt"
    (output_dir / "shuffled_sample.csv").write_bytes(_csv_bytes(sample))
    (output_dir / "shuffle_summary.txt").write_text(
        "\n".join(
            (
                "DATA SHUFFLE COMPLETE",
                "=====================",
                f"Seed: {seed}",
                f"Original: {len(rows):,} rows x {len(rows.columns)} columns",
                "PII removed: patient identifier",
                f"Retained: {len(shuffled.columns)} columns",
                f"Output sample: {sample_size:,} rows",
                "Privacy method: independent deterministic column shuffling",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    return sample_relative, summary_relative


def _file_digests(root: Path) -> dict[str, FileDigest]:
    files: dict[str, FileDigest] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"):
        content = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        files[relative] = FileDigest(sha256=_sha256(content), size_bytes=len(content))
    return files


def _selected_mapping_projection(mapping_rows: list[dict[str, str]]) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "source": row["source"],
            "target": row["target"],
            "mapping_type": row["mapping type"],
            "source_visit": row["source_visit"],
            "target_visit": row["target_visit"],
        }
        for row in mapping_rows
    )


def _populate_pack(root: Path, *, seed: int, row_count: int) -> DemoManifest:
    (root / PACK_MARKER_NAME).write_bytes(PACK_MARKER_BYTES)
    mapping_rows, mapping_content, mapping_profile = _load_mapping_rows()
    cohort_records: dict[str, CohortPackPaths] = {}
    for cohort_id, profile in COHORT_PROFILES.items():
        semantic_rows = _semantic_rows(seed, cohort_id, row_count)
        rows = _project_rows(profile, semantic_rows, mapping_rows)
        dictionary = _dictionary_frame(profile, rows, mapping_rows)
        dictionary_relative = f"cohorts/{cohort_id}/{cohort_id}_datadictionary.csv"
        rows_relative = f"dcr-input/{cohort_id}.csv"
        dictionary_path = root / dictionary_relative
        rows_path = root / rows_relative
        dictionary_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary_path.write_bytes(_csv_bytes(dictionary))
        rows_path.write_bytes(_csv_bytes(rows))

        output_dir = root / f"dcr_output_{cohort_id}"
        eda_relative, images = write_eda_assets(rows, dictionary, cohort_id, output_dir)
        sample_relative, summary_relative = _write_shuffled_sample(
            rows,
            profile,
            seed,
            output_dir,
        )
        cohort_records[cohort_id] = CohortPackPaths(
            row_count=row_count,
            dictionary=dictionary_relative,
            rows=rows_relative,
            eda=eda_relative,
            shuffled_sample=sample_relative,
            shuffle_summary=summary_relative,
            images=images,
        )

    workbook_relative = "iCARE4CVD_Cohorts.xlsx"
    (root / workbook_relative).write_bytes(_normalized_workbook_bytes(row_count))
    manifest = DemoManifest(
        schema_version=1,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        source_commit=_source_commit(),
        mapping_source={
            "repository_path": MAPPING_REPOSITORY_PATH,
            "source_commit": str(mapping_profile.get("source_commit", "unknown")),
            "sha256": _sha256(mapping_content),
        },
        workbook_relative=workbook_relative,
        cohorts=cohort_records,
        selected_mapping_rows=_selected_mapping_projection(mapping_rows),
        files=_file_digests(root),
        root=root,
    )
    (root / "manifest.json").write_bytes(manifest_bytes(manifest))
    return validate_demo_pack(root)


def _validate_force_target(output: Path) -> None:
    marker = output / PACK_MARKER_NAME
    try:
        marker_content = marker.read_bytes()
    except OSError as error:
        raise DemoPackError(
            f"Refusing to force-replace a directory that is not a generated demo pack: {output}"
        ) from error
    if marker_content != PACK_MARKER_BYTES:
        raise DemoPackError(
            f"Refusing to force-replace a directory that is not a generated demo pack: {output}"
        )
    try:
        validate_demo_pack(output)
    except DemoPackError as error:
        raise DemoPackError(
            f"Refusing to force-replace an invalid generated demo pack: {output}"
        ) from error


def generate_demo_pack(
    output_dir: Path,
    seed: int,
    rows: int,
    force: bool,
) -> DemoManifest:
    """Generate, validate, and atomically install one immutable synthetic pack."""
    output = Path(output_dir).expanduser().resolve()
    if rows <= 0:
        raise DemoPackError("rows must be greater than zero")
    if rows > MAX_ROWS_PER_COHORT:
        raise DemoPackError(f"rows must be at most {MAX_ROWS_PER_COHORT}")
    if output.exists() and not output.is_dir():
        raise DemoPackError(f"Demo pack output is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        if not force:
            raise DemoPackError(f"Refusing to overwrite non-empty demo pack directory: {output}")
        _validate_force_target(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.transaction-", dir=output.parent)
    )
    staging = transaction / "new"
    backup = transaction / "previous"
    staging.mkdir()
    preserve_transaction = False
    try:
        _populate_pack(staging, seed=seed, row_count=rows)
        had_previous_output = output.exists()
        if had_previous_output:
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except Exception:
            if backup.exists():
                try:
                    os.replace(backup, output)
                except Exception as rollback_error:
                    preserve_transaction = True
                    raise DemoPackError(
                        "Demo pack installation and rollback failed; "
                        f"the previous pack is retained at {backup}"
                    ) from rollback_error
            raise
    finally:
        if not preserve_transaction:
            shutil.rmtree(transaction, ignore_errors=True)
    return validate_demo_pack(output)
