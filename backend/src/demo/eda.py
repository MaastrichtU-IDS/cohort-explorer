"""Small deterministic EDA summaries compatible with the existing dashboard."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.demo.assets import variable_png_bytes


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count_pct(count: int, total: int) -> str:
    percentage = 100 * count / total if total else 0.0
    return f"{count} ({percentage:.2f}%)"


def _numeric_fields(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    q1 = _finite(numeric.quantile(0.25))
    q3 = _finite(numeric.quantile(0.75))
    iqr = (q3 - q1) if q1 is not None and q3 is not None else 0.0
    lower = (q1 - 1.5 * iqr) if q1 is not None else float("-inf")
    upper = (q3 + 1.5 * iqr) if q3 is not None else float("inf")
    outliers_iqr = int(((numeric < lower) | (numeric > upper)).sum())
    standard_deviation = _finite(numeric.std(ddof=1)) or 0.0
    mean = _finite(numeric.mean()) or 0.0
    outliers_z = int(((numeric - mean).abs() > 3 * standard_deviation).sum()) if standard_deviation else 0
    mode = numeric.mode()
    minimum = _finite(numeric.min())
    maximum = _finite(numeric.max())
    return {
        "mean": mean,
        "median": _finite(numeric.median()),
        "mode": _finite(mode.iloc[0]) if not mode.empty else None,
        "std dev": standard_deviation,
        "variance": _finite(numeric.var(ddof=1)),
        "min": minimum,
        "max": maximum,
        "range": (maximum - minimum) if minimum is not None and maximum is not None else None,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": _finite(numeric.skew()),
        "kurtosis": _finite(numeric.kurt()),
        "normality test": "not assessed => non-normal",
        "w_test": None,
        "outliers (iqr)": _count_pct(outliers_iqr, len(numeric)),
        "outliers (z)": outliers_z,
    }


def _categorical_fields(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str)
    counts = values.value_counts().sort_index()
    total = int(counts.sum())
    class_balance = "\n\t".join(
        f"{label} -> {(100 * count / total if total else 0):.2f}%"
        for label, count in counts.items()
    )
    return {
        "class balance": class_balance,
        "chi-square test statistic": 0.0,
        "most frequent category": str(counts.idxmax()) if not counts.empty else None,
    }


def write_eda_assets(
    rows: pd.DataFrame,
    dictionary: pd.DataFrame,
    cohort_id: str,
    output_dir: Path,
) -> tuple[str, tuple[str, ...]]:
    """Write one dashboard JSON and a lowercase PNG for every dictionary variable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_by_name = dictionary.set_index("VARIABLENAME", drop=False)
    eda: dict[str, dict[str, Any]] = {}
    images: list[str] = []

    for variable_name in dictionary["VARIABLENAME"]:
        metadata = dictionary_by_name.loc[variable_name]
        series = rows[variable_name]
        categorical = bool(str(metadata["CATEGORICAL"]).strip()) or str(metadata["VARTYPE"]).upper() == "STR"
        missing = int(series.isna().sum())
        observed = int(series.notna().sum())
        entry: dict[str, Any] = {
            "label": str(metadata["VARIABLELABEL"]),
            "type": (
                f"categorical (encoded as {series.dtype})"
                if categorical
                else f"numeric (encoded as {series.dtype})"
            ),
            "variablelabel (metadata dictionary)": str(metadata["VARIABLELABEL"]),
            "vartype (metadata dictionary)": str(metadata["VARTYPE"]),
            "units (metadata dictionary)": str(metadata["UNITS"]),
            "variable concept code (metadata dictionary)": str(metadata["VARIABLE CONCEPT CODE"]),
            "variable concept name (metadata dictionary)": str(metadata["VARIABLE CONCEPT NAME"]),
            "variable omop id (metadata dictionary)": int(metadata["VARIABLE OMOP ID"]),
            "domain (metadata dictionary)": str(metadata["DOMAIN"]),
            "visits (metadata dictionary)": str(metadata["VISITS"]),
            "visit concept name (metadata dictionary)": str(metadata["VISIT CONCEPT NAME"]),
            "count of observations (ex. missing/empty)": observed,
            "count empty": _count_pct(0, len(series)),
            "count missing": _count_pct(missing, len(series)),
            "number of unique values/categories": int(series.nunique(dropna=True)),
            "x-ticks": "",
            "y-ticks": "",
            "url": f"{variable_name.casefold()}.png",
        }
        entry.update(_categorical_fields(series) if categorical else _numeric_fields(series))
        eda[str(variable_name)] = entry

        image_name = f"{str(variable_name).casefold()}.png"
        (output_dir / image_name).write_bytes(
            variable_png_bytes(str(variable_name), series, categorical=categorical)
        )
        images.append(f"dcr_output_{cohort_id}/{image_name}")

    eda_relative = f"dcr_output_{cohort_id}/eda_output_{cohort_id}.json"
    (output_dir / f"eda_output_{cohort_id}.json").write_text(
        json.dumps(eda, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return eda_relative, tuple(images)
