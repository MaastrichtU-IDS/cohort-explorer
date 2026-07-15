"""Deterministic PNG creation and runtime/demo-pack asset path selection."""

from __future__ import annotations

import hashlib
import io
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def asset_root(settings: Any) -> Path:
    """Use the immutable pack only for offline assets, never for runtime state."""
    if getattr(settings, "offline_demo", False):
        return Path(settings.demo_pack_dir)
    return Path(settings.data_folder)


def validate_asset_component(value: str) -> str:
    """Reject route components that could alter or poison a filesystem path."""
    invalid = (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(unicodedata.category(character).startswith("C") for character in value)
    )
    if invalid:
        raise ValueError("Invalid asset component")
    return value


def contained_asset_path(root: Path, *parts: str) -> Path:
    """Resolve an asset path and fail if it leaves the configured root."""
    resolved_root = Path(root).resolve()
    try:
        candidate = resolved_root.joinpath(*parts).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("Asset path could not be resolved") from error
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError("Asset path escapes the asset root")
    return candidate


def _accent(name: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (60 + digest[0] % 140, 60 + digest[1] % 140, 60 + digest[2] % 140)


def variable_png_bytes(name: str, series: pd.Series, *, categorical: bool) -> bytes:
    """Render a compact chart without timestamps, external fonts, or metadata."""
    width, height = 640, 360
    margin = 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    accent = _accent(name)
    draw.rectangle((0, 0, width, 12), fill=accent)
    draw.line((margin, height - margin, width - margin, height - margin), fill=(55, 55, 55), width=2)
    draw.line((margin, margin, margin, height - margin), fill=(55, 55, 55), width=2)

    if categorical:
        counts = series.dropna().astype(str).value_counts().sort_index().to_numpy(dtype=float)
    else:
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        counts = np.histogram(values, bins=min(12, max(1, int(np.sqrt(len(values))))))[0] if len(values) else np.array([])

    if len(counts):
        maximum = max(float(counts.max()), 1.0)
        plot_width = width - 2 * margin
        gap = 4
        bar_width = max(2, (plot_width - gap * (len(counts) - 1)) // len(counts))
        for index, count in enumerate(counts):
            if count <= 0:
                continue
            bar_height = max(
                1,
                int((height - 2 * margin - 8) * float(count) / maximum),
            )
            left = margin + index * (bar_width + gap) + 2
            draw.rectangle(
                (left, height - margin - bar_height, left + bar_width, height - margin - 1),
                fill=accent,
            )

    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()
