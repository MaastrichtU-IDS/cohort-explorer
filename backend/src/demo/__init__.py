"""Deterministic, offline-only cohort demo pack generation."""

from src.demo.generator import generate_demo_pack
from src.demo.manifest import DemoManifest, DemoPackError, validate_demo_pack

__all__ = [
    "DemoManifest",
    "DemoPackError",
    "generate_demo_pack",
    "validate_demo_pack",
]
