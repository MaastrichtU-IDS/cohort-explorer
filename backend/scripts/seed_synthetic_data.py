"""Generate or validate the deterministic local synthetic cohort pack."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.demo.generator import generate_demo_pack
from src.demo.manifest import DemoPackError, validate_demo_pack


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Synthetic pack directory")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic root seed")
    parser.add_argument("--rows", type=int, default=2500, help="Rows generated per cohort")
    parser.add_argument("--force", action="store_true", help="Replace an existing non-empty pack")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing pack without regenerating or writing it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate:
        manifest = validate_demo_pack(args.output)
        print(f"Validated synthetic demo pack: {manifest.root}")
        return 0
    manifest = generate_demo_pack(args.output, args.seed, args.rows, args.force)
    print(
        f"Generated synthetic demo pack: {manifest.root} "
        f"({sum(record.row_count for record in manifest.cohorts.values())} total rows)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoPackError as error:
        print(f"demo pack error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
