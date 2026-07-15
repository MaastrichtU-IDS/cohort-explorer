#!/usr/bin/env python3
"""Seed only Cohort Explorer metadata into a running local demo."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

COHORTS = ("GISSI-HF", "TIME-CHF")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SeedError(RuntimeError):
    """A local seed step failed without exposing response content."""


def default_pack_path(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve the active namespace's pack without loading its secrets."""
    explicit = os.getenv("DEMO_PACK_HOST_DIR")
    if explicit:
        return Path(explicit).expanduser()

    namespace = os.getenv("COMPOSE_PROJECT_NAME", "cohort-explorer-aadcr-demo")
    state_root = Path(os.getenv("DEMO_STATE_ROOT", project_root / ".demo-state"))
    runtime_env = state_root / namespace / "runtime.env"
    if runtime_env.is_file():
        for line in runtime_env.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key == "DEMO_PACK_HOST_DIR" and value:
                return Path(value).expanduser()
    return project_root / "data" / "synthetic-demo-pack"


def _one_asset(pack: Path, filename: str) -> Path:
    matches = sorted(path for path in pack.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise SeedError(f"expected exactly one {filename} in the synthetic pack; found {len(matches)}")
    return matches[0]


def _check_response(response: httpx.Response, step: str) -> None:
    if not response.is_success:
        raise SeedError(f"{step} failed with HTTP {response.status_code}")


def seed_demo(
    pack: Path,
    *,
    base_url: str,
    central_only: bool = False,
    transport: Optional[httpx.BaseTransport] = None,
) -> dict[str, object]:
    """Authenticate locally, then upload the workbook and optional dictionaries."""
    pack = pack.expanduser().resolve()
    if not pack.is_dir():
        raise SeedError(f"synthetic pack directory does not exist: {pack}")
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
        raise SeedError("demo seeding is restricted to an HTTP loopback URL")

    workbook = _one_asset(pack, "iCARE4CVD_Cohorts.xlsx")
    dictionaries = (
        {cohort: _one_asset(pack, f"{cohort}_datadictionary.csv") for cohort in COHORTS} if not central_only else {}
    )

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        transport=transport,
        follow_redirects=False,
        timeout=httpx.Timeout(30.0),
        trust_env=False,
    ) as client:
        login = client.get("/login")
        if login.status_code not in {302, 303, 307, 308} or not client.cookies.get("token"):
            raise SeedError(f"local login failed with HTTP {login.status_code}")

        with workbook.open("rb") as handle:
            response = client.post(
                "/upload-cohorts-metadata",
                files={
                    "cohorts_metadata": (
                        workbook.name,
                        handle,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
            )
        _check_response(response, "central workbook upload")

        for cohort, dictionary in dictionaries.items():
            with dictionary.open("rb") as handle:
                response = client.post(
                    "/upload-cohort",
                    data={"cohort_id": cohort},
                    files={"cohort_dictionary": (dictionary.name, handle, "text/csv")},
                )
            _check_response(response, f"{cohort} dictionary upload")

    return {
        "central_workbook": workbook.name,
        "dictionaries": list(dictionaries),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("DEMO_BASE_URL", "http://127.0.0.1:3000"),
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=default_pack_path(),
    )
    parser.add_argument("--central-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = seed_demo(
            args.pack,
            base_url=args.base_url,
            central_only=args.central_only,
        )
    except SeedError as error:
        print(f"seed error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
