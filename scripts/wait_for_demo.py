#!/usr/bin/env python3
"""Wait for each local demo dependency and name failures precisely."""

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_PROBES = (
    "frontend=http://127.0.0.1:3001/api/health",
    "backend=http://127.0.0.1:3000/health",
    "aadcrv2=http://127.0.0.1:18000/health",
    "aadcrv2-ui=http://127.0.0.1:3002/healthz",
)
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Probe:
    name: str
    url: str


def parse_probe(value: str) -> Probe:
    name, separator, url = value.partition("=")
    parsed = urlparse(url)
    if not separator or not name or parsed.scheme not in {"http", "https"}:
        raise argparse.ArgumentTypeError("probe must be NAME=http://loopback:port/path")
    if parsed.hostname not in LOOPBACK_HOSTS:
        raise argparse.ArgumentTypeError("demo readiness probes must use a loopback host")
    return Probe(name=name, url=url)


def probe_once(probe: Probe) -> tuple[bool, str]:
    request = Request(  # noqa: S310 - parse_probe restricts URLs to HTTP loopback
        probe.url,
        headers={"Accept": "application/json, text/html"},
    )
    try:
        with urlopen(request, timeout=2.0) as response:  # noqa: S310 - loopback is enforced above
            response.read(1024)
            return 200 <= response.status < 400, f"HTTP {response.status}"
    except HTTPError as error:
        return False, f"HTTP {error.code}"
    except (URLError, TimeoutError, OSError) as error:
        return False, error.__class__.__name__


def wait_for_probes(probes: Sequence[Probe], timeout: float, interval: float) -> bool:
    deadline = time.monotonic() + timeout
    pending = {probe.name: probe for probe in probes}
    reasons: dict[str, str] = {}

    while pending:
        for name, probe in tuple(pending.items()):
            ready, reason = probe_once(probe)
            if ready:
                print(f"ready: {name}")
                pending.pop(name)
                reasons.pop(name, None)
            else:
                reasons[name] = reason
        if not pending:
            return True
        if time.monotonic() >= deadline:
            for name in sorted(pending):
                print(
                    f"not ready: {name} ({reasons.get(name, 'unknown')})",
                    file=sys.stderr,
                )
            return False
        time.sleep(max(0.0, interval))
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--probe", action="append", type=parse_probe)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    probes = args.probe or [parse_probe(value) for value in DEFAULT_PROBES]
    return 0 if wait_for_probes(probes, args.timeout, args.interval) else 1


if __name__ == "__main__":
    raise SystemExit(main())
