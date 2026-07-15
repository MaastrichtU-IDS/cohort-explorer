"""Dependency-aware health endpoint for local and deployed runtimes."""

from typing import Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import Settings


async def _is_available(client: httpx.AsyncClient, url: str) -> bool:
    try:
        response = await client.get(url)
        return response.is_success
    except (httpx.HTTPError, ValueError):
        return False


def build_health_router(
    app_settings: Settings,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> APIRouter:
    """Build a router whose probes use the provided runtime settings."""
    router = APIRouter()

    @router.get("/health")
    async def health() -> JSONResponse:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(2.0),
            follow_redirects=False,
        ) as client:
            sparql_ready = await _is_available(
                client,
                f"{app_settings.sparql_endpoint.rstrip('/')}/",
            )

            dependencies: dict[str, dict[str, str]] = {
                "sparql": {"status": "ok" if sparql_ready else "unavailable"},
            }
            ready = sparql_ready

            if app_settings.dcr_backend == "aadcrv2":
                aadcr_ready = await _is_available(
                    client,
                    f"{app_settings.aadcrv2_url.rstrip('/')}/health",
                )
                dependencies["dcr_backend"] = {
                    "status": "ok" if aadcr_ready else "unavailable",
                    "provider": "aadcrv2",
                }
                ready = ready and aadcr_ready
            else:
                dependencies["dcr_backend"] = {
                    "status": "configured",
                    "provider": app_settings.dcr_backend,
                }

        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "degraded",
                "dependencies": dependencies,
            },
        )

    return router
