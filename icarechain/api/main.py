from contextlib import asynccontextmanager

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from api.config import get_settings
from api.routes import (
    admin,
    attestations,
    auth,
    cohorts,
    commitments,
    health,
    ontology,
    providers,
    requesters,
)
from api.services.blockchain import get_blockchain_service
from api.services.cache import close_cache, configure_cache, get_cache_async
from api.services.ontology import close_ontology_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_cache(settings.cache_backend)

    try:
        cache = await get_cache_async()
    except Exception:
        configure_cache("memory")
        cache = await get_cache_async()

    try:
        service = get_blockchain_service()
        if service.w3.is_connected():
            await cache.mark_synced()
    except Exception:
        pass

    yield

    await close_ontology_service()
    await close_cache()

app = FastAPI(
    title="DUO Consent Microservice",
    description="Blockchain-backed GA4GH DUO consent management.",
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.0.3",
    )
    app.openapi_schema = schema
    return schema

app.openapi = _custom_openapi

_STATIC_DIR = Path("/app/static")
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} – Swagger UI",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
        swagger_favicon_url="/static/swagger/favicon-32x32.png",
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
        },
    )

for r in (admin, auth, providers, requesters, cohorts, attestations, commitments, ontology, health):
    app.include_router(r.router, prefix="/api")
