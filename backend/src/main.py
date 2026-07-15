import asyncio
import fcntl
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from src.admin import debug_router
from src.admin import router as admin_router
from src.auth import router as auth_router
from src.config import settings
from src.data_analysis import router as data_analysis_router
from src.dcr_backends.factory import get_dcr_backend
from src.dcr_routes import router as dcr_router
from src.decentriq import router as decentriq_router
from src.docs import router as docs_router
from src.explore import router as explore_router
from src.health import build_health_router
from src.mapping import router as mapping_router
from src.metadata_reports import build_syntax_report
from src.upload import init_triplestore
from src.upload import router as upload_router

settings.validate_runtime()
init_triplestore()
#asyncio.create_task(run_periodic_monitoring())


def _refresh_dcr_history_with_lock() -> None:
    """Run the DCR history refresh under a non-blocking file lock so that, in
    multi-worker deployments, only one worker actually performs the SDK-heavy
    refresh. Other workers see ``LOCK_EX | LOCK_NB`` fail and skip silently.
    """
    backend = get_dcr_backend()
    if not backend.capabilities.supports_room_refresh:
        logging.info(
            "Skipping startup DCR refresh: provider %s does not support it.",
            backend.provider_name,
        )
        return

    os.makedirs(settings.data_folder, exist_ok=True)
    lock_path = os.path.join(settings.data_folder, ".dcr_refresh.lock")
    try:
        with open(lock_path, "w") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logging.info(
                    "Worker %s skipping DCR refresh: another worker holds the lock.",
                    os.getpid(),
                )
                return
            logging.info("Worker %s acquired DCR refresh lock", os.getpid())
            try:
                service_user = {
                    "email": settings.decentriq_email or settings.local_auth_email,
                }
                result = asyncio.run(backend.list_rooms(service_user, refresh=True))
                logging.info("Startup DCR refresh summary: %s", result.to_dict())
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        logging.warning("Startup DCR refresh failed: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: schedule background work on startup, then yield."""
    async def _runner() -> None:
        try:
            await asyncio.to_thread(_refresh_dcr_history_with_lock)
        except Exception as exc:  # pragma: no cover - already logged inside
            logging.warning("DCR refresh task crashed: %s", exc)

    async def _report_runner() -> None:
        try:
            report = await asyncio.to_thread(build_syntax_report)
            logging.info("Startup metadata syntax report generated: %s", report.path)
        except Exception as exc:
            logging.warning("Startup metadata syntax report failed: %s", exc)

    _app.state.refresh_task = asyncio.create_task(_runner())
    _app.state.metadata_report_task = asyncio.create_task(_report_runner())
    yield


app = FastAPI(
    title="iCARE4CVD API",
    description="""Upload and explore cohorts metadata files for the [iCARE4CVD project](https://icare4cvd.eu/).""",
    lifespan=lifespan,
)

app.include_router(explore_router, tags=["explore"])
app.include_router(mapping_router, prefix="/api", tags=["mapping"])
app.include_router(data_analysis_router, prefix="/api", tags=["data-analysis"])
app.include_router(upload_router, tags=["upload"])
app.include_router(decentriq_router, tags=["upload"])
app.include_router(dcr_router, tags=["dcr"])
app.include_router(auth_router, tags=["authentication"])
app.include_router(admin_router, tags=["admin"])
if settings.dev_mode:
    app.include_router(debug_router)
app.include_router(docs_router, prefix="/docs-api", tags=["documents"])
app.include_router(build_health_router(settings), tags=["health"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the route / to /docs"""
    return RedirectResponse(url="/docs")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
