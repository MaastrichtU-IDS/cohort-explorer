from contextlib import asynccontextmanager
import asyncio
import fcntl
import logging
import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from src.auth import router as auth_router
from src.config import settings
from src.data_analysis import router as data_analysis_router
from src.decentriq import router as decentriq_router
from src.decentriq import refresh_all_dcrs_via_decentriq_api
from src.explore import router as explore_router
from src.mapping import router as mapping_router
from src.upload import init_triplestore
from src.upload import router as upload_router
from src.monitoring import run_periodic_monitoring
from src.docs import router as docs_router

init_triplestore()
#asyncio.create_task(run_periodic_monitoring())


def _refresh_dcr_history_with_lock() -> None:
    """Run the DCR history refresh under a non-blocking file lock so that, in
    multi-worker deployments, only one worker actually performs the SDK-heavy
    refresh. Other workers see ``LOCK_EX | LOCK_NB`` fail and skip silently.
    """
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
                summary = refresh_all_dcrs_via_decentriq_api()
                logging.info("Startup DCR refresh summary: %s", summary)
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

    asyncio.create_task(_runner())
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
app.include_router(auth_router, tags=["authentication"])
app.include_router(docs_router, prefix="/docs-api", tags=["documents"])


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
