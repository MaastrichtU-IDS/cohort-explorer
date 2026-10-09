from contextlib import asynccontextmanager
import asyncio
import logging
import os
import time

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from src.announcements import router as announcements_router
from src.auth import router as auth_router
from src.chat import router as chat_router
from src.code_explain import router as code_explain_router
from src.config import settings
from src.data_analysis import router as data_analysis_router
from src.data_analysis import bare_router as data_analysis_bare_router
from src.decentriq import router as decentriq_router
from src.dcr_results import router as dcr_results_router
from src.decentriq import refresh_dcr_history
from src.eda_counts import router as eda_counts_router
from src.explore import router as explore_router
from src.mapping import router as mapping_router
from src.upload import init_triplestore
from src.upload import router as upload_router
from src.monitoring import run_periodic_monitoring
from src.admin import router as admin_router
from src.docs import router as docs_router
from src.nocode import router as nocode_router

init_triplestore()
#asyncio.create_task(run_periodic_monitoring())


# How often the DCR history is refreshed from Decentriq in the background.
DCR_REFRESH_INTERVAL_SECONDS = 60 * 60


def _dcr_history_age_seconds() -> float | None:
    from src.decentriq import _dcr_history_path
    try:
        return time.time() - os.path.getmtime(_dcr_history_path())
    except OSError:
        return None


async def _refresh_dcr_history_periodically() -> None:
    """Refresh the DCR history at startup and then about every
    DCR_REFRESH_INTERVAL_SECONDS. Every worker runs this loop and checks once
    per interval, but a worker only refreshes when the history file is (nearly)
    that old and no other worker is refreshing (refresh_dcr_history(wait=False)),
    so about one refresh happens per interval in all. The 10% slack keeps a
    check that comes just before the file turns an hour old from pushing the
    refresh to the next hour."""
    first = True
    while True:
        age = _dcr_history_age_seconds()
        if first or age is None or age >= 0.9 * DCR_REFRESH_INTERVAL_SECONDS:
            try:
                summary = await asyncio.to_thread(refresh_dcr_history, False)
                logging.info("Background DCR refresh (worker %s): %s", os.getpid(), summary)
            except Exception as exc:
                logging.warning("Background DCR refresh failed: %s", exc)
        first = False
        await asyncio.sleep(DCR_REFRESH_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: schedule background work on startup, then yield."""
    task = asyncio.create_task(_refresh_dcr_history_periodically())
    yield
    task.cancel()


app = FastAPI(
    title="iCARE4CVD API",
    description="""Upload and explore cohorts metadata files for the [iCARE4CVD project](https://icare4cvd.eu/).""",
    lifespan=lifespan,
)

app.include_router(explore_router, tags=["explore"])
app.include_router(eda_counts_router, tags=["explore"])
app.include_router(mapping_router, prefix="/api", tags=["mapping"])
app.include_router(data_analysis_router, prefix="/api", tags=["data-analysis"])
app.include_router(data_analysis_bare_router, tags=["data-analysis"])
app.include_router(upload_router, tags=["upload"])
app.include_router(decentriq_router, tags=["upload"])
app.include_router(dcr_results_router, tags=["dcr-results"])
app.include_router(auth_router, tags=["authentication"])
app.include_router(admin_router, tags=["admin"])
app.include_router(announcements_router, tags=["announcements"])
app.include_router(docs_router, prefix="/docs-api", tags=["documents"])
app.include_router(chat_router, tags=["chat"])
app.include_router(code_explain_router, tags=["chat"])
app.include_router(nocode_router, tags=["nocode"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Chat-Context"],
)


@app.get("/", include_in_schema=False)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the route / to /docs"""
    return RedirectResponse(url="/docs")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
