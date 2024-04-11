from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from src.auth import router as auth_router
from src.config import settings
from src.decentriq import router as decentriq_router
from src.explore import router as explore_router
from src.mapping import router as mapping_router
from src.upload import init_triplestore
from src.upload import router as upload_router

init_triplestore()

app = FastAPI(
    title="iCARE4CVD API",
    description="""Upload and explore cohorts metadata files for the [iCARE4CVD project](https://icare4cvd.eu/).""",
)

app.include_router(explore_router, tags=["explore"])
app.include_router(mapping_router, tags=["mapping"])
app.include_router(upload_router, tags=["upload"])
app.include_router(decentriq_router, tags=["upload"])
app.include_router(auth_router, tags=["authentication"])

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
