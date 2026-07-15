import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


# NOTE: using dataclass instead of pydantic due to dependency conflict with `decentriq_platform` preventing to use pydantic v2
@dataclass
class Settings:
    frontend_url: str = field(default_factory=lambda: os.getenv("FRONTEND_URL", "http://localhost:3001"))
    api_host: str = field(default_factory=lambda: os.getenv("VIRTUAL_HOST", "localhost:3000"))
    sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7878"))

    auth_endpoint: str = field(default_factory=lambda: os.getenv("AUTH_ENDPOINT", ""))
    client_id: str = field(default_factory=lambda: os.getenv("CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("CLIENT_SECRET", ""))
    response_type: str = field(default_factory=lambda: os.getenv("RESPONSE_TYPE", "code"))
    scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email"))
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "").strip())
    local_auth_enabled: bool = field(default_factory=lambda: env_bool("LOCAL_AUTH_ENABLED", False))
    local_auth_email: str = field(
        default_factory=lambda: os.getenv("LOCAL_AUTH_EMAIL", "nikolas.molyndris@decentriq.ch").strip().lower()
    )
    session_cookie_secure: bool = field(default_factory=lambda: env_bool("SESSION_COOKIE_SECURE", True))

    decentriq_email: str = field(default_factory=lambda: os.getenv("DECENTRIQ_EMAIL", ""))
    decentriq_token: str = field(default_factory=lambda: os.getenv("DECENTRIQ_TOKEN", ""))
    admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

    data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "../data"))
    dev_mode: bool = field(default_factory=lambda: env_bool("DEV_MODE", False))
    concept_search_backend: str = field(default_factory=lambda: os.getenv("CONCEPT_SEARCH_BACKEND", "athena"))
    concept_validation_backend: str = field(
        default_factory=lambda: os.getenv("CONCEPT_VALIDATION_BACKEND", "cohortvarlinker")
    )
    mapping_generation_backend: str = field(
        default_factory=lambda: os.getenv("MAPPING_GENERATION_BACKEND", "cohortvarlinker")
    )
    offline_demo: bool = field(default_factory=lambda: env_bool("OFFLINE_DEMO", False))
    mapping_output_dir: str = field(
        default_factory=lambda: os.getenv(
            "MAPPING_OUTPUT_DIR",
            str(Path(__file__).resolve().parents[1] / "CohortVarLinker" / "data" / "mapping_output"),
        )
    )
    demo_pack_dir: str = field(default_factory=lambda: os.getenv("DEMO_PACK_DIR", "../data/synthetic-demo-pack"))
    public_api_url: str = field(default_factory=lambda: os.getenv("PUBLIC_API_URL", "http://localhost:3000"))

    def validate_runtime(self) -> None:
        providers = (
            self.concept_search_backend,
            self.concept_validation_backend,
            self.mapping_generation_backend,
        )
        if self.offline_demo and any(provider != "fixture" for provider in providers):
            raise ValueError("OFFLINE_DEMO requires fixture metadata providers")
        if self.local_auth_enabled and not self.dev_mode:
            raise ValueError("LOCAL_AUTH_ENABLED requires DEV_MODE=true")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET is required")

    @property
    def redirect_uri(self) -> str:
        if self.api_host.startswith("localhost"):
            return f"http://{self.api_host}/cb"
        else:
            return f"https://{self.api_host}/cb"

    @property
    def auth_audience(self) -> str:
        # if self.dev_mode:
        #     return "https://other-ihi-app"
        # else:
        return "https://explorer.icare4cvd.eu"

    @property
    def query_endpoint(self) -> str:
        return f"{self.sparql_endpoint}/query"

    @property
    def update_endpoint(self) -> str:
        return f"{self.sparql_endpoint}/update"

    @property
    def authorization_endpoint(self) -> str:
        return f"{self.auth_endpoint}/authorize"

    @property
    def token_endpoint(self) -> str:
        return f"{self.auth_endpoint}/oauth/token"

    @property
    def admins_list(self) -> list[str]:
        admins = [email.strip().lower() for email in self.admins.split(",") if email.strip()]
        local_admin = self.local_auth_email.strip().lower()
        if self.dev_mode and self.local_auth_enabled and local_admin and local_admin not in admins:
            admins.append(local_admin)
        return admins

    @property
    def cohort_folder(self) -> str:
        return os.path.join(self.data_folder, "cohorts")

    @property
    def logs_filepath(self) -> str:
        return os.path.join(settings.data_folder, "logs.log")


settings = Settings()

# Disable uvicorn logs, does not seems to really do much
# uvicorn_error = logging.getLogger("uvicorn.error")
# uvicorn_error.disabled = True
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True

logging.basicConfig(filename=settings.logs_filepath, level=logging.INFO, format="%(asctime)s - %(message)s")

# import warnings

# from pydantic import computed_field
# from pydantic_settings import BaseSettings, SettingsConfigDict

# warnings.simplefilter(action="ignore", category=UserWarning)

# class Settings(BaseSettings):
#     frontend_url: str = "http://localhost:3001"
#     redirect_uri: str = "http://localhost:3000/cb"

#     auth_endpoint: str = ""
#     client_id: str = ""
#     client_secret: str = ""
#     response_type: str = "code"
#     scope: str = "email read:icare4cvd-dataset-descriptions"
#     # openid required to access userinfo?
#     # read required to check that it's indeed a IHI/ICare4CVD user.

#     decentriq_email: str = ""
#     decentriq_token: str = ""

#     data_folder: str = "../data/cohorts"
#     model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

#     @computed_field()
#     def authorization_endpoint(self) -> str:
#         return f"{self.auth_endpoint}/authorize"

#     @computed_field()
#     def token_endpoint(self) -> str:
#         return f"{self.auth_endpoint}/oauth/token"

# settings = Settings()
