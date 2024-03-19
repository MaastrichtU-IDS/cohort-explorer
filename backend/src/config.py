import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")


# NOTE: using dataclass instead of pydantic due to dependency conflict with `decentriq_platform` preventing to use pydantic v2
@dataclass
class Settings:
    frontend_url: str = field(default_factory=lambda: os.getenv("FRONTEND_URL", "http://localhost:3001"))
    redirect_uri: str = field(default_factory=lambda: os.getenv("REDIRECT_URI", "http://localhost:3000/cb"))
    sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7878"))

    auth_endpoint: str = field(default_factory=lambda: os.getenv("AUTH_ENDPOINT", ""))
    client_id: str = field(default_factory=lambda: os.getenv("CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("CLIENT_SECRET", ""))
    response_type: str = field(default_factory=lambda: os.getenv("RESPONSE_TYPE", "code"))
    scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email read:icare4cvd-dataset-descriptions"))
    jwt_secret: str = field(
        default_factory=lambda: os.getenv("JWT_SECRET", "vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM")
    )

    decentriq_email: str = field(default_factory=lambda: os.getenv("DECENTRIQ_EMAIL", ""))
    decentriq_token: str = field(default_factory=lambda: os.getenv("DECENTRIQ_TOKEN", ""))
    admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

    data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "../data"))

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
        return self.admins.split(",")


settings = Settings()


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
