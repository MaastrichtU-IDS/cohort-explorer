import warnings

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

warnings.simplefilter(action="ignore", category=UserWarning)


class Settings(BaseSettings):
    frontend_url: str = "http://localhost:3001"
    redirect_uri: str = "http://localhost:3000/cb"

    auth_endpoint: str = ""
    client_id: str = ""
    client_secret: str = ""
    response_type: str = "code"
    scope: str = "email read:datasets-descriptions"
    # openid required to access userinfo?
    # read required to check that it's indeed a IHI/ICare4CVD user.

    decentriq_email: str = ""
    decentriq_token: str = ""

    data_folder: str = "../data/cohorts"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @computed_field()
    def authorization_endpoint(self) -> str:
        return f"{self.auth_endpoint}/authorize"

    @computed_field()
    def token_endpoint(self) -> str:
        return f"{self.auth_endpoint}/oauth/token"


settings = Settings()
