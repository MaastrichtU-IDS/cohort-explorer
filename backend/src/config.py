import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

def get_env_variable(key: str, default: str = None) -> str:
    return os.getenv(key, default)

@dataclass
class Settings:
    # Define your settings with default values or from environment variables
    frontend_url: str = field(default_factory=lambda: get_env_variable('FRONTEND_URL', 'http://localhost:3001'))
    redirect_uri: str = field(default_factory=lambda: get_env_variable('REDIRECT_URI', 'http://localhost:3000/cb'))

    auth_endpoint: str = field(default_factory=lambda: get_env_variable('AUTH_ENDPOINT', ''))
    client_id: str = field(default_factory=lambda: get_env_variable('CLIENT_ID', ''))
    client_secret: str = field(default_factory=lambda: get_env_variable('CLIENT_SECRET', ''))
    response_type: str = field(default_factory=lambda: get_env_variable('RESPONSE_TYPE', 'code'))
    scope: str = field(default_factory=lambda: get_env_variable('SCOPE', 'email read:datasets-descriptions'))

    decentriq_email: str = field(default_factory=lambda: get_env_variable('DECENTRIQ_EMAIL', ''))
    decentriq_token: str = field(default_factory=lambda: get_env_variable('DECENTRIQ_TOKEN', ''))

    data_folder: str = field(default_factory=lambda: get_env_variable('DATA_FOLDER', '../data/cohorts'))

    # Computed fields as regular methods
    def authorization_endpoint(self) -> str:
        return f"{self.auth_endpoint}/authorize"

    def token_endpoint(self) -> str:
        return f"{self.auth_endpoint}/oauth/token"

# Instantiate the Settings
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
#     scope: str = "email read:datasets-descriptions"
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
