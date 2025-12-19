# import logging
# import os
# from dataclasses import dataclass, field

# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv(".env")


# # NOTE: using dataclass instead of pydantic due to dependency conflict with `decentriq_platform` preventing to use pydantic v2
# @dataclass
# class Settings:

#     sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7200/repositories/icare4cvd"))

  
#     scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email"))
#     # jwt_secret: str = field(
#     #     default_factory=lambda: os.getenv("JWT_SECRET", "vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM")
#     # )

  
#     admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

#     data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "./data"))
#     # dev_mode: bool = field(default_factory=lambda: os.getenv("DEV_MODE", "false").lower() == "true")

#     @property
#     # def redirect_uri(self) -> str:
#     #     if self.api_host.startswith("localhost"):
#     #         return f"http://{self.api_host}/cb"
#     #     else:
#     #         return f"https://{self.api_host}/cb"

#     @property
#     def auth_audience(self) -> str:
#         # if self.dev_mode:
#         #     return "https://other-ihi-app"
#         # else:
#         return "https://explorer.icare4cvd.eu"

#     @property
#     def query_endpoint(self) -> str:
#         return f"{self.sparql_endpoint}/query"

#     @property
#     def update_endpoint(self) -> str:
#         return f"{self.sparql_endpoint}/update"

#     @property
#     def authorization_endpoint(self) -> str:
#         return f"{self.auth_endpoint}/authorize"

#     @property
#     def admins_list(self) -> list[str]:
#         return self.admins.split(",")

#     @property
#     def logs_filepath(self) -> str:
#         return os.path.join(settings.data_folder, "logs.log")
    
#     @property
#     def sqlite_db_filepath(self) -> str:
#         return "vocab.db"


# settings = Settings()

# # Disable uvicorn logs, does not seems to really do much
# # uvicorn_error = logging.getLogger("uvicorn.error")
# # uvicorn_error.disabled = True
# uvicorn_access = logging.getLogger("uvicorn.access")
# # uvicorn_access.disabled = True

# logging.basicConfig(filename=settings.logs_filepath, level=logging.INFO, format="%(asctime)s - %(message)s")


# import os
# import logging
# from dataclasses import dataclass, field
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv(".env")

# @dataclass
# class Settings:
#     sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7200/repositories/icare4cvd"))
    
#     scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email"))
#     admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

#     data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "./data"))

#     @property
#     def auth_audience(self) -> str:
#         return "https://explorer.icare4cvd.eu"

#     @property
#     def query_endpoint(self) -> str:
#         return f"{self.sparql_endpoint}/query"

#     @property
#     def update_endpoint(self) -> str:
#         return f"{self.sparql_endpoint}/statements"

#     @property
#     def admins_list(self) -> list[str]:
#         return self.admins.split(",")

#     @property
#     def logs_filepath(self) -> str:
#         """Log file for general application logs."""
#         return os.path.join(self.data_folder, "logs.log")
    
#     @property
#     def graphdb_log_filepath(self) -> str:
#         """Log file for GraphDB import operations."""
#         return os.path.join(self.data_folder, "graphdb_import.log")

# settings = Settings()

# # Configure logging
# logging.basicConfig(
#     filename=settings.logs_filepath,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# graphdb_logger = logging.getLogger("graphdb_import")
# graphdb_logger.setLevel(logging.INFO)

# file_handler = logging.FileHandler(settings.graphdb_log_filepath)
# file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# graphdb_logger.addHandler(file_handler)



import os
import logging
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")


# NOTE: using dataclass instead of pydantic due to dependency conflict with decentriq_platform preventing to use pydantic v2
@dataclass
class Settings:

    sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7878"))

  
    scope: str = field(default_factory=lambda: os.getenv("SCOPE", "openid email"))
    # jwt_secret: str = field(
    #     default_factory=lambda: os.getenv("JWT_SECRET", "vCitcsPBwH4BMCwEqlO1aHJSIn--usrcyxPPRbeYdHM")
    # )

  
    admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

    data_folder: str = field(default_factory=lambda: os.getenv(
        "DATA_FOLDER",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    ))
    cohort_folder: str = field(default_factory=lambda: os.path.join(
        os.getenv(
            "DATA_FOLDER",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
        ),
        "cohorts"
    ))
    output_dir: str = field(default_factory=lambda: os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/mapping_output")
    ))
    # dev_mode: bool = field(default_factory=lambda: os.getenv("DEV_MODE", "false").lower() == "true")

    @property
    # def redirect_uri(self) -> str:
    #     if self.api_host.startswith("localhost"):
    #         return f"http://{self.api_host}/cb"
    #     else:
    #         return f"https://{self.api_host}/cb"

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
    def admins_list(self) -> list[str]:
        return self.admins.split(",")

    @property
    def logs_filepath(self) -> str:
        return os.path.join(settings.data_folder, "logs.log")
    
    @property
    def sqlite_db_filepath(self) -> str:
        return "vocab.db"
    
    @property
    def vector_db_path(self) -> str:
        # return  "komal.qdrant.137.120.31.148.nip.io"
        return  "localhost"
    @property
    def concepts_file_path(self) -> str:
        # return  "komal.qdrant.137.120.31.148.nip.io"
        return  "/Users/komalgilani/Desktop/cmh/data/concept_relationship.csv"


settings = Settings()

print("in CohortVarLinker/src/config.py : ", settings.data_folder)
print("settings Cohort folder: ", settings.cohort_folder)
print("settings Output directory: ", settings.output_dir)
print("settings Concepts file path: ", settings.concepts_file_path)


# Disable uvicorn logs, does not seems to really do much
# uvicorn_error = logging.getLogger("uvicorn.error")
# uvicorn_error.disabled = True
# uvicorn_access = logging.getLogger("uvicorn.access")
# uvicorn_access.disabled = True

# logging.basicConfig(filename=settings.logs_filepath, level=logging.INFO, format="%(asctime)s - %(message)s")