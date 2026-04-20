from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
import os
from dotenv import load_dotenv


load_dotenv()  # reads .env into os.environ
@dataclass(frozen=True)
class Settings:
    # Thresholds
    SIMILARITY_THRESHOLD: float = 0.8 # lower threshold and apply adaptive gate
    ADAPTIVE_THRESHOLD: float = 0.45 # minimum score for adaptive retrival from vector db
    LIMIT: int = 5 # limit for adaptive retrival from vector db
        # Text hints for logic
    DATE_HINTS: List[str] = field(default_factory=lambda: ["visit date", "date of visit", "date of event", "event date"])
    TOGETHER_API_KEY: str = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY"))   
    # FIREWORKS_API_KEY: str = field(default_factory=lambda: os.getenv("FIREWORKS_API_KEY"))
    OPENROUTER_API_KEY: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    OLLAMA_URL = "localhost:11434"
    CROSS_CATS: Set[str] = field(default_factory=lambda: {
        "measurement", "observation", "condition_occurrence", 
        "condition_era", "observation_period"
    })
    LLM_CACHE_DIR: str = field(default_factory=lambda: os.getenv("LLM_CACHE_DIR", ".llm_cache"))
    
    DEFAULT_GRAPH_DEPTH: int = 1 # additional cross-vocabulary depth is automatically added for concepts across vocabularies, but this is the default depth for all concepts
    
    LOCAL_LLM_MODELS: List[str] = field(default_factory=lambda:  ["mistral:latest", "llama3.3:70b", "qwen2.5:72b"])
    TOGETHER_LLM_MODELS: List[str] = field(default_factory=lambda:  [ "Qwen/Qwen3-Next-80B-A3B-Instruct", "openai/gpt-oss-120b","meta-llama/Llama-3.3-70B-Instruct-Turbo" , "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"])
    LARGE_LMs: List[str] = field(default_factory=lambda:  ["gemini-2.5-flash"]) # "gpt-4.1-2025-04-14",
    
    # External Resources
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    GRAPH_REPO: str = "https://w3id.org/CMEO/graph"

    DATA_DOMAINS: List[str] = field(default_factory=lambda: ["drug_exposure","condition_occurrence","condition_era","observation","observation_era","measurement","visit_occurrence","procedure_occurrence","device_exposure","person"])
    # Derived Variable Rules (Example)
    DERIVED_VARIABLES_LIST: List[Dict[str, Any]] = field(default_factory=lambda: [
    {
        "name": "BMI",
        "omop_id": 3038553,
        "code": "loinc:39156-5",
        "label": "Body mass index (BMI) [Ratio]",
        "unit": "ucum:kg/m2",
        "required_omops": [3016723, 3025315],  # Weight, Height
        "category": "measurement",
        "data_type": "continuous_variable"
    },
    {
        "name": "eGFR_CG",
        "omop_id": 37169169,
        "code": "snomed:1556501000000100",
        "label": "Estimated creatinine clearance calculated using actual body weight Cockcroft-Gault formula",
        "unit": "ucum:ml/min",
        "required_omops": [3025315, 3016723, 3022304, 46235213],  # Height, Weight, Creatinine, Age/Gender proxy
        "category": "measurement",
        "data_type": "continuous_variable"
    }
    # CKD-EPI is prioritized  over others, eGFR (CKD-EPI and MDRD)
    ])
    admins: str = field(default_factory=lambda: os.getenv("ADMINS", ""))

    data_folder: str = field(default_factory=lambda: os.getenv("DATA_FOLDER", "./data"))
    sparql_endpoint: str = field(default_factory=lambda: os.getenv("SPARQL_ENDPOINT", "http://localhost:7879"))

    EMBEDDING_MODEL_NAME: str = "sapbert"
    FULL_EMBEDDING_MODEL_NAME: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    MODEL_CACHE_DIR: str = "../data/models"
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
    def admins_list(self) -> list[str]:
        return self.admins.split(",")

    @property
    def logs_filepath(self) -> str:
        return os.path.join(self.data_folder, "../logs.log")
    
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
        return  "../data/concept_relationship_enriched.csv"

settings = Settings()