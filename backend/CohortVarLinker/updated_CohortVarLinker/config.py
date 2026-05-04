from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
import os
from dotenv import load_dotenv


load_dotenv()  # reads .env into os.environ
@dataclass(frozen=True)
class Settings:
    # Thresholds
    ADAPTIVE_THRESHOLD: float = 0.5 # minimum score for adaptive retrival from vector db
    ADAPTIVE_ALPHA = 0.85 # upper bound
    LIMIT: int = 5 # limit for adaptive retrival from vector db

    DATE_HINTS: List[str] = field(default_factory=lambda: ["visit date", "date of visit", "date of event", "event date"])
    # TOGETHER_API_KEY: str = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY"))   
    OPENROUTER_API_KEY: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    OLLAMA_URL = "localhost:11434"
    CROSS_CATS: Set[str] = field(default_factory=lambda: {
        "measurement", "observation", "condition_occurrence", 
        "condition_era", "observation_period"
    })
    LLM_CACHE_DIR: str = field(default_factory=lambda: os.getenv("LLM_CACHE_DIR", ".llm_cache"))
    
    DEFAULT_GRAPH_DEPTH: int = 1 # additional cross-vocabulary depth is automatically added for concepts across vocabularies, but this is the default depth for all concepts
    
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
    },
    {
        "name": "eGFR_CKD_EPI", # 2021 CKD-EPI is race-free
        "omop_id": 3030354,                                  
        "code": "loinc:33914-3",
        "label": "Glomerular filtration rate/1.73 sq M.predicted by Creatinine-based formula (CKD-EPI)",
        "unit": "ucum:mL/min/{1.73_m2}",
        "required_omops": [4324383, 4265453, 46235213],       # Creatinine, Age, Sex assigned at birth
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "CKD-EPI 2021 race-free: 142*min(Scr/k,1)^a*max(Scr/k,1)^-1.200*0.9938^Age*1.012[F] ; k=0.7[F]/0.9[M], a=-0.241[F]/-0.302[M]"
    },
    {
        "name": "eGFR_MDRD",
        "omop_id": 3053283,
        "code": "loinc:48643-1",
        "label": "Glomerular filtration rate/1.73 sq M predicted by Creatinine-based formula (MDRD)",
        "unit": "ucum:mL/min/{1.73_m2}",
        "required_omops": [4324383, 4265453, 46235213],        # Creatinine, Age, Sex
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "175 * Scr^-1.154 * Age^-0.203 * 0.742[F] * 1.212[Black]"
    },
    {
        "name": "MAP",
        "omop_id": 4239021,                                  # verify
        "code": "snomed:6797001",
        "label": "Mean blood pressure",
        "unit": "ucum:mm[Hg]",
        "required_omops": [4152194, 4154790],                 # SBP, DBP
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "MAP = DBP + (SBP - DBP)/3"
    },
    {
        "name": "Pulse_Pressure",
        "omop_id": 3559113,                                   # SNOMED 
        "code": "snomed:811751000000106",
        "label": "Pulse pressure",
        "unit": "ucum:mm[Hg]",
        "required_omops": [4152194, 4154790],                 # SBP, DBP
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "PP = SBP - DBP"
    },
    {
        "name": "BSA_DuBois",
        "omop_id": 4201235,
        "code": "snomed:301898006",
        "label": "Body surface area",
        "unit": "ucum:m2",
        "required_omops": [3025315, 3036277],                 # Body Weight, Body Height 
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "DuBois: BSA = 0.007184 * W(kg)^0.425 * H(cm)^0.725"
    },
    {
        "name": "Waist_Hip_Ratio",
        "omop_id": 4087501,                                   # LOINC 9844-2 — verify
        "code": "snomed:248367009",
        "label": "Ratio of waist circumference to hip circumference",
        "unit": "ucum:1",
        "required_omops": [40329251, 4111665],                 # Waist circumference, Hip circumference — verify
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "WHR = waist / hip"
    },
    {
        "name": "LDL_Friedewald", # Friedewald is invalid when triglycerides ≥ 400 mg/dL or in non-fasting samples
        "omop_id": 3028288,
        "code": "loinc:13457-7",
        "label": "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation",
        "unit": "ucum:mg/dL",
        "required_omops": [4008265, 3007070, 3022192],        # Total cholesterol measurement, Cholesterol in HDL [Mass/volume] in Serum or Plasma, Triglyceride [Mass/volume] in Serum or Plasma
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "LDL = TC - HDL - TG/5 ; invalid if TG >= 400 mg/dL (use Martin-Hopkins or direct LDL)"
    },
    {
        "name": "Non_HDL_Cholesterol",
        "omop_id": 3044491,                                  # verify
        "code": "loinc:43396-1",
        "label": "Cholesterol non HDL [Mass/volume] in Serum or Plasma",
        "unit": "ucum:mg/dL",
        "required_omops": [4008265, 3007070],                 # Total cholesterol measurement, Cholesterol in HDL [Mass/volume] in Serum or Plasma
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "Non-HDL = TC - HDL"
    },
    {
        "name": "QTc_Bazett",
        "omop_id": 46235174,                                   
        "code": "loinc:76635-2",
        "label": "Q-T interval corrected based on Bazett formula",
        "unit": "ucum:ms",
        "required_omops": [4216826, 3027018],                 # QT interval feature, heart rate  — verify
        "category": "measurement",
        "data_type": "continuous_variable",
        "formula": "QTc = QT_ms / sqrt(RR_s) ; RR_s = 60/HR"
    }

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