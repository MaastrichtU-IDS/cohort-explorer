import os

DATA_DIR = "../data/mapping-generation"
VECTOR_PATH = "komal.qdrant.137.120.31.148.nip.io"  #:6333
CLASS = "omop_v5.4"  # class
os.environ["HF_HOME"] = f"{DATA_DIR}/resources/resources/models"
CUDA_NUM = 0  # used GPU num
LOOK_UP_FILE = f"{DATA_DIR}/lookup.csv"
CROSS_MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
EMB_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
QDRANT_PORT = 443
SYN_COLLECTION_NAME = "concept_mapping_1"
NEAREST_SAMPLE_NUM = 64
QUANT_TYPE = "scalar"
LLM_ID = "gpt-4o-mini"  #'llama3.1'
CANDIDATE_GENERATOR_BATCH_SIZE = 64
CACHE_DIR = f"{DATA_DIR}/resources/models"
LLAMA_CACHE_DIR = f"{DATA_DIR}/resources/models/llama"
RETRIEVER = "dense+sparse"
TOPK = 10
GRAPH_DATA = f"{DATA_DIR}/omop_bi_graph.pkl"
# INPUT_DATA_PATH = f"{DATA_DIR}/input/omop_v5.4/concepts.csv"
OUTPUT_DATA_PATH = f"{DATA_DIR}/concepts.jsonl"
DES_DICT_PATH = None  # description data path
LOG_FILE = f"{DATA_DIR}/logs.txt"
DES_LIMIT_LENGTH = 256
MAPPING_FILE = "src/mapping_generation/mapping_templates.json"
# MAPPING_FILE = f"{DATA_DIR}/mapping_templates.json"
