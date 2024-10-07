
import os
CLASS = 'omop_v5.4' #class
os.environ['HF_HOME'] = 'backend/resources/resources/models'
CUDA_NUM = 0 # used GPU num
LOOK_UP_FILE = "backend/data/output/lookup.csv"
CROSS_MODEL_ID = "ncbi/MedCPT-Cross-Encoder"
EMB_MODEL_NAME  ="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
VECTOR_PATH="komal.qdrant.137.120.31.148.nip.io"   #:6333
QDRANT_PORT = 443
SYN_COLLECTION_NAME = 'concept_mapping_1'
NEAREST_SAMPLE_NUM = 64
QUANT_TYPE = 'scalar'
LLM_ID = 'llama3.1'   #'gpt-4o-mini'
CANDIDATE_GENERATOR_BATCH_SIZE = 64
CACHE_DIR = 'backend/resources/models'
LLAMA_CACHE_DIR = 'backend/resources/models/llama'
DATA_DIR="backend/data"
RETRIEVER = 'dense+sparse'
TOPK = 10
GRAPH_DATA = f"{DATA_DIR}/output/omop_bi_graph.pkl"
# INPUT_DATA_PATH = f"{DATA_DIR}/input/omop_v5.4/concepts.csv"
OUTPUT_DATA_PATH = f"{DATA_DIR}/output/concepts.jsonl"
DES_DICT_PATH = None #description data path
LOG_FILE = "/workspace/mapping_tool/resources/logs/log.txt"
DES_LIMIT_LENGTH = 256
MAPPING_FILE = f"{DATA_DIR}/mapping_templates.json"