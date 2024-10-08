import csv
import json
import logging
import os
import pickle
from collections.abc import Iterable
from typing import Dict, List, Tuple

# import psutil
import pandas as pd
from json_repair import repair_json
from langchain.schema import Document
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher
from tqdm import tqdm

from .param import *
from .py_model import *

STOP_WORDS = [
    "stop",
    "start",
    "combinations",
    "combination",
    "various combinations",
    "various",
    "left",
    "right",
    "blood",
    "finding",
    "finding status",
    "status",
    "extra",
    "point in time",
    "pnt",
    "oral",
    "product",
    "oral product",
    "several",
    "types",
    "several types",
    "random",
    "nominal",
    "p time",
    "quant",
    "qual",
    "quantitative",
    "qualitative",
    "ql",
    "qn",
    "quan",
    "anti",
    "antibodies",
    "wb",
    "whole blood",
    "serum",
    "plasma",
    "diseases",
    "disorders",
    "disorder",
    "disease",
    "lab test",
    "measurements",
    "lab tests",
    "meas value",
    "measurement",
    "procedure",
    "procedures",
    "panel",
    "ordinal",
    "after",
    "before",
    "survey",
    "level",
    "levels",
    "others",
    "other",
    "p dose",
    "dose",
    "dosage",
    "frequency",
    "calc",
    "calculation",
    "calculation method",
    "method",
    "calc method",
    "calculation methods",
    "methods",
    "calc methods",
    "calculation method",
    "calculation methods",
    "measurement",
    "measurements",
    "meas value",
    "meas values",
    "meas",
    "meas val",
    "meas vals",
    "meas value",
    "meas values",
    "meas",
    "meas val",
    "vals",
    "val",
]


def save_jsonl(data, file):
    with open(file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print("Data saved to file.")


def save_docs_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            # print(doc.json())
            jsonl_file.write(doc.model_dump_json() + "\n")


def convert_to_document(data):
    try:
        page_content = data.get("kwargs", {}).get("page_content", {})
        print(f"page_content={page_content}")
        metadata = data.get("kwargs", {}).get("metadata", {})

        # Create the Document object
        document = Document(page_content=page_content, metadata=metadata)
        return document

    except Exception as e:
        print(f"Error loading document: {e}")
        return None


def load_custom_docs_from_jsonl(file_path) -> list:
    docs = []
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            try:
                obj = Document(**data)
            except Exception:
                print("document object translated into Dictionary format")
                obj = convert_to_document(data)
            docs.append(obj)

    print(f"Total Custom Documents: {len(docs)}")
    return docs


def load_docs_from_jsonl(file_path) -> list:
    docs_dict = {}
    count = 0
    with open(file_path) as jsonl_file:
        print("Opening file...")
        for line in tqdm(jsonl_file, desc="Loading Documents"):
            # if count >= 100:
            #     break
            data = json.loads(line)
            try:
                obj = Document(**data)
            except Exception:
                print("document object translated into Dictionary format")
                obj = convert_to_document(data)
            # print(f"data={obj}")
            if "vocab" in obj.metadata:
                vocab = obj.metadata["vocab"].lower()

                if vocab in [
                    "atc",
                    "loinc",
                    "ucum",
                    "rxnorm",
                    "omop extension",
                    "mesh",
                    "meddra",
                    "cancer modifier",
                    "snomed",
                    "rxnorm extension",
                ]:
                    key = (obj.page_content, json.dumps(obj.metadata, sort_keys=True))
                    if key not in docs_dict:
                        docs_dict[key] = obj
                        count += 1
            else:
                key = (obj.page_content, json.dumps(obj.metadata, sort_keys=True))
                if key not in docs_dict:
                    docs_dict[key] = obj
                    count += 1

    # Convert dictionary values to a sorted list to process documents in a specific order

    sorted_docs = (
        sorted(docs_dict.values(), key=lambda doc: doc.metadata["vocab"].lower())
        if "vocab" in docs_dict.values()
        else sorted(docs_dict.values(), key=lambda doc: doc.metadata["label"].lower())
    )
    print(f"Total Unique Documents: {len(sorted_docs)}\n")
    return sorted_docs


def save_json_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_path}")


def init_logger(log_file_path=LOG_FILE) -> logging.Logger:
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG
    # Create a file handler
    # file_handler = logging.FileHandler(log_file_path)
    # file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

    # Create a stream handler (to print to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set the logging level for the stream handler

    # Define the format for log messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    # logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


global_logger = init_logger()


def save_txt_file(file_path, data) -> None:
    with open(file_path, "a") as file:
        for item in data:
            file.write(f"{item}\n")
    print(f"Total Data = {len(data)} saved to file.")


def save_documents(filepath: str, docs) -> None:
    """Save the BM25 documents to a file."""
    with open(filepath, "wb") as f:
        pickle.dump(docs, f)


def load_documents(filepath: str) -> List[Document]:
    """Load the BM25 documents from a file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# def save_chat_history(file_path: str, history: List[HumanMessage]) -> None:
#     with open(file_path, 'wb') as f:
#         pickle.dump(history, f)

# Function to load chat history from a file
# def load_chat_history(file_path: str) -> List[HumanMessage]:
#     if os.path.exists(file_path):
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#     return []

VOCAB_CACHE = {}


def load_vocabulary(file_path=MAPPING_FILE) -> dict:
    with open(file_path) as file:
        config = json.load(file)
    return config["vocabulary_rules"]


def select_vocabulary(query_text=None, config_path=MAPPING_FILE, domain=None):
    global VOCAB_CACHE
    # Normalize the domain name to lower case or set to 'unknown' if not provided
    domain = domain.lower() if domain else "all"

    # Check if the vocabulary for the domain is alRETRIEVER_CACHEready loaded
    if domain in VOCAB_CACHE:
        selected_vocab = VOCAB_CACHE[domain]
    else:
        # Load the configuration file if the domain's vocabulary isn't cached
        vocabulary_rules = load_vocabulary(config_path)

        # Get domain-specific vocabulary or default to 'unknown' if not found
        selected_vocab = vocabulary_rules["domains"].get(domain, vocabulary_rules["domains"]["unknown"])

        # Cache the selected vocabulary
        VOCAB_CACHE[domain] = selected_vocab

    return selected_vocab


def post_process_candidates(candidates: List[Document], max=1):
    processed_candidates = []
    # seen = set()
    # print(f"Total Candidates={len(candidates)}")
    if not candidates or len(candidates) == 0:
        print("No candidates found.")
        return []

    for _, doc in enumerate(candidates[:max]):
        current_doc_dict = {
            "standard_label": doc.metadata["label"],
            "domain": f"{doc.metadata['domain']}",
            "concept_class": f"{doc.metadata['concept_class']}",
            "standard_code": f"{doc.metadata['vocab']}:{doc.metadata['scode']}",
            "standard_omop_id": str(doc.metadata["sid"]),
            "vocab": doc.metadata["vocab"],
        }
        doc_obj = RetrieverResultsModel(**current_doc_dict)
        if doc_obj not in processed_candidates:
            processed_candidates.append(doc_obj)

    return processed_candidates



def save_to_csv(data, filename):
    if not data:
        return

    fieldnames = [
        'VARIABLE NAME', 'VARIABLE LABEL', 'Domain', 'Variable Concept Label', 'Variable Concept Code','Variable Concept OMOP ID',
        'Additional Context Concept Label', 'Additional Context Concept Code','Additional Context OMOP ID','Primary to Secondary Context Relationship','Categorical Values Concept Label','Categorical Values Concept Code', 'Categorical Values Concept OMOP ID', 'UNIT', 'Unit Concept Label', 'Unit Concept Code','Unit OMOP ID'
    ]

    # Map and combine fields in the data rows
    def map_and_combine_fields(row):
        # Map fields
        mapped_row = {
            'VARIABLE NAME': row.get('VARIABLE NAME', ''),
            'VARIABLE LABEL': row.get('VARIABLE LABEL', ''),
            'Variable Concept Label': row.get('Variable Concept Label',''),
            'Variable Concept OMOP ID': row.get('Variable Concept OMOP ID',''),
            'Variable Concept Code': row.get('Variable Concept Code',''),
            'Domain': row.get('Domain', ''),
            'Additional Context Concept Label': row.get('Additional Context Concept Label', ''),
            'Additional Context Concept Code': row.get('Additional Context Concept Code', ''),
            'Additional Context OMOP ID':row.get('Additional Context OMOP ID', ''),
            'Primary to Secondary Context Relationship': row.get('Primary to Secondary Context Relationship', ''),
            'Categorical Values Concept Label': row.get('Categorical Values Concept Label', ''),
            'Categorical Values Concept Code': row.get('Categorical Values Concept Code', ''),
            'Categorical Values Concept OMOP ID':row.get('Categorical Values Concept OMOP ID', ''),
            'Unit Concept Label': row.get('Unit Concept Label', ''),
            'Unit Concept Code': row.get('Unit Concept Code', ''),
            'Unit OMOP ID': row.get('Unit OMOP ID', ''),
        }
        
        # Combine fields
        # label_ids = '|'.join(filter(None, [row.get('standard_concept_id'), row.get('additional_context_omop_ids')]))
        # label_codes = '|'.join(filter(None, [row.get('standard_code'), row.get('additional_context_codes')]))
        # mapped_row['Variable Concept OMOP ID'] = label_ids
        # mapped_row['Label Concept CODE'] = label_codes
        return mapped_row
    # if file already exists don't readd the header
    if os.path.exists(filename):
        mode='a'
    else:
        mode = 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        for row in data:
            writer.writerow(map_and_combine_fields(row))


def load_mapping(filename, domain):
    print(f"domain={domain}")
    try:
        with open(filename) as file:
            data = json.load(file)

        domain = domain if domain else "all"
        # print(f"domain={domain}")
        mapping = data["mapping_rules"].get(domain, {})

        # Get examples or default to empty list if not present
        relevance_examples = data.get("rel_relevance_examples", {}).get(domain, [])
        ranking_examples = data.get("ranking_examples", {}).get(domain, [])
        # print(f"ranking_examples={ranking_examples[:2]} for domain={domain}")
        # Format examples as string representations of dictionaries
        relevance_examples_string = [
            {
                "input": ex["input"],
                "output": str(
                    [
                        f"{{'answer': '{out['answer']}', 'relationship': '{out['relationship']}', 'explanation': '{out['explanation']}'}}"
                        for out in ex["output"]
                    ]
                ),
            }
            for ex in relevance_examples
        ]

        ranking_examples_string = [
            {
                "input": ex["input"],
                "output": str(
                    [
                        f"{{'answer': '{out['answer']}', 'score': '{out['score']}', 'explanation': '{out['explanation']}'}}"
                        for out in ex["output"]
                    ]
                ),
            }
            for ex in ranking_examples
        ]
        # print(f"ranking_examples_string={ranking_examples_string[:2]}")

        if not mapping:
            return None, ranking_examples_string, relevance_examples_string

        return (
            {
                "prompt": mapping.get("description", "No description provided."),
                "examples": mapping.get("example_output", []),
            },
            ranking_examples_string,
            relevance_examples_string,
        )

    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None, None, None
    except json.JSONDecodeError:
        print("JSON decoding error.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def parse_term(extracted_terms, domain):
    domain = domain.lower() if domain else "all"
    if domain in extracted_terms.keys():
        term = extracted_terms[domain]
        if domain == "condition":
            if "procedure" in extracted_terms:
                procedure = extracted_terms["procedure"]


def save_result_to_jsonl(array: Iterable[dict], file_path: str) -> None:
    print(f"Saving to file: {file_path}")
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            json_string = json.dumps(doc)
            print(json_string)
            jsonl_file.write(json_string + "\n")
    print(f"Saved {len(array)} documents to {file_path}")


# def exact_match_found(query_text, documents, domain=None):
#     # print(f"documents={documents}")
#     if not query_text or not documents:
#         # print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
#         return []
#     for doc in documents:
#         if "score" in doc.metadata:
#             if doc.metadata["score"] >= 0.95:
#                 # print(f"EXACT MATCH FOUND FOR QUERY using Score={query_text}")
#                 return [doc]

#     # Create a database and populate it
#     db = DictDatabase(CharacterNgramFeatureExtractor(2))
#     for doc in documents:
#         if doc.metadata.get("label", None):
#             db.add(doc.metadata["label"])

#     # Create a searcher with cosine similarity
#     searcher = Searcher(db, CosineMeasure())

#     # Normalize query text
#     results = searcher.search(query_text, 0.95)  # Set threshold to 0.95 for high similarity

#     matched_docs = []
#     selected_vocab = select_vocabulary(query_text, domain=domain)

#     for result in results:
#         for doc in documents:
#             if doc.metadata["label"].strip().lower() == result:
#                 if "vocab" in doc.metadata and doc.metadata["vocab"] in selected_vocab:
#                     # print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
#                     matched_docs.append(doc)
#     return matched_docs[:1]



def exact_match_found(query_text, documents, domain=None):
    # print(f"documents={documents}") 
    if not query_text or not documents:
        # print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
        return []
    for doc in documents:
        if 'score' in doc.metadata:
            if doc.metadata['score'] >= 0.95:
                # print(f"EXACT MATCH FOUND FOR QUERY using Score={query_text}")
                return [doc]
        
    # Create a database and populate it
    label_to_docs = {}
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        label = doc.metadata.get('label', None)
        if label:
            db.add(doc.metadata['label'])
            label_key = label.strip().lower()
            if label_key not in label_to_docs:
                label_to_docs[label_key] = []
            label_to_docs[label_key].append(doc)
            

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    results = searcher.search(query_text, 0.9)  # Set threshold to 0.95 for high similarity

    matched_docs = []
    selected_vocab = select_vocabulary(query_text, domain=domain)
 
    matched_docs = [
        doc
        for result in results
        if result in label_to_docs
        for doc in label_to_docs[result]
        if doc.metadata.get('vocab') in selected_vocab
    ]
    
    if len(matched_docs) > 1:
        domain_ = set(list({doc.metadata['domain'] for doc in matched_docs}))
        unique_domain = len(domain_) == 1
        # match_docs_vocab =select_vocabulary(query_text, domain=domain_)
        print(f"is domain unique :{unique_domain}")
        if unique_domain:
            domain_ = domain_.pop()
            match_docs_vocab =select_vocabulary(query_text, domain=domain_)
            match_docs_vocab += selected_vocab
            print(f"selected_vocab for domain={selected_vocab}.. matching docs vocab={match_docs_vocab}")
            first_priority_vocab = match_docs_vocab[0]
            matched_docs = sorted(matched_docs, key=lambda x: (x.metadata['vocab'] != first_priority_vocab, match_docs_vocab.index(x.metadata['vocab'])))

        else:
            #just sort based on select_vocabulary
            matched_docs = sorted(matched_docs, key=lambda x: selected_vocab.index(x.metadata['vocab']))
        print(f"Exact match candidates")
        pretty_print_docs(matched_docs)
    return matched_docs


def exact_match_wo_vocab(query_text, documents, domain=None):
    if not query_text or not documents:
        # print("NO DOCUMENTS FOUND FOR QUERY={query_text}")
        return []

    # Create a database and populate it
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    for doc in documents:
        if "label" in doc.metadata:
            db.add(doc.metadata["label"])

    # Create a searcher with cosine similarity
    searcher = Searcher(db, CosineMeasure())

    # Normalize query text
    results = searcher.search(query_text, 0.95)  # Set threshold to 0.95 for high similarity

    matched_docs = []
    for result in results:
        for doc in documents:
            if doc.metadata["label"] == result:
                print(f"EXACT MATCH FOUND FOR QUERY={query_text}")
                matched_docs.append(doc)
    return matched_docs[:1]


def create_document_string(doc):
    return doc.metadata.get("label", "none")


def fix_json_quotes(json_like_string):
    try:
        # Trying to convert it directly
        return repair_json(json_like_string, return_objects=True)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON after trying to fix quotes, error: {e!s}")
        return None

    """
source code from : "https://github.com/facebookresearch/contriever/blob/main/src/normalize_text.py"

"""


#: Control characters.
CONTROLS = {
    "\u0001",
    "\u0002",
    "\u0003",
    "\u0004",
    "\u0005",
    "\u0006",
    "\u0007",
    "\u0008",
    "\u000e",
    "\u000f",
    "\u0011",
    "\u0012",
    "\u0013",
    "\u0014",
    "\u0015",
    "\u0016",
    "\u0017",
    "\u0018",
    "\u0019",
    "\u001a",
    "\u001b",
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    "-",  # \u002d Hyphen-minus
    "‐",  # \u2010 Hyphen
    "‑",  # \u2011 Non-breaking hyphen
    "⁃",  # \u2043 Hyphen bullet
    "‒",  # \u2012 figure dash
    "–",  # \u2013 en dash
    "—",  # \u2014 em dash
    "―",  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    "-",  # \u002d Hyphen-minus
    "−",  # \u2212 Minus
    "－",  # \uff0d Full-width Hyphen-minus
    "⁻",  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    "+",  # \u002b Plus
    "＋",  # \uff0b Full-width Plus
    "⁺",  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    "/",  # \u002f Solidus
    "⁄",  # \u2044 Fraction slash
    "∕",  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    "~",  # \u007e Tilde
    "˜",  # \u02dc Small tilde
    "⁓",  # \u2053 Swung dash
    "∼",  # \u223c Tilde operator #in mbert vocab
    "∽",  # \u223d Reversed tilde
    "∿",  # \u223f Sine wave
    "〜",  # \u301c Wave dash #in mbert vocab
    "～",  # \uff5e Full-width tilde #in mbert vocab
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    "’",  # \u2019
    "՚",  # \u055a
    "Ꞌ",  # \ua78b
    "ꞌ",  # \ua78c
    "＇",  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    "‘",  # \u2018
    "’",  # \u2019
    "‚",  # \u201a
    "‛",  # \u201b
}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    "“",  # \u201c
    "”",  # \u201d
    "„",  # \u201e
    "‟",  # \u201f
}

#: Accent characters.
ACCENTS = {
    "`",  # \u0060
    "´",  # \u00b4
}

#: Prime characters.
PRIMES = {
    "′",  # \u2032
    "″",  # \u2033
    "‴",  # \u2034
    "‵",  # \u2035
    "‶",  # \u2036
    "‷",  # \u2037
    "⁗",  # \u2057
}

#: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES


def normalize(text):
    if text is None:
        return None
    text = str(text)
    # Replace control characters
    for control in CONTROLS:
        text = text.replace(control, "")
    text = text.replace("\u000b", " ").replace("\u000c", " ").replace("\u0085", " ")

    # Replace hyphens and minuses with '-'
    for hyphen in HYPHENS | MINUSES:
        text = text.replace(hyphen, "-")
    text = text.replace("\u00ad", "")

    # Replace various quotes with standard quotes
    for double_quote in DOUBLE_QUOTES:
        text = text.replace(double_quote, '"')
    for single_quote in SINGLE_QUOTES | APOSTROPHES | ACCENTS:
        text = text.replace(single_quote, "'")
    text = text.replace("′", "'")  # \u2032 prime
    text = text.replace("‵", "'")  # \u2035 reversed prime
    text = text.replace("″", "''")  # \u2033 double prime
    text = text.replace("‶", "''")  # \u2036 reversed double prime
    text = text.replace("‴", "'''")  # \u2034 triple prime
    text = text.replace("‷", "'''")  # \u2037 reversed triple prime
    text = text.replace("⁗", "''''")  # \u2057 quadruple prime
    text = text.replace("…", "...").replace(" . . . ", " ... ")  # \u2026

    # Replace slashes with '/'
    for slash in SLASHES:
        text = text.replace(slash, "/")

    # Ensure there's only one space between words
    text = " ".join(text.split())
    text = text.lower()
    if text in ["null", "nil", "none", "n/a", "", "nan"]:
        return None
    return text


def normalize_page_content(page_content):
    if page_content is None:
        return page_content
    page_content = page_content.strip().lower()

    if "<ent>" in page_content:
        page_content = page_content.split("<ent>")[1].split("</ent>")[0]
        if "||" in page_content:
            page_content = page_content.split("||")[0]
        if "." in page_content:
            page_content = page_content.split(".")[0]
    elif "||" in page_content:
        page_content = page_content.split("||")[0]
    elif "." in page_content:
        page_content = page_content.split(".")[0]
    # print(f"Page Content: {page_content}")
    return page_content


BASE_IRI = "http://ccb.hms.harvard.edu/t2t/"

STOP_WORDS = {
    "in",
    "the",
    "any",
    "all",
    "for",
    "and",
    "or",
    "dx",
    "on",
    "fh",
    "tx",
    "only",
    "qnorm",
    "w",
    "iqb",
    "s",
    "ds",
    "rd",
    "rdgwas",
    "ICD",
    "excluded",
    "excluding",
    "unspecified",
    "certain",
    "also",
    "undefined",
    "ordinary",
    "least",
    "squares",
    "FINNGEN",
    "elsewhere",
    "more",
    "classified",
    "classifeid",
    "unspcified",
    "unspesified",
    "specified",
    "acquired",
    "combined",
    "unspeficied",
    "not",
    "by",
    "strict",
    "wide",
    "definition",
    "definitions",
    "confirmed",
    "chapter",
    "chapters",
    "controls",
    "characterized",
    "main",
    "diagnosis",
    "hospital",
    "admissions",
    "other",
    "resulting",
    "from",
}

TEMPORAL_WORDS = {
    "age",
    "time",
    "times",
    "date",
    "initiation",
    "cessation",
    "progression",
    "duration",
    "early",
    "late",
    "later",
    "trimester",
}

QUANTITY_WORDS = {
    "hourly",
    "daily",
    "weekly",
    "monthly",
    "yearly",
    "frequently",
    "per",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "years",
    "total",
    "quantity",
    "amount",
    "level",
    "levels",
    "volume",
    "count",
    "counts",
    "percentage",
    "abundance",
    "proportion",
    "content",
    "average",
    "prevalence",
    "mean",
    "ratio",
}

BOLD = "\033[1m"
END = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def remove_duplicates(values):
    seen = set()
    return [x for x in values if not (x in seen or seen.add(x))]


def combine_if_different(list_1, list_2, separator="|") -> str:
    """
    Combines two lists with a separator if they are different. Returns a concatenated string if different, else a single value.
    """
    joined_1 = join_or_single(list_1)
    joined_2 = join_or_single(list_2)

    if joined_1 != joined_2:
        # Combine them with a separator if they are different
        return f"{joined_1}{separator}{joined_2}" if joined_1 and joined_2 else joined_1 or joined_2
    else:
        # If they are the same, return one of them
        return joined_1


def join_or_single(items, seperator="|") -> str:
    if items is None:
        return ""
    if isinstance(items, str):
        return items
    unique_items = remove_duplicates(filter(None, items))
    return "|".join(unique_items) if len(unique_items) > 1 else unique_items[0] if unique_items else ""


def create_processed_result(result_object: ProcessedResultsModel) -> dict:
    """
    Processes the input query and maps it to standardized concepts, returning the processed result.

    Parameters:
    -----------
    result_object : ProcessedResultsModel
        A dictionary containing processed documents with information like standard labels, codes, and concept IDs.

    Returns:
    --------
    dict
        A dictionary containing the processed result, including:
        - 'query_text': The original query.
        - 'revised_query': The revised version of the query.
        - 'domain': The domain of the query.
        - 'standard_label': The combined standard label for the query.
        - 'standard_code': The combined standard codes.
        - 'standard_concept_id': The combined concept IDs.
        - 'additional_context', 'categorical_values', 'unit': Processed values, status, and unit information.
    """
    additional_entities = result_object.additional_entities
    additional_entities_matches = result_object.additional_entities_matches
    categorical_values = result_object.categories
    categorical_values_matches = result_object.categories_matches
    # main_term = result_object.base_entity
    main_term_matches = result_object.base_entity_matches
    # query_text = result_object.original_query

    additional_entities_labels, additional_entities_codes, additional_entities_omop_ids = format_categorical_values(
        additional_entities_matches, additional_entities
    )
    categorical_values_labels, categorical_values_codes, categorical_values_omop_ids = format_categorical_values(
        categorical_values_matches, categorical_values
    )
    main_term_labels = main_term_matches[0].standard_label if len(main_term_matches) >= 1 else None
    main_term_codes = main_term_matches[0].standard_code if len(main_term_matches) >= 1 else None
    main_term_omop_id = main_term_matches[0].standard_omop_id if len(main_term_matches) >= 1 else None
    # if additional_entities:
    #     combined_labels = combine_if_different(main_term_labels, additional_entities_labels, separator="|")
    #     combined_codes = combine_if_different(main_term_codes, additional_entities_codes, separator="|")
    #     combined_omop_ids = combine_if_different(main_term_omop_id, additional_entities_omop_ids, separator="|")

    # else:
    #     combined_labels = join_or_single(main_term_labels)
    #     combined_codes = join_or_single(main_term_codes)
    #     combined_omop_ids = join_or_single(main_term_omop_id)
    results = {
        "VARIABLE NAME": result_object.variable_name,
        "VARIABLE LABEL": result_object.original_query,
        "Domain": result_object.domain,
        "Variable Concept Label": main_term_labels,
        "Variable Concept Code": main_term_codes,
        "Variable Concept OMOP ID": main_term_omop_id,
        "Additional Context Concept Label": additional_entities_labels,
        "Additional Context Concept Code": additional_entities_codes,
        "Additional Context OMOP ID": additional_entities_omop_ids,
        "Primary to Secondary Context Relationship": result_object.primary_to_secondary_rel,
        "Categorical Values Concept Label": categorical_values_labels,
        "Categorical Values Concept Code": categorical_values_codes,
        "Categorical Values Concept OMOP ID": categorical_values_omop_ids,
        "Unit Concept Label": result_object.unit_matches[0].standard_label
        if len(result_object.unit_matches) >= 1
        else None,
        "Unit Concept Code": result_object.unit_matches[0].standard_code
        if len(result_object.unit_matches) >= 1
        else None,
        "Unit OMOP ID": result_object.unit_matches[0].standard_omop_id
        if len(result_object.unit_matches) >= 1
        else None,
    }
    print(f"Processed Result={results}")
    return results


def create_processed_result_from_components(query_text, component_cache, query_final_result):
    def get_value(key, cache_key):
        return (
            component_cache[cache_key][key]
            if component_cache.get(cache_key) and key in component_cache[cache_key]
            else query_final_result.get(key, "")
        )

    return {
        "query_text": query_text,
        "revised_query": get_value("revised_query", "context"),
        "domain": get_value("domain", "context")
        or get_value("domain", "status")
        or get_value("domain", "unit")
        or get_value("domain", "main_query"),
        "standard_label": get_value("standard_label", "context"),
        "standard_code": get_value("standard_code", "context"),
        "standard_concept_id": get_value("standard_concept_id", "context"),
        "additional_context": get_value("additional_context", "context"),
        "additional_context_codes": get_value("additional_context_codes", "context"),
        "additional_context_omop_ids": get_value("additional_context_omop_ids", "context"),
        "categorical_values": get_value("categorical_values", "status"),
        "categorical_codes": get_value("categorical_codes", "status"),
        "categorical_omop_ids": get_value("categorical_omop_ids", "status"),
        "unit": get_value("unit", "unit"),
        "unit_code": get_value("unit_code", "unit"),
        "unit_concept_id": get_value("unit_concept_id", "unit"),
    }


def format_categorical_values(
    values_matches_documents: Dict[str, List[RetrieverResultsModel]], values_list: List[str]
) -> Tuple[str, str, str]:
    labels = []
    codes = []
    ids = []
    # print(f"values: {status}\nDocs: {status_docs}")
    if values_matches_documents is None or len(values_matches_documents) == 0:
        return None, None, None

    # Normalize the keys of status_docs to lower case for case-insensitive matching
    normalized_status_docs = {k.lower(): v for k, v in values_matches_documents.items()}
    for v_ in values_list:
        v_ = v_.strip().lower()
        if v_ in normalized_status_docs:
            docs = normalized_status_docs[v_]
            # if len(docs) > 1:
            #     labels.append(' and/or '.join(remove_duplicates([doc.standard_label for doc in docs])))
            #     codes.append(' and/or '.join(remove_duplicates([doc.standard_code for doc in docs])))
            #     ids.append(' and/or '.join(remove_duplicates(doc.standard_concept_id)) for doc in docs])))
            # elif len(docs) == 1:
            labels.append(docs[0].standard_label)
            codes.append(docs[0].standard_code)
            ids.append(docs[0].standard_omop_id)
        else:
            labels.append("na")
            codes.append("na")
            ids.append("na")

    return "|".join(labels), "|".join(codes), "|".join(ids)


def process_synonyms(synonyms_text: str) -> List[str]:
    """
    Processes the synonyms text to split by ';;' if it exists, otherwise returns the whole text as a single item list.
    If synonyms_text is empty, returns an empty list.
    """
    if synonyms_text:
        if ";;" in synonyms_text:
            return synonyms_text.split(";;")
        else:
            return [synonyms_text]
    return []


def parse_tokens(text: str, semantic_type: bool = False, domain: str = None) -> Tuple[str, List[str], str]:
    """Extract entity and synonyms from the formatted text, if present."""
    if text is None:
        return None, [], None
    text_ = text.strip().lower()
    print(f"text={text_}")
    # Initialize default values
    entity = None
    synonyms = []
    semantic_type_of_entity = None

    # Check if the text follows the new format
    if (
        "concept name:" in text_
        and "synonyms:" in text_
        and "domain:" in text_
        and "concept class:" in text_
        and "vocabulary:" in text_
    ):
        parts = text_.split(", ")
        concept_dict = {}
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                concept_dict[key.strip()] = value.strip()
            else:
                concept_dict["concept name"] = part.strip()
        entity = concept_dict.get("concept name")
        synonyms_str = concept_dict.get("synonyms", "")
        synonyms = process_synonyms(synonyms_str)
        if semantic_type:
            if domain:
                semantic_type_of_entity = concept_dict.get("domain", "")
                entity = f"{entity}||{semantic_type_of_entity}"
        global_logger.info(
            f"Entity: {entity}, Synonyms: {synonyms}, Semantic Type of Entity: {semantic_type_of_entity}"
        )
    else:
        # Fallback to the old format handling
        if "<ent>" in text_ and "<syn>" in text_:
            entity = text_.split("<ent>")[1].split("</ent>")[0]
            synonyms_text = text_.split("<syn>")[1].split("</syn>")[0]
            synonyms = process_synonyms(synonyms_text)
            global_logger.info(f"Entity: {entity}, Synonyms: {synonyms}")
        else:
            entity = text_
            synonyms = []

    return entity, synonyms, None


def combine_ent_synonyms(text: str, semantic_type=False, domain=None) -> str:
    """Extract entity and synonyms from the formatted text, if present."""
    text_ = text.strip().lower()
    if "<desc>" in text_:
        description = text_.split("<desc>")[1].split("</desc>")[0]
        global_logger.info(f"Description: {description}")
        text = text_.split("<desc>")[0]
    return text_


def print_docs(docs):
    for res, i in zip(docs, range(len(docs))):
        print("----" + str(i + 1) + "----")
        # print("LABEL: " + res.metadata["label"])
        print(f"{res.metadata['label']}:{res.metadata['domain']}")


def filter_irrelevant_domain_candidates(docs, domain) -> List[RetrieverResultsModel]:
    select_vocabs = select_vocabulary(domain=domain)

    docs_ = [doc for doc in docs if doc.metadata["vocab"] in select_vocabs]
    if len(docs_) == 0:
        # use snomed as base ontology
        select_vocabs.append("snomed")
        docs_ = [doc for doc in docs if doc.metadata["vocab"] in select_vocabs]
    return docs_


def pretty_print_docs(docs) -> None:
    for doc in docs:
        print(f"****{doc.metadata['label']}****")


# def estimate_token_cost(text,filename):
#     # Get the encoder for the GPT-4 model
#     enc = tiktoken.encoding_for_model("gpt-4")

#     # Encode the text and count the tokens
#     n_tokens = len(enc.encode(text))

#     # Save the token count to a text file
#     with open(filename, 'w') as f:
#         f.write(f"Token count: {n_tokens}")

#     print(f"Token count saved to {filename}")


def append_results_to_csv(input_file, results, output_suffix="_mapped.csv") -> None:
    """
    Reads the input CSV file, uses the number of rows according to `results` length, appends new columns,
    and saves it with a new name with the suffix '_mapped.csv'.

    Parameters:
    -----------
    input_file : str
        The path to the input CSV file.
    results : list of dict
        A list of dictionaries containing processed data. Each dictionary corresponds to a row.
    output_suffix : str, optional
        The suffix to append to the output CSV file name. Default is '_mapped.csv'.

    Returns:
    --------
    None
    """
    # Step 1: Load the input CSV file
    df = pd.read_csv(input_file)

    # Step 2: Use only the number of rows that match the length of `results`
    if len(df) != len(results):
        df = df.iloc[: len(results)]

    # Step 3: Extract data from `results` to create new columns
    new_columns_data = {
        "Variable Concept Label": [result.get("Variable Concept Label", None) for result in results],
        "Variable Concept Code": [result.get("Variable Concept Code", None) for result in results],
        "Variable Concept OMOP ID": [result.get("Variable Concept OMOP ID", None) for result in results],
        "Domain": [result.get("DOMAIN", None) for result in results],
        "Additional Context Concept Label": [
            result.get("Additional Context Concept Label", None) for result in results
        ],
        "Additional Context Concept Code": [result.get("Additional Context Concept Code", None) for result in results],
        "Additional Context OMOP ID": [result.get("Additional Context OMOP ID", None) for result in results],
        "Primary to Secondary Context Relationship": [
            result.get("Primary to Secondary Context Relationship", None) for result in results
        ],
        "Categorical Values Concept Label": [
            result.get("Categorical Values Concept Label", None) for result in results
        ],
        "Categorical Values Concept Code": [result.get("Categorical Values Concept Code", None) for result in results],
        "Categorical Values Concept OMOP ID": [
            result.get("Categorical Values Concept OMOP ID", None) for result in results
        ],
        "Unit Concept Label": [result.get("Unit Concept Label", None) for result in results],
        "Unit Concept Code": [result.get("Unit Concept Code", None) for result in results],
        "Unit OMOP ID": [result.get("Unit OMOP ID", None) for result in results],
    }

    # Step 4: Append the new columns to the dataframe
    for column_name, column_data in new_columns_data.items():
        df[column_name] = column_data

    # Step 5: Save the updated dataframe to a new CSV file
    file_name, _ = os.path.splitext(input_file)
    output_file = f"{file_name}{output_suffix}"
    df.to_csv(output_file, index=False)

    print(f"File saved: {output_file}")
    return df
