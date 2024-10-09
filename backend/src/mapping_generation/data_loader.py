import csv
import datetime
import json
import os
import re
from typing import List

import pandas as pd
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError

# import numpy as np
from tqdm import tqdm

from .graph_omop import *
from .param import *
from .py_model import QueryDecomposedModel
from .utils import global_logger as logger
from .utils import load_docs_from_jsonl, normalize, save_docs_to_jsonl


def clean_text(text):
    # Remove leading and trailing spaces
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text

def custom_data_loader(source_path):
    """
    Loads and processes data from a CSV file. Checks if the file is already mapped based on specific columns.

    Parameters:
    - source_path (str): Path to the CSV file.

    Returns:
    - docs (list): List of processed QueryDecomposedModel objects if not mapped.
    - is_mapped (bool): True if the file is already mapped, False otherwise.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(source_path, low_memory=False, header=0, index_col=None)
    except FileNotFoundError:
        print(f"Error: The file at {source_path} was not found.")
        return [], False
    except pd.errors.EmptyDataError:
        print("Error: The provided CSV file is empty.")
        return [], False
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return [], False

    # Define the required columns to check
    required_columns = {
        'Variable Concept Label',
        'Variable Concept OMOP ID',
        'Variable Concept Code'
    }

    # Normalize column names to handle case sensitivity and leading/trailing spaces
    normalized_columns = set(col.strip().lower() for col in data.columns)

    # Normalize required column names for comparison
    normalized_required = set(col.lower() for col in required_columns)

    # Check if all required columns are present
    if normalized_required.issubset(normalized_columns):
        is_mapped = True
        print("File is already mapped. Skipping further processing.")
        # Optionally, you can return early if no further processing is needed
        # return list of dictionary with all the columns
        return data.to_dict(orient='records'), is_mapped

    is_mapped = False
    print("File is not mapped. Proceeding with data processing.")

    docs = []

    # Iterate through the rows with a progress bar
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc='Retrieving input data'):
        try:
            # Extract and process the 'VARIABLE LABEL' field
            # print(f"Row: {row}")
            label = str(row.get('VARIABLE LABEL', '')).lower().strip() if pd.notna(row.get('VARIABLE LABEL')) else None
            name = str(row.get('VARIABLE NAME', '')).lower().strip() if pd.notna(row.get('VARIABLE NAME')) else None
            if not label or not name:
                print(f"Row {index} has an empty 'VARIABLE LABEL'. Skipping.")
                continue

            # Handle 'CATEGORICAL' field
            categorical_raw = row.get('CATEGORICAL', None)
            if pd.notna(categorical_raw) and str(categorical_raw).strip():
                categories = str(categorical_raw).lower().strip().replace(',', '|').split("|")
            else:
                categories = None
            visits = None
            # Handle 'UNITS' field
            unit_raw = row.get('UNITS', None)
            unit = str(unit_raw).lower().strip() if pd.notna(unit_raw) else None
            visits_raw = row.get('Visits', None)
            if pd.notna(visits_raw) and visits_raw:
                    visits = str(visits).lower().strip() if pd.notna(visits) else None
                    if visits:
                        if 'visit' not in visits and 'baseline' not in visits and ('visit' in label or 'baseline' in label or 'month' in label):
                            visits = f"visit {visits}"
                        visits = f"at {visits}"
            # Handle 'Formula' field
            formula_raw = row.get('Formula', None)
            formula = str(formula_raw).lower().strip() if pd.notna(formula_raw) else None

            # Construct the 'full_query' string
            full_query = label
            if visits:
                full_query += f" {visits}"
            if categories:
                full_query += f", categorical values: {'|'.join(categories)}"
            if unit:
                full_query += f", unit: {unit}"
            if formula:
                full_query += f", formula: {formula}"
            
            base_entity = f"{label} {formula}" if formula else label
            base_entity = f"{base_entity} {visits}" if visits else base_entity
            # Create a dictionary for the QueryDecomposedModel
            query_dict = {
                'name':name,
                'full_query': full_query,
                'base_entity':base_entity,
                'categories': categories,
                'unit': unit,
                'formula': formula,
                'domain': 'all',
                'rel': None,
                'original_label': label
            }
            # print(f"Query Dict: {query_dict}")
            # Instantiate the QueryDecomposedModel
            query_model = QueryDecomposedModel(**query_dict)

            # Append the result as a tuple (index, query_model)
            docs.append((index, query_model))

        except ValidationError as ve:
            # Handle validation errors from Pydantic
            print(f"Validation error in row {index}: {ve}")
        except Exception as ex:
            # Handle any other unexpected errors
            print(f"Unexpected error in row {index}: {ex}")

    return docs, is_mapped

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
    "quant",
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
]


class GraphLoader(BaseLoader):
    def __init__(self, graph_path=None, data_dir="/workspace/mapping_tool/data"):
        self.graph_path = graph_path if graph_path else f"{data_dir}/output/omop_bi_graph.pkl"
        self.omop = Omop(data_dir)
        self.omop.graph_path = self.graph_path
        self.graph = None

    def load_graph(self) -> List[Document]:
        self.omop.load_concepts()
        docs: List[Document] = []
        max_len = 0
        max_len_doc = None
        index = 0
        for node_id, data in tqdm(self.omop.graph.nodes(data=True)):
            # ontology_ = str(data.get('ontology')).lower()
            doc, doc_len = self._create_document(node_id, data, index)
            docs.append(doc)
            if doc_len > max_len:
                max_len = doc_len
                max_len_doc = doc
            index += 1
        print(f"Max Length of Document is {max_len_doc} with length {max_len}")
        return docs, max_len

    def _create_document(self, node_id, node, index) -> Document:
        label = node.get("desc")
        synonyms = list(node.get("synonyms", []))
        concept_class = normalize(str(node.get("concept_class")))
        domain = normalize(str(node.get("domain")))
        if 0 < len(synonyms) < 10:
            #  if 'drug' in domain or 'measurement' in domain:
            #   print(f"domain = {domain} for concept = {label}")
            align_synonyms = list(node.get("aligned_synonyms", []))
            if len(align_synonyms) > 0:
                synonyms.extend(align_synonyms)
            if len(synonyms) < 10:
                # hier_synonyms = list(node.get('hier_syn', []))
                # if len(hier_synonyms) > 0:
                #         synonyms.extend(hier_synonyms)
                maps_to_syn = list(node.get("maps_to_syn", []))
                if len(maps_to_syn) != 0:
                    synonyms.extend(maps_to_syn)
        synonyms = list(set(synonyms))
        if len(synonyms) > 1:
            synonyms_str = ";;".join([f"{syn}" for syn in synonyms[:10] if syn != label and syn != ""])
        elif len(synonyms) == 1:
            synonyms_str = synonyms[0]
        else:
            synonyms_str = ""
        ontology_ = str(node.get("ontology", "")).lower()
        has_answers = node.get("has_answers", [])
        answer_of = node.get("answer_of", None)
        #  if ontology_ in node_type or 'drug' in node_type:
        #     node_type = node_type.split(':')[1]
        #  page_content = f"<ENT>{label}||{domain}</ENT><SYN>{synonyms_str}</SYN>"
        description = f"concept name:{label}, synonyms:{synonyms_str}, domain:{domain}, concept Class:{concept_class}, vocabulary:{ontology_}"
        #  page_content  = f"{page_content}<DESC>{description}</DESC>"
        hier_terms = ";;".join(node.get("hier_syn")) if len(node.get("hier_syn")) > 0 else ""
        #  ctype_  = node.get('type')
        if domain is None:
            print(f"Domain Type is None for {node_id}")
        is_standard = node.get("type_standard", "")
        metadata_ = {
            "label": label,
            "domain": domain,
            "concept_class": concept_class,
            "vocab": ontology_,
            "parent_term": hier_terms,
            "scode": str(node.get("code")),
            "sid": str(node_id),
            "synonyms": synonyms_str,
            "answer_of": answer_of,
            "has_answers": ";;".join(has_answers),
            "is_standard": is_standard,
        }
        return Document(id=index, page_content=description, metadata=metadata_), len(description)


class ConceptLoader(BaseLoader):
    def __init__(self, concept_file, only_Syn=True):
        self.concepts = pd.read_csv(concept_file, low_memory=False, dtype=str)
        self.only_syn = only_Syn
        print(self.concepts.columns)

    def load(self) -> List[Document]:
        """Load and return documents from omop concept knowledge base"""
        docs: List[Document] = []
        for index, row in tqdm(self.concepts.iterrows(), total=self.concepts.shape[0], desc="Retrieving documents"):
            docs.append(self._create_document(index, row, self.only_syn))
        return docs

    def load_text(self) -> List[str]:
        docs: List[str] = []
        for _, row in tqdm(self.concepts.iterrows(), total=self.concepts.shape[0], desc="Retrieving text"):
            label = str(row["concept_name"]).strip()
            if not pd.isna(row["concept_synonym_name"]):
                label += f",synonyms:{row['concept_synonym_name']}"
            label = f"""{label},domain: {row['domain_id']!s}, vocabulary class: {row['concept_class_id']!s}, ontology: {row['vocabulary_id']!s}
            """
            docs.append(label)
        return docs

    def _create_document(self, index, result_row, only_Syn) -> Document:
        """Create a Document object from a query result row."""
        label = str(result_row["concept_name"])
        synonyms = set(
            result_row["concept_synonym_name"].split(";;")
            if not pd.isna(result_row["concept_synonym_name"])
            else [label]
        )

        # Updating the set with the label to ensure it includes all variants
        synonyms.add(label)

        # Check if only synonyms should be used and if synonyms are present
        if only_Syn:
            page_content = ";;".join(synonyms)
        else:
            additional_info = f", domain: {result_row['domain_id']!s}, vocabulary class: {result_row['concept_class_id']!s}, ontology: {result_row['vocabulary_id']!s}"
            page_content = f"{label}{additional_info}"

        synonyms_str = ";;".join(synonyms)  # Convert set to string
        label = synonyms_str
        return Document(
            page_content=label,
            metadata={
                "label": label,
                "synonyms": synonyms_str,
                "cid": str(result_row["concept_id"]),
                "domain": str(result_row["domain_id"]),
                "semantic_type": str(result_row["concept_class_id"]),
                "ontology": str(result_row["vocabulary_id"]),
                "code": str(result_row["concept_code"]),
                "source": f"doc:{index}",
            },
        )


def load_concepts(file_path, data_dir="/workspace/mapping_tool/data"):
    start_time = datetime.datetime.now()
    loader = GraphLoader(file_path, data_dir)
    concepts, max_len = loader.load_graph()
    end_time = datetime.datetime.now()
    # file_name = os.path.basename(file_path).split('.')[0]
    # save_docs_to_jsonl(concepts,f"{output_dir}")
    print(f"Duration: {end_time - start_time}")
    return concepts, max_len


def load_discharge_summaries(discharge_summaries_file, output_dir):
    start_time = datetime.datetime.now()
    loader = CSVLoader(file_path=discharge_summaries_file)
    discharge_docs = loader.load()
    file_name = os.path.basename(discharge_summaries_file).split(".")[0]
    save_docs_to_jsonl(discharge_docs, f"{output_dir}/{file_name}")
    end_time = datetime.datetime.now()
    print(f"Duration: {end_time - start_time}")


def split_text(documents, chunk_size, over_lap_size):
    parent_splitters = []
    for size, over_lap in tqdm(zip(chunk_size, over_lap_size), total=len(over_lap_size)):
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=over_lap, separators=".", keep_separator=True
        )
        chunks = parent_splitter.split_documents(documents)
        parent_splitters.append(chunks)
    return parent_splitters


"""
Change the output_dir to OUPUT_DATA_PATH for all concepts or SNOMED_DATA_PATH for specific vocab such as SNOMED
"""


def load_docs(data_dir=DATA_DIR, document_file_name=None, output_dir=OUTPUT_DATA_PATH, mode="inference"):
    docs = None
    logger.info(f"Mode: {mode}")
    if mode in ["recreate", "inference"]:
        if os.path.exists(output_dir):
            docs = load_docs_from_jsonl(output_dir)
            logger.info("Knowledge Base Loaded")
        elif os.path.exists(data_dir):
            concepts, max_len = load_concepts(document_file_name, data_dir)
            chunk_size = max_len
            overlap = 0
            if max_len > 1300:
                chunk_size = 1300
                overlap = 100
            docs = split_text(concepts, [chunk_size], [overlap])[0]
            save_docs_to_jsonl(docs, output_dir)
            logger.info("Documents processed into chunks")
    elif mode == "update":
        main_base = os.path.join(data_dir, "output/concepts.jsonl")
        documents = None
        if document_file_name:
            document_path = os.path.join(data_dir, f"output/{document_file_name}")
            if not os.path.exists(document_path):
                concepts, max_len = load_concepts(document_file_name, data_dir=data_dir)
                chunk_size = max_len
                overlap = 0
                if max_len > 1300:
                    chunk_size = 1300
                    overlap = 100
                docs = split_text(concepts, [chunk_size], [overlap])[0]
            else:
                print(f"Document File Name: {document_file_name}")
                docs = load_docs_from_jsonl(document_path)

        if os.path.exists(main_base):
            print(f"Main Base Exists={main_base}")
            docs = load_docs_from_jsonl(main_base)
            logger.info("Knowledge Base Loaded")
            if docs and documents:
                docs.extend(documents)
                save_docs_to_jsonl(docs, output_dir)
        else:
            print("Only updated documents loaded")
            docs = documents
        logger.info("Documents processed into chunks")

    return docs


def load_data(input_file, load_custom=False):
    try:
        queries = []
        is_mapped = False
        if load_custom:
            print("load custom data")
            # its csv file with columns VARIABLE NAME	VARIABLE LABEL	VAR TYPE	UNITS	CATEGORICAL
            queries, is_mapped = custom_data_loader(input_file)
        else:
            if input_file.endswith(".csv"):
                with open(input_file, encoding="utf-8") as file:
                    reader = csv.reader(file)
                    data = []
                    for row in reader:
                        # print(f"Row: {row}")
                        if len(row) < 2:  # Ensure the row has exactly 3 columns
                            print(f"Skipping malformed row: {row}")
                            continue
                        key = row[0].strip()
                        mention = row[1]
                        # domain = row[2].strip().lower() if len(row) > 2 else None
                        # Split values on pipe or comma and clean them
                        # answers = {val.strip().lower() for part in values.split('|') for val in part.split(",")}
                        # print(f"Key: {key}, Values: {values}, Domain: {domain}")
                        queries.append(
                            key,
                            QueryDecomposedModel(
                                full_query=mention,
                                base_entity=mention,
                                categories=[],
                                unit=None,
                                formula=None,
                                domain="all",
                                rel=None,
                                original_label=mention,
                            ),
                        )
            elif input_file.endswith(".json"):
                with open(input_file, encoding="utf-8") as file:
                    data = json.load(file)
                    queries = [
                        (
                            item["cui"],
                            QueryDecomposedModel(
                                full_query=item["query"],
                                base_entity=item["query"],
                                categories=[],
                                unit=None,
                                formula=None,
                                domain="all",
                                rel=None,
                                original_label=item["query"],
                            ),
                        )
                        for item in data
                    ]
            elif input_file.endswith(".txt"):
                with open(input_file, encoding="utf-8") as file:
                    for line in tqdm(file, desc="Loading queries from input file"):
                        line = line.strip()
                        if line == "":
                            continue
                        parts = line.split("||")
                        # print(f"Parts: {parts}")
                        # if len(parts) < 2 and len(parts) > 0:
                        if len(parts) == 2:
                            cui, query = str(parts[0]).lower(), normalize(parts[1])
                            if cui == "cui-less":
                                continue
                            queries.append(
                                (
                                    cui,
                                    QueryDecomposedModel(
                                        full_query=query,
                                        base_entity=query,
                                        categories=[],
                                        unit=None,
                                        formula=None,
                                        domain="all",
                                        rel=None,
                                        original_label=query,
                                    ),
                                )
                            )
                        elif len(parts) > 2:
                            cui, query, _ = str(parts[0]).lower(), normalize(parts[1]), normalize(parts[2])
                            if cui == "cui-less":
                                continue
                            queries.append(
                                cui,
                                QueryDecomposedModel(
                                    full_query=query,
                                    base_entity=query,
                                    categories=[],
                                    unit=None,
                                    formula=None,
                                    domain="all",
                                    rel=None,
                                    original_label=query,
                                ),
                            )
                            # print(f"CUI: {cui}, Query: {query}")

                        # else:
                        #     if cui != 'cui-less':
                        #         # print(f"CUI: {cui}, Query: {query}")
                        #         queries.append((cui, query))

        # filter duplicates
        print(f"Total queries loaded = {len(queries)}")
        # queries = list(dict.fromkeys(queries))
        # # queries = np.array(queries)
        # print(f"Total unique queries loaded = {len(queries)}")
        # random.shuffle(queries)
        return queries, is_mapped
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, is_mapped


# import argparse
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="Build a synonym graph")
#     parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
#     parser.add_argument("--output_dir", default=f"{args.data_dir}/output/concept.jsonl", type=str, help="Path to the output directory")
#     args = parser.parse_args()
#     # input_file = os.path.basename(input_file)
#     data_dir = args.data_dir
#     output_dir = args.output_dir
#     print(f"data_dir={data_dir}")
#     print(f"data_dir={data_dir}")
#     concepts, max_len = load_concepts(None, data_dir)
#     chunk_size =  max_len
#     overlap = 0
#     if max_len > 1300:
#         chunk_size = 1300
#         overlap = 100
#     documents = split_text(concepts, [chunk_size], [overlap])[0]
#     save_docs_to_jsonl(documents, output_dir)
