from typing import Any, Iterable, List, Dict
import pandas as pd
import os
# from fastembed.common.model_description import PoolingType, ModelSource
# from fastembed.embedding import FlagEmbedding as Embedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams
    

)
from .utils import load_dictionary
import glob
import uuid
from .modes import MappingType, EmbeddingType 
# from kg.utils import BOLD, CYAN, END

EMBEDDING_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# EMBEDDING_MODEL_SIZE = 768
def get_embedding_model(model_name="biolord"):
    from .lazy_model import get_model
    return get_model(backend=model_name)
# COLLECTION_NAME = "studies_metadata"

# TextEmbedding.add_custom_model(
#     model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
#     pooling=PoolingType.MEAN,
#     normalization=True,
#     sources=ModelSource(hf="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"),  # can be used with an `url` to load files from a private storage
#     dim=786
# )


def load_vectordb(collection_name:str, vectordb_path: str ="komal.qdrant.137.120.31.148.nip.io",  recreate: bool = False, model_name: str = "biolord") -> tuple[QdrantClient, 'ModelEmbedding']:
    """
    Initialize the embedding model and Qdrant vector database.
    Connects to Qdrant at the provided host (e.g., 'localhost').
    
    If recreate is True, the collection is deleted (if it exists) and then created.
    Otherwise, if the collection does not exist, it is created.
    """
    print("📥 Loading embedding model")
    embedding_model, embedding_size = get_embedding_model(model_name=model_name)
    
    # TextEmbedding(
    #     model_name=EMBEDDING_MODEL_NAME, 
    #     max_length=512, 
    #     cache_dir="/Users/komalgilani/Desktop/chexo_knowledge_graph/data/models"
    # )
    vectordb = QdrantClient(host=vectordb_path, port=6333)

    try:
        # Attempt to retrieve the collection.
        collection_info = vectordb.get_collection(collection_name)
        current_count = collection_info.points_count
        print(f"Total vectors in DB: {current_count}")
        if recreate:
            print(f"🔄 Recreating collection '{collection_name}' on {vectordb_path}")
            vectordb.delete_collection(collection_name)
            vectordb.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )
    except Exception as e:
        print(f"Collection not found or error occurred: {e}")
        print(f"Creating new collection '{collection_name}' on {vectordb_path}")
        vectordb.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )
    return vectordb, embedding_model

def get_csv_text(row: Dict[str, Any], embedding_mode: str = EmbeddingType.ED.value) -> str:
    parts = []
    row = {k.lower(): v for k, v in row.items()}

    var_label = row.get('variable label', row.get('variable name', '')).strip().lower() 
    # print(f"var_label={var_label}")
    var_concept = row.get('variable concept name', '').strip().lower() if pd.notna(row.get('variable concept name', '')) else None
    additional_context = row.get('additional context concept name', '').strip().lower() if pd.notna(row.get('additional context concept name', '')) else None

    if embedding_mode == EmbeddingType.EH.value:
        if var_label and str(var_label).strip():
            parts.append(var_label.strip())
    # Always use the label as base text if present
        if var_concept and str(var_concept).strip() and var_concept != var_label:
            parts.append(var_concept.strip())

            if additional_context and str(additional_context).strip() and additional_context != var_label and additional_context != var_concept:
                if additional_context != var_label and additional_context != var_concept:
                    parts.append(str(additional_context).strip())
    elif embedding_mode == EmbeddingType.EC.value:
        if var_concept and str(var_concept).strip():
            parts.append(var_concept.strip())
            if additional_context and str(additional_context).strip() and additional_context != var_concept:
                parts.append(str(additional_context).strip())
        else:
            if var_label and str(var_label).strip():
                parts.append(var_label.strip())
    elif embedding_mode == EmbeddingType.ED.value:
           
        if var_label and str(var_label).strip():
            parts.append(var_label.strip())
    # if var_label and str(var_label).strip():
    #     parts.append(var_label.strip())
    final_text = " ".join(parts).replace("\n", "").strip()
    # print(f"final_text={final_text}")
    return final_text




def load_csv_points(csv_path: str, study_name: str = None, embedding_mode: str = EmbeddingType.EC.value) -> List[Dict[str, Any]]:
    """
    Reads a CSV file containing your data dictionary and converts each row into a dictionary with an embedding text.
    Optionally includes additional metadata such as a study name if available.
    """
    df = load_dictionary(csv_path)
    # lowercase all column names
    df.columns = [col.lower() for col in df.columns]
    if 'variablename' in df.columns:
        # replace column name 'variablename' with 'variable name'
        df = df.rename(columns={col: 'variable name' for col in df.columns if col == 'variablename'})
    if 'variablelabel' in df.columns:
        # replace column name 'variablelabel' with 'variable label'
        df = df.rename(columns={col: 'variable label' for col in df.columns if col == 'variablelabel'})
    # study_name = str(os.path.basename(csv_path).split(".")[0]).lower()
    # print(f"study_name={study_name}")
    points = []
    print(df.head(2))
    print(f"length of data frame={len(df)}")
    for index, row in df.iterrows():
        text = get_csv_text(row, embedding_mode=embedding_mode)
        if text == "":
            continue
        # Generate a unique id (for example, use the row index or hash of the text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{study_name}_{index}"))
        
        # Add study name to metadata if available
        # 
        # make two dicts one of str and other of int
        var_data_int = {k.lower().strip().replace(" ", "_"): int(v) for k, v in row.to_dict().items() if isinstance(v, int) or (isinstance(v, float) and v.is_integer())}
        var_data_str = {k.lower().strip().replace(" ", "_"): str(v).lower().strip() for k, v in row.to_dict().items() if isinstance(v, str)}
        # replace key name 'variablelabel' with 'variable_label' in both dicts
        # if 'variablelabel' in var_data_str:
        #     var_data_str['variable_label'] = var_data_str.pop('variablelabel')
        # if 'variablename' in var_data_int:
        #     var_data_int['variable_name'] = var_data_int.pop('variablename')
        # merge two dicts
        var_data = {**var_data_str, **var_data_int}
        # keep int data types for specific fields
        # print(f"var_data before={var_data}")
        var_data["study_name"] = study_name
        # var_data['domain'] = row.get('domain', '').strip().lower() if pd.notna(row.get('domain', '')) else ''
 
        # print(f"metadata={metadata}")
        metadata = {}
        metadata["text"] = text
        metadata["id"] = point_id
        metadata["metadata"] = var_data
        # print(f"text={text}")   
        print(metadata)
        points.append(metadata)
    return points


def insert_in_db(vectordb: QdrantClient, embedding_model: 'ModelEmbedding', points: Iterable[Dict[str, Any]], collection_name:str) -> None:
    """
    Inserts a list of points into the Qdrant collection.
    Each point should have a "text" field (for embedding) and any additional metadata.
    """
    point_structs = []
    for point in points:
        text = point.get("text", "")
        if not text:
            continue  # Skip if there's no text to embed.
        vector = embedding_model.embed_text(text)
        # print(f"vector={type(vector)}")
        point_id = point.get("id", abs(hash(text)) % (10 ** 8))
        point_structs.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=point.get("metadata", {}),
            )
        )
    if point_structs:
        vectordb.upsert(collection_name=collection_name, wait=True, points=point_structs)
        print(f"Inserted {len(point_structs)} points into Qdrant.")
    else:
        print("No valid points to insert.")



def search_in_db(vectordb: QdrantClient, embedding_model: 'ModelEmbedding', query_text: str, limit: int = 20, target_study:str="gissi-hf", omop_domain:List[str]=["drug_exposure","condition_occurrence","condition_era","observation","observation_era","measurement","visit_occurrence","procedure_occurrence","device_exposure","person"], min_score:int=0.4, collection_name:str="studies_metadata") -> List[Any]:
        query_vector =  embedding_model.embed_text(query_text)
        # print(f"query_vector length={len(query_vector)}")
        results = vectordb.search(
            collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=min_score,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key='study_name',
                            match=models.MatchValue(
                                value=target_study
                            )
                        ),
                        models.FieldCondition(
                            key='domain',
                            match=models.MatchAny(
                                any=omop_domain
                            )
                        )

                    ]
                )
        
        )
        var_labels = []
        if results:
            for res in results:
                    var_labels.append(int(res.payload['variable_omop_id']))
        return var_labels


         

def search_in_db_group_by(vectordb: QdrantClient, embedding_model: 'ModelEmbedding', query_text: str, limit: int = 100, target_study:str="gissi-hf", omop_domain:str="drug_exposure", collection_name:str="studies_metadata") -> List[Any]:
    """
    Searches the Qdrant collection for points similar to the provided query text.
    Returns the top 'limit' results with payload metadata.
    """
    query_vector =  embedding_model.embed_text(query_text)
    target_ids = set()
    results = vectordb.query_points_groups(
        collection_name=collection_name,
        group_by="variable_omop_id",
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        score_threshold=0.3,
        group_size=12,
        query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key='study_name',
                            match=models.MatchValue(
                                value=target_study
                            )
                        ),
                         models.FieldCondition(
                            key='domain',
                            match=models.MatchValue(
                                value=omop_domain
                            )
                        )

                    ]
                )
        )
    
    if results:
        for res in results.groups:
             for result in res.hits:
                target_ids.add(int(result.payload['variable_omop_id']))
    # print(f"total target ids found via similarity search={len(target_ids)}")
    return list(target_ids)

def generate_studies_embeddings(dir_path:str, vectordb_path:str, collection_name:str, model_name:str, embedding_mode:str=EmbeddingType.EC.value, recreate_db=False):
   
    vectordb, embedding_model = load_vectordb(recreate=recreate_db, collection_name=collection_name, vectordb_path=vectordb_path, model_name=model_name)
    if not recreate_db:
        return vectordb, embedding_model
    for cohort_folder in os.listdir(dir_path):
        if cohort_folder.startswith('.'):  # Skip hidden files like .DS_Store
            continue
        cohort_path = os.path.join(dir_path, cohort_folder)
        print(f"cohort_path={cohort_path}")
        study_name = cohort_folder.lower()
        print(f"study_name={study_name}") 
        if os.path.isdir(cohort_path):
                # ➊ Grab every file that ends with .csv, .xlsx or .json
                patterns = ("*.csv", "*.xlsx", "*.json")
                file_candidates: list[str] = []
                for pat in patterns:
                    file_candidates.extend(glob.glob(os.path.join(cohort_path, pat)))
                cohort_metadata_file = None
                eda_file = None
                # ➋ Classify the candidates
                for file in file_candidates:
                    print(f"File: {file}")
                    # Collect *all* metadata spreadsheets
                    if file.lower().endswith((".csv", ".xlsx")):
                        cohort_metadata_file = file
                        break
        print(cohort_metadata_file)
        print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for metadata file: {cohort_metadata_file}")
        if cohort_metadata_file and os.path.exists(cohort_metadata_file):
            points = load_csv_points(cohort_metadata_file, study_name, embedding_mode)
            print(f"Number of points to insert: {len(points)} for {cohort_folder}")
            # divide points into batches of 500
            batch_size = 500
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                insert_in_db(vectordb, embedding_model, batch_points, collection_name)
                # insert_in_db(vectordb, embedding_model, points, collection_name)
            try: 
                count = vectordb.get_collection(collection_name).points_count
                print(f"Updated vector count: {count}")
            except Exception as e:
                print(f"Error retrieving vector count: {e}")
    return vectordb, embedding_model




