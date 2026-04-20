
import os
import glob
import uuid
import threading
import pandas as pd
from typing import Any, List, Dict, Tuple
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams
)

import numpy as np
from .fuzz_match import FuzzyMatcher
from .data_model import MappingType, EmbeddingType 
from .config import settings
from .utils import (
    load_dictionary,
    get_embedding_model, 
    clean_label_remove_temporal_context,
    split_categories
)

# Thread lock for Qdrant client (httpx is not thread-safe)
_qdrant_lock = threading.Lock()
_embed_cache: Dict[str, Any] = {}


def load_vectordb(collection_name:str, vectordb_path: str ="komal.qdrant.137.120.31.148.nip.io",  recreate: bool = False, model_name: str = "biolord") -> tuple[Any, Any]:
    """
    Initialize the embedding model and Qdrant vector database.
    Connects to Qdrant at the provided host (e.g., 'localhost').
    
    If recreate is True, the collection is deleted (if it exists) and then created.
    Otherwise, if the collection does not exist, it is created.
    """
    # print("📥 Loading embedding model")
    if model_name is None or collection_name is None:
        return None, None
    embedding_model, embedding_size = get_embedding_model(model_name=model_name)
    vectordb = QdrantClient(host=vectordb_path, port=6333, timeout=240)

    try:
        if recreate:
            print(f"🔄 Recreating collection '{collection_name}' on {vectordb_path}")
            vectordb.delete_collection(collection_name)
            vectordb.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(
                    m=32,  # Balanced connections (default)
                    ef_construct=200,  # Good build quality (default)
                    full_scan_threshold=10000,  # Use brute force below this size (default)
            ),
            optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=100,  # Use brute force below this size (default)
                    ),
            )
    except Exception as e:
        print(f"Collection not found or error occurred: {e}")
        print(f"Creating new collection '{collection_name}' on {vectordb_path}")
        vectordb.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(
                m=32,  # Balanced connections (default)
                ef_construct=200,  # Good build quality (default)
                full_scan_threshold=10000,  # Use brute force below this size (default)
        ),
         optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=100,  # Use brute force below this size (default)
                ),
        )
    return vectordb, embedding_model

   

def extract_field_values(row: Dict[str, Any], *field_names: str) -> List[str]:
    """
    Extract pipe-separated values from CSV fields, trying alternatives in order.
    Returns a list of stripped, lowercase values.
    
    Args:
        row: Dictionary from CSV row
        field_names: Field names to try in order (uses first one that has data)
    
    Returns:
        List of extracted values, empty list if none found
    """
    row_lower = {k.lower(): v for k, v in row.items()}
    
    for field_name in field_names:
        if field_name is None:
                continue
        field_key = field_name.lower()
        value = row_lower.get(field_key, '')
        
        if pd.notna(value) and isinstance(value, str):
            value = value.strip().lower()
            if value and value != '':
                return [v.strip() for v in value.split("|") if v.strip()]
    
    return []


def get_csv_text(row: Dict[str, Any], embedding_mode: str = EmbeddingType.ED.value) -> Tuple[List[str], List[str]]:
    """
    Generate embedding text from CSV row based on embedding mode.
    
    Args:
        row: Dictionary from CSV row
        embedding_mode: One of EmbeddingType (ED, EC, EH)
    
    Returns:
        Tuple of (embedding_text, category_labels_list)
    """
    parts = []
    row = {k.lower(): v for k, v in row.items()}

    var_label = row.get('variable label', row.get('variable name', '')).strip().lower() 
    var_label = clean_label_remove_temporal_context(var_label)
    var_concept = row.get('variable concept name', '').strip().lower() if pd.notna(row.get('variable concept name', '')) else None
    additional_context = row.get('additional context concept name', '').strip().lower() if pd.notna(row.get('additional context concept name', '')) else None
    
    # Extract categories as list (not just a set)
    categories_list = extract_field_values(row, 'categorical value concept name', None)
    # unit = row.get('units', '').strip().lower() if pd.notna(row.get('units', '')) else None
    # original_categories = row.get('original categories', '').strip().lower() if pd.notna(row.get('original categories', '')) else None
    if embedding_mode == EmbeddingType.EH.value:
        if var_label and str(var_label).strip():
            parts.append(var_label.strip())
        if var_concept and str(var_concept).strip() and var_concept != var_label:
            parts.append(var_concept.strip())
        if additional_context and str(additional_context).strip() and additional_context != var_label and additional_context != var_concept:
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
            # if unit and str(unit).strip():
            #     parts.append(f"unit:{unit.strip()}")
            # elif original_categories and len(original_categories) > 0:
            #     parts.append(f"values:{'|'.join(original_categories)}")
    
    # final_text = " ".join(parts).replace("\n", "").strip()
    
    return parts, categories_list


def _prepare_var_metadata(row: Dict[str, Any], study_name: str) -> Dict[str, Any]:
    """
    Prepare metadata dictionary from CSV row.
    Separates string and integer fields.
    """
    row_lower = {k.lower(): v for k, v in row.items()}
    
    var_data_int = {
        k.lower().strip().replace(" ", "_"): int(v) 
        for k, v in row_lower.items() 
        if isinstance(v, (int, float)) and (isinstance(v, int) or v.is_integer())
    }
    
    var_data_str = {
        k.lower().strip().replace(" ", "_"): str(v).lower().strip() 
        for k, v in row_lower.items() 
        if isinstance(v, str)
    }
    
    var_data = {**var_data_str, **var_data_int}
    var_data["study_name"] = study_name
    
    return var_data

# def _composite_concept_key(row: Dict[str, Any], text: str, 
#                            embedding_mode: str) -> str:
#     """Build a dedup key that preserves compositional identity.
    
#     Combines: variable OMOP ID + sorted additional context OMOP IDs.
#     Two rows with same base concept but different post-coordination
#     remain distinct. Rows differing only in visit are collapsed.
#     """
#     if embedding_mode == EmbeddingType.ED.value:
#         # NE mode: no ontology, label is the identity
#         return text

#     omop_id = str(row.get('variable omop id', '')).strip()
    
#     # Additional context: pipe-separated, may be empty
#     ctx_raw = row.get('additional context omop id', '')
#     if pd.notna(ctx_raw) and str(ctx_raw).strip():
#         ctx_ids = sorted(s.strip() for s in str(ctx_raw).split('|') if s.strip())
#     else:
#         ctx_ids = []

#     return f"{omop_id}||{'|'.join(ctx_ids)}"

# def load_csv_points_dedup(csv_path: str, study_name: str = None,
#                     embedding_mode: str = EmbeddingType.EC.value) -> List[Dict[str, Any]]:
#     create_category_points = embedding_mode != EmbeddingType.ED.value
#     df = load_dictionary(csv_path)
#     df.columns = [col.lower() for col in df.columns]
#     df = df.rename(columns={
#         'variablename': 'variable name', 'variablelabel': 'variable label',
#          'visits': 'visit',
#          'units': 'unit'})

#     # ── Phase 1: Group rows by unique embedding text ──
#     concept_groups = {}  # embed_text → { metadata, visits[], row_indices[] }
#     for index, row in df.iterrows():
#         if pd.isna(row.get('variable name', '')):
#             continue
#         parts, concept_categories = get_csv_text(row, embedding_mode=embedding_mode)
#         if not parts:
#             continue

#         text = " ".join(FuzzyMatcher._deduplicate_parts(parts=parts, threshold=0.8))

#         # Concept-level dedup key: (embedding_text, omop_id or variable_name)
#         # if embedding_mode == EmbeddingType.ED.value:
#         #     concept_key = text  # NE mode: label is the identity
#         # else:
#         #     omop_val = str(row.get('variable omop id', '')).strip().lower()
#         concept_key = _composite_concept_key(row, text, embedding_mode)

#         visit = str(row.get('visit','')).strip().lower()
#         var_name = str(row.get('variable name', '')).strip()

#         if concept_key not in concept_groups:
#             var_data = _prepare_var_metadata(row, study_name)
#             concept_val = str(row.get('variable omop id', row.get('variable name', '')))
#             var_data['parent_variable_concept'] = (
#                 concept_val.lower().strip() if pd.notna(concept_val) else None)
#             original_categories, category_labels = split_categories(
#                 row.get('categorical', None))
#             var_data.pop('categorical', None)
#             var_data['original_categories'] = "|".join(original_categories)
#             var_data['original_categories_labels'] = "|".join(category_labels)

#             concept_groups[concept_key] = {
#                 "text": text,
#                 "metadata": var_data,
#                 "visits": [],
#                 "var_names": [],
#                 "categories": concept_categories,
#                 "first_index": index,
#             }
#         concept_groups[concept_key]["visits"].append(visit)
#         concept_groups[concept_key]["var_names"].append(var_name)

#     # ── Phase 2: One point per unique concept ──
#     print(f"🔍 Found {len(concept_groups)} unique concepts in {csv_path}")
#     points = []
#     for ckey, group in concept_groups.items():
#         point_id = str(uuid.uuid5(
#             uuid.NAMESPACE_DNS, f"{study_name}_{group['first_index']}"))
#         meta = group["metadata"]
#         meta['parent_variable_id'] = point_id
#         meta['visits'] = "|".join(sorted(set(group["visits"])))
#         meta['variable_names'] = "|".join(sorted(set(group["var_names"])))
#         meta['visit_count'] = len(set(group["visits"]))

#         points.append({
#             "text": group["text"],
#             "id": point_id,
#             "metadata": {**meta, "is_category": 0, "point_type": "variable"},
#         })

#         # Category points — one per unique category (unchanged logic)
#         if create_category_points and group["categories"]:
#             cat_codes = extract_field_values(
#                 {k: meta.get(k, '') for k in meta},
#                 'categorical_value_concept_code')
#             cat_omop_ids = extract_field_values(
#                 {k: meta.get(k, '') for k in meta},
#                 'categorical_value_omop_id')
#             for idx, cat_label in enumerate(group["categories"]):
#                 cat_point_id = str(uuid.uuid5(
#                     uuid.NAMESPACE_DNS, f"{study_name}_cat_{cat_label}_{idx}"))
#                 cat_meta = meta.copy()
#                 cat_meta.update({
#                     'is_category': 1, 'point_type': 'category',
#                     'category_label': cat_label,
#                     'category_code': cat_codes[idx] if idx < len(cat_codes) else None,
#                     'category_omop_id': cat_omop_ids[idx] if idx < len(cat_omop_ids) else None,
#                 })
#                 points.append({"text": cat_label, "id": cat_point_id, "metadata": cat_meta})

#     print(f"📊 {len(points)} points from {csv_path} "
#           f"(deduped from {len(df)} rows, categories={'yes' if create_category_points else 'no'})")
#     return points

# def load_csv_points(csv_path: str, study_name: str = None, 
#                     embedding_mode: str = EmbeddingType.EC.value) -> List[Dict[str, Any]]:
#     """
#     Convert CSV data dictionary rows into vector DB points.
    
#     Args:
#         create_category_points: If True (ontology modes), create separate points per category
#                                 for OMOP-based subsumption search. If False (NE mode), only
#                                 create variable-level points with categories in payload.

#     """
#     create_category_points = embedding_mode != EmbeddingType.ED.value
#     df = load_dictionary(csv_path)
#     df.columns = [col.lower() for col in df.columns]
#     df = df.rename(columns={'variablename': 'variable name', 'variablelabel': 'variable label'})

#     points = []
#     for index, row in df.iterrows():
#         if pd.isna(row.get('variable name', '')):
#             continue

#         parts, concept_categories = get_csv_text(row, embedding_mode=embedding_mode)
#         if not parts:
#             continue

#         point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{study_name}_{index}"))
#         var_data = _prepare_var_metadata(row, study_name)
#         var_data['parent_variable_id'] = point_id
#         concept_val = str(row.get('variable omop id', row.get('variable name', '')))
#         var_data['parent_variable_concept'] = concept_val.lower().strip() if pd.notna(concept_val) else None

#         # Parse raw categorical column → original codes + labels
#         original_categories, category_labels = split_categories(row.get('categorical', None))
#         var_data.pop('categorical', None)
#         var_data['original_categories'] = "|".join(original_categories)
#         var_data['original_categories_labels'] = "|".join(category_labels)

#         text = " ".join(FuzzyMatcher._deduplicate_parts(parts=parts, threshold=0.8))
#         points.append({
#             "text": text,
#             "id": point_id,
#             "metadata": {**var_data, "is_category": 0, "point_type": "variable"},
#         })

#         # Category points — only for ontology modes (OMOP-based subsumption search)
#         if create_category_points and concept_categories:
#             cat_codes = extract_field_values(row, 'categorical value concept code')
#             cat_omop_ids = extract_field_values(row, 'categorical value omop id')

#             for idx, cat_label in enumerate(concept_categories):
#                 cat_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{study_name}_cat_{cat_label}_{idx}"))
#                 cat_metadata = var_data.copy()
#                 cat_metadata.update({
#                     'is_category': 1,
#                     'point_type': 'category',
#                     'category_label': cat_label,
#                     'category_code': cat_codes[idx] if idx < len(cat_codes) else None,
#                     'category_omop_id': cat_omop_ids[idx] if idx < len(cat_omop_ids) else None,
#                 })
#                 points.append({"text": cat_label, "id": cat_point_id, "metadata": cat_metadata})

#     print(f"📊 {len(points)} points from {csv_path} (categories={'yes' if create_category_points else 'no'})")
#     return points


def insert_in_db(vectordb, embedding_model, points, collection_name):
    valid_points = [(p, p["text"]) for p in points if p.get("text")]
    if not valid_points:
        print("⚠️ No valid points to insert")
        return

    # Batch encode all texts in one forward pass
    texts = [text for _, text in valid_points]
    vectors = embedding_model.embed_batch(texts, is_query=False)  # corpus side

    point_structs = []
    for (point, _), vector in zip(valid_points, vectors):
        point_structs.append(PointStruct(
            id=point.get("id", abs(hash(point["text"])) % (10**8)),
            vector=vector.tolist(),
            payload=point.get("metadata", {}),
        ))

    vectordb.upsert(collection_name=collection_name, wait=True, points=point_structs)
    print(f"✔ Inserted {len(point_structs)} points into Qdrant")

# def search_in_db(vectordb: QdrantClient, embedding_model: Any, query_text: str, 
#     limit: int = 20, target_study:list[str]=["gissi-hf"],
#      data_domain:List[str]=settings.DATA_DOMAINS, min_score:int=0.4, 
#      collection_name:str="studies_metadata",
#      mapping_mode:str=MappingType.OEH.value) -> List[Any]:
#     """
#     Search the vector database and return variable OMOP IDs.
#     """
#     query_text = clean_label_remove_temporal_context(query_text)
#     if query_text in _embed_cache:
#         query_vector = _embed_cache[query_text]
#     else:
#         query_vector = embedding_model.embed_text(query_text, is_query=True)
#         _embed_cache[query_text] = query_vector
    
#     with _qdrant_lock:
#         results = vectordb.search(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=limit,
#             # score_threshold=min_score,
#             search_params=models.SearchParams(
#                 # hnsw_ef=256,   # ≥ limit, higher = better recall, slower
#                 exact=True,    # set True to bypass HNSW for brute-force (small collections)
#             ),
#             query_filter=models.Filter(
#                 must=[
#                     models.FieldCondition(
#                         key='study_name',
#                         match=models.MatchAny(any=target_study)
#                     ),
#                     # models.FieldCondition(
#                     #     key='domain',
#                     #     match=models.MatchAny(any=data_domain)
#                     # ),
#                     models.FieldCondition(
#                         key='is_category',
#                         match=models.MatchValue(value=0)
#                     )
#                 ]
#             )
#         )
    
#     results_out = []
#     seen = set()
#     for res in results:
#         if mapping_mode == MappingType.NE.value:
#             key = res.payload.get('variable_name', '')
#         else:
#             key = int(res.payload['variable_omop_id'])
#         if key not in seen:
#             seen.add(key)
#             results_out.append(key)
#     return results_out
    

# def clear_embed_cache():
#     """Clear the embedding cache (e.g. when the embedding model changes)."""
#     _embed_cache.clear()

def _cache_key(model_name: str, text: str) -> str:
    return f"{model_name}::{text}"


def _composite_concept_key(row: Dict[str, Any], text: str,
                           embedding_mode: str) -> str:
    """Dedup key preserving compositional identity.
    
    Same base OMOP + same additional context = same concept (collapse visits).
    Same base OMOP + different context = different concepts (keep separate).
    NE mode: cleaned label text is the identity.
    """
    if embedding_mode == EmbeddingType.ED.value:
        return text

    omop_id = str(row.get('variable omop id', '')).strip()
    ctx_raw = row.get('additional context omop id', '')
    if pd.notna(ctx_raw) and str(ctx_raw).strip():
        ctx_ids = sorted(s.strip() for s in str(ctx_raw).split('|') if s.strip())
    else:
        ctx_ids = []
    return f"{omop_id}||{'|'.join(ctx_ids)}"


def _category_dedup_key(study_name: str, cat_label: str,
                        cat_omop_id: str, parent_concept_key: str) -> str:
    """Dedup key for category points: same category on same parent concept."""
    return f"{study_name}||cat||{parent_concept_key}||{cat_omop_id or cat_label}"


# ── Updated load_csv_points ──────────────────────────────────

def load_csv_points_dedup(csv_path: str, study_name: str = None,
                    embedding_mode: str = EmbeddingType.EC.value) -> List[Dict[str, Any]]:
    create_category_points = embedding_mode != EmbeddingType.ED.value
    df = load_dictionary(csv_path)
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={
        'variablename': 'variable name', 'variablelabel': 'variable label', 'visits': 'visit', 'units': 'unit'})

    # ── Phase 1: Group rows by composite concept key ──
    concept_groups = {}   # concept_key → accumulated data
    category_seen = {}    # cat_dedup_key → category point dict

    for index, row in df.iterrows():
        if pd.isna(row.get('variable name', '')):
            continue
        parts, concept_categories = get_csv_text(row, embedding_mode=embedding_mode)
        if not parts:
            continue

        text = " ".join(FuzzyMatcher._deduplicate_parts(parts=parts, threshold=0.8))
        concept_key = _composite_concept_key(row, text, embedding_mode)
        visit = str(row.get('visit', row.get('visit_occurrence', ''))).strip().lower()
        var_name = str(row.get('variable name', '')).strip()

        if concept_key not in concept_groups:
            var_data = _prepare_var_metadata(row, study_name)
            concept_val = str(row.get('variable omop id', row.get('variable name', '')))
            var_data['parent_variable_concept'] = (
                concept_val.lower().strip() if pd.notna(concept_val) else None)
            original_categories, category_labels = split_categories(
                row.get('categorical', None))
            var_data.pop('categorical', None)
            var_data['original_categories'] = "|".join(original_categories)
            var_data['original_categories_labels'] = "|".join(category_labels)

            concept_groups[concept_key] = {
                "text": text,
                "metadata": var_data,
                "visit": set(),
                "var_names": set(),
                "categories": concept_categories,
                "first_index": index,
                "row": row,  # keep first row for category extraction
            }

        concept_groups[concept_key]["visit"].add(visit)
        concept_groups[concept_key]["var_names"].add(var_name)

    # ── Phase 2: Build points ──
    points = []
    for ckey, group in concept_groups.items():
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS, f"{study_name}_{group['first_index']}"))
        meta = group["metadata"]
        meta['parent_variable_id'] = point_id
        # Store representative variable_name (first alphabetically for determinism)
        # AND all names for reference
        sorted_names = sorted(group["var_names"])
        meta['variable_name'] = sorted_names[0]  # representative for NE lookup
        meta['variable_names'] = "|".join(sorted_names)
        meta['visit'] = "|".join(sorted(group["visit"]))
        meta['visit_count'] = len(group["visit"])

        points.append({
            "text": group["text"],
            "id": point_id,
            "metadata": {**meta, "is_category": 0, "point_type": "variable"},
        })

        # ── Category points: dedup across visits ──
        if create_category_points and group["categories"]:
            row = group["row"]
            cat_codes = extract_field_values(row, 'categorical value concept code')
            cat_omop_ids = extract_field_values(row, 'categorical value omop id')

            for idx, cat_label in enumerate(group["categories"]):
                cat_omop = cat_omop_ids[idx] if idx < len(cat_omop_ids) else None
                cat_dk = _category_dedup_key(study_name, cat_label, cat_omop, ckey)

                if cat_dk in category_seen:
                    continue  # already have this category for this concept

                cat_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cat_dk))
                cat_meta = meta.copy()
                cat_meta.update({
                    'is_category': 1,
                    'point_type': 'category',
                    'category_label': cat_label,
                    'category_code': cat_codes[idx] if idx < len(cat_codes) else None,
                    'category_omop_id': cat_omop,
                })
                cat_point = {"text": cat_label, "id": cat_point_id, "metadata": cat_meta}
                points.append(cat_point)
                category_seen[cat_dk] = True

    print(f"📊 {len(points)} points from {csv_path} "
          f"(deduped from {len(df)} rows, "
          f"{len(concept_groups)} unique concepts, "
          f"{len(category_seen)} unique categories)")
    return points


# ── Updated search_in_db with adaptive threshold ─────────────
def _adaptive_gate(scores: np.ndarray, floor: float = 0.4) -> float:
    """Query-adaptive cosine cutoff: floor or natural gap, whichever is higher."""
    if len(scores) < 2:
        return floor
    gaps = scores[:-1] - scores[1:]
    max_gap = gaps.max()
    if max_gap > 0.15:
        return float(scores[int(np.argmax(gaps)) + 1]) + 0.01
    return floor

# def search_in_db(
#     vectordb: QdrantClient, embedding_model: Any, query_text: str,
#     limit: int = settings.LIMIT, target_study: list = ["gissi-hf"],
#     collection_name: str = "studies_metadata",
#     mapping_mode: str = MappingType.OEH.value,
#     start_threshold: float = settings.SIMILARITY_THRESHOLD,
#     min_threshold: float = settings.ADAPTIVE_THRESHOLD,
#     step: float = 0.1
# ) -> List[Any]:
#     """Top-down adaptive retrieval: start strict, relax until limit filled."""
#     query_text = clean_label_remove_temporal_context(query_text)
#     ck = _cache_key(embedding_model.model_name, query_text)
#     query_vector = _embed_cache.get(ck)
#     if query_vector is None:
#         query_vector = embedding_model.embed_text(query_text, is_query=True)
#         _embed_cache[ck] = query_vector

#     study_filter = models.Filter(must=[
#         models.FieldCondition(key='study_name', match=models.MatchAny(any=target_study)),
#         models.FieldCondition(key='is_category', match=models.MatchValue(value=0)),
#     ])

#     is_ne = mapping_mode == MappingType.NE.value
#     threshold = start_threshold
#     seen, out = set(), []

#     while threshold >= min_threshold and len(out) < limit:
#         with _qdrant_lock:
#             results = vectordb.search(
#                 collection_name=collection_name,
#                 query_vector=query_vector,
#                 limit=limit,              # over-fetch — Qdrant returns sorted by score
#                 score_threshold=threshold,
#                 query_filter=study_filter,
#             )
#         for r in results:
#             key = r.payload.get('variable_name', '') if is_ne else int(r.payload['variable_omop_id'])
#             if key and key not in seen:
#                 seen.add(key)
#                 out.append(key)
#                 if len(out) >= limit:
#                     break
#         if len(out) >= limit:
#             break
#         threshold = round(threshold - step, 2)

#     return out[:limit]


def search_in_db(vectordb, embedding_model, query_text,
                 limit=settings.LIMIT, target_study=["gissi-hf"],
                 collection_name="studies_metadata",
                 mapping_mode=MappingType.OEH.value,
                 min_threshold=settings.ADAPTIVE_THRESHOLD) -> List[Any]:
    query_text = clean_label_remove_temporal_context(query_text)
    ck = _cache_key(embedding_model.model_name, query_text)
    query_vector = _embed_cache.get(ck)
    if query_vector is None:
        query_vector = embedding_model.embed_text(query_text, is_query=True)
        _embed_cache[ck] = query_vector

    with _qdrant_lock:
        results = vectordb.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=min_threshold,
            query_filter=models.Filter(must=[
                models.FieldCondition(key='study_name', match=models.MatchAny(any=target_study)),
                models.FieldCondition(key='is_category', match=models.MatchValue(value=0)),
            ]),
        )
    if not results:
        return []

    scores = np.array([r.score for r in results])
    cutoff = _adaptive_gate(scores, floor=min_threshold)

    is_ne = mapping_mode == MappingType.NE.value
    seen, out = set(), []
    for r in results:
        if r.score < cutoff:
            break  # sorted descending, everything below is worse
        key = r.payload.get('variable_name', '') if is_ne else int(r.payload['variable_omop_id'])
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out[:limit]

# def search_in_db(vectordb: QdrantClient, embedding_model: Any, query_text: str,
#                  limit: int = 3, target_study: list = ["gissi-hf"],
#                  collection_name: str = "studies_metadata",
#                  mapping_mode: str = MappingType.OEH.value,
#                  min_score: float = 0.4) -> List[Any]:
#     """Search vector DB with adaptive per-query threshold.
    
#     Retrieves with low floor, then gates at the per-query score percentile.
#     With deduped index, limit=100 now covers ~100 unique concepts.
#     """
#     query_text = clean_label_remove_temporal_context(query_text)
#     # ck = f"{embedding_model.model_name}::{query_text}"
#     ck = _cache_key(embedding_model.model_name, query_text)
#     if ck in _embed_cache:
#         query_vector = _embed_cache[ck]
#     else:
#         query_vector = embedding_model.embed_text(query_text, is_query=True)
#         _embed_cache[ck] = query_vector

#     with _qdrant_lock:
#         results = vectordb.search(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=limit,
#             # score_threshold=min_score,
#             query_filter=models.Filter(must=[
#                 models.FieldCondition(
#                     key='study_name',
#                     match=models.MatchAny(any=target_study)),
#                 models.FieldCondition(
#                     key='is_category',
#                     match=models.MatchValue(value=0)),
#             ])
#         )

#     if not results:
#         return []

#     scores = np.array([r.score for r in results])
#     cutoff = _adaptive_gate(scores, floor=min_score)

#     seen, results_out = set(), []
#     is_ne = mapping_mode == MappingType.NE.value

#     for res in results:
#         if res.score < cutoff:
#             continue
#         if is_ne:
#             # Representative name → resolve_matches uses tgt_name_to_desc chain
#             key = res.payload.get('variable_name', '')
#         else:
#             key = int(res.payload['variable_omop_id'])
#         if key and key not in seen:
#             seen.add(key)
#             results_out.append(key)

#     return results_out


# ── Updated search_category_by_id ────────────────────────────
# def search_category_by_id(
#     vectordb: QdrantClient, embedding_model: Any, query_text: str,
#     limit: int = settings.LIMIT, target_study: list = ["gissi-hf"],
#     collection_name: str = "studies_metadata",
#     mapping_mode: str = MappingType.OEH.value,
#     start_threshold: float = settings.SIMILARITY_THRESHOLD, min_threshold: float = settings.ADAPTIVE_THRESHOLD,
#     step: float = 0.1
# ) -> List[Any]:
#     """Top-down adaptive retrieval: start strict, relax only when empty.
    
#     Retrieves at the highest threshold first. If no candidates are found,
#     lowers the threshold by `step` and re-queries until results are 
#     obtained or `min_threshold` is reached (Lei et al., 2025).
#     """
#     query_text = clean_label_remove_temporal_context(query_text)
#     ck = _cache_key(embedding_model.model_name, query_text)
#     if ck in _embed_cache:
#         query_vector = _embed_cache[ck]
#     else:
#         query_vector = embedding_model.embed_text(query_text, is_query=True)
#         _embed_cache[ck] = query_vector

#     study_filter = models.Filter(must=[
#         models.FieldCondition(key='study_name', match=models.MatchAny(any=target_study)),
#         models.FieldCondition(key='is_category', match=models.MatchValue(value=0)),
#     ])

#     is_ne = mapping_mode == MappingType.NE.value
#     threshold = start_threshold

#     while threshold >= min_threshold:
#         with _qdrant_lock:
#             results = vectordb.search(
#                 collection_name=collection_name,
#                 query_vector=query_vector,
#                 limit=limit,
#                 score_threshold=threshold,
#                 query_filter=study_filter,
#             )
#         if results:
#             seen, out = set(), []
#             for r in results:
#                 key = r.payload.get('variable_name', '') if is_ne else int(r.payload['variable_omop_id'])
#                 if key and key not in seen:
#                     seen.add(key)
#                     out.append(key)
#             if out:
#                 return out
#         threshold = round(threshold - step, 2)

#     return []

def search_category_by_id(vectordb: QdrantClient, embedding_model: Any,
                          query_text: str, category_id: str,
                          target_study: List[str] = None,
                          min_score: float = 0.8,
                          limit: int = 3,
                          collection_name: str = "studies_metadata") -> List[Dict[str, Any]]:
    """Search for categories by exact code match. Dedup-safe."""
    query_text = clean_label_remove_temporal_context(query_text)
    ck = f"{embedding_model.model_name}::{query_text}"
    if ck in _embed_cache:
        embed_text = _embed_cache[ck]
    else:
        embed_text = embedding_model.embed_text(query_text, is_query=True)
        _embed_cache[ck] = embed_text

    with _qdrant_lock:
        results = vectordb.search(
            collection_name=collection_name,
            query_vector=embed_text,
            limit=limit,
            score_threshold=min_score,
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key='is_category',
                    match=models.MatchValue(value=1)),
                models.FieldCondition(
                    key='category_omop_id',
                    match=models.MatchValue(value=category_id)),
            ] + ([models.FieldCondition(
                    key='study_name',
                    match=models.MatchAny(any=target_study)),
            ] if target_study else []))
        )

    # Dedup by (parent_concept, category_omop_id) — with index dedup
    # this should already be unique, but safety check
    seen, matches = set(), []
    scores = np.array([r.score for r in results])
    cutoff = _adaptive_gate(scores, floor=min_score)
    for point in results:
        if point.score < cutoff:
            continue
        dk = (point.payload.get('parent_variable_concept', ''),
              point.payload.get('category_omop_id', ''))
        if dk in seen:
            continue
        seen.add(dk)
        matches.append({
            'category_label': point.payload.get('category_label'),
            'category_code': point.payload.get('category_code'),
            'category_omop_id': point.payload.get('category_omop_id'),
            'parent_variable_id': point.payload.get('parent_variable_id'),
            'parent_variable_concept': point.payload.get('parent_variable_concept'),
            'study_name': point.payload.get('study_name'),
        })
    return matches

def generate_studies_embeddings(dir_path:str, vectordb_path:str, collection_name:str, model_name:str, embedding_mode:str=EmbeddingType.EC.value, recreate_db=False):
    """
    Generate embeddings for all studies in a directory.
    """
    vectordb, embedding_model = load_vectordb(recreate=recreate_db, collection_name=collection_name, vectordb_path=vectordb_path, model_name=model_name)
    if vectordb is None or embedding_model is None:
        print(f"❌ Error loading vector database or embedding model")
        return None, None
    # clear_embed_cache()
    if not recreate_db:
        print(f"📊 Database already exists, skipping generation")
        return vectordb, embedding_model
    
    for cohort_folder in os.listdir(dir_path):
        if cohort_folder.startswith('.'):  # Skip hidden files
            continue
        
        cohort_path = os.path.join(dir_path, cohort_folder)
        study_name = cohort_folder.lower()

        
        if os.path.isdir(cohort_path):
            # Find metadata files
            patterns = ("*.csv", "*.xlsx", "*.json")
            file_candidates: list[str] = []
            
            for pat in patterns:
                file_candidates.extend(glob.glob(os.path.join(cohort_path, pat)))
            
            cohort_metadata_file = None
            
            # Find first CSV/XLSX file
            for file in file_candidates:
                if file.lower().endswith((".csv", ".xlsx")):
                    cohort_metadata_file = file
                    break
            
            if cohort_metadata_file and os.path.exists(cohort_metadata_file):
                # print(f"📄 Found metadata file: {cohort_metadata_file}")
                
                points = load_csv_points_dedup(cohort_metadata_file, study_name, embedding_mode=embedding_mode)
                # print(f"📝 Generated {len(points)} points (variables + categories)")
                
                # Insert in batches
                batch_size = 300
                for i in range(0, len(points), batch_size):
                    batch_points = points[i:i + batch_size]
                    # batch_num = (i // batch_size) + 1
                    # print(f"\n  Batch {batch_num}: Inserting {len(batch_points)} points...")
                    insert_in_db(vectordb, embedding_model, batch_points, collection_name)
                
            else:
                print(f"⚠️ No metadata file found for {cohort_folder}")
    
    return vectordb, embedding_model



# # usage example
# if __name__ == "__main__":
#     vectordb, embedding_model = load_vectordb(recreate=False, collection_name="studies_metadata_biolord_embedding(description)", vectordb_path="localhost", model_name="biolord")
#     search_in_db(vectordb, embedding_model, "bmi", limit=100, target_study=["gissi-hf"], data_domain=settings.DATA_DOMAINS, min_score=0.4, collection_name="studies_metadata", mapping_mode=MappingType.OEH.value)