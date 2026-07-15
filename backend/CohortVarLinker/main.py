

import pandas as pd
import json
import os
import glob
# import time
# from collections import defaultdict
# from typing import Any
from updated_src.run import StudyMapper
from updated_src.variables_kg import process_variables_metadata_file
from updated_src.study_kg import generate_studies_kg
from updated_src.constraints import CategoryMapper
# from updated_src.embed_model import get_model
from updated_src.omop_graph_nx import OmopGraphNX
# from updated_src.config import settings
from updated_src.data_model import MappingType, EmbeddingType 
from updated_src.utils import (
        get_cohort_mapping_uri,
        delete_existing_triples,
        publish_graph_to_endpoint,
        OntologyNamespaces,
        get_member_studies
    )
from updated_src.vector_db import generate_studies_embeddings, _embed_cache

from updated_src.graph_similarity import _EMBED_CACHE

def clear_all_caches():
    """Clear the embedding cache (e.g. when the embedding model changes)."""
    _embed_cache.clear()
    _EMBED_CACHE.clear()
    CategoryMapper._label_embedding_cache.clear()
    CategoryMapper._label_omop_cache.clear()
    CategoryMapper._alignment_cache.clear()

def create_study_metadata_graph(file_path, recreate=False):

    if recreate:
        base_path = os.path.dirname(os.path.abspath(__file__))
        graph_file_path = f"{base_path}/data/graphs/studies_metadata.trig"
        g=generate_studies_kg(file_path)
        if len(g) > 0:
            print(f"delete_existing_triples for graph: {OntologyNamespaces.CMEO.value['graph/studies_metadata']}")
            delete_existing_triples(graph_uri=OntologyNamespaces.CMEO.value["graph/studies_metadata"])
            response=publish_graph_to_endpoint(g)
            print(f"Metadata graph published to endpoint: {response}")
            g.serialize(destination=graph_file_path, format="trig")
            print(f"Serialized graph to: {graph_file_path}")
            return g
        else:
            return None
    else:
        print("Recreate flag is set to False. Skipping processing of study metadata.")


def create_cohort_specific_metadata_graph(dir_path, recreate=False):

    base_path = os.path.dirname(os.path.abspath(__file__))
    if  recreate:
        for cohort_folder in os.listdir(dir_path):
            if cohort_folder.startswith('.'):  # Skip hidden files like .DS_Store
                continue
            cohort_path = os.path.join(dir_path, cohort_folder)
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
                    
                    if file.lower().endswith((".csv", ".xlsx")):
                        cohort_metadata_file = file
                    # Optionally single out an EDA JSON
                    if os.path.basename(file).lower().startswith("eda") and file.lower().endswith(".json"):
                        eda_file = file
                # print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for metadata file: {cohort_metadata_file}")
                if cohort_metadata_file:

                    g, cohort_id = process_variables_metadata_file(cohort_metadata_file, cohort_name=cohort_folder, eda_file_path=eda_file, study_metadata_graph_file_path=f"{base_path}/data/graphs/studies_metadata.trig")
                    if g and len(g) > 0:
                        g.serialize(f"{base_path}/data/graphs/{cohort_id}_metadata.trig", format="trig")
                        print(f"Publishing graph for cohort: {cohort_id}")
                        delete_existing_triples(
                            get_cohort_mapping_uri(cohort_id)
                        )
                        publish_graph_to_endpoint(g)
                    else:
                        print(f"Error processing metadata file for cohort: {cohort_folder}. There might be data validation errors in the file.")
                
            else:
                print(f"Skipping non-directory file: {cohort_folder}")
            # print(f"Base path: {base_path}")
    else:
        print("Recreate flag is set to False. Skipping processing of cohort metadata.")


def combine_all_mappings_to_json(source_study, target_studies, output_dir, json_path,
                                 model_name, llm_tag, mapping_mode):
    mappings = {}
    for target in target_studies:
        csv_file = os.path.join(
            output_dir,
            f"{source_study}_{target}_{model_name}+{llm_tag}_{mapping_mode}_full.csv",
        )
        if not os.path.exists(csv_file):
            print(f"⚠️ Missing: {csv_file}")
            continue
        df = pd.read_csv(csv_file)
        for row in df.to_dict(orient="records"):
            src_var = str(row["source"]).strip()
            if not src_var:
                continue
            entry = {"target_study": target, **{k: v for k, v in row.items() if k != "source"}}
            mappings.setdefault(src_var, []).append(entry)
    final_json = {k: {"from": source_study, "mappings": v} for k, v in mappings.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ saved {len(final_json)} source vars → {json_path}")

def concatenate_member_csvs_to_parent(
    source_study: str,
    parent_study: str,
    member_studies: list,
    output_dir: str,
    model_name: str,
    mapping_mode: str,
    llm:str
):
    """
    Concatenate member study CSVs into the parent study CSV.
    Adds a 'member_study' column to track the origin of each row.
    """
    parent_csv_path = f'{output_dir}/{source_study}_{parent_study}_{model_name}+{llm}_{mapping_mode}_full.csv'
    
    if not os.path.exists(parent_csv_path):
        print(f"⚠️ Parent CSV not found: {parent_csv_path}")
        return
    
    # Load parent CSV and add origin column
    parent_df = pd.read_csv(parent_csv_path)
    parent_df['member_study'] = parent_study  # Mark rows as from parent
    
    dfs_to_concat = [parent_df]
    
    # Load and append each member study CSV
    for member in member_studies:
        member_csv_path = f'{output_dir}/{source_study}_{member}_{model_name}+{llm}_{mapping_mode}_full.csv'
        
        if os.path.exists(member_csv_path):
            member_df = pd.read_csv(member_csv_path)
            member_df['member_study'] = member  # Mark rows as from member
            dfs_to_concat.append(member_df)
            print(f"  📎 Appending {member} ({len(member_df)} rows) to {parent_study}")
            # delete memeber csv after appending
            os.remove(member_csv_path)
            print(f"    🗑️ Deleted member CSV: {member_csv_path}")
        else:
            print(f"  ⚠️ Member CSV not found: {member_csv_path}")
    
    # Concatenate all DataFrames
    if len(dfs_to_concat) > 1:
        combined_df = pd.concat(dfs_to_concat, ignore_index=True)
        if 'member_study' in combined_df.columns:
            # drop this column
            combined_df.drop(columns=['member_study'], inplace=True)
            
        # Save back to parent CSV (overwrites with combined data)
        combined_df.to_csv(parent_csv_path, encoding='utf-8',index=False)
        print(f"✅ Combined {len(dfs_to_concat)} CSVs into {parent_study}: {len(combined_df)} total rows")


def generate_mapping_csv(
    source_study,
    target_studies,
    data_dir=None,
    cohort_file_path=None,
    cohorts_metadata_file=None,
    output_dir=None,
    select_relevant_studies =True
):
    """
    Generate mapping CSV files for a source study and a list of target studies.
    Args:
        source_study (str): The name of the source study.
        target_studies (list of tuple): Each tuple is (target_study_name, visit_constraint_bool).
        data_dir (str, optional): Directory containing data files. Defaults to the 'data' folder inside CohortVarLinker.
        cohort_file_path (str, optional): Path to the cohorts directory. Defaults to settings.cohort_folder.
        cohorts_metadata_file (str, optional): Path to the cohort metadata file. Defaults to f"{data_dir}/cohort_metadata_sheet_v2.csv".
        output_dir (str, optional): Directory to store output mapping CSVs. Defaults to settings.output_dir (CohortVarLinker/data/mapping_output).
    Returns:
        dict: Cache information with 'cached_pairs' and 'uncached_pairs' lists containing pair info and timestamps.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    from CohortVarLinker.updated_src.config import settings
    if cohort_file_path is None:
        cohort_file_path = settings.cohort_folder
    # Robust check: ensure all selected cohorts exist
    missing_cohorts = []
    model_name = "biolord"
    embedding_mode = EmbeddingType.EH.value  # embedding_concepts
    mapping_mode = MappingType.OEH.value # ontology + embedding_concepts
    # select_relevant_studies = True
    collection_name=f"studies_metadata_{model_name}_{embedding_mode}",
    for cohort_id in [source_study] + [t[0] for t in target_studies]:
        cohort_dir = os.path.join(cohort_file_path, cohort_id)
        if not os.path.exists(cohort_dir):
            missing_cohorts.append(cohort_id)
    if missing_cohorts:
        # If using FastAPI, raise HTTPException; otherwise, raise ValueError
        try:
            from fastapi import HTTPException
            missing_str = ", ".join(missing_cohorts)
            message = f"The metadata of the following cohorts is missing: {missing_str}"
            raise HTTPException(status_code=404, detail=message)
        except ImportError:
            missing_str = ", ".join(missing_cohorts)
            message = f"The metadata of the following cohorts is missing: {missing_str}"
            raise ValueError(message)
    if cohorts_metadata_file is None:
        cohorts_metadata_file = f"{data_dir}/queries/cohort_metadata_sheet.csv"
    if output_dir is None:
        output_dir = settings.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checking for cached or ready mapping files in directory: {os.path.abspath(output_dir)}")

    source_study = source_study.lower()
    target_studies = [t[0].lower() for t in target_studies]
    
    # add sub-studies if select_relevant_studies is True
    new_studies= []
    if select_relevant_studies:
        for tstudy in target_studies:
            member_studies = get_member_studies(tstudy)
            print(f"Member studies for {tstudy}: {member_studies}")
            new_studies.extend(member_studies)
        target_studies.extend(new_studies)
    # Check if all requested mappings already exist... 
    # Komal's comment: i dont think we should have this logic (330-342 please comment it out) here because in case of multiple target studies, we should check in next computations if mapping exist add it in the list and check next. when all available then we group by omop_id

    # Check cache status for each mapping pair and collect info
    cached_pairs = []
    uncached_pairs = []
    all_exist = True
    
    for tstudy in target_studies:
        out_filename = f'{source_study}_{tstudy}_cross_mapping.csv'
        out_path = os.path.join(output_dir, out_filename)
        print(f"Checking if {out_path} exists")

        if os.path.exists(out_path):
            # Get file modification time
            mtime = os.path.getmtime(out_path)
            cached_pairs.append({
                'source': source_study,
                'target': tstudy,
                'timestamp': mtime
            })
        else:
            all_exist = False
            uncached_pairs.append({
                'source': source_study,
                'target': tstudy
            })
    
    cache_info = {
        'cached_pairs': cached_pairs,
        'uncached_pairs': uncached_pairs
    }
    
    if all_exist:
        print("All requested mappings already exist. Skipping all computation.")
        tstudy_str = "_".join(target_studies)
        combine_all_mappings_to_json(
                source_study=source_study,
                target_studies=target_studies,
                output_dir=output_dir,
                json_path=os.path.join(output_dir, f"{source_study}_{tstudy_str}_{model_name}_{mapping_mode}.json"))
        
        return cache_info
            
    # Only run expensive computations if any mapping is missing
    create_study_metadata_graph(cohorts_metadata_file, recreate=True)
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=True)
        

    print(f"Final target studies: {target_studies}")
    # min_score_list = [0.5,0.6,0.65,0.7, 0.75, 0.8, 0.85, 0.9]
    # vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "qdrant", f"studies_metadata_{model_name}", model_name=model_name, recreate_db=True)
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "localhost", collection_name, model_name=model_name, embedding_mode=embedding_mode, recreate_db=True)
    llm_model = "litellm/gpt-oss:120b" 
    omop_graph = None if mapping_mode == MappingType.NE.value else OmopGraphNX(csv_file_path=settings.concepts_file_path)


    mapper = StudyMapper(
        vector_db=vector_db,
        vector_collection=collection_name,
        embedding_model=embedding_model, 
        omop_graph=omop_graph,
        mapping_mode=mapping_mode,
        llm_model=llm_model
        )
    llm_tag = llm_model.split("/")[-1] if llm_model and mapping_mode != MappingType.OO.value else "no_llm" 
    llm_tag = llm_tag.replace(":nitro","").replace(":free","").replace(":exacto","")
    llm_tag = llm_tag
    for tstudy in target_studies:
        out_filename = f'{source_study}_{tstudy}_cross_mapping.csv'
        out_path = os.path.join(output_dir, out_filename)
        if os.path.exists(out_path):
            print(f"Mapping already exists for {source_study} to {tstudy}, skipping computation.")
            continue
        
        print(f"Running experiment for {source_study} -> {tstudy} with model: {model_name} and mapping mode: {mapping_mode}")
        mapping_transformed = mapper.run_pipeline(
            src_study=source_study,
            tgt_study=tstudy,
            mapping_mode=mapping_mode)        

    
    
        if mapping_transformed is None or mapping_transformed.empty:
            # If possible, preserve the expected columns
            columns = getattr(mapping_transformed, 'columns', None)
            if columns is None or len(columns) == 0:
                columns = ['No mappings found']*3
            pd.DataFrame(columns=columns).to_csv(out_path, index=False)
        else:
            
            mapping_transformed.to_csv(out_path, index=False)
            
    tstudy_str = "_".join(target_studies)
    combine_all_mappings_to_json(
        source_study=source_study,
        target_studies=target_studies,
        output_dir=output_dir,
        json_path=os.path.join(output_dir, f"{source_study}_{tstudy_str}_{model_name}_{mapping_mode}.json"))
    
    return cache_info