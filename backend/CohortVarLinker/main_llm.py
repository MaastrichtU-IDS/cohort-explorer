

import pandas as pd
import json
import os
import glob
import time
from collections import defaultdict
from typing import Any
from llm.run import StudyMapper
from llm.variables_kg import process_variables_metadata_file
from llm.study_kg import generate_studies_kg
from llm.constraints import CategoryMapper
from llm.embed_model import get_model
from llm.omop_graph_nx import OmopGraphNX
from llm.config import settings
from llm.data_model import MappingType, EmbeddingType 
from llm.utils import (
        get_cohort_mapping_uri,
        delete_existing_triples,
        publish_graph_to_endpoint,
        OntologyNamespaces,
        get_member_studies
    )
from llm.vector_db import generate_studies_embeddings, _embed_cache

from llm.graph_similarity import _EMBED_CACHE

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
            # print(f"delete_existing_triples for graph: {OntologyNamespaces.CMEO.value['graph/studies_metadata']}")
            delete_existing_triples(graph_uri=OntologyNamespaces.CMEO.value["graph/studies_metadata"])
            response=publish_graph_to_endpoint(g)
            # print(f"Metadata graph published to endpoint: {response}")
            g.serialize(destination=graph_file_path, format="trig")
            # print(f"Serialized graph to: {graph_file_path}")
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
        for _, row in df.iterrows():
            src_var = str(row["source"]).strip()
            if not src_var:
                continue
            entry = {"target_study": target, **row.drop(labels=["source"]).to_dict()}
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
            # print(f"  📎 Appending {member} ({len(member_df)} rows) to {parent_study}")
            # delete memeber csv after appending
            # os.remove(member_csv_path)
            # print(f"    🗑️ Deleted member CSV: {member_csv_path}")
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

if __name__ == '__main__':
    start_time = time.time()
    data_dir = 'data'
    cohort_file_path = f"{data_dir}/cross_mapping_article_data"
    cohorts_metadata_file = f"{data_dir}/studies_metadata-2.xlsx"
    output_dir = f"{data_dir}/output/cross_mapping"

    model_name = "biolord"
    select_relevant_studies = True
    embedding_mode = EmbeddingType.EH.value  # embedding_concepts
    mapping_mode = MappingType.OEH.value # ontology + embedding_concepts
    create_study_metadata_graph(cohorts_metadata_file, recreate=True)             
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=True)      
    collection_name = f"studies_metadata_{model_name}_{embedding_mode}"      
    embedding_model, _ = get_model(model_name)     
    # llm_matcher = LocalLLMConceptMatcher(models=["llama3.3:70b", "llama3.1:latest"])
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "localhost", collection_name, model_name=model_name, embedding_mode=embedding_mode, recreate_db=True)
    source_study = "time-chf"
    target_studies = ["viennahf-register","gissi-hf","aachen-hf"]

    clear_all_caches()
    new_studies = []
    parent_to_members = defaultdict[Any, list](list)
    if select_relevant_studies:
        for tstudy in target_studies:
            member_studies = get_member_studies(tstudy)
            parent_to_members.setdefault(tstudy, []).extend(member_studies)
            new_studies.extend(member_studies)
    target_studies.extend(new_studies)

    # print(f"connected studies: {parent_to_members}")
    omop_id_tracker = {} 

    llm_model = None
    mapping_dict = {}  
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
    llm_tag = llm_tag.replace(":nitro","")
    print(f"llm_tag: {llm_tag}")
    for tstudy in target_studies:
        print(f"Running experiment for {source_study} -> {tstudy} with model: {model_name} and mapping mode: {mapping_mode}")
        mapping_transformed = mapper.run_pipeline(
            src_study=source_study,
            tgt_study=tstudy,
            mapping_mode=mapping_mode)        
        mapping_transformed.to_csv(f'{output_dir}/{model_name}/{mapping_mode}/{source_study}_{tstudy}_{model_name}+{llm_tag}_{mapping_mode}_full.csv', index=False)
    tstudy_str = "_".join(target_studies)
    
    for parent_study, members in parent_to_members.items():
        if members:
            concatenate_member_csvs_to_parent(
                source_study=source_study,
                parent_study=parent_study,
                member_studies=members,
                output_dir=f"{output_dir}/{model_name}/{mapping_mode}",
                model_name=model_name,
                mapping_mode=mapping_mode,
                llm=llm_tag
            )
    combine_all_mappings_to_json(
        source_study=source_study,
        target_studies=target_studies,
        output_dir=f"{output_dir}/{model_name}/{mapping_mode}",
        json_path=os.path.join(f"{output_dir}/{model_name}/{mapping_mode}", f"{source_study}_{tstudy_str}_{model_name}+{llm_tag}_{mapping_mode}.json"),
        model_name=model_name,
        mapping_mode=mapping_mode,
        llm_tag=llm_tag
    )
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    # add_data_access_spec(study_name="time-chf", data_policy=['disease specific research'], data_modifier=['ethics approval required'], disease_concept_code="snomed:42343007", disease_concept_label="congestive heart failure", disease_concept_omop_id="42343007", study_metadata_graph_file_path=f"{data_dir}/graphs/studies_metadata.trig")