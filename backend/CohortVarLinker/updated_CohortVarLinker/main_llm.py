
import imp
from re import T
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
            print(f"delete_existing_triples for graph: {OntologyNamespaces.CMEO.value['graph/studies_metadata']}")
            # delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/studies_metadata")
            delete_existing_triples(graph_uri=OntologyNamespaces.CMEO.value["graph/studies_metadata"])
            response=publish_graph_to_endpoint(g)
            print(f"Metadata graph published to endpoint: {response}")
            g.serialize(destination=graph_file_path, format="trig")
            print(f"Serialized graph to: {graph_file_path}")
            return g
        else:
            # print("No metadata found in the file")
            return None
    else:
        print("Recreate flag is set to False. Skipping processing of study metadata.")


def create_cohort_specific_metadata_graph(dir_path, recreate=False):

    base_path = os.path.dirname(os.path.abspath(__file__))
    # print(f"Base path: {base_path}")
    if  recreate:
        for cohort_folder in os.listdir(dir_path):
            if cohort_folder.startswith('.'):  # Skip hidden files like .DS_Store
                continue
            start_time = time.time()
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
                    # print(f"File: {file}")
                    # Collect *all* metadata spreadsheets
                    if file.lower().endswith((".csv", ".xlsx")):
                        cohort_metadata_file = file
                    # Optionally single out an EDA JSON
                    if os.path.basename(file).lower().startswith("eda") and file.lower().endswith(".json"):
                        eda_file = file
                # print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for metadata file: {cohort_metadata_file}")
                if cohort_metadata_file:
                    # if eda_file and os.path.exists(eda_file):
                    #     print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for eda file: {eda_file}")
                    g, cohort_id = process_variables_metadata_file(cohort_metadata_file, cohort_name=cohort_folder, eda_file_path=eda_file, study_metadata_graph_file_path=f"{base_path}/data/graphs/studies_metadata.trig")
                    if g and len(g) > 0:
                        # print(validate_graph(g))
                        g.serialize(f"{base_path}/data/graphs/{cohort_id}_metadata.trig", format="trig")
                        print(f"Publishing graph for cohort: {cohort_id}")
                        delete_existing_triples(
                            get_cohort_mapping_uri(cohort_id)
                        )
                        publish_graph_to_endpoint(g)
                        # print(f"Graph contains {len(g)} triples before serialization.")
                        # print(f"Graph published to endpoint: {res} for cohort: {cohort_id}")
                        
                        end_time = time.time()
                        # print(f"Time taken to process cohort: {cohort_folder} is: {end_time - start_time}")
                    else:
                        print(f"Error processing metadata file for cohort: {cohort_folder}. There might be data validation errors in the file.")
                
            else:
                print(f"Skipping non-directory file: {cohort_folder}")
            # print(f"Base path: {base_path}")
    else:
        print("Recreate flag is set to False. Skipping processing of cohort metadata.")

# def create_pld_graph(file_path, cohort_name, output_dir=None, recreate=False) -> None:
#     if recreate:
#         start_time = time.time()
#         g=add_raw_data_graph(file_path, cohort_name)
#         if len(g) > 0:
#             g.serialize(f"{output_dir}/{cohort_name}_pld.trig", format="trig")
#             # delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/{cohort_name}_pld")
#             # res=publish_graph_to_endpoint(g,graph_uri=f"{cohort_name}_pld")
            
#             delete_existing_triples(f"{get_cohort_mapping_uri(cohort_name)}_pld")
#             res=publish_graph_to_endpoint(g)

#             print(f"Graph published to endpoint: {res} for cohort: graph/{cohort_name}_pld")
#             end_time = time.time()
#             print(f"Time taken to process PLD: graph/{cohort_name}_pld is: {end_time - start_time}")
#         else:
#             print("No data found in the file")
#     else:
#         print("Recreate flag is set to False. Skipping processing of PLD data.")





def combine_all_mappings_to_json(
    source_study, target_studies, output_dir, json_path, model_name=None
):
    # Dict: {source_var: [mapping_dicts]}
    mappings = {}
    for target in target_studies:
        suffix = f"_{model_name}" if model_name else ""
        csv_file = os.path.join(output_dir, f"{source_study}_{target}{suffix}_full.csv")
        # print(f"Processing file: {csv_file}")
        if not os.path.exists(csv_file):
            # print(f"Skipping {csv_file}, does not exist.")
            continue
        df = pd.read_csv(csv_file)
        for idx, row in df.iterrows():
            # Source variable name
            src_var = str(row["source"]).strip()
            if not src_var:
                continue
            # Initialize dict for this variable if not present
            if src_var not in mappings:
                mappings[src_var] = []
            # Build mapping dict for this target study
            mapping = {"target_study": target}
            # Source columns
            for col in df.columns:
                if col.startswith("source_") or col.startswith("s") or col in [
                    "source", "somop_id", "scode", "slabel",
                    "category", "source_visit", "source_type", "source_unit", "source_data_type"
                ]:
                    mapping[f"s_{col}"] = row[col]
            # Target columns
            for col in df.columns:
                if col.startswith("target_") or col.startswith("t") or col in [
                    "target", "tomop_id", "tcode", "tlabel", 
                    "target_visit", "target_type", "target_unit", "target_data_type"
                ]:
                    mapping[f"{target}_{col}"] = row[col]
            # Extra columns (mapping_type, transformation_rule, etc.)
            for col in df.columns:
                if col not in [
                    "source", "target", "somop_id", "tomop_id",
                    "scode", "slabel", "tcode", "tlabel",
                    "category", "mapping type", "source_visit", "target_visit",
                    "source_type", "target_type", "source_unit", "target_unit",
                    "source_data_type", "target_data_type", "transformation_rule"
                ]:
                    mapping[f"{col}"] = row[col]
            # Always include mapping type and transformation rule
            if "mapping type" in df.columns:
                mapping[f"{target}_mapping_type"] = row["mapping type"]
            if "transformation_rule" in df.columns:
                mapping[f"{target}_transformation_rule"] = row["transformation_rule"]
            # Add to source variable
            
            mappings[src_var].append(mapping)
    # Compose final JSON dict
    final_json = {}
    for src_var, mapping_list in mappings.items():
        final_json[src_var] = {
            "from": source_study,
            "mappings": mapping_list
        }
    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"✅ All mappings combined and saved to {json_path}")


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
        combined_df.to_csv(parent_csv_path, index=False)
        print(f"✅ Combined {len(dfs_to_concat)} CSVs into {parent_study}: {len(combined_df)} total rows")

def run_mapping(data_dir, cohort_paths, output_dir, embedding_model_name="sapbert", 
        src_study_name:str ="time-chf", 
        target_studies_names:list[str]= ["gissi-hf"],embedding_mode = EmbeddingType.EH.value, 
        mapping_mode=MappingType.OEH.value, llm_model_name="gpt", recreate=False):
    print("all")

if __name__ == '__main__':
    start_time = time.time()
    data_dir = 'data'
    cohort_file_path = f"{data_dir}/cross_mapping_folder/cohorts_folders"
    cohorts_metadata_file = f"{data_dir}/studies_metadata.xlsx"
    output_dir = f"{data_dir}/output/cross_mapping"
    model_name = ["minilm","biolord","e5", "sapbert"]
    # already tested (with llms): biolord, openai, minilm
    model_name = "sapbert"
    select_relevant_studies = True
    embedding_mode = EmbeddingType.EH.value  # embedding_concepts
    mapping_mode = MappingType.OEH.value # ontology + embedding_concepts
    create_study_metadata_graph(cohorts_metadata_file, recreate=False)             
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=False)      
    collection_name = f"studies_metadata_{model_name}_{embedding_mode}"      
    embedding_model, _ = get_model(model_name)     
    # llm_matcher = LocalLLMConceptMatcher(models=["llama3.3:70b", "llama3.1:latest"])
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "localhost", collection_name, model_name=model_name, embedding_mode=embedding_mode, recreate_db=False)
    source_study = "time-chf"
    target_studies = ["gissi-hf","aachen-hf","viennahf-register","aric"]
    # target_studies = ["aachen-hf"]
    clear_all_caches()
    new_studies = []
    parent_to_members = defaultdict[Any, list](list)
    if select_relevant_studies:
        for tstudy in target_studies:
            member_studies = get_member_studies(tstudy)
            parent_to_members.setdefault(tstudy, []).extend(member_studies)
            new_studies.extend(member_studies)
    target_studies.extend(new_studies)

    print(f"connected studies: {parent_to_members}")
    omop_id_tracker = {} 
  
    llm_models = None
    mapping_dict = {}  
    omop_graph = None if mapping_mode == MappingType.NE.value else OmopGraphNX(csv_file_path=settings.concepts_file_path)
    mapper = StudyMapper(
         vector_db=vector_db,
         vector_collection=collection_name,
         embedding_model=embedding_model, 
         omop_graph=omop_graph,
         mapping_mode=mapping_mode,
         llm_models=llm_models
         )
    llm_tag = llm_models[0].split("/")[-1] if llm_models and mapping_mode != MappingType.OO.value else "no_llm" 
    print(f"llm_tag: {llm_tag}")
    for tstudy in target_studies:
        print(f"Running experiment for {source_study} -> {tstudy} with model: {model_name} and mapping mode: {mapping_mode}")
        mapping_transformed = mapper.run_pipeline(
            src_study=source_study,
            tgt_study=tstudy,
            mapping_mode=mapping_mode)

        # print(mapping_transformed.head(5))
        
        mapping_transformed = mapping_transformed.drop_duplicates(keep='first') if not mapping_transformed.empty else pd.DataFrame(columns=["source_variable", "target_variable", "source_omop_id", "target_omop_id"])
        mapping_transformed.to_csv(f'{output_dir}/{model_name}/{mapping_mode}/{source_study}_{tstudy}_{model_name}+{llm_tag}_{mapping_mode}_full.csv', index=False)
        
    tstudy_str = "_".join(target_studies)
    
    for parent_study, members in parent_to_members.items():
        if members:
            # print(f"\n📦 Concatenating member studies for {parent_study}: {members}")
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
        model_name=f"{model_name}_{mapping_mode}"
    )
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
   

    
    # add_data_access_spec(study_name="time-chf", data_policy=['disease specific research'], data_modifier=['ethics approval required'], disease_concept_code="snomed:42343007", disease_concept_label="congestive heart failure", disease_concept_omop_id="42343007", study_metadata_graph_file_path=f"{data_dir}/graphs/studies_metadata.trig")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    