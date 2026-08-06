

import pandas as pd
import json
import os
import glob
import time
from collections import defaultdict
from typing import Any
from src.run import StudyMapper
from src.variables_kg import process_variables_metadata_file
from src.study_kg import generate_studies_kg
from src.constraints import CategoryMapper
from src.embed_model import get_model
from src.omop_graph_nx import OmopGraphNX
from src.config import settings
from src.data_model import MappingType, EmbeddingType 
from src.utils import (
        get_cohort_mapping_uri,
        delete_existing_triples,
        publish_graph_to_endpoint,
        OntologyNamespaces,
        get_member_studies
    )
from src.utils import canonical_var_key
from src.vector_db import generate_studies_embeddings, _embed_cache

from src.graph_similarity import _EMBED_CACHE

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
            g.serialize(destination=graph_file_path, format="trig")
            # print(f"Serialized graph to: {graph_file_path}")
            return g
        else:
            return None
    # else:
    #     print("Recreate flag is set to False. Skipping processing of study metadata.")


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
                        # print(f"Publishing graph for cohort: {cohort_id}")
                        delete_existing_triples(
                            get_cohort_mapping_uri(cohort_id)
                        )
                        publish_graph_to_endpoint(g)
                    # else:
                    #     print(f"Error processing metadata file for cohort: {cohort_folder}. There might be data validation errors in the file.")
                
            # else:
            #     print(f"Skipping non-directory file: {cohort_folder}")
            # print(f"Base path: {base_path}")
    # else:
    #     print("Recreate flag is set to False. Skipping processing of cohort metadata.")


def combine_all_mappings_to_json(source_study, target_studies, output_dir, json_path,
                                 model_name, llm_tag, mapping_mode):
    mappings = {}
    for target in target_studies:
        csv_file = os.path.join(
            output_dir,
            f"{source_study}_{target}_{model_name}+{llm_tag}_{mapping_mode}_full.csv",
        )
        if not os.path.exists(csv_file):
            # print(f"⚠️ Missing: {csv_file}")
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
    # print(f"✅ saved {len(final_json)} source vars → {json_path}")

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
            os.remove(member_csv_path)
            # print(f"    🗑️ Deleted member CSV: {member_csv_path}")
        # else:
        #     print(f"  ⚠️ Member CSV not found: {member_csv_path}")
    
    # Concatenate all DataFrames
    if len(dfs_to_concat) > 1:
        combined_df = pd.concat(dfs_to_concat, ignore_index=True)
        if 'member_study' in combined_df.columns:
            # drop this column
            combined_df.drop(columns=['member_study'], inplace=True)
            
        # Save back to parent CSV (overwrites with combined data)
        combined_df.to_csv(parent_csv_path, encoding='utf-8',index=False)
        # print(f"✅ Combined {len(dfs_to_concat)} CSVs into {parent_study}: {len(combined_df)} total rows")

def load_study_variables(cohort_dir: str, study_name: str) -> list:
    """Read a study's full variable list straight from its data dictionary.

    Deliberately independent of the pipeline's VariableCollection: that one is
    narrowed by `_restrict_source_variables` to the benchmark scope, whereas the
    expansion needs every column a downstream merge will actually meet in the
    data. Reading the dictionary also means nothing in llm/run.py has to change.

    Returns [] when no dictionary is found, so a missing study degrades to
    "no expansion" instead of failing the run.
    """
    folder = None
    for entry in os.listdir(cohort_dir):
        if entry.lower() == study_name.lower() and os.path.isdir(os.path.join(cohort_dir, entry)):
            folder = os.path.join(cohort_dir, entry)
            break
    if folder is None:
        return []

    candidates = [p for p in glob.glob(os.path.join(folder, "*.csv")) if os.path.isfile(p)]
    if not candidates:
        return []

    df = pd.read_csv(sorted(candidates)[-1], low_memory=False)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    def pick(*aliases):
        return next((a for a in aliases if a in df.columns), None)

    name_col = pick("variablename", "variable_name", "name")
    visit_col = pick("visit concept name", "visits", "visit")
    omop_col = pick("variable omop id", "variable_omop_id")
    ctx_col = pick("additional context omop id", "additional_context_omop_id")
    if not name_col or not omop_col:
        return []

    cat_col = pick("categorical")
    unit_col = pick("units", "unit")
    type_col = pick("vartype", "var_type")

    return [
        {
            "name": row[name_col],
            "visit": row[visit_col] if visit_col else "",
            "main_id": row[omop_col],
            "context_ids": row[ctx_col] if ctx_col else None,
            # Encoding, used to decide whether a sibling at another timepoint is
            # really the same variable — Orthopnea3 (0/1/2/3) and Orthopnea3_01
            # (yes/no) share a concept but are not interchangeable.
            "categorical": row[cat_col] if cat_col else "",
            "units": row[unit_col] if unit_col else "",
            "vartype": row[type_col] if type_col else "",
        }
        for _, row in df.iterrows()
    ]


def get_requested_variables(gt_excel, source_study="time-chf"):
    df = pd.read_excel(gt_excel)                 # was read_excel("gt_excel") — reads a literal string
    df = df[df["source_study"].str.strip().str.lower() == source_study]
    names = df["source_var_name"].dropna().map(canonical_var_key)  # match the eval's key exactly
    return sorted(set(names))                     # dedupe; raw .tolist() has duplicates

if __name__ == '__main__':
    start_time = time.time()
    data_dir = 'data'
    cohort_file_path = f"{data_dir}/cross_mapping"
    # cohorts_metadata_file = f"{data_dir}/studies_metadata-2.xlsx"
    cohorts_metadata_file = f"{data_dir}/studies_metadata-2_all.xlsx"
    output_dir = f"{data_dir}/output/cross_mapping"

    model_name = "biolord"
    select_relevant_studies = True
   
    # expand_timepoints = False
    embedding_mode = EmbeddingType.EH.value  # embedding_concepts
    mapping_mode = MappingType.OEH.value # ontology + embedding_concepts
    create_study_metadata_graph(cohorts_metadata_file, recreate=False)             
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=False)      
    collection_name = f"studies_metadata_{model_name}_{embedding_mode}"      
    embedding_model, _ = get_model(model_name)     
    # llm_matcher = LocalLLMConceptMatcher(models=["llama3.3:70b", "llama3.1:latest"])
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "localhost", collection_name, model_name=model_name, embedding_mode=embedding_mode, recreate_db=False)
    source_study = "time-chf"
    target_studies = ["aachen-hf","biostat-chf", "viennahf-register", "gissi-hf", "tim-hf","check-hf"]
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
  
    llm_model = "litellm/gpt-oss:120b" # running in openai with temp 0.5, top_p = 0.95 and thinking low
  
    mapping_dict = {}  
    omop_graph = None if mapping_mode == MappingType.NE.value else OmopGraphNX(csv_file_path=settings.concepts_file_path)
    mapper = StudyMapper(
         vector_db=vector_db,
         vector_collection=collection_name,
         embedding_model=embedding_model, 
         omop_graph=omop_graph,
         mapping_mode=mapping_mode,
         llm_model=llm_model,
         list_of_var=get_requested_variables("/Users/komalgilani/phd_projects/CohortVarLinker/data/ground_truth_pairs.xlsx")
         )
    llm_tag = llm_model.split("/")[-1] if llm_model and mapping_mode != MappingType.OO.value else "no_llm" 
   
    llm_tag = llm_tag.replace(":120b","-120b")
    print(f"llm_tag: {llm_tag}")
    for tstudy in target_studies:
        print(f"Running experiment for {source_study} -> {tstudy} with model: {model_name} and mapping mode: {mapping_mode}")
        mapping_transformed = mapper.run_pipeline(
            src_study=source_study,
            tgt_study=tstudy,
            mapping_mode=mapping_mode)        
        csv_path = f'{output_dir}/{model_name}/{mapping_mode}/{source_study}_{tstudy}_{model_name}+{llm_tag}_{mapping_mode}_full.csv'
      
        if not mapping_transformed.empty:
            mapping_transformed = mapping_transformed[mapping_transformed["harmonization_status"].str.strip().str.lower() != "not applicable"]
        mapping_transformed.to_csv(csv_path, index=False)
    tstudy_str = "_".join(target_studies)
    
   
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