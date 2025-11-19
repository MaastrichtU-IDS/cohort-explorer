import pandas as pd
# import cProfile
# import pstats
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import os
import glob
import time
import json
from CohortVarLinker.src.variables_kg import process_variables_metadata_file#, add_raw_data_graph
from CohortVarLinker.src.study_kg import generate_studies_kg
from CohortVarLinker.src.vector_db import generate_studies_embeddings, search_in_db
from CohortVarLinker.src.utils import (
        get_cohort_mapping_uri,
        delete_existing_triples,
        publish_graph_to_endpoint,
        OntologyNamespaces,
    
    )
     
from CohortVarLinker.src.omop_graph import OmopGraphNX



from CohortVarLinker.src.fetch import map_source_target


def create_study_metadata_graph(file_path, recreate=False):

    if recreate:
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        graph_file_path = f"{base_path}/data/graphs/studies_metadata.trig"
        g=generate_studies_kg(file_path)
        print(f"length of graph: {len(g)}")
        # if isinstance(g, ConjunctiveGraph):
        #     print("Graph is a ConjunctiveGraph with quads.")
        # else:
        #     print("Graph is a standard RDF Graph with triples.")
        # print(OntologyNamespaces.CMEO.value["graph/studies_metadata"])
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
            print("No metadata found in the file")
            return None
    else:
        print("Recreate flag is set to False. Skipping processing of study metadata.")


def create_cohort_specific_metadata_graph(dir_path, recreate=False):
    # dir_path should be cohort folder which can have each sub-folder as cohort name and each cohort folder should have metadata file with same name as cohort folder
    # g = init_graph()study
    # project directory path

    base_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Base path: {base_path}")
    if  recreate:
        for cohort_folder in os.listdir(dir_path):
            if cohort_folder.startswith('.'):  # Skip hidden files like .DS_Store
                continue
            start_time = time.time()
            cohort_path = os.path.join(dir_path, cohort_folder)
            if os.path.isdir(cohort_path):
                # Select most recent CSV file with 'datadictionary' in the name
                csv_candidates = [
                    f for f in glob.glob(os.path.join(cohort_path, "*.csv"))
                    if ("datadictionary" in os.path.basename(f).lower()
                    and "noheader" not in os.path.basename(f).lower())
                ]
                cohort_metadata_file = None
                if csv_candidates:
                    cohort_metadata_file = max(csv_candidates, key=os.path.getmtime)
                    print(f"Selected metadata file for {cohort_folder}: {cohort_metadata_file}")

                # Find EDA file in sibling folder named dcr_output_<cohort_folder>
                eda_file = None
                base_dir = os.path.dirname(dir_path)
                dcr_output_folder = os.path.join(base_dir, f"dcr_output_{cohort_folder}")
                if os.path.isdir(dcr_output_folder):
                    eda_candidates = [
                        f for f in glob.glob(os.path.join(dcr_output_folder, "eda*.json"))
                        if os.path.basename(f).lower().startswith("eda") and f.lower().endswith(".json")
                    ]
                    if eda_candidates:
                        eda_file = max(eda_candidates, key=os.path.getmtime)
                        print(f"Selected EDA file for {cohort_folder}: {eda_file}")
                # print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for metadata file: {cohort_metadata_file}")
                if cohort_metadata_file:
                    if eda_file and os.path.exists(eda_file):
                        print(f"Processing cohort: {cohort_folder} at path: {cohort_path} for eda file: {eda_file}")
                    g, cohort_id = process_variables_metadata_file(cohort_metadata_file, cohort_name=cohort_folder, eda_file_path=eda_file, study_metadata_graph_file_path=f"{base_path}/data/graphs/studies_metadata.trig")
                    if g and len(g) > 0:
                        # print(validate_graph(g))
                        g.serialize(f"{base_path}/data/graphs/{cohort_id}_metadata.trig", format="trig")
                        print(f"Publishing graph for cohort: {cohort_id}")
                  
                        #delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/{cohort_id}")
                        # res=publish_graph_to_endpoint(g, graph_uri=cohort_id) These lines are works with graphDB
                        delete_existing_triples(
                            get_cohort_mapping_uri(cohort_id)
                        )
                        res = publish_graph_to_endpoint(g)
                        print(f"Graph contains {len(g)} triples before serialization.")
                        print(f"Graph published to endpoint: {res} for cohort: {cohort_id}")
                        
                        end_time = time.time()
                        print(f"Time taken to process cohort: {cohort_folder} is: {end_time - start_time}")
                    else:
                        print(f"Error processing metadata file for cohort: {cohort_folder}")
                
            else:
                print(f"Skipping non-directory file: {cohort_folder}")
            print(f"Base path: {base_path}")
    else:
        print("Recreate flag is set to False. Skipping processing of cohort metadata.")

def create_pld_graph(file_path, cohort_name, output_dir=None, recreate=False) -> None:
    if recreate:
        start_time = time.time()
        # g=add_raw_data_graph(file_path, cohort_name)
        if len(g) > 0:
            g.serialize(f"{output_dir}/{cohort_name}_pld.trig", format="trig")
            # delete_existing_triples(f"{settings.sparql_endpoint}/rdf-graphs/{cohort_name}_pld")
            # res=publish_graph_to_endpoint(g,graph_uri=f"{cohort_name}_pld")
            
            delete_existing_triples(f"{get_cohort_mapping_uri(cohort_name)}_pld")
            res=publish_graph_to_endpoint(g)

            print(f"Graph published to endpoint: {res} for cohort: graph/{cohort_name}_pld")
            end_time = time.time()
            print(f"Time taken to process PLD: graph/{cohort_name}_pld is: {end_time - start_time}")
        else:
            print("No data found in the file")
    else:
        print("Recreate flag is set to False. Skipping processing of PLD data.")


def check_if_data_exists(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX omop: <http://omop.org/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>

    ASK WHERE {
      GRAPH ?graph {
        ?s omop:Has_omop_id ?o .
      }
    }
    """
    
    sparql.setQuery(query)
    results = sparql.query().convert()
    
    return results['boolean']

def list_graphs(endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    sparql.setQuery("""
                    
    SELECT DISTINCT ?graph
    WHERE {
      GRAPH ?graph {
        ?s ?p ?o .
      }
    }
    """)
    results = sparql.query().convert()
    return [result["graph"]["value"] for result in results["results"]["bindings"]]

def get_all_predicates(endpoint_url, graph_uris):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    
    query = """
    SELECT DISTINCT ?p
    WHERE {
      VALUES ?graph { """ + " ".join(f"<{uri}>" for uri in graph_uris) + """ }
      GRAPH ?graph {
        ?s ?p ?o .
      }
    }
    """
    
    sparql.setQuery(query)
    results = sparql.query().convert()
    predicates = [result["p"]["value"] for result in results["results"]["bindings"]]
    print(f"predicates = {predicates}")
    return predicates


def cluster_variables_by_omop(endpoint_url):
    """
    Clusters variables across cohorts based on their omop_id and displays the clusters as a pandas DataFrame.

    :param endpoint_url: str, the SPARQL endpoint URL.
    :return: pd.DataFrame, DataFrame containing omop_id and associated variable names.
    """
    # Check if data exists at the endpoint
    exists = check_if_data_exists(endpoint_url)

    
    if not exists:
        return pd.DataFrame(columns=["omop_id", "variables"])
    
    # Initialize SPARQLWrapper
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    
    # Define the revised SPARQL query
    query = """
    PREFIX cmeo: <https://w3id.org/CMEO/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?omop_id (GROUP_CONCAT(?variable_label; SEPARATOR=", ") AS ?variables)
    WHERE {
      GRAPH ?graph {
        ?variable_uri rdf:type cmeo:variable .
        ?variable_uri cmeo:has_concept ?base_entity .
        ?base_entity omop:has_omop_id ?omop_id .
        ?variable_uri rdfs:label ?variable_label .
      }
    }
    GROUP BY ?omop_id
    HAVING (COUNT(?variable_uri) > 1)
    """
    
    sparql.setQuery(query)
    
    try:
        # Execute the query
        results = sparql.query().convert()
 
    except Exception as e:

        return pd.DataFrame(columns=["omop_id", "variables"])
    
    # Process the results
    clusters = []
    for result in results["results"]["bindings"]:
        omop_id = result["omop_id"]["value"]
        variables_str = result["variables"]["value"]
        variables = [var.strip() for var in variables_str.split(",")]
        clusters.append({"omop_id": omop_id, "variables": variables})

    
    # Convert to pandas DataFrame
    if clusters:
        df = pd.DataFrame(clusters)

    else:
        df = pd.DataFrame(columns=["omop_id", "variables"])
    return df


    
# def search_studyx_elements():


# from owlready2 import get_ontology

# simple intersectio with just variable names display
def combine_cross_mappings_v1(
    source_study, 
    target_studies, 
    output_dir, 
    combined_output_path
):
    """
    Combines individual study cross-mapping files into a single grouped file by OMOP ID.
    """
    omop_id_tracker = {}
    mapping_dict = {}

    for tstudy in target_studies:
        out_path = os.path.join(output_dir, f'{source_study}_{tstudy}_full.csv')
        df = pd.read_csv(out_path)
        if tstudy not in mapping_dict:
            mapping_dict[tstudy] = {}
        for _, row in df.iterrows():
            src = str(row["source"]).strip()
            tgt = str(row["target"]).strip()
            somop = str(row["somop_id"]).strip()
            tomop = str(row["tomop_id"]).strip()
            slabel = str(row.get("slabel", "")).strip()
            if src not in omop_id_tracker:
                omop_id_tracker[src] = (somop, slabel)
            mapping_dict[tstudy][src] = (tgt, tomop)

    # Group source variables by OMOP ID
    omop_to_source_vars = defaultdict(list)
    for src_var, (somop_id, slabel) in omop_id_tracker.items():
        omop_to_source_vars[somop_id].append(src_var)

    matched_rows = []
    for _, src_vars in omop_to_source_vars.items():
        row = {}
        row[source_study] = ' | '.join(sorted(set(src_vars)))
        for tstudy in target_studies:
            targets = []
            tdict = mapping_dict.get(tstudy, {})
            for src_var in src_vars:
                tgt_pair = tdict.get(src_var)
                if tgt_pair:
                    targets.append(tgt_pair[0])
            row[tstudy] = ' | '.join(sorted(set(targets))) if targets else ''
        matched_rows.append(row)

    final_df = pd.DataFrame(matched_rows)
    final_df.to_csv(combined_output_path, index=False)
    print(f"✅ Combined existing mappings saved to: {combined_output_path}")
    
def combine_cross_mappings(
    source_study,
    target_studies,
    output_dir,
    combined_output_path,
    extra_columns=None
):
    if extra_columns is None:
        extra_columns = [
            "category","mapping_type","source_visit","target_visit",
            "source_type","source_unit","source_data_type",
            "target_type","target_unit","target_data_type","transformation_rule"
        ]

    omop_id_tracker = {}
    mapping_dict = {}
    mapping_details = {}
    source_details = {}

    for tstudy in target_studies:
        out_path = os.path.join(output_dir, f'{source_study}_{tstudy}_full.csv')
        df = pd.read_csv(out_path)
        if tstudy not in mapping_dict:
            mapping_dict[tstudy] = {}
            mapping_details[tstudy] = {}
        for _, row in df.iterrows():
            src = str(row["source"]).strip()
            tgt = str(row["target"]).strip()
            somop = str(row["somop_id"]).strip()
            tomop = str(row["tomop_id"]).strip()
            slabel = str(row.get("slabel", "")).strip()
            if src not in omop_id_tracker:
                omop_id_tracker[src] = (somop, slabel)
            mapping_dict[tstudy][src] = (tgt, tomop)
            # Target variable details
            tdetail_pieces = []
            for col in extra_columns:
                if col == "transformation_rule":
                    val = row.get(col, "")
                    if pd.isna(val) or val == "":
                        val = ""
                    else:
                        val = f"{src}→{tgt}:{val}"
                    tdetail_pieces.append(f"{col}={val}")
                else:
                    val = row.get(col, "")
                    if pd.isna(val):
                        val = ""
                    tdetail_pieces.append(f"{col}={val}")
            mapping_details[tstudy][src] = (tgt, ", ".join(tdetail_pieces))
            # Source variable details (collected once per unique source var)
            if src not in source_details:
                sdetail_pieces = []
                for col in extra_columns:
                    sval = row.get(f"source_{col}", row.get(col, ""))
                    if pd.isna(sval):
                        sval = ""
                    sdetail_pieces.append(f"{col}={sval}")
                source_details[src] = ", ".join(sdetail_pieces)

    # Group source variables by OMOP ID
    omop_to_source_vars = defaultdict(list)
    for src_var, (somop_id, slabel) in omop_id_tracker.items():
        omop_to_source_vars[somop_id].append(src_var)

    matched_rows = []
    for _, src_vars in omop_to_source_vars.items():
        row = {}
        # Source study: list source vars with details
        row[source_study] = ' | '.join(
            f"{src_var}: {source_details.get(src_var, '')}" for src_var in sorted(set(src_vars))
        )
        # Each target: list mapped target vars with details
        for tstudy in target_studies:
            tdict = mapping_dict.get(tstudy, {})
            tdetail = mapping_details.get(tstudy, {})
            targets = []
            for src_var in src_vars:
                tgt_pair = tdict.get(src_var)
                if tgt_pair:
                    tgt = tgt_pair[0]
                    detail_str = tdetail.get(src_var, "")
                    targets.append(f"{tgt}: {detail_str}")
            row[tstudy] = ' | '.join(targets) if targets else ''
        matched_rows.append(row)

    final_df = pd.DataFrame(matched_rows)
    final_df.to_csv(combined_output_path, index=False)
    print(f"✅ Combined existing mappings with source and target details saved to: {combined_output_path}")
  
  

def combine_all_mappings_to_json(
    source_study, target_studies, output_dir, json_path
):
    # Dict: {source_var: [mapping_dicts]}
    mappings = {}
    for target in target_studies:
        crossmap_csv = os.path.join(output_dir, f"{source_study}_{target}_cross_mapping.csv")
        if not os.path.exists(crossmap_csv):
            print(f"Skipping {crossmap_csv}, does not exist.")
            continue
        print(f"Inside combine_all_mappings_to_json - Processing {crossmap_csv}")
        df = pd.read_csv(crossmap_csv)
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
      
def generate_mapping_csv(
    source_study,
    target_studies,
    data_dir=None,
    cohort_file_path=None,
    cohorts_metadata_file=None,
    output_dir=None
):
    """
    Generate mapping CSV files for a source study and a list of target studies.

    Args:
        source_study (str): The name of the source study.
        target_studies (list of tuple): Each tuple is (target_study_name, visit_constraint_bool).
        data_dir (str, optional): Directory containing data files. Defaults to the 'data' folder inside CohortVarLinker.
        cohort_file_path (str, optional): Path to the cohorts directory. Defaults to settings.cohort_folder.
        cohorts_metadata_file (str, optional): Path to the cohort metadata file. Defaults to f"{data_dir}/cohort_metadata_sheet_v2.csv".
        output_dir (str, optional): Directory to store output mapping CSVs. Defaults to 'mapping_output' inside CohortVarLinker.
    
    Returns:
        dict: Cache information with 'cached_pairs' and 'uncached_pairs' lists containing pair info and timestamps.
    """

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    from CohortVarLinker.src.config import settings
    if cohort_file_path is None:
        cohort_file_path = settings.cohort_folder
    # Robust check: ensure all selected cohorts exist
   
    missing_cohorts = []
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
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mapping_output')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Checking for cached or ready mapping files in directory: {os.path.abspath(output_dir)}")

    source_study = source_study.lower()
    target_studies = [t[0].lower() for t in target_studies]

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
                output_dir= "/app/CohortVarLinker/mapping_output/",
                json_path= f"/app/CohortVarLinker/mapping_output/{source_study}_omop_id_grouped_{tstudy_str}.json")
        
        return cache_info
            
    # Only run expensive computations if any mapping is missing
    create_study_metadata_graph(cohorts_metadata_file, recreate=True)
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=True)
        
    # combined_df = None
    omop_id_tracker = {}  # Track source_omop_id per variable
    
    mapping_dict = {}  # {target_study: {source_var: (target_var, target_omop_id)}}
    # Use 'qdrant' as the host when running in Docker Compose
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "qdrant", "studies_metadata", recreate_db=True)
    graph = OmopGraphNX(csv_file_path=settings.concepts_file_path)
    for tstudy in target_studies:
        out_filename = f'{source_study}_{tstudy}_cross_mapping.csv'
        out_path = os.path.join(output_dir, out_filename)
        if os.path.exists(out_path):
            print(f"Mapping already exists for {source_study} to {tstudy}, skipping computation.")
            continue
        mapping_transformed = map_source_target(
            source_study_name=source_study,
            target_study_name=tstudy,
            embedding_model=embedding_model,
            vector_db=vector_db,
            collection_name="studies_metadata",
            graph=graph,
        ) # if empty it will return empty DataFrame with header not None
    
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
        output_dir= "/app/CohortVarLinker/mapping_output/",
        json_path= f"/app/CohortVarLinker/mapping_output/{source_study}_omop_id_grouped_{tstudy_str}.json")
    
    return cache_info
    #         if tstudy not in mapping_dict:
    #             mapping_dict[tstudy] = {}
    #         for _, row in mapping_transformed.iterrows():
    #                 src = str(row["source"]).strip()
    #                 tgt = str(row["target"]).strip()
    #                 somop = str(row["somop_id"]).strip()
    #                 tomop = str(row["tomop_id"]).strip()
    #                 slabel = str(row.get("slabel", "")).strip()
    #                 if src not in omop_id_tracker:
    #                     omop_id_tracker[src] = (somop, slabel)
    #                 mapping_dict[tstudy][src] = (tgt, tomop)


    # # 1. Group TIME-CHF source variables by OMOP ID
    # omop_to_source_vars = defaultdict(list)
    # for src_var, (somop_id, slabel) in omop_id_tracker.items():
    #     omop_to_source_vars[somop_id].append(src_var)

    # matched_rows = []

    # # 2. For each OMOP ID, build a row: all TIME-CHF vars and all target study matches
    # for _, src_vars in omop_to_source_vars.items():
    #     row = {}
    #     row[source_study] = ' | '.join(sorted(set(src_vars)))
    #     for tstudy in target_studies:
    #         targets = []
    #         tdict = mapping_dict.get(tstudy, {})
    #         for src_var in src_vars:
    #             tgt_pair = tdict.get(src_var)
    #             if tgt_pair:
    #                 targets.append(tgt_pair[0])
    #         row[tstudy] = ' | '.join(sorted(set(targets))) if targets else ''
    #     matched_rows.append(row)

    # # 3. Save the DataFrame
    # final_df = pd.DataFrame(matched_rows)
    # output_path = f'{data_dir}/output/{source_study}_omop_id_grouped_all_targets.csv'  #anas please update it accordingly
    # final_df.to_csv(output_path, index=False) # merged output file with all targets where studies names are columns and source variables are grouped by omop_id
    # print(f"✅ Matched variables (grouped by source OMOP ID) saved to: {output_path}")  
    
    
