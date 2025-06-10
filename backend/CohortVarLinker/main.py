import pandas as pd
# import cProfile
# import pstats
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import glob
import time
from CohortVarLinker.src.variables_kg import process_variables_metadata_file, add_raw_data_graph
from CohortVarLinker.src.study_kg import generate_studies_kg
from CohortVarLinker.src.vector_db import generate_studies_embeddings, search_in_db
from CohortVarLinker.src.utils import (
        get_cohort_mapping_uri,
        delete_existing_triples,
        publish_graph_to_endpoint,
        OntologyNamespaces,
    
    )




from CohortVarLinker.src.fetch import find_hierarchy_of_variables, map_source_target



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
        g=add_raw_data_graph(file_path, cohort_name)
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
    target_studies = [(t[0].lower(), t[1]) for t in target_studies]
   
    # Check if all requested mappings already exist
    all_exist = True
    missing_targets = []
    for tstudy, vc in target_studies:
        suffix = 'restricted' if vc else 'full'
        out_filename = f'{source_study}_{tstudy}_{suffix}.csv'
        out_path = os.path.join(output_dir, out_filename)
        print(f"Checking if {out_path} exists")
        if not os.path.exists(out_path):
            all_exist = False
            missing_targets.append((tstudy, vc))
    if all_exist:
        print("All requested mappings already exist. Skipping all computation.")
        return

    # Only run expensive computations if any mapping is missing
    create_study_metadata_graph(cohorts_metadata_file, recreate=True)
    create_cohort_specific_metadata_graph(cohort_file_path, recreate=True)
    
    # Use 'qdrant' as the host when running in Docker Compose
    vector_db, embedding_model = generate_studies_embeddings(cohort_file_path, "qdrant", "studies_metadata", recreate_db=True)

    for tstudy, vc in target_studies:
        suffix = 'restricted' if vc else 'full'
        out_filename = f'{source_study}_{tstudy}_{suffix}.csv'
        out_path = os.path.join(output_dir, out_filename)
        if os.path.exists(out_path):
            print(f"Mapping already exists for {source_study} to {tstudy} (visit_constraint={vc}), skipping computation.")
            continue
        mapping_transformed = map_source_target(
            source_study_name=source_study,
            target_study_name=tstudy,
            embedding_model=embedding_model,
            vector_db=vector_db,
            collection_name="studies_metadata",
            visit_constraint=vc
        )
        print(mapping_transformed)
        mapping_transformed = mapping_transformed.drop_duplicates(keep='first')
        mapping_transformed.to_csv(out_path, index=False)

    # res=search_in_db(
    #     vectordb=vector_db,
    #     embedding_model=embedding_model,
    #     query_text="potassium [moles/volume] in blood",
    #     limit=100,
    #     omop_domain=["measurement"],
    #     collection_name="studies_metadata",
    #     target_study="gissi-hf",
    #     min_score=0.85
    # )
    # res =  set(res)
    # print(f"Results: {res}")
    
    # res=search_in_db(
    #     vectordb=vector_db,
    #     embedding_model=embedding_model,
    #     query_text="ace inhibitors, plain",
    #     limit=100,
    #     omop_domain=["drug_exposure"],
    #     collection_name="studies_metadata",
    #     min_score=0.85
    # )
    # res =  set(res)
    # print(f"Results: {res}")
  
    
