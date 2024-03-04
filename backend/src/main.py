import glob
import os
import shutil
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from rdflib import DCTERMS, XSD, Dataset, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DC, RDF, RDFS
from SPARQLWrapper import JSON, SPARQLWrapper
from starlette.middleware.cors import CORSMiddleware

from src.auth import get_current_user
from src.auth import router as auth_router
from src.config import settings
from src.decentriq import create_provision_dcr
from src.decentriq import router as decentriq_router

app = FastAPI(
    title="iCARE4CVD API",
    description="""Upload and explore cohorts metadata files for the [iCARE4CVD project](https://icare4cvd.eu/).""",
)
app.include_router(decentriq_router)
app.include_router(auth_router, tags=["auth"])

query_endpoint = SPARQLWrapper(f"{settings.sparql_endpoint}/query")
query_endpoint.setReturnFormat(JSON)

# Define the namespaces
ICARE = Namespace("https://w3id.org/icare4cvd/")


def init_graph(default_graph: str | None = None) -> Dataset:
    """Initialize a new RDF graph for nquads with the iCARE4CVD namespace bindings."""
    g = Dataset(store="Oxigraph", default_graph_base=default_graph)
    g.bind("icare", ICARE)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    return g


def run_query(query) -> dict[str, Any]:
    """Function to run a SPARQL query against a remote endpoint"""
    query_endpoint.setQuery(query)
    # print(sparql.query().convert())
    return query_endpoint.query().convert()


def publish_graph_to_endpoint(g: Graph, graph_uri: str | None = None) -> bool:
    """Insert the graph into the triplestore endpoint."""
    # url = f"{settings.sparql_endpoint}/store?{graph_uri}"
    url = f"{settings.sparql_endpoint}/store"
    if graph_uri:
        url += f"?graph={graph_uri}"
    headers = {"Content-Type": "application/trig"}
    g.serialize("/tmp/upload-data.trig", format="trig")
    with open("/tmp/upload-data.trig", "rb") as file:
        response = requests.post(url, headers=headers, data=file, timeout=120)

    # NOTE: Fails when we pass RDF as string directly
    # response = requests.post(url, headers=headers, data=graph_data)
    # Check response status and print result
    if not response.ok:
        print(f"Failed to upload data: {response.status_code}, {response.text}")
    return response.ok


def delete_existing_triples(cohort_uri: str):
    """Function to delete existing triples in a cohort's graph"""
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    DELETE WHERE {{
        GRAPH <{cohort_uri}> {{ ?s ?p ?o . }}
    }}
    """
    query_endpoint = SPARQLWrapper(f"{settings.sparql_endpoint}/update")
    query_endpoint.setMethod('POST')
    query_endpoint.setRequestMethod('urlencoded')
    query_endpoint.setQuery(query)
    query_endpoint.query()

def parse_categorical_string(s) -> list[dict[str, str]]:
    """Categorical string format: "value1=label1, value2=label2" or "value1=label1 | value2=label2"""
    # Split the string into items
    split_char = "," if "|" not in s else "|"
    items = s.split(split_char)
    result = []
    for item in items:
        if item:
            try:
                key, value = map(str.strip, item.split("="))
            except Exception:
                key, value = item.strip(), item.strip()
            result.append({"value": key, "label": value})
    if s and len(result) < 1:
        raise HTTPException(
            status_code=422,
            detail="Error parsing categorical string",
        )
    return result


curie_columns = {"ICD-10": "icd10", "SNOMED-CT": "snomedct", "ATC-DDD": "atc", "LOINC": "loinc"}


def create_curie_from_id(row):
    curies = []
    for column in curie_columns:
        if row[column]:
            if "," in str(row[column]):  # Handle list of IDs separated by comma
                ids = str(row[column]).split(",")
                curies.extend([f"{curie_columns[column]}:{identif.strip()}" for identif in ids])
            else:
                curies.append(f"{curie_columns[column]}:{str(row[column]).strip()}")
    return ", ".join(curies)

# TODO: add arg to get data owner email, and add it to the graph
def load_cohort_dict_file(dict_path: str, cohort_id: str, owner_email: str):
    """Parse the cohort dictionary uploaded as excel or CSV spreadsheet, and load it to the triplestore"""
    # print(f"Loading dictionary {dict_path}")
    df = pd.read_csv(dict_path) if dict_path.endswith(".csv") else pd.read_excel(dict_path)
    df = df.dropna(how="all")
    df = df.fillna("")
    df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)
    df["concept_id"] = df.apply(lambda row: create_curie_from_id(row), axis=1)

    # TODO: add metadata about the cohort, dc:creator
    cohort_uri = ICARE[f"cohort/{cohort_id.strip().replace(' ', '_')}"]
    g = init_graph()
    g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_uri))
    g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_uri))
    g.add((cohort_uri, ICARE["owner"], Literal(owner_email), cohort_uri))

    for i, row in df.iterrows():
        # Check if required columns are present
        if not row["VARIABLE NAME"] or not row["VARIABLE LABEL"] or not row["VAR TYPE"]:
            raise HTTPException(
                status_code=422,
                detail="Row is missing required data: variable_name, variable_label, or var_type",
            )
        # TODO: raise error when duplicate value for VARIABLE LABEL

        # Create a URI for the variable
        variable_uri = URIRef(f"{cohort_uri!s}/{row['VARIABLE NAME'].replace(' ', '_')}")

        # Add the type of the resource
        g.add((variable_uri, RDF.type, ICARE.Variable, cohort_uri))
        g.add((variable_uri, DCTERMS.isPartOf, cohort_uri, cohort_uri))
        g.add((variable_uri, DC.identifier, Literal(row["VARIABLE NAME"]), cohort_uri))
        g.add((variable_uri, RDFS.label, Literal(row["VARIABLE LABEL"]), cohort_uri))
        g.add((variable_uri, ICARE["index"], Literal(i, datatype=XSD.integer), cohort_uri))

        # Add properties
        for column, value in row.items():
            # if value and column not in ["categories"]:
            if column not in ["categories"] and value:
                property_uri = ICARE[column.replace(" ", "_").lower()]
                g.add((variable_uri, property_uri, Literal(value), cohort_uri))

            # Handle Category
            if column in ["categories"]:
                for index, category in enumerate(value):
                    cat_uri = URIRef(f"{variable_uri!s}/category/{index}")
                    g.add((variable_uri, ICARE["categories"], cat_uri, cohort_uri))
                    g.add((cat_uri, RDF.type, ICARE.Category, cohort_uri))
                    g.add((cat_uri, RDF.value, Literal(category["value"]), cohort_uri))
                    g.add((cat_uri, RDFS.label, Literal(category["label"]), cohort_uri))
    # print(g.serialize(format="turtle"))
    return g


@app.post(
    "/upload",
    response_description="Upload result",
    response_model={},
)
async def upload_files(
    user: Any = Depends(get_current_user),
    # cohort_id: str = Form(..., pattern="^[a-zA-Z0-9-_]+$"),
    cohort_id: str = Form(...),
    cohort_dictionary: UploadFile = File(...),
    cohort_data: UploadFile | None = None,
) -> dict[str, str]:
    """Upload files to the server"""
    # Create directory named after cohort_id
    cohorts_folder = os.path.join(settings.data_folder, "cohorts", cohort_id)
    os.makedirs(cohorts_folder, exist_ok=True)
    # print("USER", user)

    cohorts_info = get_cohorts_metadata().get(cohort_id)
    # Check if cohort already uploaded
    if len(cohorts_info.get("variables")) > 0:
        print("OWNER", cohorts_info.get("owner"))
        print("USER EMAIL", user["email"])
        if cohorts_info.get("owner") != user["email"]:
            raise HTTPException(status_code=403, detail=f"You are not the owner of cohort {cohort_id}.")
        # Check for existing data dictionary file and back it up
        for file_name in os.listdir(cohorts_folder):
            if file_name.endswith("_datadictionary.csv"):
                # Construct the backup file name with the current date
                backup_file_name = f"{file_name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d')}.csv"
                backup_file_path = os.path.join(cohorts_folder, backup_file_name)
                existing_file_path = os.path.join(cohorts_folder, file_name)
                # Rename (backup) the existing file
                os.rename(existing_file_path, backup_file_path)
                break  # Assuming there's only one data dictionary file per cohort

        # Delete graph for this file from triplestore
        cohort_uri = ICARE[f"cohort/{cohort_id.strip().replace(' ', '_')}"]
        delete_existing_triples(cohort_uri)


    # Make sure metadata file ends with _datadictionary
    metadata_filename = cohort_dictionary.filename
    filename, ext = os.path.splitext(metadata_filename)
    if not filename.endswith("_datadictionary"):
        filename += "_datadictionary"

    # Save metadata file
    metadata_path = os.path.join(cohorts_folder, filename + ext)
    with open(metadata_path, "wb") as buffer:
        shutil.copyfileobj(cohort_dictionary.file, buffer)

    try:
        g = load_cohort_dict_file(metadata_path, cohort_id, user["email"])
        publish_graph_to_endpoint(g)
    except Exception as e:
        shutil.rmtree(metadata_path)
        raise e

    cohorts_dict = get_cohorts_metadata()
    dcr_data = create_provision_dcr(user, cohorts_dict[cohort_id])
    # print(dcr_data)
    # Save data file
    if cohort_data:
        file_path = os.path.join(cohorts_folder, cohort_data.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(cohort_data.file, buffer)
    return dcr_data


get_variables_query = """PREFIX icare: <https://w3id.org/icare4cvd/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT DISTINCT ?cohortId ?cohortInstitution ?cohortType ?cohortEmail ?owner ?study_type ?study_participants
    ?study_duration ?study_ongoing ?study_population ?study_objective
    ?variable ?varName ?varLabel ?varType ?index ?count ?na ?max ?min ?units ?formula ?definition
    ?omopDomain ?conceptId ?visits ?categoryValue ?categoryLabel
WHERE {
    GRAPH ?cohortMetadataGraph {
        ?cohort a icare:Cohort ;
            dc:identifier ?cohortId ;
            icare:institution ?cohortInstitution .
        OPTIONAL { ?cohort icare:cohort_type ?cohortType . }
        OPTIONAL { ?cohort icare:email ?cohortEmail . }
        OPTIONAL { ?cohort icare:study_type ?study_type . }
        OPTIONAL { ?cohort icare:study_participants ?study_participants . }
        OPTIONAL { ?cohort icare:study_duration ?study_duration . }
        OPTIONAL { ?cohort icare:study_ongoing ?study_ongoing . }
        OPTIONAL { ?cohort icare:study_population ?study_population . }
        OPTIONAL { ?cohort icare:study_objective ?study_objective . }
    }

    OPTIONAL {
        GRAPH ?cohortVarGraph {
            OPTIONAL { ?cohort icare:owner ?owner . }
            ?variable a icare:Variable ;
                dc:identifier ?varName ;
                rdfs:label ?varLabel ;
                icare:var_type ?varType ;
                icare:index ?index ;
                dcterms:isPartOf ?cohort .
            OPTIONAL { ?variable icare:count ?count }
            OPTIONAL { ?variable icare:na ?na }
            OPTIONAL { ?variable icare:max ?max }
            OPTIONAL { ?variable icare:min ?min }
            OPTIONAL { ?variable icare:units ?units }
            OPTIONAL { ?variable icare:formula ?formula }
            OPTIONAL { ?variable icare:definition ?definition }
            OPTIONAL { ?variable icare:concept_id ?conceptId }
            OPTIONAL { ?variable icare:omop ?omopDomain }
            OPTIONAL { ?variable icare:visits ?visits }
            OPTIONAL {
                ?variable icare:categories ?category.
                ?category rdfs:label ?categoryLabel ;
                    rdf:value ?categoryValue .
            }
        }
    }
} ORDER BY ?cohort ?index
"""


def get_cohorts_metadata() -> dict[str, Any]:
    """Get all cohorts metadata from the SPARQL endpoint (infos, variables)"""
    results = run_query(get_variables_query)["results"]["bindings"]
    cohorts_with_variables = {}
    cohorts_without_variables = {}
    # print(f"Query results: {len(results)}")
    for row in results:
        cohort_id = str(row["cohortId"]["value"])
        var_id = str(row["varName"]["value"]) if "varName" in row else ""
        # Determine which dictionary to use
        target_dict = cohorts_with_variables if var_id else cohorts_without_variables

        # Initialize cohort data structure if not exists
        if cohort_id and cohort_id not in target_dict:
            target_dict[cohort_id] = {
                "cohort_id": cohort_id,
                "cohort_type": str(row["cohortType"]["value"]) if "cohortType" in row else "",
                "cohort_email": str(row["cohortEmail"]["value"]) if "cohortEmail" in row else "",
                "owner": str(row["owner"]["value"]) if "owner" in row else "",
                "institution": str(row["cohortInstitution"]["value"]),
                "study_type": str(row["study_type"]["value"]) if "study_type" in row else "",
                "study_participants": str(row["study_participants"]["value"]) if "study_participants" in row else "",
                "study_duration": str(row["study_duration"]["value"]) if "study_duration" in row else "",
                "study_ongoing": str(row["study_ongoing"]["value"]) if "study_ongoing" in row else "",
                "study_population": str(row["study_population"]["value"]) if "study_population" in row else "",
                "study_objective": str(row["study_objective"]["value"]) if "study_objective" in row else "",
                "variables": {},
            }

        # Process variables
        if "varName" in row and var_id not in target_dict[cohort_id]["variables"]:
            target_dict[cohort_id]["variables"][var_id] = {
                "var_name": var_id,
                "var_label": str(row["varLabel"]["value"]),
                "var_type": str(row["varType"]["value"]),
                "count": int(row["count"]["value"]),
                "max": str(row["max"]["value"]) if "max" in row else "",
                "min": str(row["min"]["value"]) if "min" in row else "",
                "units": str(row["units"]["value"]) if "units" in row else "",
                "visits": str(row["visits"]["value"]) if "visits" in row else "",
                "formula": str(row["formula"]["value"]) if "formula" in row else "",
                "definition": str(row["definition"]["value"]) if "definition" in row else "",
                "concept_id": str(row["conceptId"]["value"]) if "conceptId" in row else "",
                "omop_domain": str(row["omopDomain"]["value"]) if "omopDomain" in row else "",
                "index": int(row["index"]["value"]) if "index" in row else "",
                "categories": [],
            }
            if "na" in row:
                target_dict[cohort_id]["variables"][var_id]["na"] = int(row["na"]["value"])

        # Process categories of variables
        if "varName" in row and "categoryLabel" in row and "categoryValue" in row:
            target_dict[cohort_id]["variables"][var_id]["categories"].append(
                {"value": str(row["categoryValue"]["value"]), "label": str(row["categoryLabel"]["value"])}
            )

    # Merge dictionaries, cohorts with variables first
    merged_cohorts_data = {**cohorts_with_variables, **cohorts_without_variables}
    # print(merged_cohorts_data)
    # return JSONResponse(merged_cohorts_data)
    return merged_cohorts_data


@app.get("/summary")
def get_data_summary(user: Any = Depends(get_current_user)) -> dict[str, Any]:
    """Returns data dictionaries of all cohorts"""
    return get_cohorts_metadata()

@app.get("/cohort-spreadsheet/{cohort_id}")
async def get_cohort_spreasheet(cohort_id: str, user: Any = Depends(get_current_user)) -> FileResponse:
    """Download the data dictionary of a specified cohort as a spreadsheet."""
    # cohort_id = urllib.parse.unquote(cohort_id)
    cohorts_folder = os.path.join(settings.data_folder, "cohorts", cohort_id)

    # Search for a data dictionary file in the cohort's folder
    for file_name in os.listdir(cohorts_folder):
        if file_name.endswith("_datadictionary.csv") or file_name.endswith("_datadictionary.xlsx"):
            file_path = os.path.join(cohorts_folder, file_name)
            return FileResponse(path=file_path, filename=file_name, media_type="application/octet-stream")

    # If no file is found, return an error response
    raise HTTPException(status_code=404, detail=f"No data dictionary found for cohort ID '{cohort_id}'")

@app.get("/", include_in_schema=False)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the route / to /docs"""
    return RedirectResponse(url="/docs")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_triplestore():
    """Initialize triplestore with the OMOP CDM ontology and the iCARE4CVD cohorts metadata."""
    # If triples exist, skip initialization
    if run_query("ASK WHERE { GRAPH ?g {?s ?p ?o .} }")["boolean"]:
        print("⏩ Triplestore already contains data. Skipping initialization.")
        return
    # Load OMOP CDM ontology
    onto_graph_uri = URIRef("https://w3id.org/icare4cvd/omop-cdm-v6")
    g = init_graph(onto_graph_uri)
    ntriple_g = init_graph()
    ntriple_g.parse("https://raw.githubusercontent.com/vemonet/omop-cdm-owl/main/omop_cdm_v6.ttl", format="turtle")
    # Trick to convert ntriples to nquads with a given graph
    for s, p, o in ntriple_g.triples((None, None, None)):
        g.add((s, p, o, onto_graph_uri))
    # print(g.serialize(format="trig"))
    if publish_graph_to_endpoint(g):
        print(f"🦉 Triplestore initialization: added {len(g)} triples for the OMOP OWL ontology.")

    # Load cohorts data dictionaries already present in data/cohorts/
    for folder in os.listdir(os.path.join(settings.data_folder, "cohorts")):
        folder_path = os.path.join(settings.data_folder, "cohorts", folder)
        if os.path.isdir(folder_path):
            for file in glob.glob(os.path.join(folder_path, "*_datadictionary.*")):
                # TODO: currently when we reset all existing cohorts default to the main admin
                g = load_cohort_dict_file(file, folder, settings.decentriq_email)
                g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.trig", format="trig")
                if publish_graph_to_endpoint(g):
                    print(f"💾 Triplestore initialization: added {len(g)} triples for cohorts {file}.")

    # Load cohorts metadata
    df = pd.read_excel(f"{settings.data_folder}/iCARE4CVD_Cohorts.xlsx", sheet_name="Descriptions")
    df = df.fillna("")

    g = init_graph()
    for _i, row in df.iterrows():
        cohort_id = str(row["Name of Study"]).strip()
        # print(cohort_id)
        cohort_uri = ICARE[f"cohort/{cohort_id.replace(' ', '_')}"]
        cohorts_graph = ICARE["graph/metadata"]

        g.add((cohort_uri, RDF.type, ICARE.Cohort, cohorts_graph))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohorts_graph))
        g.add((cohort_uri, ICARE.institution, Literal(row["Institution"]), cohorts_graph))
        if row["Contact partner"]:
            g.add((cohort_uri, DC.creator, Literal(row["Contact partner"]), cohorts_graph))
        if row["Email"]:
            g.add((cohort_uri, ICARE.email, Literal(row["Email"]), cohorts_graph))
        if row["Type"]:
            g.add((cohort_uri, ICARE.cohort_type, Literal(row["Type"]), cohorts_graph))
        if row["Study type"]:
            g.add((cohort_uri, ICARE.study_type, Literal(row["Study type"]), cohorts_graph))
        if row["N"]:
            g.add((cohort_uri, ICARE.study_participants, Literal(row["N"]), cohorts_graph))
        if row["Study duration"]:
            g.add((cohort_uri, ICARE.study_duration, Literal(row["Study duration"]), cohorts_graph))
        if row["Ongoing"]:
            g.add((cohort_uri, ICARE.study_ongoing, Literal(row["Ongoing"]), cohorts_graph))
        if row["Patient population"]:
            g.add((cohort_uri, ICARE.study_population, Literal(row["Patient population"]), cohorts_graph))
        if row["Primary objective"]:
            g.add((cohort_uri, ICARE.study_objective, Literal(row["Primary objective"]), cohorts_graph))

    # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.ttl", format="turtle")
    if publish_graph_to_endpoint(g):
        print(f"🪪 Triplestore initialization: added {len(g)} triples for the cohorts metadata.")


init_triplestore()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
