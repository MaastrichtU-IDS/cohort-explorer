import glob
import os
import shutil
from typing import Any

import pandas as pd
import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from rdflib import DCTERMS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DC, RDF, RDFS
from SPARQLWrapper import JSON, SPARQLWrapper
from starlette.middleware.cors import CORSMiddleware

from src.auth import get_user_info
from src.auth import router as auth_router
from src.config import settings
from src.decentriq import create_provision_dcr
from src.decentriq import router as decentriq_router

app = FastAPI(
    title="iCARE4CVD data upload",
    description="""Upload data files on Maastricht University servers for the [iCARE4CVD project](https://icare4cvd.eu/).

If you are facing issues, contact [vincent.emonet@maastrichtuniversity.nl](mailto:vincent.emonet@maastrichtuniversity.nl)""",
)


g = Graph(store="Oxigraph")
query_endpoint = SPARQLWrapper(f"{settings.sparql_endpoint}/query")
query_endpoint.setReturnFormat(JSON)
# /query or /update

# Define the namespaces
ICARE = Namespace("https://w3id.org/icare4cvd/")
g.bind("icare", ICARE)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)


def run_query(query) -> dict[str, Any]:
    """Function to run a SPARQL query against a remote endpoint"""
    query_endpoint.setQuery(query)
    # print(sparql.query().convert())
    return query_endpoint.query().convert()


def publish_graph_to_endpoint(graph: Graph) -> bool:
    """Insert the graph into the triplestore endpoint."""
    url = f"{settings.sparql_endpoint}/store?default"
    headers = {"Content-Type": "text/turtle"}
    graph.serialize("/tmp/upload-data.ttl", format="ttl")
    with open("/tmp/upload-data.ttl", "rb") as file:
        response = requests.post(url, headers=headers, data=file, timeout=120)

    # NOTE: Fails when we pass RDF as string directly
    # response = requests.post(url, headers=headers, data=graph_data)
    # Check response status and print result
    if not response.ok:
        print(f"Failed to upload data: {response.status_code}, {response.text}")
    return response.ok


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


def load_cohort_dict_file(dict_path: str, cohort_id: str):
    """Parse the cohort dictionary uploaded as excel or CSV spreadsheet, and load it to the triplestore"""
    print(f"Loading dictionary {dict_path}")
    df = pd.read_csv(dict_path) if dict_path.endswith(".csv") else pd.read_excel(dict_path)
    df = df.fillna("")
    df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)
    df["concept_id"] = df.apply(lambda row: create_curie_from_id(row), axis=1)

    # TODO: add metadata about the cohort, dc:creator
    cohort_uri = ICARE[f"cohort/{cohort_id}"]
    g.add((cohort_uri, RDF.type, ICARE.Cohort))
    g.add((cohort_uri, DC.identifier, Literal(cohort_id)))

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
        g.add((variable_uri, RDF.type, ICARE.Variable))
        g.add((variable_uri, DCTERMS.isPartOf, cohort_uri))
        g.add((variable_uri, DC.identifier, Literal(row["VARIABLE NAME"])))
        g.add((variable_uri, RDFS.label, Literal(row["VARIABLE LABEL"])))
        g.add((variable_uri, ICARE["index"], Literal(i, datatype=XSD.integer)))

        # Add properties
        for column, value in row.items():
            # if value and column not in ["categories"]:
            if column not in ["categories"]:
                property_uri = ICARE[column.replace(" ", "_").lower()]
                g.add((variable_uri, property_uri, Literal(value)))

            # Handle Category
            if column in ["categories"]:
                for index, category in enumerate(value):
                    cat_uri = URIRef(f"{variable_uri!s}/category/{index}")
                    g.add((variable_uri, ICARE["categories"], cat_uri))
                    g.add((cat_uri, RDF.type, ICARE.Category))
                    g.add((cat_uri, RDF.value, Literal(category["value"])))
                    g.add((cat_uri, RDFS.label, Literal(category["label"])))
    # print(g.serialize(format="turtle"))
    return g


@app.post(
    "/upload",
    response_description="Upload result",
    response_model={},
)
async def upload_files(
    user: Any = Depends(get_user_info),
    cohort_id: str = Form(..., pattern="^[a-zA-Z0-9-_]+$"),
    cohort_dictionary: UploadFile = File(...),
    cohort_data: UploadFile | None = None,
) -> dict[str, str]:
    """Upload files to the server"""
    # Create directory named after cohort_id
    cohorts_folder = os.path.join(settings.data_folder, "cohorts", cohort_id)
    os.makedirs(cohorts_folder, exist_ok=True)
    # print("USER", user)

    # Make sure metadata file ends with -dictionary
    metadata_filename = cohort_dictionary.filename
    filename, ext = os.path.splitext(metadata_filename)
    if not filename.endswith("-dictionary"):
        filename += "-dictionary"

    # Save metadata file
    metadata_path = os.path.join(cohorts_folder, filename + ext)
    with open(metadata_path, "wb") as buffer:
        shutil.copyfileobj(cohort_dictionary.file, buffer)

    try:
        g = load_cohort_dict_file(metadata_path, cohort_id)
        publish_graph_to_endpoint(g)
    except Exception as e:
        shutil.rmtree(cohorts_folder)
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

SELECT DISTINCT ?cohortId ?cohortInstitution ?cohortType ?cohortEmail ?variable ?varName ?varLabel ?varType ?index ?count ?na ?max ?min ?units ?formula ?definition ?omopDomain ?conceptId ?visits ?categoryValue ?categoryLabel
WHERE {
    ?cohort a icare:Cohort ;
        dc:identifier ?cohortId ;
        icare:institution ?cohortInstitution .
    OPTIONAL {
        ?cohort icare:cohort_type ?cohortType .
        ?cohort icare:email ?cohortEmail .
    }

    OPTIONAL {
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
                "institution": Literal(str(row["cohortInstitution"]["value"])),
                "variables": {},
            }

        # Process variables
        if "varName" in row and var_id not in target_dict[cohort_id]["variables"]:
            target_dict[cohort_id]["variables"][var_id] = {
                "var_name": var_id,
                "var_label": str(row["varLabel"]["value"]),
                "var_type": str(row["varType"]["value"]),
                "count": int(row["count"]["value"]),
                "na": int(row["na"]["value"]),
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

        # Process categories
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
def get_data_summary(user: Any = Depends(get_user_info)) -> dict[str, Any]:
    """Returns all data dictionaries"""
    return get_cohorts_metadata()


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
    if run_query("ASK WHERE { ?s ?p ?o . }")["boolean"]:
        print("Triplestore already contains data. Skipping initialization.")
        return

    g.parse("https://raw.githubusercontent.com/vemonet/omop-cdm-owl/main/omop_cdm_v6.ttl", format="turtle")
    for folder in os.listdir(os.path.join(settings.data_folder, "cohorts")):
        folder_path = os.path.join(settings.data_folder, "cohorts", folder)
        if os.path.isdir(folder_path):
            for file in glob.glob(os.path.join(folder_path, "*-dictionary.*")):
                load_cohort_dict_file(file, folder)

    df = pd.read_csv(f"{settings.data_folder}/iCARE4CVD_Cohorts.csv")
    df = df.fillna("")

    for _i, row in df.iterrows():
        cohort_id = str(row["Id"]).strip().replace(" ", "_")
        # print(cohort_id)
        cohort_uri = ICARE[f"cohort/{cohort_id}"]

        g.add((cohort_uri, RDF.type, ICARE.Cohort))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id)))
        g.add((cohort_uri, ICARE.institution, Literal(row["Institution"])))
        if row["Contact partner"]:
            g.add((cohort_uri, DC.creator, Literal(row["Contact partner"])))
        if row["Email"]:
            g.add((cohort_uri, ICARE.email, Literal(row["Email"])))
        if row["Type"]:
            g.add((cohort_uri, ICARE.cohort_type, Literal(row["Type"])))

    # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.ttl", format="turtle")
    if publish_graph_to_endpoint(g):
        print(f"Triplestore initialized successfully with {len(g)} triples.")


init_triplestore()

app.include_router(decentriq_router)
app.include_router(auth_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
