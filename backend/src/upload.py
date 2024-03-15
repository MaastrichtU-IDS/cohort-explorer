import glob
import os
import shutil
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from rdflib import DCTERMS, XSD, Dataset, Graph, Literal, URIRef
from rdflib.namespace import DC, RDF, RDFS
from SPARQLWrapper import SPARQLWrapper

from src.auth import get_current_user
from src.config import settings
from src.decentriq import create_provision_dcr
from src.utils import ICARE, converter, init_graph, retrieve_cohorts_metadata, run_query

router = APIRouter()


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


def delete_existing_triples(cohort_uri: str | URIRef, subject="?s", predicate="?p"):
    """Function to delete existing triples in a cohort's graph"""
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    DELETE WHERE {{
        GRAPH <{cohort_uri!s}> {{ {subject} {predicate} ?o . }}
    }}
    """
    query_endpoint = SPARQLWrapper(settings.update_endpoint)
    query_endpoint.setMethod("POST")
    query_endpoint.setRequestMethod("urlencoded")
    query_endpoint.setQuery(query)
    query_endpoint.query()


def get_cohort_uri(cohort_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}"]


def get_var_uri(cohort_uri: str | URIRef, var_id: str) -> URIRef:
    return URIRef(f"{cohort_uri!s}/{var_id.replace(' ', '_')}")


def get_category_uri(var_uri: str | URIRef, category_id: str) -> URIRef:
    return URIRef(f"{var_uri!s}/category/{category_id}")


# TODO
@router.post(
    "/insert-triples",
    name="Insert triples about a cohort variable in the triplestore",
    response_description="Upload result",
)
def insert_triples(
    cohort_id: str = Form(...),
    var_id: str = Form(...),
    predicate: str = Form(...),
    value: str = Form(...),
    label: str | None = Form(None),
    category_id: str | None = Form(None),
    user: Any = Depends(get_current_user),
) -> None:
    """Insert triples about cohorts variables or variables categories into the triplestore"""
    cohort_uri = get_cohort_uri(cohort_id)
    subject_uri = get_var_uri(cohort_uri, var_id)
    if category_id:
        subject_uri = get_category_uri(subject_uri, category_id)
        # TODO: handle when a category is provided (we add triple to the category instead of the variable)
    delete_existing_triples(cohort_uri, f"<{subject_uri!s}>", predicate)
    label_part = ""
    object_uri = f"<{converter.expand(value)}>"
    if label:
        delete_existing_triples(cohort_uri, f"{object_uri}", "rdfs:label")
        label_part = f'{object_uri} rdfs:label "{label}" .'
    # TODO: some namespaces like Gender are not in the bioregistry
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    INSERT DATA {{
        GRAPH <{cohort_uri!s}> {{ <{subject_uri!s}> {predicate} {object_uri} . {label_part} }}
    }}
    """
    print(query)
    query_endpoint = SPARQLWrapper(f"{settings.sparql_endpoint}/update")
    query_endpoint.setMethod("POST")
    query_endpoint.setRequestMethod("urlencoded")
    query_endpoint.setQuery(query)
    query_endpoint.query()


def parse_categorical_string(s: str) -> list[dict[str, str]]:
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


CURIE_COLUMNS = {"ICD-10": "icd10", "SNOMED-CT": "snomedct", "ATC-DDD": "atc", "LOINC": "loinc"}


def create_uri_from_id(row):
    """Build concepts URIs from the ID provided in the various columns of the data dictionary"""
    uris_list = []
    for column in CURIE_COLUMNS:
        if row[column]:
            if "," in str(row[column]):  # Handle list of IDs separated by comma
                ids = str(row[column]).split(",")
                uris_list.extend([converter.expand(f"{CURIE_COLUMNS[column]}:{identif.strip()}") for identif in ids])
            else:
                uris_list.append(converter.expand(f"{CURIE_COLUMNS[column]}:{str(row[column]).strip()}"))
    return ", ".join(uris_list)


accepted_datatypes = ["STR", "FLOAT", "INT", "DATETIME"]


def load_cohort_dict_file(dict_path: str, cohort_id: str, owner_email: str) -> Dataset:
    """Parse the cohort dictionary uploaded as excel or CSV spreadsheet, and load it to the triplestore"""
    # print(f"Loading dictionary {dict_path}")
    # df = pd.read_csv(dict_path) if dict_path.endswith(".csv") else pd.read_excel(dict_path)
    if not dict_path.endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail="Only CSV files are supported. Please convert your file to CSV and try again.",
        )
    df = pd.read_csv(dict_path)
    df = df.dropna(how="all")
    df = df.fillna("")
    df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)
    df["concept_id"] = df.apply(lambda row: create_uri_from_id(row), axis=1)

    # TODO: add metadata about the cohort, dc:creator
    cohort_uri = get_cohort_uri(cohort_id)
    g = init_graph()
    g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_uri))
    g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_uri))
    g.add((cohort_uri, ICARE["owner"], Literal(owner_email), cohort_uri))

    for i, row in df.iterrows():
        # Check if required columns are present
        if not row["VARIABLE NAME"] or not row["VARIABLE LABEL"] or not row["VAR TYPE"]:
            raise HTTPException(
                status_code=422,
                detail=f"Row {i} is missing required data: variable_name, variable_label, or var_type",
            )
        if row["VAR TYPE"] not in accepted_datatypes:
            raise HTTPException(
                status_code=422,
                detail=f"Row {i} for variable {row['VARIABLE NAME']} is using a wrong datatype: {row['VAR TYPE']}. It should be one of: {', '.join(accepted_datatypes)}",
            )
        # TODO: raise error when duplicate value for VARIABLE LABEL?

        # Create a URI for the variable
        variable_uri = get_var_uri(cohort_uri, row["VARIABLE NAME"])

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
                if (
                    isinstance(value, str)
                    and (value.startswith("http://") or value.startswith("https://"))
                    and " " not in value
                ):
                    g.add((variable_uri, property_uri, URIRef(value), cohort_uri))
                else:
                    g.add((variable_uri, property_uri, Literal(value), cohort_uri))

            # Handle Category
            if column in ["categories"]:
                if len(value) == 1:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Row {i} for variable {row['VARIABLE NAME']} has only one category {row['categories']}. It should have at least two.",
                    )
                for index, category in enumerate(value):
                    cat_uri = get_category_uri(variable_uri, index)
                    g.add((variable_uri, ICARE["categories"], cat_uri, cohort_uri))
                    g.add((cat_uri, RDF.type, ICARE.Category, cohort_uri))
                    g.add((cat_uri, RDF.value, Literal(category["value"]), cohort_uri))
                    g.add((cat_uri, RDFS.label, Literal(category["label"]), cohort_uri))
                    # TODO: add categories
    # print(g.serialize(format="turtle"))
    return g


@router.post(
    "/upload-cohort",
    name="Upload cohort metadata file",
    response_description="Upload result",
)
async def upload_cohort(
    user: Any = Depends(get_current_user),
    # cohort_id: str = Form(..., pattern="^[a-zA-Z0-9-_\w]+$"),
    cohort_id: str = Form(...),
    cohort_dictionary: UploadFile = File(...),
    cohort_data: UploadFile | None = None,
) -> dict[str, str]:
    """Upload a cohort metadata file to the server and add its variables to the triplestore."""
    # Create directory named after cohort_id
    cohorts_folder = os.path.join(settings.data_folder, "cohorts", cohort_id)
    os.makedirs(cohorts_folder, exist_ok=True)
    owner_email = user["email"]
    cohorts_info = retrieve_cohorts_metadata().get(cohort_id)
    # Check if cohort already uploaded
    if len(cohorts_info.variables) > 0:
        authorized_users = [*settings.admins_list, owner_email]
        if user["email"] not in authorized_users:
            raise HTTPException(status_code=403, detail=f"You are not the owner of cohort {cohort_id}.")
        # Make sure we keep the original owner in case an admin edits it
        owner_email = cohorts_info.get("owner")
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

    # Store metadata file on disk in the cohorts folder
    metadata_path = os.path.join(cohorts_folder, filename + ext)
    with open(metadata_path, "wb") as buffer:
        shutil.copyfileobj(cohort_dictionary.file, buffer)

    try:
        g = load_cohort_dict_file(metadata_path, cohort_id, owner_email)
        publish_graph_to_endpoint(g)
    except Exception as e:
        os.remove(metadata_path)
        raise e

    cohorts_dict = retrieve_cohorts_metadata()
    dcr_data = create_provision_dcr(user, cohorts_dict[cohort_id])
    # print(dcr_data)
    # Save data file
    if cohort_data:
        file_path = os.path.join(cohorts_folder, cohort_data.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(cohort_data.file, buffer)
    return dcr_data


def init_triplestore() -> None:
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
        cohort_uri = get_cohort_uri(cohort_id)
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
