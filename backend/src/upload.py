import glob
import os
import shutil
from datetime import datetime
from re import sub
from typing import Any

import pandas as pd
import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from rdflib import XSD, Dataset, Graph, Literal, URIRef
from rdflib.namespace import DC, RDF, RDFS
from SPARQLWrapper import SPARQLWrapper

from src.auth import get_current_user
from src.config import settings
from src.decentriq import create_provision_dcr
from src.mapping_generation.retriever import map_csv_to_standard_codes
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


def delete_existing_triples(graph_uri: str | URIRef, subject="?s", predicate="?p"):
    """Function to delete existing triples in a cohort's graph"""
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    DELETE WHERE {{
        GRAPH <{graph_uri!s}> {{ {subject} {predicate} ?o . }}
    }}
    """
    query_endpoint = SPARQLWrapper(settings.update_endpoint)
    query_endpoint.setMethod("POST")
    query_endpoint.setRequestMethod("urlencoded")
    query_endpoint.setQuery(query)
    query_endpoint.query()


def get_cohort_uri(cohort_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}"]


def get_cohort_mapping_uri(cohort_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}/mappings"]


def get_var_uri(cohort_id: str | URIRef, var_id: str) -> URIRef:
    return ICARE[f"cohort/{cohort_id.replace(' ', '_')}/{var_id.replace(' ', '_')}"]


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
    """Insert triples about mappings for cohorts variables or variables categories into the triplestore"""
    cohort_info = retrieve_cohorts_metadata(user["email"]).get(cohort_id)
    if not cohort_info:
        raise HTTPException(
            status_code=403,
            detail=f"Cohort ID {cohort_id} does not exists",
        )
    if not cohort_info.can_edit:
        raise HTTPException(
            status_code=403,
            detail=f"User {user['email']} cannot edit cohort {cohort_id}",
        )
    graph_uri = get_cohort_mapping_uri(cohort_id)
    subject_uri = get_var_uri(cohort_id, var_id)
    if category_id:
        subject_uri = get_category_uri(subject_uri, category_id)
        # TODO: handle when a category is provided (we add triple to the category instead of the variable)
    delete_existing_triples(graph_uri, f"<{subject_uri!s}>", predicate)
    label_part = ""
    object_uri = f"<{converter.expand(value)}>"
    if label:
        delete_existing_triples(graph_uri, f"{object_uri}", "rdfs:label")
        label_part = f'{object_uri} rdfs:label "{label}" .'
    # TODO: some namespaces like Gender are not in the bioregistry
    query = f"""
    PREFIX icare: <https://w3id.org/icare4cvd/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    INSERT DATA {{
        GRAPH <{graph_uri!s}> {{ <{subject_uri!s}> {predicate} {object_uri} . {label_part} }}
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


COLUMNS_LIST = [
    "VARIABLE NAME",
    "VARIABLE LABEL",
    "VAR TYPE",
    "UNITS",
    "CATEGORICAL",
    "COUNT",
    "NA",
    "MIN",
    "MAX",
    "Definition",
    "Formula",
    "OMOP",
    "Visits",
]
ACCEPTED_DATATYPES = ["STR", "FLOAT", "INT", "DATETIME"]
# Old multiple column format for concept codes:
ID_COLUMNS_NAMESPACES = {"ICD-10": "icd10", "SNOMED-CT": "snomedct", "ATC-DDD": "atc", "LOINC": "loinc"}


def get_id_from_multi_columns(row):
    """Old format to pass concepts URIs from the ID provided in the various columns of the data dictionary (ICD-10, ATC, SNOMED...)"""
    uris_list = []

    for column in ID_COLUMNS_NAMESPACES:
        if row[column]:
            if "," in str(row[column]):  # Handle list of IDs separated by comma
                ids = str(row[column]).split(",")
                uris_list.extend(
                    [converter.expand(f"{ID_COLUMNS_NAMESPACES[column]}:{identif.strip()}") for identif in ids]
                )
            else:
                uris_list.append(converter.expand(f"{ID_COLUMNS_NAMESPACES[column]}:{str(row[column]).strip()}"))
    return ", ".join(uris_list)


def to_camelcase(s: str) -> str:
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])


def load_cohort_dict_file(dict_path: str, cohort_id: str) -> Dataset:
    """Parse the cohort dictionary uploaded as excel or CSV spreadsheet, and load it to the triplestore"""
    # print(f"Loading dictionary {dict_path}")
    # df = pd.read_csv(dict_path) if dict_path.endswith(".csv") else pd.read_excel(dict_path)
    if not dict_path.endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail="Only CSV files are supported. Please convert your file to CSV and try again.",
        )
    try:
        # Record all errors and raise them at the end
        errors = []
        df = pd.read_csv(dict_path)
        df = df.dropna(how="all")
        df = df.fillna("")
        df.columns = df.columns.str.strip()
        for column in COLUMNS_LIST:
            if column not in df.columns.values.tolist():
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing column `{column}`",
                )
        df["categories"] = df["CATEGORICAL"].apply(parse_categorical_string)

        # TODO: handle columns from Komal that maps variables:
        if "Label Concept Code" in df.columns:
            df["concept_id"] = df.apply(lambda row: str(row["Label Concept Code"]).strip(), axis=1)
        else:
            # Try to get IDs from old format multiple columns
            df["concept_id"] = df.apply(lambda row: get_id_from_multi_columns(row), axis=1)

        cohort_uri = get_cohort_uri(cohort_id)
        g = init_graph()
        g.add((cohort_uri, RDF.type, ICARE.Cohort, cohort_uri))
        g.add((cohort_uri, DC.identifier, Literal(cohort_id), cohort_uri))

        for i, row in df.iterrows():
            # Check if required columns are present
            if not row["VARIABLE NAME"] or not row["VARIABLE LABEL"] or not row["VAR TYPE"] or not row["COUNT"]:
                errors.append(f"Row {i+2} is missing required data: VARIABLE NAME, VARIABLE LABEL, VAR TYPE, or COUNT")
            if row["VAR TYPE"] not in ACCEPTED_DATATYPES:
                errors.append(
                    f"Row {i+2} for variable `{row['VARIABLE NAME']}` is using a wrong datatype: `{row['VAR TYPE']}`. It should be one of: {', '.join(ACCEPTED_DATATYPES)}"
                )
            # TODO: raise error when duplicate value for VARIABLE LABEL?

            # Create a URI for the variable
            variable_uri = get_var_uri(cohort_id, row["VARIABLE NAME"])
            g.add((cohort_uri, ICARE.hasVariable, variable_uri, cohort_uri))

            # Add the type of the resource
            g.add((variable_uri, RDF.type, ICARE.Variable, cohort_uri))
            g.add((variable_uri, DC.identifier, Literal(row["VARIABLE NAME"]), cohort_uri))
            g.add((variable_uri, RDFS.label, Literal(row["VARIABLE LABEL"]), cohort_uri))
            g.add((variable_uri, ICARE["index"], Literal(i, datatype=XSD.integer), cohort_uri))

            # Get categories code if provided
            categories_codes = []
            if row.get("Categorical Value Concept Code"):
                categories_codes = row["Categorical Value Concept Code"].split(",")
            for column, col_value in row.items():
                if column not in ["categories"] and col_value:
                    # NOTE: we literally use the column name as the property URI in camelcase (that's what I call lazy loading!)
                    property_uri = ICARE[to_camelcase(column)]
                    if (
                        isinstance(col_value, str)
                        and (col_value.startswith("http://") or col_value.startswith("https://"))
                        and " " not in col_value
                    ):
                        g.add((variable_uri, property_uri, URIRef(col_value), cohort_uri))
                    else:
                        g.add((variable_uri, property_uri, Literal(col_value), cohort_uri))

                # Handle Category
                if column in ["categories"]:
                    if len(col_value) == 1:
                        errors.append(
                            f"Row {i+2} for variable `{row['VARIABLE NAME']}` has only one category `{row['categories'][0]['value']}`. It should have at least two."
                        )
                        continue
                    for index, category in enumerate(col_value):
                        cat_uri = get_category_uri(variable_uri, index)
                        g.add((variable_uri, ICARE.categories, cat_uri, cohort_uri))
                        g.add((cat_uri, RDF.type, ICARE.VariableCategory, cohort_uri))
                        g.add((cat_uri, RDF.value, Literal(category["value"]), cohort_uri))
                        g.add((cat_uri, RDFS.label, Literal(category["label"]), cohort_uri))
                        try:
                            if categories_codes and str(categories_codes[index]).strip() != "na":
                                cat_code_uri = converter.expand(str(categories_codes[index]).strip())
                                if not cat_code_uri:
                                    errors.append(
                                        f"Row {i+2} for variable `{row['VARIABLE NAME']}` the category concept code provided for `{categories_codes[index]}` is not valid. Use one of snomedct:, icd10:, atc: or loinc: prefixes."
                                    )
                                else:
                                    g.add((cat_uri, ICARE.conceptId, URIRef(cat_code_uri), cohort_uri))
                        except Exception:
                            errors.append(
                                f"Row {i+2} for variable `{row['VARIABLE NAME']}` the {len(categories_codes)} category concept codes are not matching with {len(row['categories'])} categories provided."
                            )
        # print(g.serialize(format="turtle"))
        # Print all errors at once
        if len(errors) > 0:
            raise HTTPException(
                status_code=422,
                detail="\n\n".join(errors),
            )
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)[5:],
            # detail=str(e),
        )
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
    airlock: bool = True,
) -> dict[str, Any]:
    """Upload a cohort metadata file to the server and add its variables to the triplestore."""
    user_email = user["email"]
    cohort_info = retrieve_cohorts_metadata(user_email).get(cohort_id)
    # cohorts = retrieve_cohorts_metadata(user_email)
    if not cohort_info:
        raise HTTPException(
            status_code=403,
            detail=f"Cohort ID {cohort_id} does not exists",
        )
    if not cohort_info.can_edit:
        raise HTTPException(
            status_code=403,
            detail=f"User {user_email} cannot edit cohort {cohort_id}",
        )

    cohort_info.airlock = airlock

    # Create directory named after cohort_id
    cohorts_folder = os.path.join(settings.data_folder, "cohorts", cohort_id)
    os.makedirs(cohorts_folder, exist_ok=True)
    # Check if cohort already uploaded
    if cohort_info and len(cohort_info.variables) > 0:
        # Check for existing data dictionary file and back it up
        for file_name in os.listdir(cohorts_folder):
            if file_name.endswith("_datadictionary.csv"):
                # Construct the backup file name with the current date
                backup_file_name = f"{file_name.rsplit('.', 1)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backup_file_path = os.path.join(cohorts_folder, backup_file_name)
                existing_file_path = os.path.join(cohorts_folder, file_name)
                # Rename (backup) the existing file
                os.rename(existing_file_path, backup_file_path)
                break  # Assuming there's only one data dictionary file per cohort

    # Make sure metadata file ends with _datadictionary
    metadata_filename = cohort_dictionary.filename
    filename, ext = os.path.splitext(metadata_filename)
    if not filename.endswith("_datadictionary"):
        filename += "_datadictionary"

    # Store metadata file on disk in the cohorts folder
    metadata_path = os.path.join(cohorts_folder, filename + ext)
    with open(metadata_path, "wb") as buffer:
        shutil.copyfileobj(cohort_dictionary.file, buffer)

    # TODO: KOMAL add function that reads the CSV at `metadata_path`, add new columns, and save it back to the same file
    map_csv_to_standard_codes(metadata_path)
    try:
        g = load_cohort_dict_file(metadata_path, cohort_id)
        # Airlock preview setting goes to mapping graph because it is defined in the explorer UI
        g.add(
            (
                get_cohort_uri(cohort_id),
                ICARE.previewEnabled,
                Literal(str(airlock).lower(), datatype=XSD.boolean),
                get_cohort_mapping_uri(cohort_id),
            )
        )
        # Delete previous graph for this file from triplestore
        delete_existing_triples(
            get_cohort_mapping_uri(cohort_id), f"<{get_cohort_uri(cohort_id)!s}>", "icare:previewEnabled"
        )
        delete_existing_triples(get_cohort_uri(cohort_id))
        publish_graph_to_endpoint(g)
    except Exception as e:
        os.remove(metadata_path)
        raise e

    cohorts_dict = retrieve_cohorts_metadata(user["email"])
    try:
        dcr_data = create_provision_dcr(user, cohorts_dict.get(cohort_id))
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"The cohort was properly created in the Cohort Explorer, but there was an issue when uploading to Decentriq: {e}",
        )
    # print(dcr_data)
    # Save data file
    if cohort_data:
        file_path = os.path.join(cohorts_folder, cohort_data.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(cohort_data.file, buffer)
    return dcr_data


COHORTS_METADATA_FILEPATH = os.path.join(settings.data_folder, "iCARE4CVD_Cohorts.xlsx")


@router.post(
    "/upload-cohorts-metadata",
    name="Upload metadata file for all cohorts",
    response_description="Upload result",
)
async def upload_cohorts_metadata(
    user: Any = Depends(get_current_user),
    cohorts_metadata: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload the file with all cohorts metadata to the server and triplestore."""
    if user["email"] not in settings.admins_list:
        raise HTTPException(status_code=403, detail="You need to be admin to perform this action.")
    with open(COHORTS_METADATA_FILEPATH, "wb") as buffer:
        shutil.copyfileobj(cohorts_metadata.file, buffer)
    g = cohorts_metadata_file_to_graph(COHORTS_METADATA_FILEPATH)
    if len(g) > 0:
        delete_existing_triples(ICARE["graph/metadata"])
        publish_graph_to_endpoint(g)


def cohorts_metadata_file_to_graph(filepath: str) -> Dataset:
    df = pd.read_excel(filepath, sheet_name="Descriptions")
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
            for email in row["Email"].split(";"):
                g.add((cohort_uri, ICARE.email, Literal(email.strip()), cohorts_graph))
        if row["Type"]:
            g.add((cohort_uri, ICARE.cohortType, Literal(row["Type"]), cohorts_graph))
        if row["Study type"]:
            g.add((cohort_uri, ICARE.studyType, Literal(row["Study type"]), cohorts_graph))
        if row["N"]:
            g.add((cohort_uri, ICARE.studyParticipants, Literal(row["N"]), cohorts_graph))
        if row["Study duration"]:
            g.add((cohort_uri, ICARE.studyDuration, Literal(row["Study duration"]), cohorts_graph))
        if row["Ongoing"]:
            g.add((cohort_uri, ICARE.studyOngoing, Literal(row["Ongoing"]), cohorts_graph))
        if row["Patient population"]:
            g.add((cohort_uri, ICARE.studyPopulation, Literal(row["Patient population"]), cohorts_graph))
        if row["Primary objective"]:
            g.add((cohort_uri, ICARE.studyObjective, Literal(row["Primary objective"]), cohorts_graph))
    return g


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
    # ntriple_g.parse("https://raw.githubusercontent.com/vemonet/omop-cdm-owl/main/omop_cdm_v6.ttl", format="turtle")
    ntriple_g.parse(
        "https://raw.githubusercontent.com/MaastrichtU-IDS/cohort-explorer/main/cohort-explorer-ontology.ttl",
        format="turtle",
    )
    # Trick to convert ntriples to nquads with a given graph
    for s, p, o in ntriple_g.triples((None, None, None)):
        g.add((s, p, o, onto_graph_uri))
    # print(g.serialize(format="trig"))
    if publish_graph_to_endpoint(g):
        print(f"🦉 Triplestore initialization: added {len(g)} triples for the iCARE4CVD Cohort Explorer OWL ontology.")

    os.makedirs(os.path.join(settings.data_folder, "cohorts"), exist_ok=True)
    # Load cohorts data dictionaries already present in data/cohorts/
    for folder in os.listdir(os.path.join(settings.data_folder, "cohorts")):
        folder_path = os.path.join(settings.data_folder, "cohorts", folder)
        if os.path.isdir(folder_path):
            for file in glob.glob(os.path.join(folder_path, "*_datadictionary.*")):
                # NOTE: default airlock preview to false if we ever need to reset cohorts,
                # admins can easily ddl and reupload the cohorts with the correct airlock value
                g = load_cohort_dict_file(file, folder)
                g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.trig", format="trig")
                if publish_graph_to_endpoint(g):
                    print(f"💾 Triplestore initialization: added {len(g)} triples for cohorts {file}.")

    # Load cohorts metadata
    g = cohorts_metadata_file_to_graph(COHORTS_METADATA_FILEPATH)

    # g.serialize(f"{settings.data_folder}/cohort_explorer_triplestore.ttl", format="turtle")
    if publish_graph_to_endpoint(g):
        print(f"🪪 Triplestore initialization: added {len(g)} triples for the cohorts metadata.")
