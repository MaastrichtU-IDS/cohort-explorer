from SPARQLWrapper import SPARQLWrapper, JSON, POST,TURTLE
from rdflib import Graph, RDF, URIRef, Literal, RDFS, DC
import pandas as pd
from .utils import init_graph, OntologyNamespaces, normalize_text, STUDY_TYPES, save_graph_to_trig_file, graph_exists, extract_age_range
from .config import settings
from rdflib.namespace import XSD
import requests
from .config import settings
# use OBI, BFO and STATO where applicable 

def generate_studies_kg(filepath: str) -> Graph:
    """
    Reads cohort information from a CSV file and adds it to the RDF graph.
    :param filepath: Path to the cohort.csv file
    :return: An rdflib.Graph object with the cohort data
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        # df = df.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))
        df.columns = df.columns.str.lower()
    except UnicodeDecodeError:
        raise ValueError("Failed to read the file -- Upload with Correct CSV format")
    print(df.head(5))
   
    # df = df.fillna("")
    g = init_graph()
    metadata_graph = URIRef(OntologyNamespaces.CMEO.value + "graph/studies_metadata")
    # df.columns = df.columns.str.lower()
    # start from study design execution a process that realized plan which concretizes the study design 
    for _, row in df.iterrows():
        if pd.isna(row["study name"]):
            print("Study name is missing, skipping this row.")
            continue
        study_name = normalize_text(row["study name"])
        study_uri = URIRef(OntologyNamespaces.CMEO.value + normalize_text(row["study name"]))
        study_design_execution_uri = URIRef(study_uri + "/study_design_execution")
        g.add((study_design_execution_uri, RDF.type, OntologyNamespaces.OBI.value.study_design_execution, metadata_graph))
        
        g.add((study_design_execution_uri, DC.identifier, Literal(row["study name"], datatype=XSD.string), metadata_graph))
        
        # plan_uri = URIRef(study_uri + "/plan")
        # g.add((plan_uri, RDF.type, OntologyNamespaces.OBI.value.plan, metadata_graph))
        # g.add((study_design_execution_uri, OntologyNamespaces.OBI.value.realizes, plan_uri, metadata_graph))
      
        study_design_value =  row["study design"].lower().strip() if pd.notna(row["study design"]) else None
       
        # print(f"Study design value: {study_design_value}")
        if study_design_value:
            study_design_value = normalize_text(study_design_value)
            study_design_uri = URIRef(study_uri + "/" + study_design_value)
            g.add((study_design_uri , OntologyNamespaces.CMEO.value.has_value, Literal(study_design_value, datatype=XSD.string), metadata_graph))

            g.add((study_design_uri, DC.identifier, Literal(study_name, datatype=XSD.string), metadata_graph))
            dynamic_class_uri = URIRef(OntologyNamespaces.OBI.value + study_design_value)
            
            g.add((study_design_uri, RDF.type, dynamic_class_uri, metadata_graph))
            protocol_uri = URIRef(study_uri + "/" + study_design_value + "_protocol")
            g.add((study_design_uri,OntologyNamespaces.RO.value.has_part, protocol_uri, metadata_graph))
        else:
              study_design_uri = URIRef(study_uri + "/study_design")
              protocol_uri = URIRef(study_uri + "/protocol") 
              
              g.add((study_design_uri, RDF.type, OntologyNamespaces.OBI.value.study_design, metadata_graph))
              g.add((study_design_uri,OntologyNamespaces.RO.value.has_part, protocol_uri, metadata_graph))
              g.add((study_design_uri, DC.identifier, Literal(study_name, datatype=XSD.string), metadata_graph))
              g.add((study_design_uri , OntologyNamespaces.CMEO.value.has_value, Literal(study_design_value, datatype=XSD.string), metadata_graph))


        g.add((study_design_execution_uri, OntologyNamespaces.RO.value.concretizes, study_design_uri, metadata_graph))
        g.add((study_design_uri, OntologyNamespaces.RO.value.is_concretized_by, study_design_execution_uri, metadata_graph))
        g.add((protocol_uri, RDF.type, OntologyNamespaces.OBI.value.protocol, metadata_graph))
        study_design_variable_specification_uri = URIRef(study_uri + "/study_design_variable_specification")
        g.add((study_design_variable_specification_uri, RDFS.label, Literal("study design variable specification", datatype=XSD.string), metadata_graph))
        g.add((study_design_variable_specification_uri, RDF.type, OntologyNamespaces.CMEO.value.study_design_variable_specification,metadata_graph))
        g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, study_design_variable_specification_uri,metadata_graph))

        if row['language']:
            language = row['language'].lower().strip() if pd.notna(row['language']) else ""
          # its dct:language annotation
            g.add((study_design_execution_uri, DC.language, Literal(language, datatype=XSD.string), metadata_graph))
        if row["dataset format"]:
            data_format = row["dataset format"].lower().strip() if pd.notna(row["dataset format"]) else ""
            data_format_uri = URIRef(study_uri + "/data_format_specification")
            g.add((data_format_uri, RDF.type, OntologyNamespaces.OBI.value.data_format_specification, metadata_graph))
            g.add((data_format_uri, RDFS.label, Literal("data format specification", datatype=XSD.string), metadata_graph))
            g.add((data_format_uri, OntologyNamespaces.CMEO.value.has_value, Literal(data_format, datatype=XSD.string), metadata_graph))
            g.add((study_design_variable_specification_uri, OntologyNamespaces.RO.value.has_part, data_format_uri, metadata_graph))
                        
        if row["study type"]:
            process_type=row["study type"].lower().strip() if pd.notna(row["study type"]) else ""
            g.add((study_design_execution_uri, OntologyNamespaces.CMEO.value.has_value, Literal(process_type, datatype=XSD.string), metadata_graph)) 
            g.add((study_design_uri , RDFS.label, Literal(process_type, datatype=XSD.string), metadata_graph))
        # study design has various "has direct part" which includes primary objective, endpoints, selection criteria, etc.
        if row["study objective"] and pd.notna(row["study objective"]):
            # print(row["Primary objective"])
            po_value = row["study objective"].lower().strip() if pd.notna(row["study objective"]) else ""
            objective_uri = URIRef(study_uri + "/objective_specification") 
            g.add((protocol_uri, RDFS.label, Literal("objective specification", datatype=XSD.string), metadata_graph))
            g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, objective_uri,metadata_graph))
            g.add((objective_uri, RDF.type, OntologyNamespaces.OBI.value.objective_specification,metadata_graph))
            g.add((objective_uri, OntologyNamespaces.CMEO.value.has_value, Literal(po_value, datatype=XSD.string),metadata_graph))

        
      
        if pd.notna(row["institute"]):
            organization_uri = URIRef(study_uri + "/institute")
            g.add((organization_uri, RDF.type, OntologyNamespaces.OBI.value.organization,metadata_graph))
            g.add((study_design_execution_uri, OntologyNamespaces.RO.value.has_participant, organization_uri,metadata_graph))
            g.add((organization_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["institute"], datatype=XSD.string),metadata_graph))
        
            if pd.notna(row["study contact person"]):
                contact_uri = URIRef(study_uri + "/study_contact_person")
                study_contact_person_role_uri = URIRef(study_uri + "/study_contact_person_role")
                
                g.add((contact_uri, RDF.type, OntologyNamespaces.CMEO.value.homo_sapiens,metadata_graph))
                g.add((organization_uri, OntologyNamespaces.OBI.value.has_member, contact_uri,metadata_graph))
                g.add((contact_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["study contact person"], datatype=XSD.string),metadata_graph))
                
                g.add((study_contact_person_role_uri, RDF.type, OntologyNamespaces.OBI.value.study_contact_person_role,metadata_graph))
                g.add((contact_uri, OntologyNamespaces.RO.value.has_role, study_contact_person_role_uri,metadata_graph))
                g.add((study_contact_person_role_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["study contact person"], datatype=XSD.string),metadata_graph))

                if pd.notna(row["study contact person email address"]):
                    email_uri = URIRef(study_contact_person_role_uri + "/email_address")
                    g.add((email_uri, RDF.type, OntologyNamespaces.OBI.value.email_address,metadata_graph))
                    g.add((email_uri, OntologyNamespaces.IAO.value.is_about, contact_uri,metadata_graph))
                    g.add((email_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["study contact person email address"], datatype=XSD.string),metadata_graph))
            
            if pd.notna(row["administrator"]):
                administrator_person_uri = URIRef(study_uri + "/administrator")
                g.add((administrator_person_uri, RDF.type, OntologyNamespaces.CMEO.value.homo_sapiens,metadata_graph))
                g.add((organization_uri, OntologyNamespaces.OBI.value.has_member, contact_uri,metadata_graph))
            
                g.add((contact_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["administrator"], datatype=XSD.string),metadata_graph))
                administrator_role_uri =  URIRef(study_uri + "/administrator_role")
                g.add((administrator_role_uri, RDF.type, OntologyNamespaces.OBI.value.administrator_role,metadata_graph))
                g.add((administrator_person_uri, OntologyNamespaces.RO.value.has_role, administrator_role_uri,metadata_graph))
                g.add((administrator_role_uri, OntologyNamespaces.CMEO.value.has_value,  Literal(row["administrator"], datatype=XSD.string),metadata_graph))
            
                if pd.notna(row["administrator email address"]):
                    email_uri = URIRef(administrator_person_uri + "/email_address")
                    g.add((email_uri, RDF.type, OntologyNamespaces.OBI.value.email_address,metadata_graph))
                    g.add((email_uri, OntologyNamespaces.IAO.value.is_about, administrator_person_uri,metadata_graph))
                    g.add((email_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["administrator"], datatype=XSD.string),metadata_graph))
                    
        if pd.notna(row["number of participants"]):  #number of participants
            try:
                num_participants = int(row["number of participants"])
                num_participants_uri = URIRef(study_uri + "/number_of_study_participants_specification")
                g.add((num_participants_uri, RDF.type, OntologyNamespaces.CMEO.value.number_of_study_participants_specification,metadata_graph))
                g.add((num_participants_uri, RDFS.label, Literal("number of study participants specification", datatype=XSD.string), metadata_graph))
                g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, num_participants_uri,metadata_graph))
                g.add((num_participants_uri, OntologyNamespaces.RO.value.is_part_of, protocol_uri,metadata_graph))
                g.add((num_participants_uri, OntologyNamespaces.CMEO.value.has_value, Literal(num_participants, datatype=XSD.string),metadata_graph))
                
            except ValueError:
                pass
        g = add_study_timing(g, row, study_design_execution_uri, study_uri, metadata_graph)
        g = add_intervention_comparator(g, row, study_uri, protocol_uri, metadata_graph)
        g = add_eligibility_criterion(g, row, study_uri, protocol_uri, metadata_graph)
        g = add_timeline_specification(g, row, study_uri, protocol_uri, metadata_graph)
        g = add_outcome_specification(g, row, study_uri, protocol_uri, metadata_graph)
        
    print(f"Graph size: {len(g)}")
    return g



def add_study_timing(g: Graph, row: pd.Series, study_design_execution_uri: URIRef, study_uri: URIRef, metadata_graph: URIRef) -> None:
    
        if pd.notna(row["start date"]):
            start_date_uri = URIRef(study_design_execution_uri+ "/start_time")
            g.add((start_date_uri, RDF.type, OntologyNamespaces.SIO.value.start_time,metadata_graph))
            g.add((start_date_uri, OntologyNamespaces.IAO.value.has_time_stamp, study_design_execution_uri,metadata_graph))
            g.add((start_date_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["start date"], datatype=XSD.string),metadata_graph))
        
        if pd.notna(row["end date"]): #its study completion date

            study_completion_date_uri = URIRef(study_design_execution_uri + "/end_time")
            g.add((study_completion_date_uri, RDF.type, OntologyNamespaces.SIO.value.end_time, metadata_graph))
            g.add((study_completion_date_uri, OntologyNamespaces.IAO.value.has_time_stamp, study_design_execution_uri,metadata_graph))
            g.add((study_completion_date_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["end date"], datatype=XSD.string),metadata_graph))

        if pd.notna(row["ongoing"]):
            ongoing_status = True if row["ongoing"].lower().strip() == "yes" else False
            ongoing_uri = URIRef(study_uri + "/ongoing")
            g.add((ongoing_uri, RDF.type, OntologyNamespaces.CMEO.value.ongoing,metadata_graph))
            g.add((study_design_execution_uri, OntologyNamespaces.RO.value.has_characteristic, ongoing_uri,metadata_graph))
            g.add((ongoing_uri, OntologyNamespaces.CMEO.value.has_value, Literal(ongoing_status, datatype=XSD.boolean),metadata_graph))
            

        return g


def add_intervention_comparator(g: Graph, row: pd.Series, study_uri: URIRef, protocol_uri: URIRef, metadata_graph: URIRef) -> None:
    if pd.notna(row["interventions"]):
        interventions = row["interventions"].lower().split(";") if pd.notna(row["interventions"]) else []
        for intervention in interventions:
            intervention = intervention.strip()
            if intervention:
                intervention_uri = URIRef(study_uri + "/intervention")
                g.add((intervention_uri, RDF.type, OntologyNamespaces.CMEO.value.intervention_specification, metadata_graph))
                g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, intervention_uri, metadata_graph))
                g.add((intervention_uri, RDFS.label, Literal("intervention specification", datatype=XSD.string), metadata_graph))
                g.add((intervention_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["interventions"], datatype=XSD.string), metadata_graph))
    if pd.notna(row["comparator"]):
        comparators = row["comparator"].lower().split(";") if pd.notna(row["comparator"]) else []
        for comparator in comparators:
            comparator = comparator.strip()
            if comparator:
                comparator_uri = URIRef(study_uri + "/comparator")
                g.add((comparator_uri, RDF.type, OntologyNamespaces.CMEO.value.comparator_specification, metadata_graph))
                g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, comparator_uri, metadata_graph))
                g.add((comparator_uri, RDFS.label, Literal("comparators specification", datatype=XSD.string), metadata_graph))
                g.add((comparator_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["comparator"], datatype=XSD.string), metadata_graph))
    return g


def add_eligibility_criterion(g: Graph, row: pd.Series, study_uri: URIRef, protocol_uri: URIRef, metadata_graph: URIRef) -> None:
    
    eligibility_criterion_uri = URIRef(study_uri + "/eligibility_criterion")
    g.add((eligibility_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.eligibility_criterion, metadata_graph))
    g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, eligibility_criterion_uri, metadata_graph))
    
    human_subject_enrollement = URIRef(study_uri + "/human_subject_enrollment")
    g.add((human_subject_enrollement, RDF.type, OntologyNamespaces.CMEO.value.human_subject_enrollment, metadata_graph))
    g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.is_concretized_by, human_subject_enrollement, metadata_graph))
    
    input_population_uri = URIRef(study_uri + "/input_population")
    g.add((input_population_uri, RDF.type, OntologyNamespaces.OBI.value.population, metadata_graph))
    g.add((human_subject_enrollement, OntologyNamespaces.RO.value.has_input, input_population_uri, metadata_graph))
    
    output_population_uri = URIRef(study_uri + "/output_population")
    g.add((output_population_uri, RDF.type, OntologyNamespaces.OBI.value.population, metadata_graph))
    g.add((human_subject_enrollement, OntologyNamespaces.RO.value.has_output, output_population_uri, metadata_graph))
    
    # add morbidity quality of output population
    morbidities = row["morbidity"].lower().split(";") if pd.notna(row["morbidity"]) else []
    for morbidity in morbidities:
        morbidity = morbidity.strip()
        if morbidity:
            dynamic_morbidity_uri = URIRef(OntologyNamespaces.OBI.value + normalize_text(morbidity))
            g.add((output_population_uri, OntologyNamespaces.RO.value.has_characteristic, dynamic_morbidity_uri, metadata_graph))
            g.add((dynamic_morbidity_uri, RDF.type, OntologyNamespaces.OBI.value.morbidity, metadata_graph))
            g.add((dynamic_morbidity_uri, RDFS.label, Literal(morbidity, datatype=XSD.string), metadata_graph)) 
            g.add((dynamic_morbidity_uri, OntologyNamespaces.CMEO.value.has_value, Literal(morbidity, datatype=XSD.string), metadata_graph))    
    # add mixed sex quality of output population
    mixed_sex_quality = URIRef(study_uri + "/mixed_sex")
    g.add(( mixed_sex_quality, RDF.type, OntologyNamespaces.OBI.value.mixed_sex, metadata_graph))
    g.add((output_population_uri, OntologyNamespaces.RO.value.has_characteristic, mixed_sex_quality, metadata_graph))
    g.add((mixed_sex_quality,RDFS.label, Literal("mixed sex", datatype=XSD.string), metadata_graph))
    g.add((mixed_sex_quality, OntologyNamespaces.CMEO.value.has_value, Literal(row["mixed sex"], datatype=XSD.string), metadata_graph))

    

    # add age quality of output population
    age_quality = URIRef(study_uri + "/age_distribution")
    g.add((age_quality, RDF.type, OntologyNamespaces.OBI.value.age_distribution, metadata_graph))   
    g.add((output_population_uri, OntologyNamespaces.RO.value.has_characteristic, age_quality, metadata_graph))
    g.add((age_quality, RDFS.label, Literal("age distribution", datatype=XSD.string), metadata_graph))
    g.add((age_quality, OntologyNamespaces.CMEO.value.has_value, Literal(row["age distribution"], datatype=XSD.string), metadata_graph))
    
    g = add_inclusion_criterion(g, row, study_uri, eligibility_criterion_uri, metadata_graph)
    g = add_exclusion_criterion(g, row, study_uri, eligibility_criterion_uri, metadata_graph)
    
    site_uri = URIRef(output_population_uri + "/site")
    g.add((site_uri, RDF.type, OntologyNamespaces.BFO.value.site, metadata_graph))
    g.add((site_uri, OntologyNamespaces.IAO.value.is_about, output_population_uri, metadata_graph))
    g.add((site_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row["population location"], datatype=XSD.string), metadata_graph))
    return g
 
def add_timeline_specification(g: Graph, row: pd.Series, study_uri: URIRef, protocol_uri: URIRef, metadata_graph: URIRef) -> None:
    
    timeline_specification_uri = URIRef(study_uri + "/timeline_specification")

    g.add((timeline_specification_uri, RDF.type, OntologyNamespaces.CMEO.value.timeline_specification, metadata_graph))
    g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, timeline_specification_uri, metadata_graph))
    g.add((timeline_specification_uri, RDFS.label, Literal("frequency of data collection", datatype=XSD.string), metadata_graph))
    return g
    
def add_inclusion_criterion(g: Graph, row: pd.Series, study_uri: URIRef, eligibility_criterion_uri: URIRef, metadata_graph: URIRef) -> None:
    
    # inclusion crieria are many and all are subclass of eligibility_criterion_uri criteria
    
    # find all columns where inclusion criterion substring in name is present
    inclusion_criterion_uri = URIRef(study_uri + "/inclusion_criterion")
    g.add((inclusion_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.inclusion_criterion, metadata_graph))
    g.add((inclusion_criterion_uri, OntologyNamespaces.RO.value.part_of, eligibility_criterion_uri, metadata_graph))
    g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.has_part, inclusion_criterion_uri, metadata_graph))
    inclusion_criteria_columns = [col for col in row.index if "inclusion criterion" in col.lower()]
    for col in inclusion_criteria_columns:
        inclusion_criterion_name = normalize_text(col)
        if "age" in inclusion_criterion_name:
            add_age_group_inclusion_criterion(g, study_uri, inclusion_criterion_uri, metadata_graph, row[col])
        else:
            dynamic_inclusion_criterion_type = URIRef(OntologyNamespaces.OBI.value + inclusion_criterion_name)
            inclusion_criteria_value = row[col].lower().strip() if pd.notna(row[col]) else ""
            inc_all_values = inclusion_criteria_value.split(";") if pd.notna(row[col]) else ""
            for inclusion_criteria_value in inc_all_values:
                inclusion_criteria_value = inclusion_criteria_value.strip()
                col_inclusion_criteria_uri = URIRef(study_uri + "/" + inclusion_criterion_name)
                g.add((col_inclusion_criteria_uri, RDF.type, dynamic_inclusion_criterion_type, metadata_graph))
                g.add((col_inclusion_criteria_uri, OntologyNamespaces.RO.value.part_of, inclusion_criterion_uri, metadata_graph))
                g.add((inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, col_inclusion_criteria_uri, metadata_graph))
                g.add((col_inclusion_criteria_uri, RDFS.label, Literal(inclusion_criterion_name, datatype=XSD.string), metadata_graph)) 
                g.add((col_inclusion_criteria_uri, OntologyNamespaces.CMEO.value.has_value, Literal(inclusion_criteria_value, datatype=XSD.string), metadata_graph))
                
    return g



def add_age_group_inclusion_criterion(g: Graph, study_uri: URIRef, inclusion_criterion_uri: URIRef, metadata_graph: URIRef, inclusion_criteria_value: str) -> None:

    if pd.isna(inclusion_criteria_value):
        return g
    
    age_group_inclusion_criterion_uri =  URIRef(study_uri + "/age_group_inclusion_criterion")
    # print(f"Age group inclusion criterion URI: {age_group_inclusion_criterion_uri}")    
    g.add((age_group_inclusion_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.age_group_inclusion_criterion, metadata_graph))
    g.add((inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, age_group_inclusion_criterion_uri, metadata_graph))
    g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.RO.value.part_of, inclusion_criterion_uri, metadata_graph))
    g.add((age_group_inclusion_criterion_uri, RDFS.label, Literal("age group inclusion criterion", datatype=XSD.string), metadata_graph))
    g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.CMEO.value.has_value, Literal(inclusion_criteria_value, datatype=XSD.string), metadata_graph))
    
    agic_value_ranges = extract_age_range(inclusion_criteria_value)
    print(f"Age group inclusion criterion value ranges: {agic_value_ranges}")
    if agic_value_ranges:
        min_age, max_age = agic_value_ranges
        if min_age is not None:
            min_age = float(min_age)
           
        # print(f"Age group inclusion criterion value ranges: {min_age}, {max_age}")  
            min_age_value_specification =  URIRef(age_group_inclusion_criterion_uri + "/minimum_age_value_specification")
            g.add((min_age_value_specification, RDF.type, OntologyNamespaces.OBI.value.minimum_age_value_specification, metadata_graph))
            g.add((min_age_value_specification, OntologyNamespaces.CMEO.value.has_value, Literal(min_age, datatype=XSD.float), metadata_graph))
            g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, min_age_value_specification, metadata_graph))

        if max_age is not None:
            max_age = float(max_age)
            max_age_value_specification =  URIRef(age_group_inclusion_criterion_uri + "/maximum_age_value_specification")
            
            g.add((max_age_value_specification, RDF.type, OntologyNamespaces.OBI.value.maximum_age_value_specification, metadata_graph))
            g.add((age_group_inclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, max_age_value_specification, metadata_graph))
            g.add((max_age_value_specification, OntologyNamespaces.CMEO.value.has_value, Literal(max_age, datatype=XSD.float), metadata_graph))
        
    return g


def add_exclusion_criterion(g: Graph, row: pd.Series, study_uri: URIRef, eligibility_criterion_uri: URIRef, metadata_graph: URIRef) -> None:
    
    # inclusion crieria are many and all are subclass of eligibility_criterion_uri criteria
    
    # find all columns where inclusion criterion substring in name is present
    exclusion_criterion_uri = URIRef(study_uri + "/exclusion_criterion")
    g.add((exclusion_criterion_uri, RDF.type, OntologyNamespaces.OBI.value.exclusion_criterion, metadata_graph))
    g.add((exclusion_criterion_uri, OntologyNamespaces.RO.value.part_of, eligibility_criterion_uri, metadata_graph))
    g.add((eligibility_criterion_uri, OntologyNamespaces.RO.value.has_part, exclusion_criterion_uri, metadata_graph))
    exclusion_criteria_columns = [col for col in row.index if "exclusion criterion" in col.lower()]
    for col in exclusion_criteria_columns:
        exclusion_criterion_name = normalize_text(col)
        dynamic_exclusion_criterion_type = URIRef(OntologyNamespaces.OBI.value + "/" + exclusion_criterion_name)
        ec_all_values = row[col].lower().split(";") if pd.notna(row[col]) else ""
        for exclusion_criteria_value in ec_all_values:
            exclusion_criteria_value = exclusion_criteria_value.strip()
            col_exclusion_criteria_uri = URIRef(study_uri + "/" + exclusion_criterion_name)
            g.add((col_exclusion_criteria_uri, RDF.type, dynamic_exclusion_criterion_type, metadata_graph))
            g.add((col_exclusion_criteria_uri, OntologyNamespaces.RO.value.part_of, exclusion_criterion_uri, metadata_graph))
            g.add((exclusion_criterion_uri, OntologyNamespaces.RO.value.has_part, col_exclusion_criteria_uri, metadata_graph))
            g.add((col_exclusion_criteria_uri, RDFS.label, Literal(exclusion_criterion_name, datatype=XSD.string), metadata_graph)) 
            g.add((col_exclusion_criteria_uri, OntologyNamespaces.CMEO.value.has_value, Literal(exclusion_criteria_value, datatype=XSD.string), metadata_graph))
            
    return g
    
def add_outcome_specification(g: Graph, row: pd.Series, study_uri: URIRef, protocol_uri: URIRef, metadata_graph: URIRef) -> None:
    outcome_specification_uri =  URIRef(study_uri + "/outcome_specification")
    g.add((outcome_specification_uri, RDF.type, OntologyNamespaces.CMEO.value.outcome_specification,metadata_graph))
    g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, outcome_specification_uri,metadata_graph))
    g.add((outcome_specification_uri, RDFS.label, Literal("outcome specification", datatype=XSD.string), metadata_graph))
    
        
    if row["primary outcome specification"]:
            pendpoint_values= row["primary outcome specification"].lower().split(';') if pd.notna(row["primary outcome specification"]) else ""
            for pendpoint_value in pendpoint_values:
                pendpoint_value = pendpoint_value.strip()
                primary_outcome_uri = URIRef(study_uri + "/primary_outcome_specification")
                g.add((primary_outcome_uri, RDFS.label, Literal("primary outcome specification", datatype=XSD.string), metadata_graph))
                g.add((primary_outcome_uri, RDF.type, OntologyNamespaces.CMEO.value.primary_outcome_specification,metadata_graph))
                g.add((outcome_specification_uri, OntologyNamespaces.RO.value.has_part, primary_outcome_uri,metadata_graph))
                g.add((primary_outcome_uri, OntologyNamespaces.CMEO.value.has_value, Literal(pendpoint_value, datatype=XSD.string),metadata_graph))

    if row["secondary outcome specification"]:
            secendpoint_values= row["secondary outcome specification"].lower().split(";") if pd.notna(row["secondary outcome specification"]) else ""
            for secendpoint_value in secendpoint_values:
                secendpoint_value = secendpoint_value.strip()
                secondary_outcome_uri = URIRef(study_uri + "/secondary_outcome_specification")
                g.add((secondary_outcome_uri, RDF.type, OntologyNamespaces.CMEO.value.secondary_outcome_specification,metadata_graph))
                g.add((primary_outcome_uri, RDFS.label, Literal("secondary outcome specification", datatype=XSD.string), metadata_graph))
                g.add((outcome_specification_uri, OntologyNamespaces.RO.value.has_part, secondary_outcome_uri,metadata_graph))
                g.add((secondary_outcome_uri, OntologyNamespaces.CMEO.value.has_value, Literal(secendpoint_value, datatype=XSD.string),metadata_graph))
    return g


def update_metadata_graph(endpoint_url, cohort_uri, variable_uris, metadata_graph_path):
    """Insert metadata graph data into the Oxigraph triplestore using SPARQL Update."""
    
    # Define the named graph URI
    graph_uri = f"https://w3id.org/CMEO/graph/studies_metadata"
    print(f"📌 Graph URI: {graph_uri}")

    study_variable_design_specification_uri = URIRef(cohort_uri + "/study_design_variable_specification")
    
    if not graph_exists(graph_uri):
        print("⚠️ Metadata graph does not exist, skipping update.")
        return None
    
    # Construct SPARQL `INSERT DATA` query
    inserts = "\n".join([f"<{study_variable_design_specification_uri}> <http://purl.obolibrary.org/obo/ro.owl/has_part> <{var}> ." for var in variable_uris])

    query = f"""
        INSERT DATA {{
            GRAPH <{graph_uri}> {{
                {inserts}
            }}
        }}
    """
    
   # print(f"📌 SPARQL Update Query:\n{query}")

    # Send the SPARQL Update request to Oxigraph
    headers = {"Content-Type": "application/sparql-update"}
    response = requests.post(endpoint_url, headers=headers, data=query)

    # Handle response
    if response.status_code in (200, 201, 204):
        print(f"✅ Successfully updated metadata graph: {graph_uri}")
        reconstruct_metadata_graph(endpoint_url=endpoint_url, graph_uri=graph_uri, metadata_graph_path=metadata_graph_path)
    else:
        print(f"❌ Failed to update metadata graph: {response.status_code}, {response.text}")
        return None

def reconstruct_metadata_graph(endpoint_url, graph_uri, metadata_graph_path) -> None:
    """
    Query the entire contents of a named graph from a SPARQL endpoint using a CONSTRUCT query.
    
    :param endpoint_url: URL of the SPARQL endpoint
    :param graph_uri: URI of the named graph to query
    :return: RDF data in TRiG format as a string, or None if an error occurs
    """
    # Initialize SPARQLWrapper with the endpoint URL
    sparql = SPARQLWrapper(settings.query_endpoint)
    # Prepare the CONSTRUCT query to retrieve all triples from the specified graph
    construct_query = f"""
        CONSTRUCT {{
            ?s ?p ?o .
        }}
        WHERE {{
            GRAPH <{graph_uri}> {{
                ?s ?p ?o .
            }}
        }}
    """
    sparql.setQuery(construct_query)
    
    # Request a specific return format; while TRiG is ideal, some endpoints may not support it directly.
    # If direct TRiG support is not available, you can retrieve in Turtle and convert externally.
    # For this example, we'll set the return format to Turtle and handle accordingly.
    sparql.setReturnFormat(TURTLE)
    
    try:
        # Execute the query and retrieve the result
        turtle_data_bytes = sparql.query().convert()
        g = init_graph()
        turtle_data = turtle_data_bytes.decode('utf-8') if isinstance(turtle_data_bytes, bytes) else turtle_data_bytes
        g.parse(data=turtle_data, format="turtle")
        # result_data is in RDFLib Graph format when using TURTLE return format
        save_graph_to_trig_file(g, metadata_graph_path)
    except Exception as e:
        print(f"Error querying the graph: {e}") 

# def reconstruct_metadata_graph(graph_uri, metadata_graph_path):
#     """
#     Retrieves the entire contents of a named graph from GraphDB using the Graph Store Protocol.
#     Saves the result as a TRiG file.

#     :param repository_id: GraphDB repository ID
#     :param graph_uri: URI of the named graph to query
#     :param metadata_graph_path: Path to save the exported graph in TriG format
#     """
    
#     print(f"Retrieving named graph from: {graph_uri}")

#     headers = {"Accept": "application/trig"}  # Request TriG format

#     try:
#         # Send GET request to retrieve the graph
#         response = requests.get(graph_uri, headers=headers, timeout=300)

#         if response.status_code == 200:
#             print(f"Successfully retrieved RDF data from {graph_uri}")

#             # Parse the RDF data using rdflib
#             g = rdflib.Graph()
#             g.parse(data=response.text, format="trig")

#             # Save the RDF graph as a .trig file
#             with open(metadata_graph_path, "w", encoding="utf-8") as trig_file:
#                 trig_file.write(response.text)

#             print(f"Saved RDF data to {metadata_graph_path}")
#             return True
#         else:
#             print(f"Failed to retrieve graph: {response.status_code}, {response.text}")
#             return False

#     except Exception as e:
#         print(f"Error querying the graph: {e}")
#         return False

