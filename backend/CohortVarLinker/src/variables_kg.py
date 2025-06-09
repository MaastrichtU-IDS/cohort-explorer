# from rdflib import Dataset, Namespace
from rdflib.namespace import  XSD
from rdflib import Graph,Literal, RDF, RDFS, URIRef, DC
# from SPARQLWrapper import SPARQLWrapper, JSON, POST,TURTLE
from urllib.parse import quote
import pandas as pd
# import chardet
from .config import settings
import os
import json
# import os
from .ontology_model import  Concept
from .study_kg import update_metadata_graph
from .utils import (

    OntologyNamespaces,
    init_graph, 
    is_categorical_variable, 
    normalize_text, safe_int, determine_var_uri,
    variable_exists,
    get_study_uri,
    load_dictionary,
    extract_tick_values
)
from typing import Optional, Any

# def connect_dataelement_to_protocol(g: Graph, cohort_uri: URIRef, variable_uri: URIRef) -> Graph:
                                    
#     protocol_uri = URIRef(f"{cohort_uri}/protocol")
#     study_design_variable_specification_uri = URIRef(f"{cohort_uri}/study_design_variable_specification")
#     g.add((protocol_uri, OntologyNamespaces.RO.value.has_part, study_design_variable_specification_uri, cohort_uri))
#     g.add((study_design_variable_specification_uri, OntologyNamespaces.RO.value.has_part, variable_uri, cohort_uri))
#     g.add((variable_uri, OntologyNamespaces.RO.value.is_part_of, study_design_variable_specification_uri, cohort_uri))

def categorization_process(g: Graph, variable_uri: URIRef, category: str, cohort_uri:URIRef) -> Graph:
    if category is None:
        return g
    category_ = category.strip().lower().replace(' ','_')
    categorization_process_uri = URIRef(f"{variable_uri}/categorization_process")
    g.add((categorization_process_uri, RDF.type, OntologyNamespaces.CMEO.value.categorization_process, cohort_uri))
    g.add((variable_uri, OntologyNamespaces.OBI.value.is_specified_input_of, categorization_process_uri, cohort_uri))
    g.add((categorization_process_uri, OntologyNamespaces.OBI.value.has_specified_input, variable_uri, cohort_uri))
    category_uri = URIRef(f"{variable_uri}/category/{category_}")
    g.add((category_uri, RDF.type, OntologyNamespaces.CMEO.value.category, cohort_uri))
    g.add((categorization_process_uri, OntologyNamespaces.OBI.value.has_specified_output, category_uri, cohort_uri))
    g.add((category_uri, OntologyNamespaces.OBI.value.is_specified_output_of, categorization_process_uri, cohort_uri))
    g.add((category_uri, OntologyNamespaces.CMEO.value.has_value, Literal(category, datatype=XSD.string), cohort_uri))
    return g

def process_variables_metadata_file(file_path:str, study_metadata_graph_file_path:str, cohort_name:str,eda_file_path:str) -> tuple[Graph, str]:

        print(f"Processing metadata file: {file_path}")
        data = load_dictionary(file_path)
        if data is None or data.empty:
            return None, None   
        data.columns = data.columns.str.lower()
        print(f"colums: {data.columns}")
        # cohort id is the folder name
        cohort_id = normalize_text(cohort_name)
        cohort_uri = get_study_uri(cohort_id)
        # Example usage of settings.cohort_folder for cohort files:
        # cohort_file = os.path.join(settings.cohort_folder, cohort_id, 'somefile.csv')
        data = data.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))
        cohort_graph = URIRef(OntologyNamespaces.CMEO.value + f"graph/{cohort_id}") #named graph for the cohort
        g = init_graph(default_graph_identifier=cohort_graph)
        # convert all data into lower case
        
        count = 0
        binary_categorical, multi_class_categorical = is_categorical_variable(data)
        variables_to_update = {}
        # units_count = 0
        data.columns = [c.lower() for c in data.columns]
        varname_col = [x for x in ['variablename', 'variable name'] 
                    if x in data.columns][0]
        for rownum, row in data.iterrows():
            count += 1
            if pd.isna(row[varname_col]):
                print("Skipping row with missing variable name.")
                continue
            # Instantiate MeasuredVariable
            var_name = normalize_text(row[varname_col]) 
            var_uri = get_var_uri(cohort_id, var_name)
            g.add((var_uri, RDF.type, OntologyNamespaces.CMEO.value.data_element, cohort_graph))
           
            # data_element_domain = row['omop'].lower().strip().replace(' ','_') if pd.notna(row['omop']) else None
                # Corrected URI assignment
            if pd.notna(row['domain']):
                g=categorization_process(g, var_uri, row['domain'], cohort_graph)
            
            statistical_type_uri,statistical_type = determine_var_uri(g, cohort_id, var_name, multi_class_categorical, binary_categorical, data_type=row['vartype'])

            g.add((statistical_type_uri, RDF.type, URIRef(f"{OntologyNamespaces.OBI.value}{statistical_type}"), cohort_graph))
            g.add((statistical_type_uri, OntologyNamespaces.CMEO.value.has_value, Literal(statistical_type, datatype=XSD.string), cohort_graph))
            g.add((var_uri, OntologyNamespaces.IAO.value.is_denoted_by, statistical_type_uri, cohort_graph))
            g.add((statistical_type_uri, OntologyNamespaces.IAO.value.denotes, var_uri, cohort_graph))
            # add dc.identifier to the variable
            g.add((var_uri, DC.identifier, Literal(row['variablename']),cohort_graph))
            # add rdfs.label to the variable
            g.add((var_uri, RDFS.label, Literal(row['variablelabel']),cohort_graph)) if pd.notna(row['variablelabel']) else None
            

            # base_concept = Concept(
            #     standard_label=row['variable concept name'] if pd.notna(row['variable concept name']) else None,
            #     code=row['variable concept code'] if pd.notna(row['variable concept code']) else None,
            #     omop_id=safe_int(row['variable omop id']) if pd.notna(row['variable omop id']) else None,
            # )
            # print(f"var name = {var_name}")
            base_concept = [Concept(
                standard_label=row['variable concept name'] if pd.notna(row['variable concept name']) else None,
                code=row['variable concept code'] if pd.notna(row['variable concept code']) else None,
                omop_id=safe_int(row['variable omop id']) if pd.notna(row['variable omop id']) else None,
            )]

            if (pd.notna(row['additional context concept name']) and 
            pd.notna(row['additional context concept code']) and 
            pd.notna(row['additional context omop id'])):
                try:
                    count1 = len(str(row['additional context concept name']).split("|"))
                    count2 = len(str(row['additional context concept code']).split("|"))
                    count3 = len(str(row['additional context omop id']).split("|"))
                    if count1 == count2 == count3:
                        base_concept.extend([Concept(
                        standard_label=str(row['additional context concept name']).split("|")[i] if pd.notna(row['additional context concept name']) else None,
                        code=str(row['additional context concept code']).split("|")[i] if pd.notna(row['additional context concept code']) else None,
                        omop_id=safe_int(row['additional context omop id'].split("|")[i]) if pd.notna(row['additional context omop id']) else None,
                    ) for i in range(count1)])
                except:
                    print(f"Row number {rownum} of {cohort_name} does not have a valid string in additional context concept names/codes/omop ids.")
            else:
                print(f"Row number {rownum} of {cohort_name} has an unequal number of additional context concept names/codes/omop ids.")

            # g=add_solo_concept_info(g, var_uri, base_concept, cohort_graph)
            # g = add_concept_info(g, var_uri, base_concept, cohort_graph)
            g = add_data_type(g, var_uri, row['vartype'], cohort_graph)
            g = add_composite_concepts_info(g, var_uri, base_concept, cohort_graph)
            
            g=add_temporal_context(g=g, var_uri=var_uri, cohort_uri=cohort_graph, row=row)

            g=add_categories_to_graph(g=g, var_uri=var_uri, cohort_uri=cohort_graph,row=row)

            g=add_measurement_unit(g, var_uri=var_uri, cohort_uri=cohort_graph, row=row)

            # study_metadata_graph_file_path = f"/Users/komalgilani/Desktop/chexo_knowledge_graph/data/graphs/studies_metadata.trig"
            
            # if metadata_graph:
            #     metadata_graph.add((var_uri, OntologyNamespaces.CMEO.value.refersTo, cohort_uri))
            missing_value = row['missing'] if pd.notna(row['missing']) else None
            if missing_value:
                missing_uri = URIRef(f"{var_uri}/missing_value_specification")
                g.add((missing_uri, RDF.type, OntologyNamespaces.CMEO.value.missing_value_specification, cohort_graph))
                g.add((missing_uri, OntologyNamespaces.CMEO.value.has_value, Literal(missing_value, datatype=XSD.string), cohort_graph))
                g.add((var_uri, OntologyNamespaces.OBI.value.has_value_specification, missing_uri, cohort_graph))
                g.add((missing_uri, OntologyNamespaces.OBI.value.is_value_specification_of, missing_uri, cohort_graph))
            
            study_variable_design_specification_uri = URIRef(cohort_uri + "/study_design_variable_specification")
            g.add((study_variable_design_specification_uri, RDF.type, OntologyNamespaces.CMEO.value.study_design_variable_specification, cohort_graph))
            g.add((var_uri, OntologyNamespaces.RO.value.is_part_of, study_variable_design_specification_uri, cohort_graph))
            

            dataset_uri = URIRef(f"{var_uri}/data_set")
            g.add((dataset_uri, RDF.type, OntologyNamespaces.IAO.value.data_set, cohort_graph))
            g.add((dataset_uri, OntologyNamespaces.IAO.value.is_about,statistical_type_uri , cohort_graph))
            g = add_device_senors_for_variable(g=g, var_uri=var_uri, data_set_uri=dataset_uri, cohort_uri=cohort_graph, row_info=row)
            variables_to_update.update({row['variablename']: (var_uri, statistical_type_uri, dataset_uri)})
        vars_list = [var_uri for var_uri, _,_ in variables_to_update.values()]
        # print(f"vars_list: {vars_list}")
        update_metadata_graph(endpoint_url=settings.update_endpoint, cohort_uri=cohort_uri, variable_uris=vars_list,metadata_graph_path=study_metadata_graph_file_path)
        if eda_file_path:
            g=add_variable_eda(g, variables_to_update, cohort_graph, eda_json_file_path=eda_file_path)

        print(f"Processed {count} rows")
        # print(f"Processed {units_count} units")
        return g, cohort_id

def add_data_type(g: Graph, var_uri: URIRef, data_type: str, cohort_uri: URIRef) -> Graph:
    if data_type is None:
        return g
    data_type_uri = URIRef(f"{var_uri}/data_type/{data_type}")
    g.add((data_type_uri, RDF.type, OntologyNamespaces.CMEO.value.data_type, cohort_uri))
    g.add((data_type_uri, OntologyNamespaces.CMEO.value.has_value, Literal(data_type, datatype=XSD.string), cohort_uri))
    g.add((data_type_uri, OntologyNamespaces.IAO.value.is_about, var_uri, cohort_uri))
    return g

def add_variable_eda(g: Graph, var_uris: list[dict], cohort_uri: URIRef, eda_json_file_path:str) -> Graph:
    """
    categorical variable should have followind EDA
        "y-ticks": "0.0 - 100.0 - 200.0 - 300.0 - 400.0 - 500.0 - 600.0",
        "x-ticks": "0 - 1 - 2 - 3 - 4",
        "Chi-Square Test Statistic": "1351.95",
        "Class balance": "(0.0, 'Unknown') -> 79%\t(3.0, 'Unknown') -> 8%\t(1.0, 'Unknown') -> 7%\t(2.0, 'Unknown') -> 5%\t(nan, 'Unknown') -> 1%",
        "url": "https://explorer.icare4cvd.eu/api/variable-graph/TIME-CHF/angina",
        "Count missing": "0 (0.00%)",
        "Type": "Categorical (encoded as float64)",
        "Most frequent category": "0",
        "Number of non-null observations": "615",       
        "Number of unique values/categories": "5"
        "CATEGORICAL": "0=0| 1=I | 2=I-II | 3=II ",
        "COUNT": 615,
        "NA": 7,
        "MIN": "0.0",
        "MAX": "3.0",
    
    """
    
    with open(eda_json_file_path, 'r') as f:
        eda_data = json.load(f)
        # make all keys lower case
        eda_data = {k.lower(): v for k, v in eda_data.items()}
      
        for var_name in var_uris.keys():
            if var_name in eda_data:
                eda = eda_data.get(var_name)
                eda = {k.lower(): v for k, v in eda.items()}
                # print(f"eda data: {eda}")  
                if eda:
                    var_uri = var_uris.get(var_name)[0]
                    statistical_var_uri = var_uris.get(var_name)[1]
                    dataset_uri = var_uris.get(var_name)[2]
                    # g.add((dataset_uri, RDF.type, OntologyNamespaces.IAO.value.data_set, cohort_uri))
                    # g.add((dataset_uri, OntologyNamespaces.IAO.value.is_about,statistical_var_uri , cohort_uri))
                    # transformed_data as a process with has_specified_input dataset and statistic as output
                    transformed_data_uri = URIRef(f"{var_uri}/exploratory_data_analysis")
                    g.add((transformed_data_uri, RDF.type, OntologyNamespaces.CMEO.value.exploratory_data_analysis, cohort_uri))
                    g.add((dataset_uri, OntologyNamespaces.OBI.value.is_specified_input_of, transformed_data_uri, cohort_uri))
                    g.add((transformed_data_uri, OntologyNamespaces.OBI.value.has_specified_input, dataset_uri, cohort_uri))
                    statistic_uri = URIRef(f"{var_uri}/statistic")
                    g.add((statistic_uri, RDF.type, OntologyNamespaces.STATO.value.statistic, cohort_uri))
                    g.add((transformed_data_uri, OntologyNamespaces.OBI.value.has_specified_output, statistic_uri, cohort_uri))
                    g.add((statistic_uri, OntologyNamespaces.OBI.value.is_specified_output_of, transformed_data_uri, cohort_uri))
                    g = add_count_to_graph(g, statistic_uri, eda.get('count (metadata dictionary)'), cohort_uri)
                    g = add_min_value_to_graph(g, statistic_uri, eda.get('min'), cohort_uri)
                    g = add_max_value_to_graph(g, statistic_uri, eda.get('max'), cohort_uri)
                    g = add_missing_value_count_to_graph(g, statistic_uri, eda.get('count empty'), cohort_uri)
                    g = add_unique_values_count_to_graph(g, statistic_uri, eda.get('number of unique values/categories'), cohort_uri)
                    print(f"variable type: {eda.get('type')} for variable {var_name}")   
                    if eda.get('type') is None:
                        # print(f"Type is None for variable {var_name}")
                        continue
                    else:
                        if 'categorical' in eda.get('type','').strip().lower():
                            

                            g = add_frequency_distribution_to_graph(g, statistic_uri, eda.get('class balance'), cohort_uri)
                            g = add_mode_to_graph(g, statistic_uri, eda.get('most frequent category'), cohort_uri)
                            g = add_chi_square_test_statistic_to_graph(g, var_uri, dataset_uri, eda.get('chi-square test statistic'), cohort_uri)
                            g = add_categorical_variable_visualization(g, var_uri, cohort_uri, eda.get('url'), [eda.get('x-ticks'), eda.get('y-ticks')])
                        else:
                            g = add_histogram_visualization(g, var_uri=var_uri, cohort_uri=cohort_uri, chart_url=eda.get('url'), xy_axis=[eda.get('x-ticks'), eda.get('y-ticks')])
                            g = add_outlier_count_by_iqr_to_graph(g, statistic_uri, eda.get('outliers (iqr)'), cohort_uri)
                            g = add_outlier_count_by_z_score_to_graph(g, statistic_uri, eda.get('outliers (z)'), cohort_uri)
                            g = add_normality_test_to_graph(g, statistic_uri, eda.get('normality test'), cohort_uri)
                            g = add_wilks_shapiro_test_to_graph(g, statistic_uri, eda.get('w_test'), cohort_uri)
                            g = add_standard_deviation_to_graph(g, statistic_uri, eda.get('Std Dev'), cohort_uri)
                            g = add_mean_to_graph(g, statistic_uri, eda.get('mean'), cohort_uri)
                            g = add_median_to_graph(g, statistic_uri, eda.get('median'), cohort_uri)
                            g = add_iqr_to_graph(g, statistic_uri, eda.get('iqr'), cohort_uri)
                            g = add_q1_to_graph(g, statistic_uri, eda.get('q1'), cohort_uri)
                            g = add_q3_to_graph(g, statistic_uri, eda.get('q3'), cohort_uri)
                            g = add_variance_to_graph(g, statistic_uri, eda.get('variance'), cohort_uri)
                            g = add_kurtosis_to_graph(g, statistic_uri, eda.get('kurtosis'), cohort_uri)
                            g = add_skewness_to_graph(g, statistic_uri, eda.get('skewness'), cohort_uri)

    return g
                            
                    
def add_count_to_graph(g: Graph, statistic_uri: URIRef, count: any, cohort_uri: URIRef) -> Graph:
 
    if count is None:
        return g
    count = int(count) 
    # print(f"count: {count}")
    count_uri = URIRef(f"{statistic_uri}/count_")
    g.add((count_uri, RDF.type, OntologyNamespaces.STATO.value.count_, cohort_uri))
    g.add((count_uri, OntologyNamespaces.CMEO.value.has_value, Literal(count, datatype=XSD.integer), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, count_uri, cohort_uri))
    g.add((count_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g

def add_min_value_to_graph(g: Graph, statistic_uri: URIRef, min_value: any, cohort_uri: URIRef) -> Graph:
    if min_value is None:
        return g
    # print(f"min value = {min_value}")
    min_uri = URIRef(f"{statistic_uri}/minimum_value")

    g.add((min_uri, RDF.type, OntologyNamespaces.STATO.value.minimum_value, cohort_uri))
    if isinstance(min_value, str):
        g.add((min_uri, OntologyNamespaces.CMEO.value.has_value, Literal(min_value, datatype=XSD.string), cohort_uri))
    else:
        g.add((min_uri, OntologyNamespaces.CMEO.value.has_value, Literal(min_value, datatype=XSD.float), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, min_uri, cohort_uri))
    g.add((min_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g



def add_max_value_to_graph(g: Graph, statistic_uri: URIRef, max_value: any, cohort_uri: URIRef) -> Graph:
    if max_value is None:
        return g
    # print(f"max value = {max_value}")
    max_uri = URIRef(f"{statistic_uri}/maximum_value")
    g.add((max_uri, RDF.type, OntologyNamespaces.STATO.value.maximum_value, cohort_uri))
    if isinstance(max_value, str):
        g.add((max_uri, OntologyNamespaces.CMEO.value.has_value, Literal(max_value, datatype=XSD.string), cohort_uri))
    else:
        g.add((max_uri, OntologyNamespaces.CMEO.value.has_value, Literal(max_value, datatype=XSD.float), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, max_uri, cohort_uri))
    g.add((max_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g

def add_missing_value_count_to_graph(g: Graph, statistic_uri: URIRef, na: int, cohort_uri: URIRef) -> Graph:
    if na is None:
        return g
    na = int(na.split("(")[0].strip())
    # print(f"missing value = {na}")
    na_uri = URIRef(f"{statistic_uri}/number_of_missing_values")
    g.add((na_uri, RDF.type, OntologyNamespaces.CMEO.value.missing_values_count, cohort_uri))
    g.add((na_uri, OntologyNamespaces.CMEO.value.has_value, Literal(na, datatype=XSD.integer), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, na_uri, cohort_uri))
    g.add((na_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g

def add_missing_value_percentage_to_graph(g: Graph, statistic_uri: URIRef, missing_value_percentage: float, cohort_uri: URIRef) -> Graph:
    if missing_value_percentage is None:
        return g
    print(f"missing value percentage = {missing_value_percentage}")
    missing_value_percentage_uri = URIRef(f"{statistic_uri}/missing_value_percentage")
    g.add((missing_value_percentage_uri, RDF.type, OntologyNamespaces.CMEO.value.missing_values_percentage, cohort_uri))
    g.add((missing_value_percentage_uri, OntologyNamespaces.CMEO.value.has_value, Literal(missing_value_percentage, datatype=XSD.decimal), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, missing_value_percentage_uri, cohort_uri))
    g.add((missing_value_percentage_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g


def add_unique_values_count_to_graph(g: Graph, statistic_uri: URIRef, unique_values: any, cohort_uri: URIRef) -> Graph:
    if unique_values is None:
        return g
    unique_values_uri = URIRef(f"{statistic_uri}/number_of_unique_values")
    unique_values = int(unique_values)
    # print(f"unique values = {unique_values}")
    g.add((unique_values_uri, RDF.type, OntologyNamespaces.CMEO.value.unique_values_count, cohort_uri))
    g.add((unique_values_uri, OntologyNamespaces.CMEO.value.has_value, Literal(unique_values, datatype=XSD.integer), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, unique_values_uri, cohort_uri))
    g.add((unique_values_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g


def add_device_senors_for_variable(g: Graph, var_uri:URIRef, data_set_uri: URIRef,  cohort_uri: URIRef, row_info:pd.Series) -> Graph:
    
    data_acquisition_process_uri = URIRef(f"{var_uri}/data_acquisition_process")
    g.add((data_acquisition_process_uri, RDF.type, OntologyNamespaces.CMEO.value.data_acquisition_process, cohort_uri))
    g.add((data_set_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_acquisition_process_uri, cohort_uri))
    g.add((data_acquisition_process_uri, OntologyNamespaces.OBI.value.has_specified_output, data_set_uri, cohort_uri))
    if 'devices' in row_info and pd.notna(row_info['devices']):
        device_value = row_info['devices'].strip().lower()
        device_uri = URIRef(OntologyNamespaces.CMEO.value + f"wearable_device/{device_value.replace(' ','_')}")
        g.add((device_uri, RDF.type, OntologyNamespaces.CMEO.value.wearable_device, cohort_uri))
        g.add((device_uri, OntologyNamespaces.CMEO.value.has_value, Literal(device_value, datatype=XSD.string), cohort_uri))  
        g.add((data_acquisition_process_uri, OntologyNamespaces.OBI.value.has_specified_input, device_uri, cohort_uri))
        g.add((device_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_acquisition_process_uri, cohort_uri))
    if 'sensors' in row_info and pd.notna(row_info['sensors']):
        sensor_value = row_info['sensors'].strip().lower()
        sensor_uri =URIRef(OntologyNamespaces.CMEO.value + f"sensor/{sensor_value.replace(' ','_')}")
        g.add((sensor_uri, RDF.type, OntologyNamespaces.CMEO.value.sensor, cohort_uri))
        g.add((sensor_uri, OntologyNamespaces.CMEO.value.has_value, Literal(sensor_value, datatype=XSD.string), cohort_uri))
        g.add((device_uri, OntologyNamespaces.RO.value.has_part, sensor_uri, cohort_uri))
        g.add((sensor_uri, OntologyNamespaces.RO.value.is_part_of, device_uri, cohort_uri))
    return g

        
    
def add_frequency_distribution_to_graph(g: Graph, statistic_uri: URIRef, frequency_distribution:str, cohort_uri: URIRef) -> Graph:
    if frequency_distribution is None:
        return g
    print(f"frequency distribution = {frequency_distribution}")
    frequency_distribution_uri = URIRef(f"{statistic_uri}/frequency_distribution")
    g.add((frequency_distribution_uri, RDF.type, OntologyNamespaces.CMEO.value.frequency_distribution, cohort_uri))
    g.add((frequency_distribution_uri, OntologyNamespaces.CMEO.value.has_value, Literal(frequency_distribution, datatype=XSD.string), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, frequency_distribution_uri, cohort_uri))
    g.add((frequency_distribution_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g


def add_mode_to_graph(g: Graph, statistic_uri: URIRef, mode: any, cohort_uri: URIRef) -> Graph:
    if mode is None:
        return g
    mode_uri = URIRef(f"{statistic_uri}/mode")
    print(f"mode: {mode}")  
    
    g.add((mode_uri, RDF.type, OntologyNamespaces.STATO.value.mode, cohort_uri))
    
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, mode_uri, cohort_uri))
    g.add((mode_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    if isinstance(mode, str):
        g.add((mode_uri, OntologyNamespaces.CMEO.value.has_value, Literal(mode, datatype=XSD.string), cohort_uri))
    else:
        g.add((mode_uri, OntologyNamespaces.CMEO.value.has_value, Literal(mode, datatype=XSD.float), cohort_uri))
    return g

def add_chi_square_test_statistic_to_graph(g: Graph, var_uri:URIRef, dataset_uri: URIRef, chi_square_test_statistic: Any, cohort_uri: URIRef) -> Graph:
    if chi_square_test_statistic is None:
        return g
    print(f"chi square test statistic = {chi_square_test_statistic}")  

    chi_square_test_process = URIRef(f"{dataset_uri}/chi_square_test")
    g.add((chi_square_test_process, RDF.type, OntologyNamespaces.OBI.value.chi_square_test, cohort_uri))
    g.add((dataset_uri, OntologyNamespaces.OBI.value.is_specified_input_of, chi_square_test_process, cohort_uri))
    g.add((chi_square_test_process, OntologyNamespaces.OBI.value.has_specified_input, dataset_uri, cohort_uri))
    chi_square_test_statistic = float(chi_square_test_statistic)    
    chi_square_test_statistic_uri = URIRef(f"{var_uri}/chi_square_test_value")
    g.add((chi_square_test_statistic_uri, RDF.type, OntologyNamespaces.STATO.value.chi_square_test_statistic, cohort_uri))
    g.add((chi_square_test_process, OntologyNamespaces.OBI.value.has_specified_output, chi_square_test_statistic_uri, cohort_uri))
    g.add((chi_square_test_statistic_uri, OntologyNamespaces.OBI.value.is_specified_output_of, chi_square_test_process, cohort_uri))
    if isinstance(chi_square_test_statistic, str):
        g.add((chi_square_test_statistic_uri, OntologyNamespaces.CMEO.value.has_value, Literal(chi_square_test_statistic, datatype=XSD.string), cohort_uri))
    else:
        g.add((chi_square_test_statistic_uri, OntologyNamespaces.CMEO.value.has_value, Literal(chi_square_test_statistic, datatype=XSD.float), cohort_uri))
    
    return g


# add standard deviation to the graph
def add_standard_deviation_to_graph(g: Graph, statistic_uri: URIRef, std_dev: Any, cohort_uri: URIRef) -> Graph:
    if std_dev is None:
        return g
    print(f"std_dev: {std_dev}")
    std_dev_uri = URIRef(f"{statistic_uri}/standard_deviation")
    g.add((std_dev_uri, RDF.type, OntologyNamespaces.STATO.value.standard_deviation, cohort_uri))
    
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, std_dev_uri, cohort_uri))
    g.add((std_dev_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))

    if isinstance(std_dev, str):
        g.add((std_dev_uri, OntologyNamespaces.CMEO.value.has_value, Literal(std_dev, datatype=XSD.string), cohort_uri))
    else:
        g.add((std_dev_uri, OntologyNamespaces.CMEO.value.has_value, Literal(std_dev, datatype=XSD.float), cohort_uri))
    return g

# add mean to the graph as average_value 
def add_mean_to_graph(g: Graph, statistic_uri: URIRef, mean: Any, cohort_uri: URIRef) -> Graph:
    if mean is None:
        return g
    print(f"mean: {mean}")
    mean_uri = URIRef(f"{statistic_uri}/average_value")
    g.add((mean_uri, RDF.type, OntologyNamespaces.STATO.value.mean, cohort_uri))
    if isinstance(mean, str):
        g.add((mean_uri, OntologyNamespaces.CMEO.value.has_value, Literal(mean, datatype=XSD.string), cohort_uri))
    else:
        g.add((mean_uri, OntologyNamespaces.CMEO.value.has_value, Literal(mean, datatype=XSD.float), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, mean_uri, cohort_uri))
    g.add((mean_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g

# add median to the graph
def add_median_to_graph(g: Graph, statistic_uri: URIRef, median: Any, cohort_uri: URIRef) -> Graph:
    if median is None:
        return g
    print(f"median: {median}")
    median_uri = URIRef(f"{statistic_uri}/median")
    g.add((median_uri, RDF.type, OntologyNamespaces.STATO.value.median, cohort_uri))
   
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, median_uri, cohort_uri))
    g.add((median_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    if isinstance(median, str):
        g.add((median_uri, OntologyNamespaces.CMEO.value.has_value, Literal(median, datatype=XSD.string), cohort_uri))
    else:
        g.add((median_uri, OntologyNamespaces.CMEO.value.has_value, Literal(median, datatype=XSD.float), cohort_uri))
    return g


# add iqr  as interquartile range to the graph

def add_iqr_to_graph(g: Graph, statistic_uri: URIRef, iqr: Any, cohort_uri: URIRef) -> Graph:
    print(f"iqr: {iqr}")
    if iqr is None or iqr == "nan":
        return g
   
    iqr_uri = URIRef(f"{statistic_uri}/interquartile_range")
    # iqr = float(iqr)
    g.add((iqr_uri, RDF.type, OntologyNamespaces.STATO.value.interquartile_range, cohort_uri))
 
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, iqr_uri, cohort_uri))
    g.add((iqr_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    if isinstance(iqr, str):
         g.add((iqr_uri, OntologyNamespaces.CMEO.value.has_value, Literal(iqr, datatype=XSD.string), cohort_uri))
    else:
        g.add((iqr_uri, OntologyNamespaces.CMEO.value.has_value, Literal(iqr, datatype=XSD.float), cohort_uri))
    return g

# add Q1 to the graph
def add_q1_to_graph(g: Graph, statistic_uri: URIRef, q1: Any, cohort_uri: URIRef) -> Graph:
    if q1 is None or q1 == "nan":
        return g
    print(f"q1: {q1}")
    # q1 = float(q1)
    q1_uri = URIRef(f"{statistic_uri}/first_quartile")
    g.add((q1_uri, RDF.type, OntologyNamespaces.STATO.value.first_quartile, cohort_uri))
    
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, q1_uri, cohort_uri))
    g.add((q1_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    if isinstance(q1, str):
         g.add((q1_uri, OntologyNamespaces.CMEO.value.has_value, Literal(q1, datatype=XSD.string), cohort_uri))
    else:
        g.add((q1_uri, OntologyNamespaces.CMEO.value.has_value, Literal(q1, datatype=XSD.float), cohort_uri))
    return g

# add Q3 to the graph
def add_q3_to_graph(g: Graph, statistic_uri: URIRef, q3: Any, cohort_uri: URIRef) -> Graph:
    if q3 is None or q3 == "nan":
        return g
    print(f"q3: {q3}") 
    # q3 = float(q3)
    q3_uri = URIRef(f"{statistic_uri}/third_quartile")
    g.add((q3_uri, RDF.type, OntologyNamespaces.STATO.value.third_quartile, cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, q3_uri, cohort_uri))
    g.add((q3_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    if isinstance(q3, str):
         g.add((q3_uri, OntologyNamespaces.CMEO.value.has_value, Literal(q3, datatype=XSD.string), cohort_uri))
    else:
         g.add((q3_uri, OntologyNamespaces.CMEO.value.has_value, Literal(q3, datatype=XSD.float), cohort_uri))
    return g

# add outliers count by iqr to the graph
def add_outlier_count_by_iqr_to_graph(g: Graph, statistic_uri: URIRef, outlier_count: int, cohort_uri: URIRef) -> Graph:
    if outlier_count is None or isinstance(outlier_count, str):
        return g
    outlier_count = int(outlier_count.split("(")[0].strip())
    print(f"iqr outlier_count: {outlier_count}")
    outlier_count_uri = URIRef(f"{statistic_uri}/outlier_count_by_iqr")
    g.add((outlier_count_uri, RDF.type, OntologyNamespaces.STATO.value.outlier_count_by_iqr, cohort_uri))
    g.add((outlier_count_uri, OntologyNamespaces.CMEO.value.has_value, Literal(outlier_count, datatype=XSD.integer), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, outlier_count_uri, cohort_uri))
    g.add((outlier_count_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g

# add outlier count by z score to the graph

def add_outlier_count_by_z_score_to_graph(g: Graph, statistic_uri: URIRef, outlier_count: Any, cohort_uri: URIRef) -> Graph:
    if outlier_count is None or isinstance(outlier_count, str):
        return g
    # outlier_count = int(outlier_count.split("(")[0].strip())
    print(f"z outlier_count: {outlier_count}")
    outlier_count_uri = URIRef(f"{statistic_uri}/outlier_count_by_z_score")
    g.add((outlier_count_uri, RDF.type, OntologyNamespaces.STATO.value.outlier_count_by_z_score, cohort_uri))
    g.add((outlier_count_uri, OntologyNamespaces.CMEO.value.has_value, Literal(outlier_count, datatype=XSD.integer), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, outlier_count_uri, cohort_uri))
    g.add((outlier_count_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g



# add normality test to the graph
def add_normality_test_to_graph(g: Graph, statistic_uri: URIRef, normality_test: str, cohort_uri: URIRef) -> Graph:
    if normality_test is None:
        return g
    normality_test_uri = URIRef(f"{statistic_uri}/normality_test")
    g.add((normality_test_uri, RDF.type, OntologyNamespaces.STATO.value.normality_test, cohort_uri))
    g.add((normality_test_uri, OntologyNamespaces.CMEO.value.has_value, Literal(normality_test, datatype=XSD.string), cohort_uri))
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, normality_test_uri, cohort_uri))
    g.add((normality_test_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    return g


# add wilks shapiro test to the graph
def add_wilks_shapiro_test_to_graph(g: Graph, statistic_uri: URIRef, wilks_shapiro_test: str, cohort_uri: URIRef) -> Graph:
    if wilks_shapiro_test is None:
        return g
    wilks_shapiro_test_uri = URIRef(f"{statistic_uri}/wilks_shapiro_test")
    g.add((wilks_shapiro_test_uri, RDF.type, OntologyNamespaces.STATO.value.wilks_shapiro_test, cohort_uri))
   
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, wilks_shapiro_test_uri, cohort_uri))
    g.add((wilks_shapiro_test_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    if isinstance(wilks_shapiro_test, str):
        g.add((wilks_shapiro_test_uri, OntologyNamespaces.CMEO.value.has_value, Literal(wilks_shapiro_test, datatype=XSD.string), cohort_uri))
    else:
        wilks_shapiro_test = float(wilks_shapiro_test)
        g.add((wilks_shapiro_test_uri, OntologyNamespaces.CMEO.value.has_value, Literal(wilks_shapiro_test, datatype=XSD.float), cohort_uri))
    return g

# add skewness to the graph
def add_skewness_to_graph(g: Graph, statistic_uri: URIRef, skewness: Any, cohort_uri: URIRef) -> Graph:
    if skewness is None:
        return g
    skewness_uri = URIRef(f"{statistic_uri}/skewness")
    g.add((skewness_uri, RDF.type, OntologyNamespaces.STATO.value.skewness, cohort_uri))
    
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, skewness_uri, cohort_uri))
    g.add((skewness_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    if isinstance(skewness, str):
        g.add((skewness_uri, OntologyNamespaces.CMEO.value.has_value, Literal(skewness, datatype=XSD.string), cohort_uri))
    else:
        g.add((skewness_uri, OntologyNamespaces.CMEO.value.has_value, Literal(skewness, datatype=XSD.float), cohort_uri))
    return g

# add variance to the graph
def add_variance_to_graph(g: Graph, statistic_uri: URIRef, variance: Any, cohort_uri: URIRef) -> Graph:
    if variance is None or variance == "nan":
        return g
    print(f"variance: {variance}")
    variance_uri = URIRef(f"{statistic_uri}/variance")
    
    g.add((variance_uri, RDF.type, OntologyNamespaces.STATO.value.variance, cohort_uri))
   
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, variance_uri, cohort_uri))
    g.add((variance_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    if isinstance(variance, str):   
        g.add((variance_uri, OntologyNamespaces.CMEO.value.has_value, Literal(variance, datatype=XSD.string), cohort_uri))
    else:
        
        variance = float(variance)
        g.add((variance_uri, OntologyNamespaces.CMEO.value.has_value, Literal(variance, datatype=XSD.float), cohort_uri))
    return g


# add kurtosis to the graph
def add_kurtosis_to_graph(g: Graph, statistic_uri: URIRef, kurtosis: Any, cohort_uri: URIRef) -> Graph:
    if kurtosis is None:
        return g
    print(f"kurtosis: {kurtosis}")
    kurtosis_uri = URIRef(f"{statistic_uri}/kurtosis")
    g.add((kurtosis_uri, RDF.type, OntologyNamespaces.STATO.value.kurtosis, cohort_uri))
    
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, kurtosis_uri, cohort_uri))
    g.add((kurtosis_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))
    
    
    if isinstance(kurtosis, str):
        g.add((kurtosis_uri, OntologyNamespaces.CMEO.value.has_value, Literal(kurtosis, datatype=XSD.string), cohort_uri))
    else:
        kurtosis = float(kurtosis)
        g.add((kurtosis_uri, OntologyNamespaces.CMEO.value.has_value, Literal(kurtosis, datatype=XSD.float), cohort_uri))
    return g


def add_categorical_variable_visualization(g: Graph, var_uri: URIRef, cohort_uri: URIRef, chart_url:str, xy_axis:list[str]) -> Graph:
    if chart_url is None or xy_axis is None:
        return g
    data_visualization_process_uri = URIRef(f"{var_uri}/data_visualization_process")
    g.add((data_visualization_process_uri, RDF.type, OntologyNamespaces.CMEO.value.data_visualization_process, cohort_uri))
    g.add((var_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_visualization_process_uri, cohort_uri))
    g.add((data_visualization_process_uri, OntologyNamespaces.OBI.value.has_specified_input, var_uri, cohort_uri))
    data_visualization_output_uri = URIRef(f"{var_uri}/bar_chart")
    g.add((data_visualization_output_uri, RDF.type, OntologyNamespaces.CMEO.value.bar_chart, cohort_uri))
    g.add((data_visualization_process_uri, OntologyNamespaces.OBI.value.has_specified_output, data_visualization_output_uri, cohort_uri))
    g.add((data_visualization_output_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_visualization_process_uri, cohort_uri))
    g.add((data_visualization_output_uri, OntologyNamespaces.CMEO.value.has_value, Literal(chart_url, datatype=XSD.string), cohort_uri))

    x_axis_uri = URIRef(f"{data_visualization_output_uri}/x_axis")
    g.add((x_axis_uri, RDF.type, OntologyNamespaces.CMEO.value.x_axis, cohort_uri))
    g.add((x_axis_uri, OntologyNamespaces.CMEO.value.has_value, Literal(xy_axis[0], datatype=XSD.string), cohort_uri))
    g.add((data_visualization_output_uri, OntologyNamespaces.RO.value.has_part, x_axis_uri, cohort_uri))
    g.add((x_axis_uri, OntologyNamespaces.RO.value.is_part_of, data_visualization_output_uri, cohort_uri))
    y_axis_uri = URIRef(f"{data_visualization_output_uri}/y_axis")
    g.add((y_axis_uri, RDF.type, OntologyNamespaces.CMEO.value.y_axis, cohort_uri))
    g.add((y_axis_uri, OntologyNamespaces.CMEO.value.has_value, Literal(xy_axis[1], datatype=XSD.string), cohort_uri))
    g.add((data_visualization_output_uri, OntologyNamespaces.RO.value.has_part, y_axis_uri, cohort_uri))
    g.add((y_axis_uri, OntologyNamespaces.RO.value.is_part_of, data_visualization_output_uri, cohort_uri))
    return g


def add_histogram_visualization(g: Graph, var_uri: URIRef, cohort_uri: URIRef, chart_url:str, xy_axis:list[str]) -> Graph:
    if chart_url is None or xy_axis is None:
        return g
    
    x_ticks =  extract_tick_values(xy_axis[0])
    y_ticks = extract_tick_values(xy_axis[1])
    # print(f"x_ticks: {x_ticks}")    
    # print(f"y_ticks: {y_ticks}")
    # x_ticks = list(map(float, xy_axis[0].split(" - ")))
    # y_ticks = list(map(float, xy_axis[1].split(" - ")))
    
    #replace("text("," ").replace(")"," ").split(",")))

    # Step 2: Calculate x-axis metadata
    bin_edges = x_ticks
    num_bins = len(bin_edges) - 1
    bin_widths = [bin_edges[i+1] - bin_edges[i] for i in range(num_bins)]

    # Step 3: Get axis ranges
    # x_min = min(x_ticks)
    # x_max = max(x_ticks)
    # y_min = min(y_ticks)
    # y_max = max(y_ticks)

    # Step 4: Summary
    # histogram_info = {
    #     "x_ticks": x_ticks,
    #     "y_ticks": y_ticks,
    #     "number_of_bins": num_bins,
    #     "bin_edges": bin_edges,
    #     "bin_widths": bin_widths,
    #     "x_range": (x_min, x_max),
    #     "y_range": (y_min, y_max),
    #     "uniform_bin_width": all(w == bin_widths[0] for w in bin_widths)
    # }
  
    # Step 5: Add metadata to the graph
    data_visualization_process_uri = URIRef(f"{var_uri}/data_visualization_process")
    g.add((data_visualization_process_uri, RDF.type, OntologyNamespaces.CMEO.value.data_visualization_process, cohort_uri))
    g.add((var_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_visualization_process_uri, cohort_uri))
    g.add((data_visualization_process_uri, OntologyNamespaces.OBI.value.has_specified_input, var_uri, cohort_uri))
    histogram_uri = URIRef(f"{var_uri}/histogram")
    g.add((histogram_uri, RDF.type, OntologyNamespaces.CMEO.value.histogram, cohort_uri))
    g.add((data_visualization_process_uri, OntologyNamespaces.OBI.value.has_specified_output, histogram_uri, cohort_uri))
    g.add((histogram_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_visualization_process_uri, cohort_uri))
    g.add((histogram_uri, OntologyNamespaces.CMEO.value.has_value, Literal(chart_url, datatype=XSD.string), cohort_uri))
    
    number_of_bins_uri = URIRef(f"{histogram_uri}/number_of_bins")
    g.add((number_of_bins_uri, RDF.type, OntologyNamespaces.CMEO.value.number_of_bins, cohort_uri))
    g.add((number_of_bins_uri, OntologyNamespaces.CMEO.value.has_value, Literal(num_bins, datatype=XSD.integer), cohort_uri))
    g.add((histogram_uri, OntologyNamespaces.RO.value.has_part, number_of_bins_uri, cohort_uri))
    g.add((number_of_bins_uri, OntologyNamespaces.RO.value.is_part_of, histogram_uri, cohort_uri))

    bin_edges_uri = URIRef(f"{histogram_uri}/bin_edges")
    g.add((bin_edges_uri, RDF.type, OntologyNamespaces.CMEO.value.bin_edges, cohort_uri))
    g.add((bin_edges_uri, RDF.type, RDF.Seq))
    g.add((histogram_uri, OntologyNamespaces.RO.value.has_part, bin_edges_uri, cohort_uri))
    g.add((bin_edges_uri, OntologyNamespaces.RO.value.is_part_of, histogram_uri, cohort_uri))
    for i, value in enumerate(bin_edges):
        # print(f"bin_edges: {value}")
        g.add((bin_edges_uri, RDF[f"_{i+1}"], Literal(value, datatype=XSD.float), cohort_uri))

    return g


def add_temporal_context(g: Graph, var_uri: URIRef, cohort_uri: URIRef, row: pd.Series) -> Graph:
    visit_not_null = pd.notna(row['visits']) and row['visits']
    if visit_not_null:
        visit_labels = normalize_text(row['visits'])
        visit_uri = get_temporal_context_uri(var_uri, visit_labels)
        g.add((visit_uri, RDF.type, OntologyNamespaces.CMEO.value.visit_measurement_datum, cohort_uri))
        g.add((visit_uri, OntologyNamespaces.IAO.value.is_about, var_uri, cohort_uri))
        g.add((visit_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row['visits'], datatype=XSD.string), cohort_uri))

        
        if pd.notna(row['visit concept name']) and normalize_text(row['visit concept name']):
            concepts = Concept(
                standard_label=row['visit concept name'] if pd.notna(row['visit concept name']) else None,
                code=row['visit concept code'] if pd.notna(row['visit concept code']) else None,
                omop_id=safe_int(row['visit omop id']) if pd.notna(row['visit omop id']) else None,
            )
            add_solo_concept_info(g, visit_uri, concepts, cohort_uri)
          
            return g
         
    return g


def add_categories_to_graph(g: Graph, var_uri: URIRef, cohort_uri: URIRef, row: pd.Series) -> Graph:
    """Adds permissible information related to variable to the RDF graph for a specific variable URI."""
    # Check if there are categories to process
    datatype = row['vartype'] if pd.notna(row['vartype']) else None

    if pd.notna(row['categorical']) and datatype:
        # if row['categorical'] == '' or row['categorical'] == 'nan' or row['categorical'] == None:
        #     print("No categorical information found")
        #     return g
        categories =row['categorical'].split("|")
        updated_categories = []
        
        # Parse categories as value=defined_value pairs
        for category in categories:
            try:
                value, defined_value = map(str.strip, category.split('='))
                updated_categories.append((value, defined_value))
            except ValueError:
                continue  # Skip if format is invalid

            
        # Handle additional columns for labels, codes, and OMOP IDs, with empty fallback
        labels = row['categorical value concept name'].split("|") if pd.notna(row['categorical value concept name']) and row['categorical value concept name'] else [None] * len(categories)
        codes = row['categorical value concept code'].split("|") if pd.notna(row['categorical value concept code']) and row['categorical value concept code'] else [None] * len(categories)
        omop_ids = row['categorical value omop id'].split("|") if pd.notna(row['categorical value omop id']) and row['categorical value omop id'] else [None] * len(categories)
        
        # Add each category to the graph
        for i, (value, defined_value) in enumerate(updated_categories):
            # n_value = normalize_text(value)
            n_defined_value = normalize_text(defined_value)  
            # if value is None or value == 'nan' or value == '':
            #     print(f"value: {value} defined_value: {defined_value}")
            permissible_uri = URIRef(f"{var_uri}/categorical_value_specification/{n_defined_value}")
            g.add((permissible_uri, RDF.type, OntologyNamespaces.OBI.value.categorical_value_specification, cohort_uri))
            g.add((var_uri, OntologyNamespaces.OBI.value.has_value_specification, permissible_uri, cohort_uri))
            g.add((permissible_uri, OntologyNamespaces.OBI.value.is_value_specification_of, var_uri, cohort_uri))
            g.add((permissible_uri, RDFS.label, Literal(defined_value, datatype=XSD.string), cohort_uri))
            if datatype == 'str':
                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.string), cohort_uri))
            elif datatype == 'int':
                if value is None:  print(f"value: {value} defined_value: {defined_value}") 

                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.integer), cohort_uri))
            elif datatype == 'float':
                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.decimal), cohort_uri))
            elif datatype == 'datetime':
                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.dateTime), cohort_uri))


            # Add optional labels, codes, and OMOP IDs if present
            if i < len(labels) and len(labels) == len(updated_categories):
                # label = normalize_text(labels[i])

                # code = normalize_text(codes[i].strip()) if codes[i] else None
                # omop_id = safe_int(omop_ids[i]) if omop_ids[i] else None
                # print(f"labels={labels[i]}")
                print(f"codes={codes[i]}")
                print(f"omop_ids={omop_ids[i]}")
                print(f"label={labels[i]}")
                concept =  Concept(
                    standard_label=labels[i] if pd.notna(labels[i]) else None,
                    code=codes[i] if pd.notna(codes[i]) else None,
                    omop_id=safe_int(omop_ids[i]) if pd.notna(omop_ids[i]) else None,
                )
                g=add_solo_concept_info(g, permissible_uri, concept, cohort_uri)
                # g=add_concept_info(g, categorical_label_uri, label, code, omop_id, cohort_uri) if label and code and omop_id else g
    # else:
    #     print("No categorical information found")
           
    return g





def add_measurement_unit(g:Graph, var_uri:URIRef, cohort_uri: URIRef,row: pd.Series) -> Graph:
    """
    Adds measurement unit details to the RDF graph for a given variable URI.
    
    Parameters:
    - g: The RDF graph to add triples to.
    - var_uri: The URI of the measured variable to which the unit is related.
    - measurement_unit: An object or dictionary containing the unit details, 
      including `unit_value`, `label`, `code`, and `omop_id`.
    """
    # measurement_unit = MeasurementUnit(
    #     value=row['units'] if pd.notna(row['units']) else None,
        
    # )
   
    if pd.notna(row['units']) and pd.notna(row['unit concept name']):
        unit_concept= Concept(
            standard_label=row['unit concept name'] if pd.notna(row['unit concept name']) else None,
            code=row['unit concept code'] if pd.notna(row['unit concept code']) else None,
            omop_id=safe_int(row['unit omop id']) if pd.notna(row['unit omop id']) else None
        )
        #print(f"unit_concept: {unit_concept[0].code} {unit_concept[0].standard_label} {unit_concept[0].omop_id}")
        # Create a unique URI for the measurement unit
        unit_uri = URIRef(f"{var_uri}/measurement_unit_label")
        unit_value=row['units'] if pd.notna(row['units']) else None,
        # Add the measurement unit type to the graph
        g.add((unit_uri, RDF.type, OntologyNamespaces.OBI.value.measurement_unit_label, cohort_uri))

        g.add((unit_uri, OntologyNamespaces.CMEO.value.has_value, Literal(unit_value, datatype=XSD.string),cohort_uri))

        g= add_solo_concept_info(g, unit_uri, unit_concept, cohort_uri)
        g.add((var_uri, OntologyNamespaces.OBI.value.has_measurement_unit_label, unit_uri,cohort_uri))
        # g.add((var_uri, OntologyNamespaces.RO.value.has_part, unit_uri,cohort_uri))

    return g

def get_category_uri(var_uri: URIRef, category_value: str) -> URIRef:
    """Generates a unique and URI-safe URI for each category based on the variable URI and category value."""
    # Encode the category value to make it URI-safe
    safe_category_value = normalize_text(category_value) # This will replace unsafe characters with % encoding
    return URIRef(f"{var_uri}/category/{safe_category_value}")


def get_var_uri(cohort_id: str | URIRef, var_id: str) -> URIRef:
    safe_var_id = normalize_text(var_id)
    if safe_var_id == "":
        print("Variable ID is empty")
    # safe_cohort_id = normalize_text(cohort_id)
    return URIRef(f"{OntologyNamespaces.CMEO.value}{cohort_id}/{safe_var_id}")

def get_context_uri(var_uri: str | URIRef, context_id: str) -> URIRef: 
    safe_context_id = normalize_text(context_id)
    return URIRef(f"{var_uri!s}/context/{safe_context_id}")

def get_temporal_context_uri(var_uri: str | URIRef, temporal_context_id: str) -> URIRef:
    safe_temporal_context_id = normalize_text(temporal_context_id)
    return URIRef(f"{var_uri!s}/visit/{safe_temporal_context_id}")

def get_measurement_unit_uri(var_uri: str | URIRef, unit_label: str) -> URIRef:
    if unit_label is None:
        print("Unit label is None")
    safe_unit_label = normalize_text(unit_label)
    return URIRef(f"{var_uri!s}/unit/{safe_unit_label}")


def get_standard_label_uri(var_uri: URIRef, base_entity_label: str) -> URIRef:
    safe_base_entity_label = normalize_text(base_entity_label)
    return URIRef(f"{var_uri}/standard_label/{safe_base_entity_label}")


def get_code_uri(var_uri: URIRef, code: str) -> URIRef:
    # safe_base_entity_label = sanitize(base_entity_label)
    safe_code = normalize_text(code)
    return URIRef(f"{var_uri}/code/{safe_code}")

def get_omop_id_uri(var_uri: URIRef, omop_id: str) -> URIRef:
    # safe_base_entity_label = sanitize(base_entity_label)
    if omop_id == "" or omop_id is None:
        print("OMOP ID is empty")
    return URIRef(f"{var_uri}/omop_id/{omop_id}")

def add_context(g: Graph, var_uri: URIRef, cohort_uri: URIRef, row: pd.Series) -> Graph:
    """
    Add context to the RDF graph for a specific variable URI.
    
    :param g: RDF Graph object
    :param var_uri: URI of the variable to add context to
    :param labels: List of context labels
    :param codes: List of context codes
    :param omop_ids: List of context OMOP IDs
    """
    # visit_not_null = pd.notna(row['visits']) and row['visits']
    # if visit not null the -1 entry in Additional Context Concept Label should be about visits
    concepts = []
    context_label = ""
    if pd.notna(row['additional context concept name']) and pd.notna(row['additional context concept code']) and pd.notna(row['additional context omop id']):
        labels = row['additional context concept name'].split("|")
        codes = row['additional context concept code'].split("|")
        omop_ids = row['additional context omop id'].split("|")
        # print(f"labels: {labels} codes: {codes} omop_ids: {omop_ids}")
        for i, label in enumerate(labels):
            # label = normalize_text(label)
            code = codes[i].strip() if codes[i] else None
            omop_id = omop_ids[i] if omop_ids[i] else None
                # Create a unique URI for the context except time point
            if label and code and omop_id:
                context_label =  f"{context_label};;{label}"
                concepts.append(Concept(
                    standard_label=label,
                    code=code,
                    omop_id=omop_id,
                ))
        # uri_label = f"contextual_factor/{context_label}"
        context_uri = get_context_uri(var_uri=var_uri, context_id=context_label)
        g.add((context_uri, RDF.type, OntologyNamespaces.CMEO.value.contextual_factor, cohort_uri))
        g.add((context_uri, OntologyNamespaces.IAO.value.is_about, var_uri, cohort_uri))
        g.add((context_uri, RDFS.label, Literal(context_label, datatype=XSD.string), cohort_uri))
        g.add((context_uri, OntologyNamespaces.CMEO.value.has_value, Literal(context_label, datatype=XSD.string), cohort_uri))
        g=add_concept_info(g, context_uri, concepts, cohort_uri)
    return  g


def add_concept_info(g: Graph, linked_uri: URIRef, concepts: list[Concept], cohort_uri: URIRef) -> Graph:

    # print(f"base_concept: {base_concept} additional_concepts: {additional_concepts}")
    # data standardization is a process with has_specified_input data item and data item is output is code which has part standard label and omop id
    data_standardization_uri = URIRef(f"{linked_uri}/data_standardization")
    g.add((data_standardization_uri, RDF.type, OntologyNamespaces.CMEO.value.data_standardization,cohort_uri))
    g.add((linked_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_standardization_uri,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_input, linked_uri,cohort_uri))

    code_set_uri = URIRef(f"{linked_uri}/code_set")
    g.add((code_set_uri, RDF.type, OntologyNamespaces.CMEO.value.code_set,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_output, code_set_uri,cohort_uri))
    g.add((code_set_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_standardization_uri,cohort_uri))


    for i, concept in enumerate(concepts):
        code = concept.code.strip()
        label = concept.standard_label.strip().replace("\n","")
        omop_id = concept.omop_id
        if not code or not label or not omop_id:
            continue
        code_uri = get_code_uri(linked_uri, code)
        g.add((code_uri, RDF.type, OntologyNamespaces.CMEO.value.code,cohort_uri))
        g.add((code_set_uri, OntologyNamespaces.RO.value.has_part, code_uri,cohort_uri))
        g.add((code_uri, OntologyNamespaces.RO.value.is_part_of, code_set_uri,cohort_uri))
        g.add((code_uri, RDFS.label, Literal(label, datatype=XSD.string),cohort_uri))
        g.add((code_uri, OntologyNamespaces.CMEO.value.has_value, Literal(code, datatype=XSD.string),cohort_uri))
        omop_id_uri = get_omop_id_uri(linked_uri, omop_id)
        g.add((omop_id_uri, RDF.type, OntologyNamespaces.CMEO.value.omop_id,cohort_uri))
        g.add((code_uri, OntologyNamespaces.IAO.value.denotes, omop_id_uri,cohort_uri))
        # g.add((standard_label_uri, OntologyNamespaces.CMEO.value.has_value, Literal(label, datatype=XSD.string),cohort_uri))
        # g.add((code_uri, OntologyNamespaces.CMEO.value.has_value, Literal(code, datatype=XSD.string),cohort_uri))
        print(f"omop_id: {omop_id}")
        g.add((omop_id_uri, OntologyNamespaces.CMEO.value.has_value, Literal(omop_id, datatype=XSD.integer),cohort_uri))
    
    # print(f"omop_id: {omop_id} for {linked_uri}")
    return g



def add_composite_concepts_info(g: Graph, linked_uri: URIRef, concepts: list[Concept], cohort_uri: URIRef) -> Graph:

    data_standardization_uri = URIRef(f"{linked_uri}/data_standardization")
    g.add((data_standardization_uri, RDF.type, OntologyNamespaces.CMEO.value.data_standardization,cohort_uri))
    g.add((linked_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_standardization_uri,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_input, linked_uri,cohort_uri))

    code_set_uri = URIRef(f"{linked_uri}/code_set")

    g.add((code_set_uri, RDF.type, OntologyNamespaces.CMEO.value.code,cohort_uri))
    g.add((code_set_uri, RDF.type, RDF.Seq, cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_output, code_set_uri,cohort_uri))
    g.add((code_set_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_standardization_uri,cohort_uri))
    # print(linked_uri)
    for i, concept in enumerate(concepts):
        # print(concept)
        if concept.code is None or concept.standard_label is None or concept.omop_id is None:
            continue
        code = concept.code.strip()
        # code_only = code.split(":")[-1]
        # code_only_encoded = quote(code_only, safe='')
        code_uri = create_code_uri(code=code, cohort_uri=cohort_uri)
        label = concept.standard_label.strip().replace("\n","")
        omop_id = concept.omop_id
        g.add((code_uri, RDFS.label, Literal(label, datatype=XSD.string),cohort_uri))
        g.add((code_uri, OntologyNamespaces.CMEO.value.has_value, Literal(code, datatype=XSD.string),cohort_uri))
        omop_id_uri = URIRef(f"{OntologyNamespaces.OMOP.value}{omop_id}")
        g.add((omop_id_uri, RDF.type, OntologyNamespaces.CMEO.value.omop_id,cohort_uri))
        g.add((code_uri, OntologyNamespaces.IAO.value.denotes, omop_id_uri,cohort_uri))
    
        g.add((omop_id_uri, OntologyNamespaces.CMEO.value.has_value, Literal(omop_id, datatype=XSD.integer),cohort_uri))

        g.add((code_set_uri, RDF[f"_{i+1}"], code_uri, cohort_uri))
    # print(f"omop_id: {omop_id} for {linked_uri}")
    return g



def add_solo_concept_info(g: Graph, linked_uri: URIRef, concept: Concept, cohort_uri: URIRef) -> Graph:


    if not concept.code or not concept.standard_label or not concept.omop_id:
        return g
   
    data_standardization_uri = URIRef(f"{linked_uri}/data_standardization")
    g.add((data_standardization_uri, RDF.type, OntologyNamespaces.CMEO.value.data_standardization,cohort_uri))
    g.add((linked_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_standardization_uri,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_input, linked_uri,cohort_uri))

   
    
    # code_set_uri = URIRef(f"{linked_uri}/code_set")
    # g.add((code_set_uri, RDF.type, OntologyNamespaces.CMEO.value.code_set,cohort_uri))
    # g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_output, code_set_uri,cohort_uri))
    # g.add((code_set_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_standardization_uri,cohort_uri))


 
    # code = normalize_text(concept.code.strip())
    # label = concept.standard_label.strip().replace("\n","")
    # omop_id = concept.omop_id
    if concept.code is None or concept.standard_label is None or concept.omop_id is None:
        return g
    code = concept.code.strip()
    # code_only = code.split(":")[-1]
    # code_only_encoded = quote(code_only, safe='')
    label = concept.standard_label.strip().replace("\n","")
    omop_id = concept.omop_id
    code_uri = create_code_uri(code, cohort_uri)
 
    g.add((code_uri, RDF.type, OntologyNamespaces.CMEO.value.code,cohort_uri))
    g.add((code_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_standardization_uri,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_output, code_uri,cohort_uri))
    g.add((code_uri, RDFS.label, Literal(label, datatype=XSD.string),cohort_uri))
    # standard_label_uri = get_standard_label_uri(linked_uri, label)
    # g.add((standard_label_uri, RDF.type, OntologyNamespaces.CMEO.value.standard_label,cohort_uri))
    # g.add((code_uri, OntologyNamespaces.OBI.value.is_denoted_by, standard_label_uri,cohort_uri))
    
  
    # omop_id_uri = get_omop_id_uri(linked_uri, omop_id)
    # g.add((omop_id_uri, RDF.type, OntologyNamespaces.CMEO.value.omop_id,cohort_uri))
    # g.add((code_uri, OntologyNamespaces.IAO.value.denotes, omop_id_uri,cohort_uri))
    # g.add((standard_label_uri, OntologyNamespaces.CMEO.value.has_value, Literal(label, datatype=XSD.string),cohort_uri))
    g.add((code_uri, OntologyNamespaces.CMEO.value.has_value, Literal(code, datatype=XSD.string),cohort_uri))
    omop_id_uri = URIRef(f"{OntologyNamespaces.OMOP.value}{omop_id}")
    g.add((omop_id_uri, RDF.type, OntologyNamespaces.CMEO.value.omop_id,cohort_uri))
    g.add((code_uri, OntologyNamespaces.IAO.value.denotes, omop_id_uri,cohort_uri))
    g.add((omop_id_uri, OntologyNamespaces.CMEO.value.has_value, Literal(omop_id, datatype=XSD.integer),cohort_uri))
    
    # print(f"omop_id: {omop_id} for {linked_uri}")
    return g

def create_code_uri(code:str, cohort_uri: URIRef) -> URIRef:
    code_only = code.split(":")[-1]
    code_only_encoded = quote(code_only, safe='')
    if 'snomed' in code or 'snomedct' in code:
        code_uri = URIRef(f"{OntologyNamespaces.SNOMEDCT.value}{code_only_encoded}")
    elif 'loinc' in code:
        code_uri = URIRef(f"{OntologyNamespaces.LOINC.value}{code_only_encoded}")
    elif 'ucum' in code:
        code_uri = URIRef(f"{OntologyNamespaces.UCUM.value}{code_only_encoded}")
    elif 'rxnorm' in code:
        code_uri = URIRef(f"{OntologyNamespaces.RXNORM.value}{code_only_encoded}")
    elif 'atc' in code:
        code_uri = URIRef(f"{OntologyNamespaces.ATC.value}{code_only_encoded}")
    elif 'omop' in code:
        code_uri = URIRef(f"{OntologyNamespaces.OMOP.value}{code_only_encoded}")
    else:
        code_uri = URIRef(f"{cohort_uri}/{code_only_encoded}")
    return code_uri
    
def add_raw_data_graph(cohort_data_file_path, cohort_name) -> Graph:
    try:

        # Read header to prepare normalized column names (skip patient identifier column)
        header_data = pd.read_csv(cohort_data_file_path, nrows=0, low_memory=False)
        header_data = header_data.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))

        n_cohort_name = normalize_text(cohort_name)
        cohort_uri= f"{settings.sparql_endpoint}/rdf-graphs/{n_cohort_name}"
        print(f"cohort_uri: {cohort_uri}")
        normalized_columns = {col: normalize_text(col) for col in header_data.columns[1:]}
        var_uris = {col: get_var_uri(n_cohort_name, normalized_columns[col]) for col in normalized_columns}
        # print(f"\n top ten normalized columns: {list(normalized_columns.items())[:10]}")
        # print(f"\n top ten var_uris: {list(var_uris.items())[:10]}")
        var_exists = {col: variable_exists(cohort_uri, normalized_columns[col]) for col in normalized_columns}
       
        # Initialize the RDF graph
        cohort_graph = URIRef(OntologyNamespaces.CMEO.value + f"graph/{cohort_name}_pldata")
        g = init_graph(default_graph_identifier=cohort_graph)
        rows = 0
        chunk_size = 100

        # Process CSV file in chunks
        for chunk in pd.read_csv(cohort_data_file_path, chunksize=chunk_size, low_memory=False):
            for i, row in chunk.iterrows():
                # Extract and normalize the patient identifier (first column)
                patient_id = normalize_text(row.iloc[0])
                if not patient_id:
                    print(f"Skipping row {i}: Missing patient ID.")
                    continue

                # Create unique URIs for the patient and participant identifier
                # identifier_uri =  URIRef(OntologyNamespaces.OBI.value + f"participant_identifier/{patient_id}")
                participant_under_investigation_role_uri = URIRef(OntologyNamespaces.OBI.value + f"participant_under_investigation_role/{patient_id}")
                person_uri = URIRef(OntologyNamespaces.CMEO.value + f"person/{patient_id}")
                # g.add((identifier_uri, RDF.type, OntologyNamespaces.CMEO.value.participant_identifier,cohort_graph))
                # g.add((identifier_uri, OntologyNamespaces.CMEO.value.has_value, Literal(patient_id, datatype=XSD.string),cohort_graph))
                # particpant identifier denotes person who has role of participant under investigation role which concreatizes data item and data item is instantiated in data element
                
                g.add((person_uri, RDF.type, OntologyNamespaces.CMEO.value.person,cohort_graph))
                # g.add((person_uri, OntologyNamespaces.CMEO.value.has_identifier, identifier_uri,cohort_graph))
               
                g.add((participant_under_investigation_role_uri, RDF.type, OntologyNamespaces.CMEO.value.participant_under_investigation_role,cohort_graph))
                g.add((person_uri, OntologyNamespaces.CMEO.value.has_role, participant_under_investigation_role_uri,cohort_graph))
                g.add((participant_under_investigation_role_uri, OntologyNamespaces.CMEO.value.role_of, person_uri,cohort_graph))
                g.add((participant_under_investigation_role_uri, DC.identifier, Literal(row.iloc[0], datatype=XSD.string),cohort_graph))

                # Process each variable column (skip the first column)
                for col_name in chunk.columns[1:]:
                    # print(row[col_name])
                    var_name = normalized_columns.get(col_name)
                    # print(var_exists.get(col_name, False))
                    var_value = row[col_name]
                    # print(f"var_name: {var_name} var_value: {var_value}")
                    if not var_name or var_exists.get(var_name, False) == False:
                        # print(f"Skipping row {i}: Missing variable name or value.")
                        continue
                    print(f"Processing data point for {var_name} with value {var_value}")
                    # Create a unique URI for this data point
                    data_point_uri = URIRef(OntologyNamespaces.CMEO.value + f"data_point/{patient_id}/{var_name}")
                    dataset_uri = URIRef(f"{var_uris[col_name]}/data_set")
                    # Add triples for this data point
                    g.add((data_point_uri, RDF.type, OntologyNamespaces.OBI.value.measurement_datum,cohort_graph))
                    g.add((data_point_uri, OntologyNamespaces.RO.value.is_part_of, dataset_uri,cohort_graph))
                    g.add((data_point_uri, OntologyNamespaces.IAO.value.is_about, person_uri,cohort_graph))
                    g.add((data_point_uri, OntologyNamespaces.CMEO.value.has_value, Literal(var_value, datatype=XSD.string),cohort_graph))
                    rows += 1

        print(f"Processed {rows} data points.")
        return g    
    except Exception as e:
        print(f"Error processing raw data file: {e}")
        return None
       
        # base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # print(f"Base path: {base_path}")
        # Serialize the graph to a temporary file and post it to the SPARQL endpoint
    #     temp_file = f"{output_dir}/{cohort_name}_raw_data_graph.ttl"
    #     serialized_graph = g.serialize(format="trig")
    #     with open(temp_file, "w", encoding="utf-8") as f:
    #         f.write(serialized_graph)

    #     print(f"Serialized raw data graph to {temp_file}")

    #     headers = {"Content-Type": "application/trig"}
    #     response = requests.post(f"{settings.sparql_endpoint}/store?graph={cohort_uri}_pldata",
    #                                 headers=headers, data=serialized_graph, timeout=300)

    #     if response.status_code in (201, 204):
    #         print("Raw data successfully added to the graph.")
    #     else:
    #         print(f"Failed to publish raw data graph: {response.status_code}, {response.text}")

    # except Exception as e:
    #     print(f"Error processing raw data file: {e}")
