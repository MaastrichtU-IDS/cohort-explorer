
from rdflib.namespace import  XSD
from rdflib import Graph,Literal, RDF, RDFS, URIRef, DC

import pandas as pd
from .config import settings
import json
from dataclasses import dataclass
from .study_kg import update_metadata_graph
from .utils import (
    OntologyNamespaces,
    init_graph, 
    is_categorical_variable, 
    normalize_text, 
    safe_int,
    determine_var_uri,
    is_identifier_like_variable,
    get_study_uri,
    load_dictionary,
    extract_tick_values,
    parse_joined_string,
    create_code_uri,
    split_categories
)
from typing import Optional, Any


@dataclass
class Concept:
    standard_label: Optional[str] = None
    code: Optional[str] = None
    omop_id: Optional[int] = None
    
def add_category_annotation(g: Graph, variable_uri: URIRef, category: str, cohort_uri:URIRef) -> Graph:
    if category is None:
        return g
    category_ = category.strip().lower().replace(' ','_')
    category_uri = URIRef(f"{variable_uri}/category/{category_}")
    g.add((category_uri, RDF.type, OntologyNamespaces.CMEO.value.category, cohort_uri))
    g.add((variable_uri, OntologyNamespaces.SIO.value.has_annotation, category_uri, cohort_uri))
    g.add((category_uri, OntologyNamespaces.CMEO.value.has_value, Literal(category_), cohort_uri))
    return g

def process_variables_metadata_file(file_path:str, study_metadata_graph_file_path:str, cohort_name:str,eda_file_path:str) -> tuple[Graph, str]:

        print(f"Processing metadata file: {file_path}")
        file_path_dir = file_path.rsplit('/', 1)[0]
        data = load_dictionary(file_path)
        if data is None or data.empty:
            return None, None   
        data.columns = data.columns.str.lower()
        print(f"colums: {data.columns}")
        cohort_id = normalize_text(cohort_name)
   
        cohort_uri = get_study_uri(cohort_id)
        data = data.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))
        cohort_graph = URIRef(OntologyNamespaces.CMEO.value + f"graph/{cohort_id}") #named graph for the cohort
        print(f"Processing cohort: {cohort_graph}")
        g = init_graph(default_graph_identifier=cohort_graph)
        # convert all data into lower case
        
        count = 0
        binary_categorical, multi_class_categorical = is_categorical_variable(data)
        variables_to_update = {}
        # units_count = 0
        for _, row in data.iterrows():
            count += 1
            if pd.isna(row['variablename']):
                print("Skipping row with missing variable name.")
                continue
            # Instantiate MeasuredVariable
            var_name = normalize_text(row['variablename']) 
            
            print(f"Processing variable: {var_name}")
            var_uri = get_var_uri(cohort_id, var_name)
            g.add((var_uri, RDF.type, OntologyNamespaces.CMEO.value.data_element, cohort_graph))
           
            # data_element_domain = row['omop'].lower().strip().replace(' ','_') if pd.notna(row['omop']) else None
                # Corrected URI assignment
            if pd.notna(row['domain']):
                g=add_category_annotation(g, var_uri, row['domain'], cohort_graph)
            print(f"row['vartype']: {row['vartype']}")
            # statistical_type_uri,statistical_type = determine_var_uri(cohort_id, var_name, multi_class_categorical, binary_categorical, data_type=row['vartype'])
            
            is_identifier, reasons = is_identifier_like_variable(row)

            if is_identifier:
                print(
                    f"[IDENTIFIER] {row.get('variablename')} | "
                    f"{row.get('variablelabel')} | "
                    f"{row.get('variable concept name')} | "
                    f"{'; '.join(reasons)}"
                )

                statistical_type = "qualitative_variable"
                statistical_type_uri = URIRef(f"{var_uri}/statistical_type/{statistical_type}")

            else:
                statistical_type_uri, statistical_type = determine_var_uri(
                    cohort_id,
                    var_name,
                    multi_class_categorical,
                    binary_categorical,
                    data_type=row["vartype"],
                    unit = row['units'] if pd.notna(row['units']) else None
                )
            g.add((statistical_type_uri, RDF.type, URIRef(f"{OntologyNamespaces.CMEO.value}{statistical_type}"), cohort_graph))
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
                    count1 = len(parse_joined_string(row['additional context concept name']))
                    count2 = len(str(row['additional context concept code']).split("|"))
                    count3 = len(str(row['additional context omop id']).split("|"))
                    parsed_concept_names = parse_joined_string(row['additional context concept name'])
                    if count1 == count2 == count3:
                        base_concept.extend([Concept(
                        standard_label=parsed_concept_names[i] if pd.notna(parsed_concept_names[i]) and parsed_concept_names[i] != "na" else None,
                        code=str(row['additional context concept code']).split("|")[i] if pd.notna(row['additional context concept code']) else None,
                        omop_id=safe_int(row['additional context omop id'].split("|")[i]) if pd.notna(row['additional context omop id']) else None,
                    ) for i in range(count1)])
                    else:
                        print(f"Row number {var_name} of {cohort_name} has an unequal number of additional context concept names/codes/omop ids.")
                except:
                    print(f"Row number {var_name} of {cohort_name} does not have a valid string in additional context concept names/codes/omop ids.")
            g = add_data_type(g, var_uri, row['vartype'], cohort_graph)
            g = add_composite_concepts_info(g, var_uri, base_concept, cohort_graph)
            
            g=add_temporal_context(g=g, var_uri=var_uri, cohort_uri=cohort_graph, row=row)

            g=add_categories_to_graph(g=g, var_uri=var_uri, cohort_uri=cohort_graph,row=row)

            g=add_measurement_unit(g, var_uri=var_uri, cohort_uri=cohort_graph, row=row)
            add_missing_value_specification(g, var_uri, row['missing'], cohort_graph)
            
            g = add_formula(g, var_uri, row['formula'], cohort_graph)
            
            study_variable_design_specification_uri = URIRef(cohort_uri + "/study_design_variable_specification")
            g.add((study_variable_design_specification_uri, RDF.type, OntologyNamespaces.CMEO.value.study_design_variable_specification, cohort_graph))
            g.add((var_uri, OntologyNamespaces.RO.value.is_part_of, study_variable_design_specification_uri, cohort_graph))
            

            g , dataset_uri= add_dataset_uri(g, row, var_uri, statistical_type_uri, variables_to_update, cohort_graph)
            variables_to_update.update({row['variablename']: (var_uri, statistical_type_uri, dataset_uri)})

           
        vars_list = [var_uri for var_uri, _,_ in variables_to_update.values()]
        # print(f"vars_list: {vars_list}")
        update_metadata_graph(endpoint_url=settings.update_endpoint, cohort_uri=cohort_uri, variable_uris=vars_list,metadata_graph_path=study_metadata_graph_file_path)
        if eda_file_path:
            g=add_variable_eda(g, variables_to_update, cohort_graph, eda_json_file_path=eda_file_path)

        print(f"Processed {count} rows")
        # print(f"Processed {units_count} units")
        return g, cohort_id


def collect_endpoint_variables(g: Graph, cohort_uri: URIRef) -> list[URIRef]:
    """Collect all variables whose label, identifier, or concept name contains 'endpoint'."""
    endpoint_vars = []
    for var_uri in g.subjects(RDF.type, OntologyNamespaces.CMEO.value.data_element, cohort_uri):
        texts = []
        # variable label
        for o in g.objects(var_uri, RDFS.label, cohort_uri):
            texts.append(str(o))
        # variable identifier
        for o in g.objects(var_uri, DC.identifier, cohort_uri):
            texts.append(str(o))
        # concept labels via code_set → code → rdfs:label
        for code_set in g.objects(var_uri, OntologyNamespaces.SKOS.value.closeMatch, cohort_uri):
            for code in g.objects(code_set, OntologyNamespaces.RO.value.has_part, cohort_uri):
                for lbl in g.objects(code, RDFS.label, cohort_uri):
                    texts.append(str(lbl))
        if any('endpoint' in t.lower() for t in texts):
            endpoint_vars.append(var_uri)
    return endpoint_vars
    

def add_dataset_uri(g: Graph, row:pd.Series, var_uri: URIRef, statistical_type_uri: URIRef, variables_to_update: dict, cohort_uri: URIRef) -> Graph:
    
    dataset_uri = URIRef(f"{var_uri}/dataset")
    g.add((dataset_uri, RDF.type, OntologyNamespaces.IAO.value.dataset, cohort_uri))
    g.add((dataset_uri, OntologyNamespaces.IAO.value.is_about,statistical_type_uri , cohort_uri))
    
    g = add_device_senors_for_variable(g=g, var_uri=var_uri, data_set_uri=dataset_uri, cohort_uri=cohort_uri, row_info=row)
    return g, dataset_uri

def add_missing_value_specification(g: Graph, var_uri: URIRef, missing_value: str, cohort_uri: URIRef) -> Graph:
    if missing_value is None or pd.isna(missing_value):
        return g

    missing_uri = URIRef(f"{var_uri}/missing_value_specification")
    g.add((missing_uri, RDF.type, OntologyNamespaces.CMEO.value.missing_value_specification, cohort_uri))
    g.add((missing_uri, OntologyNamespaces.CMEO.value.has_value, Literal(missing_value, datatype=XSD.string), cohort_uri))
    # g.add((var_uri, OntologyNamespaces.OBI.value.has_value_specification, missing_uri, cohort_uri))
    g.add((missing_uri, OntologyNamespaces.OBI.value.specifies_value_of, var_uri, cohort_uri))
    g.add((var_uri, OntologyNamespaces.OBI.value.has_value_specification, missing_uri, cohort_uri))
    return g

def add_formula(g: Graph, var_uri: URIRef, formula: str, cohort_uri: URIRef) -> Graph:
    if formula is None or pd.isna(formula):
        return g
    formula_uri = URIRef(f"{var_uri}/formula")
    g.add((formula_uri, RDF.type, OntologyNamespaces.CMEO.value.formula, cohort_uri))
    g.add((formula_uri, OntologyNamespaces.CMEO.value.has_value, Literal(formula, datatype=XSD.string), cohort_uri))
    g.add((formula_uri, OntologyNamespaces.SIO.value.is_attribute_of, var_uri, cohort_uri))
    g.add((var_uri, OntologyNamespaces.SIO.value.has_attribute, formula_uri, cohort_uri))
    return g
def add_data_type(g: Graph, var_uri: URIRef, data_type: str, cohort_uri: URIRef) -> Graph:
    if data_type is None or pd.isna(data_type):
        return g
    data_type_uri = URIRef(f"{var_uri}/data_type/{data_type}")
    g.add((data_type_uri, RDF.type, OntologyNamespaces.CMEO.value.data_type, cohort_uri))
    g.add((data_type_uri, OntologyNamespaces.CMEO.value.has_value, Literal(data_type, datatype=XSD.string), cohort_uri))
    g.add((data_type_uri, OntologyNamespaces.SIO.value.is_attribute_of, var_uri, cohort_uri))
    g.add((var_uri, OntologyNamespaces.SIO.value.has_attribute, data_type_uri, cohort_uri))
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
                    # statistical_var_uri = var_uris.get(var_name)[1]
                    dataset_uri = var_uris.get(var_name)[2]
                    transformed_data_uri = URIRef(f"{var_uri}/exploratory_data_analysis")
                    g.add((transformed_data_uri, RDF.type, OntologyNamespaces.CMEO.value.exploratory_data_analysis, cohort_uri))
                    g.add((dataset_uri, OntologyNamespaces.OBI.value.is_specified_input_of, transformed_data_uri, cohort_uri))
                    g.add((transformed_data_uri, OntologyNamespaces.OBI.value.has_specified_input, dataset_uri, cohort_uri))
                    statistic_uri = URIRef(f"{var_uri}/statistic")
                    g.add((statistic_uri, RDF.type, OntologyNamespaces.STATO.value.statistic, cohort_uri))
                    g.add((statistic_uri, OntologyNamespaces.IAO.value.is_about, dataset_uri, cohort_uri))
                    g.add((transformed_data_uri, OntologyNamespaces.OBI.value.has_specified_output, statistic_uri, cohort_uri))
                    g.add((statistic_uri, OntologyNamespaces.OBI.value.is_specified_output_of, transformed_data_uri, cohort_uri))
                    g = add_count_to_graph(g, statistic_uri, eda.get('count (metadata dictionary)', None), cohort_uri)
                    g = add_min_value_to_graph(g, statistic_uri, eda.get('min', eda.get('min (metadata dictionary)', None)), cohort_uri)
                    g = add_max_value_to_graph(g, statistic_uri, eda.get('max', eda.get('max (metadata dictionary)', None)), cohort_uri)
                    g = add_missing_value_count_to_graph(g, statistic_uri, eda.get('count empty', None), cohort_uri)
                    g = add_unique_values_count_to_graph(g, statistic_uri, eda.get('number of unique values/categories', None), cohort_uri)
                    print(f"variable type: {eda.get('type', None)} for variable {var_name}")   
                    if eda.get('type', None) is None:
                        # print(f"Type is None for variable {var_name}")
                        continue
                    else:
                        if 'categorical' in eda.get('type', None).strip().lower():
                            

                            g = add_frequency_distribution_to_graph(g, statistic_uri, eda.get('class balance', None), cohort_uri)
                            g = add_mode_to_graph(g, statistic_uri, eda.get('most frequent category', None), cohort_uri)
                            g = add_chi_square_test_statistic_to_graph(g, var_uri, dataset_uri, eda.get('chi-square test statistic', None), cohort_uri)
                            g = add_categorical_variable_visualization(g, var_uri, cohort_uri, eda.get('url', None), [eda.get('x-ticks', None), eda.get('y-ticks', None)])
                        else:
                            g = add_histogram_visualization(g, var_uri=var_uri, cohort_uri=cohort_uri, chart_url=eda.get('url', None), xy_axis=[eda.get('x-ticks', None), eda.get('y-ticks', None)])
                            g = add_outlier_count_by_iqr_to_graph(g, statistic_uri, eda.get('outliers (iqr)', None), cohort_uri)
                            g = add_outlier_count_by_z_score_to_graph(g, statistic_uri, eda.get('outliers (z)', None), cohort_uri)
                            g = add_normality_test_to_graph(g, statistic_uri, eda.get('normality test', None), cohort_uri)
                            g = add_wilks_shapiro_test_to_graph(g, statistic_uri, eda.get('w_test', None), cohort_uri)
                            g = add_standard_deviation_to_graph(g, statistic_uri, eda.get('Std Dev', None), cohort_uri)
                            g = add_mean_to_graph(g, statistic_uri, eda.get('mean', None), cohort_uri)
                            g = add_median_to_graph(g, statistic_uri, eda.get('median', None), cohort_uri)
                            g = add_iqr_to_graph(g, statistic_uri, eda.get('iqr', None), cohort_uri)
                            g = add_q1_to_graph(g, statistic_uri, eda.get('q1', None), cohort_uri)
                            g = add_q3_to_graph(g, statistic_uri, eda.get('q3', None), cohort_uri)
                            g = add_variance_to_graph(g, statistic_uri, eda.get('variance', None), cohort_uri)
                            g = add_kurtosis_to_graph(g, statistic_uri, eda.get('kurtosis', None), cohort_uri)
                            g = add_skewness_to_graph(g, statistic_uri, eda.get('skewness', None), cohort_uri)

    return g
                            
def add_numeric_statistic_generic(
    g: Graph, 
    statistic_uri: URIRef, 
    value: Any, 
    stat_rdf_type: URIRef, 
    uri_suffix: str, 
    cohort_uri: URIRef
) -> Graph:
    """
    Generic function to add a numeric statistic to the graph.
    Checks if value is string and convertible to float/int before adding.
    """
    if value is None or pd.isna(value):
        return g

    final_value = None
    xsd_type = None
    
    # 1. Check logic: if str -> try convert to int, then float.
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "nan" or value == "":
            return g
            
        try:
            # Try converting to integer first
            final_value = int(value)
            xsd_type = XSD.integer
        except ValueError:
            try:
                # Try converting to float
                final_value = float(value)
                xsd_type = XSD.float
            except ValueError:
                # Not convertible, return without adding triples
                return g
    
    # Handle cases where input might already be numeric (optional safety)
    elif isinstance(value, int):
        final_value = value
        xsd_type = XSD.integer
    elif isinstance(value, float):
        final_value = value
        xsd_type = XSD.float
    else:
        return g

    # 2. Create the specific URI (e.g., .../average_value)
    specific_stat_uri = URIRef(f"{statistic_uri}/{uri_suffix}")

    # 3. Add Triples
    # Add Type
    g.add((specific_stat_uri, RDF.type, stat_rdf_type, cohort_uri))
    
    # Add Value
    g.add((specific_stat_uri, OntologyNamespaces.CMEO.value.has_value, Literal(final_value, datatype=xsd_type), cohort_uri))
    
    # Link to Parent Statistic (Bidirectional)
    g.add((statistic_uri, OntologyNamespaces.RO.value.has_part, specific_stat_uri, cohort_uri))
    g.add((specific_stat_uri, OntologyNamespaces.RO.value.is_part_of, statistic_uri, cohort_uri))

    return g                  
def add_count_to_graph(g: Graph, statistic_uri: URIRef, count: any, cohort_uri: URIRef) -> Graph:
 
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=count,
        stat_rdf_type=OntologyNamespaces.STATO.value.count_,
        uri_suffix="count_",
        cohort_uri=cohort_uri
    )

def add_min_value_to_graph(g: Graph, statistic_uri: URIRef, min_value: any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=min_value,
        stat_rdf_type=OntologyNamespaces.STATO.value.minimum_value,
        uri_suffix="minimum_value",
        cohort_uri=cohort_uri
    )


def add_max_value_to_graph(g: Graph, statistic_uri: URIRef, max_value: any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=max_value,
        stat_rdf_type=OntologyNamespaces.STATO.value.maximum_value,
        uri_suffix="maximum_value",
        cohort_uri=cohort_uri
    )

def add_missing_value_count_to_graph(g: Graph, statistic_uri: URIRef, na: int, cohort_uri: URIRef) -> Graph:

    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=na,
        stat_rdf_type=OntologyNamespaces.CMEO.value.missing_values_count,
        uri_suffix="number_of_missing_values",
        cohort_uri=cohort_uri
    )

def add_missing_value_percentage_to_graph(g: Graph, statistic_uri: URIRef, missing_value_percentage: float, cohort_uri: URIRef) -> Graph:
   
    
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=missing_value_percentage,
        stat_rdf_type=OntologyNamespaces.CMEO.value.missing_values_percentage,
        uri_suffix="missing_value_percentage",
        cohort_uri=cohort_uri
    )


def add_unique_values_count_to_graph(g: Graph, statistic_uri: URIRef, unique_values: any, cohort_uri: URIRef) -> Graph:

    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=unique_values,
        stat_rdf_type=OntologyNamespaces.CMEO.value.unique_values_count,
        uri_suffix="number_of_unique_values",
        cohort_uri=cohort_uri
    )

def add_device_senors_for_variable(g: Graph, var_uri:URIRef, data_set_uri: URIRef,  cohort_uri: URIRef, row_info:pd.Series) -> Graph:
    
    data_collection_process_uri = URIRef(f"{var_uri}/data_collection")
    g.add((data_collection_process_uri, RDF.type, OntologyNamespaces.OBCS.value.data_collection, cohort_uri))
    g.add((data_set_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_collection_process_uri, cohort_uri))
    g.add((data_collection_process_uri, OntologyNamespaces.OBI.value.has_specified_output, data_set_uri, cohort_uri))
    if 'device' in row_info and pd.notna(row_info['device']):
        device_value = row_info['device'].strip().lower().replace(' ','_')
        device_uri = URIRef(OntologyNamespaces.CMEO.value + f"wearable_device/{device_value}")
        g.add((device_uri, RDF.type, OntologyNamespaces.CMEO.value.wearable_device, cohort_uri))
        g.add((device_uri, OntologyNamespaces.CMEO.value.has_value, Literal(device_value, datatype=XSD.string), cohort_uri))  
        g.add((data_collection_process_uri, OntologyNamespaces.OBI.value.has_specified_input, device_uri, cohort_uri))
        g.add((device_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_collection_process_uri, cohort_uri))
        if 'sensor' in row_info and pd.notna(row_info['sensor']):
            sensor_value = row_info['sensor'].strip().lower().replace(' ','_')
            sensor_uri =URIRef(OntologyNamespaces.CMEO.value + f"sensor/{sensor_value}")
            g.add((sensor_uri, RDF.type, OntologyNamespaces.CMEO.value.sensor, cohort_uri))
            g.add((sensor_uri, OntologyNamespaces.CMEO.value.has_value, Literal(sensor_value, datatype=XSD.string), cohort_uri))
            g.add((device_uri, OntologyNamespaces.RO.value.has_part, sensor_uri, cohort_uri))
            g.add((sensor_uri, OntologyNamespaces.RO.value.is_part_of, device_uri, cohort_uri))
            if 'wearer location' in row_info and pd.notna(row_info['wearer location']):
                wearer_location_value = row_info['wearer location'].strip().lower().replace(' ', '_')
                wearer_location_uri = URIRef(OntologyNamespaces.CMEO.value + f"body_region/{wearer_location_value}")
                g.add((sensor_uri, OntologyNamespaces.RO.value.is_located_in, wearer_location_uri, cohort_uri))
                g.add((wearer_location_uri, RDF.type, OntologyNamespaces.CMEO.value.body_region, cohort_uri))
                g.add((data_collection_process_uri, OntologyNamespaces.RO.value.occurs_in, wearer_location_uri, cohort_uri))
                g.add((wearer_location_uri, OntologyNamespaces.CMEO.value.has_value, Literal(wearer_location_value, datatype=XSD.string), cohort_uri))

    return g

def add_frequency_distribution_to_graph(g: Graph, statistic_uri: URIRef, fd:str, cohort_uri: URIRef) -> Graph:

    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=fd,
        stat_rdf_type=OntologyNamespaces.OBCS.value.frequency_distribution,
        uri_suffix="frequency_distribution",
        cohort_uri=cohort_uri
    )


def add_mode_to_graph(g: Graph, statistic_uri: URIRef, mode: any, cohort_uri: URIRef) -> Graph:

    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=mode,
        stat_rdf_type=OntologyNamespaces.STATO.value.mode,
        uri_suffix="mode",
        cohort_uri=cohort_uri
    )

def add_chi_square_test_statistic_to_graph(g: Graph, var_uri:URIRef, dataset_uri: URIRef, chi_square_test_statistic: Any, cohort_uri: URIRef) -> Graph:
  
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=var_uri,
        value=chi_square_test_statistic,
        stat_rdf_type=OntologyNamespaces.STATO.value.chi_square_test_statistic,
        uri_suffix="chi_square_test_value",
        cohort_uri=cohort_uri
    )


# add standard deviation to the graph
def add_standard_deviation_to_graph(g: Graph, statistic_uri: URIRef, std_dev: Any, cohort_uri: URIRef) -> Graph:
  
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=std_dev,
        stat_rdf_type=OntologyNamespaces.STATO.value.standard_deviation,
        uri_suffix="standard_deviation",
        cohort_uri=cohort_uri
    )

# add mean to the graph as average_value 
def add_mean_to_graph(g: Graph, statistic_uri: URIRef, mean: Any, cohort_uri: URIRef) -> Graph:
   
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=mean,
        stat_rdf_type=OntologyNamespaces.STATO.value.mean,
        uri_suffix="average_value",
        cohort_uri=cohort_uri
    )

# add median to the graph
def add_median_to_graph(g: Graph, statistic_uri: URIRef, median: Any, cohort_uri: URIRef) -> Graph:
   
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=median,
        stat_rdf_type=OntologyNamespaces.STATO.value.median,
        uri_suffix="median",
        cohort_uri=cohort_uri
    )


# add iqr  as interquartile range to the graph

def add_iqr_to_graph(g: Graph, statistic_uri: URIRef, iqr: Any, cohort_uri: URIRef) -> Graph:
   
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=iqr,
        stat_rdf_type=OntologyNamespaces.STATO.value.interquartile_range,
        uri_suffix="interquartile_range",
        cohort_uri=cohort_uri
    )
# add Q1 to the graph
def add_q1_to_graph(g: Graph, statistic_uri: URIRef, q1: Any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=q1,
        stat_rdf_type=OntologyNamespaces.STATO.value.first_quartile,
        uri_suffix="first_quartile",
        cohort_uri=cohort_uri
    )
# add Q3 to the graph
def add_q3_to_graph(g: Graph, statistic_uri: URIRef, q3: Any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=q3,
        stat_rdf_type=OntologyNamespaces.STATO.value.third_quartile,
        uri_suffix="third_quartile",
        cohort_uri=cohort_uri
    )

# add outliers count by iqr to the graph
def add_outlier_count_by_iqr_to_graph(g: Graph, statistic_uri: URIRef, outlier_count: int, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=outlier_count,
        stat_rdf_type=OntologyNamespaces.STATO.value.outlier_count_by_iqr,
        uri_suffix="outlier_count_by_iqr",
        cohort_uri=cohort_uri
    )

# add outlier count by z score to the graph

def add_outlier_count_by_z_score_to_graph(g: Graph, statistic_uri: URIRef, outlier_count: Any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=outlier_count,
        stat_rdf_type=OntologyNamespaces.STATO.value.outlier_count_by_z_score,
        uri_suffix="outlier_count_by_z_score",
        cohort_uri=cohort_uri
    )



# add normality test to the graph
def add_normality_test_to_graph(g: Graph, statistic_uri: URIRef, normality_test: str, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=normality_test,
        stat_rdf_type=OntologyNamespaces.STATO.value.normality_test,
        uri_suffix="normality_test",
        cohort_uri=cohort_uri
    )


# add wilks shapiro test to the graph
def add_wilks_shapiro_test_to_graph(g: Graph, statistic_uri: URIRef, wilks_shapiro_test: str, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=wilks_shapiro_test,
        stat_rdf_type=OntologyNamespaces.STATO.value.wilks_shapiro_test,
        uri_suffix="wilks_shapiro_test",
        cohort_uri=cohort_uri
    )

# add skewness to the graph
def add_skewness_to_graph(g: Graph, statistic_uri: URIRef, skewness: Any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=skewness,
        stat_rdf_type=OntologyNamespaces.STATO.value.skewness,
        uri_suffix="skewness",
        cohort_uri=cohort_uri
    )

# add variance to the graph
def add_variance_to_graph(g: Graph, statistic_uri: URIRef, variance: Any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=variance,
        stat_rdf_type=OntologyNamespaces.STATO.value.variance,
        uri_suffix="variance",
        cohort_uri=cohort_uri
    )


# add kurtosis to the graph
def add_kurtosis_to_graph(g: Graph, statistic_uri: URIRef, kurtosis: Any, cohort_uri: URIRef) -> Graph:
    return add_numeric_statistic_generic(
        g=g,
        statistic_uri=statistic_uri,
        value=kurtosis,
        stat_rdf_type=OntologyNamespaces.STATO.value.kurtosis,
        uri_suffix="kurtosis",
        cohort_uri=cohort_uri
    )


def add_categorical_variable_visualization(g: Graph, var_uri: URIRef, cohort_uri: URIRef, chart_url:str, xy_axis:list[str]) -> Graph:
    if chart_url is None or xy_axis is None:
        return g
    data_visualization_process_uri = URIRef(f"{var_uri}/data_visualization_process")
    g.add((data_visualization_process_uri, RDF.type, OntologyNamespaces.CMEO.value.data_visualization_process, cohort_uri))
    g.add((var_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_visualization_process_uri, cohort_uri))
    g.add((data_visualization_process_uri, OntologyNamespaces.OBI.value.has_specified_input, var_uri, cohort_uri))
    data_visualization_output_uri = URIRef(f"{var_uri}/bar_chart")
    g.add((data_visualization_output_uri, RDF.type, OntologyNamespaces.OBCS.value.bar_chart, cohort_uri))
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
    if chart_url is None or any(x is None for x in xy_axis):
        return g
    print(f"xy_axis: {xy_axis} for var_uri: {var_uri}")
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
    g.add((histogram_uri, RDF.type, OntologyNamespaces.IAO.value.histogram, cohort_uri))
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
    if "visits" in row:
        visit_not_null = pd.notna(row['visits']) and row['visits']
        if visit_not_null:
            visit_labels = normalize_text(row['visits'])
            visit_uri = get_temporal_context_uri(var_uri, visit_labels)
            g.add((visit_uri, RDF.type, OntologyNamespaces.CMEO.value.visit_type, cohort_uri))
            g.add((visit_uri, OntologyNamespaces.SIO.value.is_attribute_of, var_uri, cohort_uri))
            g.add((var_uri, OntologyNamespaces.SIO.value.has_attribute, visit_uri, cohort_uri))
            g.add((visit_uri, OntologyNamespaces.CMEO.value.has_value, Literal(row['visits'], datatype=XSD.string), cohort_uri))

            
            if pd.notna(row['visit concept name']) and normalize_text(row['visit concept name']):
                concepts = Concept(
                    standard_label=row['visit concept name'] if pd.notna(row['visit concept name']) else None,
                    code=row['visit concept code'] if pd.notna(row['visit concept code']) else None,
                    omop_id=safe_int(row['visit omop id']) if pd.notna(row['visit omop id']) else None,
                )
                
                g= add_solo_concept_info(g, visit_uri, concepts, cohort_uri)
            
                return g
        return g
    return g




def add_categories_to_graph(g: Graph, var_uri: URIRef, cohort_uri: URIRef, row: pd.Series) -> Graph:
    """Adds permissible information related to variable to the RDF graph for a specific variable URI."""
    # Check if there are categories to process
    datatype = row['vartype'] if pd.notna(row['vartype']) else None

    if pd.notna(row['categorical']) and datatype:

        original_categories, category_labels = split_categories(row['categorical'])

        labels = row['categorical value concept name'].split("|") if pd.notna(row['categorical value concept name']) and row['categorical value concept name'] else [None] * len(original_categories)
        codes = row['categorical value concept code'].split("|") if pd.notna(row['categorical value concept code']) and row['categorical value concept code'] else [None] * len(original_categories)
        omop_ids = str(row['categorical value omop id']).split("|") if pd.notna(row['categorical value omop id']) and str(row['categorical value omop id']) else [None] * len(original_categories)
        
        # Add each category to the graph
        for i, (value, defined_value) in enumerate(zip(original_categories, category_labels)):
            data_type_value = "str" if isinstance(value, str) else "int" if isinstance(value, int) else "float" if isinstance(value, float) else "datetime" if isinstance(value, pd.Timestamp) else None
            # n_value = normalize_text(value)
            n_defined_value = normalize_text(defined_value)  

            permissible_uri = URIRef(f"{var_uri}/categorical_value_specification/{n_defined_value}")
            g.add((permissible_uri, RDF.type, OntologyNamespaces.OBI.value.categorical_value_specification, cohort_uri))
            g.add((var_uri, OntologyNamespaces.OBI.value.has_value_specification, permissible_uri, cohort_uri))
            g.add((permissible_uri, OntologyNamespaces.OBI.value.specifies_value_of, var_uri, cohort_uri))
            
            g.add((permissible_uri, RDFS.label, Literal(defined_value, datatype=XSD.string), cohort_uri))
            if data_type_value == 'str':
                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.string), cohort_uri))
            elif data_type_value == 'int':
                if value is None:  print(f"value: {value} defined_value: {defined_value}") 

                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.integer), cohort_uri))
            elif data_type_value == 'float':
                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.decimal), cohort_uri))
            elif data_type_value == 'datetime':
                g.add((permissible_uri, OntologyNamespaces.CMEO.value.has_value, Literal(value, datatype=XSD.dateTime), cohort_uri))


            # Add optional labels, codes, and OMOP IDs if present
            if i < len(labels) and len(labels) == len(original_categories):
                print(f"codes={codes[i]}")
                print(f"omop_ids={omop_ids[i]}")
                print(f"label={labels[i]}")
                concept =  Concept(
                    standard_label=labels[i] if pd.notna(labels[i]) else None,
                    code=codes[i] if pd.notna(codes[i]) else None,
                    omop_id=safe_int(omop_ids[i]) if pd.notna(omop_ids[i]) else None,
                )
                g=add_solo_concept_info(g, permissible_uri, concept, cohort_uri)
           
    return g




def add_measurement_unit(g:Graph, var_uri:URIRef, cohort_uri: URIRef,row: pd.Series) -> Graph:
    """
    Adds measurement unit details to the RDF graph for a given variable URI.
    """
    # 1. Safely extract and strip the string to catch hidden empty values
    unit_raw = row['units']
    unit_str = str(unit_raw).strip() if pd.notna(unit_raw) else ""
    
    # 2. Check that the unit string is actually not empty
    if unit_str != "" and pd.notna(row['unit concept name']):
        unit_concept= Concept(
            standard_label=row['unit concept name'] if pd.notna(row['unit concept name']) else None,
            code=row['unit concept code'] if pd.notna(row['unit concept code']) else None,
            omop_id=safe_int(row['unit omop id']) if pd.notna(row['unit omop id']) else None
        )

        unit_uri = URIRef(f"{var_uri}/measurement_unit_label")
        
        # 3. FIX: Removed the trailing comma from this line
        unit_value = unit_str 
        
        g.add((unit_uri, RDF.type, OntologyNamespaces.OBI.value.measurement_unit_label, cohort_uri))
        g.add((unit_uri, OntologyNamespaces.CMEO.value.has_value, Literal(unit_value, datatype=XSD.string),cohort_uri))
        g= add_solo_concept_info(g, unit_uri, unit_concept, cohort_uri)
        g.add((var_uri, OntologyNamespaces.OBI.value.has_measurement_unit_label, unit_uri,cohort_uri))

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
    # if unit_label is None:
    #     print("Unit label is None")
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



def add_composite_concepts_info(g: Graph, linked_uri: URIRef, concepts: list[Concept], cohort_uri: URIRef) -> Graph:

    data_standardization_uri = URIRef(f"{linked_uri}/data_standardization")
    g.add((data_standardization_uri, RDF.type, OntologyNamespaces.CMEO.value.data_standardization,cohort_uri))
    g.add((linked_uri, OntologyNamespaces.OBI.value.is_specified_input_of, data_standardization_uri,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_input, linked_uri,cohort_uri))

    code_set_uri = URIRef(f"{linked_uri}/code_set")

    g.add((code_set_uri, RDF.type, OntologyNamespaces.IAO.value.code_set,cohort_uri))
    # g.add((code_set_uri, RDF.type, RDF.Seq, cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_output, code_set_uri,cohort_uri))
    g.add((code_set_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_standardization_uri,cohort_uri))
    g.add((linked_uri, OntologyNamespaces.SKOS.value.closeMatch, code_set_uri, cohort_uri)) # for composite concepts we use closeMatch instead of exactMatch as they are not exactly defined by the code set in other vocabularies but our interpretation of them
    # g.add((code_set_uri, OntologyNamespaces.SIO.value.is_close_match_to, linked_uri, cohort_uri))
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
        g.add((code_uri, RDF.type, OntologyNamespaces.SKOS.value.concept,cohort_uri))
        g.add((code_uri, RDFS.label, Literal(label, datatype=XSD.string),cohort_uri))
        g.add((code_uri, OntologyNamespaces.CMEO.value.has_value, Literal(code, datatype=XSD.string),cohort_uri))
        omop_id_uri = URIRef(f"{OntologyNamespaces.OMOP.value}{omop_id}")
        g.add((omop_id_uri, RDF.type, OntologyNamespaces.CMEO.value.omop_id,cohort_uri))
        g.add((code_uri, OntologyNamespaces.IAO.value.denotes, omop_id_uri,cohort_uri))
    
        g.add((omop_id_uri, OntologyNamespaces.CMEO.value.has_value, Literal(omop_id, datatype=XSD.integer),cohort_uri))

        g.add((code_set_uri, OntologyNamespaces.RO.value.has_part, code_uri,cohort_uri))
        g.add((code_uri, OntologyNamespaces.RO.value.is_part_of, code_set_uri,cohort_uri))
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
    if concept.code is None or concept.standard_label is None or concept.omop_id is None:
        return g
    code = concept.code.strip()
    # code_only = code.split(":")[-1]
    # code_only_encoded = quote(code_only, safe='')
    label = concept.standard_label.strip().replace("\n","")
    omop_id = concept.omop_id
    code_uri = create_code_uri(code, cohort_uri)
 
    g.add((code_uri, RDF.type, OntologyNamespaces.SKOS.value.concept,cohort_uri))
    g.add((code_uri, OntologyNamespaces.OBI.value.is_specified_output_of, data_standardization_uri,cohort_uri))
    g.add((data_standardization_uri, OntologyNamespaces.OBI.value.has_specified_output, code_uri,cohort_uri))
    g.add((code_uri, RDFS.label, Literal(label, datatype=XSD.string),cohort_uri))
    g.add((linked_uri, OntologyNamespaces.SKOS.value.closeMatch, code_uri, cohort_uri))
    g.add((code_uri, OntologyNamespaces.CMEO.value.has_value, Literal(code, datatype=XSD.string),cohort_uri))
    omop_id_uri = URIRef(f"{OntologyNamespaces.OMOP.value}{omop_id}")
    g.add((omop_id_uri, RDF.type, OntologyNamespaces.CMEO.value.omop_id,cohort_uri))
    g.add((code_uri, OntologyNamespaces.IAO.value.denotes, omop_id_uri,cohort_uri))
    g.add((omop_id_uri, OntologyNamespaces.CMEO.value.has_value, Literal(omop_id, datatype=XSD.integer),cohort_uri))
    
    # print(f"omop_id: {omop_id} for {linked_uri}")
    return g



    

def add_all_derived_variables(
    g,
    cohort_id: str,
    cohort_graph,
    DERIVED_VARIABLES: list,
    OntologyNamespaces,
    find_var_by_omop_id_func,
    add_standardization_func
):
    """
    For each derived variable:
    - Check if output OMOP ID variable exists; if not, create with standardization.
    - For each input OMOP, find its variable URI.
    - If all exist, create derivation process linking inputs to output.
    """
    for dv in DERIVED_VARIABLES:
        # 1. Find or create output variable node by OMOP ID
        out_var_uri = find_var_by_omop_id_func(g, cohort_graph, dv["omop_id"], OntologyNamespaces)
        # created_output = False
        if out_var_uri is None:
            out_var_uri = URIRef(f"{OntologyNamespaces.CMEO.value}{cohort_id}/{dv['name'].replace(' ', '_').lower()}")
            g.add((out_var_uri, RDF.type, OntologyNamespaces.CMEO.value.data_element, cohort_graph))
            concept = [Concept(
                standard_label=dv["label"],
                code=dv["code"],
                omop_id=dv["omop_id"]
            )]
            g = add_composite_concepts_info(g, out_var_uri, concept, cohort_graph)
            g.add((out_var_uri, RDFS.label, Literal(dv["label"]), cohort_graph))
            g.add((out_var_uri, OntologyNamespaces.CMEO.value.has_omop_id, Literal(dv["omop_id"], datatype=XSD.integer), cohort_graph))
            g.add((out_var_uri, OntologyNamespaces.CMEO.value.has_value, Literal(dv["code"], datatype=XSD.string), cohort_graph))
            g = add_standardization_func(g, out_var_uri, dv, cohort_graph)
            # created_output = True

        # 2. Find input variable URIs by OMOP IDs
        input_vars = []
        missing_inputs = []
        for input_omop in dv["required_omops"]:
            input_uri = find_var_by_omop_id_func(g, cohort_graph, input_omop, OntologyNamespaces)
            if input_uri is not None:
                input_vars.append({"uri": input_uri, "omop_id": input_omop})
            else:
                missing_inputs.append(input_omop)
        if missing_inputs:
            print(f"[{dv['name']}] Skipping: required OMOP(s) not found: {missing_inputs}")
            continue

        # 3. Add data transformation process node and link inputs/outputs
        process_uri = URIRef(f"{out_var_uri}/data_transformation")
        g.add((process_uri, RDF.type, OntologyNamespaces.OBI.value.data_transformation, cohort_graph))
        g.add((process_uri, RDFS.label, Literal(f"{dv['label']} calculation process"), cohort_graph))
        # Add formula—customize per variable type if desired
        formula_str = dv.get("formula", f"Derived using OMOPs: {', '.join(map(str, dv['required_omops']))}")
        g.add((process_uri, OntologyNamespaces.CMEO.value.formula, Literal(formula_str, datatype=XSD.string), cohort_graph))
        # Inputs
        for inp in input_vars:
            g.add((process_uri, OntologyNamespaces.OBI.value.has_specified_input, inp["uri"], cohort_graph))
            # Optionally: add standardization again (idempotent, so it's safe)
            g = add_standardization_func(g, inp["uri"], inp, cohort_graph)
        # Output
        g.add((process_uri, OntologyNamespaces.OBI.value.has_specified_output, out_var_uri, cohort_graph))
        g.add((out_var_uri, OntologyNamespaces.OBI.value.is_specified_output_of, process_uri, cohort_graph))

        print(f"[{dv['name']}] Added derivation process for: {out_var_uri} using {[i['uri'] for i in input_vars]}")

    return g

# def add_raw_data_graph(cohort_data_file_path, cohort_name) -> Graph:
#     try:

#         # Read header to prepare normalized column names (skip patient identifier column)
#         header_data = pd.read_csv(cohort_data_file_path, nrows=0, low_memory=False)
#         header_data = header_data.apply(lambda col: col.map(lambda x: x.lower() if isinstance(x, str) else x))

#         n_cohort_name = normalize_text(cohort_name)
#         cohort_uri= f"{settings.sparql_endpoint}/rdf-graphs/{n_cohort_name}"
#         print(f"cohort_uri: {cohort_uri}")
#         normalized_columns = {col: normalize_text(col) for col in header_data.columns[1:]}
#         var_uris = {col: get_var_uri(n_cohort_name, normalized_columns[col]) for col in normalized_columns}
#         # print(f"\n top ten normalized columns: {list(normalized_columns.items())[:10]}")
#         # print(f"\n top ten var_uris: {list(var_uris.items())[:10]}")
#         var_exists = {col: variable_exists(cohort_uri, normalized_columns[col]) for col in normalized_columns}
       
#         # Initialize the RDF graph
#         cohort_graph = URIRef(OntologyNamespaces.CMEO.value + f"graph/{cohort_name}_pldata")
#         g = init_graph(default_graph_identifier=cohort_graph)
#         rows = 0
#         chunk_size = 100

#         # Process CSV file in chunks
#         for chunk in pd.read_csv(cohort_data_file_path, chunksize=chunk_size, low_memory=False):
#             for i, row in chunk.iterrows():
#                 # Extract and normalize the patient identifier (first column)
#                 patient_id = normalize_text(row.iloc[0])
#                 if not patient_id:
#                     print(f"Skipping row {i}: Missing patient ID.")
#                     continue

#                 # Create unique URIs for the patient and participant identifier
#                 # identifier_uri =  URIRef(OntologyNamespaces.OBI.value + f"participant_identifier/{patient_id}")
#                 participant_under_investigation_role_uri = URIRef(OntologyNamespaces.OBI.value + f"participant_under_investigation_role/{patient_id}")
#                 person_uri = URIRef(OntologyNamespaces.CMEO.value + f"person/{patient_id}")
#                 # g.add((identifier_uri, RDF.type, OntologyNamespaces.CMEO.value.participant_identifier,cohort_graph))
#                 # g.add((identifier_uri, OntologyNamespaces.CMEO.value.has_value, Literal(patient_id, datatype=XSD.string),cohort_graph))
#                 # particpant identifier denotes person who has role of participant under investigation role which concreatizes data item and data item is instantiated in data element
                
#                 g.add((person_uri, RDF.type, OntologyNamespaces.CMEO.value.person,cohort_graph))
#                 # g.add((person_uri, OntologyNamespaces.CMEO.value.has_identifier, identifier_uri,cohort_graph))
               
#                 g.add((participant_under_investigation_role_uri, RDF.type, OntologyNamespaces.CMEO.value.participant_under_investigation_role,cohort_graph))
#                 g.add((person_uri, OntologyNamespaces.CMEO.value.has_role, participant_under_investigation_role_uri,cohort_graph))
#                 g.add((participant_under_investigation_role_uri, OntologyNamespaces.CMEO.value.role_of, person_uri,cohort_graph))
#                 g.add((participant_under_investigation_role_uri, DC.identifier, Literal(row.iloc[0], datatype=XSD.string),cohort_graph))

#                 # Process each variable column (skip the first column)
#                 for col_name in chunk.columns[1:]:
#                     # print(row[col_name])
#                     var_name = normalized_columns.get(col_name)
#                     # print(var_exists.get(col_name, False))
#                     var_value = row[col_name]
#                     # print(f"var_name: {var_name} var_value: {var_value}")
#                     if not var_name or var_exists.get(var_name, False) == False:
#                         # print(f"Skipping row {i}: Missing variable name or value.")
#                         continue
#                     print(f"Processing data point for {var_name} with value {var_value}")
#                     # Create a unique URI for this data point
#                     data_point_uri = URIRef(OntologyNamespaces.CMEO.value + f"data_point/{patient_id}/{var_name}")
#                     dataset_uri = URIRef(f"{var_uris[col_name]}/dataset")
#                     # Add triples for this data point
#                     g.add((data_point_uri, RDF.type, OntologyNamespaces.OBI.value.measurement_datum,cohort_graph))
#                     g.add((data_point_uri, OntologyNamespaces.RO.value.is_part_of, dataset_uri,cohort_graph))
#                     g.add((data_point_uri, OntologyNamespaces.IAO.value.is_about, person_uri,cohort_graph))
#                     g.add((data_point_uri, OntologyNamespaces.CMEO.value.has_value, Literal(var_value, datatype=XSD.string),cohort_graph))
#                     rows += 1

#         print(f"Processed {rows} data points.")
#         return g    
#     except Exception as e:
#         print(f"Error processing raw data file: {e}")
#         return None
       
#         base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         print(f"Base path: {base_path}")
#         Serialize the graph to a temporary file and post it to the SPARQL endpoint
#         temp_file = f"{output_dir}/{cohort_name}_raw_data_graph.ttl"
#         serialized_graph = g.serialize(format="trig")
#         with open(temp_file, "w", encoding="utf-8") as f:
#             f.write(serialized_graph)

#         print(f"Serialized raw data graph to {temp_file}")

#         headers = {"Content-Type": "application/trig"}
#         response = requests.post(f"{settings.sparql_endpoint}/store?graph={cohort_uri}_pldata",
#                                     headers=headers, data=serialized_graph, timeout=300)

#         if response.status_code in (201, 204):
#             print("Raw data successfully added to the graph.")
#         else:
#             print(f"Failed to publish raw data graph: {response.status_code}, {response.text}")

#     except Exception as e:
#         print(f"Error processing raw data file: {e}")
