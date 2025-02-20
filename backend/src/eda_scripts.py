"""Exploratory Data Analysis (EDA) module."""


def c1_data_dict_check(cohort_id: str) -> str:
    return """
import pandas as pd
import decentriq_util

# Load the metadata dictionary
dictionary_df = decentriq_util.read_tabular_data(f"/input/{cohort_id}-metadata")

# Load the dataset
try:
    dataset_df = pd.read_spss(f"/input/{cohort_id}")
except Exception as e:
    dataset_df = decentriq_util.read_tabular_data(f"/input/{cohort_id}")

# Validate that the dictionary file contains the 'VARIABLE NAME' column
if 'VARIABLE NAME' not in dictionary_df.columns:
    raise ValueError("The dictionary file does not contain a 'VARIABLE NAME' column.")

# Extract 'VARIABLE NAME' column from dictionary and dataset column names
dictionary_variables = set(dictionary_df['VARIABLE NAME'].dropna().unique())
dataset_columns = set(dataset_df.columns)

# Compare the sets
in_dictionary_not_in_dataset = dictionary_variables - dataset_columns
in_dataset_not_in_dictionary = dataset_columns - dictionary_variables

# Optionally save the results to files for reference
pd.DataFrame({{'In Dataset Not in Dictionary': list(in_dataset_not_in_dictionary)}}).to_csv("/output/in_dataset_not_in_dictionary.csv",index = False)
pd.DataFrame({{'In Dictionary Not in Dataset': list(in_dictionary_not_in_dataset)}}).to_csv("/output/in_dictionary_not_in_dataset.csv",index = False)
"""


def c2_save_to_json(cohort_id: str) -> str:
    return """
import decentriq_util
import pandas as pd
import os
import json

# Load dictionary
dictionary = decentriq_util.read_tabular_data(f"/input/{cohort_id}-metadata")

# Clean column names to ensure uniformity
dictionary.columns = dictionary.columns.str.strip().str.upper()
dictionary['VARIABLE NAME'] = dictionary['VARIABLE NAME'].str.strip().str.lower()
dictionary['VAR TYPE'] = dictionary['VAR TYPE'].str.strip().str.lower()

# Define the pattern for entries to exclude non-categorical variables
include_pattern = r'\||='   # Look for strings containing either a | or =.

# Exclude rows in the dictionary where the 'CATEGORICAL' column contains the defined pattern
categorical_dict = dictionary[dictionary['CATEGORICAL'].astype(str).str.contains(include_pattern, regex=True)]

# Prepare to extract classes and their meanings, along with MIN, MAX, and VAR TYPE
class_details = {}
numerical_details = {}

for index, row in dictionary.iterrows():
    variable_name = row['VARIABLE NAME']
    var_type = row['VAR TYPE'] if 'VAR TYPE' in dictionary.columns else None
    categories_info = row['CATEGORICAL']
    min_value = row['MIN'] if 'MIN' in dictionary.columns else None
    max_value = row['MAX'] if 'MAX' in dictionary.columns else None
    missing_key = row['MISSING'] if 'MISSING' in dictionary.columns else None

    if pd.notna(categories_info) and isinstance(categories_info, str) and categories_info.strip():
        # Handle categorical variables
        categories = [item for sublist in categories_info.split('|') for item in sublist.split(',')]
        class_names = {}

        for category in categories:
                key_value = category.split('=')
                if len(key_value) == 2:
                    #print("inside if statement: ", variable_name, key_value)
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    class_names[key] = value
                else:
                    print("Encountered a possible parsing error. Check category info for variable ", variable_name, key_value)


        # Check if there is a value that corresponds to  'missing'
        if 'missing' in class_names:
            missing_key = class_names['missing']
        else:
            print("No 'missing' value for variable ", variable_name)
            #needed the line below, otherwise the "missing_key" will still store the value for the previous var
            missing_key = None
                
        # Save MIN, MAX, and VAR TYPE values if they exist to class_details
        class_details[variable_name] = {
            'categories': class_names,

            'missing': missing_key,  # Add missing indicator if found

            'min': min_value if pd.notna(min_value) else None,
            'max': max_value if pd.notna(max_value) else None,
            'var_type': var_type if pd.notna(var_type) else None
        }

    elif (pd.isna(categories_info) or categories_info.strip() == '') and var_type != 'str':
        # Handle numerical variables, if the variable has type "str" (like PatientID, the analysis does not apply to it)
        numerical_details[variable_name] = {
            'min': min_value if pd.notna(min_value) else None,
            'max': max_value if pd.notna(max_value) else None,
            'var_type': var_type if pd.notna(var_type) else None,
            'missing': missing_key
        }

json_dir = '/output/'

# Save categorical variables to a JSON file
categorical_json_path = os.path.join(json_dir, 'categorical_variables.json')
with open(categorical_json_path, 'w') as json_file:
    json.dump(class_details, json_file, indent=4)

# Save numerical variables to a JSON file
numerical_json_path = os.path.join(json_dir, 'numerical_variables.json')
with open(numerical_json_path, 'w') as json_file:
    json.dump(numerical_details, json_file, indent=4)

# Print confirmation messages and the first 5 items in a formatted way
#print(f"Categorical variables saved to {categorical_json_path}")
print(json.dumps({key: class_details[key] for key in list(class_details.keys())[:5]}, indent=4))

print(f"Numerical variables saved to {numerical_json_path}")
print(json.dumps({key: numerical_details[key] for key in list(numerical_details.keys())[:5]}, indent=4))
"""


def c3_map_missing_do_not_run(cohort_id: str) -> str:
    return """
import decentriq_util
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Load the JSON files
categorical_data = pd.read_json("/input/C2_Save_to_JSON/categorical_variables.json")
numerical_data = pd.read_json("/input/C2_Save_to_JSON/numerical_variables.json")


# Step 2: Load the dataset
df = pd.read_spss(f"/input/{cohort_id}_dataset")

# Step 3: Normalize column names in the dataset to lowercase to match JSON keys
df.columns = df.columns.str.lower()  # Convert column names to lowercase for consistency

# Normalize keys in the JSON files to lowercase for consistent comparison
numerical_data = {{k.lower(): v for k, v in numerical_data.items()}}
categorical_data = {{k.lower(): v for k, v in categorical_data.items()}}

# Extract unique missing values
unique_missing_numerical = set(item.get('missing') for item in numerical_data.values())
unique_missing_categorical = set(item.get('missing') for item in categorical_data.values())
print("Unique missing values in numerical_data:", unique_missing_numerical)
print("Unique missing values in categorical_data:", unique_missing_categorical)
print("")

# Extract unique var_type values
unique_var_types_numeric = set(item.get('var_type') for item in numerical_data.values())
unique_var_types_categorical = set(item.get('var_type') for item in categorical_data.values())
print("Unique var_type values in numerical_data:",unique_var_types_numeric)
print("Unique var_type values in categorical_data:",unique_var_types_categorical)
print("")

# Step 4: Iterate over all variables in the dataset
for column in df.columns:
    try:
        # Check if the column is numerical
        if column in numerical_data:
            min_value = numerical_data[column].get('min')
            max_value = numerical_data[column].get('max')
            var_type = numerical_data[column].get('var_type')
            var_type = var_type.lower() if var_type else None
            missing_value = numerical_data[column].get('missing')

            # Handle based on variable type
            if var_type in ['datetime', 'float', 'complex', 'int']:
                df[column] = pd.to_numeric(df[column], errors='coerce')  # Ensure column is numeric
                if not pd.isna(min_value):
                    df.loc[df[column] < float(min_value), column] = np.nan
                if not pd.isna(max_value):
                    df.loc[df[column] > float(max_value), column] = np.nan

            elif var_type in ['str', None]:
                # Skip numeric range validation for string or undefined variable types
                print(f"Skipping range validation for column: {{column}} (var_type: {{var_type}})")


            # Clean missing_value to remove brackets and convert to numeric
            if missing_value is not None:
                missing_value = str(missing_value).strip('[]')  # Remove brackets
                try:
                    missing_value = pd.to_numeric(missing_value, errors='coerce')
                except Exception:
                    pass

            # Handle missing values
            if missing_value is not None:
                df[column] = df[column].apply(lambda x: np.nan if x == missing_value else x)


        # Check if the column is categorical
        elif column in categorical_data:
            missing_value = categorical_data[column].get('missing')
            min_value = categorical_data[column].get('min')
            max_value = categorical_data[column].get('max')
            var_type = categorical_data[column].get('var_type')
            var_type = var_type.lower() if var_type else None
            categories = categorical_data[column].get('categories')

            # Handle based on variable type
            if var_type in ['float', 'int','datetime']:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                if not pd.isna(min_value) and not pd.isna(max_value):
                    df.loc[(df[column] < float(min_value)) | (df[column] > float(max_value)), column] = np.nan
            elif var_type == 'str':
                if categories:
                    # Map categories to their standardized values
                    category_mapping = {{str(k).strip(): v for k, v in categories.items()}}
                    df[column] = df[column].apply(
                        lambda x: category_mapping.get(str(x).strip(), x)
                        if pd.notna(x) and str(x).strip() in category_mapping else x
                    )

            # Clean missing_value to remove brackets and convert to numeric
            if missing_value is not None:
                missing_value = str(missing_value).strip('[]')  # Remove brackets
                try:
                    missing_value = pd.to_numeric(missing_value, errors='coerce')
                except Exception:
                    pass

            # Handle missing values
            if missing_value is not None:
                df[column] = df[column].apply(lambda x: np.nan if x == missing_value else x)


    except Exception as e:
        # Print the column name and the problematic value that caused the error
        print(f"Error occurred in column: {{column}}")
        print(f"Problematic value: {{df[column].dropna().unique()}}")
        raise e

# Step 5: Save the cleaned dataset
df.to_csv("/output/data_correct.csv",index = False, header = True)
"""


def c3_eda_data_profiling() -> str:
    return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, zscore
import warnings
import decentriq_util
import re
from datetime import datetime
warnings.filterwarnings('ignore')


# Load the dataset with corrected missing values replaced with NA from previous step
# data_correct_missing = pd.read_csv("/input/C3_map_missing_do_not_run/data_correct.csv", low_memory=False)

#Load the JSON files from C2
categorical_vars = pd.read_json("/input/c2_save_to_json/categorical_variables.json")
print("the categorical data: ", categorical_vars.keys())
numerical_vars = pd.read_json("/input/c2_save_to_json/numerical_variables.json")
#print("the numerical data: ", numerical_vars)
data = decentriq_util.read_tabular_data(f"/input/{cohort_id}")
#print(data)

#variables that should be graphed
#vars_to_graph = ['age', 'weight', 'cough1', 'angina1', 'hscrp_v6']
vars_to_graph = list(categorical_vars.columns) + list(numerical_vars.columns)
vars_to_graph = [x.strip().lower() for x in vars_to_graph]


def variable_eda(df, categorical_vars, numerical_vars):
    vars_stats = {}
    df.columns = df.columns.str.lower().str.strip()
    for column in df.columns.tolist():
        # Continuous variables
        if column in list(numerical_vars.keys()):

            #if not pd.api.types.is_numeric_dtype(df[column]):
                # Skip if the column is not numeric
                # print("Column ", column, " skipped because non-numeric")
                #continue

            # Descriptive Stats
            stats = df[column].describe()
            mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            total_missing = df[column].isnull().sum()
            missing_percent = total_missing / len(df) * 100

            # Check for numeric values before computing skewness and kurtosis
            if len(df[column].dropna()) > 0:
                #print("COLUMN: ", column, "type: ", df[column].dtype, "values: ", df[column])
                #df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column] = df[column].astype(float)
                skewness = skew(df[column].dropna(), bias=False)
                #skewness = df[column].skew()
                kurt = kurtosis(df[column].dropna(), bias=False)
            else:
                skewness = np.nan
                kurt = np.nan

            # Normality Test
            if len(df[column].dropna()) > 3:  # Shapiro requires at least 3 values
                w_test_stat, p = shapiro(df[column].dropna())
                normality = "Normal" if p > 0.05 else "Non-Normal"
                p_value_str = f"{p:.4f}"
            else:
                normality = "Insufficient Data"
                p_value_str = "N/A"

            # Outlier Detection (IQR Method)
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column].count()

            # Z-Scores for Outliers
            z_scores = zscore(df[column].dropna()) if len(df[column].dropna()) > 0 else np.array([])
            z_outliers = (np.abs(z_scores) > 3).sum() if z_scores.size > 0 else 0

            # Range Calculation
            range_value = stats['max'] - stats['min']

            # Stats Text
            stats_text = (
                f"Count:               {stats['count']}",
                f"Mean:                {stats['mean']:.2f}",
                f"Median:              {stats['50%']:.2f}",
                f"Mode:                {mode_value:.2f}",
                f"Std Dev:             {stats['std']:.2f}",
                f"Variance:            {stats['std']**2:.2f}",
                f"Range:               {range_value:.2f}", 
                f"Q1:                  {Q1:.2f}",
                f"Q3:                  {Q3:.2f}",
                f"IQR:                 {IQR:.2f}",
                f"Missing:             {total_missing} ({missing_percent:.2f}%)",
                f"Outliers (IQR):      {outliers} ({(outliers / len(df) * 100):.2f}%)",
                f"Outliers (Z):        {z_outliers}",
                f"Skewness:            {skewness:.2f}",
                f"Kurtosis:            {kurt:.2f}",
                f"W_Test:              {w_test_stat:.2f}",
                f"Normality Test: p-value={p_value_str} => {normality}",
                f"Data Type:           {df[column].dtype}"
            )

            if column in vars_to_graph:
                create_save_graph(df, column, stats_text, 'numerical')


        # Categorical variables
        elif column in categorical_vars.keys():
            value_counts = df[column].value_counts(dropna=False)
            total = len(df)

            # Get the categories mapping and normalize keys
            categories_mapping = categorical_vars[column].get("categories", [])
            categories_mapping = {str(k): v for (k, v) in categories_mapping}

            if value_counts.empty:
                stats_text = (
                    f"Variabl: {column}",
                    f"Number of Unique Categories: 0",
                    f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)"
                )
            else:
                # Chi-square test
                expected = total / len(value_counts)
                chi_square_stat = ((value_counts - expected) ** 2 / expected).sum()

                # Class balance with corrected mapping
                class_balance_text = " - ".join([
                    f"{categories_mapping.get(str(key), 'Unknown')}:  {round(count / total * 100)}%"
                    for key, count in value_counts.items()
                ])

                stats_text = (
                    f"Column: {column}",
                    f"Number of Unique Categories: {len(value_counts)}",
                    f"Most Frequent Category: {categories_mapping.get(str(value_counts.idxmax()).split('.')[0], 'Unknown')} ",
                    f"Number of observations: {value_counts.sum()}",
                    f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)",
                    f"Class Balance: {class_balance_text}",
                    f"Chi-Square Test Statistic: {chi_square_stat:.2f}"
                )

            if column in vars_to_graph:
                create_save_graph(df, column, stats_text, 'categorical')

        stats_text_dict = {i.split(":")[0].strip():i.split(":")[1].strip() for i in stats_text}
        vars_stats[column] = stats_text_dict
    return vars_stats



def create_save_graph(df, varname, stats_text, vartype):

    if vartype == 'numerical':
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Display Summary Stats
        axes[0].text(0.05, 0.9, stats_text, fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
        axes[0].axis("off")

        # Right: Plot histogram
        sns.histplot(df[varname].dropna(), kde=True, ax=axes[1])
        axes[1].set_title(f"Histogram for {varname}")
        axes[1].tick_params(axis='x')

        # Save the figure for the current feature
        plt.tight_layout()
        plt.savefig(f"/output/eda_numerical_{varname}.png")
        print(f"figure for {varname} saved!! ")
        plt.close()

    elif vartype == 'categorical':

        value_counts = df[varname].value_counts(dropna=False)
        total = len(df)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Summary stats text
        axes[0].text(0.1, 0.5, stats_text, fontsize=10, va='center', ha='left', family='monospace', wrap=True)
        axes[0].axis("off")

        # Bar chart
        if not value_counts.empty:
            colors = sns.color_palette("husl", len(value_counts))
            ax = value_counts.plot(kind='bar', color=colors, edgecolor='black', ax=axes[1])
            ax.set_title(f"Distribution in {varname}")
            ax.set_xlabel("Categories")
            ax.set_ylabel("Count")

            # Add labels to the bars
            for idx, value in enumerate(value_counts):
                percentage = (value / total) * 100
                ax.text(idx, value + total * 0.02, f"{value}({percentage:.1f}%)",
                        ha='center', fontsize=10)

            # Adjust x-axis labels to be horizontal
            ax.set_xticklabels(value_counts.index.astype(str), rotation=0, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"/output/eda_categorical_{varname}.png")
        print(f"figure for {varname} saved!! ")
        plt.close()


def integrate_eda_with_metadata(vars_stats):
    meta_data = decentriq_util.read_tabular_data("/input/TIME-CHF-metadata")
    metadata_vars = [x.lower().strip() for x in meta_data['VARIABLE NAME'].values]
    #print("vars from var_stats:", vars_stats.keys())
    #print("vars in metadata: ", metadata_vars)
    #print(" vars in common: ", [x for x in metadata_vars if x in vars_stats.keys()])
    #print(" vars no stats: ", [x for x in metadata_vars if x not in vars_stats.keys()])
    additional_cols = set()
    for s in vars_stats.values():
        additional_cols.update(s.keys())
    for c in additional_cols:
        cvals = []
        for vname in metadata_vars:
            if not vname in vars_stats or not c in vars_stats[vname]:
                cvals.append(None)
            else:
                cvals.append(vars_stats[vname][c])
        meta_data[c] =cvals
    meta_data.to_csv("/output/meta_data_enriched.csv")
    return meta_data



def generate_graph_file(df):
    max_str_length = 20

    def clean_name(text):
        return re.sub(r'[^\w\s-]', '', str(text)).strip().replace(' ', '_').lower()

    def get_xsd_type(value):
        if pd.isna(value):
            return None
        elif isinstance(value, int):
            return 'xsd:integer'
        elif isinstance(value, float):
            return 'xsd:decimal'
        elif isinstance(value, datetime):
            return 'xsd:dateTime'
        else:
            return 'xsd:string'

    def format_value(value, xsd_type):
        if pd.isna(value):
            return None
        elif xsd_type == 'xsd:decimal':
            return f'{value:.6f}'.rstrip('0').rstrip('.')
        elif xsd_type == 'xsd:string':
            return f'"{str(value)}"'
        elif xsd_type == 'xsd:dateTime':
            return value.isoformat()
        else:
            return str(value)

    domain_col_name = [x for x in df.columns if x.strip().lower() == 'domain' or x.strip().lower() == 'omop'][0]
    with open('/output/enriched_metadata_graph.ttl', 'w', encoding='utf-8') as f:
        # Write standard prefixes
        f.write('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n')
        f.write('@prefix omop: <http://omop.org/> .\n')
        
        #write domain specific prefixes
        domains = df[domain_col_name].unique()
        for domain in domains:
            domain_clean = clean_name(domain)
            f.write(f'@prefix {domain_clean}: <http://omop.org/{domain_clean}/> .\n')
        f.write('\n')
        
        for idx, row in df.iterrows():
            domain = clean_name(row[domain_col_name])
            var_name = clean_name(row['VARIABLE NAME'])
            
            # Start variable definition
            f.write(f'{domain}:{var_name}\n')
            f.write(f'    a omop:{row[domain_col_name]} ;\n')
            
            # Process all columns except VARIABLE NAME and domain
            properties = []
            for col in df.columns:
                if (col in ['VARIABLE NAME', domain_col_name] or 
                    pd.isna(row[col]) or 
                    (type(row[col]) == str and len(row[col])>max_str_length)):
                    continue
                else:
                    value = row[col]
                    xsd_type = get_xsd_type(value)
                    
                    if xsd_type is not None:
                        formatted_value = format_value(value, xsd_type)
                        if formatted_value is not None:
                            # Clean column name for property
                            prop_name = clean_name(col)
                            
                            # Add property with typed literal
                            if xsd_type == 'xsd:string':
                                properties.append(f'    omop:{prop_name} {formatted_value}^^{xsd_type}')
                            else:
                                properties.append(f'    omop:{prop_name} {formatted_value}')
            
            # Write all properties with proper punctuation
            for i, prop in enumerate(properties):
                if i == len(properties) - 1:
                    f.write(f'{prop} .\n\n')
                else:
                    f.write(f'{prop} ;\n')

    print("RDF file generated successfully!")



def generate_edgelist_graph(df):
    max_str_length = 20
    edges = []
    other_cols = [col for col in df.columns if col != 'VARIABLE NAME']
   
    for _, row in df.iterrows():
        source = row['VARIABLE NAME']
        for col in other_cols:
            target = row[col]
            if pd.isna(target) or (type(target) == str and len(target)>max_str_length):
                continue
            else:
                edges.append((str(source), str(target), col))
    with open("/output/enriched_kg.csv", 'w', encoding='utf-8') as f:
        f.write('source,target,type\n')
        for source, target, edge_type in edges:
            f.write(f'{source},{target},{edge_type}\n')
    print("Edgelist file generated successfully!")



vars_to_stats = variable_eda(data, categorical_vars, numerical_vars)
meta_data_enriched = integrate_eda_with_metadata(vars_to_stats)
generate_graph_file(meta_data_enriched)
generate_edgelist_graph(meta_data_enriched)
"""
