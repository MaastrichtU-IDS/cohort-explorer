"""Exploratory Data Analysis (EDA) module."""


def c1_data_dict_check(cohort_id: str) -> str:
    raw_script = """
import pandas as pd
import decentriq_util

# Load the metadata dictionary
dictionary_df = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")
try:
    varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in dictionary_df.columns][0]
except:
    raise ValueError("The dictionary file does not contain a 'VARIABLE NAME'/'VARIABLENAME' column.")

print("metadata variable names: ", [v for v in dictionary_df[varname_col]])

# Load the dataset
try:
    dataset_df = pd.read_csv("/input/{cohort_id}")
    #dataset_df = decentriq_util.read_tabular_data("/input/{cohort_id}")
except Exception as e:
    try:
        dataset_df = pd.read_spss("/input/{cohort_id}")
    except Exception as e2:
        raise ValueError("The dataset file does appear to be a valid CSV or SPSS file.\nCSV error: " + str(e) + "\nSPSS error: " + str(e2))
    
    

# Extract 'VARIABLE NAME' column from dictionary and dataset column names
dictionary_variables = set([x.strip() for x in dictionary_df[varname_col].unique()])
dataset_columns = set([x.strip() for x in dataset_df.columns])

# Compare the sets
in_dictionary_not_in_dataset = dictionary_variables - dataset_columns
in_dataset_not_in_dictionary = dataset_columns - dictionary_variables

# Optionally save the results to files for reference
pd.DataFrame({'In Dataset Not in Dictionary': list(in_dataset_not_in_dictionary)}).to_csv("/output/in_dataset_not_in_dictionary.csv",index = False)
pd.DataFrame({'In Dictionary Not in Dataset': list(in_dictionary_not_in_dataset)}).to_csv("/output/in_dictionary_not_in_dataset.csv",index = False)
#print("variable names: ", [v for v in dictionary_df[varname_col]])
"""
    return raw_script.replace("{cohort_id}", cohort_id)


def c2_save_to_json(cohort_id: str) -> str:
    raw_script = """import decentriq_util
import pandas as pd
import os
import json
from pprint import pprint


def _column_is_date(series):
    try:
        pd.to_datetime(series)
        return True
    except:
        return False

def _column_is_float(series):
    try:
        non_na = series.dropna()
        float_series = non_na.astype(float)
        return True
    except:
        return False

def _column_is_numeric(series):
    #meaning integers
    try:
        if series.dropna().apply(lambda x: str(x).isdigit() or str(x).endswith('.0')).all():
            return True
        else:
            return False
    except:
        return False
        
def _cast_col(series, typ):
    if typ == 'date':
        ns = pd.to_datetime(series, errors='coerce').dt.date
    elif typ == 'int':
        ns = pd.to_numeric(series, errors='coerce').astype('Int64')
    elif typ == 'float':
        ns = pd.to_numeric(series, errors='coerce')
    else:
        print("unrecognized type")
        return series, []
    invalid_cells = []
    for i, (c1, c2) in enumerate(zip(series, ns)):
        if pd.notna(c1) and pd.isna(c2):
            invalid_cells.append(i)
    return ns, invalid_cells

# Load dictionary
dictionary = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")

# Clean column names to ensure uniformity
dictionary.columns = dictionary.columns.str.strip().str.upper()

varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in dictionary.columns][0]
vartype_col = [x for x in ['VAR TYPE', 'VARTYPE'] if x in dictionary.columns][0]
varlabel_col = [x for x in ['VARIABLE LABEL', 'VARIABLELABEL', 'VAR LABEL'] if x in dictionary.columns][0]

dictionary[varname_col] = dictionary[varname_col].str.strip().str.lower()
dictionary[vartype_col] = dictionary[vartype_col].str.strip().str.lower()

try:
    data = pd.read_csv("/input/{cohort_id}", na_values=[''], keep_default_na=False)
    #data = decentriq_util.read_tabular_data("/input/{cohort_id}")
except Exception as e:
    data = pd.read_spss("/input/{cohort_id}")


#Convert whitespace-only strings to NaN
for col in data.select_dtypes(include=['object']):
    data[col] = data[col].apply(lambda x: pd.NA if isinstance(x, str) and x.isspace() else x)

data.columns = [c.lower().strip() for c in data.columns]

#for col in data.columns:
#    try:
#        if data[col].dropna().apply(lambda x: str(x).isdigit() or str(x).endswith('.0')).all():
#            data[col] = data[col].astype('Int64')
#    except:
#        continue

# Define the pattern for entries to exclude non-categorical variables
#include_pattern = r'\||='   # Look for strings containing either a | or =.
# Exclude rows in the dictionary where the 'CATEGORICAL' column contains the defined pattern
# categorical_dict = dictionary[dictionary['CATEGORICAL'].astype(str).str.contains(include_pattern, regex=True)]

vars_to_process = {}
# Prepare to extract classes and their meanings, along with MIN, MAX, and VAR TYPE
vars_details = {}
mismatched_types = {}

for index, row in dictionary.iterrows():
    variable_name = row[varname_col]
    var_type = row[vartype_col]
    categories_info = row['CATEGORICAL']

    if variable_name.lower() in ['patientid', 'pat.id', 'patiëntnummer']:
        continue
        
    if variable_name.lower() not in data.columns:
        continue

    if (pd.notna(categories_info) and isinstance(categories_info, str) and categories_info.strip() != ""):
        vars_to_process[variable_name] = 'categorical'
    elif _column_is_numeric(data[variable_name]):
        if data[variable_name].nunique()>9:
            vars_to_process[variable_name] = 'int'
        else:
            vars_to_process[variable_name] = 'categorical'
    elif _column_is_float(data[variable_name]):
        vars_to_process[variable_name] = 'float'
    elif _column_is_date(data[variable_name]):
        vars_to_process[variable_name] = 'date'
    elif data[variable_name].nunique()>20:
        #assume float:
        vars_to_process[variable_name] = 'float'
    else: #fewer than 20 unique
        #assume categorical:
        print("The following variable deemed categorical by process of elimination: ", variable_name)
        vars_to_process[variable_name] = 'categorical'

    if ((var_type.lower() == "datetime" and vars_to_process[variable_name] != "date") or 
        var_type.lower() == "str" and vars_to_process[variable_name] != "categorical"):
        mismatched_types[variable_name] = {"declared": var_type, "inferred": vars_to_process[variable_name]}

#find the mismatches between declared types (in data dictionary) and inferred types:

for index, row in dictionary.iterrows():
    variable_name = row[varname_col]
    var_type = row[vartype_col]
    if variable_name not in vars_to_process:
        continue
    else:
        t = vars_to_process[variable_name]
    categories_info = row['CATEGORICAL']

    if t == 'categorical':
        #categories = [item for sublist in categories_info.split('|') for item in sublist.split(',')]
        categories = [item.strip() for item in categories_info.split('|')]
        class_names = {}

        for category in categories:
                key_value = category.lower().split('=')
                if len(key_value) == 2:
                    #print("inside if statement: ", variable_name, key_value)
                    key = key_value[0].strip()
                    value = key_value[1].upper().strip()
                    class_names[key] = value
                elif len(key_value) == 1:  
                    #category does not have "="
                    class_names[key_value[0].strip().upper()] = key_value[0].strip().upper()
                else:
                    msg = f"Encountered a possible parsing error. Check category info for variable {variable_name}, {key_value}, Full category info: {categories_info}"
                    mismatched_types[variable_name + "-categories"] =  msg
                    print(msg)

        # Check if there is a value that corresponds to  'missing'
        if 'MISSING' in class_names.values():
            missing_key = [x[0] for x in class_names.items() if x[1] == 'MISSING'][0]
            print("MISSING value exists among categories: ", variable_name, missing_key)
        elif 'MISSING' in dictionary.columns and row['MISSING'].strip() != "":
            missing_key = str(row['MISSING']).strip()
            print(f"MISSING value {missing_key} declared for variable: ", variable_name)
        else:
            print("No 'missing' value for variable ", variable_name)
            #needed the line below, otherwise the "missing_key" will still store the value for the previous var
            missing_key = None

    else: #ints or floats or dates:
        missing_key = str(row['MISSING']).strip() if 'MISSING' in dictionary.columns and row['MISSING'].strip() != "" else None

    if missing_key == None:
        count_missing = 0
    else:
        count_missing = (data[variable_name].astype(str).str.strip().str.upper() == str(missing_key).strip().upper()).sum()

    na_count = data[variable_name].isna().sum()

    vars_details[variable_name] = {
            'var_label': row[varlabel_col],
            'missing': missing_key,
            'declared_type': var_type,
            'inferred_type': t,
            'count_missing': int(count_missing),
            'count_na': int(na_count)
    }
    if t == 'categorical':
        vars_details[variable_name]['categories'] = class_names

json_dir = '/output/'

# Save all variable details to a JSON file
vars_details_json_path = os.path.join(json_dir, 'variable_details.json')
with open(vars_details_json_path, 'w') as json_file:
    json.dump(vars_details, json_file, indent=4)

# Print confirmation messages and the first 5 items in a formatted way
print(f"Variable details saved to {vars_details_json_path}")
print(json.dumps({key: vars_details[key] for key in sorted(list(vars_details.keys()))[:-1]}, indent=4))
            
all_data_issues = [str(i) for i in mismatched_types.items()]

data_issues_json_path = os.path.join(json_dir, 'data_issues.json')
with open(data_issues_json_path, 'w') as json_file:
    json.dump(all_data_issues, json_file, indent=4)
pprint(all_data_issues)
"""
    return raw_script.replace("{cohort_id}", cohort_id)



def c3_eda_data_profiling(cohort_id: str) -> str:
    raw_script = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, zscore
import warnings
import decentriq_util
import re
from datetime import datetime
import json
import collections.abc
from collections import OrderedDict
warnings.filterwarnings('ignore')



# Load the dataset with corrected missing values replaced with NA from previous step
# data_correct_missing = pd.read_csv("/input/C3_map_missing_do_not_run/data_correct.csv", low_memory=False)

#Load the JSON files from C2
vars_details = pd.read_json("/input/c2_save_to_json/variable_details.json")

with open("/input/c2_save_to_json/data_issues.json") as f:
    data_issues = json.load(f)

try:
    data = pd.read_csv("/input/{cohort_id}", na_values=[''], keep_default_na=False)
    #data = decentriq_util.read_tabular_data("/input/{cohort_id}")
except Exception as e:
    data = pd.read_spss("/input/{cohort_id}")

for col in data.select_dtypes(include=['object']):
    data[col] = data[col].apply(lambda x: pd.NA if isinstance(x, str) and x.isspace() else x)

data.columns = [c.lower().strip() for c in data.columns]


def _cast_col(series, typ):
    if typ == 'date':
        ns = pd.to_datetime(series, errors='coerce').dt.date
    elif typ == 'int':
        ns = pd.to_numeric(series, errors='coerce').astype('Int64')
    elif typ == 'float':
        ns = pd.to_numeric(series, errors='coerce')
    elif typ == 'categorical':
        try:
            ns = series.astype('Int64').astype(str)
        except:
            ns = series.astype(str)
    else:
        print("unrecognized type")
        return series, []
    invalid_cells = []
    for i, (c1, c2) in enumerate(zip(series, ns)):
        if pd.notna(c1) and pd.isna(c2):
            invalid_cells.append(i)
    return ns, invalid_cells


#type casting the data:
for v, d in vars_details.items():
    try:
        cast_col, inv_cells = _cast_col(data[v], d['inferred_type'])
    except Exception as e:
        msg = f"Error while attempting to cast {v} to {d['inferred_type']}: {e}"
        print(msg)
        data_issues.append(msg)
        continue
    data[v] = cast_col
    if len(inv_cells)>0:
        data_issues.append(f"Column {v} the following cells appears to be invalid: {str(inv_cells)}")



#variables that should be graphed
#vars_to_graph = ['age', 'weight', 'cough1', 'angina1', 'hscrp_v6']
#vars_to_graph = ['age', 'ALCOOL', 'ALLOPURI', 'ALT', 'ALTACE', 'ALTANO', 'COLETOT', 'CREATIN', 'DALTACE', 'DATAECG', 'DATALAB']
#vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB']
#vars_to_graph = [x.lower() for x in vars_to_graph]
#vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB', 'AATHORAX', 'AATHORAXDIM', 'ACE_AT_V1']
vars_to_graph = list(vars_details.columns)
vars_to_graph = [x.strip().lower() for x in vars_to_graph]

def _lowercase_if_string(x):
    if isinstance(x, str):
        return x.lower()
    return x

def variable_eda(df, vars_details):
    vars_stats = {}
    graph_tick_data = {}
    df.columns = df.columns.str.lower().str.strip()
    for column in df.columns.tolist():
        if column not in vars_details.columns:
            continue
        # Continuous variables
        try:
            if 'missing' in vars_details[column] and vars_details[column]['missing']:
                if vars_details[column]['inferred_type'] in ['int']:
                    df[column].replace(int(vars_details[column]['missing']), pd.NA, inplace=True)
                elif vars_details[column]['inferred_type'] in ['float']:
                    df[column].replace(float(vars_details[column]['missing']), pd.NA, inplace=True)
                elif vars_details[column]['inferred_type'] in ['date']:
                    df[column].replace(vars_details[column]['missing'], pd.NA, inplace=True)
                else: #categorical
                    df[column].replace(vars_details[column]['missing'], '<missing>', inplace=True)
        
            if vars_details[column]['inferred_type'] in ['int', 'float']:

                #if not pd.api.types.is_numeric_dtype(df[column]):
                    # Skip if the column is not numeric
                    # print("Column ", column, " skipped because non-numeric")
                    #continue

                # Descriptive Stats
                stats = df[column].describe()
                mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
                value_counts = df[column].value_counts(dropna=False)
                count_missing = vars_details[column]['count_missing']
                missing_percent = count_missing / len(df) * 100
                count_na = vars_details[column]['count_na']
                #try:
                #    empty = df[column].isnull().sum() + df[column].str.strip().eq('').sum()
                #except:
                #    empty = df[column].isnull().sum()

                # Check for numeric values before computing skewness and kurtosis
                if len(df[column].dropna()) > 0:
                    #print(f"Debug {column}: dtype={df[column].dtype}, first 5 values={df[column].head()}")
                    #print(f"After dropna: dtype={df[column].dropna().dtype}, count={len(df[column].dropna())}")
                    #df[column] = pd.to_numeric(df[column], errors='coerce')
                    #column_no_na = df[column].dropna()
                    #print(f"After numeric coerce and dropping na: dtype={column_no_na.dtype}, count={column_no_na}")
                    #print("type of column no na: ", type(column_no_na))
                    #print("type of column_no_na values: ", type(column_no_na.values))
                    #df[column] = df[column].astype(float)
                    skewness = skew([_ for _ in df[column].dropna()], bias=False)
                    #skewness = df[column].skew()
                    kurt = kurtosis([_ for _ in df[column].dropna()], bias=False)
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
                z_scores = zscore([_ for _ in df[column].dropna()]) if len(df[column].dropna()) > 0 else np.array([])
                z_outliers = (np.abs(z_scores) > 3).sum() if z_scores.size > 0 else 0

                # Range Calculation
                range_value = stats['max'] - stats['min']
                count_nonnull = int(stats['count'])-int(count_missing)

                # Stats Text
                stats_text = (
                    f"Column: {column}",
                    f"Label: {vars_details[column]['var_label']}",
                    f"Type: Numeric (encoded as {df[column].dtype})",
                    f"Count of observations (ex. missing/empty): {count_nonnull}",
                    f"Count empty:         {count_na} ({(count_na/len(df[column])) * 100:.2f}%)",
                    f"Count missing:       {count_missing} ({(count_missing/len(df[column])) * 100:.2f}%)",
                    f"Code for missing value: {vars_details[column]['missing']}",
                    f"Number of Unique Values/Categories: {df[column].nunique()}",
                    f"Mean:                {stats['mean']:.2f}",
                    f"Median:              {stats['50%']:.2f}",
                    f"Mode:                {mode_value:.2f}",
                    f"Std Dev:             {stats['std']:.2f}",
                    f"Variance:            {stats['std']**2:.2f}",
                    f"Max:                 {stats['max']:.2f}",
                    f"Min:                 {stats['min']:.2f}",
                    f"Range:               {range_value:.2f}", 
                    f"Q1:                  {Q1:.2f}",
                    f"Q3:                  {Q3:.2f}",
                    f"IQR:                 {IQR:.2f}",
                    f"Outliers (IQR):      {outliers} ({(outliers / len(df)) * 100:.2f}%)",
                    f"Outliers (Z):        {z_outliers}",
                    f"Skewness:            {skewness:.2f}",
                    f"Kurtosis:            {kurt:.2f}",
                    f"W_Test:              {w_test_stat:.2f}",
                    f"Normality Test: p-value={p_value_str} => {normality}"
                )

                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'numerical')
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")


            # Categorical variables
            elif vars_details[column]['inferred_type'] == 'categorical':
                stats = df[column].describe()
                value_counts = df[column].apply(_lowercase_if_string).value_counts(dropna=False)
                total = len(df)
                

                # Get the categories mapping and normalize keys
                categories_mapping = vars_details[column].get("categories", [])
                categories_mapping = {str(k): v for (k, v) in categories_mapping.items()}
                #print("variable: ", column, "categories:", categories_mapping, value_counts )

                if value_counts.empty:
                    stats_text = (
                        f"Column: {column}",
                        f"Label: {vars_details[column]['var_label']}",
                        f"Type: Categorical (encoded as {df[column].dtype})",
                        f"Number of Unique Categories: 0",
                        f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)"
                    )
                else:
                    # Chi-square test
                    expected = total / len(value_counts)
                    chi_square_stat = ((value_counts - expected) ** 2 / expected).sum()
                    count_missing = vars_details[column]["count_missing"]
                    count_na = vars_details[column]["count_na"]
                    count_nonnull = int(stats['count'])-int(count_missing)

                    # Class balance with corrected mapping
                    class_balance_text = "\\n\\t"
                    for key, count in value_counts.items():
                        if str(key) in categories_mapping and str(key) != categories_mapping[str(key)]:
                            class_balance_text += f"{(key, categories_mapping[str(key)])} -> {round(count / total * 100, 2)}%\\n	"
                        else:
                            class_balance_text += f"{key} -> {round(count / total * 100, 2)}%\\n	"

                    stats_text = (
                        f"Column: {column}",
                        f"Label: {vars_details[column]['var_label']}",
                        f"Type: Categorical (encoded as {df[column].dtype})",
                        f"Number of unique values/categories: {len(value_counts)}",
                        f"Most frequent category: {categories_mapping.get(str(value_counts.idxmax()), 'Unknown')} ",
                        f"Count of observations (ex. missing/empty): {count_nonnull}",
                        f"Count empty: {count_na} ({(count_na/len(df[column])) * 100:.2f}%)",
                        f"Count missing: {count_missing} ({(count_missing/len(df[column])) * 100:.2f}%)",
                        f"Code for missing value: {vars_details[column]['missing']}",
                        f"Class balance: {class_balance_text}",
                        f"Chi-Square Test Statistic: {chi_square_stat:.2f}"
                    )

                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'categorical', category_mapping = categories_mapping)
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")
                        
                        
            
            elif vars_details[column]['inferred_type'] == 'date':
                try:
                    stats = pd.to_datetime(df[column], format='mixed').describe()
                except:
                    continue
                value_counts = df[column].value_counts(dropna=False)
                total = len(df)
                count_missing = vars_details[column]["count_missing"]
                count_na = vars_details[column]["count_na"]
                count_nonnull = int(stats['count'])-int(count_missing)
                stats_text = [
                        f"Column: {column}",
                        f"Label: {vars_details[column]['var_label']}",
                        f"Type: Date (encoded as {df[column].dtype})",
                        f"Number of unique values: {len(value_counts)}",
                        f"Most frequent value: {str(value_counts.idxmax()).split('.')[0]}",
                        f"Count of observations (ex. missing/empty): {count_nonnull}",
                        f"Count missing: {count_missing} ({(count_missing/len(df[column])) * 100:.2f}%)",
                        f"Count empty: {count_na} ({(count_na/len(df[column])) * 100:.2f}%)",
                        f"Mean:                {stats['mean'].date()}",
                        f"Median:              {stats['50%'].date()}",
                        f"Max:                 {stats['max'].date()}",
                        f"Min:                 {stats['min'].date()}",
                        f"Range:               {stats['max'] - stats['min']}", 
                        f"Q1:                  {stats['25%'].date()}",
                        f"Q3:                  {stats['75%'].date()}",
                        f"IQR:                 {stats['75%'] - stats['25%']}",
                ]
                #stats_text.extend([f"{k.capitalize()}: {v}" for k,v in stats.items()])
                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'datetime')
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")
            else:
                print("ELSE case: variable name ", column, "inferred type: ", vars_details[column]['inferred_type'])
                stats_text = []
            stats_text_dict = OrderedDict()
            stats_text_dict.update({i.split(":")[0].strip():i.split(":")[1].strip() for i in stats_text})
            if 'Class balance' in stats_text_dict:
                stats_text_dict['Class balance'].replace(" ->", ":")
            stats_text_dict['url'] = f"https://explorer.icare4cvd.eu/api/variable-graph/{cohort_id}/{column}"
            vars_stats[column] = stats_text_dict
        except Exception as e:
            data_issues.append(f"Failed to perform EDA on column {column}. Exception msg: {str(e)}")
            
    for col, ticks  in graph_tick_data.items():
        vars_stats[col].update(ticks)
    return vars_stats



def create_save_graph(df, varname, stats_text, vartype, category_mapping=None):
    if vartype == 'numerical':
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Display Summary Stats

        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor="lightgray")
        text_obj = axes[0].text(0.05, 0.95, '\\n'.join(stats_text), transform=axes[0].transAxes, fontsize=11, va='top', ha='left', 
        family='monospace',  bbox=props, wrap=True, linespacing=1.5)
        if hasattr(text_obj, "_get_wrap_line_width"):
            text_obj._get_wrap_line_width = lambda: 420
        #axes[0].text(0.05, 0.9, , fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
        axes[0].axis("off")

        # Right: Plot histogram
        sns.histplot(df[varname].dropna(), kde=True, ax=axes[1])
        axes[0].set_title(f"Summary Stats for {varname.upper()}", fontsize=12)
        axes[1].set_title(f"Distribution of {varname.upper()}", fontsize=12)
        axes[1].tick_params(axis='x')
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")

        # Save the figure for the current feature
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")
        #print(f"figure for {varname} saved!! ")
       #plt.close()
    elif vartype == 'datetime':
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor="lightgray")
        text_obj = axes[0].text(0.05, 0.95, '\\n'.join(stats_text), transform=axes[0].transAxes, fontsize=11, va='top', ha='left', 
        family='monospace',  bbox=props, wrap=True, linespacing=1.5)
        if hasattr(text_obj, "_get_wrap_line_width"):
            text_obj._get_wrap_line_width = lambda: 420
        #axes[0].text(0.05, 0.9, , fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
        axes[0].axis("off")
        axes[0].set_title(f"Summary Stats for {varname.upper()}", fontsize=12)
        try:
            date_vals =  pd.to_datetime(df[varname].dropna(), format='mixed')
        except:
            print("supposed date column could not be parsed: ", varname)
            return {}
    
        date_nums = mdates.date2num(date_vals)
    
        min_date = date_vals.min()
        max_date = date_vals.max()
        date_range = max_date - min_date
    
        
        if date_range.days > 365 * 10:  
            bin_freq = 'YS'  # Yearly start
            axes[1].xaxis.set_major_locator(mdates.YearLocator(base=2))
        elif date_range.days > 365 * 5: 
            bin_freq = 'YS'  # Yearly
            axes[1].xaxis.set_major_locator(mdates.YearLocator())
        if date_range.days > 365 :
            bin_freq = 'Q'  # Quarterly
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif date_range.days > 90 :
            bin_freq = 'M'  
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())  # Monthly
        elif date_range.days > 30:
            bin_freq = 'W'  # Weekly bins
            axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Weekly
        else:
            bin_freq = 'D'  # Daily
            axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        bins = mdates.date2num(pd.date_range(min_date, max_date, freq=bin_freq))
        
        axes[1].hist(date_nums, bins=bins, alpha=0.7)
        axes[1].set_title(f"Distribution of {varname.upper()}", fontsize=12)

        
        if date_range.days <= 90:
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
        elif date_range.days <= 365 * 2:
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # "2020-01" format
        else:
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].tick_params(axis='x', which='minor', bottom=False)
        
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")
        #print(f"figure for {varname} saved!! ")


    elif vartype == 'categorical':

        if df[varname].isna().sum() > 0:
            value_counts = df[varname].value_counts(dropna=False)
        else:
            value_counts = df[varname].value_counts(dropna=True)
        total = len(df)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        

        # Summary stats text
        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.8, edgecolor="lightgray")
        text_obj = axes[0].text(0.05, 0.95, '\\n'.join(stats_text), transform=axes[0].transAxes, fontsize=11, va='top', ha='left', 
        family='monospace', bbox=props, wrap=True, linespacing=1.5)
        if hasattr(text_obj, "_get_wrap_line_width"):
            text_obj._get_wrap_line_width = lambda: 400
        #axes[0].text(0.1, 0.5, stats_text, fontsize=10, va='center', ha='left', family='monospace', wrap=True)
        
        axes[0].axis("off")

        # Bar chart
        if not value_counts.empty:
            colors = sns.color_palette("husl", len(value_counts))
            ax = value_counts.plot(kind='bar', color=colors, edgecolor='black', ax=axes[1])
            axes[0].set_title(f"Summary Stats for {varname.upper()}", fontsize=12)
            axes[1].set_title(f"Distribution of {varname.upper()}", fontsize=12)
            ax.set_xlabel("Categories")
            ax.set_ylabel("Count")

            # Add labels to the bars
            if len(value_counts)>4:
                rot = 90
            else:
                rot = 0
            for idx, value in enumerate(value_counts):
                percentage = (value / total) * 100
                ax.text(idx, value + total * 0.02, f"{value}({percentage:.1f}%)",
                        ha='center', fontsize=10, rotation = rot)

            # Adjust x-axis labels to be horizontal
            xticks = []
            for v in value_counts.index.astype(str):
                if v in category_mapping:
                    xticks.append(category_mapping[v])
                else:
                    xticks.append(v)

            if len(xticks)>4:
                ax.set_xticklabels(xticks, rotation=90, fontsize=10)
            else:
                ax.set_xticklabels(xticks, rotation=0, fontsize=10)

        
        plt.ylim(0, max(value_counts.values) * 1.4)
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")

    #x_ticks = [_.get_text() for _ in axes[1].get_xticklabels()]
    #x_tick_labels = axes[1].get_xticklables()
    #y_ticks =  [_.get_text() for _ in axes[1].get_yticklabels()]
    #y_tick_labels = axes[1].get_yticklabels()
    x_ticks = axes[1].get_xticklabels()
    y_ticks =  axes[1].get_yticklabels()
    return {"x-ticks": " - ".join([str(_) for _ in x_ticks]),
    # "x-labels": " - ".join([str(_) for _ in x_tick_labels]),
            "y-ticks": " - ".join([str(_) for _ in y_ticks]), 
        #"y-labels": " - ".join([str(_) for _ in y_tick_labels])
        }


def integrate_eda_with_metadata(vars_stats):
    meta_data = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")
    varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in meta_data.columns][0]
    metadata_vars = [x.lower().strip() for x in meta_data[varname_col].values]
    meta_data.columns = [c.strip() + " (metadata dictionary)" if c.upper() != varname_col else c.strip() for c in meta_data.columns]
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


def dataframe_to_json_dicts(df):
    varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in df.columns][0]
    json_dicts = {}
    for _, row in df.iterrows():
        variable_name = row[varname_col]
        var_dict = {}
        for col in df.columns:
            if col not in [varname_col, 'Column'] and pd.notna(row[col]) and row[col] != "" :
                try:
                    valu = row[col].lower().strip()
                except Exception:
                    valu = row[col]
                var_dict[col.lower()] = _convert_numeric(valu)
        json_dicts[variable_name] = var_dict
    with open("/output/eda_output_{cohort_id}.json", 'w', encoding='utf-8') as f:
        json.dump(json_dicts, f, indent=4)


def _convert_numeric(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

vars_to_stats = variable_eda(data, vars_details)
meta_data_enriched = integrate_eda_with_metadata(vars_to_stats)
json_dicts = dataframe_to_json_dicts(meta_data_enriched)

with open('/output/data_issues.json', 'w') as json_file:
    json.dump(data_issues, json_file, indent=4)
"""
    return raw_script.replace("{cohort_id}", cohort_id)
