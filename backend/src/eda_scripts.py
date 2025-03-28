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
    dataset_df = pd.read_spss("/input/{cohort_id}")
    
    

# Extract 'VARIABLE NAME' column from dictionary and dataset column names
dictionary_variables = set(dictionary_df[varname_col].unique())
dataset_columns = set(dataset_df.columns)
print("\\\n\\\n\\\nDataset columns: ", dataset_columns)

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
    raw_script = """
import decentriq_util
import pandas as pd
import os
import json



def _column_is_date(series):
    try:
        pd.to_datetime(series)
        return True
    except (ValueError, TypeError):
        return False

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
    data = pd.read_csv("/input/{cohort_id}")
except Exception as e:
    data = pd.read_spss("/input/{cohort_id}")

data.columns = [c.lower().strip() for c in data.columns]
for col in data.columns:
    if data[col].dropna().apply(lambda x: str(x).isdigit() or str(x).endswith('.0')).all():
        data[col] = data[col].astype('Int64')

# Define the pattern for entries to exclude non-categorical variables
#include_pattern = r'\||='   # Look for strings containing either a | or =.
# Exclude rows in the dictionary where the 'CATEGORICAL' column contains the defined pattern
# categorical_dict = dictionary[dictionary['CATEGORICAL'].astype(str).str.contains(include_pattern, regex=True)]

vars_to_process = {}
# Prepare to extract classes and their meanings, along with MIN, MAX, and VAR TYPE
class_details = {}
numerical_details = {}

for index, row in dictionary.iterrows():
    variable_name = row[varname_col]
    var_type = row[vartype_col]
    categories_info = row['CATEGORICAL']

    if variable_name.lower() in ['patientid', 'pat.id']:
        continue

    if (pd.notna(categories_info) and isinstance(categories_info, str) and categories_info.strip() != ""):
        vars_to_process[variable_name] = 'categorical'
    elif data[variable_name].dropna().apply(lambda x: str(x).isdigit() or str(x).endswith('.0')).all():
        vars_to_process[variable_name] = 'numeric'
    elif _column_is_date(data[variable_name]):
        vars_to_process[variable_name] = 'date'
    else:
        #assume categorical:
        print("The following variable deemed categorical by process of elimination: ", variable_name)
        vars_to_process[variable_name] = 'categorical'



for index, row in dictionary.iterrows():
    variable_name = row[varname_col]
    if variable_name not in vars_to_process:
        continue
    else:
        t = vars_to_process[variable_name]
    categories_info = row['CATEGORICAL']

    if t == 'categorical':
        categories = [item for sublist in categories_info.split('|') for item in sublist.split(',')]
        class_names = {}

        for category in categories:
                key_value = category.lower().split('=')
                if len(key_value) == 2:
                    #print("inside if statement: ", variable_name, key_value)
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    class_names[key] = value
                elif len(key_value) == 1:  
                    #category does not have "="
                    class_names[key_value[0].strip()] = key_value[0].strip()
                else:
                    print("Encountered a possible parsing error. Check category info for variable ", variable_name, key_value)


        # Check if there is a value that corresponds to  'missing'
        if 'missing' in class_names.values():
            missing_key = [x[0] for x in class_names.items() if x[1].strip().lower() == 'missing'][0]
            print("MISSING value exists for variable: ", variable_name, missing_key)
        else:
            print("No 'missing' value for variable ", variable_name)
            #needed the line below, otherwise the "missing_key" will still store the value for the previous var
            missing_key = None
                
        # Save MIN, MAX, and VAR TYPE values if they exist to class_details
        class_details[variable_name] = {
            'categories': class_names,
            'var_label': row[varlabel_col],
            'missing': missing_key,  # Add missing indicator if found
            'var_type': t
        }

    else: #numeric or dates:
        numerical_details[variable_name] = {
            'var_type': t,
            'var_label': row[varlabel_col],
            'missing': row['MISSING'] if 'MISSING' in dictionary.columns and row['MISSING'].strip() != "" else None
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
print(f"Categorical variables saved to {categorical_json_path}")
print(json.dumps({key: class_details[key] for key in list(class_details.keys())[:-1]}, indent=4))

print(f"Numerical variables saved to {numerical_json_path}")
print(json.dumps({key: numerical_details[key] for key in list(numerical_details.keys())[:-1]}, indent=4))
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
warnings.filterwarnings('ignore')



# Load the dataset with corrected missing values replaced with NA from previous step
# data_correct_missing = pd.read_csv("/input/C3_map_missing_do_not_run/data_correct.csv", low_memory=False)

#Load the JSON files from C2
categorical_vars = pd.read_json("/input/c2_save_to_json/categorical_variables.json")
#print("the categorical data: ", categorical_vars.keys())
numerical_vars = pd.read_json("/input/c2_save_to_json/numerical_variables.json")
#print("the numerical data: ", numerical_vars)
try:
    data = pd.read_csv("/input/{cohort_id}")
    #data = decentriq_util.read_tabular_data("/input/{cohort_id}")
except Exception as e:
    data = pd.read_spss("/input/{cohort_id}")


data.columns = [c.lower().strip() for c in data.columns]
for col in data.columns:
    if data[col].dropna().apply(lambda x: str(x).isdigit() or str(x).endswith('.0')).all():
        data[col] = data[col].astype('Int64')

#variables that should be graphed
#vars_to_graph = ['age', 'weight', 'cough1', 'angina1', 'hscrp_v6']
#vars_to_graph = ['age', 'ALCOOL', 'ALLOPURI', 'ALT', 'ALTACE', 'ALTANO', 'COLETOT', 'CREATIN', 'DALTACE', 'DATAECG', 'DATALAB']
#vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB']
#vars_to_graph = [x.lower() for x in vars_to_graph]
#vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB', 'AATHORAX', 'AATHORAXDIM', 'ACE_AT_V1']
vars_to_graph = list(categorical_vars.columns) + list(numerical_vars.columns)
vars_to_graph = [x.strip().lower() for x in vars_to_graph]



def _lowercase_if_string(x):
    if isinstance(x, str):
        return x.lower()
    return x

def variable_eda(df, categorical_vars, numerical_vars):
    vars_stats = {}
    graph_tick_data = {}
    df.columns = df.columns.str.lower().str.strip()
    for column in df.columns.tolist():
        # Continuous variables
        if column in numerical_vars.keys():

            #if not pd.api.types.is_numeric_dtype(df[column]):
                # Skip if the column is not numeric
                # print("Column ", column, " skipped because non-numeric")
                #continue

            # Descriptive Stats
            stats = df[column].describe()
            mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            value_counts = df[column].value_counts(dropna=False)
            if numerical_vars[column]['missing'] == None:
                total_missing = 0
            else:
                total_missing = value_counts.get(numerical_vars[column]['missing'], 0)
            missing_percent = total_missing / len(df) * 100
            try:
                empty = df[column].isnull().sum() + df[column].str.strip().eq('').sum()
            except:
                empty = df[column].isnull().sum()

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
                f"Column: {column}",
                f"Label: {numerical_vars[column]['var_label']}",
                f"Type: Numeric (encoded as {df[column].dtype})",
                f"Number of non-null observations: {int(stats['count'])}",
                f"Number of Unique Values/Categories: {df[column].nunique()}",
                f"Mean:                {stats['mean']:.2f}",
                f"Median:              {stats['50%']:.2f}",
                f"Mode:                {mode_value:.2f}",
                f"Std Dev:             {stats['std']:.2f}",
                f"Variance:            {stats['std']**2:.2f}",
                f"Range:               {range_value:.2f}", 
                f"Q1:                  {Q1:.2f}",
                f"Q3:                  {Q3:.2f}",
                f"IQR:                 {IQR:.2f}",
                f"Count empty:         {empty} ({(empty/len(df[column])) * 100:.2f}%)",
                f"Count missing:       {total_missing} ({(total_missing/len(df[column])) * 100:.2f}%)",
                f"Outliers (IQR):      {outliers} ({(outliers / len(df)) * 100:.2f}%)",
                f"Outliers (Z):        {z_outliers}",
                f"Skewness:            {skewness:.2f}",
                f"Kurtosis:            {kurt:.2f}",
                f"W_Test:              {w_test_stat:.2f}",
                f"Normality Test: p-value={p_value_str} => {normality}"
            )

            if column in vars_to_graph:
                graph_tick_data[column] = create_save_graph(df, column, stats_text, 'numerical')
                


        # Categorical variables
        elif column in categorical_vars.keys() and categorical_vars[column]['var_type'] != 'date':
            stats = df[column].describe()
            value_counts = df[column].apply(_lowercase_if_string).value_counts(dropna=False)
            total = len(df)

            # Get the categories mapping and normalize keys
            categories_mapping = categorical_vars[column].get("categories", [])
            categories_mapping = {str(k): v for (k, v) in categories_mapping.items()}
            print("variable: ", column, "categories:", categories_mapping, value_counts, )

            if value_counts.empty:
                stats_text = (
                    f"Column: {column}",
                    f"Label: {categorical_vars[column]['var_label']}",
                    f"Type: Categorical (encoded as {df[column].dtype})",
                    f"Number of Unique Categories: 0",
                    f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)"
                )
            else:
                # Chi-square test
                expected = total / len(value_counts)
                chi_square_stat = ((value_counts - expected) ** 2 / expected).sum()
                if categorical_vars[column]['missing'] == None:
                    missing_count = 0
                else:
                    missing_count = value_counts.get(categorical_vars[column]['missing'], 0)
                
                try:
                    empty_count = df[column].isnull().sum() + df[column].str.strip().eq('').sum()
                except:
                    empty_count = df[column].isnull().sum()

                # Class balance with corrected mapping
                class_balance_text = "\\n\\t"
                for key, count in value_counts.items():
                    if str(key) in categories_mapping and str(key) != categories_mapping[str(key)]:
                        class_balance_text += f"{(key, categories_mapping[str(key)])} -> {round(count / total * 100)}%\\n\\t"
                    else:
                        class_balance_text += f"{key} -> {round(count / total * 100)}%\\n\\t"

                stats_text = (
                    f"Column: {column}",
                    f"Label: {categorical_vars[column]['var_label']}",
                    f"Type: Categorical (encoded as {df[column].dtype})",
                    f"Number of unique values/categories: {len(value_counts)}",
                    f"Most frequent category: {categories_mapping.get(str(value_counts.idxmax()), 'Unknown')} ",
                    f"Number of non-null observations: {df[column].count()}",
                    f"Code for missing value: {categorical_vars[column]['missing']}",
                    f"Count missing: {missing_count} ({(missing_count/len(df[column])) * 100:.2f}%)",
                    f"Count empty: {empty_count} ({(empty_count/len(df[column])) * 100:.2f}%)",
                    f"Class balance: {class_balance_text}",
                    f"Chi-Square Test Statistic: {chi_square_stat:.2f}"
                )

            if column in vars_to_graph:
                graph_tick_data[column] = create_save_graph(df, column, stats_text, 'categorical', category_mapping = categories_mapping)
        
        elif column in categorical_vars.keys() and categorical_vars[column]['var_type'] == 'date':
            try:
                stats = pd.to_datetime(df[column], format='mixed').describe()
            except:
                continue
            value_counts = df[column].value_counts(dropna=False)
            total = len(df)
            if categorical_vars[column]['missing'] == None:
                missing_count = 0
            else:
                missing_count = value_counts.get(categorical_vars[column]['missing'], 0)

            try:
                    empty_count = df[column].isnull().sum() + df[column].str.strip().eq('').sum()
            except:
                    empty_count = df[column].isnull().sum()
            stats_text = [
                    f"Column: {column}",
                    f"Label: {categorical_vars[column]['var_label']}",
                    f"Type: Date (encoded as {df[column].dtype})",
                    f"Number of unique values: {len(value_counts)}",
                    f"Most frequent value: {str(value_counts.idxmax()).split('.')[0]}",
                    f"Number of non-null observations: {df[column].count()}",
                    f"Count missing: {missing_count} ({(missing_count/len(df[column])) * 100:.2f}%)",
                    f"Count empty: {empty_count} ({(empty_count/len(df[column])) * 100:.2f}%)"
            ]
            stats_text.extend([f"{k.capitalize()}: {v}" for k,v in stats.items()])
            if column in vars_to_graph:
                graph_tick_data[column] = create_save_graph(df, column, stats_text, 'datetime')
        else:
            #print("ELSE case: variable name ", column, "var type: ", categorical_vars[column]['var_type'])
            stats_text = []
        stats_text_dict = {i.split(":")[0].strip():i.split(":")[1].strip() for i in stats_text}
        if 'Class balance' in stats_text_dict:
            stats_text_dict['Class balance'].replace(" ->", ":")
        stats_text_dict['url'] = f"https://explorer.icare4cvd.eu/api/variable-graph/{cohort_id}/{column}"
        vars_stats[column] = stats_text_dict
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
            text_obj._get_wrap_line_width = lambda: 400
        #axes[0].text(0.05, 0.9, , fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
        axes[0].axis("off")

        # Right: Plot histogram
        sns.histplot(df[varname].dropna(), kde=True, ax=axes[1])
        axes[0].set_title(f"Summary Stats for {varname.upper()}", fontsize=12)
        axes[1].set_title(f"Distribution of {varname.upper()}", fontsize=12)
        axes[1].tick_params(axis='x')

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
            text_obj._get_wrap_line_width = lambda: 400
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
    
        if date_range.days > 365 :
            bin_freq = 'Q'  # Quarterly
        elif date_range.days > 90 :
            bin_freq = 'M'  
        elif date_range.days > 30:
            bin_freq = 'W'  # Weekly bins
        else:
            bin_freq = 'D'  # Daily
        
        bins = mdates.date2num(pd.date_range(min_date, max_date, freq=bin_freq))
        
        axes[1].hist(date_nums, bins=bins, alpha=0.7)
        axes[1].set_title(f"Distribution of {varname.upper()}", fontsize=12)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
        #if date_range.days > 365*2:
        #    axes[1].xaxis.set_major_locator(mdates.YearLocator())
        if date_range.days > 180:
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Quarterly
        else:
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())  # Monthly
            
        axes[1].tick_params(axis='x', rotation=90)
        
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
            for idx, value in enumerate(value_counts):
                percentage = (value / total) * 100
                ax.text(idx, value + total * 0.02, f"{value}({percentage:.1f}%)",
                        ha='center', fontsize=10)

            # Adjust x-axis labels to be horizontal
            xticks = []
            for v in value_counts.index.astype(str):
                if v in category_mapping:
                    xticks.append(category_mapping[v])
                else:
                    xticks.append(v)
            ax.set_xticklabels(xticks, rotation=90, fontsize=10)

        
        plt.ylim(0, max(value_counts.values) * 1.1)
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png")
        #print(f"figure for {varname} saved!! ")

    x_ticks = axes[1].get_xticks()
    #x_tick_labels = axes[1].get_xticklabels()
    y_ticks = axes[1].get_yticks()
    #y_tick_labels = axes[1].get_yticklabels()
    return {"x-ticks": " - ".join([str(_) for _ in x_ticks]),
    # "x-labels": " - ".join([str(_) for _ in x_tick_labels]),
            "y-ticks": " - ".join([str(_) for _ in y_ticks]), 
        #"y-labels": " - ".join([str(_) for _ in y_tick_labels])
        }


def integrate_eda_with_metadata(vars_stats):
    meta_data = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")
    varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in meta_data.columns][0]
    metadata_vars = [x.lower().strip() for x in meta_data[varname_col].values]
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
                var_dict[col] = row[col]
        json_dicts[variable_name] = var_dict
    with open("/output/eda_output_{cohort_id}.json", 'w', encoding='utf-8') as f:
        json.dump(json_dicts, f, indent=4)


vars_to_stats = variable_eda(data, categorical_vars, numerical_vars)
meta_data_enriched = integrate_eda_with_metadata(vars_to_stats)
json_dicts = dataframe_to_json_dicts(meta_data_enriched)
"""
    return raw_script.replace("{cohort_id}", cohort_id)
