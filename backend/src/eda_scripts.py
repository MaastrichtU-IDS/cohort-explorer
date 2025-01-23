"""Exploratory Data Analysis (EDA) module."""

c1_data_dict_check = """
import pandas as pd
import decentriq_util

# load the TIME-CHF dictionary
dictionary_df = decentriq_util.read_tabular_data("/input/TIME_CHF_metadatadictionary")

# load the TIME-CHF dataset
dataset_df = pd.read_spss("/input/TIME_CHF_dataset")


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
pd.DataFrame({'In Dataset Not in Dictionary': list(in_dataset_not_in_dictionary)}).to_csv("/output/in_dataset_not_in_dictionary.csv",index = False)
pd.DataFrame({'In Dictionary Not in Dataset': list(in_dictionary_not_in_dataset)}).to_csv("/output/in_dictionary_not_in_dataset.csv",index = False)
"""

c2_save_to_json = """
import decentriq_util
import pandas as pd
import os
import json

# Load dictionary
dictionary = decentriq_util.read_tabular_data("/input/TIME_CHF_metadatadictionary")

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
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    class_names[key] = value

                    # Check if the value indicates a missing category
                    if any(keyword in value for keyword in ('missing')):
                        missing_key = key

        # Save MIN, MAX, and VAR TYPE values if they exist to class_details
        class_details[variable_name] = {
            'categories': class_names,

            'missing': missing_key,  # Add missing indicator if found

            'min': min_value if pd.notna(min_value) else None,
            'max': max_value if pd.notna(max_value) else None,
            'var_type': var_type if pd.notna(var_type) else None
        }

    elif pd.isna(categories_info) or categories_info.strip() == '':
        # Handle numerical variables
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
print(f"Categorical variables saved to {categorical_json_path}")
print(json.dumps({key: class_details[key] for key in list(class_details.keys())[:5]}, indent=4))

print(f"Numerical variables saved to {numerical_json_path}")
print(json.dumps({key: numerical_details[key] for key in list(numerical_details.keys())[:5]}, indent=4))
"""

c3_map_missing_do_not_run = """
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
df = pd.read_spss("/input/TIME_CHF_dataset")

# Step 3: Normalize column names in the dataset to lowercase to match JSON keys
df.columns = df.columns.str.lower()  # Convert column names to lowercase for consistency

# Normalize keys in the JSON files to lowercase for consistent comparison
numerical_data = {k.lower(): v for k, v in numerical_data.items()}
categorical_data = {k.lower(): v for k, v in categorical_data.items()}

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
                print(f"Skipping range validation for column: {column} (var_type: {var_type})")


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
                    category_mapping = {str(k).strip(): v for k, v in categories.items()}
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
        print(f"Error occurred in column: {column}")
        print(f"Problematic value: {df[column].dropna().unique()}")
        raise e

# Step 5: Save the cleaned dataset
df.to_csv("/output/data_correct.csv",index = False, header = True)
"""

c3_eda_data_profiling = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, zscore
import warnings
warnings.filterwarnings('ignore')

# EDA report for each variable


# Load the dataset with corrected missing values replaced with NA from previous step
data_correct_missing = pd.read_csv("/input/C3_map_missing_do_not_run/data_correct.csv", low_memory=False)

#Load the JSON files from C2
categorical_data = pd.read_json("/input/C2_Save_to_JSON/categorical_variables.json")
numerical_data = pd.read_json("/input/C2_Save_to_JSON/numerical_variables.json")


def variable_eda(df):
    for column in df.columns.tolist():

        # Continuous variables
        if column in list(numerical_data.keys()):

            if not pd.api.types.is_numeric_dtype(df[column]):
                # Skip if the column is not numeric
                continue

            # Descriptive Stats
            stats = df[column].describe()
            mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            total_missing = df[column].isnull().sum()
            missing_percent = total_missing / len(df) * 100

            # Check for numeric values before computing skewness and kurtosis
            if len(df[column].dropna()) > 0:
                skewness = skew(df[column].dropna(), bias=False)
                kurt = kurtosis(df[column].dropna(), bias=False)
            else:
                skewness = np.nan
                kurt = np.nan

            # Normality Test
            if len(df[column].dropna()) > 3:  # Shapiro requires at least 3 values
                stat, p = shapiro(df[column].dropna())
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
                f"Count:               {stats['count']:.0f}\n"
                f"Mean:                {stats['mean']:.2f}\n"
                f"Median:              {stats['50%']:.2f}\n"
                f"Mode:                {mode_value:.2f}\n"
                f"Std Dev:             {stats['std']:.2f}\n"
                f"Variance:            {stats['std']**2:.2f}\n"
                f"Range:               {range_value:.2f}\n"
                f"Q1:                  {Q1:.2f}\n"
                f"Q3:                  {Q3:.2f}\n"
                f"IQR:                 {IQR:.2f}\n"
                f"Missing:             {total_missing} ({missing_percent:.2f}%)\n"
                f"Outliers (IQR):      {outliers} ({(outliers / len(df) * 100):.2f}%)\n"
                f"Outliers (Z):        {z_outliers}\n"
                f"Skewness:            {skewness:.2f}\n"
                f"Kurtosis:            {kurt:.2f}\n"
                f"Normality Test:\n p-value={p_value_str} => {normality}\n"
                f"Data Type:           {df[column].dtype}\n"
            )

            # Create figure for the current feature
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Left: Display Summary Stats
            axes[0].text(0.05, 0.9, stats_text, fontsize=10, va='top', ha='left', linespacing=1.2, family='monospace', wrap=True)
            axes[0].axis("off")

            # Right: Plot histogram
            sns.histplot(df[column].dropna(), kde=True, ax=axes[1])
            axes[1].set_title(f"Histogram for {column}")
            axes[1].tick_params(axis='x')

            # Save the figure for the current feature
            plt.tight_layout()
            plt.savefig(f"/output/eda_{column}.png")
            plt.close()



        # Categorical variables
        elif column in list(categorical_data.keys()):
            value_counts = df[column].value_counts(dropna=False)
            total = len(df)

            # Get the categories mapping and normalize keys
            categories_mapping = categorical_data[column].get("categories", {})
            categories_mapping = {str(k): v for k, v in categories_mapping.items()}

            if value_counts.empty:
                stats_text = (
                    f"Column: {column}\n"
                    f"Number of Unique Categories: 0\n"
                    f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)\n"
                )
            else:
                # Chi-square test
                expected = total / len(value_counts)
                chi_square_stat = ((value_counts - expected) ** 2 / expected).sum()

                # Class balance with corrected mapping
                class_balance_text = "\n".join([
                    f"{key} = {categories_mapping.get(str(key).split('.')[0], 'Unknown')}: {count / total * 100:.2f}%"
                    for key, count in value_counts.items()
                ])

                stats_text = (
                    f"Column: {column}\n"
                    f"Number of Unique Categories: {len(value_counts)}\n"
                    f"Most Frequent Category: {categories_mapping.get(str(value_counts.idxmax()).split('.')[0], 'Unknown')} "
                    f"({value_counts.max()} observations)\n"
                    f"Missing Values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)\n"
                    f"Class Balance:\n{class_balance_text}\n"
                    f"Chi-Square Test Statistic: {chi_square_stat:.2f}\n"
                )

            # Plot summary and bar chart
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Summary stats text
            axes[0].text(0.1, 0.5, stats_text, fontsize=10, va='center', ha='left', family='monospace', wrap=True)
            axes[0].axis("off")

            # Bar chart
            if not value_counts.empty:
                colors = sns.color_palette("husl", len(value_counts))
                ax = value_counts.plot(kind='bar', color=colors, edgecolor='black', ax=axes[1])
                ax.set_title(f"Distribution in {column}")
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
            plt.savefig(f"/output/eda_{column}.png")
            plt.close()


variable_eda(data_correct_missing)
"""
