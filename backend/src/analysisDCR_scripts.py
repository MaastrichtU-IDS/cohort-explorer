"""Analysis DCR embedded scripts for data fragmentation, visualization, and exploration."""


def generate_mapping_files_section(mapping_files: list[dict] = None, include_mapping_upload_slot: bool = False) -> str:
    """Generate the commented section with mapping file paths and pandas instructions.
    
    Args:
        mapping_files: List of mapping file info dicts with 'node_name' keys
        include_mapping_upload_slot: Whether a CrossStudyMappings upload slot is included
        
    Returns:
        A string containing the commented section for the visualization script
    """
    if not mapping_files and not include_mapping_upload_slot:
        return ""
    
    lines = [
        "",
        "###############################################################################",
        "# CROSS-STUDY MAPPING FILES",
        "# The following mapping files are available in this DCR.",
        "# Uncomment and use the code below to load them for cross-cohort analysis.",
        "###############################################################################",
        "",
        "# How to load JSON mapping files using pandas:",
        "# mapping_df = pd.read_json(mapping_path)",
        "#",
        "# The mapping files contain variable mappings between cohorts.",
        "# You can use them to harmonize data across different cohorts.",
        "",
    ]
    
    # Add each mapping file path
    if mapping_files:
        for i, mapping_info in enumerate(mapping_files, 1):
            node_name = mapping_info.get('node_name', f'mapping_{i}')
            lines.append(f"# Mapping file {i}: {node_name}")
            lines.append(f"# mapping_path_{i} = \"/input/{node_name}\"")
            lines.append(f"# mapping_df_{i} = pd.read_json(mapping_path_{i})")
            lines.append("")
    
    # Add upload slot if included
    if include_mapping_upload_slot:
        lines.append("# User-uploaded mapping file (CrossStudyMappings upload slot):")
        lines.append("# cross_study_mapping_path = \"/input/CrossStudyMappings\"")
        lines.append("# cross_study_mapping_df = pd.read_json(cross_study_mapping_path)")
        lines.append("")
    
    return "\n".join(lines)


def data_fragment_script(cohort_id: str, id_variable_name: str, airlock_percentage: int) -> str:
    """Generate the data fragmentation script for a cohort.
    
    This script:
    - Loads cohort data (CSV, SPSS, or ZIP)
    - Loads metadata dictionary (CSV only)
    - Removes ID column if specified
    - Splits data based on airlock percentage
    - Caps outliers using z-scores (2 std deviations)
    - Saves the fragment to output
    
    Args:
        cohort_id: The cohort identifier
        id_variable_name: Name of the ID column to remove (or empty string)
        airlock_percentage: Percentage of data to include in the fragment
        
    Returns:
        The Python script as a string
    """
    return f"""import pandas as pd
import numpy as np
import decentriq_util
import os
import zipfile
import tempfile

# Output directory (always exists in Decentriq environment)
output_dir = "/output"
log_file = os.path.join(output_dir, "fragmentation_log.txt")

# Helper function to load data from CSV, SPSS, or zipped files (for RawDataNodeDefinition)
def load_data(file_path):
    # Try CSV first
    try:
        return pd.read_csv(file_path)
    except Exception as csv_error:
        # Try SPSS
        try:
            return pd.read_spss(file_path)
        except Exception as spss_error:
            # Try as zip file
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    
                    # Find all CSV and SPSS files in the extracted directory
                    data_files = []
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            if file.endswith(('.csv', '.sav', '.CSV', '.SAV')):
                                data_files.append(os.path.join(root, file))
                    
                    if not data_files:
                        raise ValueError("No CSV or SPSS files found in the zip archive")
                    
                    # Read the first data file found
                    data_file = data_files[0]
                    if data_file.lower().endswith('.csv'):
                        return pd.read_csv(data_file)
                    else:
                        return pd.read_spss(data_file)
            except Exception as zip_error:
                raise ValueError("Could not read file as CSV, SPSS, or ZIP. CSV error: {{}}, SPSS error: {{}}, ZIP error: {{}}".format(csv_error, spss_error, zip_error))

# Helper function to load metadata dictionary (CSV only)
def load_metadata(file_path):
    return pd.read_csv(file_path)

# Read the cohort data (RawDataNodeDefinition - use file path)
df = load_data("/input/{cohort_id}")

with open(log_file, "a") as log:
    log.write("Loaded cohort data with {{}} rows and {{}} columns\\n".format(len(df), len(df.columns)))

# Read the metadata dictionary (CSV only)
try:
    metadata_df = load_metadata("/input/{cohort_id}_metadata_dictionary")
    with open(log_file, "a") as log:
        log.write("Loaded metadata dictionary with {{}} variables\\n".format(len(metadata_df)))
except Exception as e:
    metadata_df = None
    with open(log_file, "a") as log:
        log.write("Could not load metadata dictionary: {{}}\\n".format(e))

# ID column name is passed from the room creation code (discovered using SNOMED/OMOP codes)
# Replace ID column with synthetic IDs
# IMPORTANT: Rows with the same original ID must get the same synthetic ID
id_column_expected = "{id_variable_name if id_variable_name else ''}"

# Case-insensitive matching to find the actual column name in the data
id_column = None
if id_column_expected:
    id_col_lower = id_column_expected.lower().strip()
    for col in df.columns:
        if col.lower().strip() == id_col_lower:
            id_column = col
            break

with open(log_file, "a") as log:
    if id_column:
        # Store original position of ID column
        id_col_position = df.columns.get_loc(id_column)
        
        # Create a mapping from original IDs to synthetic IDs
        # This ensures rows with the same original ID get the same synthetic ID
        unique_ids = df[id_column].unique()
        id_mapping = {{orig_id: 'AIRLOCK_' + str(i).zfill(6) for i, orig_id in enumerate(unique_ids, start=1)}}
        
        # Map original IDs to synthetic IDs
        synthetic_ids = df[id_column].map(id_mapping)
        
        # Remove the original ID column
        df = df.drop(columns=[id_column])
        
        # Insert synthetic IDs at the same position
        df.insert(id_col_position, 'Synthetic_ID', synthetic_ids)
        
        log.write("Replaced ID column '{{}}' with synthetic IDs at position {{}}\\n".format(id_column, id_col_position))
        log.write("Mapped {{}} unique original IDs to synthetic IDs\\n".format(len(unique_ids)))
    else:
        # No ID column found - add synthetic IDs at the beginning (row-based, no grouping)
        synthetic_ids = ['AIRLOCK_' + str(i).zfill(6) for i in range(1, len(df) + 1)]
        df.insert(0, 'Synthetic_ID', synthetic_ids)
        if id_column_expected:
            log.write("Expected ID column '{{}}' not found in data columns, added row-based synthetic IDs\\n".format(id_column_expected))
        else:
            log.write("No ID column specified, added row-based synthetic IDs at position 0\\n")

# Airlock percentage setting
airlock_percentage = {airlock_percentage}

# Shuffle the dataframe to ensure random split
df_full = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split based on airlock percentage
split_fraction = airlock_percentage / 100.0
split_index = int(len(df_full) * split_fraction)
df_fragment = df_full.iloc[:split_index].copy()

# Identify numeric variables from metadata dictionary
# Numeric variables: VARTYPE in ['FLOAT', 'INT'] AND CATEGORICAL is empty
numeric_vars = []
if metadata_df is not None:
    # Find the column names (case-insensitive matching)
    varname_col = None
    vartype_col = None
    categorical_col = None
    
    for col in metadata_df.columns:
        col_lower = col.lower()
        if col_lower == 'var_name' or col_lower == 'varname':
            varname_col = col
        elif col_lower == 'var_type' or col_lower == 'vartype':
            vartype_col = col
        elif col_lower == 'categorical':
            categorical_col = col
    
    if varname_col and vartype_col:
        # Build case-insensitive mapping from data columns to actual column names
        df_cols_lower = {{col.lower().strip(): col for col in df_fragment.columns}}
        
        for _, row in metadata_df.iterrows():
            var_name = str(row[varname_col]).strip()
            var_type = str(row[vartype_col]).strip().upper() if pd.notna(row[vartype_col]) else ''
            categorical = str(row[categorical_col]).strip() if categorical_col and pd.notna(row[categorical_col]) else ''
            
            # Check if numeric: VARTYPE in ['FLOAT', 'INT'] and CATEGORICAL is empty
            if var_type in ['FLOAT', 'INT'] and (categorical == '' or categorical.lower() == 'nan'):
                # Case-insensitive column matching
                var_name_lower = var_name.lower().strip()
                if var_name_lower in df_cols_lower:
                    # Use the actual column name from the data
                    actual_col_name = df_cols_lower[var_name_lower]
                    numeric_vars.append(actual_col_name)

with open(log_file, "a") as log:
    log.write("\\nIdentified {{}} numeric variables for outlier capping\\n".format(len(numeric_vars)))

# Outlier capping using z-scores (2 standard deviations)
# Calculate statistics on FULL dataset, cap on fragment only
Z_THRESHOLD = 2.0
outlier_stats = []

for var in numeric_vars:
    try:
        # Convert to numeric, coercing errors to NaN
        full_values = pd.to_numeric(df_full[var], errors='coerce')
        fragment_values = pd.to_numeric(df_fragment[var], errors='coerce')
        
        # Calculate statistics on full dataset (excluding NaN)
        mean_val = full_values.mean()
        median_val = full_values.median()
        std_val = full_values.std()
        
        if pd.isna(std_val) or std_val == 0:
            # Skip if no variation
            continue
        
        # Calculate z-score cutoffs
        lower_limit = mean_val - (Z_THRESHOLD * std_val)
        upper_limit = mean_val + (Z_THRESHOLD * std_val)
        
        # Count outliers in fragment before capping
        outliers_below = (fragment_values < lower_limit).sum()
        outliers_above = (fragment_values > upper_limit).sum()
        total_capped = outliers_below + outliers_above
        
        # Cap outliers in the fragment
        df_fragment[var] = fragment_values.clip(lower=lower_limit, upper=upper_limit)
        
        # Log statistics
        outlier_stats.append({{
            'variable': var,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'lower_limit': lower_limit,
            'upper_limit': upper_limit,
            'capped_below': outliers_below,
            'capped_above': outliers_above,
            'total_capped': total_capped
        }})
    except Exception as e:
        with open(log_file, "a") as log:
            log.write("Error processing variable {{}}: {{}}\\n".format(var, e))

# Write outlier capping summary to log
with open(log_file, "a") as log:
    log.write("\\n=== Outlier Capping Summary (Z-score threshold: {{}}) ===\\n".format(Z_THRESHOLD))
    total_vars_capped = 0
    total_values_capped = 0
    for stat in outlier_stats:
        if stat['total_capped'] > 0:
            total_vars_capped += 1
            total_values_capped += stat['total_capped']
        log.write("\\nVariable: {{}}\\n".format(stat['variable']))
        log.write("  Mean: {{:.4f}}, Median: {{:.4f}}, Std: {{:.4f}}\\n".format(stat['mean'], stat['median'], stat['std']))
        log.write("  Lower limit (mean - 2*std): {{:.4f}}\\n".format(stat['lower_limit']))
        log.write("  Upper limit (mean + 2*std): {{:.4f}}\\n".format(stat['upper_limit']))
        log.write("  Values capped below: {{}}, above: {{}}, total: {{}}\\n".format(stat['capped_below'], stat['capped_above'], stat['total_capped']))
    log.write("\\nTotal: {{}} values capped across {{}} variables\\n".format(total_values_capped, total_vars_capped))

# Save the fragment to output
output_file = os.path.join(output_dir, "{cohort_id}_data_fragment.csv")
df_fragment.to_csv(output_file, index=False)

with open(log_file, "a") as log:
    log.write("\\nData fragment saved: {{}}\\n".format(output_file))
    log.write("Fragment size: {{}} rows out of {{}} total rows ({{:.1f}}%)\\n".format(len(df_fragment), len(df_full), len(df_fragment)/len(df_full)*100))
"""


def visualization_script(fragment_node_name: str, cohort_id: str, variable_names: list[str] = None, mapping_files: list[dict] = None, include_mapping_upload_slot: bool = False) -> str:
    """Generate the data visualization script.
    
    This script:
    - Reads the full cohort data (or preview/shuffled sample)
    - Selects 5 random columns (or user-specified columns)
    - Creates histograms for numeric data and bar charts for categorical data
    - Saves visualization to PNG
    - Includes commented section with mapping file paths and instructions
    
    Args:
        fragment_node_name: Name of the data fragment node to read data from
        cohort_id: The cohort identifier (used to locate the data CSV file)
        variable_names: Optional list of variable names from the cohort (for documentation)
        mapping_files: Optional list of mapping file info dicts with 'node_name' keys
        include_mapping_upload_slot: Whether a CrossStudyMappings upload slot is included
        
    Returns:
        The Python script as a string
    """
    # Generate the list of available variables (first 20, one per line for easy editing)
    if variable_names:
        vars_sample = variable_names[:20]
        vars_list = ",\n    ".join(f'"{v}"' for v in vars_sample)
    else:
        vars_list = '"var1", "var2", "var3"  # (variable list not available)'
    return f"""
###############################################################################
# USER-CONFIGURABLE SECTION
# Modify the settings below to customize the visualization
###############################################################################

# DATA SOURCE
# -----------
# Choose which data source to visualize by uncommenting ONE of the options below:
#
# Option 1: Raw cohort data (default) - the original unprocessed dataset
DATA_FILE = "/input/{cohort_id}"
#
# Option 2: Preview/Airlock sample - only the airlocked subset (e.g., 20%) of the processed data
# DATA_FILE = "/input/preview-airlock-{cohort_id}/{cohort_id}_data_fragment.csv"
#
# Option 3: Shuffled sample - synthetic/shuffled data for testing
# DATA_FILE = "/input/{cohort_id}_shuffled_sample"

# VARIABLE SELECTION
# ------------------
# Edit the list below to select which variables to visualize.
# Remove variables you don't want, or set to None for random selection.
SELECTED_VARIABLES = [
    {vars_list}
]

# CHART SIZE
# ----------
# Width of each output PNG file in centimeters. Height is auto-calculated (width / 2.5).
# - 10 = small
# - 15 = default size
# - 25 = large 
CHART_WIDTH_CM = 15

###############################################################################
# END OF USER-CONFIGURABLE SECTION
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Helper function to load data from CSV or SPSS files
def load_data(file_path):
    # Try CSV first
    try:
        return pd.read_csv(file_path)
    except Exception as csv_error:
        # Try SPSS
        try:
            return pd.read_spss(file_path)
        except Exception as spss_error:
            raise ValueError("Could not read file as CSV or SPSS. CSV error: {{}}, SPSS error: {{}}".format(csv_error, spss_error))

# Output directory (always exists in Decentriq environment)
output_dir = "/output"

HISTOGRAM_BINS = 30
MAX_CATEGORIES = 20
NUM_RANDOM_COLUMNS = 5


# Read the data from the selected source
df = load_data(DATA_FILE)

# Log basic info
log_file = os.path.join(output_dir, "visualization_log.txt")
with open(log_file, "w") as log:
    log.write("Data source: {{}}\\n".format(DATA_FILE))
    log.write("Loaded data: {{}} rows, {{}} columns\\n".format(len(df), len(df.columns)))
    log.write("Columns: {{}}\\n\\n".format(list(df.columns)))

# Try to load metadata dictionary to identify categorical variables
categorical_vars = set()
try:
    metadata_df = pd.read_csv("/input/{cohort_id}_metadata_dictionary")
    metadata_df.columns = metadata_df.columns.str.strip().str.upper()
    # Find the variable name column
    varname_col = next((c for c in metadata_df.columns if c in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME', 'VAR_NAME']), None)
    # Categorical column is always named CATEGORICAL (columns already uppercased)
    if varname_col:
        for _, row in metadata_df.iterrows():
            var_name = str(row[varname_col]).strip()
            cat_value = str(row['CATEGORICAL']).strip() if pd.notna(row['CATEGORICAL']) else ''
            # If categorical field has any non-empty value, treat as categorical
            if cat_value and cat_value.lower() not in ['', 'nan', 'none', 'n/a']:
                categorical_vars.add(var_name)
        with open(log_file, "a") as log:
            log.write("Loaded metadata: {{}} categorical variables identified\\n\\n".format(len(categorical_vars)))
except Exception as e:
    with open(log_file, "a") as log:
        log.write("Could not load metadata dictionary: {{}}\\n\\n".format(e))

# Determine which columns to visualize
if SELECTED_VARIABLES is not None:
    # Build a case-insensitive mapping from data columns
    df_cols_lower = {{col.lower().strip(): col for col in df.columns}}
    
    # Match selected variables to actual column names (case-insensitive)
    selected_columns = []
    missing_cols = []
    for var in SELECTED_VARIABLES:
        var_lower = var.lower().strip()
        if var_lower in df_cols_lower:
            selected_columns.append(df_cols_lower[var_lower])
        else:
            missing_cols.append(var)
    
    if missing_cols:
        with open(log_file, "a") as log:
            log.write("WARNING: The following requested columns were not found: {{}}\\n".format(missing_cols))
        
        # Create comparison dataframes showing column mismatches
        # This helps users understand the difference between metadata and actual data
        metadata_vars = set(v.lower().strip() for v in SELECTED_VARIABLES)
        data_cols = set(col.lower().strip() for col in df.columns)
        
        in_metadata_not_in_data = [v for v in SELECTED_VARIABLES if v.lower().strip() not in data_cols]
        in_data_not_in_metadata = [col for col in df.columns if col.lower().strip() not in metadata_vars]
        
        # Save comparison files
        pd.DataFrame({{'In Metadata Not in Data': in_metadata_not_in_data}}).to_csv(
            os.path.join(output_dir, "columns_in_metadata_not_in_data.csv"), index=False)
        pd.DataFrame({{'In Data Not in Metadata': in_data_not_in_metadata}}).to_csv(
            os.path.join(output_dir, "columns_in_data_not_in_metadata.csv"), index=False)
        
        with open(log_file, "a") as log:
            log.write("Column mismatch detected. Comparison files saved:\\n")
            log.write("  - columns_in_metadata_not_in_data.csv ({{}} columns)\\n".format(len(in_metadata_not_in_data)))
            log.write("  - columns_in_data_not_in_metadata.csv ({{}} columns)\\n\\n".format(len(in_data_not_in_metadata)))
    
    if not selected_columns:
        # No valid columns found - log error and exit gracefully instead of raising exception
        with open(log_file, "a") as log:
            log.write("ERROR: None of the specified columns exist in the data.\\n")
            log.write("Available columns in data: {{}}\\n".format(list(df.columns)))
            log.write("Requested columns: {{}}\\n".format(SELECTED_VARIABLES))
            log.write("\\nVisualization cannot proceed. Please check the column mismatch files for details.\\n")
        
        # Create an empty summary file to indicate the issue
        with open(os.path.join(output_dir, "visualization_error.txt"), "w") as f:
            f.write("Visualization could not be completed.\\n")
            f.write("None of the specified columns were found in the data.\\n")
            f.write("Please check visualization_log.txt and column mismatch CSV files for details.\\n")
        
        # Set selected_columns to empty list to skip visualization loop
        selected_columns = []
else:
    # Select random columns
    num_cols_to_visualize = min(NUM_RANDOM_COLUMNS, len(df.columns))
    selected_columns = random.sample(list(df.columns), num_cols_to_visualize)

with open(log_file, "a") as log:
    log.write("Selected {{}} columns for visualization: {{}}\\n\\n".format(len(selected_columns), selected_columns))

# Cohort name for filenames
COHORT_NAME = "{cohort_id}"

# Create visualizations - one image per variable
saved_files = []
for col in selected_columns:
    col_data = df[col].dropna()
    
    with open(log_file, "a") as log:
        log.write("Column: {{}}\\n".format(col))
        log.write("  Non-null values: {{}}\\n".format(len(col_data)))
        log.write("  Data type: {{}}\\n".format(df[col].dtype))
    
    # Create a new figure for each variable
    # Convert cm to inches for matplotlib (1 inch = 2.54 cm)
    chart_width_inches = CHART_WIDTH_CM / 2.54
    chart_height_inches = chart_width_inches / 2.5
    fig, ax = plt.subplots(figsize=(chart_width_inches, chart_height_inches))
    
    # Check if numeric or categorical
    # Treat as categorical if: (1) metadata says it's categorical, (2) it's binary (0/1), or (3) pandas says it's not numeric
    unique_vals = col_data.unique()
    is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({{0, 1, 0.0, 1.0}})
    is_categorical_in_metadata = col.lower().strip() in {{v.lower().strip() for v in categorical_vars}}
    
    if pd.api.types.is_numeric_dtype(col_data) and not is_binary and not is_categorical_in_metadata:
        # Histogram for numeric data
        ax.hist(col_data, bins=HISTOGRAM_BINS, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of {{}} (Numeric)'.format(col))
        
        # Add statistics
        mean_val = col_data.mean()
        median_val = col_data.median()
        std_val = col_data.std()
        ax.axvline(mean_val, color='red', linestyle='--', label='Mean: {{:.2f}}'.format(mean_val))
        ax.axvline(median_val, color='green', linestyle='--', label='Median: {{:.2f}}'.format(median_val))
        ax.legend()
        
        with open(log_file, "a") as log:
            log.write("  Mean: {{:.4f}}, Median: {{:.4f}}, Std: {{:.4f}}\\n".format(mean_val, median_val, std_val))
    else:
        # Bar chart for categorical/binary data
        value_counts = col_data.value_counts().head(MAX_CATEGORIES)
        ax.barh(range(len(value_counts)), value_counts.values)
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels([str(v)[:30] for v in value_counts.index])  # Truncate long labels
        ax.set_xlabel('Count')
        chart_type = 'Binary' if is_binary else ('Categorical' if is_categorical_in_metadata else 'Categorical, top {{}}'.format(MAX_CATEGORIES))
        ax.set_title('Distribution of {{}} ({{}})'.format(col, chart_type))
        
        with open(log_file, "a") as log:
            log.write("  Unique values: {{}}\\n".format(df[col].nunique()))
            log.write("  Top 5 values: {{}}\\n".format(dict(value_counts.head(5))))
    
    # Save individual image
    plt.tight_layout()
    output_filename = "{{}}_{{}}.png".format(col, COHORT_NAME)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(output_filename)
    
    with open(log_file, "a") as log:
        log.write("  Saved to: {{}}\\n\\n".format(output_filename))

with open(log_file, "a") as log:
    log.write("Visualization complete. {{}} images saved: {{}}\\n".format(len(saved_files), saved_files))
{generate_mapping_files_section(mapping_files, include_mapping_upload_slot)}
"""


def exploration_script() -> str:
    """Generate the basic data exploration script.
    
    This script:
    - Lists all files in the input directory
    - Shows file sizes and modification dates
    - Attempts to load each file as tabular data and display info
    
    Returns:
        The Python script as a string
    """
    return """import pandas as pd
import decentriq_util
import os
from datetime import datetime

# Get all files in the /input directory
input_dir = "/input"
output_dir = "/output"
files = os.listdir(input_dir)

# Output directory (always exists in Decentriq environment)

# Open output file for writing
output_file = os.path.join(output_dir, "data_exploration_report.txt")
with open(output_file, "w") as f:
    f.write("=" * 80 + "\\n")
    f.write("BASIC DATA EXPLORATION\\n")
    f.write("=" * 80 + "\\n")
    f.write("\\n")
    
    for filename in sorted(files):
        filepath = os.path.join(input_dir, filename)
        
        # Skip if not a file
        if not os.path.isfile(filepath):
            continue
        
        f.write("\\n" + "=" * 80 + "\\n")
        f.write("FILE: {}\\n".format(filename))
        f.write("=" * 80 + "\\n")
        
        # File size in KB
        file_size_bytes = os.path.getsize(filepath)
        file_size_kb = file_size_bytes / 1024
        f.write("Size: {:.2f} KB ({:,} bytes)\\n".format(file_size_kb, file_size_bytes))
        
        # Last modified date
        mod_timestamp = os.path.getmtime(filepath)
        mod_date = datetime.fromtimestamp(mod_timestamp)
        f.write("Last Modified: {}\\n".format(mod_date.strftime('%Y-%m-%d %H:%M:%S')))
        
        # Try to load as tabular data and display info
        try:
            df = decentriq_util.read_tabular_data(filename)
            f.write("\\nDataFrame Info:\\n")
            f.write("  Rows: {:,}\\n".format(len(df)))
            f.write("  Columns: {:,}\\n".format(len(df.columns)))
            f.write("  Column names: {}\\n".format(list(df.columns)))
            
            f.write("\\nFirst 5 rows:\\n")
            f.write(df.head(5).to_string() + "\\n")
            
        except Exception as e:
            f.write("\\nCould not load as tabular data: {}\\n".format(e))
        
        f.write("\\n")
    
    f.write("=" * 80 + "\\n")
    f.write("EXPLORATION COMPLETE\\n")
    f.write("=" * 80 + "\\n")
    f.write("\\nReport written to {}\\n".format(output_file))
"""
