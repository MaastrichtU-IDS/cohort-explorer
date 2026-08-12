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
        "# Copy this code to the 'Development' tab, uncomment the code below as needed to incorporate mapping information into your analysis",
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
            lines.append(f"# mapping_path_{i} = \"/input/{node_name}\"")
            lines.append(f"# mapping_df_{i} = pd.read_json(mapping_path_{i})")
            lines.append("")
    
    # Add upload slot if included
    if include_mapping_upload_slot:
        lines.append("# cross_study_mapping_path = \"/input/CrossStudyMappings\"  #user-uploaded mapping file")
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


def visualization_script(
    fragment_node_name: str,
    cohort_id: str,
    variable_names: list[str] = None,
    mapping_files: list[dict] = None,
    include_mapping_upload_slot: bool = False,
    data_source: str = "full",
) -> str:
    """Generate the data visualization script.

    This script:
    - Reads the selected data source (full dataset, airlock sample, or shuffled sample)
    - Selects user-specified columns
    - Creates histograms for numeric data and bar charts for categorical data
    - Saves visualization to PNG
    - Includes commented section with mapping file paths and instructions

    Args:
        fragment_node_name: Name of the data fragment node (None if no airlock).
        cohort_id: The cohort identifier (used to locate the data CSV file).
        variable_names: Optional list of variable names from the cohort.
        mapping_files: Optional list of mapping file info dicts with 'node_name' keys.
        include_mapping_upload_slot: Whether a CrossStudyMappings upload slot is included.
        data_source: Which data source to select by default. One of:
            - "full"     : the full dataset (original cohort data) [default]
            - "shuffled" : the shuffled sample
            - "airlock"  : the airlock sample

    Returns:
        The Python script as a string.
    """
    # Generate the list of available variables (first 20, one per line for easy editing)
    if variable_names:
        vars_sample = variable_names[:20]
        vars_list = ",\n    ".join(f'"{v}"' for v in vars_sample)
    else:
        vars_list = '"var1", "var2", "var3"  # (variable list not available)'

    # Normalise data_source and demote "airlock" to "full" if there is no airlock node
    if data_source not in ("full", "shuffled", "airlock"):
        data_source = "full"
    has_airlock = bool(fragment_node_name)
    if data_source == "airlock" and not has_airlock:
        data_source = "full"

    # "" => uncommented (selected) ; "# " => commented (not selected)
    p_full = "" if data_source == "full" else "# "
    p_shuf = "" if data_source == "shuffled" else "# "
    p_air = "" if data_source == "airlock" else "# "

    if has_airlock:
        airlock_block = (
            f'# Option 2: Airlock sample - only the airlocked subset (e.g., 20%) of the processed data\n'
            f'{p_air}DATA_FILE = "/input/preview-airlock-{cohort_id}/{cohort_id}_data_fragment.csv"\n'
            f'{p_air}DATA_SOURCE_NAME = "{cohort_id} airlock sample"\n'
        )
    else:
        airlock_block = (
            "# Option 2: Airlock sample - (no airlock node was provisioned for this cohort)\n"
        )

    return f"""
###############################################################################
# USER-CONFIGURABLE SECTION
# Modify the settings below to customize the visualization
###############################################################################

# DATA SOURCE
# -----------
# Choose which data source to visualize by uncommenting ONE of the options below.
# DATA_SOURCE_NAME is shown on each chart so it is always clear which data
# source the visualization is based on.
#
# Option 1: Full dataset - the original unprocessed cohort dataset
{p_full}DATA_FILE = "/input/{cohort_id}"
{p_full}DATA_SOURCE_NAME = "{cohort_id} full dataset"
#
{airlock_block}#
# Option 3: Shuffled sample - synthetic/shuffled data for testing
{p_shuf}DATA_FILE = "/input/{cohort_id}_shuffled_sample"
{p_shuf}DATA_SOURCE_NAME = "{cohort_id} shuffled sample"

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

# Helper function to load data from CSV or SPSS files
def load_data(file_path, source_name):
    # Try CSV first
    try:
        return pd.read_csv(file_path)
    except Exception as csv_error:
        # Try SPSS
        try:
            return pd.read_spss(file_path)
        except Exception as spss_error:
            # Surface a clear, actionable error when the data source is not available
            raise FileNotFoundError(
                'Data source "' + source_name + '" was not found at path "' + file_path + '". '
                'Please check that it has been provisioned. '
                'If you are in "Development Mode", please check that you have selected it '
                'from the rightside drop-down menu. '
                '(CSV error: ' + str(csv_error) + ', SPSS error: ' + str(spss_error) + ')'
            )

# Output directory (always exists in Decentriq environment)
output_dir = "/output"

HISTOGRAM_BINS = 30
MAX_CATEGORIES = 20

df = load_data(DATA_FILE, DATA_SOURCE_NAME)

# Log basic info
log_file = os.path.join(output_dir, "visualization_log.txt")
with open(log_file, "w") as log:
    log.write("Data source name: {{}}\\n".format(DATA_SOURCE_NAME))
    log.write("Data source file: {{}}\\n".format(DATA_FILE))
    log.write("Loaded data: {{}} rows, {{}} columns\\n".format(len(df), len(df.columns)))
    log.write("Columns: {{}}\\n\\n".format(list(df.columns)))

# Load metadata dictionary for categorical detection and variable labels
categorical_vars = set()
metadata_df = None
try:
    metadata_df = pd.read_csv("/input/{cohort_id}_metadata_dictionary")
    with open(log_file, "a") as log:
        log.write("Metadata columns (before upper): {{}}\\n".format(list(metadata_df.columns)))
    metadata_df.columns = metadata_df.columns.str.strip().str.upper()
    with open(log_file, "a") as log:
        log.write("Metadata columns (after upper): {{}}\\n".format(list(metadata_df.columns)))
    # Build categorical set
    if 'CATEGORICAL' in metadata_df.columns:
        varname_col = 'VARIABLENAME' if 'VARIABLENAME' in metadata_df.columns else 'VARIABLE NAME'
        for _, row in metadata_df.iterrows():
            cat_val = str(row.get('CATEGORICAL', '')).strip().lower()
            if cat_val and cat_val not in ['', 'nan', 'none', 'n/a']:
                categorical_vars.add(str(row[varname_col]).strip())
    with open(log_file, "a") as log:
        log.write("Loaded metadata dictionary: {{}} rows, {{}} categorical vars\\n".format(len(metadata_df), len(categorical_vars)))
        # Show first row to debug
        if len(metadata_df) > 0:
            log.write("First row sample: {{}}\\n\\n".format(dict(metadata_df.iloc[0])))
except Exception as e:
    with open(log_file, "a") as log:
        log.write("Could not load metadata dictionary: {{}}\\n\\n".format(e))

def get_var_metadata(var_name, column):
    \"\"\"Get a metadata field from the metadata dictionary for a given variable name.\"\"\"
    if metadata_df is None:
        return ''
    varname_col = 'VARIABLENAME' if 'VARIABLENAME' in metadata_df.columns else 'VARIABLE NAME'
    if column not in metadata_df.columns:
        return ''
    mask = metadata_df[varname_col].str.strip().str.lower() == var_name.lower().strip()
    matches = metadata_df.loc[mask, column]
    if len(matches) > 0:
        val = str(matches.iloc[0]).strip()
        if val.lower() not in ['', 'nan', 'none', 'n/a']:
            return val
    return ''

def get_var_label(var_name):
    \"\"\"Get the variable label from metadata dictionary.\"\"\"
    if metadata_df is None:
        return ''
    # Try both possible column names
    if 'VARIABLELABEL' in metadata_df.columns:
        col = 'VARIABLELABEL'
    elif 'VARIABLE LABEL' in metadata_df.columns:
        col = 'VARIABLE LABEL'
    else:
        with open(log_file, "a") as log:
            log.write("WARNING: No variable label column found in metadata\\n")
        return ''
    result = get_var_metadata(var_name, col)
    with open(log_file, "a") as log:
        log.write("get_var_label('{{}}') using col='{{}}' -> '{{}}'\\n".format(var_name, col, result[:50] if result else ''))
    return result

def get_var_domain(var_name):
    \"\"\"Get the OMOP domain from metadata dictionary.\"\"\"
    return get_var_metadata(var_name, 'DOMAIN')

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
        # No valid columns found - log error gracefully
        with open(log_file, "a") as log:
            log.write("ERROR: None of the specified columns exist in the data.\\n")
            log.write("Available columns in data: {{}}\\n".format(list(df.columns)))
            log.write("Requested columns: {{}}\\n".format(SELECTED_VARIABLES))
            log.write("\\nNo charts will be generated. Please check the column mismatch files for details.\\n")

with open(log_file, "a") as log:
    log.write("Selected {{}} columns for visualization: {{}}\\n\\n".format(len(selected_columns), selected_columns))

# Cohort name for filenames
COHORT_NAME = "{cohort_id}"

# Create PNG charts - one per variable
saved_charts = []
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
    
    # Get the variable label and domain from metadata dictionary
    var_label = get_var_label(col)
    var_domain = get_var_domain(col)
    # Truncate label if too long
    if len(var_label) > 120:
        var_label = var_label[:117] + '...'

    def _build_title(data_type):
        # Title: Distribution of <var_name> (<data_type>, <domain>)
        #        <variable label from metadata>
        #        Data source: <DATA_SOURCE_NAME>
        type_info = data_type
        if var_domain:
            type_info = '{{}} - {{}}'.format(data_type, var_domain)
        lines = ['Distribution of {{}} ({{}})'.format(col, type_info)]
        if var_label:
            lines.append(var_label)
        lines.append('Data source: {{}}'.format(DATA_SOURCE_NAME))
        return '\\n'.join(lines)

    if pd.api.types.is_numeric_dtype(col_data) and not is_binary and not is_categorical_in_metadata:
        # Histogram for numeric data
        ax.hist(col_data, bins=HISTOGRAM_BINS, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(_build_title('Numeric'))

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
        data_type = 'Binary' if is_binary else 'Categorical'
        ax.set_title(_build_title(data_type))
        
        with open(log_file, "a") as log:
            log.write("  Unique values: {{}}\\n".format(df[col].nunique()))
            log.write("  Top 5 values: {{}}\\n".format(dict(value_counts.head(5))))
    
    # Save individual image
    plt.tight_layout()
    output_filename = "{{}}_{{}}.png".format(col, COHORT_NAME)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_charts.append(output_filename)
    
    with open(log_file, "a") as log:
        log.write("  Saved to: {{}}\\n\\n".format(output_filename))

with open(log_file, "a") as log:
    log.write("Visualization complete. {{}} PNG charts saved: {{}}\\n".format(len(saved_charts), saved_charts))
{generate_mapping_files_section(mapping_files, include_mapping_upload_slot)}
"""


def merge_datasets_script(
    studies_info: list[dict],
    mappings_info: list[dict] = None,
    is_shuffled_data: bool = False,
) -> str:
    """Generate the merge/pool script that combines all cohorts using the cohortpool package.

    This produces a single "merge-datasets" node script that:
    - Builds the `studies` dict expected by `cohortpool.pool`, pointing each study to its
      cohort data node and metadata dictionary node inside the DCR (`/input/<node_name>`).
    - Builds the `mappings` list from the cross-study mapping nodes available in the DCR.
    - Calls `cohortpool.pool(...)` and writes the pooled dataframe to `/output`.

    Args:
        studies_info: List of dicts, one per cohort, each with:
            - 'study_name': the cohort identifier used as the study key
            - 'data_node':  the DCR data node name (mounted at /input/<data_node>)
            - 'dict_node':  the DCR metadata dictionary node name (mounted at /input/<dict_node>)
            - 'patient_id': the patient/subject id column name in the cohort data
        mappings_info: Optional list of dicts, one per cross-study mapping, each with:
            - 'node_name': the DCR mapping node name (mounted at /input/<node_name>)
            - 'study_a':   the first study name referenced by the mapping
            - 'study_b':   the second study name referenced by the mapping
        is_shuffled_data: Whether the pooled inputs are shuffled/synthetic samples
            (True) or the full cohort data (False). Passed through to
            `cohortpool.pool(is_shuffled_data=...)`.

    Returns:
        The Python script as a string.
    """
    mappings_info = mappings_info or []

    # Build the `studies` dict literal.
    # Each study points to its data file, dictionary file (mounted under /input) and
    # the patient id column used to align records across studies.
    studies_lines = []
    for study in studies_info:
        study_name = study["study_name"]
        data_node = study["data_node"]
        dict_node = study["dict_node"]
        patient_id = study.get("patient_id", "") or ""
        studies_lines.append(
            '        "{name}": {{"data": "/input/{data}", "dictionary": "/input/{dic}", "patient_id": "{pid}"}},'.format(
                name=study_name, data=data_node, dic=dict_node, pid=patient_id
            )
        )
    studies_block = "\n".join(studies_lines) if studies_lines else ""

    # Build the `mappings` list literal (only mappings with two identifiable studies).
    mappings_lines = []
    for mapping in mappings_info:
        node_name = mapping.get("node_name")
        study_a = mapping.get("study_a")
        study_b = mapping.get("study_b")
        if not node_name or not study_a or not study_b:
            continue
        mappings_lines.append(
            '        {{"path": "/input/{path}", "study_a": "{a}", "study_b": "{b}"}},'.format(
                path=node_name, a=study_a, b=study_b
            )
        )
    mappings_block = "\n".join(mappings_lines) if mappings_lines else ""

    return f"""import os
from cohortpool import pool

# Output directory (always exists in the Decentriq environment)
output_dir = "/output"
log_file = os.path.join(output_dir, "merge_datasets_log.txt")

# Studies to pool. Each study points to its cohort data node and metadata dictionary
# node, mounted read-only under /input inside the enclave, plus the patient id column
# used to align records across studies.
studies = {{
{studies_block}
}}

# Cross-study mapping files available in this DCR (variable mappings between studies).
# Each mapping file is uploaded as a DCR data node and mounted read-only under /input.
mappings = [
{mappings_block}
]

# Preprocess: strip leading/trailing spaces from column names in each study's data CSV.
# Data owners upload raw CSVs whose headers may contain stray spaces (e.g. "Record ID "
# vs "Record ID"), which would cause cohortpool's exact column lookup to fail.
for study_id, cfg in studies.items():
    data_path = cfg["data"]
    try:
        cleaned_path = os.path.join("/tmp", os.path.basename(data_path))
        with open(data_path, "r") as src, open(cleaned_path, "w") as dst:
            header = src.readline()
            stripped = ",".join(col.strip() for col in header.rstrip("\\r\\n").split(","))
            dst.write(stripped + "\\n")
            for line in src:
                dst.write(line)
        cfg["data"] = cleaned_path
    except Exception as e:
        with open(log_file, "a") as log:
            log.write("Failed to clean columns for {{}}: {{}}\\n".format(study_id, e))

with open(log_file, "w") as log:
    log.write("Pooling {{}} studies with {{}} mapping file(s)\\n".format(len(studies), len(mappings)))
    log.write("Studies: {{}}\\n".format(list(studies.keys())))

# Pool the datasets together using the cohortpool package.
# Unit conversions, drug classes, drug target doses and core variables all come
# from the tables bundled with the package. Pass a path for any of them to use an
# edited copy instead.
result = pool(
    studies=studies,
    mappings=mappings,
    output_dir=output_dir,
    # Quality thresholds. Inclusion is decided on how many pooled patients actually
    # have a value; min_study_coverage_pct is retained only as a reported metric.
    min_study_coverage_pct=80,
    min_data_completeness_pct=50,
    # Mapping options.
    include_partial=True,
    # Partial-match transformations. Supplying this list REPLACES the built-in default
    # registry, so the two defaults must be named explicitly or they are lost.
    partial_rules=[
        "CATEGORY_DECOMPOSE",
        "DATE_TO_PRESENCE",
        "CATEGORICAL_TO_INDICATOR",
        "POSITIVE_ONLY_BINARY",
        "DATE_TO_GRANULARITY",
    ],
    minimum_studies=2,
    # Whether the pooled inputs are shuffled/synthetic samples; affects report
    # warnings only. Set from the DCR wizard's "merge on shuffled samples" switch.
    is_shuffled_data={is_shuffled_data},
)

pooled_df = result["pooled_dataframe"]

# Persist the pooled dataset so it can be used by downstream nodes / retrieved as a result.
output_file = os.path.join(output_dir, "pooled_dataset.csv")
pooled_df.to_csv(output_file, index=False)

with open(log_file, "a") as log:
    log.write("Pooled dataset saved: {{}}\\n".format(output_file))
    log.write("Pooled dataset shape: {{}} rows, {{}} columns\\n".format(len(pooled_df), len(pooled_df.columns)))
    log.write("Columns: {{}}\\n".format(list(pooled_df.columns)))
"""


def merged_data_fragment_script(
    merge_node_name: str,
    patient_id_columns: list[str],
    airlock_percentage: int,
) -> str:
    """Generate the airlock fragmentation script for the merged/pooled dataset.

    Mirrors data_fragment_script (the per-cohort airlock script), adapted to the
    pooled output of the merge node. The pooled dataset is MORE identifying than
    any single cohort: cohortpool keeps each study's original patient ID column
    and builds `pooled_patient_id` as "<study>::<original_id>". This script:
    - Loads the pooled dataset from the merge node's output
    - Drops every study's original patient ID column
    - Replaces `pooled_patient_id` with synthetic IDs (same ID -> same synthetic ID)
    - Splits the data based on the airlock percentage
    - Caps outliers using z-scores (2 std deviations), with statistics computed
      on the full pooled dataset (numeric columns detected by dtype, binary
      0/1 columns excluded)
    - Saves the fragment to output

    Args:
        merge_node_name: Name of the merge compute node (its output is mounted
            at /input/<merge_node_name>/ inside this node).
        patient_id_columns: The original patient ID column names of the pooled
            studies, to be dropped from the fragment.
        airlock_percentage: Percentage of rows to include in the fragment.

    Returns:
        The Python script as a string.
    """
    return f"""import pandas as pd
import numpy as np
import os

# Output directory (always exists in Decentriq environment)
output_dir = "/output"
log_file = os.path.join(output_dir, "merged_fragmentation_log.txt")

# Read the pooled dataset produced by the merge node (mounted under /input)
df = pd.read_csv("/input/{merge_node_name}/pooled_dataset.csv")

with open(log_file, "a") as log:
    log.write("Loaded pooled dataset with {{}} rows and {{}} columns\\n".format(len(df), len(df.columns)))

# Drop the original per-study patient ID columns. cohortpool carries each study's
# original ID column through to the pooled output, so they must not reach the airlock.
id_columns_to_drop = {patient_id_columns!r}
drop_lower = {{c.lower().strip() for c in id_columns_to_drop}}
found_id_cols = [col for col in df.columns if col.lower().strip() in drop_lower]
if found_id_cols:
    df = df.drop(columns=found_id_cols)
with open(log_file, "a") as log:
    log.write("Dropped original patient ID columns: {{}}\\n".format(found_id_cols))

# Replace pooled_patient_id ("<study>::<original_id>", i.e. it embeds the original ID)
# with synthetic IDs. Rows with the same original pooled ID get the same synthetic ID.
pooled_id_col = None
for col in df.columns:
    if col.lower().strip() == "pooled_patient_id":
        pooled_id_col = col
        break

with open(log_file, "a") as log:
    if pooled_id_col:
        id_col_position = df.columns.get_loc(pooled_id_col)
        unique_ids = df[pooled_id_col].unique()
        id_mapping = {{orig_id: 'AIRLOCK_' + str(i).zfill(6) for i, orig_id in enumerate(unique_ids, start=1)}}
        synthetic_ids = df[pooled_id_col].map(id_mapping)
        df = df.drop(columns=[pooled_id_col])
        df.insert(id_col_position, 'Synthetic_ID', synthetic_ids)
        log.write("Replaced '{{}}' with synthetic IDs at position {{}}\\n".format(pooled_id_col, id_col_position))
        log.write("Mapped {{}} unique pooled IDs to synthetic IDs\\n".format(len(unique_ids)))
    else:
        synthetic_ids = ['AIRLOCK_' + str(i).zfill(6) for i in range(1, len(df) + 1)]
        if 'Synthetic_ID' in df.columns:
            df = df.drop(columns=['Synthetic_ID'])
        df.insert(0, 'Synthetic_ID', synthetic_ids)
        log.write("No pooled_patient_id column found, added row-based synthetic IDs at position 0\\n")

# Airlock percentage setting (fixed for the merged dataset, independent of the
# per-cohort airlock settings)
airlock_percentage = {airlock_percentage}

# Shuffle the dataframe to ensure random split
df_full = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split based on airlock percentage
split_fraction = airlock_percentage / 100.0
split_index = int(len(df_full) * split_fraction)
df_fragment = df_full.iloc[:split_index].copy()

# Identify numeric variables for outlier capping. The pooled dataset has no single
# metadata dictionary, so detect by dtype and exclude binary 0/1 indicator columns
# (cohortpool emits many of those) and the ID/study columns.
excluded_cols = {{'synthetic_id', 'study_id'}}
numeric_vars = []
for col in df_full.columns:
    if col.lower().strip() in excluded_cols:
        continue
    col_data = pd.to_numeric(df_full[col], errors='coerce')
    if col_data.notna().sum() == 0:
        continue
    if not pd.api.types.is_numeric_dtype(df_full[col]):
        continue
    unique_vals = set(col_data.dropna().unique())
    if len(unique_vals) <= 2 and unique_vals.issubset({{0, 1, 0.0, 1.0}}):
        continue
    numeric_vars.append(col)

with open(log_file, "a") as log:
    log.write("\\nIdentified {{}} numeric variables for outlier capping\\n".format(len(numeric_vars)))

# Outlier capping using z-scores (2 standard deviations)
# Calculate statistics on FULL pooled dataset, cap on fragment only
Z_THRESHOLD = 2.0
outlier_stats = []

for var in numeric_vars:
    try:
        full_values = pd.to_numeric(df_full[var], errors='coerce')
        fragment_values = pd.to_numeric(df_fragment[var], errors='coerce')

        mean_val = full_values.mean()
        median_val = full_values.median()
        std_val = full_values.std()

        if pd.isna(std_val) or std_val == 0:
            continue

        lower_limit = mean_val - (Z_THRESHOLD * std_val)
        upper_limit = mean_val + (Z_THRESHOLD * std_val)

        outliers_below = (fragment_values < lower_limit).sum()
        outliers_above = (fragment_values > upper_limit).sum()
        total_capped = outliers_below + outliers_above

        df_fragment[var] = fragment_values.clip(lower=lower_limit, upper=upper_limit)

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
output_file = os.path.join(output_dir, "dataset.csv")
df_fragment.to_csv(output_file, index=False)

with open(log_file, "a") as log:
    log.write("\\nMerged data fragment saved: {{}}\\n".format(output_file))
    log.write("Fragment size: {{}} rows out of {{}} total rows ({{:.1f}}%)\\n".format(len(df_fragment), len(df_full), len(df_fragment)/len(df_full)*100 if len(df_full) else 0))
"""


def merged_data_overview_script(merge_node_name: str, preview_node_name: str) -> str:
    """Generate the example script that summarizes the FULL merged dataset.

    This node is the runnable end of the merged-data chain (airlock nodes cannot
    be accessed in production mode, so this node depends on the merge node
    directly and, by also depending on the fragment node, triggers the airlock
    content for Development-mode use). Every participant can run it.

    It reads the full pooled dataset but writes ONLY aggregate information to
    its output — dataset shape, patients per study, per-column completeness —
    plus ALL merge-process metadata produced by cohortpool (mapping tables,
    disambiguation/review decisions, provenance, audits, logs, figures, the PDF
    quality report), excluding only the patient-level data files. Intentionally
    broad while testing; prune the export once sensitive files are identified.

    Args:
        merge_node_name: Name of the merge compute node (its output, including
            cohortpool's tables/, is mounted at /input/<merge_node_name>/).
        preview_node_name: Name of the airlock node, referenced in the header
            comments so users know how to explore the fragment in Development mode.

    Returns:
        The Python script as a string.
    """
    return f"""###############################################################################
# EXAMPLE SCRIPT — OVERVIEW OF THE MERGED DATA
#
# This script reads the FULL merged (pooled) dataset produced by the
# "{merge_node_name}" node and writes AGGREGATE information only:
#   - dataset shape (rows, columns) and patients per study
#   - per-column completeness (non-empty / empty counts)
#   - ALL metadata the pooling package produced about the merge process
#     (mapping tables, disambiguation/review decisions, provenance, audits,
#     logs, figures, the PDF quality report) — everything except the merged
#     dataset itself. It never writes patient-level rows to its output.
#
# You can copy this script into the "Development" tab and adapt it further.
# IMPORTANT: in the Development tab you must select this script's inputs in the
# right-side panel. To explore the (de-identified, outlier-capped) data fragment
# itself in Development mode, select the airlock node
# "{preview_node_name}" as an input — it exposes dataset.csv.
###############################################################################

import os
import shutil
import pandas as pd

# Output directory (always exists in Decentriq environment)
output_dir = "/output"
report_file = os.path.join(output_dir, "merged_data_overview.txt")

merge_dir = "/input/{merge_node_name}"
pooled_path = os.path.join(merge_dir, "pooled_dataset.csv")

with open(report_file, "w") as report:
    report.write("MERGED DATASET OVERVIEW\\n")
    report.write("=" * 60 + "\\n\\n")

    if not os.path.exists(pooled_path):
        report.write("pooled_dataset.csv was NOT found in the merge node output.\\n")
        try:
            report.write("Files available: {{}}\\n".format(sorted(os.listdir(merge_dir))))
        except Exception as e:
            report.write("Could not list the merge output directory: {{}}\\n".format(e))
        df = None
    else:
        size_bytes = os.path.getsize(pooled_path)
        df = pd.read_csv(pooled_path)
        report.write("File: pooled_dataset.csv ({{:.2f}} KB)\\n".format(size_bytes / 1024))
        report.write("Rows (patients): {{:,}}\\n".format(len(df)))
        report.write("Columns: {{:,}}\\n\\n".format(len(df.columns)))

        # Patients per study (aggregate counts only)
        study_col = None
        for col in df.columns:
            if col.lower().strip() == "study_id":
                study_col = col
                break
        if study_col:
            report.write("Patients per study:\\n")
            for study, count in df[study_col].value_counts(dropna=False).items():
                report.write("  - {{}}: {{:,}}\\n".format(study, count))
            report.write("\\n")

        # Per-column completeness: non-empty vs empty values for every column.
        completeness = pd.DataFrame({{
            "column": df.columns,
            "non_empty_values": [int(df[c].notna().sum()) for c in df.columns],
            "empty_values": [int(df[c].isna().sum()) for c in df.columns],
        }})
        completeness["pct_missing"] = (100.0 * completeness["empty_values"] / len(df)).round(2) if len(df) else 0.0
        completeness.to_csv(os.path.join(output_dir, "column_completeness.csv"), index=False)
        report.write("Per-column completeness written to column_completeness.csv\\n")
        report.write("({{}} columns; showing the 30 with the most missing values)\\n\\n".format(len(completeness)))
        worst = completeness.sort_values("pct_missing", ascending=False).head(30)
        for _, row in worst.iterrows():
            report.write("  - {{}}: {{:,}} non-empty / {{:,}} empty ({{:.1f}}% missing)\\n".format(
                row["column"], row["non_empty_values"], row["empty_values"], row["pct_missing"]))
        report.write("\\n")

# Export ALL merge-process metadata produced by the pooling package — the
# combined cross-study mapping, the applied mapping plan, disambiguation and
# review decisions, provenance, audits, logs, figures and the PDF quality
# report — EXCEPT the merged dataset itself. The patient-level files all sit
# in the output root and start with "pooled_" (pooled_dataset.csv,
# pooled_harmonized_patient_level.csv/.parquet, pooled_harmonized_filtered.csv,
# pooled_longitudinal*.csv, pooled_provenance_patient_level.csv), so those
# are skipped; everything else is copied verbatim, preserving the tables/ and
# logs/ directory structure. During the testing phase this is intentionally
# broad — files found to be sensitive will be excluded later.
reports_dir = os.path.join(output_dir, "harmonization_reports")
os.makedirs(reports_dir, exist_ok=True)
copied = []
skipped = []
for root, dirs, files in os.walk(merge_dir):
    rel_root = os.path.relpath(root, merge_dir)
    for fname in sorted(files):
        in_output_root = rel_root == "."
        if (in_output_root and fname.lower().startswith("pooled_")) or fname.lower().endswith(".parquet"):
            skipped.append(fname)
            continue
        dst_dir = reports_dir if in_output_root else os.path.join(reports_dir, rel_root)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.copy(os.path.join(root, fname), os.path.join(dst_dir, fname))
            copied.append(fname if in_output_root else os.path.join(rel_root, fname))
        except Exception as e:
            skipped.append("{{}} (copy failed: {{}})".format(fname, e))

with open(report_file, "a") as report:
    report.write("Merge-process metadata copied to harmonization_reports/ ({{}} files):\\n".format(len(copied)))
    for name in copied:
        report.write("  [ok]      {{}}\\n".format(name))
    report.write("\\nSkipped (patient-level data, never exported):\\n")
    for name in skipped:
        report.write("  [skipped] {{}}\\n".format(name))

    report.write("\\n" + "=" * 60 + "\\n")
    report.write("OVERVIEW COMPLETE\\n")
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
