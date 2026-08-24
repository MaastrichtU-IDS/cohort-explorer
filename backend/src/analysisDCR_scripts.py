"""Analysis DCR embedded scripts: data fragmentation, visualization, exploration, and the merged-data chain."""


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

# Save the fragment to output. The file is always named "dataset.csv" (the
# cohort is identified by the airlock node/folder name, which stays per-cohort),
# matching the merged-data airlock so consumers use one predictable filename.
output_file = os.path.join(output_dir, "dataset.csv")
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
    - Draws one PNG panel per variable: histogram (log axis when skewed),
      cumulative distribution and box plots with/without outliers for numeric
      variables; counts and shares for categorical ones
    - Writes figures_guide.txt describing what each view is suited for
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
            f'{p_air}DATA_FILE = "/input/preview-airlock-{cohort_id}/dataset.csv"\n'
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

# Create PNG charts - one panel of several views per variable. No single chart
# suits every variable (a box plot squashes a skewed biomarker, a histogram
# hides the median), so numeric variables get four views side by side and
# categorical ones two; figures_guide.txt explains what each view is suited for.
saved_charts = []
guide_entries = []

def _is_skewed(vals):
    # Strongly right-skewed positive data (biomarkers, durations): also worth a log axis
    if len(vals) < 10 or (vals <= 0).any():
        return False
    med = float(vals.median())
    return med > 0 and float(vals.max()) / med > 10 and float(vals.skew()) > 2

for col in selected_columns:
    col_data = df[col].dropna()
    
    with open(log_file, "a") as log:
        log.write("Column: {{}}\\n".format(col))
        log.write("  Non-null values: {{}}\\n".format(len(col_data)))
        log.write("  Data type: {{}}\\n".format(df[col].dtype))
    
    # Convert cm to inches for matplotlib (1 inch = 2.54 cm)
    chart_width_inches = CHART_WIDTH_CM / 2.54
    
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
        vals = pd.to_numeric(col_data, errors='coerce').dropna()
        if len(vals) == 0:
            with open(log_file, "a") as log:
                log.write("  No numeric values, skipped\\n\\n")
            continue
        skew = _is_skewed(vals)
        fig, axes = plt.subplots(2, 2, figsize=(chart_width_inches, chart_width_inches * 0.8))
        lo, hi = float(vals.min()), float(vals.max())

        # (1) histogram with mean and median
        ax = axes[0][0]
        edges = np.logspace(np.log10(lo), np.log10(hi), HISTOGRAM_BINS + 1) if skew and lo > 0 and hi > lo else HISTOGRAM_BINS
        ax.hist(vals, bins=edges, edgecolor='black', alpha=0.7)
        mean_val, median_val = vals.mean(), vals.median()
        ax.axvline(mean_val, color='red', linestyle='--', label='Mean: {{:.2f}}'.format(mean_val))
        ax.axvline(median_val, color='green', linestyle='--', label='Median: {{:.2f}}'.format(median_val))
        if skew:
            ax.set_xscale('log')
        ax.set_xlabel(col + (', log scale' if skew else ''))
        ax.set_ylabel('Patients')
        ax.legend(fontsize=8)
        ax.set_title('Histogram', fontsize=10)

        # (2) cumulative distribution (evaluated on a 200-point grid)
        ax = axes[0][1]
        grid = np.logspace(np.log10(lo), np.log10(hi), 200) if skew and lo > 0 else np.linspace(lo, hi, 200)
        xs = np.sort(vals.values)
        ax.plot(grid, np.searchsorted(xs, grid, side='right') / len(xs))
        if skew:
            ax.set_xscale('log')
        ax.set_xlabel(col + (', log scale' if skew else ''))
        ax.set_ylabel('Fraction at or below')
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title('Cumulative distribution', fontsize=10)

        # (3) box plot with outliers / (4) without
        for ax, fliers, sub in ((axes[1][0], True, 'Box plot, outliers shown'), (axes[1][1], False, 'Box plot, outliers hidden')):
            ax.boxplot([vals.values], showfliers=fliers)
            ax.set_xticks([1])
            ax.set_xticklabels(['n={{}}'.format(len(vals))])
            ax.set_ylabel(col + (', log scale' if skew else ''))
            if skew:
                ax.set_yscale('log')
            ax.set_title(sub, fontsize=10)

        fig.suptitle(_build_title('Numeric'), fontsize=10)
        guide_entries.append((col, 'numeric', len(vals), skew, None))
        with open(log_file, "a") as log:
            log.write("  Mean: {{:.4f}}, Median: {{:.4f}}, Std: {{:.4f}}{{}}\\n".format(
                mean_val, median_val, vals.std(), " (log scale: right-skewed)" if skew else ""))
    else:
        # Categorical / binary: counts and shares side by side
        value_counts = col_data.astype(str).value_counts()
        shown = value_counts.head(MAX_CATEGORIES)
        fig, axes = plt.subplots(1, 2, figsize=(chart_width_inches, chart_width_inches * 0.45))
        ypos = range(len(shown))
        # "2.0" -> "2": a numeric column with blanks is read as float
        labels = [(str(v)[:-2] if str(v).endswith('.0') else str(v))[:30] for v in shown.index]
        ax = axes[0]
        ax.barh(ypos, shown.values)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Patients')
        ax.set_title('Counts', fontsize=10)
        ax = axes[1]
        pct = 100.0 * shown.values / max(len(col_data), 1)
        ax.barh(ypos, pct)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('% of patients with a value')
        ax.set_title('Shares', fontsize=10)
        data_type = 'Binary' if is_binary else 'Categorical'
        fig.suptitle(_build_title(data_type), fontsize=10)
        hidden = len(value_counts) - len(shown)
        guide_entries.append((col, data_type.lower(), len(col_data), False, hidden))
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

# Guide: what was produced and what each view is suited for
with open(os.path.join(output_dir, "figures_guide.txt"), "w") as gf:
    gf.write("FIGURES PRODUCED AND WHAT EACH VIEW IS SUITED FOR\\n")
    gf.write("Data source: {{}}\\n\\n".format(DATA_SOURCE_NAME))
    gf.write("Each numeric variable gets one image with four views:\\n")
    gf.write("  Histogram -- where the values pile up; the mean and median are marked.\\n")
    gf.write("    Less suited when the data are strongly skewed and the axis is linear; skewed\\n")
    gf.write("    variables are therefore drawn on a log axis (noted per variable below).\\n")
    gf.write("  Cumulative distribution -- the fraction of patients at or below each value;\\n")
    gf.write("    any percentile can be read off (the curve crosses 0.5 at the median).\\n")
    gf.write("  Box plot, outliers shown -- median, quartiles, whiskers (1.5 x IQR) and each\\n")
    gf.write("    value beyond them; shows how extreme the extremes are.\\n")
    gf.write("  Box plot, outliers hidden -- the same box with the axis limited to the\\n")
    gf.write("    whiskers, so the box stays readable when outliers are far out.\\n\\n")
    gf.write("Each categorical or binary variable gets one image with two views:\\n")
    gf.write("  Counts -- patients per category (top {{}} categories).\\n".format(MAX_CATEGORIES))
    gf.write("  Shares -- the same as percentages of the patients with a value.\\n\\n")
    gf.write("Variables:\\n")
    for col, kind, n, skew, hidden in guide_entries:
        note = ""
        if skew:
            note = "; log scale (strongly right-skewed)"
        if hidden:
            note = "; {{}} rare categories not drawn".format(hidden)
        gf.write("  {{}}_{{}}.png  --  {{}}, n={{}}{{}}\\n".format(col, COHORT_NAME, kind, n, note))

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
    - Calls `cohortpool.pool(...)` (longitudinal output format, monthly temporal unit,
      mirroring the reference run.py usage) and writes the published longitudinal
      patient-x-visit dataframe to `/output/pooled_dataset.csv` — the stable filename
      the downstream airlock/overview nodes read.

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

import numpy as np
import pandas as pd

# --- Compatibility shim (TEMPORARY) -------------------------------------------
# The enclave's numpy is the system-provided 1.x, and the system site-packages
# precede the custom environment on sys.path, so a newer numpy cannot be
# installed over it. cohortpool's medication normalizer produces pandas
# nullable columns (Float64/Int64); feeding those into np.isclose under
# numpy 1.x raises TypeError (isfinite on object arrays) — observed in the
# enclave at cohortpool pipeline.py:3670. Until cohortpool coerces these
# itself, wrap np.isclose to convert nullable pandas inputs to plain float64
# (pd.NA -> NaN) first: identical comparison semantics (equal_nan=False keeps
# treating missing as non-matching), and a no-op for regular inputs.
# Remove once the fix lands in cohortpool and the pinned commit is bumped.
_np_isclose = np.isclose

def _coerce_nullable(x):
    if isinstance(x, pd.Series) and pd.api.types.is_extension_array_dtype(x.dtype):
        return pd.to_numeric(x, errors="coerce").astype("float64")
    return x

def _isclose_compat(a, b, *args, **kwargs):
    return _np_isclose(_coerce_nullable(a), _coerce_nullable(b), *args, **kwargs)

np.isclose = _isclose_compat
# ------------------------------------------------------------------------------

from cohortpool import pool

# Output directory (always exists in the Decentriq environment)
output_dir = "/output"
log_file = os.path.join(output_dir, "merge_datasets_log.txt")


def wlog(message):
    # Live progress to Decentriq's worker log ("/worker/logs"), visible in the
    # platform while the computation is still running. Best-effort: outside an
    # enclave the path does not exist and the message is simply dropped.
    try:
        with open("/worker/logs", "a") as f:
            f.write(str(message) + "\\n")
    except Exception:
        pass


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

with open(log_file, "w") as log:
    log.write("Pooling {{}} studies with {{}} mapping file(s)\\n".format(len(studies), len(mappings)))
    log.write("Studies: {{}}\\n".format(list(studies.keys())))
wlog("merge-datasets: starting - {{}} studies ({{}}), {{}} mapping file(s)".format(len(studies), ", ".join(studies), len(mappings)))

# Preprocess: strip leading/trailing spaces from column names in each study's data
# CSV (written to a cleaned copy under /tmp). Data owners upload raw CSVs whose
# headers may contain stray spaces (e.g. "Record ID " vs "Record ID"), which would
# make cohortpool's exact column lookup fail. Nothing else is altered: a study
# whose patient_id is empty (no ID variable identified for the cohort) is passed
# through as-is — resolving the patient key is the merge algorithm's job.
import csv as _csv
for study_id, cfg in studies.items():
    data_path = cfg["data"]
    try:
        cleaned_path = os.path.join("/tmp", os.path.basename(data_path))
        with open(data_path, "r", newline="") as src, open(cleaned_path, "w", newline="") as dst:
            reader = _csv.reader(src)
            writer = _csv.writer(dst)
            writer.writerow([col.strip() for col in next(reader)])
            for row in reader:
                writer.writerow(row)
        cfg["data"] = cleaned_path
    except Exception as e:
        with open(log_file, "a") as log:
            log.write("Failed to clean columns for {{}}: {{}}\\n".format(study_id, e))
    if not cfg.get("patient_id"):
        with open(log_file, "a") as log:
            log.write("{{}}: no patient-ID column configured; left to the merge algorithm to resolve\\n".format(study_id))
wlog("merge-datasets: input headers cleaned; inspecting the mapping files")

# Sanity-check the mapping files' harmonization_status values BEFORE pooling.
# cohortpool silently drops every row whose status is not exactly
# 'Identical Match', 'Compatible Match' or 'Partial Match' (a missing column
# drops ALL rows), and an all-dropped mapping leads to an empty pool. Logging
# the counts up front makes that failure diagnosable from this node's log.
import csv as _csv
_ACCEPTED_STATUSES = {{"Identical Match", "Compatible Match", "Partial Match"}}
with open(log_file, "a") as log:
    for m in mappings:
        try:
            with open(m["path"], newline="") as fh:
                rows = list(_csv.DictReader(fh))
            counts = {{}}
            for r in rows:
                status = (r.get("harmonization_status") or "").strip()
                counts[status] = counts.get(status, 0) + 1
            log.write("Mapping {{}} -> {{}}: {{}} rows, harmonization_status counts: {{}}\\n".format(
                m["study_a"], m["study_b"], len(rows), counts or "(no rows)"))
            accepted = sum(n for s, n in counts.items() if s in _ACCEPTED_STATUSES)
            if accepted == 0:
                log.write("  WARNING: no row has an accepted status - cohortpool will "
                          "ignore this whole mapping file.\\n")
        except Exception as e:
            log.write("Could not inspect mapping file {{}}: {{}}\\n".format(m.get("path"), e))

wlog("merge-datasets: calling cohortpool.pool() - the long step")

# Pool the datasets together using the cohortpool package.
# Unit conversions, drug classes, drug target doses and core variables all come
# from the tables bundled with the package. Pass a path for any of them to use an
# edited copy instead.
result = pool(
    studies=studies,
    mappings=mappings,
    output_dir=output_dir,
    # "longitudinal" (patient x visit rows) — matching the reference run.py
    # usage exactly. NOTE: in this mode the patient-level view is computed but
    # NOT written, so result["pooled_dataframe"] comes back EMPTY; the
    # longitudinal frame is the published dataset (handled below).
    output_format="longitudinal",
    temporal_unit="months",
    # Quality thresholds: a variable needs values for >=50% of each cohort's
    # patients, and >=80% of pooled patients overall, to be included.
    min_completeness_per_cohort_pct=50,
    min_pooled_completeness_pct=80,
    # Mapping options.
    include_partial=True,
    # Partial-match transformations. Supplying this list REPLACES the built-in default
    # registry, so any default rule must be named explicitly or it is lost.
    partial_rules=[
        "CATEGORY_DECOMPOSE",
        "DATE_TO_PRESENCE",
        "CATEGORICAL_TO_INDICATOR",
        # A positive occurrence means yes; absence stays missing rather than
        # being read as no.
        "POSITIVE_ONLY_BINARY",
        "CATEGORY_RECODE",
        "CATEGORY_COLLAPSE",
        "DATE_TO_GRANULARITY",
    ],
    # Whether the pooled inputs are shuffled/synthetic samples; affects report
    # warnings only. Set from the DCR wizard's "merge on shuffled samples" switch.
    is_shuffled_data={is_shuffled_data},
)

wlog("merge-datasets: pool() returned; publishing the pooled dataset")

# Persist the published dataset as pooled_dataset.csv — the stable name the
# downstream nodes (airlock fragment, overview) read. With
# output_format="longitudinal" the published frame is the longitudinal one
# (patient x visit rows); result["pooled_dataframe"] is empty in that mode, so
# it only serves as a fallback should the output format ever change. Fail
# loudly if BOTH are empty: writing an empty CSV would only move the crash
# into the downstream nodes with a far less useful error.
published_df = result.get("longitudinal_dataframe")
published_kind = "longitudinal (patient x visit)"
if published_df is None or published_df.empty:
    published_df = result["pooled_dataframe"]
    published_kind = "patient-level"
if published_df.empty:
    with open(log_file, "a") as log:
        log.write("ERROR: pooling produced no data (longitudinal and patient-level "
                  "frames both empty). Most common cause: no mapping rows were "
                  "accepted (see the harmonization_status counts above; cohortpool "
                  "only accepts 'Identical Match', 'Compatible Match' and "
                  "'Partial Match').\\n")
    wlog("merge-datasets: ERROR - pooling produced no data (see merge_datasets_log.txt)")
    raise RuntimeError("Pooling produced no data; see merge_datasets_log.txt")
output_file = os.path.join(output_dir, "pooled_dataset.csv")
published_df.to_csv(output_file, index=False)

with open(log_file, "a") as log:
    log.write("Pooled dataset saved: {{}} ({{}})\\n".format(output_file, published_kind))
    log.write("Pooled dataset shape: {{}} rows, {{}} columns\\n".format(len(published_df), len(published_df.columns)))
    if "pooled_patient_id" in published_df.columns:
        log.write("Patients: {{}}\\n".format(published_df["pooled_patient_id"].nunique()))
    log.write("Columns: {{}}\\n".format(list(published_df.columns)))

# Record the output files the package reports having written.
if "output_paths" in result:
    with open(log_file, "a") as log:
        log.write("Package output paths:\\n")
        for key, path in result["output_paths"].items():
            log.write("  {{}}: {{}}\\n".format(key, path))

wlog("merge-datasets: done - pooled_dataset.csv written, {{}} rows x {{}} columns ({{}})".format(len(published_df), len(published_df.columns), published_kind))
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


def wlog(message):
    # Live progress to Decentriq's worker log ("/worker/logs"), visible in the
    # platform while the computation is still running. Best-effort: outside an
    # enclave the path does not exist and the message is simply dropped.
    try:
        with open("/worker/logs", "a") as f:
            f.write(str(message) + "\\n")
    except Exception:
        pass


# Read the pooled dataset produced by the merge node (mounted under /input)
wlog("merged-data-fragment: loading pooled_dataset.csv from the merge node")
df = pd.read_csv("/input/{merge_node_name}/pooled_dataset.csv")
wlog("merged-data-fragment: loaded {{}} rows x {{}} columns; de-identifying".format(len(df), len(df.columns)))

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
wlog("merged-data-fragment: IDs replaced; splitting the {{}}% fragment and capping outliers".format(airlock_percentage))

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

wlog("merged-data-fragment: done - dataset.csv written, {{}} of {{}} rows".format(len(df_fragment), len(df_full)))
"""


def merged_airlock_example_script(preview_node_name: str) -> str:
    """Generate the example-analysis script for the merged-data airlock.

    A documentation node: its script carries numbered instructions on how to
    work with the airlocked merged-data fragment in the Development tab (the
    key point being the input path), plus a commented minimal code snippet.
    The script is comments only — running the node executes nothing and
    produces no output; its purpose is to be read and copied into the
    Development tab. It has no dependencies, so it runs instantly and cannot
    fail in production mode.

    Args:
        preview_node_name: Name of the airlock/preview node the instructions
            refer to.

    Returns:
        The Python script as a string.
    """
    return f"""###############################################################################
# EXAMPLE ANALYSIS — WORKING WITH THE MERGED DATA IN THE AIRLOCK
#
# The airlock node "{preview_node_name}" exposes a testing
# fragment of the merged dataset (a subset of rows, synthetic IDs instead of
# the original ones, outliers capped) as the file "dataset.csv".
#
# HOW TO USE THE AIRLOCK:
#
#   1. Open the "Development" tab of this Data Clean Room and create a new
#      script (or copy this one into it).
#   2. In the right-side panel of the Development tab, select the airlock node
#      "{preview_node_name}" as an input.
#      Without this step the fragment will NOT be available to your script.
#   3. The airlock contents are then mounted read-only under:
#          /input/{preview_node_name}/
#      and the merged-data fragment itself is at:
#          /input/{preview_node_name}/dataset.csv
#   4. Load the fragment with pandas (see the snippet below), build up your
#      analysis, and write any results you want to keep to /output.
#   5. Once your analysis works, it can be added to the DCR as a compute node
#      of its own so other participants can run it.
#
# MINIMAL EXAMPLE — uncomment these lines in the Development tab:
#
# import pandas as pd
#
# df = pd.read_csv("/input/{preview_node_name}/dataset.csv")
# print("Fragment shape:", df.shape)
# summary = df.describe(include="all")
# summary.to_csv("/output/fragment_summary.csv")
#
###############################################################################
"""


def merged_data_overview_script(
    merge_node_name: str,
    preview_node_name: str,
    include_patient_level: bool = False,
) -> str:
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
        include_patient_level: When True, the patient-level pooled files are
            exported as well. Set this only when the merge pools the shuffled
            samples (synthetic data, no leak concern); with full cohort data it
            must stay False so patient-level files are never exported.

    Returns:
        The Python script as a string.
    """
    # include_patient_level is True exactly when the merge pools the SHUFFLED
    # samples; in that case the merge node has analysts and is visible in the
    # interface, so the "hidden node" explanation would be wrong.
    if include_patient_level:
        name_note = f"""# A note on the name: the merge itself does NOT happen in this script — the
# merge code lives in the "{merge_node_name}" node. Running THIS node makes
# the platform compute that node and the airlock fragment first (they are its
# dependencies), which populates the airlock for Development-mode use. Since
# this merge pools the shuffled samples (synthetic data), the merge node is
# also visible and directly runnable on its own."""
    else:
        name_note = f"""# A note on the name: the merge itself does NOT happen in this script. The
# actual merge code lives in the "{merge_node_name}" node, which the platform
# hides from the interface because no participant has direct access to it (it
# deliberately has no analysts — its full patient-level output must not be
# directly retrievable). A hidden node cannot be run by hand, so this node is
# the runnable handle for the chain: because it depends on "{merge_node_name}"
# and on the airlock fragment node, running THIS node makes the platform
# compute the merge and the fragment first, which populates the airlock for
# Development-mode use."""

    return f"""###############################################################################
# RUN THE MERGE AND CREATE THE AIRLOCK
#
{name_note}
#
# What this script itself does: it reads the FULL merged (pooled) dataset
# produced by the "{merge_node_name}" node and writes AGGREGATE information only:
#   - dataset shape (rows, columns) and patients per study
#   - per-column completeness (non-empty / empty counts)
#   - ALL metadata the pooling package produced about the merge process
#     (mapping tables, disambiguation/review decisions, provenance, audits,
#     logs, figures, the PDF quality report) — everything except the merged
#     dataset itself. It never writes patient-level rows to its output.
#
# To explore the (de-identified, outlier-capped) merged-data fragment yourself,
# see the node "example-analysis-for-merged-data-in-airlock": it explains how to
# work with the airlock "{preview_node_name}"
# in the Development tab, with a minimal code snippet to start from.
###############################################################################

import os
import shutil
import pandas as pd

# Output directory (always exists in Decentriq environment)
output_dir = "/output"
report_file = os.path.join(output_dir, "merged_data_overview.txt")


def wlog(message):
    # Live progress to Decentriq's worker log ("/worker/logs"), visible in the
    # platform while the computation is still running. Best-effort: outside an
    # enclave the path does not exist and the message is simply dropped.
    try:
        with open("/worker/logs", "a") as f:
            f.write(str(message) + "\\n")
    except Exception:
        pass


merge_dir = "/input/{merge_node_name}"
pooled_path = os.path.join(merge_dir, "pooled_dataset.csv")
wlog("merged-data-overview: reading the pooled dataset from the merge node")

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
        wlog("merged-data-overview: pooled dataset loaded - {{}} rows x {{}} columns; computing completeness".format(len(df), len(df.columns)))
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
#
# include_patient_level is set at DCR-creation time: True when the merge pools
# the SHUFFLED samples (synthetic data, so no leak concern — the patient-level
# files are exported too), False when it pools the full cohort data (the
# patient-level files are then never exported).
include_patient_level = {include_patient_level}
wlog("merged-data-overview: overview written; copying the merge-process metadata (harmonization reports)")
reports_dir = os.path.join(output_dir, "harmonization_reports")
os.makedirs(reports_dir, exist_ok=True)
copied = []
skipped = []
for root, dirs, files in os.walk(merge_dir):
    rel_root = os.path.relpath(root, merge_dir)
    for fname in sorted(files):
        in_output_root = rel_root == "."
        is_patient_level = (in_output_root and fname.lower().startswith("pooled_")) or fname.lower().endswith(".parquet")
        if is_patient_level and not include_patient_level:
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
    if include_patient_level:
        report.write("\\nPatient-level files INCLUDED (merge pools the shuffled samples).\\n")
    if skipped:
        report.write("\\nSkipped (patient-level data, not exported):\\n")
        for name in skipped:
            report.write("  [skipped] {{}}\\n".format(name))

    report.write("\\n" + "=" * 60 + "\\n")
    report.write("OVERVIEW COMPLETE\\n")

wlog("merged-data-overview: done - {{}} metadata files copied, {{}} skipped".format(len(copied), len(skipped)))
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
