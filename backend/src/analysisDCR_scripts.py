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
        # Synthetic numbers are handed out in a fresh random order (unseeded), never in
        # file or ID order: numbered in order, AIRLOCK_000001 would be the first row /
        # lowest original ID, and anyone knowing the cohort's ID list could map the
        # synthetic IDs back to real patients.
        unique_ids = df[id_column].unique()
        synthetic_numbers = np.random.default_rng().permutation(len(unique_ids)) + 1
        id_mapping = {{orig_id: 'AIRLOCK_' + str(n).zfill(6) for orig_id, n in zip(unique_ids, synthetic_numbers)}}
        
        # Map original IDs to synthetic IDs
        synthetic_ids = df[id_column].map(id_mapping)
        
        # Remove the original ID column
        df = df.drop(columns=[id_column])
        
        # Insert synthetic IDs at the same position
        df.insert(id_col_position, 'Synthetic_ID', synthetic_ids)
        
        log.write("Replaced ID column '{{}}' with synthetic IDs at position {{}}\\n".format(id_column, id_col_position))
        log.write("Mapped {{}} unique original IDs to synthetic IDs\\n".format(len(unique_ids)))
    else:
        # No ID column found - add synthetic IDs at the beginning (row-based, no grouping),
        # numbered in random order so they do not reveal the row order of the file
        synthetic_ids = ['AIRLOCK_' + str(n).zfill(6) for n in np.random.default_rng().permutation(len(df)) + 1]
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

# Patient identifier columns are never charted: one value per patient carries no
# distribution, and drawing it would put row-level values into the output.
# Matched on the NAME, case- and separator-insensitive.
import re as _re
_ID_NAMES = {{
    "id", "pid", "patientid", "patientno", "patientnr", "patientnumber", "patientcode",
    "recordid", "recordno", "subjectid", "subjectno", "usubjid", "studysubjectid",
    "caseid", "participantid", "syntheticid", "pooledpatientid",
}}


def is_id_column(name):
    key = _re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())
    return key in _ID_NAMES or key.endswith("patientid") or key.endswith("subjectid")


_id_cols = [c for c in selected_columns if is_id_column(c)]
if _id_cols:
    selected_columns = [c for c in selected_columns if not is_id_column(c)]
    with open(log_file, "a") as log:
        log.write("Identifier column(s) not charted: {{}}\\n".format(_id_cols))

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
        for ax, fliers, sub in ((axes[1][0], True, 'Box plot, outliers shown'), (axes[1][1], False, 'Box plot, outliers not drawn (same n)')):
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
    gf.write("  Box plot, outliers not drawn -- the same box with the axis limited to the\\n")
    gf.write("    whiskers, so the box stays readable when outliers are far out. The n is\\n")
    gf.write("    unchanged and so are the median, quartiles and whiskers: the outlying\\n")
    gf.write("    values are still part of the statistics, they are simply not drawn.\\n\\n")
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


# ---------------------------------------------------------------------------
# What leaves the enclave from the merge node's output
#
# The node "run-the-merge-and-create-airlock" (every participant can run it)
# copies files from the merge node's output into harmonization_reports/. It
# copies ONLY the .pdf files (cohortpool's data_quality_report.pdf and any
# other PDF report it produces), wherever they sit. Nothing else leaves the
# enclave from the merge node: no CSV, no Parquet, no JSON, no log, no PNG.
#
# Earlier versions exported every .png figure and an allowlist of named
# report files. That list had to be re-audited at every cohortpool bump, and a
# single wrong entry once let pooled_longitudinal.csv (every patient's rows,
# with real patient IDs) out. A rule on the file type is simpler to reason
# about: the PDF is a rendered report, not a data table, so a new cohortpool
# version cannot publish a new data file through it just by choosing a name.
# Deliberately NOT exported (linked data about individual patients):
#   pooled_dataset.csv, pooled_longitudinal*.csv, pooled_harmonized_*.csv/.parquet,
#   pooled_provenance_patient_level.csv, medication_implausible_dose_audit.csv,
#   longitudinal_backfilled_from_patient_level.csv - and every other non-PDF file.
# Widen MERGE_EXPORT_EXTENSIONS only after checking that files of the new type
# can hold no per-patient rows and no patient identifiers.
# ---------------------------------------------------------------------------
MERGE_EXPORT_EXTENSIONS: tuple[str, ...] = (".pdf",)


def _export_rule_comment(indent: str = "# ") -> str:
    """The export rule as a comment line, for the headers of the embedded scripts."""
    exts = ", ".join(MERGE_EXPORT_EXTENSIONS)
    return f"{indent}Exported file types: {exts} (matched on the file extension, in any sub-folder)."


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

    return f"""###############################################################################
# MERGE (POOL) THE COHORTS
#
# This node's output is never shown to participants directly. The node
# "run-the-merge-and-create-airlock" copies from it ONLY the .pdf report(s);
# everything else - the pooled dataset and the patient-level files above all,
# but also every CSV, Parquet, JSON, log and PNG - stays in the enclave.
{_export_rule_comment()}
###############################################################################

import os

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

# matplotlib's boxplot renamed `labels` to `tick_labels` in 3.9. cohortpool calls
# it by the new name (visualize.py plot_temporal_coverage), but the enclave
# imports matplotlib from the SYSTEM environment (/nix/store/...), which ships
# 3.8 - the custom environment's newer copy is installed yet shadowed, the same
# path-precedence that stops numpy being upgraded here. Accept the new keyword
# on the old version; a no-op once matplotlib itself understands it.
try:
    import inspect

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.axes

    if "tick_labels" not in inspect.signature(matplotlib.axes.Axes.boxplot).parameters:
        _orig_boxplot = matplotlib.axes.Axes.boxplot

        def _boxplot_compat(self, x, *args, **kwargs):
            if "tick_labels" in kwargs and "labels" not in kwargs:
                kwargs["labels"] = kwargs.pop("tick_labels")
            return _orig_boxplot(self, x, *args, **kwargs)

        matplotlib.axes.Axes.boxplot = _boxplot_compat
except Exception as _e:
    print("boxplot compatibility shim not applied:", _e)
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

# Stream cohortpool's log records live to the append-only /worker/logs file
# (the only path writable for live logs in the enclave). cohortpool logs via
# Python's logging module, so a handler on the root logger catches its records
# as they are emitted and appends each one to /worker/logs.
import logging as _logging

class _WorkerLogsHandler(_logging.Handler):
    def emit(self, record):
        try:
            with open("/worker/logs", "a") as f:
                f.write(self.format(record) + "\\n")
        except Exception:
            pass

_worker_handler = _WorkerLogsHandler(level=_logging.INFO)
_worker_handler.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_logging.getLogger().addHandler(_worker_handler)
_logging.getLogger().setLevel(_logging.INFO)
_logging.getLogger("cohortpool").addHandler(_worker_handler)

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
    # cohortpool opens <log_dir>/cohortpool.log as a regular file. /worker is NOT
    # writable in the enclave (only the special append-only /worker/logs file is),
    # so write the log under /output; the _WorkerLogsHandler above streams the
    # same records live to /worker/logs.
    log_dir=output_dir,
    # Quality thresholds: a variable needs values for >=20% of each cohort's
    # patients, and >=80% of pooled patients overall, to be included
    # (matching the reference run.py).
    min_completeness_per_cohort_pct=20,
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

# Record the output files the package reports having written. Newer cohortpool
# versions call this key "output_files"; accept either.
_paths = result.get("output_files") or result.get("output_paths")
if _paths:
    with open(log_file, "a") as log:
        log.write("Package output files:\\n")
        for key, path in _paths.items():
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
        # Synthetic numbers are handed out in a fresh random order (unseeded), never in
        # file or ID order: numbered in order, AIRLOCK_000001 would be the first row /
        # lowest original ID, and anyone knowing the cohort's ID list could map the
        # synthetic IDs back to real patients.
        unique_ids = df[pooled_id_col].unique()
        synthetic_numbers = np.random.default_rng().permutation(len(unique_ids)) + 1
        id_mapping = {{orig_id: 'AIRLOCK_' + str(n).zfill(6) for orig_id, n in zip(unique_ids, synthetic_numbers)}}
        synthetic_ids = df[pooled_id_col].map(id_mapping)
        df = df.drop(columns=[pooled_id_col])
        df.insert(id_col_position, 'Synthetic_ID', synthetic_ids)
        log.write("Replaced '{{}}' with synthetic IDs at position {{}}\\n".format(pooled_id_col, id_col_position))
        log.write("Mapped {{}} unique pooled IDs to synthetic IDs\\n".format(len(unique_ids)))
    else:
        synthetic_ids = ['AIRLOCK_' + str(n).zfill(6) for n in np.random.default_rng().permutation(len(df)) + 1]
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


def merged_airlock_example_script(preview_node_name: str, merge_node_name: str) -> str:
    """Generate the example-analysis script for the merged data.

    A RUNNABLE example every participant can execute and copy: it reads the
    FULL merged dataset (so the node depends on the merge node), prints every
    column with its counts (also written to /output), and draws simple
    overview figures — a completeness bar chart plus one mini-chart per column
    (histogram for numeric, top categories otherwise). Aggregates only.

    Right at the INPUT_PATH sits the note telling Development-tab users to
    point the script at the airlock fragment instead
    (/input/<preview_node>/dataset.csv).

    Args:
        preview_node_name: Name of the airlock/preview node (for the note).
        merge_node_name: Name of the merge node whose pooled_dataset.csv is read.

    Returns:
        The Python script as a string.
    """
    return f"""###############################################################################
# EXAMPLE ANALYSIS - THE MERGED DATASET
#
# A worked example you can RUN AS IS and copy as the starting point of your
# own analysis. Running it computes over the FULL merged (pooled) dataset and
# writes AGGREGATES ONLY to /output:
#   - every column name with its counts (non-empty, empty, unique values),
#     as columns_summary.csv and columns_and_counts.txt
#   - columns_completeness_p*.png - non-empty counts per column, as bars
#   - columns_overview_p*.png - one chart per column over the pooled data
#     (histogram for numeric columns, the top categories for the rest)
#   - columns_by_study_p*.png - the same columns STRATIFIED BY study_id, so
#     each variable can be read both as pooled/harmonized and per original
#     study: box plots per study for numeric columns, the category mix per
#     study for the rest.
#
# Patient identifier columns (PatientID, patient_id, Synthetic_ID,
# pooled_patient_id, ... matched case-insensitively) are counted in the tables
# but never plotted. study_id is kept - it names the cohort.
#
# Because this node depends on the "{merge_node_name}" node, running it also
# triggers the merge itself when its output is not computed yet (that step can
# take a while - progress appears in the worker log).
#
# TO BUILD YOUR OWN ANALYSIS in the "Development" tab:
#   1. Copy this script into a new Development-tab script.
#   2. Select your input in the right-side panel (see the INPUT note below),
#      adjust, run, and write anything you want to keep to /output.
#   3. A finished analysis can be added to the DCR as a compute node of its
#      own so other participants can run it.
###############################################################################

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wlog(message):
    # Live progress to Decentriq's worker log ("/worker/logs"), visible in the
    # platform while the computation is still running. Best-effort: outside an
    # enclave the path does not exist and the message is simply dropped.
    try:
        with open("/worker/logs", "a") as f:
            f.write(str(message) + "\\n")
    except Exception:
        pass


# INPUT ------------------------------------------------------------------------
# The full merged (pooled) dataset produced by the "{merge_node_name}" node:
INPUT_PATH = "/input/{merge_node_name}/pooled_dataset.csv"
#
# NOTE - USING THE AIRLOCK IN THE DEVELOPMENT TAB: the full dataset above is
# only readable by published compute nodes, not by Development-tab scripts.
# When you work in the Development tab, select the airlock node
# "{preview_node_name}" as your input and point INPUT_PATH at the
# airlocked testing fragment (subset of rows, synthetic IDs, outliers capped):
#     INPUT_PATH = "/input/{preview_node_name}/dataset.csv"
# ------------------------------------------------------------------------------

output_dir = "/output"

# Patient identifier columns are never plotted: one value per patient carries
# no distribution, and drawing it would put row-level values in the output.
# Matched on the NAME, case- and separator-insensitive ("PatientID",
# "patient_id", "Synthetic_ID", "pooled_patient_id", ...). study_id is NOT an
# identifier here - it names the cohort and is used to stratify.
_ID_NAMES = {{
    "id", "pid", "patientid", "patientno", "patientnr", "patientnumber", "patientcode",
    "pooledpatientid", "syntheticid", "recordid", "recordno", "subjectid", "subjectno",
    "usubjid", "studysubjectid", "caseid", "participantid",
}}


def is_id_column(name):
    key = re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())
    return key in _ID_NAMES or key.endswith("patientid") or key.endswith("subjectid")


wlog("example-analysis: reading the merged dataset")
df = pd.read_csv(INPUT_PATH)
wlog("example-analysis: loaded {{}} rows x {{}} columns; counting".format(len(df), len(df.columns)))

# ---- Every column with its counts --------------------------------------------
rows = []
for col in df.columns:
    non_empty = int(df[col].notna().sum())
    rows.append({{
        "column": col,
        "non_empty": non_empty,
        "empty": int(len(df) - non_empty),
        "unique_values": int(df[col].nunique(dropna=True)),
    }})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(output_dir, "columns_summary.csv"), index=False)

with open(os.path.join(output_dir, "columns_and_counts.txt"), "w") as fh:
    fh.write("MERGED DATASET: {{}} rows, {{}} columns\\n\\n".format(len(df), len(df.columns)))
    fh.write("{{:<50}} {{:>10}} {{:>10}} {{:>10}}\\n".format("column", "non-empty", "empty", "unique"))
    for r in rows:
        line = "{{:<50}} {{:>10}} {{:>10}} {{:>10}}".format(r["column"][:50], r["non_empty"], r["empty"], r["unique_values"])
        fh.write(line + "\\n")
        print(line)

# ---- Completeness bars: non-empty count per column (chunked pages) -----------
PLOT_COLUMNS = [c for c in df.columns if not is_id_column(c)]
_skipped_ids = [c for c in df.columns if is_id_column(c)]
if _skipped_ids:
    wlog("example-analysis: identifier column(s) not plotted: {{}} (still counted in the tables)".format(", ".join(_skipped_ids)))
chart_summary = summary[summary["column"].isin(PLOT_COLUMNS)].reset_index(drop=True)

CHUNK = 45
for page, start in enumerate(range(0, len(chart_summary), CHUNK), start=1):
    part = chart_summary.iloc[start:start + CHUNK]
    fig, ax = plt.subplots(figsize=(14, max(4, 0.36 * len(part))))
    ax.barh(part["column"][::-1], part["non_empty"][::-1], color="#3b6ea5")
    ax.set_xlabel("Non-empty values (of {{}} rows)".format(len(df)))
    ax.set_title("Column completeness ({{}}/{{}})".format(page, (len(chart_summary) + CHUNK - 1) // CHUNK))
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "columns_completeness_p{{}}.png".format(page)), dpi=130)
    plt.close(fig)
wlog("example-analysis: completeness charts written; drawing per-column mini-charts")

# ---- One mini-chart per column (histogram / top categories) ------------------
PER_FIG = 12   # 3 x 4 charts per page
COLS = 3
pages = (len(PLOT_COLUMNS) + PER_FIG - 1) // PER_FIG
for page in range(pages):
    cols = PLOT_COLUMNS[page * PER_FIG:(page + 1) * PER_FIG]
    nrows = (len(cols) + COLS - 1) // COLS
    fig, axes = plt.subplots(nrows, COLS, figsize=(16, 3.6 * nrows), squeeze=False)
    axes = np.atleast_2d(axes)
    for i, col in enumerate(cols):
        ax = axes[i // COLS][i % COLS]
        series = df[col].dropna()
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        try:
            if len(numeric) >= 0.9 * len(series) and numeric.nunique() > 8:
                # Strongly right-skewed positive values (biomarkers, durations):
                # bin them on a log axis, otherwise everything lands in bin 1.
                if (numeric > 0).all() and numeric.median() > 0 and numeric.max() / numeric.median() > 10:
                    _edges = np.logspace(np.log10(float(numeric.min())), np.log10(float(numeric.max())), 21)
                    ax.hist(numeric, bins=_edges, color="#3b6ea5")
                    ax.set_xscale("log")
                    ax.set_xlabel("log scale", fontsize=7)
                else:
                    ax.hist(numeric, bins=20, color="#3b6ea5")
            elif len(series) > 0 and series.nunique() > 30 and series.nunique() > 0.5 * len(series):
                # Identifier-like or free-text column: its values are (near-)
                # unique per row, so drawing them would put row-level values
                # into the output. Counted, not drawn.
                ax.text(0.5, 0.5, "identifier / free text\\n({{}} unique values)\\nnot drawn".format(series.nunique()),
                        ha="center", va="center", fontsize=9, color="#888888")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                vc = series.astype(str).value_counts().head(10)
                labels = [(v[:-2] if v.endswith(".0") else v)[:20] for v in vc.index]
                ax.barh(range(len(vc))[::-1], vc.values, color="#7a9e58")
                ax.set_yticks(range(len(vc))[::-1])
                ax.set_yticklabels(labels, fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, "not drawable", ha="center", va="center", fontsize=9)
            wlog("example-analysis: could not draw {{}}: {{}}".format(col, e))
        ax.set_title("{{}} (n={{}})".format(col[:40], len(series)), fontsize=10)
        ax.tick_params(labelsize=8)
    for j in range(len(cols), nrows * COLS):
        axes[j // COLS][j % COLS].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "columns_overview_p{{}}.png".format(page + 1)), dpi=130)
    plt.close(fig)

# ---- The same columns, stratified by study_id --------------------------------
# The pages above show each harmonized variable over the pooled data; these show
# how each original study contributes to it - the distribution per study, side
# by side. The pooled ID columns carry no information to stratify, so
# Synthetic_ID and pooled_patient_id are left out, as are identifier-like or
# free-text columns (their values are near-unique per row).
STUDY_COL = None
for _c in df.columns:
    if _c.lower().strip() == "study_id":
        STUDY_COL = _c
        break

strat_pages = 0
if STUDY_COL is None:
    wlog("example-analysis: no study_id column found - the per-study pages are skipped")
else:
    studies = sorted(str(x) for x in df[STUDY_COL].dropna().unique())
    short = {{s: s[:14] for s in studies}}
    strat_cols = []
    for col in PLOT_COLUMNS:
        if col == STUDY_COL:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        _num = pd.to_numeric(series, errors="coerce").dropna()
        _is_numeric = len(_num) >= 0.9 * len(series) and _num.nunique() > 8
        # Continuous measurements are near-unique by nature, so the
        # identifier/free-text guard only applies to non-numeric columns.
        if not _is_numeric and series.nunique() > 30 and series.nunique() > 0.5 * len(series):
            continue   # identifier-like / free text: counted on the pages above, not drawn
        strat_cols.append(col)
    wlog("example-analysis: stratifying {{}} columns by {{}} ({{}} studies)".format(len(strat_cols), STUDY_COL, len(studies)))
    strat_pages = (len(strat_cols) + PER_FIG - 1) // PER_FIG
    for page in range(strat_pages):
        cols = strat_cols[page * PER_FIG:(page + 1) * PER_FIG]
        nrows = (len(cols) + COLS - 1) // COLS
        fig, axes = plt.subplots(nrows, COLS, figsize=(16, 4.2 * nrows), squeeze=False)
        axes = np.atleast_2d(axes)
        for i, col in enumerate(cols):
            ax = axes[i // COLS][i % COLS]
            series = df[col].dropna()
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            try:
                if len(numeric) >= 0.9 * len(series) and numeric.nunique() > 8:
                    # Numeric: one box per study (outliers as small dots)
                    data, ticks = [], []
                    for st in studies:
                        vals = pd.to_numeric(df.loc[df[STUDY_COL].astype(str) == st, col], errors="coerce").dropna()
                        if len(vals) > 0:
                            data.append(vals.values)
                            ticks.append("{{}}\\n(n={{}})".format(short[st], len(vals)))
                    if not data:
                        raise ValueError("no values per study")
                    ax.boxplot(data, showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.35))
                    ax.set_xticks(range(1, len(data) + 1))
                    ax.set_xticklabels(ticks, fontsize=7)
                    # Strongly right-skewed positive values (biomarkers): on a
                    # linear axis the boxes collapse against zero.
                    if (numeric > 0).all() and numeric.median() > 0 and numeric.max() / numeric.median() > 10:
                        ax.set_yscale("log")
                        ax.set_ylabel("log scale", fontsize=7)
                else:
                    # Categorical: the category mix of each study, as shares
                    sub = df[[STUDY_COL, col]].dropna()
                    ct = pd.crosstab(sub[STUDY_COL].astype(str), sub[col].astype(str))
                    ct = ct.reindex([st for st in studies if st in ct.index])
                    top = list(ct.sum().sort_values(ascending=False).head(6).index)
                    rest = [c for c in ct.columns if c not in top]
                    plot_ct = ct[top].copy()
                    if rest:
                        plot_ct["other ({{}})".format(len(rest))] = ct[rest].sum(axis=1)
                    pct = plot_ct.div(plot_ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
                    bottom = np.zeros(len(pct))
                    for cat in pct.columns:
                        vals = pct[cat].fillna(0).values
                        label = str(cat)
                        label = (label[:-2] if label.endswith(".0") else label)[:16]
                        ax.bar(range(len(pct)), vals, bottom=bottom, label=label)
                        bottom = bottom + vals
                    ax.set_xticks(range(len(pct)))
                    ax.set_xticklabels(["{{}}\\n(n={{}})".format(short.get(st, st), int(ct.loc[st].sum())) for st in pct.index], fontsize=7)
                    ax.set_ylim(0, 100)
                    ax.set_ylabel("% of the study's values", fontsize=7)
                    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), framealpha=0.8)
            except Exception as e:
                ax.text(0.5, 0.5, "not drawable per study", ha="center", va="center", fontsize=9, color="#888888")
                wlog("example-analysis: could not stratify {{}}: {{}}".format(col, e))
            ax.set_title("{{}} by {{}}".format(col[:34], STUDY_COL), fontsize=10)
            ax.tick_params(labelsize=8)
        for j in range(len(cols), nrows * COLS):
            axes[j // COLS][j % COLS].axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "columns_by_study_p{{}}.png".format(page + 1)), dpi=130)
        plt.close(fig)

wlog("example-analysis: done - {{}} columns plotted over {{}} overview page(s) and {{}} per-study page(s)".format(
    len(PLOT_COLUMNS), pages, strat_pages))
"""


def merged_data_overview_script(
    merge_node_name: str,
    preview_node_name: str,
) -> str:
    """Generate the example script that summarizes the FULL merged dataset.

    This node is the runnable end of the merged-data chain (airlock nodes cannot
    be accessed in production mode, so this node depends on the merge node
    directly and, by also depending on the fragment node, triggers the airlock
    content for Development-mode use). Every participant can run it.

    It reads the full pooled dataset but writes ONLY aggregate information to
    its output — dataset shape, patients per study, per-column completeness —
    plus the merge node's PDF report(s), the only files of the merge node's
    output that leave the enclave (MERGE_EXPORT_EXTENSIONS). Nothing else is copied.

    Args:
        merge_node_name: Name of the merge compute node (its output, including
            cohortpool's tables/, is mounted at /input/<merge_node_name>/).
        preview_node_name: Name of the airlock node, referenced in the header
            comments so users know how to explore the fragment in Development mode.

    Returns:
        The Python script as a string.
    """
    # Tuple literal of the exported extensions, embedded in the script's copy loop.
    export_exts = ", ".join(repr(ext) for ext in MERGE_EXPORT_EXTENSIONS)
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
#   - the PDF report(s) of the pooling package, copied to harmonization_reports/
#
# EXPORT RULE - of the merge node's output, only PDF files leave the enclave.
# Everything else, including the pooled dataset, all patient-level files and
# every CSV, Parquet, JSON, log and PNG, is NOT copied and is only listed by
# name in the overview report.
{_export_rule_comment()}
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

# Copy the merge-process files that are safe to publish: ONLY the PDF
# report(s) (see the header). Everything else - the merged dataset, the
# patient-level files, every CSV / Parquet / JSON / log / PNG, anything a newer
# pooling package adds - is left out and only listed by name below.
# Sub-folders are kept for the copied files.
EXPORT_EXTENSIONS = ({export_exts},)
wlog("merged-data-overview: overview written; copying the PDF report(s) of the merge (harmonization reports)")
reports_dir = os.path.join(output_dir, "harmonization_reports")
os.makedirs(reports_dir, exist_ok=True)
copied = []
skipped = []
for root, dirs, files in os.walk(merge_dir):
    rel_root = os.path.relpath(root, merge_dir)
    for fname in sorted(files):
        rel_name = fname if rel_root == "." else os.path.join(rel_root, fname)
        if not fname.lower().endswith(EXPORT_EXTENSIONS):
            skipped.append(rel_name)
            continue
        dst_dir = reports_dir if rel_root == "." else os.path.join(reports_dir, rel_root)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.copy(os.path.join(root, fname), os.path.join(dst_dir, fname))
            copied.append(rel_name)
        except Exception as e:
            skipped.append("{{}} (copy failed: {{}})".format(rel_name, e))

with open(report_file, "a") as report:
    report.write("PDF report(s) of the merge copied to harmonization_reports/ ({{}} files):\\n".format(len(copied)))
    for name in copied:
        report.write("  [ok]      {{}}\\n".format(name))
    if skipped:
        report.write("\\nNot exported (only PDF files leave the enclave from the merge output):\\n")
        for name in skipped:
            report.write("  [skipped] {{}}\\n".format(name))

    report.write("\\n" + "=" * 60 + "\\n")
    report.write("OVERVIEW COMPLETE\\n")

wlog("merged-data-overview: done - {{}} PDF report(s) copied, {{}} other files kept inside the enclave".format(len(copied), len(skipped)))
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


# ---------------------------------------------------------------------------
# Optional R compute node ("load-R-packages")
# ---------------------------------------------------------------------------

R_PACKAGES_NODE_NAME = "load-R-packages"

# The node may reach these domains only (ProxyConfiguration.allow_list); every
# domain it accesses is recorded by the platform in accessed_domains.txt of
# the node's result.
# cloud.r-project.org is CRAN itself; packagemanager.posit.co serves the same
# packages as Linux binaries, which the script prefers when it recognises
# the image's distribution (no compilation, so far fewer failures).
# rspm-sync.rstudio.com: Posit Package Manager answers a binary download with
# a redirect to this host (seen in the run of 2026-11-06: every package, pure-R
# ones included, failed with "Couldn't resolve host name" on
# rspm-sync.rstudio.com/bin/4.3-jammy/<hash>.tar.gz), so it must be reachable too.
R_CRAN_DOMAINS = ["cloud.r-project.org", "packagemanager.posit.co", "rspm-sync.rstudio.com"]

# The node's R image is a Nix-built R (4.3.2 at the time of writing) whose
# OpenSSL has no certificate store: without one, every https fetch fails with
# "STORE routines::unregistered scheme" (seen 2026-11-02). The node therefore
# carries Mozilla's CA bundle, from the certifi package of this backend, as an
# additional file; the platform mounts such files at /input/code/<name>.
R_CA_BUNDLE_FILE = "cacert.pem"


def r_ca_bundle_content() -> str:
    """Mozilla's CA bundle (PEM), as shipped by the certifi package."""
    import certifi
    with open(certifi.where(), encoding="utf-8") as fh:
        return fh.read()


# The packages the analysts asked for that the image does NOT provide (list of
# 2026-11-09; survival, dplyr, ggplot2, glmnet, xgboost and the like are in the
# image and need no install). The node installs them at run time, Posit's
# prebuilt binaries first and CRAN sources as fallback, loads them, and runs
# one small exercise with each on every cohort's data, so the result says per
# package whether it installed, loaded and actually works here.
R_PACKAGES = [
    "writexl", "gtsummary", "labelled", "MuMIn", "merTools", "corrr", "DHARMa",
    "forestplot", "timeROC", "RMediation", "forestploter", "showtext", "sysfonts",
    "LTRCtrees", "LTRCforests", "ksamplesLTRC", "miceRanger", "Amelia", "grf",
    "caret", "isotone", "CBPS", "WeightIt",
]


def r_packages_script(inputs: list[dict], liberal_proxy: bool = False) -> str:
    """The R script of the "load-R-packages" node.

    `liberal_proxy` says the node was given ProxyConfiguration.liberal() (any
    domain) instead of the allow-list, a diagnostic mode: the script header
    says so, since the header is what participants read.

    `inputs` lists the dataset(s) the node analyses, one per cohort, each a dict with
      - 'label':    short tag used in the output file names (the cohort's node id)
      - 'path':     the cohort's full data file inside the node (/input/<data node>), or None
      - 'fallback': the cohort's shuffled sample (/input/<shuffled node>), or None; used
                    when the full data file is absent (not uploaded yet)
      - 'source':   what the input is, in words, for the script header

    The script installs the packages of R_PACKAGES (the ones the image does not
    provide), loads them, and runs one small exercise with each of them over
    each cohort's data - the full file when uploaded, the shuffled sample
    otherwise - writing per package whether it installed, loaded and ran, with
    the error when it did not (r_packages_status.csv, package_exercises.csv,
    README.txt). Base R and the image's own packages (survival, ggplot2, lme4)
    are used only for the data preparation and plain plots the exercises need.
    Every file it writes holds aggregates only: no data rows leave the node.
    """
    def q(v):
        return str(v or "").replace('"', "")
    r_inputs = ",\n".join(
        '  list(label = "{label}", path = "{path}", fallback = "{fallback}")'.format(
            label=q(i["label"]), path=q(i.get("path")), fallback=q(i.get("fallback")))
        for i in inputs
    )
    sources = "\n".join(f"#   - {i['label']}: {i['source']}" for i in inputs)
    proxy_note = (
        "#      DIAGNOSTIC MODE: the node may reach ANY domain (liberal proxy), so\n"
        "#      that accessed_domains.txt of the result lists every host the\n"
        "#      package installs actually contact; the allow-list of the regular\n"
        "#      node is then set from that list."
        if liberal_proxy else
        "#      The node is allowed outbound access to cloud.r-project.org,\n"
        "#      packagemanager.posit.co and its download host rspm-sync.rstudio.com\n"
        "#      ONLY (a transparent SNI filter; a host outside the list is cut off\n"
        "#      during the TLS handshake)."
    )
    packages = ", ".join(f'"{p}"' for p in R_PACKAGES)
    return _R_PACKAGES_SCRIPT.replace("__INPUT_SOURCES__", sources) \
                             .replace("__INPUTS__", r_inputs) \
                             .replace("__PACKAGES__", packages) \
                             .replace("__CRAN__", f"https://{R_CRAN_DOMAINS[0]}") \
                             .replace("__PROXY_NOTE__", proxy_note) \
                             .replace("__CA_BUNDLE_FILE__", R_CA_BUNDLE_FILE)


_R_PACKAGES_SCRIPT = r'''###############################################################################
# LOAD R PACKAGES - an R compute node you can run as is and copy
#
# This node shows how to work in R, with packages beyond the built-in R
# environment, on the data of this clean room. Its inputs, one per cohort:
__INPUT_SOURCES__
# For each cohort the script reads the FULL data file when the data owner has
# uploaded it, and the cohort's SHUFFLED SAMPLE otherwise (as long as the room
# has one), so it can be run before every dataset is in. A shuffled sample is
# real values shuffled independently per column: distributions and
# missingness per column are those of the real data, but every relation
# between columns is destroyed, so correlations, survival curves and
# regressions computed on it are meaningless - the outputs say which source
# each cohort's figures come from.
#
# What it does:
#   1. Installs the packages listed in PACKAGES at run time, into a library of
#      its own under R's temp dir (the image's site library is read-only).
__PROXY_NOTE__
#      The platform records every domain the node reaches in
#      accessed_domains.txt of the result. The image is a Nix-built R whose
#      OpenSSL has no certificate store, so the node brings Mozilla's CA
#      bundle along (/input/code/cacert.pem) and points R at it. The image
#      has no compiler either, so from CRAN only packages WITHOUT compiled
#      code install; the status file names each failure's reason, and a
#      package that needs compiling has to be added to the image by
#      Decentriq. Packages the environment already provides are not
#      reinstalled and never upgraded, even when a package to install wants
#      a newer version of one of them: a newer copy would shadow the image's,
#      which the other provided packages were built against (one run upgraded
#      scales under ggplot2 and died loading it); such a package is reported
#      FAILED with the version it needs. Nothing is loaded before every
#      install is done. The outcome per package goes to r_packages_status.csv
#      (already available / installed / FAILED, with the reason), with the
#      seconds its installation and its loading took; the totals are printed
#      and written to README.txt. A package whose CURRENT version does not
#      fit this image (needs a newer R, or a newer version of one of the
#      image's packages, which this node never upgrades) is taken instead from
#      Posit's snapshot of the CRAN of the image's date (see SNAPSHOT_DATE):
#      the version of that day was built for this R and these dependencies.
#      The analyses below only use a package when it loaded.
#   2. Loads every available package with library().
#   3. Runs ONE SMALL EXERCISE WITH EACH PACKAGE over each input - a model, a
#      table or a plot that the package is for - so the result says, per
#      package and cohort, whether it works in this room, not only whether it
#      installed. The data preparation (summary statistics, which columns look
#      like an event, a follow-up time, a treatment) uses base R and the image's
#      own packages (survival, ggplot2, lme4). On a shuffled sample every model
#      is a smoke test: the numbers mean nothing, the fact that the code ran does.
#      Written to /output, AGGREGATES ONLY (never data rows):
#        package_exercises.csv            per cohort and package: ok / failed / skipped, seconds,
#                                         the files written, the error or the reason
#        summary_stats_<input>.csv        n, missing, mean, sd, quartiles per numeric column (base R)
#        <package>_<input>.csv/.png/.xlsx the exercise's own output, e.g. corrr_X.csv,
#                                         forestplot_X.png, caret_X.csv, summary_stats_X.xlsx (writexl)
#        data_sources_used.csv            per cohort: full data or shuffled sample, rows, columns
#        README.txt                       what installed, loaded and ran, and what did not, and why
#        install_logs/<package>.out       R CMD INSTALL output of every package attempted
#        provided_packages.csv            the image's own packages with their versions
#        missing_packages_request.txt     the packages that cannot be installed here, grouped
#                                         by cause (compiler, newer R, image package versions)
#
# To use it on an airlock fragment in the Development tab instead, point the
# path of that cohort in INPUTS at /input/<airlock node>/dataset.csv.
###############################################################################

PACKAGES <- c(__PACKAGES__)
CRAN <- "__CRAN__"
INPUTS <- list(
__INPUTS__
)
OUT <- "/output"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
options(repos = c(CRAN = CRAN), warn = 1, timeout = 600, width = 160,
        # libcurl for every download, so the server name (SNI) the egress
        # filter checks is always sent, and redirects are followed
        download.file.method = "libcurl", url.method = "libcurl")

# The image has fontconfig but no default fonts.conf and no writable cache, so
# every plot printed "Fontconfig error: Cannot load default config file". Point
# it at a config from the store when there is one, and at a temp dir for its cache.
if (!nzchar(Sys.getenv("FONTCONFIG_FILE"))) {
  fc <- Sys.glob(c("/etc/fonts/fonts.conf", "/nix/store/*fontconfig*/etc/fonts/fonts.conf"))
  if (length(fc) > 0) Sys.setenv(FONTCONFIG_FILE = fc[1])
}
if (!nzchar(Sys.getenv("XDG_CACHE_HOME"))) Sys.setenv(XDG_CACHE_HOME = file.path(tempdir(), "cache"))
# The image has no usable /etc/localtime: without TZ every date call warns about it.
if (!nzchar(Sys.getenv("TZ"))) Sys.setenv(TZ = "UTC")

notes <- character(0)
note <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  notes <<- c(notes, msg)
}

# ---- 0. where packages can go, and whether the repositories are reachable ----
# Run every expression with its warnings and messages collected (not hidden),
# so the status table can say WHY an install failed.
capture <- function(expr) {
  msgs <- character(0)
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e) { msgs <<- c(msgs, paste("error:", conditionMessage(e))); NULL }),
    warning = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { msgs <<- c(msgs, trimws(conditionMessage(m))); invokeRestart("muffleMessage") })
  list(value = value, messages = unique(msgs))
}
one_line <- function(x, n = 240) {
  x <- gsub("[[:space:]]+", " ", paste(x, collapse = " | "))
  if (nchar(x) > n) paste0(substr(x, 1, n), "...") else x
}

cat(R.version.string, "on", R.version$platform, "\n")
lps <- .libPaths()
nix_r <- any(grepl("^/nix/store/", lps))
cat(length(lps), "library paths", if (nix_r) "(a Nix-built R: one store path per package)" else "", "\n")
cat("  first:", lps[1], "\n  last: ", lps[length(lps)], "\n")
cat("  packages already provided:", length(unique(unlist(lapply(lps, list.files)))), "\n")
# The image's packages with their versions, for the record and for choosing
# replacements: a package is installable here only when everything it pulls in
# is either pure R or already in this list (at a version it accepts).
ip <- installed.packages()
write.csv(data.frame(package = ip[, "Package"], version = ip[, "Version"], stringsAsFactors = FALSE)[order(ip[, "Package"]), ],
          file.path(OUT, "provided_packages.csv"), row.names = FALSE)
# The image's site library is usually read-only: install into a library of
# our own under R's temp dir (always writable) and put it first on the path.
LIB <- file.path(tempdir(), "rlib")
dir.create(LIB, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(LIB, .libPaths()))
cat("Packages are installed into", LIB, "\n")

proxy <- Sys.getenv(c("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"), unset = NA)
proxy <- proxy[!is.na(proxy) & proxy != ""]
cat("Proxy environment:", if (length(proxy) == 0) "(none set)" else paste(names(proxy), proxy, sep = "=", collapse = "  "), "\n")

# Certificate authorities for https. The image's OpenSSL has no certificate
# store (every https fetch failed with "STORE routines::unregistered scheme"),
# so the node carries Mozilla's CA bundle as an additional file, mounted by
# the platform under /input/code. R's libcurl reads CURL_CA_BUNDLE, OpenSSL
# itself reads SSL_CERT_FILE; both are set before the first download.
ca_candidates <- c("/input/code/__CA_BUNDLE_FILE__", "./__CA_BUNDLE_FILE__",
                   Sys.getenv("SSL_CERT_FILE"), Sys.getenv("CURL_CA_BUNDLE"),
                   "/etc/ssl/certs/ca-certificates.crt", "/etc/ssl/certs/ca-bundle.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                   Sys.glob("/nix/store/*cacert*/etc/ssl/certs/ca-bundle.crt"))
ca_candidates <- ca_candidates[nzchar(ca_candidates)]
CA_BUNDLE <- ca_candidates[file.exists(ca_candidates)][1]
if (is.na(CA_BUNDLE)) {
  note("No CA bundle found (looked for ", paste(unique(ca_candidates), collapse = ", "), "): https downloads will fail certificate verification")
} else {
  Sys.setenv(CURL_CA_BUNDLE = CA_BUNDLE, SSL_CERT_FILE = CA_BUNDLE)
  cat("CA bundle:", CA_BUNDLE, "(", file.size(CA_BUNDLE), "bytes )\n")
}

# Where packages come from. The image ships no compiler, so from CRAN only
# packages without compiled code can be installed (R builds the rest from
# source and fails). Posit Package Manager serves every CRAN package as a
# pre-built Linux binary, chosen by the distribution in the URL and the R
# version in the user agent; R CMD INSTALL recognises such a tarball and
# only unpacks it. The image is Nix-built, not Ubuntu, but its C library is
# newer than Ubuntu 22.04's (glibc 2.38 vs 2.35, probe of 2026-11-06) and
# the C++/Fortran runtimes its own compiled packages need are in the Nix
# store (preloaded below), so the Ubuntu 22.04 binaries are tried first for
# every image; CRAN sources remain the fallback for what Posit lacks.
os <- if (file.exists("/etc/os-release")) readLines("/etc/os-release", warn = FALSE) else character(0)
field <- function(key) { v <- sub(paste0("^", key, "="), "", grep(paste0("^", key, "="), os, value = TRUE)); gsub('"', "", v)[1] }
distro <- field("ID"); codename <- field("VERSION_CODENAME")
BINARY_CODENAME <- if (!is.na(distro) && !is.na(codename) && distro %in% c("ubuntu", "debian")) codename else "jammy"
cat("Distribution:", if (length(os) == 0) "no /etc/os-release" else paste(distro, codename),
    if (nix_r) "(Nix-built R)" else "", "- Posit binaries built for", BINARY_CODENAME, "tried first, CRAN sources as fallback\n")
PPM <- sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", BINARY_CODENAME)
# The same repository frozen at a date consistent with the image's own package
# versions (R 4.3.2 of 2023-10-31; ggplot2 3.4.4, glue 1.6.2, i.e. the CRAN of
# November 2023, nixpkgs 23.11). A package whose current version needs a newer
# R or a newer version of an image package than the image has is taken from
# here instead: the version of that day was built for this R and for these
# dependency versions, as a prebuilt binary too. Older than the image is
# harmless (an older package asks for older dependencies); newer is not.
SNAPSHOT_DATE <- "2023-11-30"
SNAPSHOT <- sprintf("https://packagemanager.posit.co/cran/__linux__/%s/%s", BINARY_CODENAME, SNAPSHOT_DATE)
AVAILABLE_SNAPSHOT <- NULL   # its package index, fetched on first use
# The user agent Posit reads to serve a binary: "R/<version> R (<version> <platform> <arch> <os>)".
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
                                paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
cat("User agent for the repositories:", getOption("HTTPUserAgent"), "\n")
REPOS <- c(PPM = PPM, CRAN = CRAN)
options(repos = REPOS)

reachable <- character(0)
for (nm in names(REPOS)) {
  r <- capture(available.packages(repos = REPOS[[nm]]))
  n <- if (is.null(r$value)) 0 else nrow(r$value)
  cat(nm, "(", REPOS[[nm]], "): ", if (n > 0) paste(n, "packages listed") else paste("NOT reachable -", one_line(r$messages)), "\n", sep = "")
  if (n > 0) reachable <- c(reachable, nm)
}
if (length(reachable) == 0) {
  note("No package repository is reachable from this node: nothing can be installed. ",
       "Check the node's internet access (proxy allow-list) and accessed_domains.txt in this result.")
  # ---- network diagnostics: what exactly fails between this node and the repositories?
  # Each probe is one request; together they tell a certificate-trust problem
  # (verification off succeeds), a protocol problem (TLS 1.2 only succeeds)
  # and an HTTPS-only block (plain http succeeds) apart.
  cat("\n-- Network diagnostics --\n")
  env <- Sys.getenv(); nm <- names(env)
  sel <- nm[grepl("proxy|ssl|cert|curl|ca_bundle|decentriq", nm, ignore.case = TRUE)]
  cat("Environment variables mentioning proxy/ssl/cert/curl:",
      if (length(sel) > 0) paste(sel, env[sel], sep = "=", collapse = "  ") else "(none)", "\n")
  for (d in c("/etc/ssl/certs", "/etc/pki", "/usr/local/share/ca-certificates", "/usr/share/ca-certificates", "/run/secrets", "/etc/decentriq")) {
    if (dir.exists(d)) {
      f <- list.files(d, pattern = "[.](crt|pem)$", recursive = TRUE, full.names = TRUE)
      cat("  ", d, ": ", length(f), " certificate file(s)", if (length(f) > 0) paste0(", e.g. ", paste(head(basename(f), 4), collapse = ", ")) else "", "\n", sep = "")
    }
  }
  if (requireNamespace("curl", quietly = TRUE)) {
    cv <- curl::curl_version()
    cat("libcurl ", cv$version, " with ", cv$ssl_version, "\n", sep = "")
    probe <- function(label, url, ...) {
      r <- tryCatch({
        h <- curl::new_handle(timeout = 30, ...)
        res <- curl::curl_fetch_memory(url, handle = h)
        paste0("HTTP ", res$status_code, ", ", length(res$content), " bytes")
      }, error = function(e) paste("FAILED:", conditionMessage(e)))
      cat("  ", label, ": ", one_line(r, 160), "\n", sep = "")
    }
    probe("https CRAN, certificate verification on ", "https://cloud.r-project.org/src/contrib/PACKAGES")
    probe("https CRAN, certificate verification OFF", "https://cloud.r-project.org/src/contrib/PACKAGES", ssl_verifypeer = FALSE, ssl_verifyhost = FALSE)
    probe("https CRAN, TLS 1.2 only               ", "https://cloud.r-project.org/src/contrib/PACKAGES", sslversion = 6L)
    probe("http  CRAN (no TLS)                    ", "http://cloud.r-project.org/src/contrib/PACKAGES")
    if (!is.null(PPM)) {
      probe("https Posit PM, verification on        ", paste0(PPM, "/src/contrib/PACKAGES"))
      probe("https Posit PM, verification OFF       ", paste0(PPM, "/src/contrib/PACKAGES"), ssl_verifypeer = FALSE, ssl_verifyhost = FALSE)
    }
  } else {
    cat("The 'curl' R package is not available; falling back to download.file probes\n")
    for (u in c("http://cloud.r-project.org/src/contrib/PACKAGES", "https://cloud.r-project.org/src/contrib/PACKAGES")) {
      r <- capture(download.file(u, tempfile(), quiet = TRUE))
      cat("  ", u, ": ", if (is.null(r$value)) one_line(r$messages, 160) else "OK", "\n", sep = "")
    }
  }
}
REPOS <- REPOS[reachable]

# ---- 1. install what is missing ---------------------------------------------
# Installed-or-not is answered from the library paths, without loading anything:
# requireNamespace() would load the package and its imports, and a namespace
# loaded now cannot be swapped for another version later.
# Binaries built on Ubuntu expect libstdc++, libgfortran, libgomp, libquadmath
# and libgcc_s at the usual system paths, which a Nix image does not have; the
# dynamic loader does, however, satisfy a dependency by a library that is
# already loaded in the process under that name. The runtimes exist in the
# Nix store (the image's own compiled packages were built against them), so
# they are loaded here by full path, globally, before any package install.
runtime_libs <- c("libstdc++.so.6", "libgfortran.so.5", "libgfortran.so.4", "libgomp.so.1", "libquadmath.so.0", "libgcc_s.so.1")
preloaded <- character(0)
for (lib in runtime_libs) {
  found <- c(Sys.glob(file.path("/nix/store", "*", "lib", lib)), Sys.glob(file.path("/usr/lib", "*", lib)), Sys.glob(file.path("/usr/lib", lib)))
  found <- found[file.exists(found)]
  if (length(found) == 0) next
  ok <- tryCatch({ dyn.load(found[1], local = FALSE); TRUE }, error = function(e) FALSE)
  if (ok) preloaded <- c(preloaded, found[1])
}
cat("Runtime libraries preloaded for prebuilt binaries:", if (length(preloaded) > 0) paste(basename(preloaded), collapse = ", ") else "(none found)", "\n")

is_installed <- function(pkg, where = .libPaths()) nzchar(system.file(package = pkg, lib.loc = where))

# The packages the image provides are never upgraded. install.packages() would
# fetch a newer version of a dependency it finds too old; a pure-R one (scales,
# say) installs fine into our library, which comes first on the path, and then
# shadows the version the image's other packages (ggplot2) were built against.
# Hiding the provided packages from the list of what the repositories offer
# keeps them out of the dependency resolution: a package that needs a newer
# version of one of them fails to install, with that requirement as the reason.
PROVIDED <- rownames(installed.packages())
AVAILABLE <- NULL
AVAILABLE_CRAN <- NULL  # the CRAN-only index, fetched on the first source retry
if (length(REPOS) > 0) {
  AVAILABLE <- available.packages(repos = REPOS)
  AVAILABLE <- AVAILABLE[!(rownames(AVAILABLE) %in% PROVIDED), , drop = FALSE]
  cat(length(PROVIDED), "provided packages are excluded from installation and upgrade;",
      nrow(AVAILABLE), "others can be installed\n")
}

INSTALL_LOGS <- file.path(OUT, "install_logs")
dir.create(INSTALL_LOGS, showWarnings = FALSE)

# Why an install failed, from install.packages()'s messages and the .out files
# of the packages it attempted (the failing one is usually a dependency):
#   compiler      - the image has no C/C++/Fortran toolchain, so any package
#                   with compiled code fails ("gcc: not found", "make: not found")
#   newer R       - CRAN only has a build for a newer R than the image's
#   image package - a dependency needs a newer version of a package the image
#                   provides, which this node never upgrades (see PROVIDED)
#   other         - anything else: the .out file says what
# Every cause but "network" gets a second chance from the snapshot (below).
failure_cause <- function(pkg, msgs) {
  m <- paste(msgs, collapse = " ")
  if (grepl("not available for this version of R", m)) return("newer R")
  if (grepl("Couldn't resolve host|Could not resolve host|SSL connect error|Connection timed out|Failed to connect", m)) {
    host <- regmatches(m, regexpr("https?://[A-Za-z0-9.-]+", m))
    return(paste0("network (", if (length(host) > 0) sub("^https?://", "", host[1]) else "host not reachable", ")"))
  }
  if (grepl("dependenc(y|ies) '.*' (is|are) not available", m)) {
    need <- regmatches(m, regexpr("dependenc(y|ies) '[^|]*' (is|are) not available", m))
    deps <- regmatches(need, gregexpr("'[^']+'", need))[[1]]
    deps <- gsub("'", "", deps)
    if (length(deps) > 0 && all(deps %in% PROVIDED)) return(paste0("image package (", paste(deps, collapse = ", "), ")"))
  }
  # the packages whose install was attempted for this one: their .out files
  tried <- regmatches(m, gregexpr("installation of package '[^']+' had non-zero exit status", m))[[1]]
  tried <- unique(c(gsub("installation of package '([^']+)'.*", "\\1", tried), pkg))
  for (dep in tried) {
    f <- file.path(INSTALL_LOGS, paste0(dep, ".out"))
    if (!file.exists(f)) next
    txt <- paste(readLines(f, warn = FALSE), collapse = "\n")
    if (grepl("(gcc|cc|g\\+\\+|gfortran|make|clang)[^\n]*(not found|No such file)|no acceptable C compiler|C\\+\\+ compiler cannot create|cannot run C compiled programs", txt))
      return(paste0("compiler (", dep, ")"))
    if (grepl("fatal error: [^\n]*\\.h: No such file|configure: error", txt))
      return(paste0("system library (", dep, ")"))
    if (grepl("cannot open shared object file", txt)) {
      lib <- regmatches(txt, regexpr("[A-Za-z0-9_+.-]+[.]so[.0-9]*(?=: cannot open shared object file)", txt, perl = TRUE))
      return(paste0("system library (", dep, if (length(lib) > 0) paste0(": ", lib[1]) else "", ")"))
    }
  }
  "other"
}

# The snapshot's package index (minus the image's packages), fetched once.
snapshot_index <- function() {
  if (is.null(AVAILABLE_SNAPSHOT)) {
    r <- capture(available.packages(repos = c(SNAPSHOT = SNAPSHOT)))
    idx <- if (is.null(r$value)) matrix(character(0), 0, 0) else r$value[!(rownames(r$value) %in% PROVIDED), , drop = FALSE]
    cat("Snapshot of ", SNAPSHOT_DATE, " (", SNAPSHOT, "): ",
        if (nrow(idx) > 0) paste(nrow(idx), "packages listed") else paste("NOT reachable -", one_line(r$messages)), "\n", sep = "")
    AVAILABLE_SNAPSHOT <<- idx
  }
  AVAILABLE_SNAPSHOT
}
in_snapshot <- function(pkg) { idx <- snapshot_index(); nrow(idx) > 0 && pkg %in% rownames(idx) }
# Install a package's version of the snapshot date into our library (its
# dependencies from the snapshot too, those already in the library kept).
# Returns install.packages()'s messages; is_installed() says whether it worked.
install_snapshot <- function(pkg) {
  idx <- snapshot_index()
  if (nrow(idx) == 0) return(paste("snapshot repository", SNAPSHOT, "not reachable"))
  if (!(pkg %in% rownames(idx))) return(paste0("'", pkg, "' is not in the snapshot of ", SNAPSHOT_DATE))
  r <- capture(install.packages(pkg, lib = LIB, repos = c(SNAPSHOT = SNAPSHOT), available = idx, quiet = TRUE,
                                dependencies = c("Depends", "Imports", "LinkingTo"), keep_outputs = INSTALL_LOGS))
  r$messages
}

status <- data.frame(package = PACKAGES, status = NA_character_, source = "",
                     install_seconds = NA_real_, load_seconds = NA_real_,
                     cause = "", reason = "", stringsAsFactors = FALSE)
t_install <- Sys.time()
for (i in seq_along(PACKAGES)) {
  pkg <- PACKAGES[i]
  if (is_installed(pkg)) {
    status$status[i] <- "already available"
    status$source[i] <- "image"
    next
  }
  if (length(REPOS) == 0) {
    status$status[i] <- "FAILED"
    status$reason[i] <- "no repository reachable"
    next
  }
  t0 <- Sys.time()
  cat("Installing", pkg, "...\n")
  # keep_outputs: R CMD INSTALL runs as a child process, so its output (the
  # compiler error, the missing header) never reaches capture(); R writes it to
  # <package>.out in that directory instead, one file per package it attempts,
  # dependencies included. The files go to /output/install_logs.
  r <- capture(install.packages(pkg, lib = LIB, repos = REPOS, available = AVAILABLE, quiet = TRUE,
                                dependencies = c("Depends", "Imports", "LinkingTo"),
                                keep_outputs = INSTALL_LOGS))
  # A download that failed (a redirect to a host outside the allow-list, a
  # binary Posit has not built) must not cost the package: the same install
  # is tried again from CRAN sources alone, which is what the node did before
  # the binaries were tried first. Pure-R packages always install this way.
  if (!is_installed(pkg, LIB) && "PPM" %in% names(REPOS) &&
      grepl("cannot open URL|Couldn't resolve host|download of package|status was", paste(r$messages, collapse = " "))) {
    cat("  binary download failed; retrying", pkg, "from CRAN sources\n")
    if (is.null(AVAILABLE_CRAN)) {
      AVAILABLE_CRAN <- available.packages(repos = c(CRAN = CRAN))
      AVAILABLE_CRAN <- AVAILABLE_CRAN[!(rownames(AVAILABLE_CRAN) %in% PROVIDED), , drop = FALSE]
    }
    r2 <- capture(install.packages(pkg, lib = LIB, repos = c(CRAN = CRAN), available = AVAILABLE_CRAN, quiet = TRUE,
                                   dependencies = c("Depends", "Imports", "LinkingTo"),
                                   keep_outputs = INSTALL_LOGS))
    r$messages <- unique(c(r$messages, paste("source retry:", r2$messages)))
  }
  # Still not installed, and not for want of network: the current version does
  # not fit this image (a newer R, or a newer version of an image package,
  # asked by the package itself or by a dependency). The version of the
  # snapshot date is installed instead, with the current version's problem
  # kept as the reason.
  if (!is_installed(pkg, LIB) && length(REPOS) > 0 && !grepl("^network", failure_cause(pkg, r$messages))) {
    cat("  the current version does not install here (", failure_cause(pkg, r$messages), "); trying the version of ", SNAPSHOT_DATE, "\n", sep = "")
    m3 <- install_snapshot(pkg)
    if (is_installed(pkg, LIB)) {
      status$source[i] <- paste("snapshot", SNAPSHOT_DATE)
      status$reason[i] <- one_line(c(paste0("current CRAN version does not fit this image (", failure_cause(pkg, r$messages),
                                            "); installed version ", as.character(packageVersion(pkg, lib.loc = LIB)), " of ", SNAPSHOT_DATE, " instead:"), r$messages))
      cat("  installed ", pkg, " ", as.character(packageVersion(pkg, lib.loc = LIB)), " from the snapshot\n", sep = "")
    } else {
      r$messages <- unique(c(r$messages, paste("snapshot retry:", m3)))
    }
  }
  status$install_seconds[i] <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  if (is_installed(pkg, LIB)) {
    status$status[i] <- "installed"
    if (!nzchar(status$source[i])) status$source[i] <- "current CRAN"
  } else {
    status$status[i] <- "FAILED"
    msgs <- if (length(r$messages) > 0) r$messages else "no message from install.packages; see the console log above"
    status$cause[i] <- failure_cause(pkg, msgs)
    status$reason[i] <- one_line(msgs)
    cat("  FAILED (", status$cause[i], "): ", status$reason[i], "\n", sep = "")
  }
}
# ---- 1b. do the installed packages accept the image's package versions? -----
# A current prebuilt package installs fine and then fails to load when it (or
# a dependency installed with it) imports a newer version of an image package
# than the image has (run of 2026-11-09: gtsummary wanted glue >= 1.8.0 against
# 1.6.2, WeightIt wanted ggplot2 >= 4.0.0 against 3.4.4). The requirements in
# the DESCRIPTION files of our library are checked against the versions on the
# path; a package with an unmet one is replaced by its version of the snapshot
# date, which asked for what the image has. Repeated while anything changes.
dep_requirements <- function(p) {
  d <- tryCatch(suppressWarnings(packageDescription(p, lib.loc = LIB)), error = function(e) NULL)
  if (is.null(d) || !is.list(d)) return(NULL)
  items <- unlist(lapply(c("Depends", "Imports", "LinkingTo"), function(f)
    if (is.null(d[[f]]) || is.na(d[[f]])) character(0) else strsplit(gsub("[[:space:]]+", " ", d[[f]]), ",")[[1]]))
  items <- trimws(items)
  items <- items[nzchar(items)]
  if (length(items) == 0) return(NULL)
  dep <- sub("^([A-Za-z0-9.]+).*$", "\\1", items)
  has_min <- grepl("\\(\\s*>=?\\s*[0-9.-]+\\s*\\)", items)
  min_version <- ifelse(has_min, sub("^.*\\(\\s*>=?\\s*([0-9.-]+)\\s*\\).*$", "\\1", items), NA_character_)
  data.frame(dep = dep, min_version = min_version, stringsAsFactors = FALSE)[dep != "R", , drop = FALSE]
}
unmet_requirements <- function(p) {
  req <- dep_requirements(p)
  if (is.null(req) || nrow(req) == 0) return(character(0))
  out <- character(0)
  for (k in seq_len(nrow(req))) {
    if (is.na(req$min_version[k])) next
    inst <- tryCatch(as.character(packageVersion(req$dep[k])), error = function(e) NA_character_)
    if (is.na(inst)) next
    if (utils::compareVersion(inst, req$min_version[k]) < 0)
      out <- c(out, sprintf("%s >= %s (the image has %s)", req$dep[k], req$min_version[k], inst))
  }
  out
}
replaced <- character(0)
for (round in 1:3) {
  changed <- FALSE
  for (p in rownames(installed.packages(lib.loc = LIB, noCache = TRUE))) {
    u <- unmet_requirements(p)
    if (length(u) == 0) next
    old <- as.character(packageVersion(p, lib.loc = LIB))
    i <- match(p, PACKAGES)
    if (!in_snapshot(p)) {
      note("Installed ", p, " ", old, " needs ", paste(u, collapse = ", "), " and has no version in the snapshot of ",
           SNAPSHOT_DATE, ": it, and what imports it, will not load")
      next
    }
    cat("Installed ", p, " ", old, " needs ", paste(u, collapse = ", "), "; replacing it with its version of ", SNAPSHOT_DATE, "\n", sep = "")
    suppressMessages(remove.packages(p, lib = LIB))
    m <- install_snapshot(p)
    if (is_installed(p, LIB)) {
      changed <- TRUE
      new <- as.character(packageVersion(p, lib.loc = LIB))
      replaced <- c(replaced, sprintf("%s %s -> %s (%s needed %s)", p, old, new, old, paste(u, collapse = ", ")))
      cat("  installed ", p, " ", new, " from the snapshot\n", sep = "")
      if (!is.na(i)) {
        status$source[i] <- paste("snapshot", SNAPSHOT_DATE)
        status$reason[i] <- one_line(paste0("current version ", old, " needs ", paste(u, collapse = ", "), "; installed version ", new, " of ", SNAPSHOT_DATE, " instead"))
      }
    } else {
      note("  ", p, " could not be reinstalled from the snapshot (", one_line(m, 160), "): what imports it will not load")
      if (!is.na(i)) {
        status$status[i] <- "FAILED"
        status$cause[i] <- paste0("image package (", paste(sub(" >=.*", "", u), collapse = ", "), ")")
        status$reason[i] <- one_line(c(paste0("version ", old, " needs ", paste(u, collapse = ", "), "; snapshot retry:"), m))
      }
    }
  }
  if (!changed) break
}
total_install <- round(as.numeric(difftime(Sys.time(), t_install, units = "secs")), 1)

# ---- 2. load every available package, timing each library() call -----------
# A package can be installed and still not load (a system library the enclave
# lacks, a dependency version it does not like): that is a FAILED row with
# the load error as its reason, never the end of the run.
# A shared library a prebuilt binary needs may be in the image under a path
# the dynamic linker does not search (libfreetype.so.6 for sysfonts, run of
# 2026-11-09): it is looked for in the Nix store and the usual library
# directories, loaded into the process, and the package is tried again.
find_shared_lib <- function(name) {
  hits <- Sys.glob(c(file.path("/nix/store/*/lib", name), file.path("/nix/store/*/lib", paste0(name, ".*")),
                     file.path(c("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/lib/x86_64-linux-gnu"), name)))
  hits[file.exists(hits)]
}
preloaded_libs <- character(0)
load_with_libs <- function(pkg) {
  for (attempt in 1:6) {
    err <- tryCatch({ suppressPackageStartupMessages(library(pkg, character.only = TRUE)); NULL },
                    error = function(e) conditionMessage(e))
    if (is.null(err)) return(NULL)
    lib <- regmatches(err, regexpr("[A-Za-z0-9_+.-]+[.]so[.0-9]*(?=: cannot open shared object file)", err, perl = TRUE))
    if (length(lib) == 0) return(err)
    hit <- find_shared_lib(lib[1])
    if (length(hit) == 0) return(err)
    ok <- tryCatch({ dyn.load(hit[1], local = FALSE); TRUE }, error = function(e) FALSE)
    if (!ok) return(err)
    cat("  preloaded ", lib[1], " from ", hit[1], " for ", pkg, "\n", sep = "")
    preloaded_libs <<- c(preloaded_libs, paste0(lib[1], " (", hit[1], ")"))
  }
  err
}
loaded <- character(0)
try_load <- function(i) {
  pkg <- PACKAGES[i]
  t0 <- Sys.time()
  err <- load_with_libs(pkg)
  status$load_seconds[i] <<- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2)
  if (is.null(err)) {
    loaded <<- c(loaded, pkg)
    if (status$status[i] == "FAILED") { status$status[i] <<- "installed"; status$cause[i] <<- ""; status$reason[i] <<- "" }
  } else {
    status$status[i] <<- "FAILED"
    status$reason[i] <<- one_line(paste("installed but does not load:", err))
    lib <- regmatches(err, regexpr("[A-Za-z0-9_+.-]+[.]so[.0-9]*(?=: cannot open shared object file)", err, perl = TRUE))
    status$cause[i] <<- if (length(lib) > 0) paste0("system library (", lib[1], ")") else "load error"
    cat("Loading ", pkg, " ... FAILED (", status$cause[i], "): ", status$reason[i], "\n", sep = "")
  }
}
t_load <- Sys.time()
for (i in seq_along(PACKAGES)) if (is_installed(PACKAGES[i])) try_load(i)
# A second pass for what failed to load: a package may only have needed one
# loaded after it (showtext needs sysfonts, whose library was found above).
failed_load <- which(status$status == "FAILED" & !is.na(status$load_seconds))
for (i in failed_load) if (is_installed(PACKAGES[i])) { cat("Retrying ", PACKAGES[i], " ...\n", sep = ""); try_load(i) }
total_load <- round(as.numeric(difftime(Sys.time(), t_load, units = "secs")), 1)
if (length(replaced) > 0) note("Replaced by the version of ", SNAPSHOT_DATE, " because the current version needs newer image packages: ", paste(replaced, collapse = "; "))
if (length(preloaded_libs) > 0) note("Shared libraries of the image preloaded for prebuilt binaries: ", paste(unique(preloaded_libs), collapse = ", "))

write.csv(status, file.path(OUT, "r_packages_status.csv"), row.names = FALSE)
cat("\nPackage status (seconds to install / to load; full reasons in r_packages_status.csv):\n")
shown <- status
shown$reason <- ifelse(nchar(shown$reason) > 70, paste0(substr(shown$reason, 1, 70), "..."), shown$reason)
print(shown, row.names = FALSE)

# What to ask the image maintainers for, grouped by cause: the packages the
# node cannot install here and why. Written to missing_packages_request.txt.
failed_rows <- status[status$status == "FAILED", ]
if (nrow(failed_rows) > 0) {
  kind <- sub(" \\(.*", "", failed_rows$cause)
  req <- c(paste0("R ", R.version$major, ".", R.version$minor, " image, run of ", format(Sys.time(), "%Y-%m-%d")),
           paste(nrow(failed_rows), "of", length(PACKAGES), "requested packages cannot be installed in this image:"), "")
  for (k in c("compiler", "system library", "newer R", "image package", "network", "load error", "other")) {
    rows <- failed_rows[kind == k, ]
    if (nrow(rows) == 0) next
    head_line <- switch(k,
      "compiler" = "Need a C/C++/Fortran compiler (the image has none), or the packages pre-installed:",
      "system library" = "Need a system library or header the image lacks (see install_logs/<package>.out):",
      "newer R" = paste0("Need a newer R than ", R.version$major, ".", R.version$minor, " (CRAN has no build for this version, and the snapshot of ", SNAPSHOT_DATE, " none either):"),
      "image package" = paste0("Need a newer version of a package the image provides (this node never upgrades those; the snapshot of ", SNAPSHOT_DATE, " did not help):"),
      "network" = "A download was refused by the node's internet allow-list (the host in brackets must be allowed; accessed_domains.txt lists it):",
      "load error" = "Installed (prebuilt binary) but failed to load for another reason (see r_packages_status.csv):",
      "Failed for another reason (see install_logs/<package>.out):")
    req <- c(req, head_line, paste0("  ", rows$package, ifelse(nchar(rows$cause) > nchar(k), paste0("  [", sub("^[^(]*\\(", "", sub("\\)$", "", rows$cause)), "]"), "")), "")
  }
  writeLines(req, file.path(OUT, "missing_packages_request.txt"))
  cat("\n", paste(req, collapse = "\n"), "\n", sep = "")
}
note("Loaded packages: ", paste(loaded, collapse = ", "))
failed <- status$package[status$status == "FAILED"]
if (length(failed) > 0) note("Not available (install failed): ", paste(failed, collapse = ", "))
note("Total time to install the missing packages: ", total_install, " s (",
     sum(status$status == "installed"), " installed, ", sum(status$status == "already available"),
     " already available, ", length(failed), " failed)")
note("Total time to load the ", length(loaded), " available packages: ", total_load, " s")
slowest <- status[!is.na(status$load_seconds), ]
slowest <- head(slowest[order(-slowest$load_seconds), ], 5)
if (nrow(slowest) > 0) note("Slowest to load: ", paste0(slowest$package, " (", slowest$load_seconds, " s)", collapse = ", "))
has <- function(pkg) pkg %in% loaded

# ---- helpers ----------------------------------------------------------------
present <- function(path) nzchar(path) && file.exists(path) && !dir.exists(path) && file.size(path) > 0
read_table <- function(path) {
  na <- c("", "NA", "NaN", "null", "NULL", "None")
  if (has("data.table")) {
    as.data.frame(data.table::fread(path, na.strings = na, showProgress = FALSE))
  } else {
    read.csv(path, na.strings = na, check.names = FALSE, stringsAsFactors = FALSE)
  }
}
# The file to analyse for one input: the full data when uploaded, else the
# shuffled sample; NULL (with a note) when neither is there.
pick_source <- function(inp) {
  if (present(inp$path)) return(list(path = inp$path, kind = "full data"))
  if (present(inp$fallback)) return(list(path = inp$fallback, kind = "shuffled sample"))
  NULL
}

# Columns that look like a follow-up time / an event indicator / a study label,
# recognised by name and content. Returns NULL when nothing fits.
find_time_col <- function(df, exclude = character(0)) {
  cand <- names(df)[grepl("time|follow|fu_|duration|days|months|years|survival", names(df), ignore.case = TRUE)]
  cand <- setdiff(cand, exclude)
  cand <- cand[vapply(cand, function(c) is.numeric(df[[c]]) && sum(!is.na(df[[c]])) > 20, logical(1))]
  if (length(cand) == 0) NULL else cand[which.max(vapply(cand, function(c) sum(!is.na(df[[c]])), numeric(1)))]
}
as_event <- function(x) {
  if (is.logical(x)) return(as.integer(x))
  if (is.numeric(x) && all(x[!is.na(x)] %in% c(0, 1))) return(as.integer(x))
  v <- tolower(trimws(as.character(x)))
  if (all(v[!is.na(v)] %in% c("yes", "no"))) return(as.integer(v == "yes"))
  if (all(v[!is.na(v)] %in% c("true", "false"))) return(as.integer(v == "true"))
  NULL
}
find_event_col <- function(df) {
  cand <- names(df)[grepl("death|dead|died|event|status|mortal|outcome", names(df), ignore.case = TRUE)]
  for (c in cand) {
    ev <- as_event(df[[c]])
    if (!is.null(ev) && sum(!is.na(ev)) > 20 && length(unique(na.omit(ev))) == 2) return(c)
  }
  NULL
}
find_group_col <- function(df) {
  cand <- names(df)[grepl("^(study|cohort|source_study|study_name|dataset)$", names(df), ignore.case = TRUE)]
  for (c in cand) {
    k <- length(unique(na.omit(df[[c]])))
    if (k >= 2 && k <= 12) return(c)
  }
  NULL
}
write_table <- function(x, file) {
  x <- as.data.frame(x)
  if (has("readr")) readr::write_csv(x, file.path(OUT, file)) else write.csv(x, file.path(OUT, file), row.names = FALSE)
}
save_plot <- function(p, file, width = 10, height = 6) {
  ggplot2::ggsave(file.path(OUT, file), p, width = width, height = height, dpi = 120, bg = "white")
  note("wrote ", file)
}

# ---- 3. one exercise per package, on every input ---------------------------
# avail(): the package loaded here, or the image provides it (survival,
# ggplot2, lme4 ...) - used for the preparation and plain plots only.
avail <- function(pkg) has(pkg) || suppressWarnings(requireNamespace(pkg, quietly = TRUE))
exercises <- data.frame(cohort = character(0), package = character(0), exercise = character(0),
                        status = character(0), seconds = numeric(0), outputs = character(0),
                        detail = character(0), stringsAsFactors = FALSE)
record <- function(label, pkg, name, status, secs, outputs, detail) {
  exercises <<- rbind(exercises, data.frame(cohort = label, package = pkg, exercise = name, status = status,
                                            seconds = round(secs, 1), outputs = paste(outputs, collapse = "; "),
                                            detail = one_line(detail, 300), stringsAsFactors = FALSE))
  cat(sprintf("  %-13s %-8s %s%s\n", pkg, status, name, if (nzchar(detail)) paste0(" - ", one_line(detail, 160)) else ""))
}
# run_exercise(): skipped when the package did not load or the data lacks what
# the exercise needs; failed with the error otherwise; ok with its outputs.
run_exercise <- function(label, pkg, name, needs_ok, needs_msg, fn) {
  if (!has(pkg)) { record(label, pkg, name, "skipped", 0, "", "package not loaded"); return(invisible()) }
  if (!needs_ok) { record(label, pkg, name, "skipped", 0, "", needs_msg); return(invisible()) }
  t0 <- Sys.time()
  res <- tryCatch(fn(), error = function(e) structure(list(error = conditionMessage(e)), class = "exercise_error"))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(res, "exercise_error")) {
    record(label, pkg, name, "failed", secs, "", res$error)
  } else {
    record(label, pkg, name, "ok", secs, res$outputs, if (is.null(res$detail)) "" else res$detail)
  }
  invisible()
}
out_file <- function(pkg, label, ext) paste0(pkg, "_", label, ".", ext)
save_png <- function(file, expr, width = 1000, height = 650) {
  png(file.path(OUT, file), width = width, height = height)
  on.exit(dev.off(), add = TRUE)
  force(expr)
  file
}
# Columns that look like a binary treatment / exposure: 0/1 with both arms present.
find_binary_col <- function(df, exclude = character(0), min_arm = 10) {
  for (c in setdiff(names(df), exclude)) {
    ev <- as_event(df[[c]])
    if (!is.null(ev) && sum(!is.na(ev)) > 2 * min_arm) {
      tab <- table(ev)
      if (length(tab) == 2 && min(tab) >= min_arm) return(c)
    }
  }
  NULL
}
glm_table <- function(fit) {
  co <- as.data.frame(summary(fit)$coefficients)
  data.frame(term = rownames(co), estimate = round(co[, 1], 4), std_error = round(co[, 2], 4),
             odds_ratio = round(exp(co[, 1]), 4), lower95 = round(exp(co[, 1] - 1.96 * co[, 2]), 4),
             upper95 = round(exp(co[, 1] + 1.96 * co[, 2]), 4), p_value = signif(co[, 4], 3),
             stringsAsFactors = FALSE)
}

sources_used <- data.frame(cohort = character(0), source = character(0), file = character(0), rows = integer(0), columns = integer(0))
# One value or NA: a statistic of an odd column must never change the row count.
scalar <- function(v) if (length(v) == 1) as.numeric(v) else NA_real_
analyse_cohort <- function(inp) {
  label <- inp$label
  src <- pick_source(inp)
  if (is.null(src)) {
    note(label, ": neither the full data file (", inp$path, ") nor a shuffled sample (",
         if (nzchar(inp$fallback)) inp$fallback else "none in this room", ") is available yet - skipped")
    return(invisible())
  }
  cat("\n==== ", label, " - ", src$kind, " (", src$path, ") ====\n", sep = "")
  df <- tryCatch(read_table(src$path), error = function(e) { note(label, ": could not read ", src$path, " - ", conditionMessage(e)); NULL })
  if (is.null(df)) return(invisible())
  # Column names as R expects them, unique and syntactic: a header with an
  # empty or repeated name gives NULL when the column is taken by name (the
  # run of 2026-11-09 halted on exactly that in the summary statistics).
  fixed <- make.names(names(df), unique = TRUE)
  if (any(fixed != names(df))) {
    note(label, ": ", sum(fixed != names(df)), " column name(s) made unique and syntactic, e.g. ",
         paste(head(sprintf("'%s' -> %s", names(df)[fixed != names(df)], fixed[fixed != names(df)]), 3), collapse = ", "))
    names(df) <- fixed
  }
  note(label, ": ", src$kind, ", ", nrow(df), " rows x ", ncol(df), " columns",
       if (src$kind == "shuffled sample") " (values shuffled per column: every model below is a smoke test)" else "")
  sources_used <<- rbind(sources_used, data.frame(cohort = label, source = src$kind, file = src$path, rows = nrow(df), columns = ncol(df)))

  # ---- preparation (base R, image packages): what the exercises work on ----
  complete <- vapply(df, function(x) sum(!is.na(x)), numeric(1))
  num_cols <- names(df)[vapply(df, is.numeric, logical(1))]
  num_cols <- num_cols[vapply(num_cols, function(c) sd(df[[c]], na.rm = TRUE) > 0, logical(1))]
  num_cols <- num_cols[order(-complete[num_cols])]
  event_col <- find_event_col(df)
  time_col <- if (is.null(event_col)) NULL else find_time_col(df, exclude = event_col)
  treat_col <- find_binary_col(df, exclude = c(event_col, time_col))
  covs <- head(setdiff(num_cols, c(event_col, time_col, treat_col)), 4)
  # md: complete cases of the columns the models use (event, time, treatment, covariates)
  used <- c(covs, event_col, time_col, treat_col)
  md <- df[, used, drop = FALSE]
  if (!is.null(event_col)) md$event <- as_event(df[[event_col]])
  if (!is.null(treat_col)) md$treat <- as_event(df[[treat_col]])
  md_rows <- which(complete.cases(md))   # the rows of df the models use
  md <- md[md_rows, , drop = FALSE]
  has_event <- !is.null(event_col) && nrow(md) >= 30 && length(covs) >= 1 && length(unique(md$event)) == 2
  has_time <- has_event && !is.null(time_col) && all(md[[time_col]] >= 0)
  has_treat <- !is.null(treat_col) && nrow(md) >= 30 && length(covs) >= 1 && length(unique(md$treat)) == 2
  cat("prepared: ", length(num_cols), " varying numeric columns; event: ", if (is.null(event_col)) "-" else event_col,
      "; time: ", if (is.null(time_col)) "-" else time_col, "; treatment: ", if (is.null(treat_col)) "-" else treat_col,
      "; covariates: ", paste(covs, collapse = ", "), "; complete rows: ", nrow(md), "\n", sep = "")
  if (has_event) {
    fit_glm <- glm(reformulate(covs, "event"), data = md, family = binomial())
    or_tab <- glm_table(fit_glm)
  }

  # summary statistics of every numeric column (base R) - the table several exercises reuse
  stats <- NULL
  if (length(num_cols) > 0) {
    stats <- do.call(rbind, lapply(num_cols, function(c) {
      x <- df[[c]]
      data.frame(column = c, n = sum(!is.na(x)), missing = sum(is.na(x)),
                 mean = scalar(mean(x, na.rm = TRUE)), sd = scalar(sd(x, na.rm = TRUE)),
                 min = scalar(suppressWarnings(min(x, na.rm = TRUE))), q25 = scalar(quantile(x, 0.25, na.rm = TRUE, names = FALSE)),
                 median = scalar(median(x, na.rm = TRUE)), q75 = scalar(quantile(x, 0.75, na.rm = TRUE, names = FALSE)),
                 max = scalar(suppressWarnings(max(x, na.rm = TRUE))), stringsAsFactors = FALSE)
    }))
    stats[, -1] <- lapply(stats[, -1], function(v) round(v, 4))
    write.csv(stats, file.path(OUT, paste0("summary_stats_", label, ".csv")), row.names = FALSE)
    note("wrote summary_stats_", label, ".csv (", length(num_cols), " numeric columns)")
  }
  small <- df[, head(num_cols, 6), drop = FALSE]   # a few complete numeric columns for the table/imputation exercises

  cat("-- exercises --\n")
  # writexl: the summary statistics as an Excel workbook
  run_exercise(label, "writexl", "write the summary statistics to .xlsx", !is.null(stats), "no numeric column", function() {
    f <- paste0("summary_stats_", label, ".xlsx")
    writexl::write_xlsx(list(summary_stats = stats, data_source = sources_used[sources_used$cohort == label, ]), file.path(OUT, f))
    list(outputs = f, detail = paste(nrow(stats), "rows"))
  })
  # labelled: variable labels on the columns, then a data dictionary with look_for()
  run_exercise(label, "labelled", "set variable labels and build a dictionary with look_for()", ncol(small) > 0, "no numeric column", function() {
    lab <- small
    labels <- setNames(as.list(gsub("[_.]", " ", names(lab))), names(lab))
    lab <- labelled::set_variable_labels(lab, .labels = labels)
    dict <- as.data.frame(labelled::look_for(lab, details = FALSE))
    f <- out_file("labelled", label, "csv")
    write.csv(dict[, intersect(c("pos", "variable", "label", "col_type"), names(dict))], file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(nrow(dict), "variables labelled"))
  })
  # gtsummary: a descriptive table of a few columns (by treatment arm when there is one)
  run_exercise(label, "gtsummary", "tbl_summary() of the most complete numeric columns", ncol(small) > 0, "no numeric column", function() {
    dat <- small
    if (has_treat) { dat <- small[md_rows, , drop = FALSE]; dat$treat <- factor(md$treat) }
    tbl <- if (has_treat) gtsummary::tbl_summary(dat, by = "treat") else gtsummary::tbl_summary(dat)
    out <- as.data.frame(gtsummary::as_tibble(tbl))
    f <- out_file("gtsummary", label, "csv")
    write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(nrow(out), "table rows"))
  })
  # corrr: correlations of numeric columns, as a table and a plot
  run_exercise(label, "corrr", "correlate() the numeric columns, rplot()", length(num_cols) >= 2, "fewer than two varying numeric columns", function() {
    cols <- head(num_cols, 12)
    cm <- corrr::correlate(df[, cols], use = "pairwise.complete.obs", quiet = TRUE)
    f <- out_file("corrr", label, "csv")
    write.csv(as.data.frame(cm), file.path(OUT, f), row.names = FALSE)
    outs <- f
    if (avail("ggplot2")) {
      p <- corrr::rplot(corrr::shave(cm), print_cor = TRUE) + ggplot2::labs(title = paste("Correlations -", label))
      g <- out_file("corrr", label, "png"); ggplot2::ggsave(file.path(OUT, g), p, width = 9, height = 7, dpi = 120, bg = "white")
      outs <- c(outs, g)
    }
    list(outputs = outs, detail = paste(length(cols), "columns"))
  })
  # MuMIn: all-subsets model selection of the event model
  run_exercise(label, "MuMIn", "dredge() the logistic model of the event", has_event, "no event column with enough complete rows", function() {
    fit <- glm(reformulate(covs, "event"), data = md, family = binomial(), na.action = na.fail)
    dd <- MuMIn::dredge(fit, trace = FALSE)
    out <- head(as.data.frame(dd), 12)
    f <- out_file("MuMIn", label, "csv")
    write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(nrow(dd), "candidate models, best AICc", round(min(dd$AICc), 1)))
  })
  # merTools: prediction intervals of a mixed model with a random intercept
  run_exercise(label, "merTools", "predictInterval() of a random-intercept lmer model", avail("lme4") && length(covs) >= 2 && nrow(md) >= 40,
               "needs lme4 (image) and two covariates", function() {
    dat <- md[, covs[1:2]]; names(dat) <- c("y", "x")
    grp <- find_group_col(df)
    dat$g <- if (!is.null(grp)) factor(df[[grp]][md_rows]) else cut(dat$x, breaks = 4, labels = FALSE)
    dat <- dat[complete.cases(dat), ]
    fit <- lme4::lmer(y ~ x + (1 | g), data = dat)
    pi <- merTools::predictInterval(fit, newdata = head(dat, 50), n.sims = 200, level = 0.9, stat = "median")
    out <- data.frame(statistic = c("median fit", "median lower 90%", "median upper 90%", "n rows predicted"),
                      value = c(median(pi$fit), median(pi$lwr), median(pi$upr), nrow(pi)))
    f <- out_file("merTools", label, "csv")
    write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste("random intercept on", if (!is.null(grp)) grp else "4 bins of x"))
  })
  # DHARMa: residual diagnostics of the event model
  run_exercise(label, "DHARMa", "simulateResiduals() + testDispersion() on the event model", has_event, "no event column with enough complete rows", function() {
    sim <- DHARMa::simulateResiduals(fit_glm, n = 100, plot = FALSE)
    disp <- DHARMa::testDispersion(sim, plot = FALSE)
    unif <- DHARMa::testUniformity(sim, plot = FALSE)
    out <- data.frame(test = c("dispersion", "uniformity (KS)"), statistic = c(disp$statistic, unif$statistic), p_value = c(disp$p.value, unif$p.value))
    f <- out_file("DHARMa", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    g <- save_png(out_file("DHARMa", label, "png"), plot(sim), width = 1100, height = 550)
    list(outputs = c(f, g), detail = sprintf("dispersion p=%.3f", disp$p.value))
  })
  # forestplot / forestploter: the odds ratios of the event model as forest plots
  run_exercise(label, "forestplot", "forestplot() of the event model's odds ratios", has_event, "no event column with enough complete rows", function() {
    t <- or_tab[or_tab$term != "(Intercept)", ]
    g <- save_png(out_file("forestplot", label, "png"), {
      print(forestplot::forestplot(labeltext = cbind(c("term", t$term), c("OR (95% CI)", sprintf("%.2f (%.2f-%.2f)", t$odds_ratio, t$lower95, t$upper95))),
                                   mean = c(NA, t$odds_ratio), lower = c(NA, t$lower95), upper = c(NA, t$upper95),
                                   is.summary = c(TRUE, rep(FALSE, nrow(t))), zero = 1, xlog = TRUE,
                                   title = paste0("Odds ratios of ", event_col, " - ", label)))
    }, width = 900, height = 450)
    list(outputs = g, detail = paste(nrow(t), "terms"))
  })
  run_exercise(label, "forestploter", "forest() of the same odds ratios", has_event, "no event column with enough complete rows", function() {
    t <- or_tab[or_tab$term != "(Intercept)", ]
    dt <- data.frame(Term = t$term, ` ` = paste(rep(" ", 24), collapse = ""), `OR (95% CI)` = sprintf("%.2f (%.2f-%.2f)", t$odds_ratio, t$lower95, t$upper95),
                     check.names = FALSE, stringsAsFactors = FALSE)
    p <- forestploter::forest(dt, est = t$odds_ratio, lower = t$lower95, upper = t$upper95, ci_column = 2, ref_line = 1, x_trans = "log",
                              title = paste0("Odds ratios of ", event_col, " - ", label))
    g <- save_png(out_file("forestploter", label, "png"), plot(p), width = 900, height = 450)
    list(outputs = g, detail = paste(nrow(t), "terms"))
  })
  # timeROC: time-dependent ROC of the first covariate as a marker for the event
  run_exercise(label, "timeROC", "timeROC() of the first covariate as marker, AUC at three horizons", has_time, "needs an event and a follow-up time", function() {
    tt <- unname(quantile(md[[time_col]][md[[time_col]] > 0], c(0.25, 0.5, 0.75)))
    roc <- timeROC::timeROC(T = md[[time_col]], delta = md$event, marker = md[[covs[1]]], cause = 1, times = tt, iid = FALSE)
    out <- data.frame(time = roc$times, AUC = round(roc$AUC, 4), cases = roc$Stats[, 1], survivors = roc$Stats[, 2])
    f <- out_file("timeROC", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    g <- save_png(out_file("timeROC", label, "png"), plot(roc, time = tt[2], title = FALSE), width = 650, height = 650)
    list(outputs = c(f, g), detail = sprintf("marker %s, AUC at median time %.3f", covs[1], roc$AUC[2]))
  })
  # RMediation: a mediation effect (a x b) with its distribution-of-the-product CI
  run_exercise(label, "RMediation", "medci() for covariate 1 -> covariate 2 -> event", has_event && length(covs) >= 2, "needs an event and two covariates", function() {
    a_fit <- lm(reformulate(covs[1], covs[2]), data = md)
    b_fit <- glm(reformulate(covs[1:2], "event"), data = md, family = binomial())
    a <- coef(summary(a_fit))[covs[1], ]; b <- coef(summary(b_fit))[covs[2], ]
    ci <- RMediation::medci(mu.x = a[1], mu.y = b[1], se.x = a[2], se.y = b[2], type = "dop")
    out <- data.frame(path = c("a (x -> m)", "b (m -> y | x)", "a x b"), estimate = c(a[1], b[1], ci$Estimate),
                      lower95 = c(NA, NA, ci$`95% CI`[1]), upper95 = c(NA, NA, ci$`95% CI`[2]))
    f <- out_file("RMediation", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = sprintf("x=%s, m=%s", covs[1], covs[2]))
  })
  # sysfonts / showtext: the fonts the image has, and a plot rendered through showtext
  run_exercise(label, "sysfonts", "font_files() and font_families() of the image", TRUE, "", function() {
    ff <- sysfonts::font_files()
    fam <- sysfonts::font_families()
    out <- data.frame(what = c("font files", "families registered"), count = c(nrow(ff), length(fam)),
                      examples = c(paste(head(unique(ff$family), 8), collapse = "; "), paste(head(fam, 8), collapse = "; ")))
    f <- out_file("sysfonts", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(nrow(ff), "font files on the image"))
  })
  run_exercise(label, "showtext", "showtext_auto() and a plot with text", TRUE, "", function() {
    showtext::showtext_auto()
    on.exit(showtext::showtext_auto(FALSE), add = TRUE)
    g <- save_png(out_file("showtext", label, "png"), {
      plot.new(); text(0.5, 0.6, paste("showtext renders this text -", label), cex = 2)
      text(0.5, 0.4, paste("families:", paste(head(sysfonts::font_families(), 4), collapse = ", ")), cex = 1.2)
    }, width = 800, height = 400)
    list(outputs = g, detail = "rendered")
  })
  # LTRCtrees / LTRCforests: survival trees and forests for left-truncated right-censored data
  # (no truncation time in these files: entry at 0, so they run as ordinary survival trees)
  run_exercise(label, "LTRCtrees", "LTRCART() survival tree of time-to-event on the covariates", has_time, "needs an event and a follow-up time", function() {
    dat <- md; dat$start <- 0; dat$stop <- pmax(dat[[time_col]], 1e-6)
    fit <- LTRCtrees::LTRCART(reformulate(covs, "survival::Surv(start, stop, event)"), data = dat)
    g <- save_png(out_file("LTRCtrees", label, "png"), { plot(fit); text(fit, cex = 0.8) }, width = 1000, height = 650)
    list(outputs = g, detail = paste(nrow(fit$frame), "nodes"))
  })
  run_exercise(label, "LTRCforests", "ltrccif() conditional inference forest (50 trees)", has_time, "needs an event and a follow-up time", function() {
    dat <- md; dat$start <- 0; dat$stop <- pmax(dat[[time_col]], 1e-6); dat$id <- seq_len(nrow(dat))
    fit <- LTRCforests::ltrccif(reformulate(covs, "survival::Surv(start, stop, event)"), data = dat, id = id, ntree = 50)
    list(outputs = character(0), detail = paste("forest of", fit$ntree, "trees fitted"))
  })
  run_exercise(label, "ksamplesLTRC", "load and list the exported functions", TRUE, "", function() {
    ex <- sort(getNamespaceExports("ksamplesLTRC"))
    f <- out_file("ksamplesLTRC", label, "csv"); write.csv(data.frame(exported_function = ex), file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(length(ex), "exports:", paste(head(ex, 6), collapse = ", ")))
  })
  # miceRanger / Amelia: imputation of the most complete numeric columns
  run_exercise(label, "miceRanger", "miceRanger() one imputation, two iterations", ncol(small) >= 2 && nrow(small) >= 30, "needs two numeric columns", function() {
    dat <- small[rowSums(!is.na(small)) >= 1, ]
    imp <- miceRanger::miceRanger(dat, m = 1, maxiter = 2, verbose = FALSE)
    comp <- miceRanger::completeData(imp)[[1]]
    out <- data.frame(column = names(dat), missing_before = colSums(is.na(dat)), missing_after = colSums(is.na(as.data.frame(comp))))
    f <- out_file("miceRanger", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(sum(out$missing_before), "cells imputed"))
  })
  run_exercise(label, "Amelia", "amelia() one imputation of the numeric columns", ncol(small) >= 2 && nrow(small) >= 30, "needs two numeric columns", function() {
    dat <- small[rowSums(!is.na(small)) >= 1, ]
    am <- Amelia::amelia(dat, m = 1, p2s = 0)
    comp <- am$imputations[[1]]
    out <- data.frame(column = names(dat), missing_before = colSums(is.na(dat)), missing_after = colSums(is.na(comp)))
    f <- out_file("Amelia", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste(sum(out$missing_before), "cells imputed"))
  })
  # grf: a causal forest when a binary treatment exists, a regression forest otherwise
  run_exercise(label, "grf", if (has_treat) "causal_forest() and its average treatment effect" else "regression_forest() and variable importance",
               (has_treat || has_event) && length(covs) >= 2, "needs two covariates and an event or a treatment", function() {
    X <- as.matrix(md[, covs])
    if (has_treat) {
      Y <- if (has_event) md$event else md[[covs[1]]]
      cf <- grf::causal_forest(X, Y, md$treat, num.trees = 200)
      ate <- grf::average_treatment_effect(cf)
      out <- data.frame(statistic = c("ATE", "std error", "n"), value = c(ate[["estimate"]], ate[["std.err"]], nrow(X)))
      detail <- sprintf("ATE %.3f (se %.3f) of %s on %s", ate[["estimate"]], ate[["std.err"]], treat_col, if (has_event) event_col else covs[1])
    } else {
      rf <- grf::regression_forest(X, md$event, num.trees = 200)
      vi <- grf::variable_importance(rf)
      out <- data.frame(covariate = covs, importance = round(as.numeric(vi), 4))
      detail <- paste("most important:", covs[which.max(vi)])
    }
    f <- out_file("grf", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = detail)
  })
  # caret: 3-fold cross-validated logistic model of the event
  run_exercise(label, "caret", "train() a glm with 3-fold cross-validation", has_event && nrow(md) >= 60, "needs an event and 60 complete rows", function() {
    dat <- md[, covs]; dat$event <- factor(ifelse(md$event == 1, "yes", "no"))
    ctrl <- caret::trainControl(method = "cv", number = 3)
    fit <- caret::train(event ~ ., data = dat, method = "glm", family = binomial(), trControl = ctrl)
    out <- fit$results
    f <- out_file("caret", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = sprintf("accuracy %.3f, kappa %.3f", out$Accuracy[1], out$Kappa[1]))
  })
  # isotone: isotonic regression of covariate 2 on covariate 1 (fitted means per decile)
  run_exercise(label, "isotone", "gpava() isotonic regression of covariate 2 on covariate 1", length(covs) >= 2 && nrow(md) >= 30, "needs two covariates", function() {
    o <- order(md[[covs[1]]])
    fit <- isotone::gpava(z = md[[covs[1]]][o], y = md[[covs[2]]][o])
    dec <- cut(seq_along(fit$x), breaks = 10, labels = FALSE)
    out <- data.frame(decile_of_x = 1:10, mean_x = tapply(fit$z, dec, mean), fitted_mean_y = tapply(fit$x, dec, mean))
    f <- out_file("isotone", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = sprintf("%d distinct fitted levels", length(unique(fit$x))))
  })
  # CBPS / WeightIt: propensity-score weights for the treatment, and the balance they give
  run_exercise(label, "CBPS", "CBPS() covariate-balancing propensity score, balance table", has_treat, "no binary treatment column", function() {
    dat <- md[, covs]; dat$treat <- md$treat
    fit <- CBPS::CBPS(reformulate(covs, "treat"), data = dat, ATT = 0, standardize = TRUE)
    bal <- CBPS::balance(fit)
    out <- data.frame(covariate = rownames(bal$balanced), balanced = round(bal$balanced[, 1], 4), original = round(bal$original[, 1], 4))
    f <- out_file("CBPS", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste("treatment", treat_col))
  })
  run_exercise(label, "WeightIt", "weightit() inverse-probability weights (glm), effective sample sizes", has_treat, "no binary treatment column", function() {
    dat <- md[, covs]; dat$treat <- md$treat
    w <- WeightIt::weightit(reformulate(covs, "treat"), data = dat, method = "glm", estimand = "ATE")
    sw <- summary(w)
    ess <- as.data.frame(sw$effective.sample.size)
    out <- data.frame(statistic = c("ESS control (unweighted)", "ESS treated (unweighted)", "ESS control (weighted)", "ESS treated (weighted)"),
                      value = c(ess[1, 1], ess[1, 2], ess[2, 1], ess[2, 2]))
    f <- out_file("WeightIt", label, "csv"); write.csv(out, file.path(OUT, f), row.names = FALSE)
    list(outputs = f, detail = paste("treatment", treat_col, "- weights range", paste(round(range(w$weights), 2), collapse = " to ")))
  })
}
# One cohort's failure (an unexpected column, a model that cannot be fit) is a
# note, never the end of the run: the other cohorts and the files below follow.
for (inp in INPUTS) tryCatch(analyse_cohort(inp), error = function(e)
  note(inp$label, ": analysis stopped - ", conditionMessage(e), " (the other cohorts and the result files are unaffected)"))

write.csv(exercises, file.path(OUT, "package_exercises.csv"), row.names = FALSE)
cat("\nExercises per package and cohort (package_exercises.csv):\n")
if (nrow(exercises) > 0) {
  shown <- exercises; shown$detail <- ifelse(nchar(shown$detail) > 60, paste0(substr(shown$detail, 1, 60), "..."), shown$detail)
  print(shown[, c("cohort", "package", "status", "seconds", "detail")], row.names = FALSE)
}
# The status table gets an "exercise" column: what happened when the package was used.
status$exercise <- vapply(status$package, function(p) {
  e <- exercises[exercises$package == p, ]
  if (nrow(e) == 0) return(if (status$status[status$package == p] == "FAILED") "not installed" else "no exercise")
  n_ok <- sum(e$status == "ok"); n_fail <- sum(e$status == "failed"); n_skip <- sum(e$status == "skipped")
  paste(c(if (n_ok) paste("ok on", n_ok), if (n_fail) paste("failed on", n_fail), if (n_skip) paste("skipped on", n_skip)), collapse = ", ")
}, character(1))
write.csv(status, file.path(OUT, "r_packages_status.csv"), row.names = FALSE)
note("Exercises: ", sum(exercises$status == "ok"), " ok, ", sum(exercises$status == "failed"), " failed, ",
     sum(exercises$status == "skipped"), " skipped, over ", length(unique(exercises$cohort)), " cohort(s)")
failed_ex <- exercises[exercises$status == "failed", ]
if (nrow(failed_ex) > 0) for (i in seq_len(nrow(failed_ex))) note("  failed: ", failed_ex$package[i], " on ", failed_ex$cohort[i], " - ", failed_ex$detail[i])

write.csv(sources_used, file.path(OUT, "data_sources_used.csv"), row.names = FALSE)
writeLines(c("load-R-packages - run summary", paste0("R ", R.version.string), "",
             "Data source per cohort (full data, or the shuffled sample while the full data is not uploaded):",
             if (nrow(sources_used) > 0) paste0("  ", sources_used$cohort, ": ", sources_used$source) else "  (none available)",
             "", notes),
           file.path(OUT, "README.txt"))
cat("\nDone. Files in", OUT, ":\n")
print(list.files(OUT))
'''


# ---------------------------------------------------------------------------
# Optional Python probe node ("probe-R-in-python")
# ---------------------------------------------------------------------------
# Answers, in one run, whether R can be driven from a Python compute node with
# rpy2 - and whether that route would install the R packages the R node cannot.
# rpy2 is a bridge, not an R distribution: it links against an R that must
# already be in the Python image, and it brings no compiler. So the probe
# reports the image's distribution, whether R, Rscript and a C/C++/Fortran
# toolchain are on the path, the R version, whether PyPI is reachable, whether
# `pip install rpy2` succeeds, and, when R is there, whether one pure-R and
# one compiled CRAN package install - from Posit's Linux binaries when the
# distribution is one they serve. Reads no data; writes a report only.

RPY2_PROBE_NODE_NAME = "probe-R-in-python"
RPY2_PROBE_DOMAINS = ["pypi.org", "files.pythonhosted.org"] + R_CRAN_DOMAINS

_RPY2_PROBE_SCRIPT = r'''###############################################################################
# probe-R-in-python: can R be used from this Python compute node (rpy2), and
# would that route install the R packages the R node cannot?
#
# Reads no data. Writes to /output:
#   r_in_python_probe.txt   the report, step by step, with a verdict at the end
#   r_in_python_probe.json  the same findings as data
#   accessed_domains.txt    (written by the platform) the hosts contacted
# Internet access is limited to PyPI and the two R package repositories.
###############################################################################
import json, os, platform, shutil, subprocess, sys, time, ssl, urllib.request

OUT = "/output"
os.makedirs(OUT, exist_ok=True)
CA_BUNDLE = "/input/code/cacert.pem"   # Mozilla's bundle, attached to the node
report, findings = [], {}


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    report.append(line)


def run(cmd, timeout=600, env=None):
    """(returncode, last lines of output) of a command; -1 when it cannot start."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out = (r.stdout + "\n" + r.stderr).strip()
        return r.returncode, out
    except FileNotFoundError:
        return -1, "not found"
    except subprocess.TimeoutExpired:
        return -2, "timed out after %ss" % timeout
    except Exception as e:
        return -3, "%s: %s" % (type(e).__name__, e)


def tail(text, n=12):
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n      ".join(lines[-n:]) if lines else "(no output)"


# ---- 1. the image ---------------------------------------------------------
say("==== 1. The image")
say("Python", platform.python_version(), "at", sys.executable)
say("Platform:", platform.platform())
distro = {}
try:
    with open("/etc/os-release") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.rstrip().split("=", 1)
                distro[k] = v.strip('"')
    say("Distribution:", distro.get("PRETTY_NAME") or distro.get("ID"), "(", distro.get("ID"), distro.get("VERSION_CODENAME") or distro.get("VERSION_ID"), ")")
except OSError:
    say("Distribution: no /etc/os-release")
findings["python"] = platform.python_version()
findings["distro"] = distro
nix = os.path.isdir("/nix/store")
say("Nix store present:", nix)
findings["nix"] = nix

# ---- 2. toolchain and R on the path ---------------------------------------
say("\n==== 2. Toolchain and R")
tools = {}
for t in ("R", "Rscript", "gcc", "cc", "g++", "gfortran", "make", "pkg-config", "conda", "mamba", "micromamba"):
    tools[t] = shutil.which(t)
    say("  %-11s %s" % (t, tools[t] or "-"))
findings["tools"] = tools
r_present = bool(tools["R"] or tools["Rscript"])
if r_present:
    rc, out = run([tools["R"] or tools["Rscript"], "--version"], timeout=60)
    say("R --version:", tail(out, 2))
    findings["r_version"] = out.splitlines()[0] if out else None
    rc, out = run([tools["R"] or "R", "RHOME"], timeout=60)
    say("R home:", out.strip())
    findings["r_home"] = out.strip()
    rc, out = run([tools["R"] or "R", "CMD", "config", "CC"], timeout=60)
    say("R CMD config CC:", out.strip())
    findings["r_cc"] = out.strip()
    # headers: rpy2 needs them to build
    inc = os.path.join(findings.get("r_home") or "", "include", "R.h")
    say("R headers (R.h):", "present" if os.path.exists(inc) else "missing at %s" % inc)
    findings["r_headers"] = os.path.exists(inc)
else:
    say("R is not on the path: rpy2 cannot be built or run in this image.")
compiler = bool(tools["gcc"] or tools["cc"])
findings["compiler"] = compiler

# ---- 3. network: PyPI and the R repositories over TLS -----------------------
say("\n==== 3. Network")
ctx = ssl.create_default_context(cafile=CA_BUNDLE) if os.path.exists(CA_BUNDLE) else ssl.create_default_context()
net = {}
for name, url in (("pypi", "https://pypi.org/simple/rpy2/"),
                  ("cran", "https://cloud.r-project.org/src/contrib/PACKAGES"),
                  ("posit", "https://packagemanager.posit.co/cran/latest/src/contrib/PACKAGES")):
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cohort-explorer-probe"}), timeout=60, context=ctx) as resp:
            body = resp.read(200000)
            net[name] = "HTTP %s, %d bytes, %.1fs" % (resp.status, len(body), time.time() - t0)
    except Exception as e:
        net[name] = "FAILED: %s" % (str(e)[:160])
    say("  %-6s %s" % (name, net[name]))
findings["network"] = net

# ---- 4. pip and rpy2 ------------------------------------------------------
say("\n==== 4. pip and rpy2")
env = dict(os.environ)
if os.path.exists(CA_BUNDLE):
    env["PIP_CERT"] = CA_BUNDLE
    env["SSL_CERT_FILE"] = CA_BUNDLE
rc, out = run([sys.executable, "-m", "pip", "--version"], timeout=60, env=env)
say("pip:", out.strip() if rc == 0 else "not available (%s)" % tail(out, 2))
findings["pip"] = rc == 0
rpy2 = {"download": None, "install": None, "import": None}
if rc == 0:
    rc, out = run([sys.executable, "-m", "pip", "download", "--no-deps", "-d", "/tmp/rpy2dl", "rpy2"], timeout=300, env=env)
    rpy2["download"] = "ok" if rc == 0 else "FAILED: " + tail(out, 4)
    say("pip download rpy2:", rpy2["download"])
    if rc == 0:
        files = os.listdir("/tmp/rpy2dl") if os.path.isdir("/tmp/rpy2dl") else []
        say("  got:", ", ".join(files))
        findings["rpy2_artifact"] = files
    rc, out = run([sys.executable, "-m", "pip", "install", "--target", "/tmp/rpy2lib", "rpy2"], timeout=900, env=env)
    rpy2["install"] = "ok" if rc == 0 else "FAILED: " + tail(out, 8)
    say("pip install rpy2:", rpy2["install"])
    if rc == 0:
        sys.path.insert(0, "/tmp/rpy2lib")
        try:
            import rpy2.robjects as ro
            rpy2["import"] = "ok: " + str(ro.r("R.version.string")[0])
        except Exception as e:
            rpy2["import"] = "FAILED: %s: %s" % (type(e).__name__, str(e)[:300])
        say("import rpy2.robjects:", rpy2["import"])
findings["rpy2"] = rpy2

# ---- 5. R packages through this image's R -----------------------------------
say("\n==== 5. R packages (only when R is present)")
pkgs = {}
if r_present:
    did = (distro.get("ID") or "").lower()
    code = (distro.get("VERSION_CODENAME") or "").lower()
    binary = None
    if did in ("ubuntu", "debian") and code:
        binary = "https://packagemanager.posit.co/cran/__linux__/%s/latest" % code
    repos = ([binary] if binary else []) + ["https://cloud.r-project.org"]
    say("Repositories:", ", ".join(repos), "(Posit Linux binaries need no compiler)" if binary else "(sources only: compiled packages need a toolchain)")
    findings["r_repos"] = repos
    rscript = tools["Rscript"] or tools["R"]
    for pkg, kind in (("isotone", "pure R, needs the compiled nnls"), ("nnls", "compiled C"), ("corrplot", "pure R, no dependencies")):
        r_code = (
            "options(repos=c(%s), timeout=600, warn=1);" % ",".join('"%s"' % r for r in repos)
            + ('Sys.setenv(CURL_CA_BUNDLE="%s", SSL_CERT_FILE="%s");' % (CA_BUNDLE, CA_BUNDLE) if os.path.exists(CA_BUNDLE) else "")
            + "lib <- file.path(tempdir(), 'lib'); dir.create(lib, showWarnings=FALSE);"
            + "install.packages('%s', lib=lib, quiet=TRUE);" % pkg
            + "cat(if (nzchar(system.file(package='%s', lib.loc=lib))) 'INSTALLED' else 'NOT INSTALLED', '\\n')" % pkg
        )
        rc, out = run([rscript, "-e", r_code], timeout=900, env=env)
        ok = "INSTALLED" in out and "NOT INSTALLED" not in out
        pkgs[pkg] = {"kind": kind, "installed": ok, "log": tail(out, 6)}
        say("  %-9s (%s): %s" % (pkg, kind, "installed" if ok else "FAILED\n      " + tail(out, 6)))
else:
    say("skipped")
findings["r_packages"] = pkgs

# ---- 6. verdict -----------------------------------------------------------
say("\n==== 6. Verdict")
if not r_present:
    verdict = ("This Python image has no R, so rpy2 cannot be installed or run here. A Python node cannot be the way to "
               "install R packages unless Decentriq provides a Python image with R (and a compiler, or a Debian/Ubuntu "
               "base so Posit's binary packages work).")
elif rpy2["install"] != "ok":
    verdict = ("R is present but rpy2 did not install (see step 4): without a compiler and R headers the bridge cannot be built.")
elif pkgs.get("nnls", {}).get("installed"):
    verdict = ("rpy2 works and a compiled CRAN package installed: this image CAN install the packages the R node cannot. "
               "Driving R from Python with rpy2 is a viable route.")
elif pkgs.get("corrplot", {}).get("installed"):
    verdict = ("rpy2 works and pure-R packages install, but a compiled package did not: the same limit as the R node. "
               "A Python node changes nothing for the 12 packages that need a compiler.")
else:
    verdict = "rpy2 works but no package installed from the repositories (see step 5): check the network and the repositories."
say(verdict)
findings["verdict"] = verdict
findings["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(os.path.join(OUT, "r_in_python_probe.txt"), "w") as fh:
    fh.write("\n".join(report) + "\n")
with open(os.path.join(OUT, "r_in_python_probe.json"), "w") as fh:
    json.dump(findings, fh, indent=2, default=str)
print("Done. Files in", OUT, ":", sorted(os.listdir(OUT)))
'''


def rpy2_probe_script() -> str:
    """The Python script of the "probe-R-in-python" node (no inputs)."""
    return _RPY2_PROBE_SCRIPT
