"""Analysis DCR embedded scripts for data fragmentation, visualization, and exploration."""


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
                raise ValueError(f"Could not read file as CSV, SPSS, or ZIP. CSV error: {{csv_error}}, SPSS error: {{spss_error}}, ZIP error: {{zip_error}}")

# Helper function to load metadata dictionary (CSV only)
def load_metadata(file_path):
    return pd.read_csv(file_path)

# Read the cohort data (RawDataNodeDefinition - use file path)
df = load_data("/input/{cohort_id}")

with open(log_file, "a") as log:
    log.write(f"Loaded cohort data with {{len(df)}} rows and {{len(df.columns)}} columns\\n")

# Read the metadata dictionary (CSV only)
try:
    metadata_df = load_metadata("/input/{cohort_id}_metadata_dictionary")
    with open(log_file, "a") as log:
        log.write(f"Loaded metadata dictionary with {{len(metadata_df)}} variables\\n")
except Exception as e:
    metadata_df = None
    with open(log_file, "a") as log:
        log.write(f"Could not load metadata dictionary: {{e}}\\n")

# ID column name is passed from the room creation code (discovered using OMOP ID 4086934)
# Replace ID column with synthetic IDs
# IMPORTANT: Rows with the same original ID must get the same synthetic ID
id_column = "{id_variable_name if id_variable_name else ''}"
with open(log_file, "a") as log:
    if id_column and id_column in df.columns:
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
        
        log.write(f"Replaced ID column '{{id_column}}' with synthetic IDs at position {{id_col_position}}\\n")
        log.write(f"Mapped {{len(unique_ids)}} unique original IDs to synthetic IDs\\n")
    else:
        # No ID column found - add synthetic IDs at the beginning (row-based, no grouping)
        synthetic_ids = ['AIRLOCK_' + str(i).zfill(6) for i in range(1, len(df) + 1)]
        df.insert(0, 'Synthetic_ID', synthetic_ids)
        log.write(f"ID column not found or not specified, added row-based synthetic IDs at position 0\\n")

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
        # Normalize column names: lowercase, strip, collapse multiple spaces
        def normalize_col(col):
            return ' '.join(col.lower().strip().split())
        
        # Build normalized mapping from data columns to actual column names
        df_cols_normalized = {{normalize_col(col): col for col in df_fragment.columns}}
        
        for _, row in metadata_df.iterrows():
            var_name = str(row[varname_col]).strip()
            var_type = str(row[vartype_col]).strip().upper() if pd.notna(row[vartype_col]) else ''
            categorical = str(row[categorical_col]).strip() if categorical_col and pd.notna(row[categorical_col]) else ''
            
            # Check if numeric: VARTYPE in ['FLOAT', 'INT'] and CATEGORICAL is empty
            if var_type in ['FLOAT', 'INT'] and (categorical == '' or categorical.lower() == 'nan'):
                # Normalized column matching (case-insensitive, space-normalized)
                var_name_normalized = normalize_col(var_name)
                if var_name_normalized in df_cols_normalized:
                    # Use the actual column name from the data
                    actual_col_name = df_cols_normalized[var_name_normalized]
                    numeric_vars.append(actual_col_name)

with open(log_file, "a") as log:
    log.write(f"\\nIdentified {{len(numeric_vars)}} numeric variables for outlier capping\\n")

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
            log.write(f"Error processing variable {{var}}: {{e}}\\n")

# Write outlier capping summary to log
with open(log_file, "a") as log:
    log.write(f"\\n=== Outlier Capping Summary (Z-score threshold: {{Z_THRESHOLD}}) ===\\n")
    total_vars_capped = 0
    total_values_capped = 0
    for stat in outlier_stats:
        if stat['total_capped'] > 0:
            total_vars_capped += 1
            total_values_capped += stat['total_capped']
        log.write(f"\\nVariable: {{stat['variable']}}\\n")
        log.write(f"  Mean: {{stat['mean']:.4f}}, Median: {{stat['median']:.4f}}, Std: {{stat['std']:.4f}}\\n")
        log.write(f"  Lower limit (mean - 2*std): {{stat['lower_limit']:.4f}}\\n")
        log.write(f"  Upper limit (mean + 2*std): {{stat['upper_limit']:.4f}}\\n")
        log.write(f"  Values capped below: {{stat['capped_below']}}, above: {{stat['capped_above']}}, total: {{stat['total_capped']}}\\n")
    log.write(f"\\nTotal: {{total_values_capped}} values capped across {{total_vars_capped}} variables\\n")

# Save the fragment to output
output_file = os.path.join(output_dir, "{cohort_id}_data_fragment.csv")
df_fragment.to_csv(output_file, index=False)

with open(log_file, "a") as log:
    log.write(f"\\nData fragment saved: {{output_file}}\\n")
    log.write(f"Fragment size: {{len(df_fragment)}} rows out of {{len(df_full)}} total rows ({{len(df_fragment)/len(df_full)*100:.1f}}%)\\n")
"""


def visualization_script(fragment_node_name: str, cohort_id: str, variable_names: list[str] = None) -> str:
    """Generate the data visualization script.
    
    This script:
    - Reads the full cohort data (or preview/shuffled sample)
    - Selects 5 random columns (or user-specified columns)
    - Creates histograms for numeric data and bar charts for categorical data
    - Saves visualization to PNG
    
    Args:
        fragment_node_name: Name of the data fragment node to read data from
        cohort_id: The cohort identifier (used to locate the data CSV file)
        variable_names: Optional list of variable names from the cohort (for documentation)
        
    Returns:
        The Python script as a string
    """
    # Generate the list of available variables (first 20, one per line for easy editing)
    if variable_names:
        vars_sample = variable_names[:20]
        vars_list = ",\n    ".join(f'"{v}"' for v in vars_sample)
    else:
        vars_list = '"var1", "var2", "var3"  # (variable list not available)'
    return f"""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tempfile
import zipfile

###############################################################################
# USER-CONFIGURABLE SECTION
# Modify the settings below to customize the visualization
###############################################################################

# DATA SOURCE
# -----------
# Choose which data source to visualize by uncommenting ONE of the options below:
#
# Option 1: The full cohort data - the original dataset.
# Note this cannot be accessed in development without approval of the data owners for the new script!
DATA_FILE = "/input/{cohort_id}"
#
# Option 2: Preview/Airlock sample - only the airlocked subset (20%) of the processed data and without ID column
# Note: this can be used in development stage for the purpose of refining the script. 
# To use the airlock subset: 
#   (a) select preview-airlock-{cohort_id} from the input dropdown menu to the right
#   (b) uncomment the line below    
# DATA_FILE = "/input/preview-airlock-{cohort_id}/{cohort_id}_data_fragment.csv"
#
# Option 3: Shuffled sample - the columns have been independently shuffled.
# To use this as data source:
#    (a) select {cohort_id}_shuffled_sample from the input dropdown menu to the right
#    (b) uncomment the line below 
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
# Controls the overall size of the output image.
# - 0.5 = small (50% of default, good for quick previews)
# - 1.0 = default size
# - 1.5 = large (150% of default, good for presentations)
CHART_SCALE = 0.5

###############################################################################
# END OF USER-CONFIGURABLE SECTION
###############################################################################

# Helper function to load data from CSV, SPSS, or zipped files
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
                raise ValueError(f"Could not read file as CSV, SPSS, or ZIP. CSV error: {{csv_error}}, SPSS error: {{spss_error}}, ZIP error: {{zip_error}}")

# Output directory (always exists in Decentriq environment)
output_dir = "/output"
OUTPUT_FILENAME = "sample_dataViz_{cohort_id}.png"

HISTOGRAM_BINS = 30
MAX_CATEGORIES = 20
NUM_RANDOM_COLUMNS = 5


# Read the data from the selected source
df = load_data(DATA_FILE)

# Log basic info
log_file = os.path.join(output_dir, "visualization_log.txt")
with open(log_file, "w") as log:
    log.write(f"Data source: {{DATA_FILE}}\\n")
    log.write(f"Loaded data: {{len(df)}} rows, {{len(df.columns)}} columns\\n")
    log.write(f"Columns: {{list(df.columns)}}\\n\\n")

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
            log.write(f"Loaded metadata: {{len(categorical_vars)}} categorical variables identified\\n\\n")
except Exception as e:
    with open(log_file, "a") as log:
        log.write(f"Could not load metadata dictionary: {{e}}\\n\\n")

# Determine which columns to visualize
if SELECTED_VARIABLES is not None:
    # Build a case-insensitive mapping from data columns
    # Also normalize spaces (replace multiple spaces with single, strip)
    def normalize_col(col):
        return ' '.join(col.lower().strip().split())
    
    df_cols_normalized = {{normalize_col(col): col for col in df.columns}}
    
    # Log the mapping for debugging
    with open(log_file, "a") as log:
        log.write(f"Column mapping (normalized -> actual):\\n")
        for norm, actual in list(df_cols_normalized.items())[:10]:
            log.write(f"  '{{norm}}' -> '{{actual}}'\\n")
        if len(df_cols_normalized) > 10:
            log.write(f"  ... and {{len(df_cols_normalized) - 10}} more\\n")
        log.write(f"\\nSELECTED_VARIABLES (first 10):\\n")
        for var in SELECTED_VARIABLES[:10]:
            log.write(f"  '{{var}}' -> normalized: '{{normalize_col(var)}}'\\n")
        log.write("\\n")
    
    # Match selected variables to actual column names (case-insensitive, space-normalized)
    selected_columns = []
    missing_cols = []
    for var in SELECTED_VARIABLES:
        var_normalized = normalize_col(var)
        if var_normalized in df_cols_normalized:
            selected_columns.append(df_cols_normalized[var_normalized])
        else:
            missing_cols.append(var)
    
    if missing_cols:
        with open(log_file, "a") as log:
            log.write(f"WARNING: The following requested columns were not found: {{missing_cols}}\\n")
    if not selected_columns:
        raise ValueError(f"None of the specified columns exist in the data. Available columns: {{list(df.columns)}}")
else:
    # Select random columns
    num_cols_to_visualize = min(NUM_RANDOM_COLUMNS, len(df.columns))
    selected_columns = random.sample(list(df.columns), num_cols_to_visualize)

with open(log_file, "a") as log:
    log.write(f"Selected {{len(selected_columns)}} columns for visualization: {{selected_columns}}\\n\\n")

# Create visualizations
num_cols = len(selected_columns)
fig, axes = plt.subplots(num_cols, 1, figsize=(10 * CHART_SCALE, 4 * num_cols * CHART_SCALE))
if num_cols == 1:
    axes = [axes]

for idx, col in enumerate(selected_columns):
    ax = axes[idx]
    col_data = df[col].dropna()
    
    with open(log_file, "a") as log:
        log.write(f"Column: {{col}}\\n")
        log.write(f"  Non-null values: {{len(col_data)}}\\n")
        log.write(f"  Data type: {{df[col].dtype}}\\n")
    
    # Check if numeric or categorical
    # Treat as categorical if: (1) metadata says it's categorical, (2) it's binary (0/1), or (3) pandas says it's not numeric
    unique_vals = col_data.unique()
    is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({{0, 1, 0.0, 1.0}})
    is_categorical_in_metadata = normalize_col(col) in {{normalize_col(v) for v in categorical_vars}}
    
    if pd.api.types.is_numeric_dtype(col_data) and not is_binary and not is_categorical_in_metadata:
        # Histogram for numeric data
        ax.hist(col_data, bins=HISTOGRAM_BINS, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {{col}} (Numeric)')
        
        # Add statistics
        mean_val = col_data.mean()
        median_val = col_data.median()
        std_val = col_data.std()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {{mean_val:.2f}}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {{median_val:.2f}}')
        ax.legend()
        
        with open(log_file, "a") as log:
            log.write(f"  Mean: {{mean_val:.4f}}, Median: {{median_val:.4f}}, Std: {{std_val:.4f}}\\n\\n")
    else:
        # Bar chart for categorical/binary data
        value_counts = col_data.value_counts().head(MAX_CATEGORIES)
        ax.barh(range(len(value_counts)), value_counts.values)
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels([str(v)[:30] for v in value_counts.index])  # Truncate long labels
        ax.set_xlabel('Count')
        chart_type = 'Binary' if is_binary else ('Categorical' if is_categorical_in_metadata else f'Categorical, top {{MAX_CATEGORIES}}')
        ax.set_title(f'Distribution of {{col}} ({{chart_type}})')
        
        with open(log_file, "a") as log:
            log.write(f"  Unique values: {{df[col].nunique()}}\\n")
            log.write(f"  Top 5 values: {{dict(value_counts.head(5))}}\\n\\n")

plt.tight_layout()
output_path = os.path.join(output_dir, OUTPUT_FILENAME)
plt.savefig(output_path, dpi=int(150 * CHART_SCALE), bbox_inches='tight')
plt.close()

with open(log_file, "a") as log:
    log.write(f"Visualization saved to {{OUTPUT_FILENAME}}\\n")
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
        
        f.write(f"\\n{'=' * 80}\\n")
        f.write(f"FILE: {filename}\\n")
        f.write(f"{'=' * 80}\\n")
        
        # File size in KB
        file_size_bytes = os.path.getsize(filepath)
        file_size_kb = file_size_bytes / 1024
        f.write(f"Size: {file_size_kb:.2f} KB ({file_size_bytes:,} bytes)\\n")
        
        # Last modified date
        mod_timestamp = os.path.getmtime(filepath)
        mod_date = datetime.fromtimestamp(mod_timestamp)
        f.write(f"Last Modified: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}\\n")
        
        # Try to load as tabular data and display info
        try:
            df = decentriq_util.read_tabular_data(filename)
            f.write(f"\\nDataFrame Info:\\n")
            f.write(f"  Rows: {len(df):,}\\n")
            f.write(f"  Columns: {len(df.columns):,}\\n")
            f.write(f"  Column names: {list(df.columns)}\\n")
            
            f.write(f"\\nFirst 5 rows:\\n")
            f.write(df.head(5).to_string() + "\\n")
            
        except Exception as e:
            f.write(f"\\nCould not load as tabular data: {e}\\n")
        
        f.write("\\n")
    
    f.write("=" * 80 + "\\n")
    f.write("EXPLORATION COMPLETE\\n")
    f.write("=" * 80 + "\\n")
    f.write(f"\\nReport written to {output_file}\\n")
"""
