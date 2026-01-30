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
data_node_id = "{cohort_id.replace(' ', '-')}"
df = load_data(f"/input/{{data_node_id}}")

with open(log_file, "a") as log:
    log.write(f"Loaded cohort data with {{len(df)}} rows and {{len(df.columns)}} columns\\n")

# Read the metadata dictionary (CSV only)
metadata_node_id = "{cohort_id.replace(' ', '-')}_metadata_dictionary"
try:
    metadata_df = load_metadata(f"/input/{{metadata_node_id}}")
    with open(log_file, "a") as log:
        log.write(f"Loaded metadata dictionary with {{len(metadata_df)}} variables\\n")
except Exception as e:
    metadata_df = None
    with open(log_file, "a") as log:
        log.write(f"Could not load metadata dictionary: {{e}}\\n")

# Remove ID column if it exists
id_column = "{id_variable_name if id_variable_name else ''}"
with open(log_file, "a") as log:
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])
        log.write(f"Removed ID column: {{id_column}}\\n")
    else:
        log.write(f"ID column not found or not specified\\n")

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
        for _, row in metadata_df.iterrows():
            var_name = str(row[varname_col]).strip()
            var_type = str(row[vartype_col]).strip().upper() if pd.notna(row[vartype_col]) else ''
            categorical = str(row[categorical_col]).strip() if categorical_col and pd.notna(row[categorical_col]) else ''
            
            # Check if numeric: VARTYPE in ['FLOAT', 'INT'] and CATEGORICAL is empty
            if var_type in ['FLOAT', 'INT'] and (categorical == '' or categorical.lower() == 'nan'):
                if var_name in df_fragment.columns:
                    numeric_vars.append(var_name)

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
# DATA_FILE = "/input/preview-fragment-{cohort_id}/{cohort_id}_data_fragment.csv"
#
# Option 3: Shuffled sample - synthetic/shuffled data for testing
# DATA_FILE = "/input/{cohort_id}_shuffled_sample/{cohort_id}_shuffled_sample.csv"

# VARIABLE SELECTION
# ------------------
# Edit the list below to select which variables to visualize.
# Remove variables you don't want, or set to None for random selection.
SELECTED_VARIABLES = [
    {vars_list}
]


###############################################################################
# END OF USER-CONFIGURABLE SECTION
###############################################################################

# Output directory (always exists in Decentriq environment)
output_dir = "/output"
OUTPUT_FILENAME = "sample_dataViz_{cohort_id}.png"

HISTOGRAM_BINS = 30
MAX_CATEGORIES = 20
OUTPUT_DPI = 150
NUM_RANDOM_COLUMNS = 5


# Read the data from the selected source
df = pd.read_csv(DATA_FILE)

# Log basic info
log_file = os.path.join(output_dir, "visualization_log.txt")
with open(log_file, "w") as log:
    log.write(f"Data source: {{DATA_FILE}}\\n")
    log.write(f"Loaded data: {{len(df)}} rows, {{len(df.columns)}} columns\\n")
    log.write(f"Columns: {{list(df.columns)}}\\n\\n")

# Determine which columns to visualize
if SELECTED_VARIABLES is not None:
    # Use user-specified variables (filter to only those that exist in the data)
    selected_columns = [col for col in SELECTED_VARIABLES if col in df.columns]
    missing_cols = [col for col in SELECTED_VARIABLES if col not in df.columns]
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
fig, axes = plt.subplots(num_cols, 1, figsize=(10, 4 * num_cols))
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
    if pd.api.types.is_numeric_dtype(col_data):
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
        # Bar chart for categorical data
        value_counts = col_data.value_counts().head(MAX_CATEGORIES)
        ax.barh(range(len(value_counts)), value_counts.values)
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels([str(v)[:30] for v in value_counts.index])  # Truncate long labels
        ax.set_xlabel('Count')
        ax.set_title(f'Distribution of {{col}} (Categorical, top {{MAX_CATEGORIES}})')
        
        with open(log_file, "a") as log:
            log.write(f"  Unique values: {{df[col].nunique()}}\\n")
            log.write(f"  Top 5 values: {{dict(value_counts.head(5))}}\\n\\n")

plt.tight_layout()
output_path = os.path.join(output_dir, OUTPUT_FILENAME)
plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches='tight')
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
