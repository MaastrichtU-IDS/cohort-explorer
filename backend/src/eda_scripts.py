"""Exploratory Data Analysis (EDA) module."""


# Shared helper code injected into both c2 and c3 via the {shared_helpers}
# placeholder, so the two nodes always agree on how data is loaded and how
# each cell is classified.
#
# Cell-state model (mutually exclusive, must sum to the number of rows):
#   valid          - parsed successfully to the variable's inferred type
#   empty          - blank / whitespace-only cell, nothing was recorded
#   coded_missing  - matches a missing code declared in the data dictionary
#   invalid        - a value is present but does not fit the expected type
_SHARED_HELPERS = '''
import math as _math
import re as _re

def _norm_token(v):
    # Normalise a raw cell/dictionary token so that "999", "999.0" and 999.0
    # all compare equal. Non-numeric tokens compare case-insensitively.
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except Exception:
        pass
    s = str(v).strip()
    if s == '':
        return ''
    try:
        f = float(s)
        if _math.isnan(f) or _math.isinf(f):
            return s.upper()
        if f == int(f):
            return str(int(f))
        return '%.10g' % f
    except Exception:
        return s.upper()

def _canon_value_token(v):
    # Categorical codes are often written as "1.0" by exporters. The dictionary
    # declares them as "1", so canonicalise numeric-looking tokens; leave text
    # values untouched.
    s = str(v).strip()
    if s == '':
        return s
    try:
        f = float(s)
        if _math.isnan(f) or _math.isinf(f):
            return s
        if f == int(f):
            return str(int(f))
        return '%.10g' % f
    except Exception:
        return s

def _parse_missing_codes(raw_value):
    # The dictionary MISSING column may declare several codes separated by "|".
    if raw_value is None:
        return []
    try:
        if pd.isna(raw_value):
            return []
    except Exception:
        pass
    s = str(raw_value).strip()
    if s == '' or s.lower() == 'nan':
        return []
    out = []
    for part in s.split('|'):
        p = part.strip()
        if p != '' and p not in out:
            out.append(p)
    return out

def _read_csv_as_text(path):
    # dtype=str + keep_default_na=False + na_values=[] means pandas performs no
    # NA conversion at all: an empty cell arrives as '' and strings such as
    # "NA" or "NaN" stay literal. We decide what counts as missing, not pandas.
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)

def _spss_as_text(path, notes):
    # Prefer pyreadstat so that SPSS *user-defined* missing values survive as
    # their declared codes; only SPSS system-missing becomes a blank cell.
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sav(path, user_missing=True)
        notes.append("SPSS file read with user-defined missing values preserved (pyreadstat).")
    except Exception as e:
        df = pd.read_spss(path)
        notes.append(
            "SPSS file read via pandas fallback (%s). User-defined missing values may already "
            "have been converted to blanks, so 'coded missing' counts could be understated." % str(e)
        )
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        col = df[c]
        out[c] = col.map(lambda x: '' if (x is None or (isinstance(x, float) and _math.isnan(x))) else str(x))
    return out

def load_data_as_text(file_path, notes=None):
    # Returns a DataFrame where every cell is a string and blank cells are ''.
    if notes is None:
        notes = []
    try:
        return _read_csv_as_text(file_path), notes
    except Exception as csv_error:
        try:
            return _spss_as_text(file_path, notes), notes
        except Exception as spss_error:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    data_files = []
                    for root, dirs, files in os.walk(tmpdir):
                        for fn in files:
                            if fn.endswith(('.csv', '.sav', '.CSV', '.SAV')):
                                data_files.append(os.path.join(root, fn))
                    if not data_files:
                        raise ValueError("No CSV or SPSS files found in the zip archive")
                    data_files = sorted(data_files)
                    frames = []
                    headers = None
                    for data_file in data_files:
                        if data_file.lower().endswith('.csv'):
                            d = _read_csv_as_text(data_file)
                        else:
                            d = _spss_as_text(data_file, notes)
                        cols = [str(c).strip().lower() for c in d.columns]
                        if headers is None:
                            headers = cols
                            first_file = os.path.basename(data_file)
                        elif set(cols) != set(headers):
                            only_first = sorted(set(headers) - set(cols))
                            only_this = sorted(set(cols) - set(headers))
                            raise ValueError(
                                "Header mismatch between files in the archive; refusing to continue. "
                                "'%s' vs '%s'. Columns only in the first file: %s. "
                                "Columns only in this file: %s."
                                % (first_file, os.path.basename(data_file), only_first, only_this)
                            )
                        frames.append(d)
                    if not frames:
                        raise ValueError("Could not read any files from the zip archive")
                    return pd.concat(frames, ignore_index=True), notes
            except Exception as zip_error:
                raise ValueError(
                    "Could not read file as CSV, SPSS, or ZIP. CSV error: %s, SPSS error: %s, ZIP error: %s"
                    % (csv_error, spss_error, zip_error)
                )

def candidate_tokens(raw, missing_codes):
    # Tokens that are neither blank nor a declared missing code, i.e. the values
    # that should be used when inferring a variable's type.
    txt = raw.astype('string').str.strip()
    empty_mask = txt.isna() | (txt == '')
    if missing_codes:
        norm_codes = set(_norm_token(c) for c in missing_codes)
        coded_mask = (~empty_mask) & txt.map(_norm_token).isin(norm_codes)
    else:
        coded_mask = pd.Series(False, index=raw.index)
    return txt[(~empty_mask) & (~coded_mask)]

def classify_series(raw, missing_codes, typ):
    # Partition a raw (string) column into valid / empty / coded_missing / invalid.
    n_rows = int(len(raw))
    txt = raw.astype('string').str.strip()
    empty_mask = txt.isna() | (txt == '')
    norm = txt.map(_norm_token)

    by_code = {}
    norm_to_code = {}
    for c in (missing_codes or []):
        norm_to_code[_norm_token(c)] = c
    if norm_to_code:
        coded_mask = (~empty_mask) & norm.isin(list(norm_to_code.keys()))
        for nc, orig in norm_to_code.items():
            by_code[str(orig)] = int(((~empty_mask) & (norm == nc)).sum())
    else:
        coded_mask = pd.Series(False, index=raw.index)

    rest_mask = (~empty_mask) & (~coded_mask)
    candidate = txt[rest_mask]

    if typ in ('int', 'float'):
        parsed = pd.to_numeric(candidate, errors='coerce')
        valid = parsed.dropna()
        if typ == 'int':
            try:
                valid = valid.astype('Int64')
            except Exception:
                pass
    elif typ == 'date':
        parsed = pd.to_datetime(candidate, errors='coerce')
        valid = parsed.dropna()
    else:
        valid = candidate.map(_canon_value_token).astype(object)

    n_empty = int(empty_mask.sum())
    n_coded = int(coded_mask.sum())
    n_valid = int(len(valid))
    n_invalid = int(rest_mask.sum()) - n_valid

    completeness = {
        'n_rows': n_rows,
        'n_valid': n_valid,
        'n_empty': n_empty,
        'n_coded_missing': n_coded,
        'n_invalid': n_invalid,
        'n_missing_total': n_empty + n_coded,
        'pct_empty': round(n_empty / n_rows * 100, 4) if n_rows else 0.0,
        'pct_coded_missing': round(n_coded / n_rows * 100, 4) if n_rows else 0.0,
        'pct_missing_total': round((n_empty + n_coded) / n_rows * 100, 4) if n_rows else 0.0,
        'pct_invalid': round(n_invalid / n_rows * 100, 4) if n_rows else 0.0,
        'missing_codes_declared': [str(c) for c in (missing_codes or [])],
        'coded_missing_by_code': by_code,
    }
    completeness['partition_check_ok'] = bool(
        n_valid + n_empty + n_coded + n_invalid == n_rows
    )
    return completeness, valid
'''


def c1_data_dict_check(cohort_id: str) -> str:
    raw_script = """
import pandas as pd
import decentriq_util
import zipfile
import os
import tempfile

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
                    
                    # Read and concatenate all files
                    dfs = []
                    for data_file in data_files:
                        try:
                            if data_file.lower().endswith('.csv'):
                                dfs.append(pd.read_csv(data_file))
                            else:
                                dfs.append(pd.read_spss(data_file))
                        except Exception as e:
                            print(f"Warning: Could not read {data_file}: {e}")
                    
                    if not dfs:
                        raise ValueError("Could not read any files from the zip archive")
                    
                    # Concatenate all dataframes
                    return pd.concat(dfs, ignore_index=True)
            except Exception as zip_error:
                raise ValueError(f"Could not read file as CSV, SPSS, or ZIP. CSV error: {csv_error}, SPSS error: {spss_error}, ZIP error: {zip_error}")

# Load the metadata dictionary
dictionary_df = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")
try:
    varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in dictionary_df.columns][0]
except:
    raise ValueError("The dictionary file does not contain a 'VARIABLE NAME'/'VARIABLENAME' column.")

print("metadata variable names: ", [v for v in dictionary_df[varname_col]])

# Load the dataset using the helper function
dataset_df = load_data("/input/{cohort_id}")
    
    

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
import zipfile
import tempfile
{shared_helpers}

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
    #meaning integers (handles negatives and trailing .0)
    import re
    _int_re = re.compile(r'^[+-]?\\d+(\\.0+)?$')
    try:
        non_na = series.dropna()
        if len(non_na) == 0:
            return False
        return non_na.apply(lambda x: bool(_int_re.match(str(x).strip()))).all()
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

# Load the data as raw text. Nothing is auto-converted: a blank cell arrives
# as '' and literal strings such as "NA" stay literal, so we - not pandas -
# decide what counts as missing.
load_notes = []
data, load_notes = load_data_as_text("/input/{cohort_id}", load_notes)

data.columns = [str(c).lower().strip() for c in data.columns]

#for col in data.columns:
#    try:
#        if data[col].dropna().apply(lambda x: str(x).isdigit() or str(x).endswith('.0')).all():
#            data[col] = data[col].astype('Int64')
#    except:
#        continue

# Define the pattern for entries to exclude non-categorical variables
#include_pattern = r'\\||='   # Look for strings containing either a | or =.
# Exclude rows in the dictionary where the 'CATEGORICAL' column contains the defined pattern
# categorical_dict = dictionary[dictionary['CATEGORICAL'].astype(str).str.contains(include_pattern, regex=True)]

vars_to_process = {}
# Prepare to extract classes and their meanings, along with MIN, MAX, and VAR TYPE
vars_details = {}
mismatched_types = {}

for _i, _note in enumerate(load_notes):
    mismatched_types['data-loading-note-%d' % _i] = _note

for index, row in dictionary.iterrows():
    variable_name = row[varname_col]
    var_type = row[vartype_col]
    categories_info = row['CATEGORICAL'] if 'CATEGORICAL' in dictionary.columns else None

    if variable_name.lower() in ['patientid', 'pat.id', 'patiëntnummer']:
        continue

    if variable_name.lower() not in data.columns:
        continue

    is_categorical_by_dict = (pd.notna(categories_info)
                              and isinstance(categories_info, str)
                              and categories_info.strip() != "")

    # A variable may declare several missing codes, e.g. "999|-1".
    missing_codes = _parse_missing_codes(row['MISSING']) if 'MISSING' in dictionary.columns else []

    class_names = {}
    if is_categorical_by_dict:
        for category in [item.strip() for item in categories_info.split('|')]:
            key_value = category.lower().split('=')
            if len(key_value) == 2:
                class_names[key_value[0].strip()] = key_value[1].upper().strip()
            elif len(key_value) == 1:
                #category does not have "="
                class_names[key_value[0].strip().upper()] = key_value[0].strip().upper()
            else:
                msg = f"Encountered a possible parsing error. Check category info for variable {variable_name}, {key_value}, Full category info: {categories_info}"
                mismatched_types[variable_name + "-categories"] = msg
                print(msg)
        # A category whose label is MISSING also declares a missing code.
        for _k, _v in class_names.items():
            if _v == 'MISSING' and _k not in missing_codes:
                missing_codes.append(_k)
                print("MISSING value exists among categories: ", variable_name, _k)

    if missing_codes:
        print(f"Missing code(s) {missing_codes} declared for variable: ", variable_name)
    else:
        print("No 'missing' value for variable ", variable_name)

    # Infer the type from values that are neither blank nor a declared missing
    # code, so that codes like 999 cannot distort the inference.
    cands = candidate_tokens(data[variable_name], missing_codes)

    if is_categorical_by_dict:
        t = 'categorical'
    elif len(cands) == 0:
        t = 'categorical'
    elif _column_is_numeric(cands):
        t = 'int' if cands.nunique() > 9 else 'categorical'
    elif _column_is_float(cands):
        t = 'float'
    elif _column_is_date(cands):
        t = 'date'
    elif cands.nunique() > 20:
        #assume float:
        t = 'float'
    else: #fewer than 20 unique
        #assume categorical:
        print("The following variable deemed categorical by process of elimination: ", variable_name)
        t = 'categorical'

    vars_to_process[variable_name] = t

    #find the mismatches between declared types (in data dictionary) and inferred types:
    if ((var_type.lower() == "datetime" and t != "date") or
        (var_type.lower() == "str" and t != "categorical")):
        mismatched_types[variable_name] = {"declared": var_type, "inferred": t}

    completeness, _valid_values = classify_series(data[variable_name], missing_codes, t)

    vars_details[variable_name] = {
            'var_label': row[varlabel_col],
            'missing_codes': [str(c) for c in missing_codes],
            # kept for backwards compatibility with earlier consumers
            'missing': (str(missing_codes[0]) if missing_codes else None),
            'declared_type': var_type,
            'inferred_type': t,
            'completeness': completeness,
            'count_missing': completeness['n_coded_missing'],
            'count_na': completeness['n_empty']
    }
    if t == 'categorical':
        vars_details[variable_name]['categories'] = class_names

    if not completeness['partition_check_ok']:
        mismatched_types[variable_name + "-partition"] = (
            "Cell-state counts (valid/empty/coded missing/invalid) do not sum to the row count."
        )
    if completeness['n_invalid'] > 0:
        mismatched_types[variable_name + "-invalid"] = (
            "%d value(s) are present but could not be read as %s."
            % (completeness['n_invalid'], t)
        )

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
    return raw_script.replace("{shared_helpers}", _SHARED_HELPERS).replace("{cohort_id}", cohort_id)



def c3_eda_data_profiling(cohort_id: str) -> str:
    raw_script = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, zscore, normaltest, median_abs_deviation, trim_mean, entropy
import warnings
import decentriq_util
import re
from datetime import datetime
import json
import collections.abc
from collections import OrderedDict
import zipfile
import tempfile
import os
warnings.filterwarnings('ignore')
{shared_helpers}

# Load the dataset with corrected missing values replaced with NA from previous step
# data_correct_missing = pd.read_csv("/input/C3_map_missing_do_not_run/data_correct.csv", low_memory=False)

#Load the JSON files from C2
vars_details = pd.read_json("/input/c2_save_to_json/variable_details.json")

with open("/input/c2_save_to_json/data_issues.json") as f:
    data_issues = json.load(f)

# Load the data as raw text (pandas performs no NA conversion), then rebuild a
# clean frame holding ONLY the valid values. Empty, coded-missing and invalid
# cells all become NaN here; their counts are preserved in completeness_by_var
# so the two kinds of missing are never conflated.
load_notes = []
raw_data, load_notes = load_data_as_text("/input/{cohort_id}", load_notes)
raw_data.columns = [str(c).lower().strip() for c in raw_data.columns]
for _note in load_notes:
    data_issues.append(_note)

n_rows_total = int(len(raw_data))
completeness_by_var = {}
data = pd.DataFrame(index=raw_data.index)

def _missing_codes_of(d):
    try:
        codes = d.get('missing_codes')
    except Exception:
        codes = None
    if codes is None:
        return []
    # read_json may hand this back as a list, a numpy array or a bare scalar.
    if isinstance(codes, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(c) for c in list(codes) if str(c).strip() != '']
    try:
        if pd.isna(codes):
            return []
    except Exception:
        pass
    s = str(codes).strip()
    return _parse_missing_codes(s) if s else []

for v in list(vars_details.columns):
    if v not in raw_data.columns:
        continue
    d = vars_details[v]
    typ = d['inferred_type']
    try:
        comp, valid = classify_series(raw_data[v], _missing_codes_of(d), typ)
    except Exception as e:
        msg = f"Failed to classify column {v} as {typ}: {e}"
        print(msg)
        data_issues.append(msg)
        continue
    completeness_by_var[v] = comp
    data[v] = valid.reindex(raw_data.index)
    if comp['n_invalid'] > 0:
        data_issues.append(
            f"Column {v}: {comp['n_invalid']} value(s) are present but could not be read as {typ}."
        )
    if not comp['partition_check_ok']:
        data_issues.append(
            f"Column {v}: cell-state counts do not sum to the row count ({comp['n_rows']})."
        )



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

import math

def _to_native(x):
    # Convert numpy/pandas scalars to JSON-native values; NaN/inf -> None
    if x is None:
        return None
    try:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return None
            return round(xf, 6)
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
    except Exception:
        pass
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x

def _int0(v):
    try:
        if v is None:
            return 0
        if isinstance(v, float) and math.isnan(v):
            return 0
        return int(v)
    except Exception:
        return 0

def _numeric_structured(series, comp):
    # All measures are aggregate (no per-row data points leave the enclave).
    # `series` holds only the valid values; every other cell state is
    # accounted for in `comp`.
    out = {}
    vals = pd.to_numeric(series, errors='coerce').dropna().astype(float)
    n = int(vals.shape[0])
    out['n'] = n
    out['completeness'] = comp
    out['n_unique'] = int(vals.nunique())
    if n == 0:
        return out
    q = vals.quantile([0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    mean = float(vals.mean())
    std = float(vals.std())
    out['mean'] = _to_native(mean)
    out['std'] = _to_native(std)
    out['variance'] = _to_native(std ** 2)
    out['cv'] = _to_native(std / mean) if mean != 0 else None
    out['sem'] = _to_native(std / np.sqrt(n))
    out['min'] = _to_native(float(vals.min()))
    out['max'] = _to_native(float(vals.max()))
    out['range'] = _to_native(float(vals.max()) - float(vals.min()))
    out['p1'] = _to_native(q.loc[0.01])
    out['p5'] = _to_native(q.loc[0.05])
    out['p10'] = _to_native(q.loc[0.10])
    out['q1'] = _to_native(q.loc[0.25])
    out['median'] = _to_native(q.loc[0.5])
    out['q3'] = _to_native(q.loc[0.75])
    out['p90'] = _to_native(q.loc[0.90])
    out['p95'] = _to_native(q.loc[0.95])
    out['p99'] = _to_native(q.loc[0.99])
    iqr_val = float(q.loc[0.75]) - float(q.loc[0.25])
    out['iqr'] = _to_native(iqr_val)
    med = float(vals.median())
    mad = float(median_abs_deviation(vals, scale=1.0))
    out['mad'] = _to_native(mad)
    try:
        out['trimmed_mean_10'] = _to_native(float(trim_mean(vals, 0.1)))
    except Exception:
        out['trimmed_mean_10'] = None
    out['skewness'] = _to_native(skew(vals, bias=False)) if n > 2 else None
    out['kurtosis'] = _to_native(kurtosis(vals, bias=False)) if n > 3 else None
    out['zero_fraction'] = _to_native(float((vals == 0).mean()))
    lb = float(q.loc[0.25]) - 1.5 * iqr_val
    ub = float(q.loc[0.75]) + 1.5 * iqr_val
    oi = int(((vals < lb) | (vals > ub)).sum())
    out['outliers_iqr'] = {'count': oi, 'pct': _to_native(oi / n * 100), 'lower_bound': _to_native(lb), 'upper_bound': _to_native(ub)}
    if mad > 0:
        mz = 0.6745 * (vals - med) / mad
        om = int((mz.abs() > 3.5).sum())
    else:
        om = 0
    out['outliers_mad'] = {'count': om, 'pct': _to_native(om / n * 100), 'threshold': 3.5}
    norm = {}
    if n >= 8:
        try:
            _k2, pv = normaltest(vals)
            norm['dagostino_p'] = _to_native(float(pv))
        except Exception:
            pass
    if n >= 3:
        try:
            sub = vals.sample(5000, random_state=42) if n > 5000 else vals
            _w, pw = shapiro(sub)
            norm['shapiro_p'] = _to_native(float(pw))
            norm['shapiro_n'] = int(len(sub))
        except Exception:
            pass
    _pv = norm.get('dagostino_p', norm.get('shapiro_p'))
    norm['is_normal'] = bool(_pv > 0.05) if _pv is not None else None
    out['normality'] = norm
    try:
        nb = int(min(50, max(10, int(np.sqrt(n)))))
        counts, edges = np.histogram(vals, bins=nb)
        out['histogram'] = {'counts': [int(c) for c in counts], 'bin_edges': [_to_native(float(e)) for e in edges]}
    except Exception:
        pass
    return out

def _categorical_structured(series, comp, categories_mapping):
    out = {}
    categories_mapping = categories_mapping or {}
    out['completeness'] = comp
    out['n_unique'] = int(series.nunique(dropna=True))
    s_norm = series.apply(_lowercase_if_string)
    # Only valid observations form the category distribution; empty and
    # coded-missing cells are reported in `completeness` instead.
    vc_valid = s_norm.value_counts(dropna=True)
    total = int(vc_valid.sum())
    out['n'] = total
    dist = []
    for k, c in vc_valid.items():
        label = categories_mapping.get(str(k), str(k))
        dist.append({'value': str(k), 'label': label, 'count': int(c), 'pct': _to_native(c / total * 100) if total else None})
    out['distribution'] = dist
    tot_valid = total
    if tot_valid > 0 and len(vc_valid) > 0:
        probs = (vc_valid / tot_valid).values
        ent = float(entropy(probs, base=2))
        out['entropy_bits'] = _to_native(ent)
        out['normalized_entropy'] = _to_native(ent / np.log2(len(probs))) if len(probs) > 1 else _to_native(0.0)
        out['gini_impurity'] = _to_native(float(1 - np.sum(probs ** 2)))
        out['effective_n_categories'] = _to_native(float(2 ** ent))
        out['imbalance_ratio'] = _to_native(float(vc_valid.max() / vc_valid.min())) if vc_valid.min() > 0 else None
        top = vc_valid.idxmax()
        out['most_frequent'] = {'value': str(top), 'label': categories_mapping.get(str(top), str(top)), 'pct': _to_native(float(vc_valid.max()) / total * 100) if total else None}
    return out

def _date_structured(series, comp):
    out = {}
    d = pd.to_datetime(series, errors='coerce').dropna()
    out['n'] = int(len(d))
    out['n_unique'] = int(d.nunique())
    out['completeness'] = comp
    if len(d) == 0:
        return out
    out['min'] = str(d.min().date())
    out['max'] = str(d.max().date())
    out['range_days'] = int((d.max() - d.min()).days)
    try:
        q = d.quantile([0.25, 0.5, 0.75])
        out['q1'] = str(q.loc[0.25].date())
        out['median'] = str(q.loc[0.5].date())
        out['q3'] = str(q.loc[0.75].date())
    except Exception:
        pass
    out['future_dates'] = int((d > pd.Timestamp.now()).sum())
    return out

def write_structured_v2(structured_stats):
    # Writes a typed, aggregate-only EDA output to a NEW file (v2), never
    # overwriting the legacy eda_output_{cohort_id}.json.
    try:
        meta = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")
        varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in meta.columns][0]
        meta_lookup = {}
        for _, r in meta.iterrows():
            vn = str(r[varname_col]).lower().strip()
            meta_lookup[vn] = {c.strip().lower(): (None if pd.isna(r[c]) else r[c]) for c in meta.columns}
    except Exception as e:
        print("v2: could not read metadata for enrichment:", e)
        meta_lookup = {}
    variables = {}
    for vname, st in structured_stats.items():
        entry = dict(st)
        m = meta_lookup.get(vname, {})
        entry['metadata'] = {
            'label': m.get('variable label') or m.get('variablelabel') or m.get('var label'),
            'var_type': m.get('var type') or m.get('vartype'),
            'units': m.get('units'),
            'concept_code': m.get('variable concept code'),
            'concept_name': m.get('variable concept name'),
            'omop_id': m.get('variable omop id'),
            'domain': m.get('domain'),
            'visits': m.get('visits'),
        }
        entry['graph_url'] = f"https://explorer.icare4cvd.eu/api/variable-graph/{cohort_id}/{vname}"
        variables[vname] = entry
    output = {
        'schema_version': '2.0',
        'cohort_id': '{cohort_id}',
        'generated_at': datetime.now().isoformat(),
        'n_rows': int(n_rows_total),
        'n_variables': int(len(variables)),
        'variables': variables,
        'data_issues': data_issues,
    }
    with open("/output/eda_output_v2_{cohort_id}.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, default=str)
    print("v2 structured EDA output written:", len(variables), "variables")

def variable_eda(df, vars_details):
    vars_stats = {}
    structured = {}
    graph_tick_data = {}
    df.columns = df.columns.str.lower().str.strip()
    for column in df.columns.tolist():
        if column not in vars_details.columns:
            continue
        # Continuous variables
        try:
            # `df` already holds only the valid values: blank, coded-missing and
            # invalid cells were separated out when the frame was built, and the
            # counts for each of those states live in `comp`.
            comp = completeness_by_var.get(column, {})
            count_nonnull = comp.get('n_valid', int(df[column].notna().sum()))
            count_na = comp.get('n_empty', 0)
            count_missing = comp.get('n_coded_missing', 0)
            count_invalid = comp.get('n_invalid', 0)
            count_missing_total = count_na + count_missing
            missing_codes_txt = ', '.join(comp.get('missing_codes_declared', [])) or 'none'

            if vars_details[column]['inferred_type'] in ['int', 'float']:

                #if not pd.api.types.is_numeric_dtype(df[column]):
                    # Skip if the column is not numeric
                    # print("Column ", column, " skipped because non-numeric")
                    #continue

                # Descriptive Stats
                stats = df[column].describe()
                mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
                value_counts = df[column].value_counts(dropna=False)
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

                # Normality Test (Shapiro capped: unreliable/invalid for very large n)
                if len(df[column].dropna()) > 3:  # Shapiro requires at least 3 values
                    _shapiro_input = df[column].dropna()
                    if len(_shapiro_input) > 5000:
                        _shapiro_input = _shapiro_input.sample(5000, random_state=42)
                    w_test_stat, p = shapiro(_shapiro_input)
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

                # Stats Text
                stats_text = (
                    f"Column: {column}",
                    f"Label: {vars_details[column]['var_label']}",
                    f"Type: Numeric (encoded as {df[column].dtype})",
                    f"Count of valid observations: {count_nonnull}",
                    f"Count empty (blank cells): {count_na} ({comp.get('pct_empty', 0):.2f}%)",
                    f"Count coded missing: {count_missing} ({comp.get('pct_coded_missing', 0):.2f}%)",
                    f"Count missing total: {count_missing_total} ({comp.get('pct_missing_total', 0):.2f}%)",
                    f"Count invalid: {count_invalid} ({comp.get('pct_invalid', 0):.2f}%)",
                    f"Missing code(s) declared: {missing_codes_txt}",
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

                try:
                    structured[column] = _numeric_structured(df[column], comp)
                    structured[column]['type'] = 'numeric'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured numeric stats for {column}: {str(e)}")

                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'numerical')
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")


            # Categorical variables
            elif vars_details[column]['inferred_type'] == 'categorical':
                stats = df[column].describe()
                # Only valid observations: blanks and coded-missing values are
                # reported separately rather than shown as categories.
                value_counts = df[column].apply(_lowercase_if_string).value_counts(dropna=True)
                total = int(value_counts.sum())

                # Get the categories mapping and normalize keys
                categories_mapping = vars_details[column].get("categories", None)
                if not isinstance(categories_mapping, dict):
                    categories_mapping = {}
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
                        f"Count of valid observations: {count_nonnull}",
                        f"Count empty (blank cells): {count_na} ({comp.get('pct_empty', 0):.2f}%)",
                        f"Count coded missing: {count_missing} ({comp.get('pct_coded_missing', 0):.2f}%)",
                        f"Count missing total: {count_missing_total} ({comp.get('pct_missing_total', 0):.2f}%)",
                        f"Count invalid: {count_invalid} ({comp.get('pct_invalid', 0):.2f}%)",
                        f"Missing code(s) declared: {missing_codes_txt}",
                        f"Class balance: {class_balance_text}",
                        f"Chi-Square Test Statistic: {chi_square_stat:.2f}"
                    )

                try:
                    structured[column] = _categorical_structured(df[column], comp, categories_mapping)
                    structured[column]['type'] = 'categorical'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured categorical stats for {column}: {str(e)}")

                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'categorical', category_mapping = categories_mapping)
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")
                        
                        
            
            elif vars_details[column]['inferred_type'] == 'date':
                try:
                    stats = pd.to_datetime(df[column], errors='coerce').describe()
                except:
                    continue
                value_counts = df[column].value_counts(dropna=True)
                total = int(value_counts.sum())
                stats_text = [
                        f"Column: {column}",
                        f"Label: {vars_details[column]['var_label']}",
                        f"Type: Date (encoded as {df[column].dtype})",
                        f"Number of unique values: {len(value_counts)}",
                        f"Most frequent value: {str(value_counts.idxmax()).split('.')[0]}",
                        f"Count of valid observations: {count_nonnull}",
                        f"Count empty (blank cells): {count_na} ({comp.get('pct_empty', 0):.2f}%)",
                        f"Count coded missing: {count_missing} ({comp.get('pct_coded_missing', 0):.2f}%)",
                        f"Count missing total: {count_missing_total} ({comp.get('pct_missing_total', 0):.2f}%)",
                        f"Count invalid: {count_invalid} ({comp.get('pct_invalid', 0):.2f}%)",
                        f"Missing code(s) declared: {missing_codes_txt}",
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
                try:
                    structured[column] = _date_structured(df[column], comp)
                    structured[column]['type'] = 'date'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured date stats for {column}: {str(e)}")

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
    return vars_stats, structured



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
            plt.close('all')
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
    plt.close('all')
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
    # NOTE: written under a v2 name so that previously generated
    # eda_output_{cohort_id}.json files are never overwritten on re-runs.
    with open("/output/eda_output_flat_v2_{cohort_id}.json", 'w', encoding='utf-8') as f:
        json.dump(json_dicts, f, indent=4)


def _convert_numeric(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

vars_to_stats, structured_stats = variable_eda(data, vars_details)
meta_data_enriched = integrate_eda_with_metadata(vars_to_stats)
json_dicts = dataframe_to_json_dicts(meta_data_enriched)

try:
    write_structured_v2(structured_stats)
except Exception as e:
    print("Failed to write v2 structured EDA output:", e)
    data_issues.append(f"Failed to write v2 structured EDA output: {str(e)}")

with open('/output/data_issues.json', 'w') as json_file:
    json.dump(data_issues, json_file, indent=4)
"""
    return raw_script.replace("{shared_helpers}", _SHARED_HELPERS).replace("{cohort_id}", cohort_id)


def shuffle_data(cohort_id: str) -> str:
    raw_script = """
import pandas as pd
import numpy as np
import decentriq_util
from datetime import datetime
import zipfile
import tempfile
import os

# Helper function to load data from CSV, SPSS, or zipped files
def load_data(file_path):
    # Try CSV first
    try:
        return pd.read_csv(file_path, na_values=[''], keep_default_na=False)
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
                    
                    # Read and concatenate all files
                    dfs = []
                    for data_file in data_files:
                        try:
                            if data_file.lower().endswith('.csv'):
                                dfs.append(pd.read_csv(data_file, na_values=[''], keep_default_na=False))
                            else:
                                dfs.append(pd.read_spss(data_file))
                        except Exception as e:
                            print(f"Warning: Could not read {data_file}: {e}")
                    
                    if not dfs:
                        raise ValueError("Could not read any files from the zip archive")
                    
                    # Concatenate all dataframes
                    return pd.concat(dfs, ignore_index=True)
            except Exception as zip_error:
                raise ValueError(f"Could not read file as CSV, SPSS, or ZIP. CSV error: {csv_error}, SPSS error: {spss_error}, ZIP error: {zip_error}")

# Configuration
SAMPLE_SIZE = 500  # Number of rows to output
SAMPLE_FRACTION = 0.2  # Alternative: fraction of data to sample
RANDOM_SEED = 42  # For reproducibility

# Load the metadata dictionary to find patient ID variable
dictionary_df = decentriq_util.read_tabular_data("/input/{cohort_id}-metadata")

# Clean column names to ensure uniformity
dictionary_df.columns = dictionary_df.columns.str.strip().str.upper()

# Find the patient ID variable (OMOP ID 4086934)
patient_id_var = None
if 'VARIABLE OMOP ID' in dictionary_df.columns:
    varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in dictionary_df.columns]
    if varname_col:
        varname_col = varname_col[0]
        patient_id_rows = dictionary_df[dictionary_df['VARIABLE OMOP ID'] == '4086934']
        if not patient_id_rows.empty:
            patient_id_var = patient_id_rows.iloc[0][varname_col]
            print(f"Found patient ID variable with OMOP ID 4086934: {patient_id_var}")
        else:
            print("No variable found with OMOP ID 4086934 in metadata dictionary")
    else:
        print("Could not find variable name column in metadata dictionary")
else:
    print("'VARIABLE OMOP ID' column not found in metadata dictionary")

# EXPLICITLY DEFINE PII COLUMNS TO REMOVE
# Modify this list based on the cohort
PII_COLUMNS = []

# Add the patient ID variable to PII columns if found
if patient_id_var:
    PII_COLUMNS.append(patient_id_var)
    print(f"Added {patient_id_var} to PII columns to be removed")

# Read the input data using helper function
df = load_data("/input/{cohort_id}")

# Store original shape
original_rows = len(df)
original_cols = len(df.columns)

# Filter to only variables present in the metadata dictionary
# (exclude any data columns not described in the dictionary)
dict_varname_col = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in dictionary_df.columns]
if dict_varname_col:
    dict_variables = set(dictionary_df[dict_varname_col[0]].dropna().str.strip().tolist())
    # Keep only columns that appear in the dictionary (case-insensitive match)
    dict_variables_lower = {v.lower() for v in dict_variables}
    cols_to_keep = [col for col in df.columns if col.strip().lower() in dict_variables_lower]
    cols_excluded = [col for col in df.columns if col.strip().lower() not in dict_variables_lower]
    if cols_excluded:
        print(f"Excluding {len(cols_excluded)} columns not in metadata dictionary: {cols_excluded[:20]}")
    df = df[cols_to_keep]
    print(f"Retained {len(cols_to_keep)} columns matching metadata dictionary")

# Remove PII columns
columns_to_drop = [col for col in PII_COLUMNS if col in df.columns]
df_anonymized = df.drop(columns=columns_to_drop, errors='ignore')

# Shuffle each column independently
df_shuffled = pd.DataFrame()

for column in df_anonymized.columns:
    # Get non-null values
    non_null_mask = ~df_anonymized[column].isna()
    non_null_values = df_anonymized.loc[non_null_mask, column].values
    
    if len(non_null_values) > 0:
        # Create a copy of the column
        new_column = df_anonymized[column].copy()
        
        # Shuffle the non-null values
        np.random.seed(RANDOM_SEED + hash(column) % 10000)
        shuffled_values = non_null_values.copy()
        np.random.shuffle(shuffled_values)
        
        # Replace non-null values with shuffled ones
        new_column[non_null_mask] = shuffled_values
        df_shuffled[column] = new_column
    else:
        # Column is all nulls, preserve as is
        df_shuffled[column] = df_anonymized[column]

# Sample the shuffled data
n_samples = min(SAMPLE_SIZE, len(df_shuffled))
if len(df_shuffled) * SAMPLE_FRACTION < n_samples:
    n_samples = int(len(df_shuffled) * SAMPLE_FRACTION)

df_sample = df_shuffled.sample(n=n_samples, random_state=RANDOM_SEED)
df_sample = df_sample.reset_index(drop=True)

# Add synthetic IDs for reference
df_sample.insert(0, 'Synthetic_ID', ['SYNTH_' + str(i).zfill(5) for i in range(1, len(df_sample) + 1)])

# Save the shuffled sample
df_sample.to_csv('/output/shuffled_sample.csv', index=False)

# Create a simple summary report
summary_template = '''DATA SHUFFLE COMPLETE
====================
Timestamp: {}

Original: {:,} rows × {} columns
Excluded (not in metadata dict): {} columns
PII Removed: {} columns
Retained: {} columns
Output Sample: {:,} rows

Privacy Method: Independent column shuffling
- Each column shuffled separately
- All correlations destroyed
- No patient reconstruction possible

Removed PII columns: {}
'''
summary = summary_template.format(
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    original_rows,
    original_cols,
    len(cols_excluded) if 'cols_excluded' in dir() else 0,
    len(columns_to_drop),
    len(df_anonymized.columns),
    len(df_sample),
    ', '.join(columns_to_drop) if columns_to_drop else 'None'
)

with open('/output/shuffle_summary.txt', 'w') as f:
    f.write(summary)
"""
    return raw_script.replace("{cohort_id}", cohort_id)
