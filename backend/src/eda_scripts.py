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
'''


# Chart-styling helpers shared between c3 (per-variable EDA plots) and
# c4 (longitudinal plots). Kept separate from _SHARED_HELPERS because c2
# never imports matplotlib.
_CHART_STYLE_HELPERS = '''
import textwrap as _textwrap

# Classic matplotlib/seaborn defaults are kept deliberately: the only shared
# styling is the summary-stats panel and a light grid.
_CATEGORY_PALETTE = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
_ACCENT = _CATEGORY_PALETTE[0]
_ACCENT_DARK = _CATEGORY_PALETTE[0]
_GRID = '#CCCCCC'
_TEXT_MUTED = 'black'
_STATS_FONTSIZE = 10
_STATS_LINESPACING = 1.5
_STATS_MAX_CHARS = 62
# Sentinel entry that separates groups of related stats with a blank line.
_GROUP_BREAK = ''
# Shown on the chart title and in the stats panel whenever full calendar dates
# are rolled up before charting.
_MONTH_AGG_NOTE = 'values aggregated to month+year'

def _has_day_granularity(dates):
    # True when the values are full calendar dates rather than already
    # month- or year-level values (in which case every value shares one day).
    try:
        present = dates.dropna()
        if present.empty:
            return False
        return int(present.dt.day.nunique()) > 1
    except (AttributeError, TypeError, ValueError):
        return False

def _clean_axis(ax, grid_axis='y'):
    # Keep the default frame; add a faint grid only.
    if grid_axis:
        ax.grid(axis=grid_axis, color=_GRID, linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)

def _panel_title(ax, title):
    ax.set_title(title, fontsize=12)

def _fmt_stat(value, decimals=2, suffix=''):
    # Panel-safe rendering of a v2 structured measure, which may be None when
    # the sample was too small to compute it.
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num != num:
        return 'N/A'
    return f"{num:,.{decimals}f}{suffix}"

def _fmt_count(count, pct=None, small_pct_cutoff=5.0):
    # A zero count needs no "(0%)" suffix. Otherwise the share is rounded to a
    # whole number, except for small shares where the decimals carry the
    # information (0.36% and 4% should not both collapse to the same digit).
    if count is None:
        return 'N/A'
    try:
        n = int(count)
    except (TypeError, ValueError):
        return str(count)
    if n == 0 or pct is None:
        return str(n)
    try:
        share = float(pct)
    except (TypeError, ValueError):
        return str(n)
    if share != share:
        return str(n)
    if abs(share) < small_pct_cutoff:
        return f"{n} ({share:.2f}%)"
    return f"{n} ({round(share):.0f}%)"

def _fmt_pct(pct, small_pct_cutoff=5.0):
    # Same rounding rule as _fmt_count: shares of 5% and above carry no useful
    # decimals, while below that the decimals are the information.
    if pct is None:
        return 'N/A'
    try:
        share = float(pct)
    except (TypeError, ValueError):
        return str(pct)
    if share != share:
        return 'N/A'
    if abs(share) < small_pct_cutoff:
        return f"{share:.2f}%"
    return f"{round(share):.0f}%"

def _fmt_pct_dual(pct_valid, pct_all, small_pct_cutoff=5.0):
    # "% of valid / % of all rows". When missingness is zero (or too small to
    # survive rounding) the two are identical, so only one is shown.
    left = _fmt_pct(pct_valid, small_pct_cutoff)
    right = _fmt_pct(pct_all, small_pct_cutoff)
    if left == right:
        return left
    return f"{left} / {right}"

def _format_stats_lines(stats_text, max_chars=_STATS_MAX_CHARS, label_pad_cap=20):
    # Split every entry on its first colon, then pad labels to a common width so
    # that all values start in the same column. Values that are too long are
    # wrapped and hanging-indented instead of breaking mid-entry.
    rows = []
    for raw in stats_text:
        line = str(raw)
        if not line.strip():
            rows.append((None, None))
        elif ':' in line:
            label, value = line.split(':', 1)
            rows.append((' '.join(label.split()), ' '.join(value.split())))
        else:
            rows.append((' '.join(line.split()), None))
    if not rows:
        return []
    label_width = min(max((len(lbl) for lbl, _ in rows if lbl), default=0), label_pad_cap)
    out = []
    for label, value in rows:
        if label is None:
            # Group separator: never leading, never doubled.
            if out and out[-1] != '':
                out.append('')
            continue
        if value is None:
            out.extend(_textwrap.wrap(label, max_chars) or [''])
            continue
        head = (label + ':').ljust(label_width + 2)
        if not head.endswith(' '):
            # Label is longer than the common gutter; keep at least one space.
            head = head + ' '
        indent = ' ' * len(head)
        avail = max(12, max_chars - len(head))
        chunks = _textwrap.wrap(value, avail) or ['']
        out.append(head + chunks[0])
        for extra in chunks[1:]:
            out.append(indent + extra)
    return out

def _stats_figure_height(stats_lines, minimum=6.0, maximum=18.0):
    # Size the canvas to the panel content so the chart is never left floating
    # in a mostly empty figure. Capped: an uncapped panel with one line per
    # category can otherwise ask for a canvas hundreds of inches tall.
    line_in = (_STATS_FONTSIZE * _STATS_LINESPACING) / 72.0
    return min(maximum, max(minimum, len(stats_lines) * line_in + 1.3))

def _render_stats_panel(ax, title, stats_lines):
    ax.axis('off')
    _panel_title(ax, title)
    props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.8, edgecolor='lightgray')
    return ax.text(0.02, 0.98, '\\n'.join(stats_lines), transform=ax.transAxes,
                    fontsize=_STATS_FONTSIZE, va='top', ha='left', family='monospace',
                    bbox=props, linespacing=_STATS_LINESPACING, wrap=False)
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

    def _dict_val(col):
        if col not in dictionary.columns:
            return None
        v = row[col]
        if pd.isna(v):
            return None
        v = str(v).strip()
        return v if v and v.lower() != 'na' else None

    vars_details[variable_name] = {
            'var_label': row[varlabel_col],
            'missing_codes': [str(c) for c in missing_codes],
            # kept for backwards compatibility with earlier consumers
            'missing': (str(missing_codes[0]) if missing_codes else None),
            'declared_type': var_type,
            'inferred_type': t,
            'completeness': completeness,
            'count_missing': completeness['n_coded_missing'],
            'count_na': completeness['n_empty'],
            # Used by longitudinal_analysis to identify longitudinal
            # families (same concept measured at different visits) and to
            # detect the patient-id column.
            'omop_id': _dict_val('VARIABLE OMOP ID'),
            'concept_code': _dict_val('VARIABLE CONCEPT CODE'),
            # Two variables only belong to the same longitudinal family if they
            # share the concept *and* the additional context, so the context is
            # part of the grouping key rather than decoration.
            'additional_context': _dict_val('ADDITIONAL CONTEXT CONCEPT NAME'),
            'units': _dict_val('UNITS'),
            'visits': _dict_val('VISITS'),
            'visit_concept_name': _dict_val('VISIT CONCEPT NAME'),
            'visit_concept_code': _dict_val('VISIT CONCEPT CODE'),
            'visit_omop_id': _dict_val('VISIT OMOP ID'),
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
{chart_style}

# Study/cohort this DCR belongs to; shown on every panel and chart title.
COHORT_ID = "{cohort_id}"

def _meta_value(details, key):
    # Read an optional metadata-dictionary field, treating NaN and the various
    # spellings of "not applicable" as absent.
    try:
        value = details.get(key)
    except Exception:
        return ''
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in ('', 'na', 'n/a', 'nan', 'none', 'null', '-', '--', 'not applicable'):
        return ''
    return text

def _identity_lines(column, details):
    # Study / column / label, plus the visit type and code when the metadata
    # dictionary declares them.
    lines = [
        f"Study: {COHORT_ID}",
        f"Column: {column}",
        f"Label: {_meta_value(details, 'var_label') or column}",
    ]
    visit_type = _meta_value(details, 'visits')
    visit_code = _meta_value(details, 'visit_concept_code')
    visit_name = _meta_value(details, 'visit_concept_name')
    if visit_type:
        lines.append(f"Visit type: {visit_type}")
    if visit_name:
        lines.append(f"Visit concept: {visit_name}")
    if visit_code:
        lines.append(f"Visit code: {visit_code}")
    return lines

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
# Columns are collected first and concatenated once: assigning 1400 columns one
# at a time fragments the pandas block manager and gets progressively slower.
_data_cols = {}

for v in list(vars_details.columns):
    if v not in raw_data.columns:
        continue
    d = vars_details[v]
    typ = d['inferred_type']
    try:
        comp, valid = classify_series(raw_data[v], _missing_codes_of(d), typ)
    except Exception as e:
        msg = f"Failed to classify column {v} as {typ}: {e}"
        print(msg, flush=True)
        data_issues.append(msg)
        continue
    completeness_by_var[v] = comp
    _data_cols[v] = valid.reindex(raw_data.index)
    if comp['n_invalid'] > 0:
        data_issues.append(
            f"Column {v}: {comp['n_invalid']} value(s) are present but could not be read as {typ}."
        )
    if not comp['partition_check_ok']:
        data_issues.append(
            f"Column {v}: cell-state counts do not sum to the row count ({comp['n_rows']})."
        )

if _data_cols:
    data = pd.concat(_data_cols, axis=1)
    data = data[list(_data_cols.keys())]
else:
    data = pd.DataFrame(index=raw_data.index)



#variables that should be graphed
#vars_to_graph = ['age', 'weight', 'cough1', 'angina1', 'hscrp_v6']
#vars_to_graph = ['age', 'ALCOOL', 'ALLOPURI', 'ALT', 'ALTACE', 'ALTANO', 'COLETOT', 'CREATIN', 'DALTACE', 'DATAECG', 'DATALAB']
#vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB']
#vars_to_graph = [x.lower() for x in vars_to_graph]
#vars_to_graph = ['age', 'ALCOOL', 'DATAECG', 'DATALAB', 'AATHORAX', 'AATHORAXDIM', 'ACE_AT_V1']
# Patient/subject identifiers hold no analytical signal and must never be
# profiled or charted. Every cohort names that column differently, so they are
# recognised by their standardised concept instead of by name.
_PATIENT_OMOP_ID = '4086934'
_PATIENT_CONCEPT_CODE = '184107009'

def _is_patient_id(details):
    omop_id = _meta_value(details, 'omop_id')
    if omop_id.endswith('.0'):
        omop_id = omop_id[:-2]
    if omop_id == _PATIENT_OMOP_ID:
        return True
    return _PATIENT_CONCEPT_CODE in _meta_value(details, 'concept_code')

patient_id_cols = set()
for _v in list(vars_details.columns):
    if _is_patient_id(vars_details[_v]):
        patient_id_cols.add(str(_v).strip().lower())
        data_issues.append(
            f"Variable {_v} is a patient identifier (SNOMED:{_PATIENT_CONCEPT_CODE} / "
            f"OMOP {_PATIENT_OMOP_ID}); excluded from profiling."
        )

vars_to_graph = list(vars_details.columns)
vars_to_graph = [x.strip().lower() for x in vars_to_graph]
vars_to_graph = [x for x in vars_to_graph if x not in patient_id_cols]

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
    # `pct` is a share of the valid observations; `pct_of_total` is a share of
    # every row in the cohort. They differ by the variable's missingness, so
    # both are stated rather than left to the consumer to reconstruct.
    n_rows = comp.get('n_rows') if isinstance(comp, dict) else None
    dist = []
    for k, c in vc_valid.items():
        label = categories_mapping.get(str(k), str(k))
        dist.append({
            'value': str(k),
            'label': label,
            'count': int(c),
            'pct': _to_native(c / total * 100) if total else None,
            'pct_of_total': _to_native(c / n_rows * 100) if n_rows else None,
        })
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
    import time as _time
    _t0 = _time.time()
    _all_cols = df.columns.tolist()
    _done = 0
    for column in _all_cols:
        if column not in vars_details.columns:
            continue
        if column in patient_id_cols:
            # No stats, no chart, no structured entry for identifiers.
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
            # Set by the categorical branch only; re-injected into the flat
            # output below because the per-category panel lines carry no colon.
            class_balance_flat = None

            if vars_details[column]['inferred_type'] in ['int', 'float']:

                #if not pd.api.types.is_numeric_dtype(df[column]):
                    # Skip if the column is not numeric
                    # print("Column ", column, " skipped because non-numeric")
                    #continue

                # Descriptive Stats
                stats = df[column].describe()
                mode_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
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
                _col_vals = df[column]
                outliers = int(((_col_vals < lower_bound) | (_col_vals > upper_bound)).sum())

                # Z-Scores for Outliers
                z_scores = zscore([_ for _ in df[column].dropna()]) if len(df[column].dropna()) > 0 else np.array([])
                z_outliers = (np.abs(z_scores) > 3).sum() if z_scores.size > 0 else 0

                # Range Calculation
                range_value = stats['max'] - stats['min']

                # Structured v2 measures are computed first so the panel can
                # reuse them instead of recomputing.
                try:
                    structured[column] = _numeric_structured(df[column], comp)
                    structured[column]['type'] = 'numeric'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured numeric stats for {column}: {str(e)}")
                _sv = structured.get(column, {})
                _out_mad = _sv.get('outliers_mad') or {}
                _norm = _sv.get('normality') or {}
                _zero_frac = _sv.get('zero_fraction')

                # Stats Text. _GROUP_BREAK entries render as blank lines so the
                # panel reads as labelled blocks rather than one long list.
                stats_text = tuple(_identity_lines(column, vars_details[column])) + (
                    f"Type: Numeric (encoded as {df[column].dtype})",
                    _GROUP_BREAK,
                    f"Count of valid observations: {count_nonnull} of {n_rows_total} rows",
                    f"Count empty (blank cells, % of all rows): {_fmt_count(count_na, comp.get('pct_empty', 0))}",
                    f"Count coded missing (% of all rows): {_fmt_count(count_missing, comp.get('pct_coded_missing', 0))}",
                    f"Count missing total (% of all rows): {_fmt_count(count_missing_total, comp.get('pct_missing_total', 0))}",
                    f"Count invalid (% of all rows): {_fmt_count(count_invalid, comp.get('pct_invalid', 0))}",
                    f"Missing code(s) declared: {missing_codes_txt}",
                    f"Number of Unique Values/Categories: {df[column].nunique()}",
                    _GROUP_BREAK,
                    f"Mean: {stats['mean']:.2f}",
                    f"Median: {stats['50%']:.2f}",
                    f"Mode: {mode_value:.2f}",
                    f"Trimmed mean (10%): {_fmt_stat(_sv.get('trimmed_mean_10'))}",
                    _GROUP_BREAK,
                    f"Std Dev: {stats['std']:.2f}",
                    f"Variance: {stats['std']**2:.2f}",
                    f"CV: {_fmt_stat(_sv.get('cv'), 3)}",
                    f"MAD: {_fmt_stat(_sv.get('mad'))}",
                    f"Min: {stats['min']:.2f}",
                    f"Max: {stats['max']:.2f}",
                    f"Range: {range_value:.2f}", 
                    _GROUP_BREAK,
                    f"Q1: {Q1:.2f}",
                    f"Q3: {Q3:.2f}",
                    f"IQR: {IQR:.2f}",
                    f"P5 / P95: {_fmt_stat(_sv.get('p5'))} / {_fmt_stat(_sv.get('p95'))}",
                    _GROUP_BREAK,
                    f"Zero fraction (% of valid): {_fmt_pct(None if _zero_frac is None else _zero_frac * 100)}",
                    f"Outliers (IQR, % of valid): {_fmt_count(outliers, (outliers / count_nonnull) * 100 if count_nonnull else None)}",
                    f"Outliers (MAD, % of valid): {_fmt_count(_out_mad.get('count'), _out_mad.get('pct'))}",
                    f"Outliers (Z): {_fmt_count(z_outliers, (z_outliers / count_nonnull) * 100 if count_nonnull else None)}",
                    _GROUP_BREAK,
                    f"Skewness: {skewness:.2f}",
                    f"Kurtosis: {kurt:.2f}",
                    f"W_Test: {w_test_stat:.2f}",
                    f"Normality Test: p-value={p_value_str} => {normality}",
                    f"Normality (D'Agostino) p-value: {_fmt_stat(_norm.get('dagostino_p'), 4)}"
                )

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

                # Structured v2 measures are computed first so the panel can
                # reuse them instead of recomputing.
                try:
                    structured[column] = _categorical_structured(df[column], comp, categories_mapping)
                    structured[column]['type'] = 'categorical'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured categorical stats for {column}: {str(e)}")
                _sv = structured.get(column, {})

                if value_counts.empty:
                    stats_text = tuple(_identity_lines(column, vars_details[column])) + (
                        f"Type: Categorical (encoded as {df[column].dtype})",
                        _GROUP_BREAK,
                        f"Number of Unique Categories: 0",
                        f"Missing Values (% of all rows): {_fmt_count(df[column].isnull().sum(), df[column].isnull().mean() * 100)}"
                    )
                else:
                    # Chi-square test
                    expected = total / len(value_counts)
                    chi_square_stat = ((value_counts - expected) ** 2 / expected).sum()

                    # Class balance, one panel entry per category: a single
                    # entry with embedded newlines is reflowed into one blob by
                    # _format_stats_lines. Percentages are given against the
                    # valid total and against all rows, because for a variable
                    # with heavy missingness those are very different claims.
                    # Only the first N categories are listed; the full
                    # distribution always goes to the structured JSON output.
                    _MAX_CLASS_BALANCE_LINES = 30
                    _cb_items = list(value_counts.items())
                    _cb_hidden = max(0, len(_cb_items) - _MAX_CLASS_BALANCE_LINES)
                    class_balance_lines = []
                    _cb_flat = []
                    for key, count in _cb_items[:_MAX_CLASS_BALANCE_LINES]:
                        _mapped = categories_mapping.get(str(key))
                        if _mapped and str(key) != str(_mapped):
                            _cat_label = f"{key} ({_mapped})"
                        else:
                            _cat_label = f"{key}"
                        _pct_valid = (count / total * 100) if total else None
                        _pct_all = (count / n_rows_total * 100) if n_rows_total else None
                        _cb_entry = f"{_cat_label} -> {int(count)} ({_fmt_pct_dual(_pct_valid, _pct_all)})"
                        class_balance_lines.append(f"- {_cb_entry}")
                        _cb_flat.append(_cb_entry)
                    if _cb_hidden:
                        _cb_more = f"... and {_cb_hidden} more categories (see JSON output)"
                        class_balance_lines.append(f"- {_cb_more}")
                        _cb_flat.append(_cb_more)
                    # No colon anywhere in this value: the flat-JSON builder
                    # keeps only the text between the first and second colon.
                    class_balance_flat = "; ".join(_cb_flat)

                    stats_text = tuple(_identity_lines(column, vars_details[column])) + (
                        f"Type: Categorical (encoded as {df[column].dtype})",
                        _GROUP_BREAK,
                        f"Number of unique values/categories: {len(value_counts)}",
                        f"Most frequent category: {categories_mapping.get(str(value_counts.idxmax()), 'Unknown')} ",
                        _GROUP_BREAK,
                        f"Count of valid observations: {count_nonnull} of {n_rows_total} rows",
                        f"Count empty (blank cells, % of all rows): {_fmt_count(count_na, comp.get('pct_empty', 0))}",
                        f"Count coded missing (% of all rows): {_fmt_count(count_missing, comp.get('pct_coded_missing', 0))}",
                        f"Count missing total (% of all rows): {_fmt_count(count_missing_total, comp.get('pct_missing_total', 0))}",
                        f"Count invalid (% of all rows): {_fmt_count(count_invalid, comp.get('pct_invalid', 0))}",
                        f"Missing code(s) declared: {missing_codes_txt}",
                        _GROUP_BREAK,
                        f"Entropy (bits): {_fmt_stat(_sv.get('entropy_bits'), 3)}",
                        f"Normalized entropy: {_fmt_stat(_sv.get('normalized_entropy'), 3)}",
                        f"Gini impurity: {_fmt_stat(_sv.get('gini_impurity'), 3)}",
                        f"Imbalance ratio: {_fmt_stat(_sv.get('imbalance_ratio'))}",
                        f"Chi-Square Test Statistic: {chi_square_stat:.2f}",
                        _GROUP_BREAK,
                        "Class balance (n, % of valid / % of all rows)",
                    ) + tuple(class_balance_lines)

                # A bar chart with hundreds of categories draws one bar, one
                # rotated tick label and one annotation per category, and is
                # unreadable regardless of how long it takes to render.
                _MAX_CHARTED_CATEGORIES = 60
                if column in vars_to_graph and len(value_counts) > _MAX_CHARTED_CATEGORIES:
                    data_issues.append(
                        f"Column {column}: {len(value_counts)} distinct categories exceeds the "
                        f"charting limit of {_MAX_CHARTED_CATEGORIES}; chart skipped."
                    )
                elif column in vars_to_graph:
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
                # Structured v2 measures are computed first so the panel can
                # reuse them instead of recomputing.
                try:
                    structured[column] = _date_structured(df[column], comp)
                    structured[column]['type'] = 'date'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured date stats for {column}: {str(e)}")
                _sv = structured.get(column, {})
                # Full calendar dates are charted at month+year resolution, so
                # the panel says so rather than leaving the chart to explain it.
                if _has_day_granularity(pd.to_datetime(df[column], errors='coerce')):
                    _granularity_lines = [
                        f"Value granularity: full date (year, month, day)",
                        f"Chart note: {_MONTH_AGG_NOTE}",
                    ]
                else:
                    _granularity_lines = []
                stats_text = _identity_lines(column, vars_details[column]) + [
                        f"Type: Date (encoded as {df[column].dtype})",
                ] + _granularity_lines + [
                        _GROUP_BREAK,
                        f"Number of unique values: {len(value_counts)}",
                        f"Most frequent value: {str(value_counts.idxmax()).split('.')[0]}",
                        _GROUP_BREAK,
                        f"Count of valid observations: {count_nonnull} of {n_rows_total} rows",
                        f"Count empty (blank cells, % of all rows): {_fmt_count(count_na, comp.get('pct_empty', 0))}",
                        f"Count coded missing (% of all rows): {_fmt_count(count_missing, comp.get('pct_coded_missing', 0))}",
                        f"Count missing total (% of all rows): {_fmt_count(count_missing_total, comp.get('pct_missing_total', 0))}",
                        f"Count invalid (% of all rows): {_fmt_count(count_invalid, comp.get('pct_invalid', 0))}",
                        f"Missing code(s) declared: {missing_codes_txt}",
                        _GROUP_BREAK,
                        f"Mean: {stats['mean'].date()}",
                        f"Median: {stats['50%'].date()}",
                        f"Min: {stats['min'].date()}",
                        f"Max: {stats['max'].date()}",
                        f"Range: {stats['max'] - stats['min']}", 
                        f"Range (days): {_sv.get('range_days', 'N/A')}",
                        _GROUP_BREAK,
                        f"Q1: {stats['25%'].date()}",
                        f"Q3: {stats['75%'].date()}",
                        f"IQR: {stats['75%'] - stats['25%']}",
                        _GROUP_BREAK,
                        f"Dates in the future: {_sv.get('future_dates', 'N/A')}",
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
            # Group separators carry no key/value pair, so skip them here.
            stats_text_dict.update({i.split(":")[0].strip():i.split(":")[1].strip() for i in stats_text if ":" in i})
            if class_balance_flat:
                stats_text_dict['Class balance (n, % of valid / % of all rows)'] = class_balance_flat
            stats_text_dict['url'] = f"https://explorer.icare4cvd.eu/api/variable-graph/{cohort_id}/{column}"
            vars_stats[column] = stats_text_dict
        except Exception as e:
            data_issues.append(f"Failed to perform EDA on column {column}. Exception msg: {str(e)}")
        # Enclave stdout is buffered, so this must be flushed to be visible.
        _done += 1
        if _done % 25 == 0:
            print(f"[{_time.time() - _t0:7.1f}s] {_done}/{len(_all_cols)} processed (last: {column})", flush=True)

    for col, ticks  in graph_tick_data.items():
        vars_stats[col].update(ticks)
    return vars_stats, structured



def create_save_graph(df, varname, stats_text, vartype, category_mapping=None):
    # Drop any figure orphaned by an exception in a previous call: the caller
    # swallows plotting errors, so the close at the end of this function can be
    # skipped. With hundreds of variables those orphans would accumulate.
    plt.close('all')
    stats_lines = _format_stats_lines(stats_text)
    fig_height = _stats_figure_height(stats_lines)
    chart_title = f"Distribution of {varname.upper()} ({COHORT_ID})"
    if vartype == 'numerical':
        fig, axes = plt.subplots(1, 2, figsize=(12, fig_height))

        # Left: Summary stats card
        _render_stats_panel(axes[0], f"Summary Stats for {varname.upper()}", stats_lines)

        # Right: Histogram with a count-scaled KDE on the same axis, so the
        # curve height is comparable to the bars.
        col_vals = pd.to_numeric(df[varname], errors='coerce').dropna()
        n_unique = col_vals.nunique()
        hist_kwargs = {}
        discrete = False
        if n_unique > 1 and len(col_vals) and float((col_vals % 1 != 0).sum()) == 0 and n_unique <= 30:
            # Integer-valued with few distinct values: one bar per value instead
            # of arbitrary auto-bins that split integers unevenly.
            hist_kwargs['discrete'] = True
            discrete = True
        if n_unique > 1:
            # cut=0 stops the gaussian tails from stretching the x-axis past the
            # observed data range.
            hist_kwargs['kde'] = True
            hist_kwargs['kde_kws'] = {'cut': 0}
        sns.histplot(col_vals, ax=axes[1], **hist_kwargs)
        _panel_title(axes[1], chart_title)
        _clean_axis(axes[1])
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")

        if n_unique > 1:
            low, high = float(col_vals.min()), float(col_vals.max())
            pad = max(0.02 * (high - low), 0.6 if discrete else 0.0)
            axes[1].set_xlim(low - pad, high + pad)
        if len(col_vals):
            mean_val, median_val = float(col_vals.mean()), float(col_vals.median())
            axes[1].axvline(mean_val, color='firebrick', linestyle='--', linewidth=1.4,
                            label=f"Mean = {mean_val:,.2f}")
            axes[1].axvline(median_val, color='darkslategray', linestyle=':', linewidth=1.6,
                            label=f"Median = {median_val:,.2f}")
            axes[1].legend(fontsize=9, frameon=False)

        # Save the figure for the current feature
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png", dpi=100)
        #print(f"figure for {varname} saved!! ")
    elif vartype == 'datetime':
        fig, axes = plt.subplots(1, 2, figsize=(12, fig_height))

        _render_stats_panel(axes[0], f"Summary Stats for {varname.upper()}", stats_lines)
        try:
            date_vals = pd.to_datetime(df[varname].dropna(), errors='coerce').dropna()
        except:
            print("supposed date column could not be parsed: ", varname)
            plt.close('all')
            return {}
    
        # Full calendar dates are never charted at day resolution: they are
        # truncated to the first of their month so a bar can only ever say
        # "month + year".
        month_aggregated = _has_day_granularity(date_vals)
        if month_aggregated:
            date_vals = date_vals.dt.to_period('M').dt.to_timestamp()

        if date_vals.empty:
            print("date column has no parseable values: ", varname)
            plt.close('all')
            return {}

        date_nums = mdates.date2num(date_vals)

        min_date = date_vals.min()
        max_date = date_vals.max()
        date_range = max_date - min_date

        if date_range.days > 365 * 10:
            bin_period = 'Y'  # Yearly
            axes[1].xaxis.set_major_locator(mdates.YearLocator(base=2))
        elif date_range.days > 365 * 5:
            bin_period = 'Y'  # Yearly
            axes[1].xaxis.set_major_locator(mdates.YearLocator())
        elif date_range.days > 365:
            bin_period = 'Q'  # Quarterly
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif date_range.days > 90:
            bin_period = 'M'
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())  # Monthly
        elif date_range.days > 30 and not month_aggregated:
            bin_period = 'W'  # Weekly bins
            axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Weekly
        elif month_aggregated:
            # One bar per month, never per day.
            bin_period = 'M'
            axes[1].xaxis.set_major_locator(mdates.MonthLocator())
        else:
            bin_period = 'D'  # Daily
            axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))

        # Bin on period boundaries: a plain date_range anchored on min_date rolls
        # *forward* to the next year/quarter/week start, which silently drops
        # every observation before that first edge. The trailing edge is needed
        # too, since histogram bins are half-open and the last period would
        # otherwise be counted inside the previous bar.
        bin_periods = pd.period_range(min_date, max_date, freq=bin_period)
        bin_edges = bin_periods.to_timestamp().append(
            pd.DatetimeIndex([(bin_periods[-1] + 1).to_timestamp()])
        )
        bins = mdates.date2num(bin_edges)

        axes[1].hist(date_nums, bins=bins, alpha=0.7)
        _panel_title(axes[1], f"{chart_title} ({_MONTH_AGG_NOTE})" if month_aggregated else chart_title)
        _clean_axis(axes[1])
        axes[1].set_xlabel("Month" if month_aggregated else "Date")
        axes[1].set_ylabel("Count")

        if date_range.days <= 90 and not month_aggregated:
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        elif date_range.days <= 365 * 2:
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # "2020-01" format
        else:
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].tick_params(axis='x', which='minor', bottom=False)
        
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png", dpi=100)
        #print(f"figure for {varname} saved!! ")


    elif vartype == 'categorical':

        # Valid observations only, matching the stats panel: blanks and
        # coded-missing values are reported there as counts rather than drawn
        # as a category, so the percentages below share the panel's denominator.
        value_counts = df[varname].apply(_lowercase_if_string).value_counts(dropna=True)
        total = int(value_counts.sum())
        fig, axes = plt.subplots(1, 2, figsize=(12, fig_height))

        _render_stats_panel(axes[0], f"Summary Stats for {varname.upper()}", stats_lines)

        # Bar chart
        if not value_counts.empty:
            colors = sns.color_palette("husl", len(value_counts))
            ax = value_counts.plot(kind='bar', color=colors, edgecolor='black', ax=axes[1])
            _panel_title(axes[1], chart_title)
            _clean_axis(axes[1])
            ax.set_xlabel(f"Categories (n={total} valid; percentages are of valid)")
            ax.set_ylabel("Count")

            # Add labels to the bars
            if len(value_counts)>4:
                rot = 90
            else:
                rot = 0
            for idx, value in enumerate(value_counts):
                percentage = (value / total) * 100 if total else None
                ax.text(idx, value + total * 0.02, f"{value} ({_fmt_pct(percentage)})",
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

        
        if not value_counts.empty:
            axes[1].set_ylim(0, max(value_counts.values) * 1.4)
        plt.tight_layout()
        plt.savefig(f"/output/{varname.lower()}.png", dpi=100)

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
    return (raw_script.replace("{shared_helpers}", _SHARED_HELPERS)
            .replace("{chart_style}", _CHART_STYLE_HELPERS)
            .replace("{cohort_id}", cohort_id))


def longitudinal_analysis(cohort_id: str) -> str:
    raw_script = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import re
import json
import decentriq_util
import zipfile
import tempfile
import os
import warnings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
warnings.filterwarnings('ignore')
{shared_helpers}

# ---------------------------------------------------------------------------
# longitudinal_analysis
#
# Identifies "longitudinal families": sets of variables in the cohort that
# measure the same standardised concept (same VARIABLE OMOP ID, or VARIABLE
# CONCEPT CODE when no OMOP ID is available) at different visits/timepoints
# (distinct VISIT CONCEPT NAME / VISITS values). For each family it charts
# every patient's trajectory without ever displaying patient identifiers:
#   - numeric families: patients are grouped into a small number of "bands"
#     by the *shape* of their trajectory (baseline-relative, so patients with
#     very different absolute ranges can still land in the same band), then
#     each band is drawn as a median line + IQR ribbon. Every patient
#     contributes to exactly one band.
#   - categorical families: an alluvial (flow) diagram shows how patients
#     move between categories across consecutive visits.
# ---------------------------------------------------------------------------

data_issues = []
try:
    with open("/input/c2_save_to_json/data_issues.json") as f:
        data_issues = json.load(f)
except Exception:
    pass

vars_details = pd.read_json("/input/c2_save_to_json/variable_details.json")

load_notes = []
raw_data, load_notes = load_data_as_text("/input/{cohort_id}", load_notes)
raw_data.columns = [str(c).lower().strip() for c in raw_data.columns]
for _note in load_notes:
    data_issues.append(_note)

data = pd.DataFrame(index=raw_data.index)
var_meta = {}
for v in list(vars_details.columns):
    if v not in raw_data.columns:
        continue
    d = vars_details[v]
    typ = d['inferred_type']
    try:
        comp, valid = classify_series(raw_data[v], _missing_codes_of(d), typ)
    except Exception as e:
        data_issues.append(f"c4: failed to classify column {v}: {e}")
        continue
    data[v] = valid.reindex(raw_data.index)

    def _dget(key):
        try:
            val = d.get(key)
        except Exception:
            val = None
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        s = str(val).strip()
        return s if s else None

    var_meta[v] = {
        'var_type': typ,
        'var_label': _dget('var_label') or v,
        'omop_id': _dget('omop_id'),
        'concept_code': _dget('concept_code'),
        'additional_context': _dget('additional_context'),
        'units': _dget('units'),
        'visits': _dget('visits'),
        'visit_concept_name': _dget('visit_concept_name'),
        'categories': (d.get('categories') if typ == 'categorical' else None),
    }

# ---- Identify the patient-id column (never charted/shown) -----------------
_PATIENT_OMOP_ID = '4086934'
_PATIENT_CONCEPT_CODE = 'snomed:184107009'
patient_id_col = None
for v, m in var_meta.items():
    if (m['omop_id'] == _PATIENT_OMOP_ID or
            (m['concept_code'] and m['concept_code'].lower() == _PATIENT_CONCEPT_CODE)):
        patient_id_col = v
        break

# ---- Identify longitudinal families ----------------------------------------
def _visit_label(m):
    return m['visit_concept_name'] or m['visits']

_BASELINE_RE = re.compile(r'(baseline|screening|enrol{1,2}ment|day\\s*0\\b|week\\s*0\\b|visit\\s*0\\b)', re.I)
_END_RE = re.compile(r'(end\\s*of\\s*(study|trial)|\\bfinal\\b|\\blast\\b|\\beos\\b|study\\s*end|follow[- ]?up\\s*end)', re.I)
_TOKEN_RE = re.compile(r'[a-z]+|\\d+(?:\\.\\d+)?')

# Days per time unit, so "1 year" can be compared against "3 months" instead of
# comparing the bare numbers 1 and 3.
_VISIT_UNIT_DAYS = {
    'd': 1.0, 'day': 1.0, 'days': 1.0,
    'w': 7.0, 'wk': 7.0, 'wks': 7.0, 'week': 7.0, 'weeks': 7.0,
    'm': 30.44, 'mo': 30.44, 'mos': 30.44, 'mon': 30.44, 'month': 30.44, 'months': 30.44,
    'y': 365.25, 'yr': 365.25, 'yrs': 365.25, 'year': 365.25, 'years': 365.25,
}

def _visit_offset_days(text):
    # Returns the visit offset in days, or None when the label carries no
    # number/unit pair. The unit may sit on either side of the number
    # ("3 months", "month 3").
    tokens = _TOKEN_RE.findall(text)
    for i, tok in enumerate(tokens):
        if not tok[0].isdigit():
            continue
        for neighbour in (i + 1, i - 1):
            if 0 <= neighbour < len(tokens) and tokens[neighbour] in _VISIT_UNIT_DAYS:
                return float(tok) * _VISIT_UNIT_DAYS[tokens[neighbour]]
    return None

def _visit_ordinal(text):
    # "visit 2" / "v3" style labels have an order but no duration.
    tokens = _TOKEN_RE.findall(text)
    for tok in tokens:
        if tok[0].isdigit():
            return float(tok)
    return None

def _visit_sort_key(label):
    # Tiers keep unlike labelling schemes from interleaving: baseline first,
    # then real durations, then bare ordinals, then unrecognised, then end-of-study.
    if not label:
        return (3.0, 0.0, '')
    s = str(label).strip().lower()
    if _BASELINE_RE.search(s):
        return (0.0, 0.0, s)
    if _END_RE.search(s):
        return (4.0, 0.0, s)
    offset = _visit_offset_days(s)
    if offset is not None:
        return (1.0, offset, s)
    ordinal = _visit_ordinal(s)
    if ordinal is not None:
        return (2.0, ordinal, s)
    return (3.0, 0.0, s)

# Visit boilerplate and time words: what is left after removing these from the
# member variable names is the concept the family measures.
_LABEL_STOPWORDS = {
    'visit', 'visits', 'at', 'on', 'in', 'of', 'the', 'for', 'to', 'from', 'by',
    'baseline', 'base', 'screening', 'enrolment', 'enrollment', 'entry',
    'follow', 'followup', 'fup', 'fu', 'up', 'end', 'study', 'trial', 'final',
    'last', 'first', 'time', 'timepoint', 'point', 'v', 'val', 'value',
    'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
    'd', 'w', 'wk', 'wks', 'mo', 'mos', 'mon', 'y', 'yr', 'yrs',
}

def _concept_tokens(name):
    tokens = []
    for raw in re.split(r'[^a-z0-9]+', str(name).lower()):
        if not raw or raw[0].isdigit():
            continue
        # weight1 / weight2 are one concept at two visits, so a trailing visit
        # index is not part of the name. Kept when stripping would leave almost
        # nothing behind, so that names like "o2" survive intact.
        stripped = raw.rstrip('0123456789')
        token = stripped if len(stripped) >= 2 else raw
        if token not in _LABEL_STOPWORDS:
            tokens.append(token)
    return tokens

def _family_label(members, fallback):
    # Members of a family are named to a pattern (blood_pressure_on_visit_1,
    # blood_pressure_at_end_of_study), so the tokens they share name the
    # concept. A common prefix is the usual case; otherwise fall back to the
    # tokens common to every member, in the order the first member uses them.
    token_lists = [_concept_tokens(v) for v in members]
    token_lists = [t for t in token_lists if t]
    if not token_lists:
        return fallback
    shared = []
    for group in zip(*token_lists):
        if len(set(group)) == 1:
            shared.append(group[0])
        else:
            break
    if not shared:
        common = set(token_lists[0])
        for tokens in token_lists[1:]:
            common &= set(tokens)
        for tok in token_lists[0]:
            if tok in common and tok not in shared:
                shared.append(tok)
    label = ' '.join(shared).strip()
    return label or fallback

def _context_key(m):
    # Same concept measured with a different additional context is a different
    # thing (e.g. "systolic" vs "diastolic" context on one BP concept), so the
    # context joins the grouping key. Order within a pipe-list is not meaningful.
    context = m['additional_context']
    if not context:
        return 'no-additional-context'
    parts = sorted(p.strip().lower() for p in str(context).split('|') if p.strip())
    return '|'.join(parts) or 'no-additional-context'

groups = {}
for v, m in var_meta.items():
    if v == patient_id_col:
        continue
    concept = m['omop_id'] or m['concept_code']
    if not concept or not _visit_label(m):
        continue
    groups.setdefault((str(concept), _context_key(m)), []).append(v)

families = []
for (concept, context), members in groups.items():
    key = f"{concept} / {context}"
    if len(members) < 2:
        continue
    labels = set(_visit_label(var_meta[v]) for v in members)
    if len(labels) < 2:
        continue
    type_counts = {}
    for v in members:
        t = var_meta[v]['var_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    fam_type = max(type_counts, key=type_counts.get)
    if len(type_counts) > 1:
        data_issues.append(
            f"Longitudinal family {key}: member variables disagree on inferred type "
            f"({type_counts}); treating the whole family as '{fam_type}' and skipping "
            f"members of a different type."
        )
    members = [v for v in members if var_meta[v]['var_type'] == fam_type]
    members = sorted(members, key=lambda v: _visit_sort_key(_visit_label(var_meta[v])))
    units = next((var_meta[v]['units'] for v in members if var_meta[v]['units']), None)
    families.append({
        'key': key,
        'concept': str(concept),
        'context': context,
        'var_type': fam_type,
        'var_label': _family_label(members, var_meta[members[0]]['var_label']),
        'units': units,
        'members': members,
        'visit_labels': [_visit_label(var_meta[v]) for v in members],
    })

# ---- Numeric family: baseline-relative trajectory banding -----------------
def _process_numeric_family(fam):
    fam_df = data[fam['members']].apply(pd.to_numeric, errors='coerce')
    fam_df.columns = fam['visit_labels']
    n_visits = fam_df.shape[1]
    if n_visits < 2:
        return None
    baseline = fam_df.iloc[:, 0]
    usable = baseline.notna() & (fam_df.notna().sum(axis=1) >= 2)
    sub = fam_df[usable].reset_index(drop=True)
    n_patients = int(len(sub))
    if n_patients < 5:
        data_issues.append(f"Longitudinal family {fam['key']}: fewer than 5 usable patient trajectories, skipping chart.")
        return None

    baseline_vals = sub.iloc[:, 0]
    denom = baseline_vals.abs().replace(0, np.nan)
    norm = sub.sub(baseline_vals, axis=0)
    norm_pct = norm.div(denom, axis=0)
    norm_pct = norm_pct.where(~norm_pct.isna(), norm)
    mat = norm_pct.to_numpy(dtype=float)

    MAX_FIT = 600
    if n_patients > MAX_FIT:
        rng = np.random.RandomState(42)
        fit_idx = rng.choice(n_patients, MAX_FIT, replace=False)
    else:
        fit_idx = np.arange(n_patients)
    fit_mat = mat[fit_idx]
    nf = len(fit_idx)

    def _nan_rmse(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if not mask.any():
            return 1.0
        return float(np.sqrt(np.nanmean((a[mask] - b[mask]) ** 2)))

    if nf < 4:
        fit_labels = np.ones(nf, dtype=int)
    else:
        dist = np.zeros((nf, nf))
        for i in range(nf):
            for j in range(i + 1, nf):
                dd = _nan_rmse(fit_mat[i], fit_mat[j])
                dist[i, j] = dist[j, i] = dd
        k = max(2, min(6, round((nf / 2) ** 0.5)))
        k = min(k, max(1, nf - 1))
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        fit_labels = fcluster(Z, k, criterion='maxclust')

    centroids = {b: np.nanmean(fit_mat[fit_labels == b], axis=0) for b in set(fit_labels)}
    labels = np.array([min(centroids, key=lambda b: _nan_rmse(mat[i], centroids[b])) for i in range(n_patients)])

    bands = []
    for b in sorted(set(labels)):
        band_mask = labels == b
        band_df = sub[band_mask]
        band_norm = norm_pct[band_mask]
        end_col = band_norm.iloc[:, -1]
        end_shift = float(end_col.mean(skipna=True)) if end_col.notna().any() else 0.0
        if end_shift > 0.05:
            trend_label = 'rising'
        elif end_shift < -0.05:
            trend_label = 'declining'
        else:
            trend_label = 'stable'
        bands.append({
            'band_label': trend_label,
            'end_shift': None if pd.isna(end_shift) else float(end_shift),
            'n_patients': int(band_mask.sum()),
            'median': [None if pd.isna(v) else float(v) for v in band_df.median(axis=0, skipna=True)],
            'q1': [None if pd.isna(v) else float(v) for v in band_df.quantile(0.25, axis=0)],
            'q3': [None if pd.isna(v) else float(v) for v in band_df.quantile(0.75, axis=0)],
        })
    return {'n_patients': n_patients, 'visit_labels': fam['visit_labels'], 'bands': bands}

# Trend direction is carried by colour, so a legend entry can be read without
# matching it back to an arbitrary palette index.
_TREND_COLORS = {'declining': 'tab:blue', 'stable': 'tab:gray', 'rising': 'tab:red'}
_TREND_ORDER = {'declining': 0, 'stable': 1, 'rising': 2}
# Bands this small are individual or near-individual trajectories: they are the
# outliers we want to see, so they are drawn on top and named as such.
_OUTLIER_BAND_MAX = 3

def _plot_numeric_family(ax, result, var_label, units=None):
    x = np.arange(len(result['visit_labels']))
    bands_sorted = sorted(
        result['bands'],
        key=lambda b: (_TREND_ORDER.get(b['band_label'], 1),
                       -abs(b.get('end_shift') or 0.0)),
    )
    # Several bands can share a trend; a dash pattern plus the magnitude in the
    # label keeps them apart.
    styles = ['-', '--', ':', '-.']
    seen_trend = {}
    for band in bands_sorted:
        trend = band['band_label']
        rank = seen_trend.get(trend, 0)
        seen_trend[trend] = rank + 1
        color = _TREND_COLORS.get(trend, 'tab:purple')
        median = np.array([np.nan if v is None else v for v in band['median']], dtype=float)
        q1 = np.array([np.nan if v is None else v for v in band['q1']], dtype=float)
        q3 = np.array([np.nan if v is None else v for v in band['q3']], dtype=float)
        n = band['n_patients']
        shift = band.get('end_shift')
        parts = [trend.capitalize()]
        if shift is not None:
            parts.append(f"{shift * 100:+.0f}%")
        parts.append(f"(n={n}" + (", outlier)" if n <= _OUTLIER_BAND_MAX else ")"))
        outlier = n <= _OUTLIER_BAND_MAX
        ax.plot(x, median, color=color, linestyle=styles[rank % len(styles)],
                linewidth=1.6 if outlier else 2.2,
                marker='D' if outlier else 'o', markersize=5 if outlier else 4,
                label=' '.join(parts), zorder=5 if outlier else 3)
        # An interquartile ribbon needs a distribution behind it; for a handful
        # of patients q1 and q3 collapse onto the median and only add clutter.
        if not outlier:
            ax.fill_between(x, q1, q3, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(result['visit_labels'], rotation=45 if len(x) > 4 else 0,
                        ha='right' if len(x) > 4 else 'center')
    ax.set_title(f"Longitudinal Trend - {var_label.upper()}")
    ax.set_xlabel("Visit")
    ax.set_ylabel(f"Value ({units})" if units else "Value")
    ax.legend(frameon=False, fontsize=8, loc='best',
              title=f"n={result['n_patients']} patients", title_fontsize=8)

# ---- Categorical family: alluvial (flow) diagram ---------------------------
def _draw_ribbon(ax, x0, x1, y0s, y1s, y0t, y1t, color):
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0s),
        (x0, y1s),
        (xm, y1s), (xm, y1t), (x1, y1t),
        (x1, y0t),
        (xm, y0t), (xm, y0s), (x0, y0s),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    patch = mpatches.PathPatch(MplPath(verts, codes), facecolor=color, edgecolor='none', alpha=0.45, zorder=2)
    ax.add_patch(patch)

def _process_categorical_family(fam):
    fam_df = data[fam['members']].copy()
    fam_df.columns = fam['visit_labels']
    n_patients = int(len(fam_df))
    if n_patients < 5:
        data_issues.append(f"Longitudinal family {fam['key']}: fewer than 5 patients, skipping chart.")
        return None
    label_maps = [var_meta[v]['categories'] or {} for v in fam['members']]
    return {'fam_df': fam_df, 'n_patients': n_patients, 'label_maps': label_maps}

def _plot_categorical_family(ax, processed, var_label):
    fam_df = processed['fam_df']
    label_maps = processed['label_maps']
    visits = list(fam_df.columns)
    n_v = len(visits)
    n_patients = processed['n_patients']

    def _lab(vi, val):
        if pd.isna(val):
            return 'N/A'
        return label_maps[vi].get(str(val), str(val))

    cols = [fam_df.iloc[:, vi].map(lambda val, vi=vi: _lab(vi, val)) for vi in range(n_v)]
    all_cats = []
    for c in cols:
        for cat in c.unique():
            if cat not in all_cats:
                all_cats.append(cat)
    # N/A sits at the bottom of every column so that loss to follow-up reads as
    # a floor rather than sweeping its ribbons across the whole diagram.
    all_cats = sorted(set(all_cats), key=lambda c: (c != 'N/A', c))
    non_na = [c for c in all_cats if c != 'N/A']
    _default_colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    color_map = {cat: _default_colors[i % len(_default_colors)] for i, cat in enumerate(non_na)}
    color_map['N/A'] = '#CBD5E1'

    node_gap = max(1.0, 0.02 * n_patients)
    x_pos = np.linspace(0, 1, n_v) if n_v > 1 else np.array([0.5])

    # Lay every column out first: the height of the tallest column decides
    # which nodes are too thin to hold their label inline.
    node_rects = {}
    column_counts = []
    max_height = 0.0
    for vi, c in enumerate(cols):
        counts = c.value_counts()
        counts = counts.reindex([cat for cat in all_cats if cat in counts.index])
        column_counts.append(counts)
        y = 0.0
        for cat, cnt in counts.items():
            node_rects[(vi, cat)] = (y, y + cnt)
            y = y + cnt + node_gap
        max_height = max(max_height, y)

    for vi, counts in enumerate(column_counts):
        for cat, cnt in counts.items():
            y0, y1 = node_rects[(vi, cat)]
            ax.add_patch(mpatches.Rectangle((x_pos[vi] - 0.012, y0), 0.024, y1 - y0,
                                             facecolor=color_map[cat], edgecolor='white',
                                             linewidth=0.5, zorder=3))
            # Small categories are often the interesting ones, so they keep
            # their label; it just moves beside the node when it cannot fit.
            if (y1 - y0) >= max_height * 0.045:
                ax.text(x_pos[vi], (y0 + y1) / 2, f"{cat}\\n{int(cnt)}", ha='center', va='center',
                        fontsize=8, zorder=4)
            else:
                to_left = vi == n_v - 1
                ax.annotate(f"{cat} {int(cnt)}", xy=(x_pos[vi], (y0 + y1) / 2),
                            xytext=(-7 if to_left else 7, 0), textcoords='offset points',
                            ha='right' if to_left else 'left', va='center',
                            fontsize=7, zorder=6,
                            bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                                      edgecolor='none', alpha=0.75))

    for vi in range(n_v - 1):
        trans = pd.crosstab(cols[vi], cols[vi + 1])
        y_off_src = {cat: node_rects[(vi, cat)][0] for cat in all_cats if (vi, cat) in node_rects}
        y_off_tgt = {cat: node_rects[(vi + 1, cat)][0] for cat in all_cats if (vi + 1, cat) in node_rects}
        for src in trans.index:
            for tgt in trans.columns:
                cnt = trans.loc[src, tgt]
                if cnt <= 0:
                    continue
                y0s = y_off_src[src]; y1s = y0s + cnt; y_off_src[src] = y1s
                y0t = y_off_tgt[tgt]; y1t = y0t + cnt; y_off_tgt[tgt] = y1t
                _draw_ribbon(ax, x_pos[vi] + 0.012, x_pos[vi + 1] - 0.012, y0s, y1s, y0t, y1t, color_map[src])

    # Room for the labels on the first and last columns, which are centred on
    # the node and would otherwise be cut off at the axes edge.
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-max_height * 0.02, max_height * 1.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(visits, rotation=45 if n_v > 4 else 0, ha='right' if n_v > 4 else 'center')
    ax.set_yticks([])
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(
        f"n={n_patients} patients - N/A = no recorded value at that visit "
        f"(missed visit or withdrawal)", fontsize=8)
    ax.set_title(f"Category Flow - {var_label.upper()}")

# ---- Run over every identified family --------------------------------------
longitudinal_json = {}
for fam in families:
    name_slug = re.sub(r'[^a-z0-9]+', '_', str(fam['var_label']).lower()).strip('_')
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        if fam['var_type'] in ('int', 'float'):
            result = _process_numeric_family(fam)
            if result is None:
                plt.close(fig)
                continue
            _plot_numeric_family(ax, result, fam['var_label'], fam['units'])
            chart_suffix = 'longitudinal-trend'
            longitudinal_json[fam['key']] = {
                'var_type': fam['var_type'],
                'var_label': fam['var_label'],
                'concept': fam['concept'],
                'additional_context': fam['context'],
                'units': fam['units'],
                'member_variables': fam['members'],
                'chart_type': 'trajectory_bands',
                **result,
            }
        elif fam['var_type'] == 'categorical':
            processed = _process_categorical_family(fam)
            if processed is None:
                plt.close(fig)
                continue
            _plot_categorical_family(ax, processed, fam['var_label'])
            chart_suffix = 'category_flow'
            longitudinal_json[fam['key']] = {
                'var_type': fam['var_type'],
                'var_label': fam['var_label'],
                'concept': fam['concept'],
                'additional_context': fam['context'],
                'member_variables': fam['members'],
                'visit_labels': fam['visit_labels'],
                'chart_type': 'alluvial',
                'n_patients': processed['n_patients'],
            }
        else:
            plt.close(fig)
            continue
        plt.tight_layout()
        plt.savefig(f"/output/{name_slug}_{chart_suffix}.png", dpi=160, bbox_inches='tight')
    except Exception as e:
        data_issues.append(f"Longitudinal family {fam['key']}: failed to build chart: {e}")
    finally:
        plt.close('all')

with open('/output/eda_longitudinal_v1_{cohort_id}.json', 'w') as f:
    json.dump(longitudinal_json, f, indent=4)

with open('/output/data_issues.json', 'w') as f:
    json.dump(data_issues, f, indent=4)
"""
    return (raw_script.replace("{shared_helpers}", _SHARED_HELPERS)
            .replace("{cohort_id}", cohort_id))


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
