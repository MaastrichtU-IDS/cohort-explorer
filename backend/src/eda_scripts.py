"""Exploratory Data Analysis (EDA) module."""

import json as _json


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

def _missing_code_variants(p):
    # Some dictionaries write numeric missing codes with a space (or comma) where
    # a decimal or thousands separator belongs, e.g. "9999 000" for 9999.000 or
    # 9999000. The literal token "9999 000" never matches a numeric cell, so we
    # register every plausible reading:
    #   - the combined decimal form  (9999.000)
    #   - the combined integer form  (9999000)
    #   - the head as a standalone code (9999)
    # The tail is deliberately NOT registered as a standalone code: a tail such
    # as "000" normalises to 0 and would silently classify every genuine zero
    # in the data as coded-missing.
    m = _re.match(r'^([+-]?[0-9]+)[ ,]([0-9]{3})$', p)
    if not m:
        return [p]
    head, tail = m.group(1), m.group(2)
    out = []
    for v in (head + '.' + tail, head + tail, head):
        if v not in out:
            out.append(v)
    return out

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
        if p == '':
            continue
        for code in _missing_code_variants(p):
            if code not in out:
                out.append(code)
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

def _fmt_num(value, decimals=2):
    # Plain fixed-point rendering with NO thousands separators: these values
    # end up in the flat JSON output, whose consumers parse them back into
    # numbers. None/NaN render as 'N/A'.
    if value is None:
        return 'N/A'
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num != num:
        return 'N/A'
    return f"{num:.{decimals}f}"

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
    elif cands.nunique() <= 20:
        #few distinct values, none of the clean types matched: categorical
        print("The following variable deemed categorical by process of elimination: ", variable_name)
        t = 'categorical'
    else:
        # High-cardinality mixed content. The clean checks above require EVERY
        # candidate to parse, so one stray annotation ("n/a", "12,5", "<40")
        # used to push the whole column into "assume float", marking 100% of a
        # genuinely textual column invalid. Decide by parse RATE instead: a
        # column that is mostly numbers is numeric with a few invalid cells; a
        # column that is mostly words is free text.
        _num_rate = float(pd.to_numeric(cands, errors='coerce').notna().mean())
        try:
            _date_rate = float(pd.to_datetime(cands, errors='coerce').notna().mean())
        except Exception:
            _date_rate = 0.0
        if _num_rate >= 0.5 and _num_rate >= _date_rate:
            t = 'float'
        elif _date_rate >= 0.5:
            t = 'date'
        else:
            t = 'text'
        msg = ("Mixed-content variable %s: %.0f%% of non-missing values parse as numeric, "
               "%.0f%% as dates -> inferred '%s'" % (variable_name, _num_rate * 100, _date_rate * 100, t))
        print(msg)
        if t == 'text':
            mismatched_types[variable_name + "-text"] = msg

    vars_to_process[variable_name] = t

    #find the mismatches between declared types (in data dictionary) and inferred types:
    if ((var_type.lower() == "datetime" and t != "date") or
        (var_type.lower() == "str" and t not in ("categorical", "text"))):
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
            'concept_name': _dict_val('VARIABLE CONCEPT NAME'),
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



def c3_eda_data_profiling(cohort_id: str, stratifier_config: dict | None = None) -> str:
    """Generate the c3 profiling script.

    stratifier_config, chosen by the DCR creator at provision time:
      {"excluded_defaults": ["sex" | "age", ...],   # default stratifiers to skip
       "custom_variables": ["varname", ...]}        # extra stratifier columns
    """
    _cfg = {"excluded_defaults": [], "custom_variables": []}
    for _k in list(_cfg.keys()):
        if isinstance(stratifier_config, dict) and isinstance(stratifier_config.get(_k), list):
            _cfg[_k] = [str(x) for x in stratifier_config[_k]]
    raw_script = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, zscore, normaltest, median_abs_deviation, trim_mean, entropy, chi2
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, spearmanr, pearsonr, chi2_contingency
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

# Stratified-analysis configuration, baked in at DCR provision time. The DCR
# creator can opt out of default stratifiers and/or add custom ones by name.
_STRATIFIER_CONFIG = {stratifier_config}

# Default stratifiers are recognised by standardised concept, never by column
# name (the same variable is called sex/DEM_SEX/Gender/GESCHLECHT across
# cohorts, and codings even flip: 0=male in one cohort, 1=male in another).
_DEFAULT_STRATIFIERS = {
    'sex': {'omop_ids': ['46235213'], 'kind': 'categorical'},   # LOINC 76689-9 "Sex assigned at birth"
    'age': {'omop_ids': ['3022304'], 'kind': 'numeric'},        # LOINC 30525-0 "Age"
}

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
_PATIENT_OMOP_IDS = ('4086934', '40757164')
_PATIENT_CONCEPT_CODE = '184107009'

def _is_patient_id(details):
    omop_id = _meta_value(details, 'omop_id')
    if omop_id.endswith('.0'):
        omop_id = omop_id[:-2]
    if omop_id in _PATIENT_OMOP_IDS:
        return True
    concept_code = _meta_value(details, 'concept_code').strip().lower()
    if _PATIENT_CONCEPT_CODE in concept_code:
        return True
    # Last fallback: a patient-id OMOP concept written into the concept code
    # itself (e.g. "omop:4086934"), when neither of the two checks above hit.
    if concept_code.endswith('.0'):
        concept_code = concept_code[:-2]
    return any(concept_code == pid or concept_code == 'omop:' + pid for pid in _PATIENT_OMOP_IDS)

patient_id_cols = set()
for _v in list(vars_details.columns):
    if _is_patient_id(vars_details[_v]):
        patient_id_cols.add(str(_v).strip().lower())
        data_issues.append(
            f"Variable {_v} is a patient identifier (SNOMED:{_PATIENT_CONCEPT_CODE} / "
            f"OMOP {'/'.join(_PATIENT_OMOP_IDS)}); excluded from profiling."
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
    try:
        _modes = vals.mode()
        out['mode'] = _to_native(float(_modes.iloc[0])) if len(_modes) else None
    except Exception:
        out['mode'] = None
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
    # Classic z-score rule (sample std, ddof=1); complements IQR and MAD above.
    if std and std > 0:
        oz = int((((vals - mean) / std).abs() > 3).sum())
    else:
        oz = 0
    out['outliers_z'] = {'count': oz, 'pct': _to_native(oz / n * 100), 'threshold': 3}
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
            norm['shapiro_w'] = _to_native(float(_w))
            norm['shapiro_p'] = _to_native(float(pw))
            norm['shapiro_n'] = int(len(sub))
        except Exception:
            pass
    # Fall back to Shapiro when D'Agostino is absent OR present-but-None
    # (a NaN p-value is stored as None).
    _pv = norm.get('dagostino_p')
    if _pv is None:
        _pv = norm.get('shapiro_p')
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
    s_norm = series.apply(_lowercase_if_string)
    # Case variants of one token ("Male"/"MALE") are a single category, so the
    # unique count is taken on the same normalised values the distribution
    # below is built from — otherwise the two disagree.
    out['n_unique'] = int(s_norm.nunique(dropna=True))
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
        # Goodness-of-fit against a uniform distribution over the observed
        # categories (H0: all categories equally frequent). The raw statistic
        # alone scales with n and category count, so dof and p-value are
        # reported with it; `valid` is False when the expected count per
        # category is below 5, where the chi-square approximation is unreliable.
        if len(vc_valid) > 1:
            expected = tot_valid / len(vc_valid)
            stat = float((((vc_valid - expected) ** 2) / expected).sum())
            dof = int(len(vc_valid) - 1)
            try:
                pval = _to_native(float(chi2.sf(stat, dof)))
            except Exception:
                pval = None
            out['chi_square_uniform'] = {
                'statistic': _to_native(stat),
                'dof': dof,
                'p_value': pval,
                'expected_count_per_category': _to_native(expected),
                'valid': bool(expected >= 5),
            }
    return out

def _date_structured(series, comp):
    out = {}
    d = pd.to_datetime(series, errors='coerce').dropna()
    out['n'] = int(len(d))
    out['n_unique'] = int(d.nunique())
    out['completeness'] = comp
    if len(d) == 0:
        return out
    # Summary statistics keep full day-level detail (min/max/quartiles are
    # exact dates); only the CHART aggregates full dates to month+year. The
    # granularity field records which kind of values the column holds.
    out['granularity'] = 'day' if _has_day_granularity(d) else 'month or coarser'
    _fmt_date = lambda ts: str(ts.date())
    out['min'] = _fmt_date(d.min())
    out['max'] = _fmt_date(d.max())
    out['mean'] = _fmt_date(d.mean())
    out['range_days'] = int((d.max() - d.min()).days)
    try:
        q = d.quantile([0.25, 0.5, 0.75])
        out['q1'] = _fmt_date(q.loc[0.25])
        out['median'] = _fmt_date(q.loc[0.5])
        out['q3'] = _fmt_date(q.loc[0.75])
        out['iqr_days'] = int((q.loc[0.75] - q.loc[0.25]).days)
    except Exception:
        pass
    try:
        _vc = d.value_counts()
        out['most_frequent'] = {'value': _fmt_date(_vc.idxmax()), 'count': int(_vc.max())}
    except Exception:
        pass
    out['future_dates'] = int((d > pd.Timestamp.now()).sum())
    return out

def _text_structured(series, comp):
    # Free-text columns: no distribution chart is meaningful, but the variable
    # should still be visible in the outputs with honest summaries instead of
    # being force-parsed as float and reported 100% invalid.
    out = {'completeness': comp}
    s = series.dropna().astype(str)
    out['n'] = int(len(s))
    out['n_unique'] = int(s.nunique())
    if len(s) == 0:
        return out
    lengths = s.str.len()
    out['length'] = {
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': _to_native(float(lengths.mean())),
    }
    out['pct_containing_digit'] = _to_native(float(s.str.contains('[0-9]').mean()) * 100)
    vc = s.value_counts()
    out['top_values'] = [
        {'value': str(k)[:80], 'count': int(c)} for k, c in vc.head(10).items()
    ]
    return out

# ---------------------------------------------------------------------------
# Stratified analysis: cross-variable context against common stratifiers
# (sex, age by default; configurable at DCR provision time). JSON-only: the
# interactive visualisation lives in the frontend, not in enclave PNGs.
#
# Robustness principles:
#   - effect sizes are first-class (Hedges g, Cliff's delta, Cramér's V,
#     epsilon²), not just p-values;
#   - rank-based measures (Mann-Whitney, Spearman, Kruskal-Wallis) are the
#     PRIMARY tests, since biomarkers are rarely normal;
#   - with ~a thousand variables tested, raw p-values WILL contain false
#     positives, so Benjamini-Hochberg q-values are attached per test family;
#   - strata below a minimum size get descriptives but no formal test.
# ---------------------------------------------------------------------------
_MIN_STRATUM_N = 10       # smallest stratum that still gets a formal test
_MIN_CORR_N = 10          # minimum complete pairs for a correlation
_AGE_BAND_EDGES = [0, 45, 55, 65, 75, 200]
_AGE_BAND_LABELS = ['<45', '45-54', '55-64', '65-74', '>=75']
_MAX_CONTINGENCY_CATEGORIES = 20
_NOTABLE_Q_CUTOFF = 0.05
_MAX_NOTABLE = 50

def _bh_qvalues(pairs):
    # Benjamini-Hochberg step-up. pairs: [(key, p)] -> {key: q}.
    items = [(k, float(p)) for k, p in pairs if p is not None]
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    out = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        k, p = items[rank - 1]
        q = min(prev, p * m / rank)
        prev = q
        out[k] = _to_native(q)
    return out

def _hedges_g(a, b):
    # Bias-corrected standardised mean difference and its approximate SE.
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None, None
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    if sp2 <= 0:
        return None, None
    d = (float(a.mean()) - float(b.mean())) / math.sqrt(sp2)
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = d * j
    se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2.0 * (n1 + n2)))
    return g, se

def _numeric_vs_groups(vals, groups):
    # Numeric target split by a categorical stratifier (e.g. weight by sex).
    dfp = pd.DataFrame({'v': pd.to_numeric(vals, errors='coerce'), 'g': groups}).dropna()
    # Nullable Float64/Int64 -> plain float64: scipy's tests reject pandas
    # masked arrays (safe here, all NA rows are already gone).
    dfp['v'] = dfp['v'].astype(float)
    per, arrays = {}, {}
    for g, sub in dfp.groupby('g'):
        s = sub['v']
        per[str(g)] = {
            'n': int(len(s)),
            'mean': _to_native(float(s.mean())),
            'std': _to_native(float(s.std())) if len(s) > 1 else None,
            'median': _to_native(float(s.median())),
            'q1': _to_native(float(s.quantile(0.25))),
            'q3': _to_native(float(s.quantile(0.75))),
        }
        arrays[str(g)] = s
    out = {'by_stratum': per}
    big = sorted(arrays.items(), key=lambda kv: -len(kv[1]))[:2]
    if len(big) == 2 and all(len(s) >= _MIN_STRATUM_N for _, s in big):
        (ga, a), (gb, b) = big
        comp = {'strata_compared': [ga, gb]}
        try:
            _t, p_t = ttest_ind(a, b, equal_var=False)
            comp['welch_p'] = _to_native(float(p_t))
        except Exception:
            pass
        try:
            u, p_u = mannwhitneyu(a, b, alternative='two-sided')
            comp['mannwhitney_p'] = _to_native(float(p_u))
            # Cliff's delta in [-1, 1]: robust dominance of the first stratum.
            comp['cliffs_delta'] = _to_native(2.0 * float(u) / (len(a) * len(b)) - 1.0)
        except Exception:
            pass
        g_val, g_se = _hedges_g(a, b)
        if g_val is not None:
            comp['hedges_g'] = _to_native(g_val)
            comp['hedges_g_ci95'] = [_to_native(g_val - 1.96 * g_se), _to_native(g_val + 1.96 * g_se)]
        out['comparison'] = comp
    else:
        out['comparison'] = None
        out['comparison_note'] = 'skipped: fewer than two strata with n >= %d' % _MIN_STRATUM_N
    return out

def _numeric_vs_numeric(vals, strat_vals):
    # Numeric target vs numeric stratifier (e.g. biomarker vs age).
    dfp = pd.DataFrame({'v': pd.to_numeric(vals, errors='coerce'),
                        's': pd.to_numeric(strat_vals, errors='coerce')}).dropna().astype(float)
    n = int(len(dfp))
    out = {'n': n}
    if n < _MIN_CORR_N:
        out['note'] = 'skipped: fewer than %d complete pairs' % _MIN_CORR_N
        return out
    try:
        rho, p_s = spearmanr(dfp['v'], dfp['s'])
        out['spearman_rho'] = _to_native(float(rho))
        out['spearman_p'] = _to_native(float(p_s))
    except Exception:
        pass
    try:
        r, p_p = pearsonr(dfp['v'], dfp['s'])
        out['pearson_r'] = _to_native(float(r))
        out['pearson_p'] = _to_native(float(p_p))
    except Exception:
        pass
    return out

def _age_band_summaries(vals, age_vals):
    dfp = pd.DataFrame({'v': pd.to_numeric(vals, errors='coerce'),
                        'a': pd.to_numeric(age_vals, errors='coerce')}).dropna().astype(float)
    if dfp.empty:
        return None
    bands = pd.cut(dfp['a'], bins=_AGE_BAND_EDGES, labels=_AGE_BAND_LABELS, right=False)
    rows = []
    for lbl, sub in dfp.groupby(bands, observed=True):
        if len(sub) == 0:
            continue
        rows.append({'band': str(lbl), 'n': int(len(sub)), 'median': _to_native(float(sub['v'].median()))})
    return rows or None

def _categorical_vs_groups(vals, groups):
    # Categorical target vs categorical stratifier: contingency + chi-square
    # independence + Cramér's V. Rare categories beyond the cap fold into
    # '(other)' so the table stays bounded.
    dfp = pd.DataFrame({'v': vals, 'g': groups}).dropna()
    if dfp.empty:
        return None
    top = dfp['v'].value_counts().head(_MAX_CONTINGENCY_CATEGORIES).index
    dfp['v'] = dfp['v'].where(dfp['v'].isin(top), other='(other)')
    tab = pd.crosstab(dfp['g'], dfp['v'])
    out = {
        'contingency': {str(g): {str(c): int(n) for c, n in row.items()} for g, row in tab.iterrows()},
        'pct_within_stratum': {str(g): {str(c): _to_native(n / row.sum() * 100) for c, n in row.items()}
                               for g, row in tab.iterrows()},
    }
    if tab.shape[0] > 1 and tab.shape[1] > 1:
        try:
            stat, p, dof, expected = chi2_contingency(tab)
            n = int(tab.values.sum())
            k = min(tab.shape) - 1
            out['chi2'] = {'statistic': _to_native(float(stat)), 'dof': int(dof),
                           'p_value': _to_native(float(p)), 'valid': bool(expected.min() >= 5)}
            out['cramers_v'] = _to_native(math.sqrt(float(stat) / (n * k))) if n > 0 and k > 0 else None
        except Exception:
            pass
    return out

def _groups_vs_numeric(vals, strat_vals):
    # Categorical target vs numeric stratifier: stratifier distribution per
    # category (e.g. median age per NYHA class) + Kruskal-Wallis + epsilon².
    dfp = pd.DataFrame({'v': vals, 'a': pd.to_numeric(strat_vals, errors='coerce')}).dropna()
    if dfp.empty:
        return None
    dfp['a'] = dfp['a'].astype(float)
    keep = dfp['v'].value_counts().head(_MAX_CONTINGENCY_CATEGORIES).index
    per, arrays = {}, []
    for cat, sub in dfp[dfp['v'].isin(keep)].groupby('v'):
        a = sub['a']
        per[str(cat)] = {'n': int(len(a)), 'median': _to_native(float(a.median())),
                         'q1': _to_native(float(a.quantile(0.25))), 'q3': _to_native(float(a.quantile(0.75)))}
        if len(a) >= _MIN_STRATUM_N:
            arrays.append(a.values)
    out = {'stratifier_by_category': per}
    if len(arrays) > 1:
        try:
            h, p = kruskal(*arrays)
            n = sum(len(x) for x in arrays)
            k = len(arrays)
            out['kruskal'] = {'H': _to_native(float(h)), 'p_value': _to_native(float(p)),
                              'epsilon_sq': _to_native((float(h) - k + 1) / (n - k)) if n > k else None}
        except Exception:
            pass
    return out

def _missingness_vs_stratifier(missing_mask, strat_series, kind):
    # Is *being missing* associated with the stratifier? Flags systematic
    # collection gaps (e.g. a biomarker only measured in men, or echo data
    # missing in the oldest patients) that per-variable profiling cannot see.
    n_missing = int(missing_mask.sum())
    n_present = int((~missing_mask).sum())
    if n_missing < _MIN_STRATUM_N or n_present < _MIN_STRATUM_N:
        return None
    if kind == 'categorical':
        dfp = pd.DataFrame({'m': missing_mask, 'g': strat_series}).dropna()
        tab = pd.crosstab(dfp['g'], dfp['m'])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            return None
        out = {'pct_missing_by_stratum': {str(g): _to_native(row.get(True, 0) / row.sum() * 100)
                                          for g, row in tab.iterrows()}}
        try:
            _stat, p, _dof, expected = chi2_contingency(tab)
            out['chi2_p'] = _to_native(float(p))
            out['valid'] = bool(expected.min() >= 5)
        except Exception:
            pass
        return out
    a = pd.to_numeric(strat_series[missing_mask], errors='coerce').dropna().astype(float)
    b = pd.to_numeric(strat_series[~missing_mask], errors='coerce').dropna().astype(float)
    if len(a) < _MIN_STRATUM_N or len(b) < _MIN_STRATUM_N:
        return None
    out = {'stratifier_median_when_missing': _to_native(float(a.median())),
           'stratifier_median_when_present': _to_native(float(b.median()))}
    try:
        _u, p = mannwhitneyu(a, b, alternative='two-sided')
        out['mannwhitney_p'] = _to_native(float(p))
    except Exception:
        pass
    return out

def _detect_stratifiers():
    cfg = _STRATIFIER_CONFIG if isinstance(_STRATIFIER_CONFIG, dict) else {}
    excluded = set(str(x).strip().lower() for x in (cfg.get('excluded_defaults') or []))
    found = {}
    for name, spec in _DEFAULT_STRATIFIERS.items():
        if name in excluded:
            data_issues.append("Stratifier '%s' excluded by DCR configuration." % name)
            continue
        cands = []
        for v in vars_details.columns:
            vl = str(v).strip().lower()
            if vl in patient_id_cols or vl not in data.columns:
                continue
            omop = _meta_value(vars_details[v], 'omop_id')
            if omop.endswith('.0'):
                omop = omop[:-2]
            if omop in spec['omop_ids']:
                cands.append(vl)
        if not cands:
            data_issues.append("Default stratifier '%s' (OMOP %s) not found in this cohort; skipped."
                               % (name, '/'.join(spec['omop_ids'])))
            continue
        # Several columns can share the concept (e.g. age at every visit);
        # prefer the least-missing one, which is normally the baseline value.
        cands_sorted = sorted(cands, key=lambda c: -_int0(completeness_by_var.get(c, {}).get('n_valid')))
        found[name] = {'column': cands_sorted[0], 'kind': spec['kind'], 'source': 'default',
                       'all_candidates': cands}
    for raw in (cfg.get('custom_variables') or []):
        vl = str(raw).strip().lower()
        if vl in [s['column'] for s in found.values()] or vl in found:
            continue
        if vl not in data.columns or vl not in vars_details.columns:
            data_issues.append("Custom stratifier '%s' not found in the dataset; skipped." % raw)
            continue
        t = vars_details[vl]['inferred_type']
        kind = 'numeric' if t in ('int', 'float') else ('categorical' if t == 'categorical' else None)
        if kind is None:
            data_issues.append("Custom stratifier '%s' has unsupported type '%s'; skipped." % (raw, t))
            continue
        found[vl] = {'column': vl, 'kind': kind, 'source': 'user'}
    # Materialise the stratum series. Categorical strata are formed on the
    # dictionary LABELS, never on raw codes: cohorts flip codings (0=male in
    # one study, 1=male in another) and only the labels are comparable.
    for name, s in found.items():
        col = s['column']
        if s['kind'] == 'categorical':
            mapping = vars_details[col].get('categories', None)
            mapping = mapping if isinstance(mapping, dict) else {}
            mapping = {_canon_value_token(str(k)).lower(): v for k, v in mapping.items()}

            def _label_of(x, _m=mapping):
                # Observed values are canonicalised + lowercased exactly like
                # the mapping keys, so case variants of one stratum merge and
                # the dictionary label is used whenever one is declared.
                if pd.isna(x):
                    return None
                key = _canon_value_token(str(x)).lower()
                return _m.get(key, key)

            ser = data[col].map(_label_of)
            s['series'] = ser
            s['strata_n'] = {str(k): int(v) for k, v in ser.value_counts().items()}
        else:
            ser = pd.to_numeric(data[col], errors='coerce')
            s['series'] = ser
            s['n_valid'] = int(ser.notna().sum())
            s['median'] = _to_native(float(ser.median())) if ser.notna().any() else None
    return found

def stratified_analysis(structured):
    strats = _detect_stratifiers()
    if not strats:
        return {}
    fdr_registry = []   # (family, variable, dict-to-annotate, primary p)
    for column in data.columns:
        if column not in vars_details.columns or column in patient_id_cols:
            continue
        entry = structured.get(column)
        if entry is None:
            continue
        typ = vars_details[column]['inferred_type']
        blocks = {}
        for name, s in strats.items():
            if column == s['column']:
                continue
            p_primary, eff = None, None
            try:
                if s['kind'] == 'categorical':
                    if typ in ('int', 'float'):
                        block = _numeric_vs_groups(data[column], s['series'])
                        comp = block.get('comparison') or {}
                        p_primary, eff = comp.get('mannwhitney_p'), comp.get('hedges_g')
                    elif typ == 'categorical':
                        block = _categorical_vs_groups(data[column], s['series']) or {}
                        p_primary = (block.get('chi2') or {}).get('p_value')
                        eff = block.get('cramers_v')
                    else:
                        block = {}
                else:
                    if typ in ('int', 'float'):
                        block = _numeric_vs_numeric(data[column], s['series'])
                        if name == 'age':
                            bands = _age_band_summaries(data[column], s['series'])
                            if bands:
                                block['by_age_band'] = bands
                        p_primary, eff = block.get('spearman_p'), block.get('spearman_rho')
                    elif typ == 'categorical':
                        block = _groups_vs_numeric(data[column], s['series']) or {}
                        kw = block.get('kruskal') or {}
                        p_primary, eff = kw.get('p_value'), kw.get('epsilon_sq')
                    else:
                        block = {}
                miss = _missingness_vs_stratifier(data[column].isna(), s['series'], s['kind'])
                if miss:
                    block['missingness'] = miss
                    mp = miss.get('chi2_p', miss.get('mannwhitney_p'))
                    if mp is not None:
                        fdr_registry.append(('missingness_vs_' + name, column, miss, mp))
            except Exception as e:
                data_issues.append("Stratified analysis failed for %s vs %s: %s" % (column, name, str(e)))
                continue
            if block:
                if p_primary is not None:
                    block['primary_p'] = _to_native(p_primary)
                    block['primary_effect'] = _to_native(eff) if eff is not None else None
                    fdr_registry.append(('association_vs_' + name, column, block, p_primary))
                blocks[name] = block
        if blocks:
            entry['stratified'] = blocks
    # BH-FDR within each family of tests, written back into the same dicts.
    families = {}
    for fam, var, ref, p in fdr_registry:
        families.setdefault(fam, []).append((var, ref, p))
    for fam, items in families.items():
        qmap = _bh_qvalues([(i, p) for i, (_v, _r, p) in enumerate(items)])
        for i, (_v, ref, _p) in enumerate(items):
            if i in qmap:
                ref['q_value'] = qmap[i]
    # Metadata + a ready-to-render shortlist of associations that survive FDR.
    meta = {'config': _STRATIFIER_CONFIG, 'stratifiers': {}}
    for name, s in strats.items():
        info = {'column': s['column'], 'kind': s['kind'], 'source': s['source']}
        if len(s.get('all_candidates', [])) > 1:
            info['other_candidates'] = [c for c in s['all_candidates'] if c != s['column']]
        if s['kind'] == 'categorical':
            info['strata_n'] = s['strata_n']
        else:
            info['n_valid'] = s['n_valid']
            info['median'] = s['median']
        meta['stratifiers'][name] = info
    notable = {}
    for fam, items in families.items():
        rows = []
        for var, ref, p in items:
            q = ref.get('q_value')
            if q is not None and q < _NOTABLE_Q_CUTOFF:
                rows.append({'variable': var, 'q_value': q,
                             'effect': ref.get('primary_effect'), 'p_value': _to_native(p)})
        rows.sort(key=lambda r: abs(r['effect']) if isinstance(r.get('effect'), (int, float)) else 0.0,
                  reverse=True)
        if rows:
            notable[fam] = rows[:_MAX_NOTABLE]
    meta['notable_associations'] = notable
    meta['n_tests_by_family'] = {fam: len(items) for fam, items in families.items()}
    meta['fdr_note'] = ("q_value is the Benjamini-Hochberg FDR-adjusted p-value, computed within each "
                        "test family across all variables. With this many tests, raw p-values alone "
                        "will contain false positives; screen on q_value and judge magnitude by the "
                        "effect size, not the p-value.")
    return meta


def write_structured_v2(structured_stats, stratifier_meta=None):
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
        'schema_version': '2.1',
        'cohort_id': '{cohort_id}',
        'generated_at': datetime.now().isoformat(),
        'n_rows': int(n_rows_total),
        'n_variables': int(len(variables)),
        'stratified_analysis': stratifier_meta or {},
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

                # The v2 structured measures are the single source of truth;
                # the panel below only RENDERS them. (The two used to be
                # computed independently and drifted.)
                try:
                    structured[column] = _numeric_structured(df[column], comp)
                    structured[column]['type'] = 'numeric'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured numeric stats for {column}: {str(e)}")
                _sv = structured.get(column, {})
                _out_iqr = _sv.get('outliers_iqr') or {}
                _out_mad = _sv.get('outliers_mad') or {}
                _out_z = _sv.get('outliers_z') or {}
                _norm = _sv.get('normality') or {}
                _zero_frac = _sv.get('zero_fraction')

                # Human-readable verdict for the Shapiro test.
                _shapiro_p = _norm.get('shapiro_p')
                if _shapiro_p is None:
                    _normality_line = "p-value=N/A => Insufficient Data"
                else:
                    _normality_line = "p-value=%.4f => %s" % (
                        _shapiro_p, "Normal" if _shapiro_p > 0.05 else "Non-Normal")

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
                    f"Number of Unique Values/Categories: {_sv.get('n_unique', 0)}",
                    _GROUP_BREAK,
                    f"Mean: {_fmt_num(_sv.get('mean'))}",
                    f"Median: {_fmt_num(_sv.get('median'))}",
                    f"Mode: {_fmt_num(_sv.get('mode'))}",
                    f"Trimmed mean (10%): {_fmt_stat(_sv.get('trimmed_mean_10'))}",
                    _GROUP_BREAK,
                    f"Std Dev: {_fmt_num(_sv.get('std'))}",
                    f"Variance: {_fmt_num(_sv.get('variance'))}",
                    f"CV: {_fmt_stat(_sv.get('cv'), 3)}",
                    f"MAD: {_fmt_stat(_sv.get('mad'))}",
                    f"Min: {_fmt_num(_sv.get('min'))}",
                    f"Max: {_fmt_num(_sv.get('max'))}",
                    f"Range: {_fmt_num(_sv.get('range'))}",
                    _GROUP_BREAK,
                    f"Q1: {_fmt_num(_sv.get('q1'))}",
                    f"Q3: {_fmt_num(_sv.get('q3'))}",
                    f"IQR: {_fmt_num(_sv.get('iqr'))}",
                    f"P5 / P95: {_fmt_stat(_sv.get('p5'))} / {_fmt_stat(_sv.get('p95'))}",
                    _GROUP_BREAK,
                    f"Zero fraction (% of valid): {_fmt_pct(None if _zero_frac is None else _zero_frac * 100)}",
                    f"Outliers (IQR, % of valid): {_fmt_count(_out_iqr.get('count'), _out_iqr.get('pct'))}",
                    f"Outliers (MAD, % of valid): {_fmt_count(_out_mad.get('count'), _out_mad.get('pct'))}",
                    f"Outliers (Z): {_fmt_count(_out_z.get('count'), _out_z.get('pct'))}",
                    _GROUP_BREAK,
                    f"Skewness: {_fmt_num(_sv.get('skewness'))}",
                    f"Kurtosis: {_fmt_num(_sv.get('kurtosis'))}",
                    f"W_Test: {_fmt_stat(_norm.get('shapiro_w'))}",
                    f"Normality Test: {_normality_line}",
                    f"Normality (D'Agostino) p-value: {_fmt_stat(_norm.get('dagostino_p'), 4)}"
                )

                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'numerical')
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")


            # Categorical variables
            elif vars_details[column]['inferred_type'] == 'categorical':
                # Get the categories mapping and normalize keys
                categories_mapping = vars_details[column].get("categories", None)
                if not isinstance(categories_mapping, dict):
                    categories_mapping = {}
                # Observed values were canonicalised ("1.0" -> "1") and
                # lowercased before counting, so dictionary keys must be
                # normalised the same way or lookups miss (e.g. categories
                # declared without "=" are stored with uppercase keys).
                categories_mapping = {_canon_value_token(str(k)).lower(): v for (k, v) in categories_mapping.items()}

                # The v2 structured measures are the single source of truth;
                # the panel below only RENDERS them.
                try:
                    structured[column] = _categorical_structured(df[column], comp, categories_mapping)
                    structured[column]['type'] = 'categorical'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured categorical stats for {column}: {str(e)}")
                _sv = structured.get(column, {})
                _dist = _sv.get('distribution') or []

                if not _dist:
                    stats_text = tuple(_identity_lines(column, vars_details[column])) + (
                        f"Type: Categorical (encoded as {df[column].dtype})",
                        _GROUP_BREAK,
                        f"Number of Unique Categories: 0",
                        f"Missing Values (% of all rows): {_fmt_count(df[column].isnull().sum(), df[column].isnull().mean() * 100)}"
                    )
                else:
                    # Class balance, one panel entry per category: a single
                    # entry with embedded newlines is reflowed into one blob by
                    # _format_stats_lines. Percentages are given against the
                    # valid total and against all rows, because for a variable
                    # with heavy missingness those are very different claims.
                    # Only the first N categories are listed; the full
                    # distribution always goes to the structured JSON output.
                    _MAX_CLASS_BALANCE_LINES = 30
                    _cb_hidden = max(0, len(_dist) - _MAX_CLASS_BALANCE_LINES)
                    class_balance_lines = []
                    _cb_flat = []
                    for _entry in _dist[:_MAX_CLASS_BALANCE_LINES]:
                        _key, _mapped = _entry['value'], _entry['label']
                        if _mapped and str(_key) != str(_mapped):
                            _cat_label = f"{_key} ({_mapped})"
                        else:
                            _cat_label = f"{_key}"
                        _cb_entry = f"{_cat_label} -> {_entry['count']} ({_fmt_pct_dual(_entry['pct'], _entry['pct_of_total'])})"
                        class_balance_lines.append(f"- {_cb_entry}")
                        _cb_flat.append(_cb_entry)
                    if _cb_hidden:
                        _cb_more = f"... and {_cb_hidden} more categories (see JSON output)"
                        class_balance_lines.append(f"- {_cb_more}")
                        _cb_flat.append(_cb_more)
                    # No colon anywhere in this value: the flat-JSON builder
                    # keeps only the text between the first and second colon.
                    class_balance_flat = "; ".join(_cb_flat)

                    _mf = _sv.get('most_frequent') or {}
                    # Chi-square goodness of fit vs uniform; the statistic alone
                    # scales with n, so df and p-value are shown with it.
                    _chi = _sv.get('chi_square_uniform') or {}
                    if _chi:
                        _chi_line = "stat=%s, df=%s, p=%s%s" % (
                            _fmt_num(_chi.get('statistic')), _chi.get('dof', 'N/A'),
                            _fmt_stat(_chi.get('p_value'), 4),
                            "" if _chi.get('valid', True) else " (low expected counts; approximate)")
                    else:
                        _chi_line = "N/A (single category)"

                    stats_text = tuple(_identity_lines(column, vars_details[column])) + (
                        f"Type: Categorical (encoded as {df[column].dtype})",
                        _GROUP_BREAK,
                        f"Number of unique values/categories: {_sv.get('n_unique', len(_dist))}",
                        f"Most frequent category: {_mf.get('label', _mf.get('value', 'N/A'))} ",
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
                        f"Chi-square vs uniform: {_chi_line}",
                        _GROUP_BREAK,
                        "Class balance (n, % of valid / % of all rows)",
                    ) + tuple(class_balance_lines)

                # A bar chart with hundreds of categories draws one bar, one
                # rotated tick label and one annotation per category, and is
                # unreadable regardless of how long it takes to render.
                _MAX_CHARTED_CATEGORIES = 60
                if column in vars_to_graph and len(_dist) > _MAX_CHARTED_CATEGORIES:
                    data_issues.append(
                        f"Column {column}: {len(_dist)} distinct categories exceeds the "
                        f"charting limit of {_MAX_CHARTED_CATEGORIES}; chart skipped."
                    )
                elif column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'categorical', category_mapping = categories_mapping)
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")
                        
                        
            
            elif vars_details[column]['inferred_type'] == 'date':
                # The v2 structured measures are the single source of truth;
                # the panel below only RENDERS them. Series.describe() is
                # deliberately avoided: the keys it returns for datetime input
                # differ across pandas versions (1.x returns count/unique/
                # top/freq unless told otherwise; 2.x returns mean/min/
                # quantiles/max), so relying on its layout ties the script to
                # the enclave's pandas.
                try:
                    structured[column] = _date_structured(df[column], comp)
                    structured[column]['type'] = 'date'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured date stats for {column}: {str(e)}")
                _sv = structured.get(column, {})
                # Full calendar dates are charted at month+year resolution, so
                # the panel says so; the summary statistics below keep full
                # day-level detail.
                if str(_sv.get('granularity', '')).startswith('day'):
                    _granularity_lines = [
                        f"Value granularity: full date (year, month, day)",
                        f"Chart note: {_MONTH_AGG_NOTE}",
                    ]
                else:
                    _granularity_lines = []
                _count_lines = [
                        f"Count of valid observations: {count_nonnull} of {n_rows_total} rows",
                        f"Count empty (blank cells, % of all rows): {_fmt_count(count_na, comp.get('pct_empty', 0))}",
                        f"Count coded missing (% of all rows): {_fmt_count(count_missing, comp.get('pct_coded_missing', 0))}",
                        f"Count missing total (% of all rows): {_fmt_count(count_missing_total, comp.get('pct_missing_total', 0))}",
                        f"Count invalid (% of all rows): {_fmt_count(count_invalid, comp.get('pct_invalid', 0))}",
                        f"Missing code(s) declared: {missing_codes_txt}",
                ]
                if not _sv.get('n'):
                    stats_text = _identity_lines(column, vars_details[column]) + [
                            f"Type: Date (encoded as {df[column].dtype})",
                            _GROUP_BREAK,
                    ] + _count_lines + [
                            _GROUP_BREAK,
                            "No valid date values to summarise.",
                    ]
                else:
                    _mf = _sv.get('most_frequent') or {}
                    stats_text = _identity_lines(column, vars_details[column]) + [
                            f"Type: Date (encoded as {df[column].dtype})",
                    ] + _granularity_lines + [
                            _GROUP_BREAK,
                            f"Number of unique values: {_sv.get('n_unique', 'N/A')}",
                            f"Most frequent value: {_mf.get('value', 'N/A')}",
                            _GROUP_BREAK,
                    ] + _count_lines + [
                            _GROUP_BREAK,
                            f"Mean: {_sv.get('mean', 'N/A')}",
                            f"Median: {_sv.get('median', 'N/A')}",
                            f"Min: {_sv.get('min', 'N/A')}",
                            f"Max: {_sv.get('max', 'N/A')}",
                            f"Range (days): {_sv.get('range_days', 'N/A')}",
                            _GROUP_BREAK,
                            f"Q1: {_sv.get('q1', 'N/A')}",
                            f"Q3: {_sv.get('q3', 'N/A')}",
                            f"IQR (days): {_sv.get('iqr_days', 'N/A')}",
                            _GROUP_BREAK,
                            f"Dates in the future: {_sv.get('future_dates', 'N/A')}",
                    ]

                if column in vars_to_graph:
                    try:
                        graph_tick_data[column] = create_save_graph(df, column, stats_text, 'datetime')
                    except Exception as e:
                        data_issues.append(f"Failed to create a graph for column {column}. Exception msg: {str(e)}")

            elif vars_details[column]['inferred_type'] == 'text':
                # Free text: no chart (a bar per unique string is meaningless),
                # but the variable stays visible with honest text summaries.
                try:
                    structured[column] = _text_structured(df[column], comp)
                    structured[column]['type'] = 'text'
                    structured[column]['label'] = vars_details[column]['var_label']
                except Exception as e:
                    data_issues.append(f"Failed to compute structured text stats for {column}: {str(e)}")
                _sv = structured.get(column, {})
                _len = _sv.get('length') or {}
                _top_lines = []
                for _entry in (_sv.get('top_values') or [])[:5]:
                    # No colon in these values; the flat-JSON builder keeps
                    # only the text between the first and second colon.
                    _top_lines.append(f"- {_entry['value']} -> {_entry['count']}")
                stats_text = tuple(_identity_lines(column, vars_details[column])) + (
                    f"Type: Text (free text; no chart rendered)",
                    _GROUP_BREAK,
                    f"Count of valid observations: {count_nonnull} of {n_rows_total} rows",
                    f"Count empty (blank cells, % of all rows): {_fmt_count(count_na, comp.get('pct_empty', 0))}",
                    f"Count coded missing (% of all rows): {_fmt_count(count_missing, comp.get('pct_coded_missing', 0))}",
                    f"Count missing total (% of all rows): {_fmt_count(count_missing_total, comp.get('pct_missing_total', 0))}",
                    f"Missing code(s) declared: {missing_codes_txt}",
                    _GROUP_BREAK,
                    f"Number of unique values: {_sv.get('n_unique', 0)}",
                    f"Value length (min / mean / max): {_len.get('min', 'N/A')} / {_fmt_num(_len.get('mean'), 1)} / {_len.get('max', 'N/A')}",
                    f"Values containing a digit (% of valid): {_fmt_pct(_sv.get('pct_containing_digit'))}",
                    _GROUP_BREAK,
                    "Most frequent values (n)",
                ) + tuple(_top_lines)

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
    stratifier_meta = stratified_analysis(structured_stats)
    print("Stratified analysis complete:", list((stratifier_meta.get('stratifiers') or {}).keys()), flush=True)
except Exception as e:
    print("Stratified analysis failed:", e)
    data_issues.append(f"Stratified analysis failed: {str(e)}")
    stratifier_meta = {}

try:
    write_structured_v2(structured_stats, stratifier_meta)
except Exception as e:
    print("Failed to write v2 structured EDA output:", e)
    data_issues.append(f"Failed to write v2 structured EDA output: {str(e)}")

with open('/output/data_issues.json', 'w') as json_file:
    json.dump(data_issues, json_file, indent=4)
"""
    return (raw_script.replace("{shared_helpers}", _SHARED_HELPERS)
            .replace("{chart_style}", _CHART_STYLE_HELPERS)
            .replace("{stratifier_config}", _json.dumps(_cfg))
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
import textwrap as _textwrap
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
        'concept_name': _dget('concept_name'),
        'additional_context': _dget('additional_context'),
        'units': _dget('units'),
        'visits': _dget('visits'),
        'visit_concept_name': _dget('visit_concept_name'),
        'categories': (d.get('categories') if typ == 'categorical' else None),
    }

# ---- Identify the patient-id column (never charted/shown) -----------------
_PATIENT_OMOP_IDS = ('4086934', '40757164')
_PATIENT_CONCEPT_CODE = 'snomed:184107009'
_PATIENT_ID_CODES = tuple(['omop:' + pid for pid in _PATIENT_OMOP_IDS] + list(_PATIENT_OMOP_IDS))
patient_id_col = None
for v, m in var_meta.items():
    if (m['omop_id'] in _PATIENT_OMOP_IDS or
            (m['concept_code'] and m['concept_code'].lower() == _PATIENT_CONCEPT_CODE) or
            (m['concept_code'] and m['concept_code'].lower().strip() in _PATIENT_ID_CODES)):
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

def _strip_visit_words(text):
    # Removes the visit/time boilerplate from a single human-readable label, so
    # that "Weight at baseline" names the family as "Weight" rather than
    # claiming the whole family was measured at baseline. Returns '' when
    # nothing but boilerplate is left, so the caller can try something else.
    return ' '.join(_concept_tokens(text)).strip()

def _shared_tokens(names):
    # The tokens a set of names has in common. A shared prefix is the usual
    # case (blood_pressure_visit_1, blood_pressure_end_of_study); otherwise the
    # tokens present in every name, in the order the first name uses them.
    token_lists = [_concept_tokens(n) for n in names]
    token_lists = [t for t in token_lists if t]
    if not token_lists:
        return ''
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
    return ' '.join(shared).strip()

def _family_label(members, member_labels, fallback):
    # Preference order: what the member *column names* share, then what their
    # human-readable *labels* share, then the first label with its visit words
    # stripped. Every step goes through _concept_tokens, so no step can leak
    # visit boilerplate into the family name.
    for candidate in (_shared_tokens(members),
                      _shared_tokens(member_labels),
                      _strip_visit_words(member_labels[0] if member_labels else '')):
        if candidate:
            return candidate
    return fallback

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
    # Member labels are in visit order, because `members` is already sorted.
    member_labels = [var_meta[v]['var_label'] for v in members]
    # The standardised names as the metadata dictionary spells them, shown
    # verbatim on the chart. `context` above is the normalised grouping key.
    concept_name = next((var_meta[v]['concept_name'] for v in members if var_meta[v]['concept_name']), None)
    context_name = next((var_meta[v]['additional_context'] for v in members if var_meta[v]['additional_context']), None)
    # Last resort for the family name. The raw member label is deliberately not
    # used: when every name and label is pure visit boilerplate ("baseline",
    # "end of study") it would name the family after a single visit.
    fam_fallback = concept_name or f"concept {concept}"
    families.append({
        'key': key,
        'concept': str(concept),
        'context': context,
        'var_type': fam_type,
        'var_label': _family_label(members, member_labels, fam_fallback),
        'concept_name': concept_name,
        'context_name': context_name,
        'units': units,
        'members': members,
        'member_labels': member_labels,
        'visit_labels': [_visit_label(var_meta[v]) for v in members],
    })

# ---- Chart titling ---------------------------------------------------------
def _family_subtitle(fam):
    # The standardised concept the family measures, plus the additional context
    # that distinguishes it from other families on the same concept.
    concept = fam.get('concept_name') or fam.get('concept') or 'unknown concept'
    context = fam.get('context_name') or 'none'
    return f"Concept: {concept}    Additional context: {context}"

def _family_members_line(fam, width=120, max_lines=3):
    # The member variables, in the same visit order the chart uses, so a reader
    # can tell which column produced which point.
    labels = [str(l) for l in fam.get('member_labels') or fam.get('members') or []]
    if not labels:
        return ''
    text = "Variables (in visit order): " + "  \u2192  ".join(labels)
    wrapped = _textwrap.wrap(text, width) or ['']
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1] + ' ...'
    return "\\n".join(wrapped)

def _set_chart_titles(ax, main_title, fam):
    # The family name goes on the figure so it stays the visually dominant
    # line; the two descriptive lines go on the axes title beneath it. Both are
    # accounted for by tight_layout, so neither can be clipped.
    fig = ax.get_figure()
    fig.suptitle(main_title, fontsize=13)
    lines = [_family_subtitle(fam)]
    members_line = _family_members_line(fam)
    if members_line:
        lines.append(members_line)
    ax.set_title("\\n".join(lines), fontsize=8)

# ---- Numeric family: baseline-relative trajectory banding -----------------
def _process_numeric_family(fam):
    fam_df = data[fam['members']].apply(pd.to_numeric, errors='coerce')
    fam_df.columns = fam['visit_labels']
    n_visits = fam_df.shape[1]
    if n_visits < 2:
        return None
    # A patient contributes as long as they have at least two measurements
    # anywhere in the family; the first *scheduled* visit no longer has to be
    # present. This keeps patients who enrolled late or missed baseline but
    # were still followed over time.
    usable = fam_df.notna().sum(axis=1) >= 2
    sub = fam_df[usable].reset_index(drop=True)
    n_patients = int(len(sub))
    if n_patients < 5:
        data_issues.append(f"Longitudinal family {fam['key']}: fewer than 5 usable patient trajectories, skipping chart.")
        return None

    # ---- x positions for slope: real elapsed time when every visit resolves to
    # a duration/baseline, otherwise evenly-spaced visit order. Using time-aware
    # spacing keeps a "+10% over 3 months" band from being read the same as
    # "+10% over 3 years".
    _keys = [_visit_sort_key(lbl) for lbl in fam['visit_labels']]
    _tvals = [k[1] for k in _keys]
    _tiers = {k[0] for k in _keys}
    if _tiers <= {0.0, 1.0} and all(_tvals[i] < _tvals[i + 1] for i in range(len(_tvals) - 1)):
        x_pos = np.array(_tvals, dtype=float)
    else:
        x_pos = np.arange(n_visits, dtype=float)
    x_span = float(x_pos.max() - x_pos.min()) or 1.0

    # ---- Per-patient baseline = each patient's OWN first recorded value, so a
    # patient anchored at visit 2 is still expressed as change-from-their-start
    # and stays comparable, by shape, to a patient anchored at baseline.
    arr = sub.to_numpy(dtype=float)
    baseline_vals = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        nn = np.where(~np.isnan(arr[i]))[0]
        if len(nn):
            baseline_vals[i] = arr[i][nn[0]]
    baseline_vals = pd.Series(baseline_vals, index=sub.index)
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

    def _silhouette(dist_mat, lab):
        # Average silhouette width from a precomputed distance matrix (numpy
        # only). Used to pick the band count instead of a fixed rule.
        lab = np.asarray(lab)
        clusters = np.unique(lab)
        if len(clusters) < 2:
            return -1.0
        scores = []
        for i in range(len(lab)):
            same = lab == lab[i]
            same[i] = False
            a = float(np.mean(dist_mat[i, same])) if same.any() else 0.0
            b = np.inf
            for c in clusters:
                if c == lab[i]:
                    continue
                other = lab == c
                if other.any():
                    b = min(b, float(np.mean(dist_mat[i, other])))
            if b == np.inf:
                scores.append(0.0)
            else:
                denom_s = max(a, b)
                scores.append((b - a) / denom_s if denom_s > 0 else 0.0)
        return float(np.mean(scores)) if scores else -1.0

    if nf < 4:
        fit_labels = np.ones(nf, dtype=int)
    else:
        dist = np.zeros((nf, nf))
        for i in range(nf):
            for j in range(i + 1, nf):
                dd = _nan_rmse(fit_mat[i], fit_mat[j])
                dist[i, j] = dist[j, i] = dd
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        # Pick the band count by the best average silhouette over candidate
        # values, rather than assuming sqrt(n/2) bands regardless of structure.
        k_max = min(6, nf - 1)
        best_score, fit_labels = -np.inf, None
        for k in range(2, k_max + 1):
            cand = fcluster(Z, k, criterion='maxclust')
            if len(set(cand)) < 2:
                continue
            score = _silhouette(dist, cand)
            if score > best_score:
                best_score, fit_labels = score, cand
        if fit_labels is None:
            fit_labels = np.ones(nf, dtype=int)

    centroids = {b: np.nanmean(fit_mat[fit_labels == b], axis=0) for b in set(fit_labels)}
    labels = np.array([min(centroids, key=lambda b: _nan_rmse(mat[i], centroids[b])) for i in range(n_patients)])

    bands = []
    for b in sorted(set(labels)):
        band_mask = labels == b
        band_df = sub[band_mask]
        band_norm = norm_pct[band_mask]
        # ---- Trend from the WHOLE trajectory: least-squares slope of the
        # band's per-visit median (baseline-relative) against visit time. The
        # implied end-to-end change (slope x span) drives the label, so a band
        # that dips then recovers is not mislabelled from its endpoint alone.
        med_norm = band_norm.median(axis=0, skipna=True).to_numpy(dtype=float)
        fit_mask = ~np.isnan(med_norm)
        if fit_mask.sum() >= 2:
            slope = float(np.polyfit(x_pos[fit_mask], med_norm[fit_mask], 1)[0])
        else:
            slope = 0.0
        trend_change = slope * x_span
        end_col = band_norm.iloc[:, -1]
        end_shift = float(end_col.mean(skipna=True)) if end_col.notna().any() else 0.0
        if trend_change > 0.05:
            trend_label = 'rising'
        elif trend_change < -0.05:
            trend_label = 'declining'
        else:
            trend_label = 'stable'
        bands.append({
            'band_label': trend_label,
            'end_shift': None if pd.isna(end_shift) else float(end_shift),
            'trend_change': None if pd.isna(trend_change) else float(trend_change),
            'trend_slope': None if pd.isna(slope) else float(slope),
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

def _plot_numeric_family(ax, result, fam, units=None):
    x = np.arange(len(result['visit_labels']))
    bands_sorted = sorted(
        result['bands'],
        key=lambda b: (_TREND_ORDER.get(b['band_label'], 1),
                       -abs((b.get('trend_change') if b.get('trend_change') is not None else b.get('end_shift')) or 0.0)),
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
        shift = band.get('trend_change')
        if shift is None:
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
    _set_chart_titles(ax, f"Longitudinal Trend - {str(fam['var_label']).upper()}", fam)
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

def _plot_categorical_family(ax, processed, fam):
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
    _set_chart_titles(ax, f"Category Flow - {str(fam['var_label']).upper()}", fam)

# ---- Run over every identified family --------------------------------------
longitudinal_json = {}
for fam in families:
    name_slug = re.sub(r'[^a-z0-9]+', '_', str(fam['var_label']).lower()).strip('_')
    # fig, ax = plt.subplots(figsize=(10, 6))
    try:
        if fam['var_type'] in ('int', 'float'):
            result = _process_numeric_family(fam)
            if result is None:
                # plt.close(fig)
                continue
            # _plot_numeric_family(ax, result, fam, fam['units'])
            chart_suffix = 'longitudinal-trend'
            longitudinal_json[fam['key']] = {
                'var_type': fam['var_type'],
                'var_label': fam['var_label'],
                'concept': fam['concept'],
                'concept_name': fam['concept_name'],
                'additional_context': fam['context'],
                'additional_context_name': fam['context_name'],
                'units': fam['units'],
                'member_variables': fam['members'],
                'member_labels': fam['member_labels'],
                'chart_type': 'trajectory_bands',
                **result,
            }
        elif fam['var_type'] == 'categorical':
            processed = _process_categorical_family(fam)
            if processed is None:
                # plt.close(fig)
                continue
            # _plot_categorical_family(ax, processed, fam)
            chart_suffix = 'category_flow'
            longitudinal_json[fam['key']] = {
                'var_type': fam['var_type'],
                'var_label': fam['var_label'],
                'concept': fam['concept'],
                'concept_name': fam['concept_name'],
                'additional_context': fam['context'],
                'additional_context_name': fam['context_name'],
                'member_variables': fam['members'],
                'member_labels': fam['member_labels'],
                'visit_labels': fam['visit_labels'],
                'chart_type': 'alluvial',
                'n_patients': processed['n_patients'],
            }
        else:
            # plt.close(fig)
            continue
        # plt.tight_layout()
        # plt.savefig(f"/output/{name_slug}_{chart_suffix}.png", dpi=160, bbox_inches='tight')
    except Exception as e:
        data_issues.append(f"Longitudinal family {fam['key']}: failed to build chart: {e}")
    finally:
        pass
        # plt.close('all')

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

# Find the patient ID variable: OMOP ID 4086934 or 40757164 first, then the
# patient-id concept in the VARIABLE CONCEPT CODE column (snomed:184107009,
# or the OMOP concept codes as the last fallback).
patient_id_var = None
varname_cols = [x for x in ['VARIABLE NAME', 'VARIABLENAME', 'VAR NAME'] if x in dictionary_df.columns]
if varname_cols:
    varname_col = varname_cols[0]
    if 'VARIABLE OMOP ID' in dictionary_df.columns:
        patient_id_rows = dictionary_df[dictionary_df['VARIABLE OMOP ID'].isin(['4086934', '40757164'])]
        if not patient_id_rows.empty:
            patient_id_var = patient_id_rows.iloc[0][varname_col]
            print(f"Found patient ID variable by OMOP ID: {patient_id_var}")
    else:
        print("'VARIABLE OMOP ID' column not found in metadata dictionary")
    if patient_id_var is None and 'VARIABLE CONCEPT CODE' in dictionary_df.columns:
        codes = dictionary_df['VARIABLE CONCEPT CODE'].astype(str).str.strip().str.lower()
        code_rows = dictionary_df[codes.isin(['snomed:184107009', '184107009', 'omop:4086934', '4086934', 'omop:40757164', '40757164'])]
        if not code_rows.empty:
            patient_id_var = code_rows.iloc[0][varname_col]
            print(f"Found patient ID variable via concept code: {patient_id_var}")
    if patient_id_var is None:
        print("No patient ID variable found in metadata dictionary (OMOP ID 4086934 / concept code)")
else:
    print("Could not find variable name column in metadata dictionary")

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
