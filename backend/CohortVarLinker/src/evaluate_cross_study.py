
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score
)
import re
import os
import warnings
warnings.filterwarnings('ignore')
from llm.utils import clean_label_remove_temporal_context, setup_logger, determine_var_uri , canonical_var_key

logger = setup_logger('eval.log')

# import unicodedata


_INSTANCE_PREFIX_RE = re.compile(r"^\s*\d+\s*\.\s*")  # leading event index: "1.", "10."

# def canonical_var_key(x, strip_instance_prefix: bool = True) -> str:
#     """Robust key for comparing variable names across CSVs.""

#     Encoding-invariant (mojibake repaired, diacritics folded) and
#     separator-invariant ('.', '_', '-', whitespace all unified), so names
#     differing only in punctuation or accent encoding collapse to one key.
#     """
#     if pd.isna(x):
#         return ""
#     s = str(x).strip()

#     # 1) Repair mojibake deterministically: ftfy if available, else fixed table.
#     try:
#         from ftfy import fix_text
#         s = fix_text(s)
#     except Exception:
#         for bad, good in {
#             "Ã¤": "ä", "Ã¶": "ö", "Ã¼": "ü", "ÃŸ": "ß",
#             "Ã„": "Ä", "Ã–": "Ö", "Ãœ": "Ü",
#             "√§": "ä", "√∂": "ö", "√º": "ü", "â¤": "ä",
#         }.items():
#             s = s.replace(bad, good)

#     # 2) Normalise + casefold + strip diacritics (ä->a, é->e): key no longer
#     #    depends on how the accent was encoded on disk.
#     s = unicodedata.normalize("NFKC", s).casefold()
#     s = "".join(ch for ch in unicodedata.normalize("NFKD", s)
#                 if not unicodedata.combining(ch))

#     # 3) Optional: drop a leading repeated-event index so per-instance source
#     #    variables align with a single base dictionary entry.
#     if strip_instance_prefix:
#         s = _INSTANCE_PREFIX_RE.sub("", s)

#     # 4) Unify ALL separators so 'hf.first.diagnosed' == 'hf_first_diagnosed'.
#     s = re.sub(r"[\s._\-]+", "_", s)
#     s = re.sub(r"[^a-z0-9_]+", "", s)
#     return s.strip("_").lower()


# Set style for all plots
PALETTE = {
    # Primary comparison pair (ground truth vs predicted)
    'primary':   '#2066A8',   # dark blue   (GeoDataViz sequential)
    'secondary': '#D4764E',   # muted terracotta / orange
    # Metric triad (accuracy / F1-weighted / F1-macro)
    'metric_1':  '#298C8C',   # teal        (image 2, colorblind-safe)
    'metric_2':  '#3594CC',   # medium blue (image 2 sequential)
    'metric_3':  '#EA801C',   # amber       (image 2 divergent pair)
    # Per-class metrics (precision / recall / F1)
    'precision': '#EA801C',   # amber
    'recall':    '#2066A8',   # dark blue
    'f1':        '#AF58BA',   # muted purple (GeoDataViz qualitative, CB-safe)
    # Supplementary / accent
    'accent':    '#AF58BA',   # muted purple
    'neutral':   '#888888',   # mid-gray
    'alert':     '#E8601C',   # deep orange  (avoids pure red)
    'good':      '#298C8C',   # teal         (avoids pure green)
    # Heatmap colormaps — colorblind-safe (Crameri 2024)
    'cmap_seq':  'YlGnBu',   # sequential   (blue-dominant, CB-safe)
    'cmap_div':  'PuOr',     # diverging    (purple-orange, CB-safe; avoids RdYlGn)
}

# Mode palette for consistent color coding across plots
MODE_COLORS = {
    'OO':  '#298C8C',   # teal
    'NE':  '#3594CC',   # blue
    'OEH': '#EA801C',   # amber
    'OEC': '#AF58BA',   # purple
    'OED': '#D4764E',   # terracotta
}

def _mode_color(mode: str) -> str:
    """Return a consistent color for a given mode, with fallback."""
    return MODE_COLORS.get(mode, PALETTE['neutral'])

# Set style for all plots
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.linewidth':    0.8,
    'axes.edgecolor':    '#333333',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})
sns.set_palette([PALETTE['primary'], PALETTE['secondary'], PALETTE['metric_1'],
                 PALETTE['accent'], PALETTE['metric_3']])

STUDY_FOLDER_ALIASES = {
    'gissi-hf_outcomes': 'gissi-hf',
}

# Evaluation must stay at the annotated variable-pair level.
# Dictionary labels are stored only as auxiliary/debug columns.
EVAL_KEY_COLS = ["source_study", "target_study", "src_var", "tgt_var"]

def _normalise_study_name(x) -> str:
    return str(x).strip().lower()

def ensure_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure human-readable display columns exist.
    Variable-level evaluation uses src_var/tgt_var as keys;
    src_var_orig/tgt_var_orig are only for output display.
    """
    d = df.copy()

    if "src_var_orig" not in d.columns:
        if "src_var_raw" in d.columns:
            d["src_var_orig"] = d["src_var_raw"]
        else:
            d["src_var_orig"] = d["src_var"]

    if "tgt_var_orig" not in d.columns:
        if "tgt_var_raw" in d.columns:
            d["tgt_var_orig"] = d["tgt_var_raw"]
        else:
            d["tgt_var_orig"] = d["tgt_var"]

    return d

def _require_columns(df: pd.DataFrame, cols: list[str], frame_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{frame_name} is missing required column(s): {missing}")

def _dedupe_eval_keys(df: pd.DataFrame, label_col: str | None = None, frame_name: str = "dataframe") -> pd.DataFrame:
    """Keep one row per variable-level evaluation key and log true key duplicates.

    This is not label-level collapsing. It only protects metric merges from
    accidental repeated rows with the same source/target study and raw variable key.
    """
    _require_columns(df, EVAL_KEY_COLS, frame_name)
    dup_mask = df.duplicated(EVAL_KEY_COLS, keep=False)
    if dup_mask.any():
        log_cols = [c for c in [
            "source_study", "target_study", "src_var_raw", "tgt_var_raw",
            "src_var", "tgt_var", label_col, "domain"
        ] if c and c in df.columns]
        logger.warning(
            f"{frame_name}: found {int(dup_mask.sum())} rows sharing the same "
            f"variable-level evaluation key {EVAL_KEY_COLS}; keeping the first row per key."
        )
        logger.warning("Duplicate variable-level rows:\n" + df.loc[dup_mask, log_cols].to_string(index=False))
    return df.drop_duplicates(EVAL_KEY_COLS, keep="first").reset_index(drop=True)

def normalize_text_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def split_joined_values(x):
    """
    Handles category strings like:
    - 'yes||no'
    - '0||1'
    - '0=no||1=yes'
    - ''
    """
    x = normalize_text_value(x)

    if not x or x in {"nan", "none", "null", "[]"}:
        return []

    # Support both || and | just in case
    if "||" in x:
        parts = x.split("||")
    elif "|" in x:
        parts = x.split("|")
    else:
        parts = [x]

    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        # If encoded as 0=no, keep the label part if available
        if "=" in p:
            left, right = p.split("=", 1)
            p = right.strip() if right.strip() else left.strip()

        cleaned.append(p)

    return cleaned


def canonical_category_set(x):
    values = split_joined_values(x)

    bool_aliases = {
        "yes": "1", "true": "1", "y": "1", "positive": "1", "present": "1", "on": "1",
        "no": "0", "false": "0", "n": "0", "negative": "0", "absent": "0", "off": "0",
    }

    return {bool_aliases.get(v.strip().lower(), v.strip().lower()) for v in values}


def infer_variable_structure(vartype, categories_labels=None, original_categories=None):
    """
    Infer statistical/structural variable type from raw datatype and categories.

    Returns:
    - continuous
    - binary
    - multiclass
    - qualitative
    - datetime
    - unknown
    """
    vt = normalize_text_value(vartype)

    cat_set = canonical_category_set(categories_labels)
    if not cat_set:
        cat_set = canonical_category_set(original_categories)

    n_cats = len(cat_set)

    # Datetime variables are structurally different from continuous measurements.
    if vt in {"datetime", "date", "timestamp"}:
        return "datetime"

    # Numeric variables can be continuous or encoded categorical.
    if vt in {"float", "float64", "double", "int", "int64", "integer", "numeric"}:
        if n_cats == 0:
            return "continuous"
        if n_cats == 2:
            return "binary"
        if n_cats > 2:
            return "multiclass"
        return "continuous"

    # String variables can be categorical or free text.
    if vt in {"str", "string", "object", "text"}:
        if n_cats == 2:
            return "binary"
        if n_cats > 2:
            return "multiclass"
        return "qualitative"

    # Fallback if vartype is missing but categories exist.
    if n_cats == 2:
        return "binary"
    if n_cats > 2:
        return "multiclass"

    return "unknown"


# def normalize_text_value(x):
#     if pd.isna(x):
#         return ""
#     return str(x).strip().lower()


def has_categories(x):
    x = normalize_text_value(x)
    return bool(x and x not in {"nan", "none", "null", "[]"})


def normalize_var_type(x):
    x = normalize_text_value(x)

    aliases = {
        "continuous_variable": "continuous",
        "continuous": "continuous",
        "numeric": "continuous",
        "float": "continuous",
        "integer": "continuous",

        "binary_class_variable": "binary",
        "binary": "binary",
        "boolean": "binary",

        "multi_class_variable": "multiclass",
        "multiclass": "multiclass",
        "multi-class": "multiclass",
        "categorical": "multiclass",

        "qualitative_variable": "qualitative",
        "qualitative": "qualitative",
    }

    return aliases.get(x, x)


def same_category_values(src_vals, tgt_vals):
    src = normalize_text_value(src_vals)
    tgt = normalize_text_value(tgt_vals)

    if not src or not tgt:
        return False

    src_set = {v.strip().lower() for v in src.split("||") if v.strip()}
    tgt_set = {v.strip().lower() for v in tgt.split("||") if v.strip()}

    bool_aliases = {
        "yes": "1", "true": "1", "y": "1", "positive": "1", "present": "1",
        "no": "0", "false": "0", "n": "0", "negative": "0", "absent": "0",
    }

    src_set = {bool_aliases.get(v, v) for v in src_set}
    tgt_set = {bool_aliases.get(v, v) for v in tgt_set}

    return src_set == tgt_set


def derive_structure_pattern(row):
    s_type = normalize_var_type(row.get("source_type", ""))
    t_type = normalize_var_type(row.get("target_type", ""))

    s_unit = normalize_text_value(row.get("source_unit", ""))
    t_unit = normalize_text_value(row.get("target_unit", ""))

    s_cats = row.get("source_categories_labels", "")
    t_cats = row.get("target_categories_labels", "")

    s_visit = normalize_text_value(row.get("source_visit", ""))
    t_visit = normalize_text_value(row.get("target_visit", ""))

    relation = normalize_text_value(row.get("mapping_relation", ""))
    source_context = normalize_text_value(row.get("source_composite_code_labels", ""))
    target_context = normalize_text_value(row.get("target_composite_code_labels", ""))

    # 1. Temporal mismatch should be captured separately because it affects harmonisation.
    if s_visit and t_visit and s_visit != t_visit:
        return "temporal-context mismatch"

    # 2. Medication class-member / hierarchy relations.
    if relation in {"symbolic:broadmatch", "symbolic:narrowmatch"}:
        return "hierarchical concept relation"

    # 3. Context asymmetry.
    if bool(source_context) != bool(target_context):
        return "context-asymmetric pair"

    # 4. Continuous--continuous.
    if s_type == "continuous" and t_type == "continuous":
        if s_unit and t_unit:
            if s_unit == t_unit:
                return "continuous--continuous, same unit"
            return "continuous--continuous, different unit"
        if bool(s_unit) != bool(t_unit):
            return "continuous--continuous, one-sided unit"
        return "continuous--continuous, no unit"

    # 5. Binary--binary.
    if s_type == "binary" and t_type == "binary":
        if same_category_values(s_cats, t_cats):
            return "binary--binary, equivalent value set"
        return "binary--binary, different value encoding"

    # 6. Binary--multiclass / multiclass--binary.
    if {s_type, t_type} == {"binary", "multiclass"}:
        return "binary--multiclass granularity mismatch"

    # 7. Multiclass--multiclass.
    if s_type == "multiclass" and t_type == "multiclass":
        if same_category_values(s_cats, t_cats):
            return "multiclass--multiclass, equivalent value set"
        return "multiclass--multiclass, different value set"

    # 8. Continuous--binary.
    if {s_type, t_type} == {"continuous", "binary"}:
        return "continuous--binary transformation"

    # 9. Continuous--multiclass.
    if {s_type, t_type} == {"continuous", "multiclass"}:
        return "continuous--multiclass transformation"

    # 10. Fallback.
    if s_type and t_type:
        return f"{s_type}--{t_type}"

    return "unclassified structure"
    
def load_dictionaries(cohorts_dir: str) -> dict[str, dict[str, str]]:
    """
    cohorts_dir/
      time-chf/*.csv
      gissi-hf/*.csv
      gissi-hf_outcomes/*.csv   # merged into 'gissi-hf'
      ...
    Each CSV must have columns 'variablename' and 'variablelabel' (case-insensitive).
    Returns: {study_name_lower: {variablename_lower: canonical_label}}
    """
    import glob
    if not os.path.isdir(cohorts_dir):
        raise FileNotFoundError(f"Cohorts directory not found: {cohorts_dir}")

    maps: dict[str, dict[str, str]] = {}
    for entry in sorted(os.listdir(cohorts_dir)):
        study_dir = os.path.join(cohorts_dir, entry)
        if not os.path.isdir(study_dir) or entry.startswith('.'):
            continue
        csvs = sorted(glob.glob(os.path.join(study_dir, '*.csv')))
        if not csvs:
            logger.warning(f"[{entry}] no CSV in {study_dir} — skipping")
            continue
        if len(csvs) > 1:
            logger.warning(f"[{entry}] {len(csvs)} CSVs, using first: {os.path.basename(csvs[0])}")
        path = csvs[0]

        for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
            try:
                df = pd.read_csv(path, encoding=enc); break
            except UnicodeDecodeError:
                continue
        df.columns = df.columns.str.strip().str.lower()
        if 'variablename' not in df.columns or 'variablelabel' not in df.columns:
            raise ValueError(f"[{entry}] {path}: expected 'variablename' and 'variablelabel', "
                             f"got {list(df.columns)}")

        # Route aliased folders onto their canonical study key and MERGE
        # (do not overwrite) so split dictionaries accumulate into one map.
        study_key = STUDY_FOLDER_ALIASES.get(entry.lower(), entry.lower())
        m = maps.setdefault(study_key, {})

        added = 0
        for _, r in df[['variablename', 'variablelabel']].iterrows():
            name = canonical_var_key(r['variablename'])
            raw   = str(r['variablelabel'] or '').strip().lower()
            label = clean_label_remove_temporal_context(raw) or name
            label = label.replace(" ", "_")
            if name in m and m[name] != label:
                logger.warning(f"[{entry}] label conflict for '{name}': "
                               f"'{m[name]}' vs '{label}' — keeping first")
            else:
                if name not in m:
                    added += 1
                m.setdefault(name, label)

        logger.info(f"[{entry}] loaded {added} entries from {path} "
                    f"→ study key '{study_key}' (total {len(m)})")

    if not maps:
        raise FileNotFoundError(f"No study dictionaries found under {cohorts_dir}")
    return maps
    


def add_label_keys(df: pd.DataFrame, s_map: dict, t_map: dict) -> pd.DataFrame:
    """Attach dictionary-label keys without changing the evaluation identity.

    src_var/tgt_var remain canonical variable-name keys. src_label_key/tgt_label_key
    are only for debugging, reporting label-level collisions, or optional concept-level
    analyses.
    """
    d = df.copy()
    d["src_label_key"] = d["src_var"].map(lambda x: s_map.get(x, x))
    d["tgt_label_key"] = d["tgt_var"].map(lambda x: t_map.get(x, x))
    return d


def log_label_level_collisions(df: pd.DataFrame, label_col: str, frame_name: str) -> None:
    """Log label-level collisions without dropping rows."""
    keys = [c for c in ["source_study", "target_study", "src_label_key", "tgt_label_key"] if c in df.columns]
    if not keys:
        return
    dup_mask = df.duplicated(keys, keep=False)
    if not dup_mask.any():
        return
    log_cols = [c for c in [
        "source_study", "target_study", "src_var_raw", "tgt_var_raw",
        "src_var", "tgt_var", "src_label_key", "tgt_label_key", label_col, "domain"
    ] if c in df.columns]
    logger.warning(
        f"{frame_name}: found {int(dup_mask.sum())} rows involved in label-level collisions "
        f"after dictionary-label mapping. These rows are preserved for variable-level evaluation. "
        f"Collision keys={keys}; label_col='{label_col}'."
    )
    # logger.warning("Label-level collision rows preserved:\n" + df.loc[dup_mask, log_cols].to_string(index=False))


def _collapse_by_label(df, label_col, s_map, t_map, rank=None):
    rank = rank or {
        'identical match': 0,
        # 'complete match': 1,
        'compatible match': 1,
        'partial match': 2,
        'not applicable': 3
    }

    d = df.assign(
        src_key=df['src_var'].map(lambda x: s_map.get(x, x)),
        tgt_key=df['tgt_var'].map(lambda x: t_map.get(x, x)),
        _rank=df[label_col].map(rank).fillna(99),
    )

    # Sort first so the best-ranked harmonization label is kept.
    d_sorted = d.sort_values('_rank', kind='mergesort').copy()

    # Important: collapse inside each study pair, not globally across all studies.
    collapse_keys = [
        c for c in [
            'source_study',
            'target_study',
            'src_key',
            'tgt_key'
        ]
        if c in d_sorted.columns
    ]

    dup_mask = d_sorted.duplicated(collapse_keys, keep='first')
    skipped = d_sorted.loc[dup_mask].copy()
    kept = d_sorted.loc[~dup_mask].copy()

    if not skipped.empty:
        log_cols = [
            c for c in [
                'source_study',
                'target_study',
                'src_var_raw',
                'tgt_var_raw',
                'src_var',
                'tgt_var',
                'src_key',
                'tgt_key',
                label_col,
                '_rank',
                'domain'
            ]
            if c in skipped.columns
        ]

        logger.warning(
            f"_collapse_by_label found {len(skipped)} label-level collisions after dictionary-label mapping. "
            f"These are not necessarily duplicate raw GT/prediction variable pairs. "
            f"Collision keys={collapse_keys}; label_col='{label_col}'."
        )
        logger.warning(
            "Skipped collapsed rows:\n"
            + skipped[log_cols].to_string(index=False)
        )

        collapsed_summary = (
            d_sorted
            .groupby(collapse_keys)
            .size()
            .reset_index(name='n_rows_before_collapse')
            .query("n_rows_before_collapse > 1")
            .sort_values('n_rows_before_collapse', ascending=False)
        )

        logger.warning(
    "Label-level collision summary after dictionary-label mapping:\n"
    + collapsed_summary.to_string(index=False)
)

    d = (
        kept
        .drop(columns='_rank')
        .rename(columns={
            'src_var': 'src_var_orig',
            'tgt_var': 'tgt_var_orig',
            'src_key': 'src_var',
            'tgt_key': 'tgt_var'
        })
        .reset_index(drop=True)
    )

    # Use raw display names if present.
    if 'src_var_raw' in d.columns:
        d['src_var_orig'] = d['src_var_raw']
    if 'tgt_var_raw' in d.columns:
        d['tgt_var_orig'] = d['tgt_var_raw']

    base_cols = ['src_var', 'tgt_var', 'src_var_orig', 'tgt_var_orig', label_col]

    extra_cols = [
        c for c in [
            'source_study',
            'target_study',
            'domain'
        ]
        if c in d.columns
    ]

    return d[base_cols + extra_cols]





def load_predictions(file_path: str, s_map: dict, t_map: dict,
                     source_study: str | None = None,
                     target_study: str | None = None,
                     collapse_temporal: bool = True) -> pd.DataFrame:
    """Read predictions at variable-pair level.

    collapse_temporal is kept for backwards-compatible calls but no longer
    triggers dictionary-label collapse. Main evaluation must use
    source_study + target_study + canonical source variable + canonical target variable.
    """
    for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
        try:
            p_df = pd.read_csv(file_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        p_df = pd.read_csv(file_path, encoding='utf-8', encoding_errors="replace")

    # Normalize column names once.
    p_df.columns = p_df.columns.str.strip().str.lower()

    # These are the structural columns needed for variable-structure error analysis.
    structure_cols = [
        "source_type", "target_type",
        "source_unit", "target_unit",
        "source_categories_labels", "target_categories_labels",
        "source_original_categories", "target_original_categories",
        "source_categories_omop_ids", "target_categories_omop_ids",
        "source_visit", "target_visit",
        "mapping_relation", "context_match_type",
        "source_composite_code_labels", "target_composite_code_labels",
        "source_composite_code_omop_ids", "target_composite_code_omop_ids",
        "category", "source_label", "target_label",
        "slabel", "tlabel", "somop_id", "tomop_id",
        "scode", "tcode", "sim_score",
    ]

    _require_columns(p_df, ["source", "target", "harmonization_status"], "prediction file")

    pred = pd.DataFrame({
        'source_study': _normalise_study_name(source_study) if source_study is not None else "",
        'target_study': _normalise_study_name(target_study) if target_study is not None else "",
        'src_var': p_df['source'].map(canonical_var_key),
        'tgt_var': p_df['target'].map(canonical_var_key),
        'src_var_raw': p_df['source'].astype(str),
        'tgt_var_raw': p_df['target'].astype(str),
        'predicted class': p_df['harmonization_status'].astype(str).str.strip().str.lower(),
    })

    # Attach structural metadata from the prediction file.
    for col in [c for c in structure_cols if c in p_df.columns]:
        pred[col] = p_df[col]

    if source_study is None or target_study is None:
        logger.warning(
            "load_predictions was called without source_study/target_study. "
            "Study-aware evaluation keys will be incomplete."
        )

    missing_s = set(pred['src_var']) - set(s_map)
    missing_t = set(pred['tgt_var']) - set(t_map)

    if missing_s:
        logger.warning(
            f"{len(missing_s)} prediction source names absent from source dictionary: "
            f"{sorted(missing_s)[:10]}"
        )
    if missing_t:
        logger.warning(
            f"{len(missing_t)} prediction target names absent from target dictionary: "
            f"{sorted(missing_t)[:10]}"
        )

    pred = add_label_keys(pred, s_map, t_map)
    pred = ensure_display_columns(pred)
    log_label_level_collisions(pred, 'predicted class', 'predictions')
    pred = _dedupe_eval_keys(pred, label_col='predicted class', frame_name='predictions')
    return pred

def load_ground_truth(file_path: str, source_study: str, target_study: str,
                      s_map: dict | None = None, t_map: dict | None = None,
                      collapse_temporal: bool = True) -> pd.DataFrame:

    t_df = pd.read_excel(file_path, sheet_name=0)
    t_df = t_df.drop_duplicates(
    subset=["source_study", "target_study", "source_var_name", "target_var_name"],
    keep="first"
)
    logger.info(f"before collaspe GT total pairs: {len(t_df)} | study pairs:\n"
                f"{t_df.groupby(['source_study','target_study'])['harmonization level'].value_counts()}")

    src_l, tgt_l = source_study.lower(), target_study.lower()

    mask = (t_df['source_study'].str.strip().str.lower() == src_l) & \
           (t_df['target_study'].str.strip().str.lower() == tgt_l)
    gt = pd.DataFrame({
    'source_study': t_df.loc[mask, 'source_study'].astype(str).str.strip().str.lower(),
    'target_study': t_df.loc[mask, 'target_study'].astype(str).str.strip().str.lower(),

    'src_var': t_df.loc[mask, 'source_var_name'].map(canonical_var_key),
    'tgt_var': t_df.loc[mask, 'target_var_name'].map(canonical_var_key),
    'src_var_raw': t_df.loc[mask, 'source_var_name'].astype(str),
    'tgt_var_raw': t_df.loc[mask, 'target_var_name'].astype(str),
    'correct class': t_df.loc[mask, 'harmonization level'].astype(str).str.strip().str.lower(),
    'domain': t_df.loc[mask, 'domain'].astype(str).str.strip().str.lower(),
}).reset_index(drop=True)
    if not collapse_temporal or s_map is None:
        return gt
    missing_s = set(gt['src_var']) - set(s_map)
    missing_t = set(gt['tgt_var']) - set(t_map)
    if missing_s: logger.warning(f"{len(missing_s)} GT source names absent from dictionaries: {sorted(missing_s)[:10]}")
    if missing_t: logger.warning(f"{len(missing_t)} GT target names absent from dictionaries: {sorted(missing_t)[:10]}")

    d_final = add_label_keys(gt, s_map, t_map)
    d_final = ensure_display_columns(d_final)
    log_label_level_collisions(d_final, 'correct class', 'ground truth')
    d_final = _dedupe_eval_keys(d_final, label_col='correct class', frame_name='ground truth')

    logger.info(f"after variable-level GT load total pairs: {len(d_final)} | study pairs:\n"
                f"{d_final.groupby(['source_study','target_study'])['correct class'].value_counts()}")
    logger.info(f"ground truth total pairs are {len(d_final)}")
    return d_final



def analyze_class_distribution(ground_truth: pd.DataFrame, predictions: pd.DataFrame,
                                study_pair: str = "") -> dict:
    """Analyze and compare class distributions between ground truth and predictions."""
    _require_columns(ground_truth, EVAL_KEY_COLS + ['correct class'], 'ground_truth')
    _require_columns(predictions, EVAL_KEY_COLS + ['predicted class'], 'predictions')

    gt_m = ground_truth[EVAL_KEY_COLS + ['correct class']]
    pr_m = predictions[EVAL_KEY_COLS + ['predicted class']]
    merged_df = pd.merge(gt_m, pr_m, on=EVAL_KEY_COLS, how='left')
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')

    gt_dist = ground_truth['correct class'].value_counts()
    pred_dist = merged_df['predicted class'].value_counts()
    all_classes = sorted(set(gt_dist.index) | set(pred_dist.index))

    comparison = pd.DataFrame({
        'class': all_classes,
        'gt_count': [gt_dist.get(c, 0) for c in all_classes],
        'pred_count': [pred_dist.get(c, 0) for c in all_classes],
    })
    comparison['gt_pct'] = (comparison['gt_count'] / comparison['gt_count'].sum() * 100).round(1)
    comparison['pred_pct'] = (comparison['pred_count'] / comparison['pred_count'].sum() * 100).round(1)
    comparison['diff_pct'] = (comparison['pred_pct'] - comparison['gt_pct']).round(1)

    total_samples = len(ground_truth)
    min_class_count = gt_dist.min() if len(gt_dist) > 0 else 0
    max_class_count = gt_dist.max() if len(gt_dist) > 0 else 0
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

    warnings_list = []
    for cls in all_classes:
        count = gt_dist.get(cls, 0)
        if count < 5:
            warnings_list.append(f"⚠️  Class '{cls}' has only {count} samples - metrics unreliable")
    if imbalance_ratio > 5:
        warnings_list.append(f"⚠️  High class imbalance (ratio {imbalance_ratio:.1f}:1) - consider macro-averaged metrics")

    return {
        'comparison': comparison,
        'gt_distribution': gt_dist,
        'pred_distribution': pred_dist,
        'imbalance_ratio': imbalance_ratio,
        'total_samples': total_samples,
        'warnings': warnings_list,
        'study_pair': study_pair
    }


def compute_comprehensive_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Compute comprehensive metrics at variable-pair level."""
    _require_columns(ground_truth, EVAL_KEY_COLS + ['correct class'], 'ground_truth')
    _require_columns(predictions, EVAL_KEY_COLS + ['predicted class'], 'predictions')

    gt_m = ground_truth[EVAL_KEY_COLS + ['correct class']]
    pr_m = predictions[EVAL_KEY_COLS + ['predicted class']]
    merged_df = pd.merge(gt_m, pr_m, on=EVAL_KEY_COLS, how='left')
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')

    y_true = merged_df['correct class']
    y_pred = merged_df['predicted class']
    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class_metrics = pd.DataFrame({
        'class': labels,
        'precision': precision.round(3),
        'recall': recall.round(3),
        'f1_score': f1.round(3),
        'support': support
    })

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'labels': labels,
        'y_true': y_true,
        'y_pred': y_pred,
        'classification_report': report
    }

def compute_domain_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(ground_truth, EVAL_KEY_COLS + ['correct class', 'domain'], 'ground_truth')
    _require_columns(predictions, EVAL_KEY_COLS + ['predicted class'], 'predictions')

    gt_m = ground_truth[EVAL_KEY_COLS + ['correct class', 'domain']]
    pr_m = predictions[EVAL_KEY_COLS + ['predicted class']]

    merged = pd.merge(gt_m, pr_m, on=EVAL_KEY_COLS, how='left')
    merged['predicted class'] = merged['predicted class'].fillna('not applicable')
    merged['domain'] = merged['domain'].fillna('unknown').astype(str).str.strip().str.lower()

    rows = []

    for domain, g in merged.groupby('domain'):
        y_true = g['correct class']
        y_pred = g['predicted class']

        rows.append({
            'domain': domain,
            'n_pairs': len(g),
            'n_classes_present': y_true.nunique(),
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'f1_weighted': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
            'f1_micro': round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4),
            'class_distribution': dict(y_true.value_counts())
        })

    return pd.DataFrame(rows)

def compute_collective_metrics(collective_data: list, source_study: str) -> pd.DataFrame:
    """
    Across-target summary for each (model, mode), computed as the unweighted MEAN
    of per-target metrics (each target study weighted equally). This matches the
    aggregation used in the comparison figures (mean across study pairs), so the
    summary table and the figures report the same numbers.
    """
    from collections import OrderedDict
    buckets = OrderedDict()
    for d in collective_data:
        key = (d['model'], d['mode'])
        b = buckets.setdefault(key, {'yt': [], 'yp': [], 'targets': []})
        b['yt'].append(pd.Series(d['y_true']).reset_index(drop=True))
        b['yp'].append(pd.Series(d['y_pred']).reset_index(drop=True))
        b['targets'].append(d.get('target_study'))

    rows = []
    for (model, mode), b in buckets.items():
        acc, f1m, f1w, f1mi, n_pairs = [], [], [], [], 0
        for yt, yp in zip(b['yt'], b['yp']):
            acc.append(accuracy_score(yt, yp))
            f1m.append(f1_score(yt, yp, average='macro',    zero_division=0))
            f1w.append(f1_score(yt, yp, average='weighted', zero_division=0))
            f1mi.append(f1_score(yt, yp, average='micro',   zero_division=0))
            n_pairs += len(yt)
        rows.append({
            'model': model,
            'mode': mode,
            'source_study': source_study,
            'target_scope': 'MEAN_OF_TARGETS',
            'n_targets': len([t for t in b['targets'] if t is not None]),
            'n_pairs': int(n_pairs),
            'accuracy':    round(float(np.mean(acc)),  4),
            'f1_macro':    round(float(np.mean(f1m)),  4),
            'f1_weighted': round(float(np.mean(f1w)),  4),
            'f1_micro':    round(float(np.mean(f1mi)), 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('f1_macro', ascending=False).reset_index(drop=True)
    return df


POSITIVE_GT_CLASSES = {
    "identical match",
    # "complete match",
    "compatible match",
    "partial match",
}


def _pair_set(df: pd.DataFrame) -> set[tuple]:
    _require_columns(df, EVAL_KEY_COLS, 'pair dataframe')
    return set(df[EVAL_KEY_COLS].itertuples(index=False, name=None))

def _label_pair_set(df: pd.DataFrame) -> set[tuple]:
    """
    Diagnostic-only pair set using temporal-neutral dictionary labels.

    Used for predicted_not_in_ground_truth only.
    Do not use this for the main metric merge.
    """
    _require_columns(
        df,
        ["source_study", "target_study", "src_label_key", "tgt_label_key"],
        "label-pair dataframe"
    )

    return set(
        df[["source_study", "target_study", "src_label_key", "tgt_label_key"]]
        .itertuples(index=False, name=None)
    )


def _rows_for_pairs(df: pd.DataFrame, pairs: set[tuple]) -> pd.DataFrame:
    if df.empty or not pairs:
        return df.iloc[0:0].copy()

    _require_columns(df, EVAL_KEY_COLS, 'pair dataframe')
    pair_index = pd.MultiIndex.from_tuples(list(pairs), names=EVAL_KEY_COLS)
    row_index = pd.MultiIndex.from_frame(df[EVAL_KEY_COLS])
    mask = row_index.isin(pair_index)
    return df.loc[mask].copy()


def _display_pair_df(df: pd.DataFrame) -> pd.DataFrame:
    name_rename = {"src_var_orig": "src_var", "tgt_var_orig": "tgt_var"}
    return (
        df.drop(columns=["src_var", "tgt_var"], errors="ignore")
          .rename(columns=name_rename)
          .copy()
    )

def summarize_errors_by_structure(merged):
    label_rank = {
        "identical match": 0,
        "compatible match": 1,
        "partial match": 2,
        "not applicable": 3,
    }

    merged = merged.copy()

    merged["true_rank"] = merged["correct class"].map(label_rank)
    merged["pred_rank"] = merged["predicted class"].map(label_rank)
    merged["error_delta"] = merged["pred_rank"] - merged["true_rank"]

    merged["is_correct"] = merged["correct class"] == merged["predicted class"]
    merged["over_harmonisation"] = merged["error_delta"] < 0
    merged["under_harmonisation"] = merged["error_delta"] > 0
    merged["severe_error"] = merged["error_delta"].abs() >= 2

    summary = (
        merged
        .groupby("structure_pattern")
        .agg(
            n_pairs=("structure_pattern", "size"),
            accuracy=("is_correct", "mean"),
            over_harmonisation_rate=("over_harmonisation", "mean"),
            under_harmonisation_rate=("under_harmonisation", "mean"),
            severe_error_rate=("severe_error", "mean"),
        )
        .reset_index()
    )

    for col in [
        "accuracy",
        "over_harmonisation_rate",
        "under_harmonisation_rate",
        "severe_error_rate",
    ]:
        summary[col] = summary[col].round(3)

    summary = summary.sort_values(
        ["severe_error_rate", "over_harmonisation_rate", "n_pairs"],
        ascending=[False, False, False],
    )

    return summary


def evaluate_predictions(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    data_dir: str,
    predict_studies_names: str,
    source_study: str,
    target_study: str
) -> dict:
    """Evaluate final classification and variable-level candidate-generation coverage.

    - Classification metrics use all GT variable pairs.
    - Candidate-retrieval missing-GT report uses only positive/harmonizable GT pairs.
    - Predicted-not-in-GT remains against the full GT annotation set.
    """
    _require_columns(ground_truth, EVAL_KEY_COLS + ["src_var_orig", "tgt_var_orig", "correct class"], 'ground_truth')
    _require_columns(predictions, EVAL_KEY_COLS + ["src_var_orig", "tgt_var_orig", "predicted class"], 'predictions')

    gt_cols = EVAL_KEY_COLS + [
        c for c in ground_truth.columns
        if c not in EVAL_KEY_COLS and c in {
            "src_var_orig", "tgt_var_orig", "correct class", "domain",
            "src_label_key", "tgt_label_key"
        }
    ]
    pred_cols = EVAL_KEY_COLS + [c for c in predictions.columns if c not in EVAL_KEY_COLS]

    gt_m = ground_truth[gt_cols]
    pr_m = predictions[pred_cols]

    merged = pd.merge(
        gt_m,
        pr_m,
        on=EVAL_KEY_COLS,
        how="left",
        suffixes=("_gt", "_pred"),
    )

    merged["predicted class"] = merged["predicted class"].fillna("not applicable")
    merged["structure_pattern"] = merged.apply(derive_structure_pattern, axis=1)
    structure_summary = summarize_errors_by_structure(merged)

    structure_summary.to_csv(
        f"{data_dir}/{predict_studies_names}_error_by_variable_structure.csv",
        index=False
    )
    total = len(merged)
    correct = int((merged["correct class"] == merged["predicted class"]).sum())
    accuracy = correct / total if total else 0.0

    all_incorrect_cols = [
        c for c in [
            "source_study", "target_study", "src_var_orig_gt", "tgt_var_orig_gt",
            "src_var", "tgt_var", "src_label_key_gt", "tgt_label_key_gt",
            "correct class", "predicted class", "structure_pattern"
        ] if c in merged.columns
    ]

    all_incorrect = (
        merged.loc[merged["correct class"] != merged["predicted class"], all_incorrect_cols]
        .rename(columns={
            "src_var_orig_gt": "src_var_raw",
            "tgt_var_orig_gt": "tgt_var_raw",
        })
        .copy()
    )

    all_incorrect.to_csv(
        f"{data_dir}/incorrect_predictions_{predict_studies_names}.csv",
        encoding="utf-8",
        index=False,
    )

    logger.info("Incorrect Predictions:")
    logger.info(all_incorrect)

    # ------------------------------------------------------------------
    # Diagnostic 1:
    # Predicted pairs outside the GT reference.
    #
    # Use temporal-neutral label-pair keys here, not raw variable names.
    # This avoids counting follow-up variables as outside GT when the GT
    # contains only the baseline version.
    # ------------------------------------------------------------------

    gt_label_pairs = _label_pair_set(ground_truth)
    pred_label_pairs = _label_pair_set(predictions)

    pred_tmp = predictions.copy()
    pred_tmp["__label_pair"] = list(
        pred_tmp[["source_study", "target_study", "src_label_key", "tgt_label_key"]]
        .itertuples(index=False, name=None)
    )

    predicted_not_in_gt_pairs = pred_label_pairs - gt_label_pairs

    not_in_gt_df = pred_tmp[
        pred_tmp["__label_pair"].isin(predicted_not_in_gt_pairs)
    ].drop(columns="__label_pair")

    not_in_gt_df = _display_pair_df(not_in_gt_df)


    # ------------------------------------------------------------------
    # Diagnostic 2:
    # Positive GT pairs not retrieved by candidate generation.
    #
    # Also use temporal-neutral label-pair keys here. Otherwise, a baseline
    # GT pair can be falsely counted as "not retrieved" when only its
    # follow-up versions were predicted.
    # ------------------------------------------------------------------

    gt_positive = ground_truth[
        ground_truth["correct class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(POSITIVE_GT_CLASSES)
    ].copy()

    gt_positive_label_pairs = _label_pair_set(gt_positive)
    positive_gt_not_retrieved_pairs = gt_positive_label_pairs - pred_label_pairs

    gt_positive_tmp = gt_positive.copy()
    gt_positive_tmp["__label_pair"] = list(
        gt_positive_tmp[["source_study", "target_study", "src_label_key", "tgt_label_key"]]
        .itertuples(index=False, name=None)
    )

    not_in_pred_df = gt_positive_tmp[
        gt_positive_tmp["__label_pair"].isin(positive_gt_not_retrieved_pairs)
    ].drop(columns="__label_pair")

    not_in_pred_df = _display_pair_df(not_in_pred_df)

    positive_candidate_recall = (
        1.0 - len(positive_gt_not_retrieved_pairs) / len(gt_positive_label_pairs)
        if gt_positive_label_pairs else 0.0
    )


    # ------------------------------------------------------------------
    # Save diagnostic files
    # ------------------------------------------------------------------

    not_in_gt_df.to_csv(
        f"{data_dir}/predicted_not_in_ground_truth_{predict_studies_names}.csv",
        encoding="utf-8",
        index=False,
    )

    not_in_pred_df.to_csv(
        f"{data_dir}/positive_gt_not_retrieved_{predict_studies_names}.csv",
        encoding="utf-8",
        index=False,
    )


    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    logger.info(
        f"\n{'=' * 60}\n"
        f"Predicted variable pairs NOT in ground truth: {len(not_in_gt_df)} rows "
        f"({len(predicted_not_in_gt_pairs)} unique temporal-neutral label pairs)\n"
        f"Saved to: {data_dir}/predicted_not_in_ground_truth_{predict_studies_names}.csv\n"
        f"{'=' * 60}"
    )

    logger.info(
        f"\n{'=' * 60}\n"
        f"Positive GT variable pairs NOT retrieved: {len(not_in_pred_df)} rows "
        f"({len(positive_gt_not_retrieved_pairs)} unique temporal-neutral label pairs)\n"
        f"Positive candidate recall: {positive_candidate_recall:.3f}\n"
        f"Saved to: {data_dir}/positive_gt_not_retrieved_{predict_studies_names}.csv\n"
        f"{'=' * 60}"
    )
    # logger.info(not_in_pred_df.to_string(index=False))

    return {
        "source_study": source_study,
        "target_study": target_study,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "incorrect_predictions": all_incorrect,
        "not_in_ground_truth": not_in_gt_df,
        "not_in_predictions": not_in_pred_df,
        "positive_candidate_recall": positive_candidate_recall,

        # temporal-neutral diagnostic counts
        "n_positive_gt_pairs": len(gt_positive_label_pairs),
        "n_positive_gt_not_retrieved": len(positive_gt_not_retrieved_pairs),
        "n_predicted_not_in_ground_truth": len(predicted_not_in_gt_pairs),
    }
def plot_class_distribution_comparison(dist_results: dict, output_path: str = None):
    """Create side-by-side bar chart comparing GT vs Predicted class distributions."""
    comparison = dist_results['comparison']
    study_pair = dist_results.get('study_pair', '')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(comparison))
    width = 0.35
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, comparison['gt_count'], width, label='Ground Truth', color=PALETTE['primary'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, comparison['pred_count'], width, label='Predicted', color=PALETTE['secondary'], alpha=0.8)
    ax1.set_xlabel('Harmonization Level', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Class Distribution: Count\n{study_pair}', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison['class'], rotation=45, ha='right')
    ax1.legend()
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, comparison['gt_pct'], width, label='Ground Truth %', color=PALETTE['primary'], alpha=0.8)
    bars4 = ax2.bar(x + width/2, comparison['pred_pct'], width, label='Predicted %', color=PALETTE['secondary'], alpha=0.8)
    ax2.set_xlabel('Harmonization Level', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title(f'Class Distribution: Percentage\n{study_pair}', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison['class'], rotation=45, ha='right')
    ax2.legend()
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plot to: {output_path}")
    plt.close()
    return fig


def plot_confusion_matrix(metrics: dict, study_pair: str = "", output_path: str = None):
    """Plot confusion matrix heatmap."""
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap=PALETTE['cmap_seq'], xticklabels=labels, 
                yticklabels=labels, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Ground Truth', fontsize=11)
    ax1.set_title(f'Confusion Matrix (Counts)\n{study_pair}', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap=PALETTE['cmap_div'], xticklabels=labels, 
                yticklabels=labels, ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Recall'})
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Ground Truth', fontsize=11)
    ax2.set_title(f'Confusion Matrix (Normalized = Recall)\n{study_pair}', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to: {output_path}")
    plt.close()
    return fig


def plot_per_class_metrics(metrics: dict, study_pair: str = "", output_path: str = None):
    """Plot per-class precision, recall, F1 as grouped bar chart."""
    pcm = metrics['per_class_metrics']
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pcm))
    width = 0.25
    
    ax.bar(x - width, pcm['precision'], width, label='Precision', color=PALETTE['precision'], alpha=0.8)
    ax.bar(x, pcm['recall'], width, label='Recall', color=PALETTE['recall'], alpha=0.8)
    ax.bar(x + width, pcm['f1_score'], width, label='F1 Score', color=PALETTE['f1'], alpha=0.8)
    
    ax.set_xlabel('Harmonization Level', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Per-Class Metrics\n{study_pair}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pcm['class'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    
    for i, (cls, support) in enumerate(zip(pcm['class'], pcm['support'])):
        ax.annotate(f'n={support}', xy=(i, -0.08), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=9, color=PALETTE['neutral'])
    
    ax.axhline(y=0.5, color=PALETTE['neutral'], linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved per-class metrics to: {output_path}")
    plt.close()
    return fig


def plot_multi_study_comparison(all_study_metrics: list, output_path: str = None):
    """Create comparison visualization across multiple study pairs."""
    study_names = [m['study_pair'] for m in all_study_metrics]
    f1_macro = [m['metrics']['f1_macro'] for m in all_study_metrics]
    imbalance_ratios = [m['distribution']['imbalance_ratio'] for m in all_study_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    x = np.arange(len(study_names))
    bars = ax1.bar(x, f1_macro, color=PALETTE['metric_2'], alpha=0.85)
    ax1.set_ylabel('F1 (Macro)', fontsize=11)
    ax1.set_title('F1 (Macro) by Study Pair', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(study_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.5, linewidth=0.8)
    for bar, val in zip(bars, f1_macro):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[1]
    colors = [PALETTE['alert'] if r > 5 else PALETTE['good'] for r in imbalance_ratios]
    bars = ax2.bar(study_names, imbalance_ratios, color=colors, alpha=0.8)
    ax2.set_ylabel('Imbalance Ratio', fontsize=11)
    ax2.set_title('Class Imbalance Ratio by Study Pair\n(Red = High Imbalance > 5:1)', fontsize=12, fontweight='bold')
    ax2.axhline(y=5, color=PALETTE['alert'], linestyle='--', alpha=0.7, linewidth=1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    for bar, ratio in zip(bars, imbalance_ratios):
        ax2.annotate(f'{ratio:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    ax3 = axes[2]
    ax3.axis('off')
    table_data = []
    for m in all_study_metrics:
        table_data.append([
            m['study_pair'],
            f"{m['metrics']['f1_macro']:.3f}",
            f"{m['distribution']['imbalance_ratio']:.1f}",
            f"{m['distribution']['total_samples']}"
        ])
    table = ax3.table(
        cellText=table_data,
        colLabels=['Study Pair', 'F1(M)', 'Imbalance', 'N'],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax3.set_title('Summary Table', fontsize=12, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-study comparison to: {output_path}")
    plt.close()
    return fig


# ============================================================================
# CROSS-MODEL / CROSS-MODE COMPARISON FUNCTIONS
# ============================================================================

def plot_metric_heatmap(master_df: pd.DataFrame, metric: str, output_path: str = None,
                        title: str = None):
    """
    Plot a heatmap of a given metric with rows = (model, mode) and columns = study pair.
    OO rows are labelled as 'OO (baseline)' instead of showing model 'N/A'.
    """
    df = master_df.copy()
    # Relabel OO so it reads as a baseline, not as a model
    df.loc[df['mode'] == 'OO', 'model'] = 'OO (baseline)'
    df.loc[df['mode'] == 'OO', 'mode'] = '—'

    pivot = df.pivot_table(
        index=['model', 'mode'], columns='study_pair', values=metric, aggfunc='first'
    )
    # Sort rows by mean metric value (best at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2.5), max(6, len(pivot) * 0.7)))
    
    sns.heatmap(pivot.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    ax.set_title(title or f'{metric.replace("_", " ").title()} by Model × Mode × Study Pair',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Model / Mode', fontsize=11)
    ax.set_xlabel('Study Pair', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved heatmap to: {output_path}")
    plt.close()
    return fig


def plot_aggregate_comparison(master_df: pd.DataFrame, output_path: str = None):
    """
    Single horizontal grouped-bar chart comparing all models across all modes
    by F1 (Weighted), averaged across study pairs.
    Rows = models (sorted best→worst), bars grouped/coloured by mode.
    OO is model-independent: shown as a separate hatched bar at the bottom.
    """
    # Separate OO baseline from model-specific results
    oo_df = master_df[master_df['mode'] == 'OO']
    model_df = master_df[(master_df['mode'] != 'OO') & (master_df['model'] != 'N/A')]
    has_oo = not oo_df.empty
    logger.info(f"  plot_aggregate_comparison: OO rows in master_df = {len(oo_df)} "
                f"({'will show baseline' if has_oo else 'OO NOT FOUND — check file discovery'})")

    agg = model_df.groupby(['model', 'mode']).agg(
        f1_macro=('f1_macro', 'mean'),
    ).reset_index()

    if agg.empty:
        logger.info("  No model-specific results to plot in aggregate comparison.")
        return None

    models = (agg.groupby('model')['f1_macro'].mean()
                  .sort_values(ascending=True).index.tolist())
    modes  = sorted(agg['mode'].unique())
    n_modes = len(modes)

    has_oo = not oo_df.empty
    gap = 0.15  # visual gap between OO and model rows

    bar_h   = 0.8 / max(n_modes, 1)
    # Model rows start after the OO row + gap
    if has_oo:
        y_base = np.arange(len(models)) + 1 + gap
    else:
        y_base = np.arange(len(models))

    fig, ax = plt.subplots(
        figsize=(10, max(4, (len(models) + has_oo) * 0.65 + 1.5))
    )

    # --- OO bar (single, hatched) ---
    if has_oo:
        oo_mean = oo_df['f1_macro'].mean()
        ax.barh(0, oo_mean, height=0.5,
                color=_mode_color('OO'), alpha=0.85,
                edgecolor='white', linewidth=0.3,
                hatch='///', label='OO (no model)')
        ax.text(oo_mean + 0.006, 0, f'{oo_mean:.3f}',
                va='center', ha='left', fontsize=8,
                fontweight='bold', color=_mode_color('OO'))

    # --- Model-specific bars ---
    for i, mode in enumerate(modes):
        mode_data = agg[agg['mode'] == mode].set_index('model')
        vals = [mode_data.loc[m, 'f1_macro'] if m in mode_data.index else 0
                for m in models]
        offset = (i - (n_modes - 1) / 2) * bar_h
        bars = ax.barh(y_base + offset, vals, height=bar_h,
                       label=mode, color=_mode_color(mode),
                       alpha=0.85, edgecolor='white', linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(v + 0.006, bar.get_y() + bar.get_height() / 2,
                        f'{v:.3f}', va='center', ha='left', fontsize=7.5,
                        fontweight='bold', color=_mode_color(mode))

    # --- Y-axis ---
    all_ticks = ([0] if has_oo else []) + y_base.tolist()
    all_labels = (['OO (baseline)'] if has_oo else []) + models
    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.set_xlabel('F1 (Macro)', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.axvline(x=0.8, color=PALETTE['good'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.legend(title='Mode', fontsize=9, title_fontsize=10,
              loc='lower right', framealpha=0.9)
    ax.set_title('F1 (Macro) by Model × Mode  (avg. across study pairs)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved aggregate comparison to: {output_path}")
    plt.close()
    return fig


def plot_mode_comparison_per_model(master_df: pd.DataFrame, output_path: str = None):
    """
    Single vertical grouped-bar chart comparing all modes across all models
    by F1 (Macro), averaged across study pairs.
    X-axis = modes, bars grouped/coloured by model.
    OO is model-independent: shown as a single hatched bar at its own x-position.
    """
    # Separate OO baseline from model-specific results
    oo_df = master_df[master_df['mode'] == 'OO']
    model_df = master_df[(master_df['mode'] != 'OO') & (master_df['model'] != 'N/A')]
    logger.info(f"  plot_mode_comparison_per_model: {len(oo_df)} OO rows, {len(model_df)} model rows"
                f"  (modes in master_df: {sorted(master_df['mode'].unique())})")
    logger.info(f"  models in master_df: {sorted(master_df['model'].unique())}")

    agg = model_df.groupby(['model', 'mode']).agg(
        f1_macro=('f1_macro', 'mean'),
    ).reset_index()

    if agg.empty:
        logger.info("  No model-specific results to plot in mode comparison.")
        return None

    model_modes = sorted(agg['mode'].unique())
    models = sorted(agg['model'].unique())
    n_models = len(models)

    _model_cmap = plt.cm.get_cmap('tab10', max(n_models, 3))
    model_colors = {m: _model_cmap(i) for i, m in enumerate(models)}

    # Build x-positions: OO first (position 0), then a gap, then model-specific modes
    has_oo = not oo_df.empty
    gap = 0.15  # visual gap between OO and model-specific modes
    if has_oo:
        x_model_modes = np.arange(len(model_modes)) + 1 + gap
    else:
        x_model_modes = np.arange(len(model_modes)) + 0.25

    bar_w = 0.85 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, (len(model_modes) + has_oo) * 2.5), 6))

    # --- OO bar (single, hatched, centered) ---
    if has_oo:
        oo_mean = oo_df['f1_macro'].mean()
        ax.bar(0, oo_mean, width=0.5,
               color=_mode_color('OO'), alpha=0.85,
               edgecolor='white', linewidth=0.3,
               hatch='///', label='OO (no model)')
        ax.annotate(f'{oo_mean:.3f}',
                    xy=(0, oo_mean), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=_mode_color('OO'))

    # --- Model-specific bars ---
    for i, model in enumerate(models):
        model_data = agg[agg['model'] == model].set_index('mode')
        vals = [model_data.loc[m, 'f1_macro'] if m in model_data.index else 0
                for m in model_modes]
        offset = (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x_model_modes + offset, vals, width=bar_w,
                      label=model, color=model_colors[model],
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6.5,
                            fontweight='bold', color=model_colors[model])

    # --- X-axis ---
    all_ticks = ([0] if has_oo else []) + x_model_modes.tolist()
    all_labels = (['OO'] if has_oo else []) + model_modes
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=11)
    ax.set_xlabel('Mode', fontsize=11)
    ax.set_ylabel('F1 (Macro)', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.legend(title='Model', fontsize=8, title_fontsize=9,
    loc='upper center', bbox_to_anchor=(0.5, -0.1),
    framealpha=0.9, ncol=5)
    ax.set_title('F1 (Macro) by Mode × Model  (avg. across study pairs)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved mode comparison to: {output_path}")
    plt.close()
    return fig


def plot_per_class_heatmap(all_per_class: pd.DataFrame, metric: str = 'f1_score',
                           output_path: str = None):
    """
    Heatmap of per-class F1 (or precision/recall) across model×mode combinations.
    Rows = harmonization class, Columns = model+mode.
    """
    all_per_class = all_per_class.copy()
    # Label OO as baseline instead of showing N/A as a model
    oo_mask = all_per_class['mode'] == 'OO'
    all_per_class.loc[oo_mask, 'config'] = 'OO (baseline)'
    all_per_class.loc[~oo_mask, 'config'] = (
        all_per_class.loc[~oo_mask, 'model'] + ' / ' + all_per_class.loc[~oo_mask, 'mode']
    )

    pivot = all_per_class.pivot_table(
        index='class', columns='config', values=metric, aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5),
                                     max(5, len(pivot) * 0.8)))
    
    sns.heatmap(pivot.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    ax.set_title(f'Per-Class {metric.replace("_", " ").title()} (avg across study pairs)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Mapping Class', fontsize=11)
    ax.set_xlabel('Model / Mode', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved per-class heatmap to: {output_path}")
    plt.close()
    return fig


def plot_best_configs_radar(master_df: pd.DataFrame, output_path: str = None):
    """
    Bar chart showing the best model+mode config per study pair (by F1 macro).
    """
    best = master_df.loc[master_df.groupby('study_pair')['f1_macro'].idxmax()]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(best))
    bars = ax.bar(x, best['f1_macro'], color=PALETTE['f1'], alpha=0.85, edgecolor=PALETTE['neutral'], linewidth=0.5)
    
    for bar, (_, row) in zip(bars, best.iterrows()):
        h = bar.get_height()
        label = 'OO (baseline)' if row['mode'] == 'OO' else f"{row['model']}/{row['mode']}"
        ax.annotate(f"{label}\n{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(best['study_pair'], rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('F1 (Macro)', fontsize=11)
    ax.set_title('Best Model+Mode Configuration per Study Pair (by F1 Macro)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color=PALETTE['good'], linestyle='--', alpha=0.4, linewidth=0.8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved best configs chart to: {output_path}")
    plt.close()
    return fig

def plot_coverage_comparison(summary_df: pd.DataFrame, output_path: str = None):
    """Grouped bar: total predicted vs unique-to-mode, per target study."""
    modes = sorted(summary_df['mode'].unique())
    targets = sorted(summary_df['target_study'].unique())
    x = np.arange(len(targets))
    bar_w = 0.8 / max(len(modes), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(targets) * 2.5), 5))
    for i, m in enumerate(modes):
        sub = summary_df[summary_df['mode'] == m].set_index('target_study')
        vals_total  = [sub.loc[t, 'total_predicted']     if t in sub.index else 0 for t in targets]
        vals_unique = [sub.loc[t, 'unique_to_this_mode'] if t in sub.index else 0 for t in targets]
        off = (i - (len(modes) - 1) / 2) * bar_w
        ax1.bar(x + off, vals_total,  bar_w, label=m, color=_mode_color(m), alpha=0.85)
        ax2.bar(x + off, vals_unique, bar_w, label=m, color=_mode_color(m), alpha=0.85)

    for ax, title in [(ax1, 'Total Predicted Pairs by Mode'),
                      (ax2, 'Pairs Unique to One Mode')]:
        ax.set_xticks(x); ax.set_xticklabels(targets, rotation=30, ha='right')
        ax.set_ylabel('Count'); ax.set_xlabel('Target Study')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(title='Mode', fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved coverage comparison to: {output_path}")
    plt.close()
    return fig
def analyze_prediction_coverage(all_predictions: list, output_dir: str,
                                exclude_not_applicable: bool = False) -> tuple:
    """Set ops on canonical variable pairs; unique-rows CSV uses raw variable names."""
    mode_pairs = {}
    pair_names = {}
    for item in all_predictions:
        df = item['pred_df']
        if exclude_not_applicable:
            df = df[df['predicted class'] != 'not applicable']
        if df.empty:
            continue
        _require_columns(df, EVAL_KEY_COLS + ['src_var_orig', 'tgt_var_orig'], 'prediction coverage dataframe')
        src_study = item.get('source_study', df['source_study'].iloc[0])
        tgt_study = item.get('target_study', df['target_study'].iloc[0])
        key = (item['mode'], src_study, tgt_study)
        bucket = mode_pairs.setdefault(key, set())
        for row in df.itertuples(index=False):
            pair = (getattr(row, 'source_study'), getattr(row, 'target_study'), getattr(row, 'src_var'), getattr(row, 'tgt_var'))
            bucket.add(pair)
            pair_names.setdefault((item['mode'],) + pair, (getattr(row, 'src_var_orig'), getattr(row, 'tgt_var_orig')))

    study_pairs = sorted({(s, t) for _, s, t in mode_pairs})
    modes = sorted({m for m, _, _ in mode_pairs})

    summary, unique_rows = [], []
    for src, tgt in study_pairs:
        ts_map = {m: mode_pairs.get((m, src, tgt), set()) for m in modes}
        union_all = set().union(*ts_map.values()) if ts_map else set()
        nonempty = [ts_map[m] for m in modes if ts_map[m]]
        shared_all = set.intersection(*nonempty) if len(nonempty) == len(modes) else set()

        for m in modes:
            mine = ts_map[m]
            others = set().union(*(p for mm, p in ts_map.items() if mm != m))
            unique = mine - others
            summary.append({
                'source_study': src,
                'target_study': tgt,
                'mode': m,
                'total_predicted': len(mine),
                'unique_to_this_mode': len(unique),
                'shared_with_all_modes': len(shared_all),
                'pct_of_union': round(len(mine) / max(len(union_all), 1) * 100, 1),
            })
            for pair in sorted(unique):
                s_nm, t_nm = pair_names.get((m,) + pair, (pair[2], pair[3]))
                unique_rows.append({
                    'source_study': pair[0],
                    'target_study': pair[1],
                    'mode': m,
                    'src_var': s_nm,
                    'tgt_var': t_nm,
                    'src_key': pair[2],
                    'tgt_key': pair[3],
                })

    summary_df = pd.DataFrame(summary)
    unique_df = pd.DataFrame(unique_rows)
    tag = '_matches_only' if exclude_not_applicable else '_all'
    summary_df.to_csv(os.path.join(output_dir, f'prediction_coverage_summary{tag}.csv'), encoding='utf-8', index=False)
    unique_df.to_csv(os.path.join(output_dir, f'predictions_unique_to_mode{tag}.csv'), encoding='utf-8', index=False)
    logger.info(f"  Coverage summary ({len(summary_df)} rows) + unique pairs ({len(unique_df)} rows) "
                f"saved [{tag.strip('_')}]")
    return summary_df, unique_df

def run_full_evaluation(base_dir: str, cohorts_dir:str, ground_truth_file: str, model_names: list,
                        modes: list, source_study: str, target_studies: list,
                        output_dir: str):
    """
    Run evaluation across ALL models × modes × study pairs.
    Returns a master DataFrame and per-class DataFrame with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    dict_maps = load_dictionaries(cohorts_dir)
    master_rows = []
    all_predictions = []
    per_class_rows = []
    domain_rows = []
    all_incorrect = []
    all_not_in_gt = []
    all_not_in_pred = []
    skipped = []
    collective_data = []   # per (model, mode): aligned y_true/y_pred for pooling across targets
    
    model_specific_modes = [m for m in modes if m != 'OO']
    has_oo = 'OO' in modes
    
    total_combos = (len(model_names) * len(model_specific_modes) * len(target_studies)
                    + (len(target_studies) if has_oo else 0))
    combo_idx = 0
    
    logger.info("=" * 90)
    logger.info("FULL CROSS-MODEL × CROSS-MODE EVALUATION")
    logger.info(f"Models: {model_names}")
    logger.info(f"Model-specific modes: {model_specific_modes}")
    if has_oo:
        logger.info(f"Model-independent modes: OO (ontology only — evaluated once)")
    logger.info(f"Source: {source_study}  →  Targets: {target_studies}")
    logger.info(f"Total configurations: {total_combos}")
    logger.info("=" * 90)
    compute_study_pair_class_imbalance(ground_truth_file=ground_truth_file, output_dir= output_dir)

    def _evaluate_combo(model_label, mode, data_dir, filename_model):
        nonlocal combo_idx
        results = []
        for target_study in target_studies:
            combo_idx += 1
            study_pair = f"{source_study} → {target_study}"
            config_label = f"{model_label}/{mode}" if model_label != "N/A" else mode
            predict_studies_names = f"{source_study}_{target_study}"
            pred_file = os.path.join(
                data_dir,
                f"time-chf_{target_study}_{filename_model}_{mode}_full.csv"
            )
            
            logger.info(f"\n[{combo_idx}/{total_combos}] {config_label} | {study_pair}")
            logger.info("-" * 60)
            
            if not os.path.exists(pred_file):
                logger.info(f"  SKIPPED: Prediction file not found: {pred_file}")
                skipped.append({
                    'model': model_label, 'mode': mode,
                    'study_pair': study_pair, 'reason': 'file_not_found'
                })
                continue
            
            pred = load_predictions(pred_file,
                        dict_maps[source_study.lower()],
                        dict_maps[target_study.lower()],
                        source_study=source_study,
                        target_study=target_study)
            gt   = load_ground_truth(ground_truth_file, source_study, target_study,
                                    dict_maps[source_study.lower()],
                                    dict_maps[target_study.lower()])
            if len(gt) == 0:
                logger.info(f"  SKIPPED: No ground truth for {study_pair}")
                skipped.append({
                    'model': model_label, 'mode': mode,
                    'study_pair': study_pair, 'reason': 'no_ground_truth'
                })
                continue
            
            

            
            metrics = compute_comprehensive_metrics(gt, pred)
            dist = analyze_class_distribution(gt, pred, study_pair)
            domain_df = compute_domain_metrics(gt, pred)

            for _, row in domain_df.iterrows():
                domain_rows.append({
                    'model': model_label,
                    'mode': mode,
                    'source_study': source_study,
                    'target_study': target_study,
                    'study_pair': study_pair,
                    'domain': row['domain'],
                    'n_pairs': row['n_pairs'],
                    'n_classes_present': row['n_classes_present'],
                    'accuracy': row['accuracy'],
                    'f1_macro': row['f1_macro'],
                    'f1_weighted': row['f1_weighted'],
                    'f1_micro': row['f1_micro'],
                    'class_distribution': row['class_distribution'],
                })
            collective_data.append({'model': model_label, 'mode': mode,
                                     'target_study': target_study,
                                     'y_true': metrics['y_true'], 'y_pred': metrics['y_pred']})
            
            combo_output_dir = os.path.join(output_dir, mode) if model_label == "N/A" else os.path.join(output_dir, model_label, mode)
            os.makedirs(combo_output_dir, exist_ok=True)
            basic = evaluate_predictions(gt, pred, combo_output_dir, predict_studies_names, source_study, target_study)
            all_predictions.append({'model': model_label, 'mode': mode,
                        'source_study': source_study, 'target_study': target_study, 'pred_df': pred})
            master_rows.append({
                    "model": model_label,
                    "mode": mode,
                    "source_study": source_study,
                    "target_study": target_study,
                    "study_pair": study_pair,
                    "total_gt_pairs": basic["total"],
                    "correct": basic["correct"],
                    "accuracy": round(metrics["accuracy"], 4),
                    "f1_weighted": round(metrics["f1_weighted"], 4),
                    "f1_macro": round(metrics["f1_macro"], 4),
                    "f1_micro": round(metrics["f1_micro"], 4),
                    "imbalance_ratio": round(dist["imbalance_ratio"], 2),
                    "predicted_not_in_gt": len(basic["not_in_ground_truth"]),
                    "positive_gt_not_retrieved": len(basic["not_in_predictions"]),
                    "positive_candidate_recall": round(basic["positive_candidate_recall"], 4),
                    "warnings": "; ".join(dist["warnings"]) if dist["warnings"] else "",
                })
            
            pcm = metrics['per_class_metrics']
            for _, row in pcm.iterrows():
                per_class_rows.append({
                    'model': model_label,
                    'mode': mode,
                    'study_pair': study_pair,
                    'class': row['class'],
                    'precision': row['precision'],
                    'recall': row['recall'],
                    'f1_score': row['f1_score'],
                    'support': row['support'],
                })
            
            if len(basic['incorrect_predictions']) > 0:
                inc = basic['incorrect_predictions'].copy()
                inc['model'] = model_label
                inc['mode'] = mode
                all_incorrect.append(inc)
            
            if len(basic['not_in_ground_truth']) > 0:
                nig = basic['not_in_ground_truth'].copy()
                nig['model'] = model_label
                nig['mode'] = mode
                all_not_in_gt.append(nig)
            
            if len(basic['not_in_predictions']) > 0:
                nip = basic['not_in_predictions'].copy()
                nip['model'] = model_label
                nip['mode'] = mode
                all_not_in_pred.append(nip)
            
            logger.info(f"  Accuracy: {metrics['accuracy']:.3f}  |  "
                  f"F1(W): {metrics['f1_weighted']:.3f}  |  "
                  f"F1(M): {metrics['f1_macro']:.3f}  |  "
                  f"Correct: {basic['correct']}/{basic['total']}")
            


            plot_confusion_matrix(
                metrics, f"{config_label} | {study_pair}",
                os.path.join(combo_output_dir, f"{predict_studies_names}_confusion_matrix.png")
            )
            plot_per_class_metrics(
                metrics, f"{config_label} | {study_pair}",
                os.path.join(combo_output_dir, f"{predict_studies_names}_per_class_metrics.png")
            )
    
    # -----------------------------------------------------------------
    # 1. Evaluate OO once (model-independent)
    # -----------------------------------------------------------------
    if has_oo:
        import glob
        
        oo_data_dir = None
        oo_filename_model = None
        first_target = target_studies[0]
        
        # Candidate directories to search for OO files
        candidate_dirs = [
            os.path.join(base_dir, "OO"),                          # base_dir/OO/
        ]
        for m in model_names:
            candidate_dirs.append(os.path.join(base_dir, m, "OO")) # base_dir/{model}/OO/
        
        logger.info(f"\n{'='*60}")
        logger.info("OO FILE DISCOVERY — searching for OO prediction files")
        logger.info(f"{'='*60}")
        logger.info(f"  base_dir = {base_dir}")
        logger.info(f"  first_target = {first_target}")
        logger.info(f"  candidate_dirs = {candidate_dirs}")
        
        for cdir in candidate_dirs:
            if not os.path.isdir(cdir):
                logger.info(f"  ✗ {cdir}  — directory does not exist")
                continue
            
            # List ALL csv files in this directory for diagnostics
            all_csvs = glob.glob(os.path.join(cdir, "*.csv"))
            logger.info(f"  ✓ {cdir}  — exists, contains {len(all_csvs)} CSV file(s)")
            if all_csvs:
                for f in sorted(all_csvs)[:10]:  # show up to 10
                    logger.info(f"      {os.path.basename(f)}")
                if len(all_csvs) > 10:
                    logger.info(f"      ... and {len(all_csvs) - 10} more")
            
            # Strategy 1: time-chf_{target}_{model}_OO_full.csv  (model token present)
            pattern1 = os.path.join(cdir, f"time-chf_{first_target}_*_OO_full.csv")
            matches1 = glob.glob(pattern1)
            
            # Strategy 2: time-chf_{target}_OO_full.csv  (no model token)
            pattern2 = os.path.join(cdir, f"time-chf_{first_target}_OO_full.csv")
            match2_exists = os.path.exists(pattern2)
            
            # Strategy 3: broader — any file containing OO and full
            pattern3 = os.path.join(cdir, f"*{first_target}*OO*full*.csv")
            matches3 = glob.glob(pattern3)
            
            # Strategy 4: any *_OO_full.csv at all
            pattern4 = os.path.join(cdir, "*_OO_full.csv")
            matches4 = glob.glob(pattern4)
            
            logger.info(f"    Pattern '{os.path.basename(pattern1)}': {len(matches1)} match(es)")
            logger.info(f"    Pattern '{os.path.basename(pattern2)}': {'FOUND' if match2_exists else 'not found'}")
            logger.info(f"    Pattern '*{first_target}*OO*full*.csv': {len(matches3)} match(es)")
            logger.info(f"    Pattern '*_OO_full.csv': {len(matches4)} match(es)")
            
            if matches1:
                # Extract model token from filename
                basename = os.path.basename(matches1[0])
                prefix = f"time-chf_{first_target}_"
                suffix = "_OO_full.csv"
                if basename.startswith(prefix) and basename.endswith(suffix):
                    oo_filename_model = basename[len(prefix):-len(suffix)]
                    oo_data_dir = cdir
                    logger.info(f"  → MATCH (strategy 1): model token = '{oo_filename_model}'")
                    break
            
            if match2_exists:
                # No model token — need special handling in _evaluate_combo
                # We'll pass empty string as model token
                oo_filename_model = ""
                oo_data_dir = cdir
                logger.info(f"  → MATCH (strategy 2): no model token in filename")
                break
            
            if matches3:
                # Try to parse the first match
                basename = os.path.basename(matches3[0])
                logger.info(f"  → MATCH (strategy 3): {basename}")
                # Try to extract model token
                prefix = f"time-chf_{first_target}_"
                suffix = "_OO_full.csv"
                if basename.startswith(prefix) and basename.endswith(suffix):
                    oo_filename_model = basename[len(prefix):-len(suffix)]
                elif basename == f"time-chf_{first_target}_OO_full.csv":
                    oo_filename_model = ""
                else:
                    # Can't parse, use glob-based approach for _evaluate_combo
                    oo_filename_model = "__GLOB__"
                oo_data_dir = cdir
                break
            
            if matches4:
                basename = os.path.basename(matches4[0])
                logger.info(f"  → MATCH (strategy 4): {basename}")
                oo_filename_model = "__GLOB__"
                oo_data_dir = cdir
                break
        
        if oo_data_dir and oo_filename_model is not None:
            if oo_filename_model == "__GLOB__":
                # Can't use _evaluate_combo's filename template; evaluate directly with glob
                logger.info("  Using glob-based OO evaluation (non-standard filenames)")
                for target_study in target_studies:
                    combo_idx += 1
                    study_pair = f"{source_study} → {target_study}"
                    predict_studies_names = f"{source_study}_{target_study}"
                    
                    # Find the OO file for this target
                    oo_glob = glob.glob(os.path.join(oo_data_dir, f"*{target_study}*OO*full*.csv"))
                    if not oo_glob:
                        logger.info(f"  SKIPPED OO for {target_study}: no matching file in {oo_data_dir}")
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'file_not_found'})
                        continue
                    
                    pred_file = oo_glob[0]
                    logger.info(f"\n[{combo_idx}/{total_combos}] OO | {study_pair}")
                    logger.info(f"  Using: {pred_file}")
                    
                    pred = load_predictions(pred_file,
                        dict_maps[source_study.lower()],
                        dict_maps[target_study.lower()],
                        source_study=source_study,
                        target_study=target_study)
                    gt   = load_ground_truth(ground_truth_file, source_study, target_study,
                                            dict_maps[source_study.lower()],
                                            dict_maps[target_study.lower()])
                    if len(gt) == 0:
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'no_ground_truth'})
                        continue
                    metrics = compute_comprehensive_metrics(gt, pred)
                    dist = analyze_class_distribution(gt, pred, study_pair)
                    domain_df = compute_domain_metrics(gt, pred)

                    for _, row in domain_df.iterrows():
                        domain_rows.append({
                            'model': 'N/A',
                        'mode': 'OO',
                            'source_study': source_study,
                            'target_study': target_study,
                            'study_pair': study_pair,
                            'domain': row['domain'],
                            'n_pairs': row['n_pairs'],
                            'n_classes_present': row['n_classes_present'],
                            'accuracy': row['accuracy'],
                            'f1_macro': row['f1_macro'],
                            'f1_weighted': row['f1_weighted'],
                            'f1_micro': row['f1_micro'],
                            'class_distribution': row['class_distribution'],
                        })
                    collective_data.append({'model': 'N/A', 'mode': 'OO',
                                             'target_study': target_study,
                                             'y_true': metrics['y_true'], 'y_pred': metrics['y_pred']})
                    combo_output_dir = os.path.join(output_dir, "OO")
                    os.makedirs(combo_output_dir, exist_ok=True)
                    basic = evaluate_predictions(gt, pred, combo_output_dir, predict_studies_names, source_study, target_study)
                    all_predictions.append({'model': 'N/A', 'mode': 'OO',
                                            'source_study': source_study,
                                            'target_study': target_study, 'pred_df': pred})
                    if len(basic['incorrect_predictions']) > 0:
                        inc = basic['incorrect_predictions'].copy()
                        inc['model'] = 'N/A'
                        inc['mode'] = 'OO'
                        all_incorrect.append(inc)
                    if len(basic['not_in_ground_truth']) > 0:
                        nig = basic['not_in_ground_truth'].copy()
                        nig['model'] = 'N/A'
                        nig['mode'] = 'OO'
                        all_not_in_gt.append(nig)
                    if len(basic['not_in_predictions']) > 0:
                        nip = basic['not_in_predictions'].copy()
                        nip['model'] = 'N/A'
                        nip['mode'] = 'OO'
                        all_not_in_pred.append(nip)
                    
                    master_rows.append({
                        'model': 'N/A', 'mode': 'OO',
                        "source_study": source_study,
                        "target_study": target_study,
                        "study_pair": study_pair,
                        "total_gt_pairs": basic["total"],
                        "correct": basic["correct"],
                        "accuracy": round(metrics["accuracy"], 4),
                        "f1_weighted": round(metrics["f1_weighted"], 4),
                        "f1_macro": round(metrics["f1_macro"], 4),
                        "f1_micro": round(metrics["f1_micro"], 4),
                        "imbalance_ratio": round(dist["imbalance_ratio"], 2),
                        "predicted_not_in_gt": len(basic["not_in_ground_truth"]),
                        "positive_gt_not_retrieved": len(basic["not_in_predictions"]),
                        "positive_candidate_recall": round(basic["positive_candidate_recall"], 4),
                        "warnings": "; ".join(dist["warnings"]) if dist["warnings"] else ""
                    })

            
                    for _, row in metrics['per_class_metrics'].iterrows():
                        per_class_rows.append({
                            'model': 'N/A', 'mode': 'OO', 'study_pair': study_pair,
                            'class': row['class'], 'precision': row['precision'],
                            'recall': row['recall'], 'f1_score': row['f1_score'],
                            'support': row['support'],
                        })
                    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}  |  "
                                f"F1(W): {metrics['f1_weighted']:.3f}  |  "
                                f"Correct: {basic['correct']}/{basic['total']}")
            elif oo_filename_model == "":
                # Filename has no model token: time-chf_{target}_OO_full.csv
                # We need a custom call since _evaluate_combo adds _{model}_ in the middle
                logger.info("  Using no-model-token OO evaluation")
                for target_study in target_studies:
                    combo_idx += 1
                    study_pair = f"{source_study} → {target_study}"
                    predict_studies_names = f"{source_study}_{target_study}"
                    pred_file = os.path.join(oo_data_dir, f"time-chf_{target_study}_OO_full.csv")
                    
                    logger.info(f"\n[{combo_idx}/{total_combos}] OO | {study_pair}")
                    
                    if not os.path.exists(pred_file):
                        logger.info(f"  SKIPPED: {pred_file} not found")
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'file_not_found'})
                        continue
                    
                    pred = load_predictions(pred_file,
                        dict_maps[source_study.lower()],
                        dict_maps[target_study.lower()],
                        source_study=source_study,
                        target_study=target_study)
                    gt   = load_ground_truth(ground_truth_file, source_study, target_study,
                                            dict_maps[source_study.lower()],
                                            dict_maps[target_study.lower()])
                    if len(gt) == 0:
                        skipped.append({'model': 'N/A', 'mode': 'OO',
                                        'study_pair': study_pair, 'reason': 'no_ground_truth'})
                        continue
                    metrics = compute_comprehensive_metrics(gt, pred)
                    dist = analyze_class_distribution(gt, pred, study_pair)
                    domain_df = compute_domain_metrics(gt, pred)
                    for _, row in domain_df.iterrows():
                        domain_rows.append({
                            'model': 'N/A',
                            'mode': 'OO',
                            'source_study': source_study,
                            'target_study': target_study,
                            'study_pair': study_pair,
                            'domain': row['domain'],
                            'n_pairs': row['n_pairs'],
                            'n_classes_present': row['n_classes_present'],
                            'accuracy': row['accuracy'],
                            'f1_macro': row['f1_macro'],
                            'f1_weighted': row['f1_weighted'],
                            'f1_micro': row['f1_micro'],
                            'class_distribution': row['class_distribution'],
                        })
                    collective_data.append({'model': 'N/A', 'mode': 'OO',
                                             'target_study': target_study,
                                             'y_true': metrics['y_true'], 'y_pred': metrics['y_pred']})
                    combo_output_dir = os.path.join(output_dir, "OO")
                    os.makedirs(combo_output_dir, exist_ok=True)
                    basic = evaluate_predictions(gt, pred, combo_output_dir, predict_studies_names, source_study, target_study)
                    all_predictions.append({'model': 'N/A', 'mode': 'OO',
                                            'source_study': source_study,
                                            'target_study': target_study, 'pred_df': pred})
                    if len(basic['incorrect_predictions']) > 0:
                        inc = basic['incorrect_predictions'].copy()
                        inc['model'] = 'N/A'
                        inc['mode'] = 'OO'
                        all_incorrect.append(inc)
                    if len(basic['not_in_ground_truth']) > 0:
                        nig = basic['not_in_ground_truth'].copy()
                        nig['model'] = 'N/A'
                        nig['mode'] = 'OO'
                        all_not_in_gt.append(nig)
                    if len(basic['not_in_predictions']) > 0:
                        nip = basic['not_in_predictions'].copy()
                        nip['model'] = 'N/A'
                        nip['mode'] = 'OO'
                        all_not_in_pred.append(nip)
                    
                    master_rows.append({
                        'model': 'N/A', 'mode': 'OO',
                        "source_study": source_study,
                        "target_study": target_study,
                        "study_pair": study_pair,
                        "total_gt_pairs": basic["total"],
                        "correct": basic["correct"],
                        "accuracy": round(metrics["accuracy"], 4),
                        "f1_weighted": round(metrics["f1_weighted"], 4),
                        "f1_macro": round(metrics["f1_macro"], 4),
                        "f1_micro": round(metrics["f1_micro"], 4),
                        "imbalance_ratio": round(dist["imbalance_ratio"], 2),
                        "predicted_not_in_gt": len(basic["not_in_ground_truth"]),
                        "positive_gt_not_retrieved": len(basic["not_in_predictions"]),
                        "positive_candidate_recall": round(basic["positive_candidate_recall"], 4),
                        "warnings": "; ".join(dist["warnings"]) if dist["warnings"] else ""
                    })
                    for _, row in metrics['per_class_metrics'].iterrows():
                        per_class_rows.append({
                            'model': 'N/A', 'mode': 'OO', 'study_pair': study_pair,
                            'class': row['class'], 'precision': row['precision'],
                            'recall': row['recall'], 'f1_score': row['f1_score'],
                            'support': row['support'],
                        })
                    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}  |  "
                                f"F1(W): {metrics['f1_weighted']:.3f}  |  "
                                f"Correct: {basic['correct']}/{basic['total']}")
            else:
                # Standard case: model token found, use _evaluate_combo
                _evaluate_combo("N/A", "OO", oo_data_dir, oo_filename_model)
        else:
            logger.info("\n  ⚠️  SKIPPED OO: Could not find any OO prediction files.")
            logger.info(f"     Searched directories: {candidate_dirs}")
            logger.info(f"     TIP: Check your OO output directory and filenames.")
            logger.info(f"     Expected patterns like: time-chf_{{target}}_*_OO_full.csv")
            logger.info(f"     Or: time-chf_{{target}}_OO_full.csv")
            for target_study in target_studies:
                combo_idx += 1
                skipped.append({
                    'model': 'N/A', 'mode': 'OO',
                    'study_pair': f"{source_study} → {target_study}",
                    'reason': 'file_not_found'
                })
    
    # -----------------------------------------------------------------
    # 2. Evaluate model-specific modes (NE, OEH, OEC, OED, ...)
    # -----------------------------------------------------------------
    for model_name in model_names:
        for mode in model_specific_modes:
            data_dir = os.path.join(base_dir, model_name, mode)
            _evaluate_combo(model_name, mode, data_dir, model_name)
    
    # =========================================================================
    # BUILD MASTER DATAFRAMES
    # =========================================================================
    master_df = pd.DataFrame(master_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    domain_df = pd.DataFrame(domain_rows)
    skipped_df = pd.DataFrame(skipped) if skipped else pd.DataFrame()
    
    if master_df.empty:
        logger.info("\nNo valid results found. Check that prediction files exist.")
        return master_df, per_class_df
    
    # =========================================================================
    # SAVE MASTER CSVs
    # =========================================================================
    master_path = os.path.join(output_dir, "master_evaluation_summary.csv")
    master_df.to_csv(master_path, encoding='utf-8', index=False)
    logger.info(f"\n  Saved master summary ({len(master_df)} rows) to: {master_path}")
    
    per_class_path = os.path.join(output_dir, "master_per_class_metrics.csv")
    per_class_df.to_csv(per_class_path, encoding='utf-8', index=False)
    logger.info(f"  Saved per-class metrics ({len(per_class_df)} rows) to: {per_class_path}")

    if not domain_df.empty:
        domain_path = os.path.join(output_dir, "master_domain_metrics.csv")
        domain_df.to_csv(domain_path, encoding='utf-8', index=False)
        logger.info(f"Saved domain metrics ({len(domain_df)} rows) to: {domain_path}")
    pooled_domain_df = (
                domain_df
                .groupby(['model', 'mode', 'domain'], as_index=False)
                .agg(
                    n_pairs=('n_pairs', 'sum'),
                    mean_accuracy=('accuracy', 'mean'),
                    mean_f1_macro=('f1_macro', 'mean'),
                    mean_f1_weighted=('f1_weighted', 'mean')
                )
            )
    pooled_domain_df.to_csv(
    os.path.join(output_dir, "pooled_domain_metrics.csv"),
    encoding='utf-8',
    index=False
)
    # Collective (pooled-over-all-targets) metrics per model × mode
    collective_df = compute_collective_metrics(collective_data, source_study)
    if not collective_df.empty:
        collective_path = os.path.join(output_dir, "collective_evaluation_summary.csv")
        collective_df.to_csv(collective_path, encoding='utf-8', index=False)
        logger.info(f"  Saved collective (pooled-over-targets) summary "
                    f"({len(collective_df)} rows) to: {collective_path}")
    
    if all_incorrect:
        inc_df = pd.concat(all_incorrect, ignore_index=True)
        inc_path = os.path.join(output_dir, "all_incorrect_predictions.csv")
        inc_df.to_csv(inc_path, encoding='utf-8', index=False)
        logger.info(f"  Saved all incorrect predictions ({len(inc_df)} rows) to: {inc_path}")
    
    if all_not_in_gt:
        nig_df = pd.concat(all_not_in_gt, ignore_index=True)
        nig_path = os.path.join(output_dir, "all_predicted_not_in_ground_truth.csv")
        nig_df.to_csv(nig_path,encoding='utf-8',  index=False)
        logger.info(f"  Saved all predicted-not-in-GT ({len(nig_df)} rows) to: {nig_path}")
    
    if all_not_in_pred:
        nip_df = pd.concat(all_not_in_pred, ignore_index=True)
        nip_path = os.path.join(output_dir, "all_gt_not_in_predictions.csv")
        nip_df.to_csv(nip_path,encoding='utf-8', index=False)
        logger.info(f"  Saved all GT-not-in-predictions ({len(nip_df)} rows) to: {nip_path}")
    
    if not skipped_df.empty:
        skip_path = os.path.join(output_dir, "skipped_configurations.csv")
        skipped_df.to_csv(skip_path,encoding='utf-8', index=False)
        logger.info(f"  Saved skipped configurations ({len(skipped_df)} rows) to: {skip_path}")
    


    cov_all, uniq_all = analyze_prediction_coverage(all_predictions, output_dir,
                                                exclude_not_applicable=False)
    cov_m,   uniq_m   = analyze_prediction_coverage(all_predictions, output_dir,
                                                    exclude_not_applicable=True)

    # =========================================================================
    # GENERATE CROSS-MODEL / CROSS-MODE PLOTS
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("GENERATING CROSS-MODEL × CROSS-MODE COMPARISON PLOTS")
    logger.info("=" * 90)
    

    plot_coverage_comparison(cov_all, os.path.join(output_dir, 'coverage_comparison_all.png'))
    plot_coverage_comparison(cov_m,   os.path.join(output_dir, 'coverage_comparison_matches_only.png'))
    for metric in ['f1_macro']:
        plot_metric_heatmap(
            master_df, metric,
            os.path.join(output_dir, f"heatmap_{metric}.png"),
            title=f'{metric.replace("_", " ").title()} — All Models × Modes × Study Pairs'
        )
    
    plot_aggregate_comparison(
        master_df,
        os.path.join(output_dir, "comparison_models_by_mode.png")
    )
    
    plot_mode_comparison_per_model(
        master_df,
        os.path.join(output_dir, "comparison_modes_by_model.png")
    )
    
    if not per_class_df.empty:
        for metric in ['f1_score', 'precision', 'recall']:
            plot_per_class_heatmap(
                per_class_df.copy(), metric,
                os.path.join(output_dir, f"per_class_heatmap_{metric}.png")
            )
    
    plot_best_configs_radar(
        master_df,
        os.path.join(output_dir, "best_config_per_study_pair.png")
    )
    
    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 90)
    logger.info("FINAL MASTER SUMMARY")
    logger.info("=" * 90)
    logger.info(master_df.to_string(index=False))
    
    if not collective_df.empty:
        logger.info("\n" + "=" * 90)
        logger.info(f"COLLECTIVE METRICS — pooled over ALL targets ({source_study} → all), per model × mode")
        logger.info("=" * 90)
        logger.info(collective_df.to_string(index=False))
    
    # Ranking by F1 Macro (averaged across study pairs)
    # ranking = master_df.groupby(['model', 'mode']).agg({
    #     'f1_macro': 'mean',
    # }).reset_index().sort_values('f1_macro', ascending=False)
    # ranking.columns = ['Model', 'Mode', 'Avg F1(M)']
    
    # logger.info("\n" + "=" * 90)
    # logger.info("RANKING BY AVG F1 MACRO (across all study pairs)")
    # logger.info("=" * 90)
    # for rank, (_, row) in enumerate(ranking.iterrows(), 1):
    #     avg_f1_macro = round(row['Avg F1(M)'], 2)
    #     model_label = 'OO (baseline)' if row['Mode'] == 'OO' else f"{row['Model']:>10s} / {row['Mode']:<5s}"
    #     logger.info(f"  #{rank}  {model_label}  |  "
    #           f"F1(M)={avg_f1_macro:>5.3f}")
    
    # ranking_path = os.path.join(output_dir, "ranking_by_f1_macro.csv")
    # ranking.to_csv(ranking_path,encoding='utf-8', index=False)
    # logger.info(f"\n  Saved ranking to: {ranking_path}")
    
    return master_df, per_class_df


def evaluate_f1_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Evaluates predictions against ground truth at variable-pair level."""
    _require_columns(ground_truth, EVAL_KEY_COLS + ['correct class'], 'ground_truth')
    _require_columns(predictions, EVAL_KEY_COLS + ['predicted class'], 'predictions')

    gt_m = ground_truth[EVAL_KEY_COLS + ['correct class']]
    pr_m = predictions[EVAL_KEY_COLS + ['predicted class']]
    merged_df = pd.merge(gt_m, pr_m, on=EVAL_KEY_COLS, how='left')
    merged_df['predicted class'] = merged_df['predicted class'].fillna('not applicable')

    y_true = merged_df['correct class']
    y_pred = merged_df['predicted class']

    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    logger.info("Classification Report:")
    logger.info(report)

    return {
        'f1_score': f1_weighted,
        'f1_macro': f1_macro
    }

def compute_study_pair_class_imbalance(
    ground_truth_file: str,
    positive_only: bool = False,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Per (source_study, target_study) harmonization-level distribution and imbalance ratio.

    The imbalance ratio is the majority-class count divided by the minority-class
    count, computed over classes that are *present* in the pair (zero-support
    classes are ignored, consistent with analyze_class_distribution). With
    positive_only=True the 'not applicable' tier is excluded, so the ratio reflects
    imbalance among the harmonizable tiers (identical / complete / compatible /
    partial) only.

    Counts are taken on de-duplicated (source_var, target_var) pairs within each
    study pair, i.e. before temporal collapsing — they reproduce the raw GT
    value_counts logged at load time.
    """
    gt = pd.read_excel(ground_truth_file, sheet_name=0)
    gt = gt.drop_duplicates(
        subset=["source_study", "target_study", "source_var_name", "target_var_name"],
        keep="first",
    )

    gt["__study_src"] = gt["source_study"].astype(str).str.strip().str.lower()
    gt["__study_tgt"] = gt["target_study"].astype(str).str.strip().str.lower()
    gt["__level"] = gt["harmonization level"].astype(str).str.strip().str.lower()

    if positive_only:
        gt = gt[gt["__level"].isin(POSITIVE_GT_CLASSES)]

    level_order = [
        "identical match", "compatible match",
        "partial match", "not applicable",
    ]

    rows = []
    for (src, tgt), g in gt.groupby(["__study_src", "__study_tgt"]):
        counts = g["__level"].value_counts()
        present = counts[counts > 0]
        if present.empty:
            continue
        imbalance = present.max() / present.min()

        row = {
            "source_study": src,
            "target_study": tgt,
            "n_pairs": int(counts.sum()),
            "n_classes_present": int((counts > 0).sum()),
        }
        for lvl in level_order:
            row[lvl] = int(counts.get(lvl, 0))
        row["majority_class"] = present.idxmax()
        row["minority_class"] = present.idxmin()
        row["imbalance_ratio"] = round(float(imbalance), 2)
        rows.append(row)

    cols = (
        ["source_study", "target_study", "n_pairs", "n_classes_present"]
        + level_order
        + ["majority_class", "minority_class", "imbalance_ratio"]
    )
    out = (
        pd.DataFrame(rows)[cols]
        .sort_values(["source_study", "target_study"])
        .reset_index(drop=True)
    )

    if output_dir is not None:
        tag = "positive_only" if positive_only else "all_classes"
        path = os.path.join(output_dir, f"study_pair_class_imbalance_{tag}.csv")
        out.to_csv(path, encoding="utf-8", index=False)
        logger.info(f" Saved study-pair imbalance table ({len(out)} rows) to: {path}")

    return out


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    # "openai+no_llm", "qwen3-0.6b+no_llm",
    #                     "biolord+gemini-2.5-flash-lite", "sapbert+gemini-2.5-flash-lite", "openai+gemini-2.5-flash-lite", "qwen3-0.6b+gemini-2.5-flash-lite",
    #                     "biolord+gpt-oss-120b", "sapbert+gpt-oss-120b", "openai+gpt-oss-120b", "qwen3-0.6b+gpt-oss-120b",
    #                     "sapbert+llama-4-maverick"
    parser = argparse.ArgumentParser(description="Comprehensive cross-model × cross-mode evaluation")
    parser.add_argument('--models', nargs='+',
                        default=["biolord+no_llm","openai+no_llm","sapbert+no_llm", "biolord+gpt-oss_120b_local","biolord+gemini-3.1-flash-lite_ng","biolord+gemini-3.1-flash-lite","biolord+deepseek-v4-pro","biolord+deepseek-v4-flash_ng","biolord+gpt-oss-120b_ng","sapbert+gemini-3.1-flash-lite","sapbert+gpt-oss-120b","biolord+gpt-oss-120b","sapbert+deepseek-v4-flash","biolord+deepseek-v4-flash","openai+gemini-3.1-flash-lite","openai+gpt-oss-120b","openai+deepseek-v4-flash"],
                        help='Embedding model names to evaluate')
    parser.add_argument('--modes', nargs='+', default=["OO","NE", "OEH"],
                        help='Mapping modes to evaluate')
    parser.add_argument('--source', default="time-chf", help='Source study name')
    parser.add_argument('--targets', nargs='+', default=["aachen-hf", "gissi-hf",  "viennahf-register", "biostat-chf"],
                        help='Target study names')
    parser.add_argument('--single-model', type=str, default=None,
                        help='Run evaluation for a single model only (e.g. sapbert)')
    parser.add_argument('--single-mode', type=str, default=None,
                        help='Run evaluation for a single mode only (e.g. NE)')
    
    args = parser.parse_args()
    
    base_dir = "/Users/komalgilani/phd_projects/CohortVarLinker/data/output/cross_mapping"
    ground_truth_file = "/Users/komalgilani/phd_projects/CohortVarLinker/data/ground_truth_pairs.xlsx"
    output_dir = os.path.join(base_dir, "evaluation_results")
    cohorts_dir = "/Users/komalgilani/phd_projects/CohortVarLinker/data/cross_mapping_article_data"
    model_names = [args.single_model] if args.single_model else args.models
    modes = [args.single_mode] if args.single_mode else args.modes
    
    master_df, per_class_df = run_full_evaluation(
        base_dir=base_dir,
        cohorts_dir=cohorts_dir,
        ground_truth_file=ground_truth_file,
        model_names=model_names,
        modes=modes,
        source_study=args.source,
        target_studies=args.targets,
        output_dir=output_dir,
    )
    
    logger.info(f"\nAll outputs saved to: {output_dir}")
