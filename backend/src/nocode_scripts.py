"""No-code DCR analysis: generate the enclave script from an analysis spec.

A domain expert assembles a spec in the no-code DCR wizard (frontend
/nocode-dcr): an analysis *kind*, the cohorts, and a *mapping* that says
which variable in each cohort plays which role and how their values/units are
harmonized. This module turns that spec into the Python script that runs inside
the Decentriq enclave on the FULL cohort data and writes figures (PNG), tables
(CSV) and a provenance note. Every figure carries a subtext stating exactly
which mapping produced it.

The script is COMPOSED from segments so it contains only what the chosen
analysis needs: single-cohort scripts have no cross-cohort harmonization, a
distribution script carries no correlation code, scipy is imported only when a
test is computed, and so on. Logging and robustness (missing codes, unit
factors, value maps, column lookup) are kept in every variant.

Small-cell suppression is OPTIONAL: spec "suppression_k" (default 0 = off)
hides counts below k and blanks bins/cells below k. Outliers and actual
extremes are shown by default.

Spec shape:
{
  "analysis": {"kind": "distribution" | "stratified" | "correlation" | "crosstab" | "compare" | "pooled",
               "title": "...", "suppression_k": 0, "bins": 20,
               "roles": {"variable": "<harmonized name>", "group": "...", "x": "...", "y": "..."}},
  "cohorts": ["TIME-CHF", "Aachen-HF"],
  "nodes": {"TIME-CHF": {"data": "TIME-CHF", "dictionary": "TIME-CHF_metadata_dictionary"}, ...},
  "mapping": {"id", "name", "created_by", "created_at",
              "variables": [{"harmonized_name", "label", "type": "categorical"|"numeric", "unit",
                             "members": {cohort: {"var_name", "unit"}},
                             "value_map": {cohort: {raw: harmonized}},
                             "unit_conversion": {cohort: {"factor", "from", "to"}},
                             "evidence": [...]}]}
}
"""

from __future__ import annotations

import json
import re
from typing import Any

ANALYSIS_KINDS = {
    "stratified": {
        "label": "One variable stratified by another",
        "roles": ["variable", "group"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "Compare the distribution of a variable between groups (e.g. weight by sex, NYHA class by diabetes).",
        "explain": (
            "Splits the patients into groups defined by a categorical variable and shows the variable of interest within "
            "each group. Numeric: one distribution curve per group plus box plots (median, quartiles, outliers) and a "
            "summary table per group. Categorical: grouped bars and a counts table."
        ),
    },
    "correlation": {
        "label": "Relationship between two numeric variables",
        "roles": ["x", "y"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "Scatter plot, correlation coefficients with confidence intervals and p-values, and binned means.",
        "explain": (
            "Plots every patient as a point (x vs y) with a least-squares line, and reports Pearson and Spearman "
            "correlation coefficients with confidence intervals and p-values. A second panel shows the mean of y within "
            "deciles of x, which makes a non-linear relationship visible."
        ),
    },
    "crosstab": {
        "label": "Cross-tabulation of two categorical variables",
        "roles": ["x", "y"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "Counts and percentages for every combination of two categorical variables, with a chi-square test.",
        "explain": (
            "Counts how many patients fall into each combination of two categorical variables, as a table with row "
            "percentages and a stacked bar chart, plus a chi-square test of independence (with Cramer's V as an effect size)."
        ),
    },
    "compare": {
        "label": "Harmonized variable across multiple cohorts",
        "roles": ["variable"],
        "optional_roles": ["group"],
        "min_cohorts": 2,
        "max_cohorts": 6,
        "blurb": "The same (harmonized) variable side by side in each cohort and pooled into one distribution, optionally stratified by a harmonized group.",
        "explain": (
            "After you have harmonized the variable (matched the variables across cohorts and aligned their values or "
            "units), two views are produced. Side by side: overlaid distribution curves (or percentage bars), a summary "
            "table per cohort and standardized mean differences between cohorts. Pooled: all cohorts stacked into one "
            "distribution coloured by cohort, with a pooled summary. Optionally the pooled data is also stratified by a "
            "second harmonized categorical variable."
        ),
    },
}

# Kinds that are no longer offered in the wizard but may exist in saved specs
# and already-created DCRs; the generator still accepts them.
_LEGACY_KINDS = {"distribution", "pooled"}


def _parse_missing_codes(raw: Any) -> list[str]:
    raw = str(raw or "").strip()
    if not raw or raw.lower() in ("nan", "na", "none"):
        return []
    parts = [p.strip() for p in re.split(r"[|,;]", raw) if p.strip()]
    return [p.split("=")[0].strip() for p in parts]   # "999=unknown" -> "999"


def bake_variable_metadata(spec: dict[str, Any], cohorts_by_id: dict[str, Any]) -> dict[str, Any]:
    """Copy, for every member variable, the declared missing-value codes from
    the cohort's data dictionary (on the explorer server) into the spec, so the
    enclave script needs no dictionary node at all: the handful of variables it
    touches are fully described at DCR-creation time."""
    import csv

    dict_rows: dict[str, dict[str, dict]] = {}   # cohort -> lower var name -> row
    for cohort_id in spec.get("cohorts") or []:
        cohort = cohorts_by_id.get(cohort_id)
        if cohort is None:
            continue
        try:
            path = cohort.metadata_filepath
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                cols = {(c or "").strip().upper(): c for c in (reader.fieldnames or [])}
                name_col = cols.get("VARIABLENAME") or cols.get("VARIABLE NAME")
                miss_col = cols.get("MISSING")
                rows = {}
                if name_col:
                    for r in reader:
                        rows[str(r.get(name_col) or "").strip().lower()] = {"missing": r.get(miss_col, "") if miss_col else ""}
                dict_rows[cohort_id] = rows
        except Exception:
            dict_rows[cohort_id] = {}
    for hv in (spec.get("mapping") or {}).get("variables") or []:
        for cohort_id, member in (hv.get("members") or {}).items():
            if not member or not member.get("var_name"):
                continue
            row = dict_rows.get(cohort_id, {}).get(str(member["var_name"]).strip().lower())
            member["missing_codes"] = _parse_missing_codes(row["missing"]) if row else []
    return spec


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "analysis"


def short_var(name: str, n: int = 16) -> str:
    """A harmonized variable name without its _pooled/_harmonized suffix, as a
    dashed slug cut to n characters: 'ntprobnp_pooled' -> 'ntprobnp'."""
    base = re.sub(r"_(pooled|harmonized)$", "", str(name or ""))
    return _slug(base)[:n].rstrip("-") if base else ""


def nocode_node_name(index: int, spec: dict[str, Any]) -> str:
    """Compute node name: the analysis type followed by the variables in play,
    e.g. pool-and-stratify-ntprobnp-by-nyha, crosstab-sex-nyha. (The node has
    code in it; 'no-code' is about how it was built, so it is not in the name.)"""
    a = spec.get("analysis", {})
    kind = a.get("kind", "analysis")
    roles = a.get("roles", {}) or {}
    if kind == "compare":
        label = "pool-and-stratify" if roles.get("group") else "pool-and-compare"
    else:
        label = {"stratified": "stratify", "correlation": "correlate", "crosstab": "crosstab",
                 "pooled": "pool", "distribution": "distribution"}.get(kind, _slug(kind))
    if kind in ("correlation", "crosstab"):
        parts = [short_var(roles.get("x")), short_var(roles.get("y"))]
    else:
        parts = [short_var(roles.get("variable"))]
        if roles.get("group"):
            parts += ["by", short_var(roles.get("group"))]
    name = "-".join([label] + [p for p in parts if p])
    if index > 1:
        name = f"{name}-{index}"
    return name[:64].rstrip("-")


def describe_spec(spec: dict[str, Any]) -> str:
    """Plain-language recipe of what the script will compute (used for the DCR
    description and the wizard's review step)."""
    a = spec.get("analysis", {})
    kind = a.get("kind", "")
    roles = a.get("roles", {})
    meta = ANALYSIS_KINDS.get(kind, {})
    # The cohort names are deliberately not repeated here: they are visible in the DCR's data section.
    parts = [f"{meta.get('label', kind)}."]
    for role, hv in roles.items():
        if hv:
            parts.append(f"{role}: {hv}")
    k = int(a.get("suppression_k", 0) or 0)
    if k > 0:
        parts.append(f"Small cells below {k} are suppressed.")
    if spec.get("data_source") == "shuffled":
        parts.append("COMPUTED ON SHUFFLED SAMPLES (code test only; no results can be drawn).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Script segments. Plain strings (no f-strings) with __PLACEHOLDERS__.
# ---------------------------------------------------------------------------

HEADER = r'''###############################################################################
# NO-CODE DCR ANALYSIS — generated by the iCARE4CVD Cohort Explorer
#
# Assembled from the choices made in the no-code DCR wizard:
#   __DESCRIPTION__
#
# Reads the cohort data inside the enclave and writes figures, summary tables
# and a provenance note (/output). This script contains only the code paths
# this analysis needs.
###############################################################################
import json
import math
import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
__SCIPY_IMPORT__
SPEC = json.loads(r"""__SPEC_JSON__""")

OUT = "/output"
FIG_DIR = os.path.join(OUT, "figures")
TAB_DIR = os.path.join(OUT, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
LOG = os.path.join(OUT, "nocode_log.txt")

ANALYSIS = SPEC["analysis"]
KIND = ANALYSIS["kind"]
K = int(ANALYSIS.get("suppression_k", 0) or 0)   # 0 = no suppression
BINS = int(ANALYSIS.get("bins", 20))
ROLES = ANALYSIS.get("roles", {})
COHORTS = SPEC["cohorts"]
NODES = SPEC["nodes"]
MAPPING = SPEC["mapping"]
HVARS = {v["harmonized_name"]: v for v in MAPPING.get("variables", [])}
TITLE = ANALYSIS.get("title") or KIND
SHUFFLED = SPEC.get("data_source") == "shuffled"

captions = []   # {"figure"/"table": path, "caption": text, "provenance": text}
notes = []


def log(msg):
    with open(LOG, "a") as fh:
        fh.write(str(msg) + "\n")


# ---- loading -----------------------------------------------------------------

def load_table(path):
    """Read a cohort data node. RawDataNodeDefinition mounts the uploaded file
    as-is at /input/<node>. Tries CSV (several encodings, separator detected),
    then SPSS if the file is one, and fails with the real reasons."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s does not exist inside the enclave: the data of this cohort has not been "
            "provisioned by its data owner yet (or the node is empty)." % path)
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        log("%s is a directory with %d file(s); using the first" % (path, len(files)))
        if not files:
            raise FileNotFoundError("%s is an empty directory" % path)
        path = os.path.join(path, files[0])
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head.startswith(b"$FL2") or head.startswith(b"$FL3"):
        return pd.read_spss(path, convert_categoricals=False)
    errors = []
    # utf-8-sig reads plain UTF-8 too and drops a byte-order mark; latin-1
    # decodes any byte sequence, so it is the last resort.
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                header = fh.readline()
            # separator = the candidate that occurs most in the header line
            counts = dict((sep, header.count(sep)) for sep in (",", ";", "\t", "|"))
            sep = max(counts, key=counts.get) if any(counts.values()) else ","
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            log("%s: read as CSV (%s, separator %r), %d rows x %d columns" % (os.path.basename(path), enc, sep, df.shape[0], df.shape[1]))
            return df
        except Exception as e:
            errors.append("%s: %s" % (enc, str(e)[:160]))
    raise ValueError("Could not read %s as CSV. Attempts: %s" % (path, " | ".join(errors)))


def load_missing_codes(dict_path):
    """{variable_name -> set(missing codes)} from the dictionary's MISSING column."""
    codes = {}
    try:
        d = pd.read_csv(dict_path, dtype=str).fillna("")
        cols = {c.strip().upper(): c for c in d.columns}
        name_col = cols.get("VARIABLENAME") or cols.get("VARIABLE NAME")
        miss_col = cols.get("MISSING")
        if name_col and miss_col:
            for _, row in d.iterrows():
                raw = str(row[miss_col]).strip()
                if raw and raw.lower() not in ("nan", "na", "none", ""):
                    parts = [p.strip() for p in re.split(r"[|,;]", raw) if p.strip()]
                    parts = [p.split("=")[0].strip() for p in parts]   # "999=unknown" -> "999"
                    codes[str(row[name_col]).strip()] = set(parts)
    except Exception as e:
        log("dictionary not parsed for missing codes: %s" % e)
    return codes


def find_column(df, name):
    if name in df.columns:
        return name
    low = {c.lower().strip(): c for c in df.columns}
    return low.get(str(name).lower().strip())


def prepare_column(df, hv, cohort, missing):
    """One harmonized variable's column for one cohort: declared missing codes
    -> NaN, numeric coercion and unit factor, or value map for categoricals."""
    member = (hv.get("members") or {}).get(cohort)
    if not member or not member.get("var_name"):
        return pd.Series(np.nan, index=df.index)
    col = find_column(df, member["var_name"])
    if col is None:
        log("%s: column '%s' not found for %s" % (cohort, member["var_name"], hv["harmonized_name"]))
        return pd.Series(np.nan, index=df.index)
    s = df[col]
    # Declared missing-value codes: baked into the spec at DCR creation; the
    # dictionary lookup only serves specs that still carry a dictionary node.
    mc = set(member.get("missing_codes") or []) or missing.get(member["var_name"], set())
    if mc:
        # Compare as text, with and without a trailing ".0": a numeric column
        # with blanks is read as float, so the code 999 arrives as "999.0".
        mc = set(str(c).strip() for c in mc) | set(re.sub(r"\.0$", "", str(c).strip()) for c in mc)
        key = s.astype(str).str.strip()
        s = s.mask(key.isin(mc) | key.str.replace(r"\.0$", "", regex=True).isin(mc))
    if hv.get("type") == "numeric":
        s = pd.to_numeric(s, errors="coerce")
        conv = (hv.get("unit_conversion") or {}).get(cohort)
        if conv and conv.get("factor") not in (None, "", 1, 1.0):
            s = s * float(conv["factor"])
            log("%s: %s scaled by %s (%s -> %s)" % (cohort, member["var_name"], conv["factor"], conv.get("from"), conv.get("to")))
        return s
    vmap = dict((hv.get("value_map") or {}).get(cohort) or {})
    # "__MISSING__" carries the policy for empty cells, NaN and the dictionary's
    # declared missing codes (masked above): "" (default) = those patients are
    # excluded; a name (the wizard uses "<missing>") = they are kept as that one
    # category, the same in every cohort.
    missing_target = vmap.pop("__MISSING__", "") or ""
    is_missing = s.isna() | (s.astype(str).str.strip() == "")
    if vmap:
        norm = {str(k).strip().lower(): v for k, v in vmap.items()}
        key = s.astype(str).str.strip().str.lower()
        key2 = key.str.replace(r"\.0$", "", regex=True)     # "1.0" should match "1"
        mapped = key.map(norm)
        mapped = mapped.where(mapped.notna(), key2.map(norm))
        mapped = mapped.replace("", np.nan)        # "" in the value map = treated as missing
        unmapped = int((~is_missing & mapped.isna()).sum())
        if unmapped:
            log("%s: %d value(s) of %s not in the value map -> missing" % (cohort, unmapped, member["var_name"]))
        mapped = mapped.where(~is_missing, np.nan)
    else:
        mapped = s.where(~is_missing, np.nan).astype(object)
    if missing_target:
        mapped = mapped.where(~is_missing, missing_target)
        log("%s: %d empty/missing value(s) of %s kept as '%s'" % (cohort, int(is_missing.sum()), member["var_name"], missing_target))
    return mapped

'''

LOAD_SINGLE = r'''
# ---- data (single cohort) ----------------------------------------------------
COHORT = COHORTS[0]
_raw = load_table("/input/" + NODES[COHORT]["data"])
_missing = load_missing_codes("/input/" + NODES[COHORT]["dictionary"]) if NODES[COHORT].get("dictionary") else {}
data = pd.DataFrame(index=_raw.index)
data["cohort"] = COHORT
for _name, _hv in HVARS.items():
    data[_name] = prepare_column(_raw, _hv, COHORT, _missing)
log("%s: %d rows loaded; columns prepared: %s" % (COHORT, len(data), list(HVARS.keys())))

'''

LOAD_MULTI = r'''
# ---- data (harmonized across cohorts) ----------------------------------------
frames = []
for _c in COHORTS:
    _raw = load_table("/input/" + NODES[_c]["data"])
    _missing = load_missing_codes("/input/" + NODES[_c]["dictionary"]) if NODES[_c].get("dictionary") else {}
    _out = pd.DataFrame(index=_raw.index)
    _out["cohort"] = _c
    for _name, _hv in HVARS.items():
        _out[_name] = prepare_column(_raw, _hv, _c, _missing)
    frames.append(_out)
    log("%s: %d rows loaded" % (_c, len(_out)))
data = pd.concat(frames, ignore_index=True)

'''

PROVENANCE = r'''
# ---- provenance + output helpers ---------------------------------------------

def short_cohort(c):
    """'TIME-CHF' -> 'TIME': a two-part cohort name keeps its first part."""
    parts = str(c).split("-")
    return parts[0] if len(parts) == 2 and parts[0] else str(c)


def provenance_lines(used, with_values=False):
    """Compact record of a harmonized variable, one line each:
    'hname: TIME::BNP1 -- Aachen::NT.pro.BNP (x0.001) (same LOINC code)'.
    with_values adds each cohort's value map in braces, {1 -> I, (missing) -> (excluded)}."""
    lines = []
    for hname in used:
        hv = HVARS.get(hname)
        if not hv:
            continue
        members = []
        for c in COHORTS:
            m = (hv.get("members") or {}).get(c)
            if not m or not m.get("var_name"):
                continue
            piece = "%s::%s" % (short_cohort(c), m["var_name"])
            conv = (hv.get("unit_conversion") or {}).get(c)
            if conv and conv.get("factor") not in (None, "", 1, 1.0):
                piece += " (x%s)" % conv["factor"]
            vmap = (hv.get("value_map") or {}).get(c) or {}
            if with_values and vmap:
                piece += " {%s}" % ", ".join("%s -> %s" % ("(missing)" if k == "__MISSING__" else k, v or "(excluded)")
                                             for k, v in vmap.items())
            members.append(piece)
        codes = []
        for e in hv.get("evidence") or []:
            if e.get("type") == "code":
                sysname = e.get("system") or ""
                # just the vocabulary, not the code itself: "same SNOMED code"
                tag = ("same %s" % sysname) if sysname.upper().endswith("ID") else ("same %s code" % sysname).replace("  ", " ").strip()
                if tag not in codes:
                    codes.append(tag)
        line = "%s: %s" % (hname, " -- ".join(members))
        if codes:
            line += " (%s)" % "; ".join(codes)
        lines.append(line)
    return lines


def footer_text(used):
    """Figure subtext: the shuffled-samples banner and suppression note when
    they apply, then the compact mapping lines."""
    head = ""
    if SHUFFLED:
        head = "SHUFFLED SAMPLES - code test only, columns shuffled independently: no results can be drawn."
    if K > 0:
        head += " Cells with n<%d suppressed." % K
    lines = provenance_lines(used)
    return "\n".join(([head.strip()] if head.strip() else []) + lines)


def save_fig(fig, name, used, caption):
    txt = footer_text(used)
    wrapped = "\n".join(textwrap.fill(l, 150) for l in txt.split("\n"))
    n_lines = wrapped.count("\n") + 1
    # Reserve a band at the bottom for the footer and lay the axes out above
    # it (tight_layout accounts for rotated tick labels and axis titles).
    footer_in = 0.2 + 0.12 * n_lines
    band = min(0.5, footer_in / fig.get_size_inches()[1])
    try:
        fig.tight_layout(rect=[0, band, 1, 1])
    except Exception:
        fig.subplots_adjust(bottom=band + 0.12)
    fig.text(0.01, 0.01, wrapped, fontsize=6.5, va="bottom", ha="left", color="#444444", family="monospace")
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    captions.append({"figure": "figures/" + name, "caption": caption, "provenance": txt})
    log("figure: " + name)


def save_table(df, name, caption=None):
    df.to_csv(os.path.join(TAB_DIR, name), index=False)
    captions.append({"table": "tables/" + name, "caption": caption or name})
    log("table: " + name)


def save_stats(values, name, caption=None):
    """A handful of named statistics (one 'row') as a plain-text file: one
    'name: value' line each. CSV is overkill for a single row."""
    lines = []
    for key, val in values.items():
        if val is None or val == "":
            continue
        if isinstance(val, float):
            val = ("%.4g" % val) if (abs(val) < 1e-3 and val != 0) else ("%.4f" % val).rstrip("0").rstrip(".")
        lines.append("%s: %s" % (key, val))
    with open(os.path.join(TAB_DIR, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    captions.append({"text": "tables/" + name, "caption": caption or name})
    log("stats: " + name)


def fmt_count(n):
    return str(int(n)) if (K <= 0 or int(n) >= K) else "<%d" % K


def sup_note(what="bins"):
    """Axis-label suffix describing suppression, empty when it is off."""
    return " (%s with n<%d blanked)" % (what, K) if K > 0 else ""


def axis_label(hv):
    return "%s%s" % (hv.get("label") or hv["harmonized_name"], " (%s)" % hv["unit"] if hv.get("unit") else "")

'''

H_NUMERIC_SUMMARY = r'''
def numeric_summary(s, label):
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    n = len(valid)
    row = {"group": label, "n": fmt_count(n), "missing": int(s.isna().sum())}
    keys = ["mean", "sd", "median", "q1", "q3", "min", "max", "p5", "p95"]
    if n > 0 and (K <= 0 or n >= K):
        row.update({
            "mean": round(float(valid.mean()), 3),
            "sd": round(float(valid.std(ddof=1)), 3) if n > 1 else None,
            "median": round(float(valid.median()), 3),
            "q1": round(float(valid.quantile(0.25)), 3),
            "q3": round(float(valid.quantile(0.75)), 3),
            "min": round(float(valid.min()), 3),
            "max": round(float(valid.max()), 3),
            "p5": round(float(valid.quantile(0.05)), 3),
            "p95": round(float(valid.quantile(0.95)), 3),
        })
    else:
        row.update({k: None for k in keys})
    return row

'''

H_CATEGORY_COUNTS = r'''
def category_counts(s, label=None):
    s = s.dropna().astype(str)
    vc = s.value_counts()
    rows = []
    for val, n in vc.items():
        rows.append({"group": label, "value": val, "count": fmt_count(n),
                     "percent": round(100.0 * n / len(s), 1) if (K <= 0 or n >= K) and len(s) else None})
    return rows

'''

H_SMD = r'''
def smd(a, b):
    """Standardized mean difference between two numeric samples."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return round(float((a.mean() - b.mean()) / pooled), 3) if pooled > 0 else None

'''

H_FIGSETS = r'''
# ---- figure sets -------------------------------------------------------------
# Every analysis draws several views of the same numbers, because no single
# chart suits every variable: a box plot squashes a skewed biomarker, a
# histogram hides the medians, a violin needs enough patients. The file
# figures_guide.txt lists what was produced and what each view is suited for.

GUIDE = []   # (figure file, title, explanation lines)
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def guide(name, title, *lines):
    GUIDE.append(("figures/" + name, title, [l for l in lines if l]))


def skewed(values):
    """Strongly right-skewed positive data (biomarkers, durations, costs):
    such variables are also drawn on a log axis."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(v) < 10 or (v <= 0).any():
        return False
    med = float(v.median())
    return med > 0 and float(v.max()) / med > 10 and float(v.skew()) > 2


def group_series(df, var, groupcol, order=None):
    """[(group, values)] for the groups of groupcol (in `order` when given),
    small groups suppressed when K > 0."""
    g = df[groupcol].astype(str).where(df[groupcol].notna())
    present = set(g.dropna())
    names = ([x for x in order if x in present] + sorted(present - set(order))) if order else sorted(present)
    out = []
    for name in names:
        s = pd.to_numeric(df.loc[g == name, var], errors="coerce").dropna()
        if len(s) > 0 and (K <= 0 or len(s) >= K):
            out.append((name, s))
    return out


def group_order(gv):
    """Harmonized values in the order of the value mapping (the order the user
    saw in the workbench), so ordered classes keep their order."""
    seen = []
    for vm in (gv.get("value_map") or {}).values():
        for k, v in vm.items():
            if v and v not in seen:
                seen.append(v)
    return seen or None


def log_axis(ax, axis, label):
    if axis == "y":
        ax.set_yscale("log")
        ax.set_ylabel(label + ", log scale")
    else:
        ax.set_xscale("log")
        ax.set_xlabel(label + ", log scale")


def tick_names(ax, names, ns=None):
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(["%s (n=%d)" % (g, n) for g, n in zip(names, ns)] if ns else names, rotation=30, ha="right")


def figures_numeric_by_group(df, var, groupcol, group_label, title, prefix, used, order=None):
    """Views of one numeric variable across groups: box plots with and without
    outliers, cumulative curves, violins, medians with IQR, histograms, and
    (several cohorts pooled) one box per cohort within each group."""
    hv = HVARS[var]
    series = group_series(df, var, groupcol, order)
    if not series:
        notes.append("%s by %s: no group with enough values, figures skipped" % (var, groupcol))
        return
    label = axis_label(hv)
    allv = pd.concat([s for _, s in series])
    logy = skewed(allv)
    names = [g for g, _ in series]
    ns = [len(s) for _, s in series]
    small = [g for g, s in series if len(s) < 10]
    log_note = "Log scale: the values are strongly right-skewed, a linear axis would squash most of the data against zero." if logy else ""
    small_note = ("Groups with fewer than 10 patients (%s): their shapes and quartiles are unreliable." % ", ".join(small)) if small else ""
    width = max(7, 1.5 * len(series) + 3)

    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.boxplot([s.values for _, s in series], showfliers=True)
    tick_names(ax, names, ns)
    ax.set_ylabel(label)
    if logy:
        log_axis(ax, "y", label)
    ax.set_title("Box plots, outliers shown")
    fig.suptitle(title)
    save_fig(fig, prefix + "_box.png", used, "%s: box plots, outliers shown" % title)
    guide(prefix + "_box.png", "Box plots, outliers shown",
          "Median, quartiles and whiskers (1.5 x IQR) per group; circles are the values beyond the whiskers.",
          "Suited for: comparing typical values and spread between groups, and seeing how extreme the extremes are.",
          "Less suited when: a few very large values dominate the axis; then read the version without outliers.",
          log_note, small_note)

    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.boxplot([s.values for _, s in series], showfliers=False)
    tick_names(ax, names, ns)
    ax.set_ylabel(label)
    if logy:
        log_axis(ax, "y", label)
    ax.set_title("Box plots, outliers hidden")
    fig.suptitle(title)
    save_fig(fig, prefix + "_box_no_outliers.png", used, "%s: box plots, outliers hidden" % title)
    guide(prefix + "_box_no_outliers.png", "Box plots, outliers hidden",
          "Same boxes and whiskers, the axis limited to the whiskers so the boxes themselves are readable.",
          "Suited for: comparing medians and interquartile ranges when outliers would otherwise compress the picture.",
          "Less suited when: the extremes are the point of the question; they are not drawn here.", log_note)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    lo, hi = float(allv.min()), float(allv.max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 200) if logy and lo > 0 else np.linspace(lo, hi, 200)
    for g, s in series:
        xs = np.sort(s.values)
        ax.plot(grid, np.searchsorted(xs, grid, side="right") / len(xs), label="%s (n=%d)" % (g, len(s)))
    ax.set_xlabel(label)
    ax.set_ylabel("Fraction of patients at or below the value")
    ax.set_ylim(0, 1.02)
    if logy:
        log_axis(ax, "x", label)
    ax.grid(True, alpha=0.3)
    ax.legend(title=group_label)
    ax.set_title("Cumulative distribution per group")
    fig.suptitle(title)
    save_fig(fig, prefix + "_ecdf.png", used, "%s: cumulative distributions" % title)
    guide(prefix + "_ecdf.png", "Cumulative distribution per group (ECDF)",
          "For every value on the x axis, the fraction of the group at or below it (evaluated on a 200-point grid).",
          "Suited for: comparing whole distributions without bins or smoothing, and reading off any percentile "
          "(where a curve crosses 0.5 is the group's median).",
          "Less suited when: the audience is unused to cumulative curves; box plots are quicker to read.", log_note)

    big = [(g, s) for g, s in series if len(s) >= 5]
    if big:
        fig, ax = plt.subplots(figsize=(width, 5.5))
        data = [np.log10(s.values) if logy else s.values for _, s in big]
        ax.violinplot(data, showmedians=True, showextrema=False)
        tick_names(ax, [g for g, _ in big], [len(s) for _, s in big])
        ax.set_ylabel(("log10 " + label) if logy else label)
        ax.set_title("Violin plots (smoothed shape, median marked)")
        fig.suptitle(title)
        save_fig(fig, prefix + "_violin.png", used, "%s: violin plots" % title)
        guide(prefix + "_violin.png", "Violin plots",
              "The smoothed shape of each group's distribution (mirrored), with the median as a line.",
              "Suited for: spotting two-peaked or lopsided distributions that a box plot hides.",
              "Less suited when: groups are small (under about 30 patients); the smoothing then invents shape." +
              (" Groups with fewer than 5 patients are left out." if len(big) < len(series) else ""),
              "Drawn on log10 values." if logy else "", small_note)

    fig, ax = plt.subplots(figsize=(width, 5.5))
    x = np.arange(len(series))
    meds = [float(s.median()) for _, s in series]
    q1 = [float(s.quantile(0.25)) for _, s in series]
    q3 = [float(s.quantile(0.75)) for _, s in series]
    ax.fill_between(x, q1, q3, alpha=0.2, color=PALETTE[0], label="interquartile range")
    ax.plot(x, meds, marker="o", color=PALETTE[0], label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s (n=%d)" % (g, n) for g, n in zip(names, ns)], rotation=30, ha="right")
    ax.set_ylabel(label)
    if logy:
        log_axis(ax, "y", label)
    ax.legend()
    ax.set_title("Median and interquartile range across groups")
    fig.suptitle(title)
    save_fig(fig, prefix + "_medians.png", used, "%s: medians across groups" % title)
    guide(prefix + "_medians.png", "Median and interquartile range across groups",
          "One point per group (its median) with the interquartile range as a band, in the order of the value mapping.",
          "Suited for: ordered groups (stages, classes, age bands), where the line shows the trend.",
          "Less suited when: the groups have no natural order; the connecting line then means nothing, read it as dots.", log_note)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    edges = np.logspace(np.log10(lo), np.log10(hi), BINS + 1) if logy and lo > 0 and hi > lo else np.histogram_bin_edges(allv, bins=BINS)
    for g, s in series:
        counts, _ = np.histogram(s, bins=edges)
        if K > 0:
            counts = np.where(counts >= K, counts, 0)
        ax.step(edges[:-1], 100.0 * counts / max(counts.sum(), 1), where="post", label="%s (n=%d)" % (g, len(s)))
    ax.set_xlabel(label)
    ax.set_ylabel("% of the group per bin" + sup_note())
    if logy:
        log_axis(ax, "x", label)
    ax.legend(title=group_label)
    ax.set_title("Histograms per group (%d bins)" % BINS)
    fig.suptitle(title)
    save_fig(fig, prefix + "_hist.png", used, "%s: histograms" % title)
    guide(prefix + "_hist.png", "Histograms per group",
          "The share of each group falling in each of %d bins, drawn as steps so the groups can be overlaid." % BINS,
          "Suited for: seeing where the bulk of each group lies and how the groups overlap.",
          "Less suited when: groups are small or many; the steps then cross and become hard to follow.",
          "Bins are equal on the log scale." if logy else "")

    if "cohort" in df.columns and groupcol != "cohort" and df["cohort"].nunique() > 1:
        present = [c for c in COHORTS if (df["cohort"] == c).any()]
        w = 0.8 / len(present)
        fig, ax = plt.subplots(figsize=(max(9, 2.0 * len(series) + 3), 5.5))
        for k, c in enumerate(present):
            data, positions = [], []
            for i, (g, _) in enumerate(series):
                s = pd.to_numeric(df.loc[(df["cohort"] == c) & (df[groupcol].astype(str) == g), var], errors="coerce").dropna()
                if len(s) > 0 and (K <= 0 or len(s) >= K):
                    data.append(s.values)
                    positions.append(i + k * w)
            if data:
                bp = ax.boxplot(data, positions=positions, widths=w * 0.9, showfliers=False, patch_artist=True)
                for box in bp["boxes"]:
                    box.set_facecolor(PALETTE[k % len(PALETTE)])
                    box.set_alpha(0.6)
            ax.plot([], [], color=PALETTE[k % len(PALETTE)], linewidth=8, alpha=0.6, label=c)
        ax.set_xticks(np.arange(len(series)) + w * (len(present) - 1) / 2)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(label)
        if logy:
            log_axis(ax, "y", label)
        ax.legend(title="cohort")
        ax.set_title("Per cohort within each group (outliers hidden)")
        fig.suptitle(title)
        save_fig(fig, prefix + "_box_by_cohort.png", used, "%s: per cohort within each group" % title)
        guide(prefix + "_box_by_cohort.png", "Per cohort within each group",
              "One box per cohort inside every group, outliers hidden.",
              "Suited for: checking that the pooled pattern holds in each cohort rather than being driven by one of them.",
              "Less suited when: a cohort contributes only a few patients to a group; its box is then just noise.", log_note)

    if len(series) >= 2 and all(len(s) >= 2 for _, s in series):
        try:
            h, p = stats.kruskal(*[s.values for _, s in series])
            n_total, k = int(sum(ns)), len(series)
            save_stats({"groups": k, "n": n_total, "kruskal_wallis_H": round(float(h), 3), "p_value": float(p),
                        "epsilon_squared": round((float(h) - k + 1) / (n_total - k), 3) if n_total > k else None,
                        "reading": "Kruskal-Wallis asks whether the groups differ in their distribution (rank-based, "
                                   "no normality assumed); epsilon squared is the effect size (0 none, 1 complete separation)."},
                       prefix + "_tests.txt", "%s: test of group differences" % title)
        except Exception as e:
            log("Kruskal-Wallis skipped: %s" % e)


def figures_categorical_by_group(ct, x_label, y_label, title, prefix, used):
    """Views of a contingency table (rows = the categories on the x axis,
    columns = the categories stacked or grouped): stacked counts, shares,
    grouped bars, and a heat map with counts and row percentages."""
    shown = ct.where(ct >= K, 0) if K > 0 else ct
    rows = [str(i) for i in shown.index]
    cols = [str(c) for c in shown.columns]
    if len(rows) == 0 or len(cols) == 0:
        notes.append("%s: nothing to draw" % title)
        return
    width = max(7, 0.9 * len(rows) + 4)

    fig, ax = plt.subplots(figsize=(width, 5.5))
    bottom = np.zeros(len(rows))
    for c in shown.columns:
        ax.bar(rows, shown[c].values, bottom=bottom, label=str(c))
        bottom += shown[c].values
    ax.set_ylabel("Patients" + sup_note("cells"))
    ax.set_xlabel(x_label)
    ax.legend(title=y_label)
    ax.set_title("Stacked counts")
    fig.suptitle(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, prefix + "_stacked.png", used, "%s: stacked counts" % title)
    guide(prefix + "_stacked.png", "Stacked counts",
          "Number of patients per %s, split by %s." % (x_label, y_label),
          "Suited for: seeing group sizes and the raw composition at once.",
          "Less suited when: the groups differ a lot in size; compare shares in the 100% stacked version instead.")

    pct = shown.div(shown.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(width, 5.5))
    bottom = np.zeros(len(rows))
    for c in shown.columns:
        vals = pct[c].fillna(0).values
        ax.bar(rows, vals, bottom=bottom, label=str(c))
        bottom += vals
    ax.set_ylabel("%% of the patients in each %s" % x_label)
    ax.set_xlabel(x_label)
    ax.set_ylim(0, 100)
    ax.legend(title=y_label)
    ax.set_title("Shares (100% stacked)")
    fig.suptitle(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, prefix + "_stacked_pct.png", used, "%s: shares" % title)
    guide(prefix + "_stacked_pct.png", "Shares, 100% stacked",
          "The composition of each %s in percent, so groups of different sizes can be compared." % x_label,
          "Suited for: comparing proportions between groups.",
          "Less suited when: group sizes matter for the reading; the counts are in the stacked version and the table.")

    fig, ax = plt.subplots(figsize=(max(width, 0.4 * len(rows) * len(cols) + 3), 5.5))
    x = np.arange(len(rows))
    w = 0.8 / max(len(cols), 1)
    for k, c in enumerate(shown.columns):
        ax.bar(x + k * w, shown[c].values, width=w, label=str(c))
    ax.set_xticks(x + w * (len(cols) - 1) / 2)
    ax.set_xticklabels(rows, rotation=30, ha="right")
    ax.set_ylabel("Patients" + sup_note("cells"))
    ax.set_xlabel(x_label)
    ax.legend(title=y_label)
    ax.set_title("Grouped counts")
    fig.suptitle(title)
    save_fig(fig, prefix + "_grouped.png", used, "%s: grouped counts" % title)
    guide(prefix + "_grouped.png", "Grouped counts",
          "The same counts side by side instead of stacked.",
          "Suited for: comparing one category of %s across the groups (bars of the same colour)." % y_label,
          "Less suited when: there are many categories; the bars become thin.")

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(cols) + 3), max(4, 0.6 * len(rows) + 2)))
    im = ax.imshow(pct.fillna(0).values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel(y_label)
    ax.set_ylabel(x_label)
    for i in range(len(rows)):
        for j in range(len(cols)):
            n = int(shown.iloc[i, j])
            v = pct.iloc[i, j]
            txt = "%d\n(%.0f%%)" % (n, v) if not np.isnan(v) else ("<%d" % K if K > 0 else "0")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if (not np.isnan(v) and v > 55) else "black")
    fig.colorbar(im, ax=ax, label="% of the row")
    ax.set_title("Heat map: counts and row percentages")
    fig.suptitle(title)
    save_fig(fig, prefix + "_heatmap.png", used, "%s: heat map" % title)
    guide(prefix + "_heatmap.png", "Heat map with counts and row percentages",
          "Every cell of the table as a coloured tile (row percentage) with the count and percentage written in it.",
          "Suited for: tables with many categories, where bars get crowded; finding the dominant cells at a glance.",
          "Less suited when: there are only two or three categories; bars are then simpler.")


def figures_numeric_pair(df, x, y, title, prefix, used):
    """Views of two numeric variables: scatter with fit, density (hexagonal
    bins), binned medians with IQR, and the binned medians per cohort."""
    hx, hy = HVARS[x], HVARS[y]
    sub = df[[x, y] + (["cohort"] if "cohort" in df.columns else [])].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])
    n = len(sub)
    if n < 4:
        notes.append("%s vs %s: fewer than 4 complete pairs, figures skipped" % (x, y))
        return
    lx, ly = axis_label(hx), axis_label(hy)
    logx, logy = skewed(sub[x]), skewed(sub[y])
    log_note = ("Log scale on %s: strongly right-skewed values." % " and ".join([a for a, f in (("x", logx), ("y", logy)) if f])) if (logx or logy) else ""

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub[x], sub[y], s=10, alpha=0.5, color=PALETTE[0], edgecolors="none")
    try:
        fx = np.log10(sub[x]) if logx else sub[x]
        fy = np.log10(sub[y]) if logy else sub[y]
        slope, intercept = np.polyfit(fx, fy, 1)
        xs = np.linspace(float(fx.min()), float(fx.max()), 50)
        ys = slope * xs + intercept
        ax.plot(10 ** xs if logx else xs, 10 ** ys if logy else ys, color="#c0392b", linewidth=1.2,
                label="least-squares fit" + (" (on log values)" if (logx or logy) else ""))
        ax.legend()
    except Exception as e:
        log("fit line skipped: %s" % e)
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    if logx:
        log_axis(ax, "x", lx)
    if logy:
        log_axis(ax, "y", ly)
    ax.set_title("Scatter with fitted line (n=%d)" % n)
    fig.suptitle(title)
    save_fig(fig, prefix + "_scatter.png", used, "%s: scatter" % title)
    guide(prefix + "_scatter.png", "Scatter with fitted line",
          "One dot per patient and the least-squares line.",
          "Suited for: seeing the shape of the relationship (linear, curved, none) and unusual patients.",
          "Less suited when: there are thousands of patients; dots pile up and the density plot is clearer.", log_note)

    fig, ax = plt.subplots(figsize=(8, 6))
    hx_vals = np.log10(sub[x]) if logx else sub[x]
    hy_vals = np.log10(sub[y]) if logy else sub[y]
    hb = ax.hexbin(hx_vals, hy_vals, gridsize=25, cmap="Blues", mincnt=max(1, K))
    fig.colorbar(hb, ax=ax, label="patients per cell" + (" (cells with n<%d blank)" % K if K > 0 else ""))
    ax.set_xlabel(("log10 " + lx) if logx else lx)
    ax.set_ylabel(("log10 " + ly) if logy else ly)
    ax.set_title("Density (hexagonal bins)")
    fig.suptitle(title)
    save_fig(fig, prefix + "_density.png", used, "%s: density" % title)
    guide(prefix + "_density.png", "Density, hexagonal bins",
          "Patients counted in hexagonal cells; darker means more patients.",
          "Suited for: large samples where a scatter turns into a blob; shows where most patients are.",
          "Less suited when: the sample is small; most cells then hold one or two patients.", log_note)

    try:
        q = pd.qcut(sub[x], q=10, duplicates="drop")
        g = sub.groupby(q, observed=True)[y]
        agg = g.agg(["median", "mean", "count"])
        agg["q1"] = g.quantile(0.25)
        agg["q3"] = g.quantile(0.75)
        centers = np.array([iv.mid for iv in agg.index])
        ok = (agg["count"] >= K).values if K > 0 else (agg["count"] > 0).values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.fill_between(centers[ok], agg["q1"].values[ok], agg["q3"].values[ok], alpha=0.2, color=PALETTE[0], label="interquartile range")
        ax.plot(centers[ok], agg["median"].values[ok], marker="o", color=PALETTE[0], label="median")
        ax.plot(centers[ok], agg["mean"].values[ok], marker="s", linestyle="--", color=PALETTE[1], label="mean")
        ax.set_xlabel("%s (deciles)" % lx)
        ax.set_ylabel(ly)
        if logx:
            log_axis(ax, "x", lx)
        if logy:
            log_axis(ax, "y", ly)
        ax.legend()
        ax.set_title("Binned medians and means")
        fig.suptitle(title)
        save_fig(fig, prefix + "_binned.png", used, "%s: binned medians" % title)
        guide(prefix + "_binned.png", "Binned medians and means",
              "Patients split into deciles of %s; for each, the median (with IQR band) and the mean of %s." % (lx, ly),
              "Suited for: reading the typical %s at each level of %s without the noise of single patients; "
              "curved relationships show up here." % (ly, lx),
              "Less suited when: the sample is small; ten bins then hold only a handful of patients each.", log_note)
        save_table(pd.DataFrame({"x_bin": [str(iv) for iv in agg.index], "median_y": agg["median"].round(3).values,
                                 "mean_y": agg["mean"].round(3).values, "q1_y": agg["q1"].round(3).values,
                                 "q3_y": agg["q3"].round(3).values, "n": [fmt_count(c) for c in agg["count"].values]}),
                   prefix + "_binned.csv", "%s by deciles of %s" % (y, x))
        if "cohort" in sub.columns and sub["cohort"].nunique() > 1:
            edges_x = [iv.left for iv in agg.index] + [agg.index[-1].right]
            fig, ax = plt.subplots(figsize=(8, 6))
            for k, c in enumerate([c for c in COHORTS if (sub["cohort"] == c).any()]):
                part = sub[sub["cohort"] == c]
                gc = part.groupby(pd.cut(part[x], bins=edges_x, include_lowest=True), observed=True)[y].agg(["median", "count"])
                ok_c = (gc["count"] >= K).values if K > 0 else (gc["count"] > 0).values
                cc = np.array([iv.mid for iv in gc.index])
                ax.plot(cc[ok_c], gc["median"].values[ok_c], marker="o", color=PALETTE[k % len(PALETTE)], label="%s (n=%d)" % (c, len(part)))
            ax.set_xlabel("%s (pooled deciles)" % lx)
            ax.set_ylabel("median %s" % ly)
            if logx:
                log_axis(ax, "x", lx)
            if logy:
                log_axis(ax, "y", ly)
            ax.legend(title="cohort")
            ax.set_title("Binned medians per cohort")
            fig.suptitle(title)
            save_fig(fig, prefix + "_binned_by_cohort.png", used, "%s: binned medians per cohort" % title)
            guide(prefix + "_binned_by_cohort.png", "Binned medians per cohort",
                  "The binned medians drawn separately for each cohort, on the same bins.",
                  "Suited for: checking that the relationship is the same in every cohort before trusting the pooled one.",
                  "Less suited when: a cohort is small; its line then jumps around.", log_note)
    except Exception as e:
        log("binned views skipped: %s" % e)


def write_guide():
    if not GUIDE:
        return
    with open(os.path.join(OUT, "figures_guide.txt"), "w") as fh:
        fh.write("FIGURES PRODUCED AND WHAT EACH IS SUITED FOR\n")
        fh.write("Analysis: %s\n\n" % TITLE)
        for f, t, lines in GUIDE:
            fh.write("%s  --  %s\n" % (f, t))
            for l in lines:
                fh.write("    %s\n" % l)
            fh.write("\n")
    captions.insert(0, {"doc": "figures_guide.txt", "caption": "Figures produced and what each is suited for"})

'''

A_DISTRIBUTION = r'''
# ---- analysis: distribution (legacy kind) ------------------------------------

def run_distribution(df, var, label, group_label=None, prefix="distribution"):
    hv = HVARS[var]
    if hv.get("type") == "numeric":
        save_table(pd.DataFrame([numeric_summary(df[var], group_label or label)]), prefix + "_summary.csv", "Summary of %s" % var)
        tmp = df[[var]].copy()
        tmp["__all__"] = "all patients"
        figures_numeric_by_group(tmp, var, "__all__", "", label, prefix, [var])
    else:
        rows = category_counts(df[var], group_label or label)
        save_table(pd.DataFrame(rows), prefix + "_counts.csv", "Counts of %s" % var)
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), pd.Series(["patients"] * len(df), index=df.index))
        figures_categorical_by_group(ct, hv.get("label") or var, "", label, prefix, [var])

'''

A_STRATIFIED = r'''
# ---- analysis: one variable broken down by another ---------------------------

def run_stratified(df, var, group, title, prefix="stratified"):
    hv, gv = HVARS[var], HVARS[group]
    used = [var, group]
    order = group_order(gv)
    glabel = gv.get("label") or group
    if hv.get("type") == "numeric":
        rows = [numeric_summary(s, g) for g, s in group_series(df, var, group, order)]
        save_table(pd.DataFrame(rows), prefix + "_summary.csv", "%s by %s" % (var, group))
        figures_numeric_by_group(df, var, group, glabel, title, prefix, used, order)
    else:
        ct = pd.crosstab(df[group].astype(str).where(df[group].notna()), df[var].astype(str).where(df[var].notna()))
        if order:
            ct = ct.reindex([g for g in order if g in ct.index] + [g for g in ct.index if g not in order])
        table = ct.astype(object).copy()
        for i in table.index:
            for j in table.columns:
                table.loc[i, j] = fmt_count(ct.loc[i, j])
        table.insert(0, group + " \\ " + var, table.index)
        save_table(table.reset_index(drop=True), prefix + "_counts.csv", "%s by %s" % (var, group))
        figures_categorical_by_group(ct, glabel, hv.get("label") or var, title, prefix, used)

'''

A_CORRELATION = r'''
# ---- analysis: relationship between two numeric variables --------------------

def run_correlation(df, x, y, title, prefix="correlation"):
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    row = {"n": fmt_count(n)}
    if n >= 4:
        r_p, p_p = stats.pearsonr(sub[x], sub[y])
        r_s, p_s = stats.spearmanr(sub[x], sub[y])
        z = math.atanh(max(min(float(r_p), 0.999999), -0.999999))
        se = 1.0 / math.sqrt(n - 3)
        row.update({"pearson_r": round(float(r_p), 3), "pearson_p": float(p_p),
                    "pearson_ci95_low": round(math.tanh(z - 1.96 * se), 3),
                    "pearson_ci95_high": round(math.tanh(z + 1.96 * se), 3),
                    "spearman_rho": round(float(r_s), 3), "spearman_p": float(p_s),
                    "reading": "Pearson measures a straight-line relationship, Spearman any monotone one; "
                               "both run from -1 to 1, 0 meaning none."})
    save_stats(row, prefix + "_coefficients.txt", "Correlation of %s and %s" % (x, y))
    figures_numeric_pair(df, x, y, title, prefix, [x, y])

'''

A_CROSSTAB = r'''
# ---- analysis: cross-tabulation ----------------------------------------------

def run_crosstab(df, x, y, title, prefix="crosstab"):
    hx, hy = HVARS[x], HVARS[y]
    a = df[x].astype(str).where(df[x].notna())
    b = df[y].astype(str).where(df[y].notna())
    ct = pd.crosstab(a, b)
    ox, oy = group_order(hx), group_order(hy)
    if ox:
        ct = ct.reindex([g for g in ox if g in ct.index] + [g for g in ct.index if g not in ox])
    if oy:
        ct = ct[[g for g in oy if g in ct.columns] + [g for g in ct.columns if g not in oy]]
    table = ct.astype(object).copy()
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    for i in table.index:
        for j in table.columns:
            n = ct.loc[i, j]
            table.loc[i, j] = ("%d (%.1f%%)" % (n, pct.loc[i, j])) if (K <= 0 or n >= K) else "<%d" % K
    table.insert(0, x + " \\ " + y, table.index)
    save_table(table.reset_index(drop=True), prefix + "_counts.csv", "%s by %s (row %%)" % (x, y))
    stat = {"n": fmt_count(int(ct.values.sum()))}
    if ct.shape[0] > 1 and ct.shape[1] > 1 and ct.values.sum() > 0:
        chi2, p, dof, expected = stats.chi2_contingency(ct.values)
        v = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        stat.update({"chi_square": round(float(chi2), 3), "dof": int(dof), "p_value": float(p), "cramers_v": round(v, 3),
                     "note": "expected counts < 5 in some cells" if (expected < 5).any() else "",
                     "reading": "Chi-square asks whether the two variables are independent; Cramer's V is the "
                                "strength of association (0 none, 1 complete)."})
    save_stats(stat, prefix + "_chi_square.txt", "Chi-square test of independence")
    figures_categorical_by_group(ct, hx.get("label") or x, hy.get("label") or y, title, prefix, [x, y])

'''

A_COMPARE = r'''
# ---- analysis: compare one variable across cohorts ---------------------------

def run_compare(df, var, title, prefix="compare"):
    hv = HVARS[var]
    if hv.get("type") == "numeric":
        rows = [numeric_summary(df.loc[df["cohort"] == c, var], c) for c in COHORTS]
        for i in range(len(COHORTS)):
            for j in range(i + 1, len(COHORTS)):
                rows.append({"group": "SMD %s vs %s" % (COHORTS[i], COHORTS[j]),
                             "mean": smd(df.loc[df["cohort"] == COHORTS[i], var], df.loc[df["cohort"] == COHORTS[j], var])})
        save_table(pd.DataFrame(rows), prefix + "_summary.csv", "%s per cohort" % var)
        figures_numeric_by_group(df, var, "cohort", "cohort", title, prefix, [var], COHORTS)
    else:
        rows = []
        for c in COHORTS:
            rows.extend(category_counts(df.loc[df["cohort"] == c, var], c))
        save_table(pd.DataFrame(rows), prefix + "_counts.csv", "%s per cohort" % var)
        ct = pd.crosstab(df["cohort"], df[var].astype(str).where(df[var].notna()))
        ct = ct.reindex([c for c in COHORTS if c in ct.index])
        order = group_order(hv)
        if order:
            ct = ct[[g for g in order if g in ct.columns] + [g for g in ct.columns if g not in order]]
        figures_categorical_by_group(ct, "cohort", hv.get("label") or var, title, prefix, [var])

'''

A_POOLED = r'''
# ---- analysis: pooled distribution -------------------------------------------

def run_pooled(df, var, group, title, prefix="pooled"):
    hv = HVARS[var]
    used = [var] + ([group] if group else [])
    if hv.get("type") == "numeric":
        rows = [numeric_summary(df[var], "pooled")] + [numeric_summary(df.loc[df["cohort"] == c, var], c) for c in COHORTS]
    else:
        rows = category_counts(df[var], "pooled")
        for c in COHORTS:
            rows.extend(category_counts(df.loc[df["cohort"] == c, var], c))
    save_table(pd.DataFrame(rows), prefix + "_summary.csv", "Pooled %s" % var)
    if hv.get("type") == "numeric":
        allv = pd.to_numeric(df[var], errors="coerce").dropna()
        if len(allv) == 0:
            notes.append("pooled: no values available")
            return
        logx = skewed(allv)
        lo, hi = float(allv.min()), float(allv.max())
        edges = np.logspace(np.log10(lo), np.log10(hi), BINS + 1) if logx and lo > 0 and hi > lo else np.histogram_bin_edges(allv, bins=BINS)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bottom = np.zeros(len(edges) - 1)
        for c in COHORTS:
            s = pd.to_numeric(df.loc[df["cohort"] == c, var], errors="coerce").dropna()
            counts, _ = np.histogram(s, bins=edges)
            if K > 0:
                counts = np.where(counts >= K, counts, 0)
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", bottom=bottom, label=c, edgecolor="white")
            bottom = bottom + counts
        ax.set_xlabel(axis_label(hv))
        if logx:
            log_axis(ax, "x", axis_label(hv))
        ax.set_ylabel("Patients, stacked by cohort" + sup_note())
        ax.set_title("Pooled histogram, stacked by cohort")
        fig.suptitle(title)
        ax.legend()
        save_fig(fig, prefix + "_hist.png", used, "%s: pooled histogram" % title)
        guide(prefix + "_hist.png", "Pooled histogram, stacked by cohort",
              "All cohorts together in %d bins, each bar split by cohort." % BINS,
              "Suited for: the overall distribution of the pooled data and each cohort's contribution to it.",
              "Less suited when: comparing cohort shapes; the per-cohort box plots and cumulative curves do that.",
              "Bins are equal on the log scale: the values are strongly right-skewed." if logx else "")
    else:
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), df["cohort"])
        order = group_order(hv)
        if order:
            ct = ct.reindex([g for g in order if g in ct.index] + [g for g in ct.index if g not in order])
        ct = ct[[c for c in COHORTS if c in ct.columns]]
        figures_categorical_by_group(ct, hv.get("label") or var, "cohort", title, prefix, used)
    if group:
        run_stratified(df, var, group, title, prefix="pooled_by_group")

'''

DISPATCH = {
    "distribution": 'run_distribution(data, ROLES["variable"], TITLE)',
    "stratified": 'run_stratified(data, ROLES["variable"], ROLES["group"], TITLE)',
    "correlation": 'run_correlation(data, ROLES["x"], ROLES["y"], TITLE)',
    "crosstab": 'run_crosstab(data, ROLES["x"], ROLES["y"], TITLE)',
    "compare": 'run_compare(data, ROLES["variable"], TITLE)\nrun_pooled(data, ROLES["variable"], ROLES.get("group"), TITLE)',
    "pooled": 'run_pooled(data, ROLES["variable"], ROLES.get("group"), TITLE)',
}

FOOTER = r'''
# ---- run ---------------------------------------------------------------------
log("kind=%s cohorts=%s rows=%d" % (KIND, COHORTS, len(data)))
_used = [v for v in ROLES.values() if v]
for _v in _used:
    if _v not in HVARS:
        raise ValueError("The analysis refers to '%s', which is not among the harmonized variables %s" % (_v, list(HVARS)))
if len(set(_used)) != len(_used):
    raise ValueError("The same harmonized variable is used in two roles of the analysis: %s" % ROLES)
__DISPATCH__

write_guide()

with open(os.path.join(OUT, "provenance.md"), "w") as fh:
    fh.write("# Mapping record\n\n")
    fh.write("**Analysis:** %s (%s)\n\n" % (TITLE, KIND))
    fh.write("**Cohorts:** %s\n\n" % ", ".join(COHORTS))
    if SHUFFLED:
        fh.write("**DATA SOURCE: SHUFFLED SAMPLES.** A small fragment with independently shuffled columns, "
                 "intended to test that the analysis code runs. No actual results can be drawn from these figures.\n\n")
    if K > 0:
        fh.write("**Suppression:** counts below %d suppressed; bins/cells below %d blanked.\n\n" % (K, K))
    fh.write("**Variables** (harmonized name: cohort::source variable, unit factor, {value map}):\n\n")
    for line in provenance_lines(list(HVARS.keys()), with_values=True):
        fh.write("- %s\n" % line)
    if notes:
        fh.write("\n**Notes:**\n")
        for n_ in notes:
            fh.write("- %s\n" % n_)

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump({"title": TITLE, "kind": KIND, "cohorts": COHORTS, "suppression_k": K, "items": captions,
               "data_source": "shuffled" if SHUFFLED else "full",
               "notes": notes}, fh, indent=2)
log("done")
'''

# Which helper/analysis segments each kind needs.
_NEEDS = {
    "distribution": [H_NUMERIC_SUMMARY, H_CATEGORY_COUNTS, H_FIGSETS, A_DISTRIBUTION],
    "stratified": [H_NUMERIC_SUMMARY, H_FIGSETS, A_STRATIFIED],
    "correlation": [H_FIGSETS, A_CORRELATION],
    "crosstab": [H_FIGSETS, A_CROSSTAB],
    "compare": [H_NUMERIC_SUMMARY, H_CATEGORY_COUNTS, H_SMD, H_FIGSETS, A_STRATIFIED, A_COMPARE, A_POOLED],
    "pooled": [H_NUMERIC_SUMMARY, H_CATEGORY_COUNTS, H_FIGSETS, A_STRATIFIED, A_POOLED],
}
# scipy is part of the enclave image and every figure set uses it (Kruskal-Wallis, chi-square, correlations).
_USES_SCIPY = set(ANALYSIS_KINDS) | _LEGACY_KINDS


def nocode_analysis_script(spec: dict[str, Any]) -> str:
    """Compose the enclave script for one no-code-dcr spec: only the
    loading mode, helpers and analysis the spec actually uses."""
    kind = spec.get("analysis", {}).get("kind", "distribution")
    if kind not in ANALYSIS_KINDS and kind not in _LEGACY_KINDS:
        raise ValueError(f"unknown analysis kind: {kind}")
    cohorts = spec.get("cohorts") or []
    spec_json = json.dumps(spec, ensure_ascii=False).replace('"""', '\\"\\"\\"')
    parts = [
        HEADER
        .replace("__DESCRIPTION__", describe_spec(spec).replace("\n", " "))
        .replace("__SCIPY_IMPORT__", "from scipy import stats\n" if kind in _USES_SCIPY else "")
        .replace("__SPEC_JSON__", spec_json),
        LOAD_MULTI if len(cohorts) > 1 else LOAD_SINGLE,
        PROVENANCE,
        *_NEEDS[kind],
        FOOTER.replace("__DISPATCH__", DISPATCH[kind]),
    ]
    return "".join(parts)
