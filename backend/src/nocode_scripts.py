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


def nocode_node_name(index: int, spec: dict[str, Any]) -> str:
    kind = spec.get("analysis", {}).get("kind", "analysis")
    title = spec.get("analysis", {}).get("title") or kind
    return f"nocode-{index}-{_slug(kind)}-{_slug(title)[:40]}"


def describe_spec(spec: dict[str, Any]) -> str:
    """Plain-language recipe of what the script will compute (used for the DCR
    description and the wizard's review step)."""
    a = spec.get("analysis", {})
    kind = a.get("kind", "")
    roles = a.get("roles", {})
    cohorts = spec.get("cohorts", [])
    meta = ANALYSIS_KINDS.get(kind, {})
    parts = [f"{meta.get('label', kind)} in {', '.join(cohorts)}."]
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
    """CSV first, then SPSS; RawDataNodeDefinition mounts the uploaded file as-is."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_spss(path)


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
        s = s.mask(s.astype(str).str.strip().isin(mc))
    if hv.get("type") == "numeric":
        s = pd.to_numeric(s, errors="coerce")
        conv = (hv.get("unit_conversion") or {}).get(cohort)
        if conv and conv.get("factor") not in (None, "", 1, 1.0):
            s = s * float(conv["factor"])
            log("%s: %s scaled by %s (%s -> %s)" % (cohort, member["var_name"], conv["factor"], conv.get("from"), conv.get("to")))
        return s
    vmap = (hv.get("value_map") or {}).get(cohort) or {}
    if vmap:
        norm = {str(k).strip().lower(): v for k, v in vmap.items()}
        key = s.astype(str).str.strip().str.lower()
        key2 = key.str.replace(r"\.0$", "", regex=True)     # "1.0" should match "1"
        mapped = key.map(norm)
        mapped = mapped.where(mapped.notna(), key2.map(norm))
        mapped = mapped.replace("", np.nan)        # "" in the value map = treated as missing
        unmapped = int((s.notna() & mapped.isna()).sum())
        if unmapped:
            log("%s: %d value(s) of %s not in the value map -> missing" % (cohort, unmapped, member["var_name"]))
        return mapped.where(s.notna(), np.nan)
    return s.where(s.notna(), np.nan).astype(object)

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

def provenance_lines(used):
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
            piece = "%s [%s]" % (m["var_name"], c)
            vmap = (hv.get("value_map") or {}).get(c) or {}
            if vmap:
                pairs = ", ".join("%s->%s" % (k, v or "(missing)") for k, v in list(vmap.items())[:6])
                piece += " (%s%s)" % (pairs, ", ..." if len(vmap) > 6 else "")
            conv = (hv.get("unit_conversion") or {}).get(c)
            if conv and conv.get("factor") not in (None, "", 1, 1.0):
                piece += " (x%s %s->%s)" % (conv["factor"], conv.get("from", "?"), conv.get("to", "?"))
            members.append(piece)
        ev = []
        for e in hv.get("evidence") or []:
            t = e.get("type")
            if t == "code":
                ev.append("same %s %s" % (e.get("system") or "code", e.get("detail", "")))
            elif t == "cache":
                ev.append("computed mapping %s (%s)" % (e.get("file", ""), e.get("status", "")))
            elif t == "text":
                ev.append("text match %d%%" % round(float(e.get("score", 0)) * 100))
            elif t == "ai":
                ev.append("AI suggestion")
            elif t == "manual":
                ev.append("chosen manually")
        line = "%s := %s" % (hname, " | ".join(members))
        if ev:
            line += "; evidence: " + "; ".join(ev)
        lines.append(line)
    return lines


def footer_text(used):
    head = "Mapping '%s' (by %s)." % (MAPPING.get("name") or "unnamed", MAPPING.get("created_by") or "unknown")
    if SHUFFLED:
        head = "SHUFFLED SAMPLES - code test only, columns shuffled independently: no results can be drawn. " + head
    if K > 0:
        head += " Cells with n<%d suppressed." % K
    lines = provenance_lines(used)
    return head + ("\n" + "\n".join(lines) if lines else "")


def save_fig(fig, name, used, caption):
    txt = footer_text(used)
    wrapped = "\n".join(textwrap.fill(l, 150) for l in txt.split("\n"))
    n_lines = wrapped.count("\n") + 1
    fig.subplots_adjust(bottom=0.12 + 0.025 * n_lines)
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

H_HIST = r'''
def hist_counts(s, bins):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0 or (K > 0 and len(s) < K):
        return None, None
    counts, edges = np.histogram(s, bins=bins)
    if K > 0:
        counts = np.where(counts >= K, counts, 0)
    return counts, edges

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

A_DISTRIBUTION = r'''
# ---- analysis: distribution --------------------------------------------------

def run_distribution(df, var, label, group_label=None, prefix="distribution"):
    hv = HVARS[var]
    if hv.get("type") == "numeric":
        counts, edges = hist_counts(df[var], BINS)
        save_table(pd.DataFrame([numeric_summary(df[var], group_label or label)]), prefix + "_summary.csv", "Summary of %s" % var)
        if counts is None:
            notes.append("%s: no values available, figure skipped" % var)
            return
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#3b6ea5", edgecolor="white")
        ax.set_xlabel(axis_label(hv))
        ax.set_ylabel("Patients" + sup_note())
        ax.set_title(label)
        save_fig(fig, prefix + ".png", [var], "Distribution of %s" % (hv.get("label") or var))
    else:
        rows = category_counts(df[var], group_label or label)
        save_table(pd.DataFrame(rows), prefix + "_counts.csv", "Counts of %s" % var)
        shown = [(r["value"], int(r["count"])) for r in rows if not str(r["count"]).startswith("<")]
        if not shown:
            notes.append("%s: no categories to show" % var)
            return
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar([v for v, _ in shown], [n for _, n in shown], color="#3b6ea5")
        ax.set_ylabel("Patients")
        hidden = len(rows) - len(shown)
        ax.set_title(label + (" (%d small categories hidden)" % hidden if hidden else ""))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save_fig(fig, prefix + ".png", [var], "Distribution of %s" % (hv.get("label") or var))

'''

A_STRATIFIED = r'''
# ---- analysis: one variable broken down by another ---------------------------

def run_stratified(df, var, group, title, prefix="stratified"):
    hv, gv = HVARS[var], HVARS[group]
    used = [var, group]
    groups = sorted(df[group].dropna().astype(str).unique())
    if hv.get("type") == "numeric":
        rows, series = [], []
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for g in groups:
            s = pd.to_numeric(df.loc[df[group].astype(str) == g, var], errors="coerce").dropna()
            rows.append(numeric_summary(s, g))
            if len(s) > 0 and (K <= 0 or len(s) >= K):
                series.append((g, s))
                counts, edges = np.histogram(s, bins=BINS)
                if K > 0:
                    counts = np.where(counts >= K, counts, 0)
                dens = counts.astype(float) / max(counts.sum(), 1) / np.diff(edges)
                axes[0].step(edges[:-1], dens, where="post", label="%s (n=%d)" % (g, len(s)))
        save_table(pd.DataFrame(rows), prefix + "_summary.csv", "%s by %s" % (var, group))
        axes[0].set_xlabel(axis_label(hv))
        axes[0].set_ylabel("Density" + sup_note())
        axes[0].set_title("Distribution per group")
        axes[0].legend(title=gv.get("label") or group)
        if series:
            axes[1].boxplot([s.values for _, s in series], labels=[g for g, _ in series], showfliers=True)
            axes[1].set_ylabel(axis_label(hv))
            axes[1].set_title("Box plots (outliers shown)")
            plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
        fig.suptitle(title)
        save_fig(fig, prefix + ".png", used, "%s by %s" % (hv.get("label") or var, gv.get("label") or group))
    else:
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), df[group].astype(str).where(df[group].notna()))
        table = ct.astype(object).copy()
        for i in table.index:
            for j in table.columns:
                table.loc[i, j] = fmt_count(ct.loc[i, j])
        table.insert(0, var, table.index)
        save_table(table.reset_index(drop=True), prefix + "_counts.csv", "%s by %s" % (var, group))
        shown = ct.where(ct >= K, 0) if K > 0 else ct
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(shown.index))
        width = 0.8 / max(len(shown.columns), 1)
        for k, g in enumerate(shown.columns):
            ax.bar(x + k * width, shown[g].values, width=width, label=str(g))
        ax.set_xticks(x + width * (len(shown.columns) - 1) / 2)
        ax.set_xticklabels([str(i) for i in shown.index], rotation=30, ha="right")
        ax.set_ylabel("Patients" + sup_note("cells"))
        ax.set_title(title)
        ax.legend(title=gv.get("label") or group)
        save_fig(fig, prefix + ".png", used, "%s by %s" % (hv.get("label") or var, gv.get("label") or group))

'''

A_CORRELATION = r'''
# ---- analysis: relationship between two numeric variables --------------------

def run_correlation(df, x, y, title, prefix="correlation"):
    hx, hy = HVARS[x], HVARS[y]
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
                    "spearman_rho": round(float(r_s), 3), "spearman_p": float(p_s)})
    save_stats(row, prefix + "_coefficients.txt", "Correlation of %s and %s" % (x, y))
    if n < 4:
        notes.append("correlation: fewer than 4 complete pairs, figure skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(sub[x], sub[y], s=10, alpha=0.5, color="#3b6ea5", edgecolors="none")
    try:
        slope, intercept = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(float(sub[x].min()), float(sub[x].max()), 50)
        axes[0].plot(xs, slope * xs + intercept, color="#c0392b", linewidth=1.2, label="least-squares fit")
        axes[0].legend()
    except Exception as e:
        log("fit line skipped: %s" % e)
    axes[0].set_xlabel(axis_label(hx))
    axes[0].set_ylabel(axis_label(hy))
    axes[0].set_title("Scatter (n=%d)" % n)
    try:
        q = pd.qcut(sub[x], q=10, duplicates="drop")
        g = sub.groupby(q, observed=True)[y].agg(["mean", "count"])
        centers = np.array([iv.mid for iv in g.index])
        ok = (g["count"] >= K).values if K > 0 else (g["count"] > 0).values
        axes[1].plot(centers[ok], g["mean"].values[ok], marker="o", color="#3b6ea5")
        save_table(pd.DataFrame({"x_bin": [str(iv) for iv in g.index], "mean_y": g["mean"].round(3).values,
                                 "n": [fmt_count(c) for c in g["count"].values]}),
                   prefix + "_binned_means.csv", "Mean of %s by deciles of %s" % (y, x))
    except Exception as e:
        log("binned means skipped: %s" % e)
    axes[1].set_xlabel("%s (deciles)" % (hx.get("label") or x))
    axes[1].set_ylabel("mean %s" % (hy.get("label") or y))
    axes[1].set_title("Binned means")
    fig.suptitle("%s. Pearson r=%.2f (p=%.3g), Spearman rho=%.2f (p=%.3g), n=%d" % (
        title, row.get("pearson_r", float("nan")), row.get("pearson_p", float("nan")),
        row.get("spearman_rho", float("nan")), row.get("spearman_p", float("nan")), n))
    save_fig(fig, prefix + ".png", [x, y], "Relationship between %s and %s" % (hx.get("label") or x, hy.get("label") or y))

'''

A_CROSSTAB = r'''
# ---- analysis: cross-tabulation ----------------------------------------------

def run_crosstab(df, x, y, title, prefix="crosstab"):
    hx, hy = HVARS[x], HVARS[y]
    a = df[x].astype(str).where(df[x].notna())
    b = df[y].astype(str).where(df[y].notna())
    ct = pd.crosstab(a, b)
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
                     "note": "expected counts < 5 in some cells" if (expected < 5).any() else ""})
    save_stats(stat, prefix + "_chi_square.txt", "Chi-square test of independence")
    shown = ct.where(ct >= K, 0) if K > 0 else ct
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(shown.index))
    for col in shown.columns:
        ax.bar([str(i) for i in shown.index], shown[col].values, bottom=bottom, label=str(col))
        bottom += shown[col].values
    ax.set_ylabel("Patients" + sup_note("cells"))
    ax.set_xlabel(hx.get("label") or x)
    ax.legend(title=hy.get("label") or y)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, prefix + ".png", [x, y], "%s by %s" % (hx.get("label") or x, hy.get("label") or y))

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
        allv = pd.to_numeric(df[var], errors="coerce").dropna()
        if len(allv) == 0:
            notes.append("%s: no values available" % var)
            return
        edges = np.histogram_bin_edges(allv, bins=BINS)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for c in COHORTS:
            s = pd.to_numeric(df.loc[df["cohort"] == c, var], errors="coerce").dropna()
            if len(s) > 0 and (K <= 0 or len(s) >= K):
                counts, _ = np.histogram(s, bins=edges)
                if K > 0:
                    counts = np.where(counts >= K, counts, 0)
                dens = counts.astype(float) / max(counts.sum(), 1) / np.diff(edges)
                ax.step(edges[:-1], dens, where="post", label="%s (n=%d)" % (c, len(s)))
        ax.set_xlabel(axis_label(hv))
        ax.set_ylabel("Density" + sup_note())
        ax.set_title(title)
        ax.legend()
        save_fig(fig, prefix + ".png", [var], "%s across cohorts" % (hv.get("label") or var))
    else:
        rows = []
        for c in COHORTS:
            rows.extend(category_counts(df.loc[df["cohort"] == c, var], c))
        save_table(pd.DataFrame(rows), prefix + "_counts.csv", "%s per cohort" % var)
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), df["cohort"])
        pct = ct.div(ct.sum(axis=0), axis=1) * 100
        if K > 0:
            pct = pct.where(ct >= K, 0)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(pct.index))
        width = 0.8 / max(len(pct.columns), 1)
        for k, c in enumerate(pct.columns):
            ax.bar(x + k * width, pct[c].values, width=width, label=str(c))
        ax.set_xticks(x + width * (len(pct.columns) - 1) / 2)
        ax.set_xticklabels([str(i) for i in pct.index], rotation=30, ha="right")
        ax.set_ylabel("% of cohort" + sup_note("cells"))
        ax.set_title(title)
        ax.legend()
        save_fig(fig, prefix + ".png", [var], "%s across cohorts" % (hv.get("label") or var))

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
        edges = np.histogram_bin_edges(allv, bins=BINS)
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
        ax.set_ylabel("Patients, stacked by cohort" + sup_note())
        ax.set_title(title)
        ax.legend()
        save_fig(fig, prefix + ".png", used, "Pooled distribution of %s" % (hv.get("label") or var))
    else:
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), df["cohort"])
        shown = ct.where(ct >= K, 0) if K > 0 else ct
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bottom = np.zeros(len(shown.index))
        for c in shown.columns:
            ax.bar([str(i) for i in shown.index], shown[c].values, bottom=bottom, label=str(c))
            bottom += shown[c].values
        ax.set_ylabel("Patients, stacked by cohort" + sup_note("cells"))
        ax.set_title(title)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save_fig(fig, prefix + ".png", used, "Pooled distribution of %s" % (hv.get("label") or var))
    if group:
        run_stratified(df, var, group, title + " by " + (HVARS[group].get("label") or group), prefix="pooled_by_group")

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
__DISPATCH__

with open(os.path.join(OUT, "provenance.md"), "w") as fh:
    fh.write("# No-code DCR analysis provenance\n\n")
    fh.write("**Analysis:** %s (%s)\n\n" % (TITLE, KIND))
    fh.write("**Cohorts:** %s\n\n" % ", ".join(COHORTS))
    if SHUFFLED:
        fh.write("**DATA SOURCE: SHUFFLED SAMPLES.** A small fragment with independently shuffled columns, "
                 "intended to test that the analysis code runs. No actual results can be drawn from these figures.\n\n")
    fh.write(("**Suppression:** counts below %d suppressed; bins/cells below %d blanked.\n\n" % (K, K)) if K > 0
             else "**Suppression:** none (all counts and values shown).\n\n")
    fh.write("**Mapping:** %s (id %s, by %s, %s)\n\n" % (MAPPING.get("name", "unnamed"), MAPPING.get("id", "-"),
                                                           MAPPING.get("created_by", "-"), MAPPING.get("created_at", "-")))
    for line in provenance_lines(list(HVARS.keys())):
        fh.write("- %s\n" % line)
    if notes:
        fh.write("\n**Notes:**\n")
        for n_ in notes:
            fh.write("- %s\n" % n_)

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump({"title": TITLE, "kind": KIND, "cohorts": COHORTS, "suppression_k": K, "items": captions,
               "data_source": "shuffled" if SHUFFLED else "full",
               "notes": notes, "mapping_name": MAPPING.get("name"), "mapping_id": MAPPING.get("id")}, fh, indent=2)
log("done")
'''

# Which helper/analysis segments each kind needs.
_NEEDS = {
    "distribution": [H_NUMERIC_SUMMARY, H_CATEGORY_COUNTS, H_HIST, A_DISTRIBUTION],
    "stratified": [H_NUMERIC_SUMMARY, A_STRATIFIED],
    "correlation": [A_CORRELATION],
    "crosstab": [A_CROSSTAB],
    "compare": [H_NUMERIC_SUMMARY, H_CATEGORY_COUNTS, H_SMD, H_HIST, A_STRATIFIED, A_COMPARE, A_POOLED],
    "pooled": [H_NUMERIC_SUMMARY, H_CATEGORY_COUNTS, A_STRATIFIED, A_POOLED],
}
_USES_SCIPY = {"correlation", "crosstab"}


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
