"""Guided (no-code) analysis: generate the enclave script from an analysis spec.

A domain expert assembles a spec in the guided wizard (frontend
/guided-analysis): an analysis *kind*, the cohorts, and a *mapping* that says
which variable in each cohort plays which role and how their values/units are
harmonized. This module turns that spec into the Python script that runs inside
the Decentriq enclave on the FULL cohort data and writes
figures (PNG), tables (CSV) and a provenance note. Every figure carries a
subtext stating exactly which mapping produced it.

Outputs are figures and summary tables (no data files). Small-cell suppression
is OPTIONAL: spec "suppression_k" (default 0 = off) hides counts below k and
blanks bins/cells below k. Outliers and actual extremes are shown by default —
the values are not linked to any identifying information.

Spec shape (all keys optional unless stated):
{
  "analysis": {
    "kind": "distribution" | "stratified" | "correlation" | "crosstab" | "compare" | "pooled",
    "title": "Body weight by sex",
    "suppression_k": 0,
    "bins": 20,
    "roles": {"variable": "<harmonized name>", "group": "...", "x": "...", "y": "..."}
  },
  "cohorts": ["TIME-CHF", "Aachen-HF"],
  "nodes": {"TIME-CHF": {"data": "TIME-CHF", "dictionary": "TIME-CHF_metadata_dictionary"}, ...},
  "mapping": {
    "id": "...", "name": "...", "created_by": "...", "created_at": "...",
    "variables": [
      {
        "harmonized_name": "sex", "label": "Sex", "type": "categorical" | "numeric",
        "members": {"TIME-CHF": {"var_name": "gender", "unit": ""}, "Aachen-HF": {"var_name": "Geschlecht", "unit": ""}},
        "value_map": {"TIME-CHF": {"1": "male", "2": "female"}, "Aachen-HF": {"M": "male", "W": "female"}},
        "unit_conversion": {"Aachen-HF": {"factor": 0.4536, "from": "lb", "to": "kg"}},
        "evidence": [{"type": "code", "detail": "loinc:46098-0"}, {"type": "cache", "file": "...", "status": "Identical Match"}],
        "notes": ""
      }
    ]
  }
}
"""

from __future__ import annotations

import json
import re
from typing import Any

ANALYSIS_KINDS = {
    "distribution": {
        "label": "Distribution of one variable",
        "roles": ["variable"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "How is a variable distributed in a cohort? Histogram or bar chart plus a summary table.",
    },
    "stratified": {
        "label": "One variable broken down by another",
        "roles": ["variable", "group"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "Compare the distribution of a variable between groups (e.g. weight by sex, NYHA class by diabetes).",
    },
    "correlation": {
        "label": "Relationship between two numeric variables",
        "roles": ["x", "y"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "Scatter plot, correlation coefficients with confidence intervals and p-values, and binned means.",
    },
    "crosstab": {
        "label": "Cross-tabulation of two categorical variables",
        "roles": ["x", "y"],
        "min_cohorts": 1,
        "max_cohorts": 1,
        "blurb": "Counts and percentages for every combination of two categorical variables, with a chi-square test.",
    },
    "compare": {
        "label": "Compare one variable across cohorts",
        "roles": ["variable"],
        "min_cohorts": 2,
        "max_cohorts": 6,
        "blurb": "The same (harmonized) variable side by side in each cohort, with summaries and standardized differences.",
    },
    "pooled": {
        "label": "Pooled distribution across cohorts",
        "roles": ["variable"],
        "optional_roles": ["group"],
        "min_cohorts": 2,
        "max_cohorts": 6,
        "blurb": "Merge a harmonized variable from several cohorts into one distribution, optionally broken down by a harmonized group.",
    },
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "analysis"


def guided_node_name(index: int, spec: dict[str, Any]) -> str:
    kind = spec.get("analysis", {}).get("kind", "analysis")
    title = spec.get("analysis", {}).get("title") or kind
    return f"guided-{index}-{_slug(kind)}-{_slug(title)[:40]}"


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
    return " ".join(parts)


# ---------------------------------------------------------------------------
# The enclave script. A plain template with ONE placeholder for the JSON spec
# (no f-string, so braces are safe).
# ---------------------------------------------------------------------------
_SCRIPT_TEMPLATE = r'''###############################################################################
# GUIDED ANALYSIS — generated by the iCARE4CVD Cohort Explorer
#
# This script was assembled from choices made in the guided (no-code) wizard.
# It reads the full cohort data inside the enclave and writes figures, summary
# tables and a provenance note.
#
#   __DESCRIPTION__
###############################################################################
import csv
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
from scipy import stats

SPEC = json.loads(r"""__SPEC_JSON__""")

OUT = "/output"
FIG_DIR = os.path.join(OUT, "figures")
TAB_DIR = os.path.join(OUT, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
LOG = os.path.join(OUT, "guided_log.txt")

ANALYSIS = SPEC["analysis"]
KIND = ANALYSIS["kind"]
K = int(ANALYSIS.get("suppression_k", 0) or 0)   # 0 = no suppression
BINS = int(ANALYSIS.get("bins", 20))
ROLES = ANALYSIS.get("roles", {})
COHORTS = SPEC["cohorts"]
NODES = SPEC["nodes"]
MAPPING = SPEC["mapping"]
HVARS = {v["harmonized_name"]: v for v in MAPPING.get("variables", [])}

captions = []   # {"figure": ..., "table": ..., "caption": ...}
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
                    # "999=unknown" style -> keep the code before '='
                    parts = [p.split("=")[0].strip() for p in parts]
                    codes[str(row[name_col]).strip()] = set(parts)
    except Exception as e:
        log("dictionary not parsed for missing codes: %s" % e)
    return codes


def find_column(df, name):
    if name in df.columns:
        return name
    low = {c.lower().strip(): c for c in df.columns}
    return low.get(str(name).lower().strip())


# ---- harmonization -----------------------------------------------------------

def harmonize_cohort(cohort):
    node = NODES[cohort]
    df = load_table("/input/" + node["data"])
    missing = load_missing_codes("/input/" + node["dictionary"]) if node.get("dictionary") else {}
    out = pd.DataFrame(index=df.index)
    out["cohort"] = cohort
    for hname, hv in HVARS.items():
        member = (hv.get("members") or {}).get(cohort)
        if not member or not member.get("var_name"):
            out[hname] = np.nan
            continue
        col = find_column(df, member["var_name"])
        if col is None:
            log("%s: column '%s' not found for %s" % (cohort, member["var_name"], hname))
            out[hname] = np.nan
            continue
        s = df[col]
        # declared missing codes -> NaN (compared as stripped strings)
        mc = missing.get(member["var_name"], set())
        if mc:
            s = s.mask(s.astype(str).str.strip().isin(mc))
        if hv.get("type") == "numeric":
            s = pd.to_numeric(s, errors="coerce")
            conv = (hv.get("unit_conversion") or {}).get(cohort)
            if conv and conv.get("factor") not in (None, "", 1, 1.0):
                s = s * float(conv["factor"])
        else:
            vmap = (hv.get("value_map") or {}).get(cohort) or {}
            if vmap:
                norm = {str(k).strip().lower(): v for k, v in vmap.items()}
                key = s.astype(str).str.strip().str.lower()
                # numeric-looking codes: "1.0" should match "1"
                key2 = key.str.replace(r"\.0$", "", regex=True)
                mapped = key.map(norm)
                mapped = mapped.where(mapped.notna(), key2.map(norm))
                s = mapped.where(s.notna(), np.nan)
            else:
                s = s.where(s.notna(), np.nan).astype(object)
        out[hname] = s
    return out


frames = []
for c in COHORTS:
    try:
        frames.append(harmonize_cohort(c))
        log("%s: loaded %d rows" % (c, len(frames[-1])))
    except Exception as e:
        log("%s: FAILED to load (%s)" % (c, e))
        raise
data = pd.concat(frames, ignore_index=True)


# ---- provenance text ---------------------------------------------------------

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
                pairs = ", ".join("%s->%s" % (k, v) for k, v in list(vmap.items())[:6])
                if len(vmap) > 6:
                    pairs += ", ..."
                piece += " (%s)" % pairs
            conv = (hv.get("unit_conversion") or {}).get(c)
            if conv and conv.get("factor") not in (None, "", 1, 1.0):
                piece += " (x%s %s->%s)" % (conv["factor"], conv.get("from", "?"), conv.get("to", "?"))
            members.append(piece)
        ev = []
        for e in hv.get("evidence") or []:
            t = e.get("type")
            if t == "code":
                ev.append("shared code %s" % e.get("detail", ""))
            elif t == "cache":
                ev.append("cached mapping %s (%s)" % (e.get("file", ""), e.get("status", "")))
            elif t == "text":
                ev.append("text similarity %.2f" % float(e.get("score", 0)))
            elif t == "ai":
                ev.append("AI suggestion")
            elif t == "manual":
                ev.append("chosen manually")
        line = "%s := %s" % (hname, " | ".join(members))
        if ev:
            line += " — evidence: " + "; ".join(ev)
        lines.append(line)
    return lines


def footer_text(used):
    lines = provenance_lines(used)
    mname = MAPPING.get("name") or "unnamed"
    who = MAPPING.get("created_by") or "unknown"
    head = "Mapping '%s' (by %s)." % (mname, who) + (" Cells with n<%d suppressed." % K if K > 0 else "")
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
    path = os.path.join(TAB_DIR, name)
    df.to_csv(path, index=False)
    captions.append({"table": "tables/" + name, "caption": caption or name})
    log("table: " + name)


# ---- helpers with suppression ------------------------------------------------

def suppress_count(n):
    return int(n) if int(n) >= K else None


def fmt_count(n):
    return str(int(n)) if (K <= 0 or int(n) >= K) else "<%d" % K


def sup_note(what="bins"):
    """Axis-label suffix describing suppression, empty when it is off."""
    return " (%s with n<%d blanked)" % (what, K) if K > 0 else ""


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


def category_counts(s, label=None):
    s = s.dropna().astype(str)
    vc = s.value_counts()
    rows = []
    for val, n in vc.items():
        rows.append({"group": label, "value": val, "count": fmt_count(n),
                     "percent": round(100.0 * n / len(s), 1) if (K <= 0 or n >= K) and len(s) else None})
    return rows


def hist_counts(s, bins):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0 or (K > 0 and len(s) < K):
        return None, None
    counts, edges = np.histogram(s, bins=bins)
    if K > 0:
        counts = np.where(counts >= K, counts, 0)
    return counts, edges


def smd(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return round(float((a.mean() - b.mean()) / pooled), 3) if pooled > 0 else None


def plain_title(text):
    return str(text)


# ---- analyses ----------------------------------------------------------------

def run_distribution(df, var, label, group_label=None, fname_prefix="distribution"):
    hv = HVARS[var]
    used = [var]
    if hv.get("type") == "numeric":
        counts, edges = hist_counts(df[var], BINS)
        rows = [numeric_summary(df[var], group_label or label)]
        save_table(pd.DataFrame(rows), fname_prefix + "_summary.csv", "Summary of %s" % var)
        if counts is None:
            notes.append("%s: no values available, figure skipped" % var)
            return
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#3b6ea5", edgecolor="white")
        ax.set_xlabel("%s%s" % (hv.get("label") or var, " (%s)" % hv.get("unit") if hv.get("unit") else ""))
        ax.set_ylabel("Patients" + sup_note())
        ax.set_title(plain_title(label))
        save_fig(fig, fname_prefix + ".png", used, "Distribution of %s" % (hv.get("label") or var))
    else:
        rows = category_counts(df[var], group_label or label)
        save_table(pd.DataFrame(rows), fname_prefix + "_counts.csv", "Counts of %s" % var)
        shown = [(r["value"], int(r["count"])) for r in rows if not str(r["count"]).startswith("<")]
        hidden = len(rows) - len(shown)
        if not shown:
            notes.append("%s: no categories to show" % var)
            return
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar([v for v, _ in shown], [n for _, n in shown], color="#3b6ea5")
        ax.set_ylabel("Patients")
        ax.set_title(plain_title(label) + (" (%d small categories hidden)" % hidden if hidden else ""))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save_fig(fig, fname_prefix + ".png", used, "Distribution of %s" % (hv.get("label") or var))


def run_stratified(df, var, group, title, fname_prefix="stratified"):
    hv, gv = HVARS[var], HVARS[group]
    used = [var, group]
    groups = [g for g in df[group].dropna().astype(str).unique()]
    groups.sort()
    if hv.get("type") == "numeric":
        rows = []
        series = []
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for g in groups:
            s = pd.to_numeric(df.loc[df[group].astype(str) == g, var], errors="coerce").dropna()
            rows.append(numeric_summary(s, g))
            if len(s) > 0 and (K <= 0 or len(s) >= K):
                series.append((g, s))
                counts, edges = np.histogram(s, bins=BINS, density=False)
                if K > 0:
                    counts = np.where(counts >= K, counts, 0)
                dens = counts.astype(float) / max(counts.sum(), 1) / np.diff(edges)
                axes[0].step(edges[:-1], dens, where="post", label="%s (n=%d)" % (g, len(s)))
        save_table(pd.DataFrame(rows), fname_prefix + "_summary.csv", "%s by %s" % (var, group))
        axes[0].set_xlabel(hv.get("label") or var)
        axes[0].set_ylabel("Density" + sup_note())
        axes[0].set_title("Distribution per group")
        axes[0].legend(title=gv.get("label") or group)
        if series:
            axes[1].boxplot([s.values for _, s in series], labels=[g for g, _ in series], showfliers=True)
            axes[1].set_ylabel(hv.get("label") or var)
            axes[1].set_title("Box plots (outliers shown)")
            plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
        fig.suptitle(plain_title(title))
        save_fig(fig, fname_prefix + ".png", used, "%s by %s" % (hv.get("label") or var, gv.get("label") or group))
    else:
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), df[group].astype(str).where(df[group].notna()))
        table = ct.copy().astype(object)
        for i in table.index:
            for j in table.columns:
                table.loc[i, j] = fmt_count(ct.loc[i, j])
        table.insert(0, var, table.index)
        save_table(table.reset_index(drop=True), fname_prefix + "_counts.csv", "%s by %s" % (var, group))
        shown = ct.where(ct >= K, 0) if K > 0 else ct
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(len(shown.index))
        width = 0.8 / max(len(shown.columns), 1)
        for k, g in enumerate(shown.columns):
            ax.bar(x + k * width, shown[g].values, width=width, label=str(g))
        ax.set_xticks(x + width * (len(shown.columns) - 1) / 2)
        ax.set_xticklabels([str(i) for i in shown.index], rotation=30, ha="right")
        ax.set_ylabel("Patients" + sup_note("cells"))
        ax.set_title(plain_title(title))
        ax.legend(title=gv.get("label") or group)
        save_fig(fig, fname_prefix + ".png", used, "%s by %s" % (hv.get("label") or var, gv.get("label") or group))


def run_correlation(df, x, y, title, fname_prefix="correlation"):
    hx, hy = HVARS[x], HVARS[y]
    used = [x, y]
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    rows = [{"n": fmt_count(n)}]
    if n >= 4:
        r_p, p_p = stats.pearsonr(sub[x], sub[y])
        r_s, p_s = stats.spearmanr(sub[x], sub[y])
        z = math.atanh(max(min(float(r_p), 0.999999), -0.999999))
        se = 1.0 / math.sqrt(n - 3)
        rows[0].update({"pearson_r": round(float(r_p), 3), "pearson_p": float(p_p),
                        "pearson_ci95_low": round(math.tanh(z - 1.96 * se), 3),
                        "pearson_ci95_high": round(math.tanh(z + 1.96 * se), 3),
                        "spearman_rho": round(float(r_s), 3), "spearman_p": float(p_s)})
    save_table(pd.DataFrame(rows), fname_prefix + "_coefficients.csv", "Correlation of %s and %s" % (x, y))
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
    except Exception:
        pass
    axes[0].set_xlabel(hx.get("label") or x)
    axes[0].set_ylabel(hy.get("label") or y)
    axes[0].set_title("Scatter (n=%d)" % n)
    # binned means: deciles of x
    try:
        q = pd.qcut(sub[x], q=10, duplicates="drop")
        g = sub.groupby(q, observed=True)[y].agg(["mean", "count"])
        centers = [iv.mid for iv in g.index]
        ok = (g["count"] >= K) if K > 0 else (g["count"] > 0)
        axes[1].plot(np.array(centers)[ok.values], g["mean"][ok].values, marker="o", color="#3b6ea5")
        binned = pd.DataFrame({"x_bin": [str(iv) for iv in g.index], "mean_y": g["mean"].round(3).values,
                               "n": [fmt_count(c) for c in g["count"].values]})
        save_table(binned, fname_prefix + "_binned_means.csv", "Mean of %s by deciles of %s" % (y, x))
    except Exception as e:
        log("binned means failed: %s" % e)
    axes[1].set_xlabel("%s (deciles)" % (hx.get("label") or x))
    axes[1].set_ylabel("mean %s" % (hy.get("label") or y))
    axes[1].set_title("Binned means")
    fig.suptitle("%s  —  Pearson r=%.2f (p=%.3g), Spearman rho=%.2f (p=%.3g), n=%d" % (
        plain_title(title), rows[0].get("pearson_r", float("nan")), rows[0].get("pearson_p", float("nan")),
        rows[0].get("spearman_rho", float("nan")), rows[0].get("spearman_p", float("nan")), n))
    save_fig(fig, fname_prefix + ".png", used, "Relationship between %s and %s" % (hx.get("label") or x, hy.get("label") or y))


def run_crosstab(df, x, y, title, fname_prefix="crosstab"):
    hx, hy = HVARS[x], HVARS[y]
    used = [x, y]
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
    save_table(table.reset_index(drop=True), fname_prefix + "_counts.csv", "%s by %s (row %%)" % (x, y))
    stat_rows = [{"n": fmt_count(int(ct.values.sum()))}]
    if ct.shape[0] > 1 and ct.shape[1] > 1 and ct.values.sum() > 0:
        chi2, p, dof, expected = stats.chi2_contingency(ct.values)
        v = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else None
        stat_rows[0].update({"chi_square": round(float(chi2), 3), "dof": int(dof), "p_value": float(p),
                             "cramers_v": round(v, 3) if v is not None else None,
                             "note": "expected counts < 5 in some cells" if (expected < 5).any() else ""})
    save_table(pd.DataFrame(stat_rows), fname_prefix + "_chi_square.csv", "Chi-square test")
    shown = ct.where(ct >= K, 0) if K > 0 else ct
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(shown.index))
    for col in shown.columns:
        ax.bar([str(i) for i in shown.index], shown[col].values, bottom=bottom, label=str(col))
        bottom += shown[col].values
    ax.set_ylabel("Patients" + sup_note("cells"))
    ax.set_xlabel(hx.get("label") or x)
    ax.legend(title=hy.get("label") or y)
    ax.set_title(plain_title(title))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, fname_prefix + ".png", used, "%s by %s" % (hx.get("label") or x, hy.get("label") or y))


def run_compare(df, var, title, fname_prefix="compare"):
    hv = HVARS[var]
    used = [var]
    if hv.get("type") == "numeric":
        rows = [numeric_summary(df.loc[df["cohort"] == c, var], c) for c in COHORTS]
        # pairwise standardized mean differences
        for i in range(len(COHORTS)):
            for j in range(i + 1, len(COHORTS)):
                rows.append({"group": "SMD %s vs %s" % (COHORTS[i], COHORTS[j]),
                             "mean": smd(df.loc[df["cohort"] == COHORTS[i], var], df.loc[df["cohort"] == COHORTS[j], var])})
        save_table(pd.DataFrame(rows), fname_prefix + "_summary.csv", "%s per cohort" % var)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        allv = pd.to_numeric(df[var], errors="coerce").dropna()
        edges = np.histogram_bin_edges(allv, bins=BINS) if len(allv) else None
        for c in COHORTS:
            s = pd.to_numeric(df.loc[df["cohort"] == c, var], errors="coerce").dropna()
            if len(s) > 0 and (K <= 0 or len(s) >= K) and edges is not None:
                counts, _ = np.histogram(s, bins=edges)
                if K > 0:
                    counts = np.where(counts >= K, counts, 0)
                dens = counts.astype(float) / max(counts.sum(), 1) / np.diff(edges)
                ax.step(edges[:-1], dens, where="post", label="%s (n=%d)" % (c, len(s)))
        ax.set_xlabel("%s%s" % (hv.get("label") or var, " (%s)" % hv.get("unit") if hv.get("unit") else ""))
        ax.set_ylabel("Density" + sup_note())
        ax.set_title(plain_title(title))
        ax.legend()
        save_fig(fig, fname_prefix + ".png", used, "%s across cohorts" % (hv.get("label") or var))
    else:
        rows = []
        for c in COHORTS:
            rows.extend(category_counts(df.loc[df["cohort"] == c, var], c))
        save_table(pd.DataFrame(rows), fname_prefix + "_counts.csv", "%s per cohort" % var)
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
        ax.set_title(plain_title(title))
        ax.legend()
        save_fig(fig, fname_prefix + ".png", used, "%s across cohorts" % (hv.get("label") or var))


def run_pooled(df, var, group, title, fname_prefix="pooled"):
    hv = HVARS[var]
    used = [var] + ([group] if group else [])
    rows = [numeric_summary(df[var], "pooled")] if hv.get("type") == "numeric" else category_counts(df[var], "pooled")
    for c in COHORTS:
        sub = df.loc[df["cohort"] == c, var]
        rows.extend([numeric_summary(sub, c)] if hv.get("type") == "numeric" else category_counts(sub, c))
    save_table(pd.DataFrame(rows), fname_prefix + "_summary.csv", "Pooled %s" % var)
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
        ax.set_xlabel("%s%s" % (hv.get("label") or var, " (%s)" % hv.get("unit") if hv.get("unit") else ""))
        ax.set_ylabel("Patients, stacked by cohort" + sup_note())
        ax.set_title(plain_title(title))
        ax.legend()
        save_fig(fig, fname_prefix + ".png", used, "Pooled distribution of %s" % (hv.get("label") or var))
    else:
        ct = pd.crosstab(df[var].astype(str).where(df[var].notna()), df["cohort"])
        shown = ct.where(ct >= K, 0) if K > 0 else ct
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bottom = np.zeros(len(shown.index))
        for c in shown.columns:
            ax.bar([str(i) for i in shown.index], shown[c].values, bottom=bottom, label=str(c))
            bottom += shown[c].values
        ax.set_ylabel("Patients, stacked by cohort" + sup_note("cells"))
        ax.set_title(plain_title(title))
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save_fig(fig, fname_prefix + ".png", used, "Pooled distribution of %s" % (hv.get("label") or var))
    if group:
        run_stratified(df, var, group, title + " by " + (HVARS[group].get("label") or group), fname_prefix="pooled_by_group")


# ---- dispatch ----------------------------------------------------------------

title = ANALYSIS.get("title") or KIND
log("kind=%s cohorts=%s rows=%d" % (KIND, COHORTS, len(data)))
if KIND == "distribution":
    run_distribution(data, ROLES["variable"], title)
elif KIND == "stratified":
    run_stratified(data, ROLES["variable"], ROLES["group"], title)
elif KIND == "correlation":
    run_correlation(data, ROLES["x"], ROLES["y"], title)
elif KIND == "crosstab":
    run_crosstab(data, ROLES["x"], ROLES["y"], title)
elif KIND == "compare":
    run_compare(data, ROLES["variable"], title)
elif KIND == "pooled":
    run_pooled(data, ROLES["variable"], ROLES.get("group"), title)
else:
    raise ValueError("unknown analysis kind: %s" % KIND)

# ---- provenance + summary ----------------------------------------------------
with open(os.path.join(OUT, "provenance.md"), "w") as fh:
    fh.write("# Guided analysis provenance\n\n")
    fh.write("**Analysis:** %s (%s)\n\n" % (title, KIND))
    fh.write("**Cohorts:** %s\n\n" % ", ".join(COHORTS))
    fh.write(("**Suppression:** counts below %d are suppressed; bins/cells below %d blanked.\n\n" % (K, K)) if K > 0 else "**Suppression:** none (all counts and values shown).\n\n")
    fh.write("**Mapping:** %s (id %s, by %s, %s)\n\n" % (MAPPING.get("name", "unnamed"), MAPPING.get("id", "-"),
                                                           MAPPING.get("created_by", "-"), MAPPING.get("created_at", "-")))
    for line in provenance_lines(list(HVARS.keys())):
        fh.write("- %s\n" % line)
    if notes:
        fh.write("\n**Notes:**\n")
        for n_ in notes:
            fh.write("- %s\n" % n_)

with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump({"title": title, "kind": KIND, "cohorts": COHORTS, "suppression_k": K,
               "items": captions, "notes": notes, "mapping_name": MAPPING.get("name"),
               "mapping_id": MAPPING.get("id")}, fh, indent=2)
log("done")
'''


def guided_analysis_script(spec: dict[str, Any]) -> str:
    """Render the enclave script for one guided-analysis spec."""
    spec_json = json.dumps(spec, ensure_ascii=False)
    # The spec is embedded in a raw triple-quoted string; guard the one sequence
    # that could terminate it early.
    spec_json = spec_json.replace('"""', '\\"\\"\\"')
    return (
        _SCRIPT_TEMPLATE
        .replace("__SPEC_JSON__", spec_json)
        .replace("__DESCRIPTION__", describe_spec(spec).replace("\n", " "))
    )
