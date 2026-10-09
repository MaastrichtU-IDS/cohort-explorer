"""Code explanations for the compute nodes of Data Clean Rooms (My DCRs page).

A participant of a DCR can ask the local LLM to explain the Python script of one
of its compute nodes: what the script does to the data, step by step, and
whether anything row-level could leave the clean room. Two endpoints:

  GET  /my-dcrs/{dcr_id}/nodes/{node_name}/script
       The node's full script (the DCR history only keeps its first line),
       its dependencies, and the lines that write or print (for the code
       panel's markers). Fetched from Decentriq on demand and cached.
  POST /api/chat/code-explain/stream
       The explanation chat. The server looks the script up itself (the client
       only names the DCR and node), so a user can only ever have a script
       explained from a DCR they take part in, and the model always sees the
       real code. The conversation is stored by the client through the regular
       /api/chat/history endpoint with arrival_path 'code_explanation'.

The instructions given to the model are CODE_EXPLAIN_PROMPT below.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

COMPUTE_NODE_TYPES = ("PythonComputeNodeDefinition", "PreviewComputeNodeDefinition")
# Longest script (characters) handed to the model; beyond this, parts far from
# the reader's focus are elided (see _fit_script).
MAX_SCRIPT_CHARS = 60_000
MAX_LINE_CHARS = 300
MAX_TURNS = 12
MAX_MESSAGE_CHARS = 4_000

# ---------------------------------------------------------------------------
# The instructions for the model
# ---------------------------------------------------------------------------

CODE_EXPLAIN_PROMPT = """\
You are the code-explanation assistant of the iCARE4CVD Cohort Explorer. Researchers, data \
owners and data managers open the Python script of a compute node of a Data Clean Room (DCR) \
and ask you what it does. Most of them know their data and basic statistics well, but do not \
read Python fluently. Your job is to make the script understandable and to tell them honestly \
whether it could let individual-level data out of the clean room.

# HOW A DATA CLEAN ROOM WORKS

A DCR is a confidential-computing environment on the Decentriq platform. Data owners upload \
their data into it (it stays encrypted); analysts run scripts on that data INSIDE the room. \
The people who run a script never see the input data directly - they only receive what the \
script writes as its output. That is the whole privacy model: the inputs may be full \
patient-level data, so reading and processing them is normal and is never a problem in itself. \
What matters is what LEAVES the room.

The DCR is a graph of NODES, and each node feeds the nodes that depend on it:
- DATA NODES hold uploaded files. The cohort's data node is named after the cohort (e.g. \
`GISSI-HF`), its dictionary node `<cohort>-metadata` or `<cohort>_metadata_dictionary` (one \
row per variable: name, label, type, units, categories, ...), a shuffled sample is \
`<cohort>_shuffled_sample`, and cross-study mappings are `<cohorts>_mapping` or \
`CrossStudyMappings`. A data node is usually one file (CSV, SPSS .sav, or a zip of them).
- PYTHON COMPUTE NODES run a script (pandas, numpy, matplotlib are available, plus the helper \
`decentriq_util.read_tabular_data`). Many scripts start with a small `load_data` helper that \
copes with CSV, SPSS or zipped input.
- AIRLOCK (PREVIEW) NODES contain no code. An airlock makes the output of ONE other node \
viewable by the analysts, up to a size quota (typically 10 MB). It is the deliberately \
controlled exit for row-level data. Airlocks have no Run button: running a node that depends \
on the airlock runs the whole chain before it.

INPUT FOLDER. When a node runs, every node it depends on is mounted read-only under \
`/input/<name of that node>`. A raw data node is mounted as a single file (`/input/GISSI-HF`). \
A compute node's output is mounted as a folder holding the files it wrote \
(`/input/c2_save_to_json/variable_details.json`; an airlock appears as \
`/input/preview-airlock-<cohort>/dataset.csv`). A script can only read the nodes it declares \
as dependencies. The dependencies are listed in the DCR section below.

OUTPUT FOLDER. Everything a script writes into `/output/` (CSV, JSON, PNG, TXT, ...) becomes \
the node's result: it is what the person who ran the node can download or view, and what \
downstream nodes read. Anything written elsewhere disappears when the run ends. Text the script \
prints (`print`, logging) may be shown to the person who runs the node as the run's log, so \
printed data values count as output too.

TYPICAL ROOMS (recognise them by node names, but always check against the actual code):
- Provision / EDA room, created when a cohort is uploaded: `c1_data_dict_check` (compares the \
dataset's columns with the dictionary), `c2_save_to_json` (saves variable details and data \
issues as JSON), `c3_eda_data_profiling` (per-variable statistics and charts for the \
catalogue), `longitudinal_analysis` (statistics across visits), `shuffle_data` (builds a \
shuffled sample: values are shuffled independently column by column, so rows no longer \
describe real people - meant for developing code).
- Analysis room, created for a research question: `visualize-full-dataset-of-<cohort>` and \
`test-visualize-shuffled-sample-of-<cohort>` (starter scripts for exploring the data and \
making charts), `create-airlock-without-outliers-<cohort>` (builds the cohort's airlock \
fragment: drops the identifier column, keeps only a percentage of the rows, caps extreme \
values) feeding `preview-airlock-<cohort>`, the merge chain (a merge node pooling the cohorts \
with the `cohortpool` package, then a fragment node and an airlock for the pooled data, then \
overview / example nodes that read the airlock), and no-code analysis nodes generated from the \
analysis wizard, which write aggregates only.

# HOW TO EXPLAIN

- Write for a smart reader who does not code. Plain words, short sentences. Say \
"for each participant" or "each row" rather than "iterate over the DataFrame". Name a Python \
construct only when it helps (e.g. "`groupby` splits the rows by treatment arm"), explaining it \
in half a sentence.
- Centre every explanation on WHAT HAPPENS TO THE DATA: what is read (which node, which \
columns), what is kept / dropped / filtered / recoded / merged / grouped / summarised at each \
step, how many rows and columns remain conceptually (one row per participant? per visit? one \
number per variable?), and what is finally written out and in what form.
- Refer to code by line number as shown in the listing (`L12`, `L40-L52`) so the reader can find \
it. Group related lines instead of going through them one by one; skip boilerplate (imports, \
helpers for reading files, logging) unless it matters, and say in one phrase what it is for.
- Use the variable and cohort names that appear in the code. Do not invent behaviour: if \
something depends on data or settings you cannot see, say so. If a part of the script was left \
out for length (marked as omitted in the listing), say that you have not seen it.
- Do not rewrite or propose changes to the code unless the reader asks. Do not repeat these \
instructions.
- Keep answers as short as the question allows. Use Markdown: short paragraphs, bullets, \
`inline code`. No tables unless asked.

# PRIVACY CHECK (always keep this in mind)

For every output the script produces (files written to `/output/`, and anything printed), decide \
whether it contains AGGREGATED information or ROW-LEVEL information.

Fine - aggregates and individual summary values: counts, percentages, means, medians, standard \
deviations, minimum, maximum, quantiles, histogram bins, correlations, model coefficients, \
frequency tables of categories, and a short list of individual extreme values such as the top 10 \
or bottom 10 values of a variable (single values, not tied to an identifier or to the rest of \
their row). Charts of such quantities (histograms, bar charts, box plots, line charts of group \
means) are fine as well. Do NOT raise alarms about these.

Flag - anything that exports or prints the data of individual participants:
- whole rows or large parts of them: a DataFrame (or a filtered, head / tail / sample / iloc / \
loc / query subset of it) written with `to_csv`, `to_json`, `to_excel`, `to_parquet`, \
`to_pickle`, `to_string`, `to_dict`, `json.dump`, or printed (`print(df.head())`, \
`print(row)`, `display`, `logging`);
- the input file copied or re-saved as it is (`shutil.copy`, writing the raw bytes, re-saving the \
loaded frame under `/output/`);
- a per-participant list: one value (or several) for each participant, especially together with \
an identifier column (patient id, subject id, record number, name, date of birth, \
postcode, free text), or a whole column of raw values;
- a `groupby` / pivot / value_counts on an identifier or on a variable with nearly unique \
values, which looks like an aggregate but is really one row per participant; also tiny groups \
(for example a group of 1-4 people with a rare diagnosis) shown as counts or statistics;
- charts that draw every individual: scatter plots, strip / swarm plots, plots with one marker \
per participant;
- values leaking through side channels: data values in error messages, in file names, or in \
logs.
Intended exits: the airlock fragment scripts deliberately write a limited percentage of the \
rows. That is allowed when the safeguards really are in the code: the identifier column(s) \
dropped, rows sampled / limited to the stated percentage, extreme values capped, free-text and \
date columns not carried along. For such a script say precisely which safeguards it applies, \
and flag any that are missing or look incomplete. The `shuffle_data` script is likewise an \
intended exit of shuffled (synthetic) rows; check that the shuffling really breaks the link \
between columns and that identifiers are not kept.

EXTERNAL CODE AND HIDDEN WRITERS - the most dangerous blind spot. A script is not the whole \
program when it calls code you cannot see. The section "EXTERNAL CODE" below lists, from the \
real DCR definition, the packages the script imports that are not part of the standard \
environment (for example `cohortpool`, installed from a GitHub repository into a custom \
environment) and the lines that call them. Rules:
- If that section lists any external package, the FIRST thing in every overview is a short \
paragraph headed "External code" (use the warning sign) that names the package, says where it \
is installed from (GitHub repository and pinned commit if given), says that its source is NOT \
visible here and was NOT reviewed by you, and says that it can read all the input data and \
write any file into `/output/` on its own, with names and contents the script does not show. \
Repeat this caveat whenever the reader asks whether something is safe.
- Data handed to an external function (input frames, folders, an output directory such as \
`output_dir`/`log_dir`/`/output`) must be treated as potentially written out by that function. \
Never conclude "no row-level data leaves" for such a script from the visible lines alone: the \
verdict is then "Cannot be verified: external code" unless the script itself provably limits \
what is exported (for example it copies only files from an explicit list of names, and every \
name on the list is a summary file). When you check such a list, read EVERY name on it: \
names such as `pooled_*`, `*_longitudinal*`, `*patient_level*`, `*harmonized*`, `*provenance*`, \
`*_dataset*` are suspects: they often hold one row per participant (or per visit), and a \
"longitudinal" file often holds every patient's rows. But audit and summary tables can carry \
the same words (a per-variable "..._longitudinal.csv" audit is fine), so say it is the name \
that makes you suspicious and that you cannot see the contents, and tell the reader how to \
confirm: run the node (or ask a participant who may run it) and open the named output file \
to see whether it holds one row per participant or visit, or only summaries. Name the \
file(s) and the line(s) that write them.
- Merging is not a leak. Combining cohorts inside the room, and writing diagnostics or aggregate \
information about the merged data (coverage, completeness, mapping and harmonization audits, \
unit-conversion and value-range checks, per-variable or per-cohort summaries, counts), is \
expected and needed: do not flag it. Merged data becomes a problem only when what is written \
out contains the cohorts' individual-level data: a table that is a superset or an extension of \
the cohort rows (every participant's rows still there, possibly with extra harmonized or derived \
columns, or with an original or pooled patient identifier), in any format. Renaming, re-saving, \
converting (CSV to Parquet/JSON) or copying such a file does not change that.
- Do not assume a DataFrame is row-level just because it derives from the input. Decide from \
what it holds: one row per participant or per visit carrying input values (or identifiers) is \
row-level, whatever the table is called; tables with one row per variable, cohort, rule, file \
or issue, or with counts and statistics, are diagnostics or aggregates and are fine. A diagnostic \
that lists individual participants (for example the identifiers or values of the patients having \
an issue) is row-level.
- Who can see an output matters: the DCR section tells you whether anyone is allowed to run the \
node and which nodes read its output. Row-level output of a node that participants can run \
directly, or that feeds an airlock directly, is a leak. Row-level output that only goes to a \
downstream node is a leak if that downstream node (or an airlock on it) passes it on; name the \
downstream node and say that the safety of the whole chain depends on it.
- Printing and logging count too: when the DCR section says logs are shown on success or on \
error, anything the script or an external package logs is visible to the reader.

When you give an overview of a script, always end with a short section titled "Privacy check" that \
(1) lists each output the script produces and says whether it is aggregated or row-level, and \
(2) gives a verdict, one of: "No row-level data leaves this script"; "Possible leak:" followed \
by the line numbers, what exactly would be exported, and why it matters; or "Cannot be verified: \
external code" when an external package may write files you cannot see (say which). Be specific and calibrated: say \
"looks aggregate-only" rather than "guaranteed safe", mention what you could not verify, and \
never cry wolf over aggregates. A list of AUTOMATED HINTS (lines that write, print, call external \
code, access rows, or name suspicious files) is given with the script: use it as a starting point to find the exits, but judge from \
the code, because a hinted line is not necessarily a problem and some exits are not hinted.

When the reader asks about specific lines, answer about those lines in the context of the \
script, and add a privacy remark only if those lines write, print or reshape row-level data. \
If asked something unrelated to this script or to DCRs, say briefly that you can only help \
with this script.
"""

OVERVIEW_TASK = (
    "The reader asked for an overview of the whole script. Structure the answer as: "
    "**What it does** (two or three sentences), **What it reads** (which input nodes and "
    "columns), **Step by step** (the main stages, each with its line range and what happens "
    "to the data), **What it writes** (every output file, its form and content), and finally "
    "**Privacy check** as described in the instructions."
)
FOLLOWUP_TASK = (
    "Answer the reader's latest message about this script. Be focused and shorter than an "
    "overview. If the message carries a line selection, explain exactly those lines first."
)

# ---------------------------------------------------------------------------
# Fetching the node scripts from Decentriq
# ---------------------------------------------------------------------------

_DCR_CACHE_TTL = 3600.0
_DCR_CACHE_MAX = 64
_dcr_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_dcr_cache_lock = threading.Lock()


_CREDENTIAL_RE = re.compile(r"(://)[^/@\s]+@")


def _redact(text: str) -> str:
    """Requirements can embed an access token in a git URL: never pass it on."""
    return _CREDENTIAL_RE.sub(r"\1***@", text)


def _fetch_dcr_nodes(dcr_id: str) -> dict[str, Any]:
    """{'nodes': [{name, type, dependencies, script?}]} of a DCR, from Decentriq."""
    with _dcr_cache_lock:
        hit = _dcr_cache.get(dcr_id)
        if hit and time.time() - hit[0] < _DCR_CACHE_TTL:
            return hit[1]
    import decentriq_platform as dq

    try:
        client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
        dcr = client.retrieve_analytics_dcr(dcr_id=dcr_id)
    except Exception as exc:
        logger.warning("Could not retrieve DCR %s for code explanation: %s", dcr_id, exc)
        raise HTTPException(status_code=502, detail="Could not retrieve the data clean room from Decentriq.")
    nodes: list[dict[str, Any]] = []
    for node_def in getattr(dcr, "node_definitions", []) or []:
        deps = getattr(node_def, "dependencies", None)
        if deps is None and getattr(node_def, "dependency", None):
            deps = [node_def.dependency]
        node: dict[str, Any] = {
            "name": getattr(node_def, "name", None),
            "type": type(node_def).__name__,
            "dependencies": [str(d) for d in (deps or [])],
        }
        script = getattr(node_def, "script", None)
        if script:
            node["script"] = str(script)
        env = getattr(node_def, "custom_environment", None)
        if env:
            node["environment"] = str(env)
        reqs = getattr(node_def, "requirements_txt", None)
        if reqs:
            node["requirements"] = _redact(str(reqs))
        node["logs"] = [flag for flag in ("on_error", "on_success") if getattr(node_def, f"enable_logs_{flag}", False)]
        nodes.append(node)
    result = {"nodes": nodes}
    with _dcr_cache_lock:
        if len(_dcr_cache) >= _DCR_CACHE_MAX:
            oldest = min(_dcr_cache, key=lambda k: _dcr_cache[k][0])
            _dcr_cache.pop(oldest, None)
        _dcr_cache[dcr_id] = (time.time(), result)
    return result


def _user_email(user: Any) -> str:
    return (user.get("email") or "").strip().lower() if isinstance(user, dict) else ""


def _participant_record(dcr_id: str, user: Any) -> dict[str, Any]:
    """The DCR's history record, provided the user takes part in it (or is an admin)."""
    from src.decentriq import get_dcrs_for_participant, get_all_dcrs

    email = _user_email(user)
    if not email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    pool = get_all_dcrs() if email in settings.admins_list else get_dcrs_for_participant(email)
    for record in pool:
        if record.get("id") == dcr_id:
            return record
    raise HTTPException(status_code=404, detail="Data clean room not found among your DCRs")


def _load_node(dcr_id: str, node_name: str, user: Any) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """(history record, the node, all nodes of the DCR); 404 unless the node has a script."""
    record = _participant_record(dcr_id, user)
    nodes = _fetch_dcr_nodes(dcr_id)["nodes"]
    node = next((n for n in nodes if n["name"] == node_name), None)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found in this data clean room")
    if not node.get("script"):
        raise HTTPException(status_code=404, detail="This node has no code to explain")
    return record, node, nodes


# ---------------------------------------------------------------------------
# Static hints: lines that write, print or pick out rows
# ---------------------------------------------------------------------------

_OUTPUT_RE = re.compile(
    r"\.to_(?:csv|json|excel|parquet|pickle|feather|hdf|stata|sql|clipboard)\(|\bjson\.dump\(|\bpickle\.dump\(|"
    r"\.savefig\(|\bshutil\.(?:copy|move)\w*\(|\bopen\([^)\n]*['\"][wax]b?\+?['\"]|\.write\("
)
_PRINT_RE = re.compile(
    r"\bprint\(|\bdisplay\(|\blogging\.\w+\(|\blog\.(?:info|warning|error|debug)\(|\bwlog\(|\.to_string\(|"
    r"\.to_markdown\(|\.to_html\(|sys\.std(?:out|err)"
)
_ROWS_RE = re.compile(r"\.(?:head|tail|sample|nlargest|nsmallest)\(|\.iloc\[|\.iterrows\(|\.itertuples\(|\.to_dict\(")


HINT_TEXT = {
    "output": "writes a file",
    "print": "prints / logs",
    "rows": "picks out rows",
    "external": "calls external code (may write files you cannot see)",
    "name": "names a file that sounds like patient-level data",
}


def static_hints(lines: list[str]) -> list[dict[str, Any]]:
    """Lines that write a file ('output'), print ('print') or pick out rows ('rows')."""
    hints: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        code = line.split("#", 1)[0]
        if not code.strip():
            continue
        kind = "output" if _OUTPUT_RE.search(code) else "print" if _PRINT_RE.search(code) else "rows" if _ROWS_RE.search(code) else None
        if kind:
            hints.append({"line": i, "kind": kind})
    return hints


# ---------------------------------------------------------------------------
# External code and suspicious names (deterministic, so the reader sees them
# whatever the model says)
# ---------------------------------------------------------------------------

# Modules every DCR Python environment ships with; anything else is external.
_BUNDLED_MODULES = {
    "pandas", "numpy", "matplotlib", "scipy", "seaborn", "sklearn", "statsmodels", "pyarrow",
    "openpyxl", "pyreadstat", "decentriq_util", "PIL", "lifelines", "xlrd",
}
_STDLIB = set(getattr(__import__("sys"), "stdlib_module_names", ())) | {"__future__"}
_SENSITIVE_NAME_RE = re.compile(
    r"""['"]([^'"\n/]*(?:pooled|longitudinal|patient[_-]?level|harmoni[sz]ed|provenance|per[_-]?patient|individual|row[_-]?level)"""
    r"""[^'"\n/]*\.(?:csv|parquet|json|xlsx?|feather|pkl|pickle|tsv))['"]""", re.I)
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(?:@|==|>=|<=|~=|<|>|\[|$)")


def external_code_report(node: dict[str, Any], lines: list[str], nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Imports outside the standard environment, the custom environment's
    packages (access tokens redacted), the lines calling external code, and
    string literals naming files that sound like patient-level tables."""
    import ast

    aliases: dict[str, str] = {}  # name used in the script -> top-level module
    imported: dict[str, int] = {}  # top-level module -> first line
    try:
        for n in ast.walk(ast.parse(node["script"])):
            if isinstance(n, ast.Import):
                for a in n.names:
                    top = a.name.split(".")[0]
                    aliases[(a.asname or a.name).split(".")[0]] = top
                    imported.setdefault(top, n.lineno)
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                top = n.module.split(".")[0]
                imported.setdefault(top, n.lineno)
                for a in n.names:
                    aliases[a.asname or a.name] = top
    except SyntaxError:
        for i, l in enumerate(lines, start=1):
            m = re.match(r"\s*(?:import|from)\s+([A-Za-z_]\w*)", l)
            if m:
                imported.setdefault(m.group(1), i)
                aliases.setdefault(m.group(1), m.group(1))
    external = sorted(m for m in imported if m not in _STDLIB and m not in _BUNDLED_MODULES)

    env_name = node.get("environment")
    env_node = next((n for n in nodes if n["name"] == env_name), None) if env_name else None
    requirements = (env_node or {}).get("requirements", "")
    packages = []
    for l in requirements.splitlines():
        l = l.strip()
        if not l or l.startswith("#"):
            continue
        git = "github.com" in l or l.startswith("git+")
        m = _REQ_NAME_RE.match(l)
        packages.append({"name": (m.group(1) if m else l.split("/")[-1].split("@")[0]).removesuffix(".git"), "spec": l, "git": git})

    call_lines = []
    ext_names = {a for a, top in aliases.items() if top in external}
    if ext_names:
        call_re = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(ext_names)) + r")\s*(?:\.\w+)*\(")
        for i, l in enumerate(lines, start=1):
            if call_re.search(l.split("#", 1)[0]):
                call_lines.append(i)
    names = []
    for i, l in enumerate(lines, start=1):
        for m in _SENSITIVE_NAME_RE.finditer(l):
            names.append({"line": i, "name": m.group(1)})
    return {
        "external_modules": [{"module": m, "line": imported[m]} for m in external],
        "environment": env_name,
        "packages": packages,
        "external_call_lines": call_lines[:60],
        "sensitive_names": names[:80],
        "logs_shown": node.get("logs", []),
    }


def external_notice(report: dict[str, Any]) -> str:
    """A fixed warning (Markdown) when the script depends on code that is not visible here."""
    mods = [m["module"] for m in report["external_modules"]]
    if not mods and not report["packages"]:
        return ""
    src = []
    for p in report["packages"]:
        if p["git"]:
            src.append(f"`{p['spec']}`")
    where = (" It is installed from a GitHub repository into a custom environment (" + ", ".join(src) + ")."
             if src else " It is installed into a custom environment" + (f" (`{report['environment']}`)" if report["environment"] else "") + ".")
    names = ", ".join(f"`{m}`" for m in (mods or [p["name"] for p in report["packages"]]))
    return (
        f"⚠️ **External code.** This script imports {names}, which is not part of the standard "
        f"analysis environment.{where} Its source code is **not shown here and was not reviewed**: "
        "it can write files into `/output/` on its own. We recommend that the user test this node on "
        "shuffled or synthetic data and examine the files in the output to determine whether row level "
        "data is being written.\n\n"
    )


def _dcr_access_section(record: dict[str, Any], node: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    """Who may run the node and which nodes consume its output."""
    name = node["name"]
    runners = [p for p in record.get("participants") or [] if name in (p.get("analyst_of") or [])]
    dependents = [(n["name"], n["type"]) for n in nodes if name in n["dependencies"]]
    out = [
        f"Participants allowed to run `{name}` directly: {len(runners)}"
        + (" (nobody: it only runs as a dependency of other nodes)" if not runners else ""),
        "Nodes that read its output: " + (", ".join(
            f"`{n}` ({'airlock - its content is shown to the participants' if t == 'PreviewComputeNodeDefinition' else 'python'})"
            for n, t in dependents) or "none"),
    ]
    logs = node.get("logs") or []
    out.append("Logs shown to the person who runs it: " + (" and ".join(logs).replace("_", " ") if logs else "no"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Building the model's context
# ---------------------------------------------------------------------------


def _fit_script(lines: list[str], focus: tuple[int, int] | None, hint_lines: list[int]) -> tuple[str, bool]:
    """The numbered listing handed to the model. Over-long lines are cut; when the
    script exceeds MAX_SCRIPT_CHARS, the lines near the focus (the reader's
    selection), the start and the hinted lines are kept, the rest elided."""
    numbered = [f"L{i}: {l if len(l) <= MAX_LINE_CHARS else l[:MAX_LINE_CHARS] + ' ...(line cut)'}"
                for i, l in enumerate(lines, start=1)]
    if sum(len(l) + 1 for l in numbered) <= MAX_SCRIPT_CHARS:
        return "\n".join(numbered), False
    keep = [False] * len(lines)
    budget = MAX_SCRIPT_CHARS

    def take(lo: int, hi: int) -> None:
        nonlocal budget
        for idx in range(max(0, lo), min(len(lines), hi)):
            if not keep[idx] and budget > 0:
                keep[idx] = True
                budget -= len(numbered[idx]) + 1

    if focus:
        take(focus[0] - 1 - 80, focus[1] + 80)
    for h in hint_lines:
        take(h - 3, h + 2)
    take(0, len(lines))
    out: list[str] = []
    skipped_from = None
    for idx, kept in enumerate(keep):
        if kept:
            if skipped_from is not None:
                out.append(f"... (lines {skipped_from + 1}-{idx} omitted for length) ...")
                skipped_from = None
            out.append(numbered[idx])
        elif skipped_from is None:
            skipped_from = idx
    if skipped_from is not None:
        out.append(f"... (lines {skipped_from + 1}-{len(lines)} omitted for length) ...")
    return "\n".join(out), True


def _dcr_section(record: dict[str, Any], node: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    out = [f"Title: {record.get('title') or '(untitled)'}"]
    if record.get("cohorts"):
        out.append("Cohorts: " + ", ".join(record["cohorts"]))
    desc = (record.get("description") or "").strip()
    if desc:
        out.append("Description: " + desc[:600])
    out.append("Nodes (type; depends on):")
    for n in nodes[:80]:
        kind = {"PythonComputeNodeDefinition": "python compute", "PreviewComputeNodeDefinition": "airlock",
                "RawDataNodeDefinition": "data (file)", "TableDataNodeDefinition": "data (table)"}.get(n["type"], n["type"])
        deps = ", ".join(n["dependencies"]) or "-"
        mark = "   <== THE NODE BEING EXPLAINED" if n["name"] == node["name"] else ""
        out.append(f"- {n['name']} ({kind}; depends on: {deps}){mark}")
    if len(nodes) > 80:
        out.append(f"- ... {len(nodes) - 80} more nodes")
    return "\n".join(out)


def _external_section(report: dict[str, Any]) -> str:
    if not report["external_modules"] and not report["packages"]:
        return "No imports outside the standard environment (standard library, pandas, numpy, matplotlib, scipy, ...)."
    out = ["The script imports packages outside the standard environment. Their source is NOT visible to you."]
    for m in report["external_modules"]:
        out.append(f"- imports `{m['module']}` (L{m['line']})")
    if report["environment"]:
        out.append(f"Custom environment `{report['environment']}` installs:")
        for p in report["packages"]:
            out.append(f"- {p['spec']}" + ("  <== installed from a GitHub repository" if p["git"] else ""))
    if report["external_call_lines"]:
        out.append("Lines calling external code: " + ", ".join(f"L{n}" for n in report["external_call_lines"][:30]))
    return "\n".join(out)


def _clean_messages(raw: Any) -> list[dict[str, str]]:
    """Roles user/assistant only, last MAX_TURNS, with a line selection carried by a
    user message folded into its text so the history is self-contained."""
    out: list[dict[str, str]] = []
    for m in raw if isinstance(raw, list) else []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("detailed") or m.get("content") or "" if m["role"] == "assistant" else m.get("content") or ""
        content = str(content).strip()[:MAX_MESSAGE_CHARS]
        if not content:
            continue
        lines = m.get("codeLines")
        if m["role"] == "user" and isinstance(lines, list) and len(lines) == 2 and all(isinstance(x, int) for x in lines):
            content = f"[About lines L{lines[0]}-L{lines[1]} of the script]\n{content}"
        out.append({"role": m["role"], "content": content})
    return out[-MAX_TURNS:]


def _selection(raw: Any, n_lines: int) -> tuple[int, int] | None:
    if isinstance(raw, list) and len(raw) == 2 and all(isinstance(x, int) for x in raw):
        a, b = sorted(raw)
        if 1 <= a <= n_lines:
            return a, min(b, n_lines)
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/my-dcrs/{dcr_id}/nodes/{node_name}/script", name="Full script of a compute node, for the code explanation view")
def get_node_script(dcr_id: str, node_name: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    record, node, nodes = _load_node(dcr_id, node_name, user)
    lines = node["script"].split("\n")
    report = external_code_report(node, lines, nodes)
    return {
        "dcr_id": dcr_id,
        "dcr_title": record.get("title") or "",
        "node_name": node["name"],
        "node_type": node["type"],
        "dependencies": node["dependencies"],
        "script": node["script"],
        "line_count": len(lines),
        "hints": static_hints(lines),
        "external": report,
        "external_notice": external_notice(report),
    }


@router.post("/api/chat/code-explain/stream", name="Stream an explanation of a compute node's script")
def code_explain_stream(body: dict[str, Any], user: Any = Depends(get_current_user)) -> StreamingResponse:
    from src.chat import _get_openai_client, stream_completion

    client = _get_openai_client()
    dcr_id = str(body.get("dcr_id") or "")
    node_name = str(body.get("node_name") or "")
    record, node, nodes = _load_node(dcr_id, node_name, user)
    messages = _clean_messages(body.get("messages"))
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="messages must end with a user message")

    lines = node["script"].split("\n")
    focus = _selection(body.get("selected_lines"), len(lines))
    hints = static_hints(lines)
    report = external_code_report(node, lines, nodes)
    notice = external_notice(report)
    for ln in report["external_call_lines"]:
        hints.append({"line": ln, "kind": "external"})
    for sn in report["sensitive_names"]:
        hints.append({"line": sn["line"], "kind": "name"})
    # The most telling hints first, capped so a print-heavy script cannot flood the context.
    rank = {"external": 0, "output": 1, "name": 2, "rows": 3, "print": 4}
    shown = sorted(sorted(hints, key=lambda h: rank[h["kind"]])[:80], key=lambda h: h["line"])
    listing, truncated = _fit_script(lines, focus, [h["line"] for h in shown])
    hint_text = "\n".join(
        f"- L{h['line']}: {HINT_TEXT[h['kind']]}"
        f" - {lines[h['line'] - 1].strip()[:140]}" for h in shown) or "(none found)"

    is_overview = bool(body.get("overview"))
    task = OVERVIEW_TASK if is_overview else FOLLOWUP_TASK
    if notice:
        task += ("\n\nThis script depends on external code (see EXTERNAL CODE). " +
                 ("The fixed warning paragraph about it is already shown to the reader above your answer: do not "
                  "repeat it at length, but build the Privacy check on it. " if is_overview else "") +
                 "Do not give a clean verdict on the visible lines alone.")
    if focus:
        sel = "\n".join(f"L{i}: {lines[i - 1][:MAX_LINE_CHARS]}" for i in range(focus[0], min(focus[1], focus[0] + 60) + 1))
        task += f"\n\nThe reader has selected lines L{focus[0]}-L{focus[1]}:\n{sel}"
    system = "\n\n".join([
        CODE_EXPLAIN_PROMPT,
        "# THIS DATA CLEAN ROOM\n\n" + _dcr_section(record, node, nodes),
        f"# SCRIPT OF THE NODE `{node['name']}` (python compute node; reads {', '.join(node['dependencies']) or 'no other nodes'})"
        + ("\n\nNOTE: the script is long; parts were omitted where marked." if truncated else "")
        + "\n\n" + listing,
        "# EXTERNAL CODE\n\n" + _external_section(report) + "\n\n" + _dcr_access_section(record, node, nodes),
        "# AUTOMATED HINTS (lines that write, print, call external code, pick out rows or name suspicious files; a starting point, not a verdict)\n\n" + hint_text,
        "# YOUR TASK FOR THIS REPLY\n\n" + task,
    ])
    full_messages = [{"role": "system", "content": system}] + messages
    try:
        temperature = float(body.get("temperature", 0.2))
    except (TypeError, ValueError):
        temperature = 0.2
    def _stream() -> Any:
        # The external-code warning is fixed text, not the model's: an overview
        # always starts with it, however the model answers.
        if notice and is_overview:
            yield notice
        yield from stream_completion(client, settings.litellm_model, full_messages, temperature, "code-explain")

    return StreamingResponse(
        _stream(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Chat-Context": json.dumps({"mode": "code-explain", "approx_tokens": len(system) // 4})},
    )
