"""Room-level AI explanation of a Data Clean Room (the "Explain the code" overlay).

Where code_explain.py explains ONE compute node's script in depth, this module
explains the ROOM: what happens in it overall, how data flows from the data
nodes through the compute nodes to what the analysts can see, and whether any
individual-level data reaches an analyst-accessible output. Compute nodes get a
line or two each; the reader asks follow-ups in the chat or opens a node.

Deterministic part (no model): the dependency graph and the EXITS, i.e. every
place where output becomes visible to participants: airlocks (with their
quota) and compute nodes that at least one participant may run directly.
Those facts are given to the model and returned to the UI, which draws the graph.

  GET  /my-dcrs/{dcr_id}/overview               the room structure for the graph
  POST /api/chat/code-explain/room/stream       streamed room explanation / follow-up
"""

import json
import logging
import re
import threading
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
from src.config import settings
from src.code_explain import (
    DCR_BACKGROUND,
    MAX_LINE_CHARS,
    _clean_messages,
    _fetch_dcr_nodes,
    _participant_record,
    external_code_report,
    static_hints,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Per-node budget in the model's context, so a room with 70 nodes still fits.
DIGEST_HEADER_LINES = 6
DIGEST_OUTPUT_LINES = 10
DIGEST_LINE_CHARS = 140

ROOM_PROMPT = """\
You are the assistant that explains a Data Clean Room (DCR) of the iCARE4CVD Cohort Explorer \
to the researchers, data owners and data managers who take part in it. They know their data and \
basic statistics; most do not read Python. They want to understand what happens in the room AS A \
WHOLE, and to be reassured, honestly, about what can and cannot leave it.

""" + DCR_BACKGROUND + """

# WHAT YOU ARE GIVEN

Not the full scripts. For every compute node you get a DIGEST: its dependencies, whether \
participants may run it directly, whether its logs are shown, the packages it needs outside the \
standard environment, the first comment lines of its script, and the lines that write files or \
print. Below that, the list of EXITS: every place where output becomes visible to participants, \
derived from the room's definition (not from you). Judge from these facts; when a verdict \
needs a detail you do not have, say so and tell the reader to open that node for a full \
explanation, where the whole script is available.

# HOW TO EXPLAIN

- Write for a smart reader who does not code. Plain words, short sentences, Markdown with short \
sections and bullets. Use the node and cohort names as they appear. No tables unless asked.
- Explain the FLOW: which data goes in, what each stage does to it in a phrase, and where the \
results come out. Group nodes that play the same role for several cohorts ("for each cohort \
there is a ... node") instead of repeating yourself per cohort.
- Do not dwell on individual nodes: one or two lines each in the summary list, no line numbers. \
The reader can open any node for the step-by-step view.
- Never invent behaviour. A digest shows only the writing and printing lines; the processing \
between them is not visible to you. Say "appears to" where you infer.

# DATA SAFEGUARDS (the section the reader cares most about)

The question is whether INDIVIDUAL-LEVEL data (rows of participants, per-participant values, \
identifiers) reaches any ANALYST-ACCESSIBLE output. Analyst-accessible outputs are exactly the \
EXITS: the contents of airlocks, and the outputs and shown logs of nodes that participants may \
run directly. Output that only feeds another node is not accessible, unless that node or an \
airlock passes it on. Aggregates (counts, percentages, means, distributions, charts of them, \
per-variable or per-cohort summaries, mapping and harmonization audits) are fine and expected: do \
not raise alarms about them. The intended, controlled exits of row-level data are the airlocks: a \
fragment node drops identifiers, keeps a stated percentage of rows and caps extreme values, and \
the airlock shows that fragment up to its quota; the shuffled samples are synthetic rows \
(columns shuffled independently). Say plainly which exits carry aggregates only, which carry \
deliberately limited row-level fragments and under which safeguards, and whether anything else \
could carry individual-level data. External packages whose code is not visible (listed per node) \
can write files the digest does not show: name them and say that their output cannot be verified \
from here. Be calibrated: "looks aggregate-only", "cannot be verified", "would export rows" - \
never "guaranteed safe".
"""

ROOM_OVERVIEW_TASK = (
    "The reader asked for an overview of the whole room. Structure the answer as:\n"
    "**What this room does** (three to five sentences: which cohorts, what the research purpose is "
    "according to the description, what kinds of results analysts can get out).\n"
    "**How the data flows** (a short narrative of the dependency graph: data nodes -> per-cohort "
    "processing -> airlocks / merge chain -> what analysts run; mention that the reader can see the "
    "graph on the left).\n"
    "**Data safeguards** (go through the EXITS one group at a time; for each say what it exposes and "
    "whether it is aggregate-only, a deliberately limited row-level fragment with its safeguards, or "
    "cannot be verified; end with one clear sentence answering whether any individual-level data is "
    "written or exported to an analyst-accessible output, with the caveats that apply).\n"
    "**Compute nodes at a glance** (one or two lines per node or per group of same-role nodes: what it does "
    "and WHAT IT WRITES, naming its output files and saying in a few words what each contains, e.g. "
    "'hist_age.png: a histogram of ages'; mark outputs that participants can see; no line numbers).\n"
    "**Data provisioning** (one or two sentences: which data nodes already hold data and which do not, and "
    "what therefore cannot run yet; skip this section if the status is unknown).\n"
    "Close with one sentence inviting follow-up questions or opening a node for its full script."
)
ROOM_FOLLOWUP_TASK = (
    "Answer the reader's latest message about this room. Be focused and shorter than an overview. "
    "If the answer depends on script details you do not have, say which node to open."
)

_DATA_TYPES = ("RawDataNodeDefinition", "TableDataNodeDefinition")


def _kind(n: dict[str, Any]) -> str:
    """Role of a node for the graph and the digests."""
    t, name = n["type"], n.get("name") or ""
    if t == "PreviewComputeNodeDefinition":
        return "airlock"
    if t == "PythonEnvironmentComputeNodeDefinition":
        return "environment"
    if t == "PythonComputeNodeDefinition":
        return "python"
    if t in _DATA_TYPES:
        if name.endswith("_metadata_dictionary") or name.endswith("-metadata"):
            return "dictionary"
        if name.endswith("_shuffled_sample"):
            return "shuffled"
        if name.endswith("_mapping") or name == "CrossStudyMappings":
            return "mapping"
        return "cohort_data"
    return "other"


def _runners(record: dict[str, Any], name: str) -> list[str]:
    return sorted(p.get("email") or "" for p in record.get("participants") or []
                  if name in (p.get("analyst_of") or []))


# Which data nodes hold a published dataset. Short-lived cache: owners provision
# data while a room is open, and this is one enclave round-trip per lookup.
_PUBLISHED_TTL = 60.0
_published_cache: dict[str, tuple[float, Optional[dict[str, dict[str, Any]]]]] = {}
_published_lock = threading.Lock()


def fetch_published_datasets(dcr_id: str) -> Optional[dict[str, dict[str, Any]]]:
    """{data node name: {user, timestamp}} for the datasets published to the
    room, or None when the platform could not be asked (status unknown)."""
    with _published_lock:
        hit = _published_cache.get(dcr_id)
        if hit and time.time() - hit[0] < _PUBLISHED_TTL:
            return hit[1]
    result: Optional[dict[str, dict[str, Any]]] = None
    try:
        import decentriq_platform as dq
        client = dq.create_client(settings.decentriq_email, settings.decentriq_token)
        dcr = client.retrieve_analytics_dcr(dcr_id=dcr_id)
        response = dcr.session.retrieve_published_datasets(dcr_id)
        result = {}
        for ds in getattr(response, "publishedDatasets", []) or []:
            leaf = str(getattr(ds, "leafId", "") or "")
            name = leaf[:-5] if leaf.endswith("_leaf") else leaf
            result[name] = {"user": getattr(ds, "user", None), "timestamp": int(getattr(ds, "timestamp", 0) or 0)}
    except Exception as exc:
        logger.warning("Published datasets of DCR %s could not be retrieved: %s", dcr_id, exc)
    with _published_lock:
        _published_cache[dcr_id] = (time.time(), result)
    return result


# File names a script writes under /output, from its literals: "/output/x.csv",
# os.path.join(output_dir, "x.csv"), f"{output_dir}/x.csv", OUTPUT_DIR / "x.csv".
_OUT_LITERAL_RE = re.compile(r"/output/([A-Za-z0-9_\-./{}]+)")
_OUT_JOIN_RE = re.compile(
    r"(?:os\.path\.join\(\s*)?(?:output_dir|OUTPUT_DIR|out_dir|OUT_DIR|output_path|OUTPUT_PATH|outdir)\s*"
    r"(?:,|/|\}/)\s*f?['\"]([^'\"\n]+)['\"]")
_OUT_FSTRING_RE = re.compile(r"\{(?:output_dir|OUTPUT_DIR|out_dir|OUT_DIR|output_path|outdir)\}/([A-Za-z0-9_\-./{}]+)")


def output_files(lines: list[str]) -> list[str]:
    """Distinct output file names mentioned by a script, in order of appearance;
    f-string placeholders are kept as '{...}'. Directories end with '/'."""
    seen: list[str] = []
    for raw in lines:
        code = raw.split("#", 1)[0]
        for rx in (_OUT_LITERAL_RE, _OUT_JOIN_RE, _OUT_FSTRING_RE):
            for m in rx.finditer(code):
                name = m.group(1).strip().rstrip("'\")")
                name = re.sub(r"\{[^}]*\}", "{...}", name)
                if name and name not in ("{...}",) and name not in seen and len(name) < 80:
                    seen.append(name)
    return seen[:12]


def room_structure(record: dict[str, Any], nodes: list[dict[str, Any]],
                   published: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, Any]:
    """Nodes, edges and exits of the room, for the UI and for the model."""
    by_name = {n["name"]: n for n in nodes}
    out_nodes = []
    for n in nodes:
        kind = _kind(n)
        item: dict[str, Any] = {
            "name": n["name"], "type": n["type"], "kind": kind,
            "dependencies": [d for d in n["dependencies"] if d in by_name],
            "runners": len(_runners(record, n["name"])),
            "logs": n.get("logs") or [],
        }
        if n.get("environment"):
            item["environment"] = n["environment"]
        if n.get("quota_bytes"):
            item["quota_bytes"] = n["quota_bytes"]
        if kind == "python" and n.get("script"):
            lines = n["script"].split("\n")
            report = external_code_report(n, lines, nodes)
            item["external_packages"] = [p["name"] for p in report["packages"]] or [m["module"] for m in report["external_modules"]]
            hints = static_hints(lines)
            item["writes"] = sum(1 for h in hints if h["kind"] == "output")
            item["prints"] = any(h["kind"] == "print" for h in hints)
            item["outputs"] = output_files(lines)
            # External code given an output folder can write files the script does not name.
            item["external_writes"] = bool(item["external_packages"]) and bool(report["external_call_lines"])
        if kind in ("cohort_data", "dictionary", "shuffled", "mapping", "other") and n["type"] in _DATA_TYPES:
            if published is None:
                item["provisioned"] = None
            else:
                item["provisioned"] = n["name"] in published
                if item["provisioned"]:
                    item["provisioned_by"] = published[n["name"]].get("user")
                    item["provisioned_at"] = published[n["name"]].get("timestamp")
        out_nodes.append(item)
    edges = [{"from": d, "to": n["name"]} for n in out_nodes for d in n["dependencies"]]
    dependents: dict[str, list[str]] = {}
    for e in edges:
        dependents.setdefault(e["from"], []).append(e["to"])

    exits = []
    for n in out_nodes:
        if n["kind"] == "airlock":
            src = n["dependencies"][0] if n["dependencies"] else None
            exits.append({"node": n["name"], "kind": "airlock", "shows_output_of": src,
                          "quota_bytes": n.get("quota_bytes"), "viewers": n["runners"]})
        elif n["kind"] == "python" and n["runners"]:
            exits.append({"node": n["name"], "kind": "runnable", "runners": n["runners"],
                          "logs": n["logs"], "feeds": dependents.get(n["name"], [])})
    data_nodes = [n for n in out_nodes if "provisioned" in n]
    return {
        "dcr_id": record.get("id"),
        "title": record.get("title") or "",
        "description": (record.get("description") or "").strip(),
        "cohorts": record.get("cohorts") or [],
        "participants": len(record.get("participants") or []),
        "nodes": out_nodes,
        "edges": edges,
        "exits": exits,
        "provisioning": {
            "known": published is not None,
            "data_nodes": len(data_nodes),
            "provisioned": sum(1 for n in data_nodes if n.get("provisioned")),
        },
    }


def _digest(n: dict[str, Any], record: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    """A compute node as the model sees it: facts plus the lines that matter."""
    lines = n["script"].split("\n")
    report = external_code_report(n, lines, nodes)
    runners = _runners(record, n["name"])
    head = [f"## `{n['name']}` (python; depends on: {', '.join(n['dependencies']) or '-'})"]
    head.append("Participants may run it directly: " + (f"yes ({len(runners)})" if runners else "no (runs only as a dependency)"))
    if n.get("logs"):
        head.append("Logs shown to the person who runs it: " + ", ".join(n["logs"]).replace("_", " "))
    if report["packages"] or report["external_modules"]:
        pk = [p["spec"] for p in report["packages"]] or [m["module"] for m in report["external_modules"]]
        head.append("EXTERNAL CODE (source not visible): " + "; ".join(pk)[:300])
    # The script's own words about itself: leading comments / docstring lines.
    comments = []
    for raw in lines[:40]:
        t = raw.strip()
        if t.startswith("#") or t.startswith('"""') or t.startswith("'''"):
            t = t.lstrip("#").strip().strip('"').strip("'").strip()
            if t and not t.startswith("!") and not t.lower().startswith("import"):
                comments.append(t[:DIGEST_LINE_CHARS])
        if len(comments) >= DIGEST_HEADER_LINES:
            break
    if comments:
        head.append("Script header: " + " | ".join(comments))
    outs = output_files(lines)
    if outs:
        head.append("Output files it names: " + ", ".join(outs))
    hints = static_hints(lines) + [{"line": ln, "kind": "external"} for ln in report["external_call_lines"]]
    rank = {"external": 0, "output": 1, "print": 2, "rows": 3}
    shown = sorted(sorted(hints, key=lambda h: rank[h["kind"]])[:DIGEST_OUTPUT_LINES], key=lambda h: h["line"])
    if shown:
        head.append("Lines that write, print or call external code:")
        for h in shown:
            head.append(f"  - L{h['line']} ({h['kind']}): {lines[h['line'] - 1].strip()[:DIGEST_LINE_CHARS]}")
    else:
        head.append("No file-writing, printing or external-call lines found by the static scan.")
    if report["sensitive_names"]:
        names = sorted({sn["name"] for sn in report["sensitive_names"]})[:8]
        head.append("File names that sound patient-level (check contents): " + ", ".join(names))
    return "\n".join(head)


def _room_context(structure: dict[str, Any], record: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    kinds = {"cohort_data": "cohort data file", "dictionary": "dictionary", "shuffled": "shuffled sample",
             "mapping": "cross-study mapping", "airlock": "AIRLOCK", "environment": "custom python environment",
             "python": "python compute", "other": "node"}
    out = [f"Title: {structure['title'] or '(untitled)'}"]
    if structure["cohorts"]:
        out.append("Cohorts: " + ", ".join(structure["cohorts"]))
    if structure["description"]:
        out.append("Description: " + structure["description"][:1200])
    out.append(f"Participants: {structure['participants']}")
    out.append("\n# NODES (kind; depends on)")
    prov = structure.get("provisioning") or {}
    if prov.get("known"):
        out.append(f"Data provisioned so far: {prov['provisioned']} of {prov['data_nodes']} data nodes hold a dataset "
                   "(a compute node cannot run until every data node it reads is provisioned).")
    else:
        out.append("Provisioning status of the data nodes could not be retrieved.")
    for n in structure["nodes"]:
        extra = f"; quota {n['quota_bytes'] // 1_000_000} MB" if n.get("quota_bytes") else ""
        if "provisioned" in n and n["provisioned"] is not None:
            extra += "; DATA PROVISIONED" if n["provisioned"] else "; no data yet"
        if n.get("outputs"):
            extra += "; writes: " + ", ".join(n["outputs"][:6]) + (" ..." if len(n["outputs"]) > 6 else "")
        out.append(f"- {n['name']} ({kinds.get(n['kind'], n['kind'])}; depends on: {', '.join(n['dependencies']) or '-'}{extra})")
    out.append("\n# EXITS (every analyst-accessible output, from the room definition)")
    if not structure["exits"]:
        out.append("None: no airlock and no compute node that a participant may run directly.")
    for e in structure["exits"]:
        if e["kind"] == "airlock":
            out.append(f"- AIRLOCK `{e['node']}` shows the output of `{e['shows_output_of']}` to {e['viewers']} participant(s)"
                       + (f", up to {e['quota_bytes'] // 1_000_000} MB" if e.get("quota_bytes") else ""))
        else:
            out.append(f"- `{e['node']}` may be run by {e['runners']} participant(s): its /output files"
                       + (" and its logs (" + ", ".join(e["logs"]).replace("_", " ") + ")" if e.get("logs") else "")
                       + " are visible to them" + (f"; it also feeds {', '.join(e['feeds'])}" if e.get("feeds") else ""))
    out.append("\n# COMPUTE NODE DIGESTS")
    for n in nodes:
        if n["type"] == "PythonComputeNodeDefinition" and n.get("script"):
            out.append(_digest(n, record, nodes))
    return "\n".join(out)


@router.get("/my-dcrs/{dcr_id}/overview", name="Structure of a DCR (nodes, edges, exits) for the room explanation")
def get_room_overview(dcr_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    record = _participant_record(dcr_id, user)
    nodes = _fetch_dcr_nodes(dcr_id)["nodes"]
    return room_structure(record, nodes, fetch_published_datasets(dcr_id))


@router.post("/api/chat/code-explain/room/stream", name="Stream an explanation of the whole DCR")
def room_explain_stream(body: dict[str, Any], user: Any = Depends(get_current_user)) -> StreamingResponse:
    from src.chat import _get_openai_client, stream_completion

    client = _get_openai_client()
    dcr_id = str(body.get("dcr_id") or "")
    record = _participant_record(dcr_id, user)
    nodes = _fetch_dcr_nodes(dcr_id)["nodes"]
    messages = _clean_messages(body.get("messages"))
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="messages must end with a user message")
    structure = room_structure(record, nodes, fetch_published_datasets(dcr_id))
    is_overview = bool(body.get("overview"))
    system = "\n\n".join([
        ROOM_PROMPT,
        "# THIS DATA CLEAN ROOM\n\n" + _room_context(structure, record, nodes),
        "# YOUR TASK FOR THIS REPLY\n\n" + (ROOM_OVERVIEW_TASK if is_overview else ROOM_FOLLOWUP_TASK),
    ])
    full_messages = [{"role": "system", "content": system}] + messages
    try:
        temperature = float(body.get("temperature", 0.2))
    except (TypeError, ValueError):
        temperature = 0.2
    return StreamingResponse(
        stream_completion(client, settings.litellm_model, full_messages, temperature, "code-explain-room"),
        media_type="text/plain; charset=utf-8",
        headers={"X-Chat-Context": json.dumps({"mode": "code-explain-room", "approx_tokens": len(system) // 4,
                                              "nodes": len(nodes), "exits": len(structure["exits"])})},
    )
