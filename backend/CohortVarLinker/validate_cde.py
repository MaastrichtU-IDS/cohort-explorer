#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install pandas requests

import sys
import pandas as pd
import requests
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import re
from pathlib import Path
from requests.exceptions import RequestException  # add this import
from CohortVarLinker.src.omop_graph import OmopGraphNX
import time
from CohortVarLinker.src.config import settings
from CohortVarLinker.src.utils import load_dictionary
"""_summary_

CDE dictionary validator (format, structure, semantics)

Checks implemented:

1.  Check all required columns exist (values can be empty, but columns must be present).
2.  Normalize column names: 'variablename' → 'variable name', 'variablelabel' → 'variable label'.
3.  Check 'categorical' column format based on 'vartype' (int/float/str/datetime).
4. Check the values in OMOP ID column must be Integar (or float if .0 gets added). For all Cocnept Code must have prefix of one of the allowed vocabularies. For all concept name must be non-empty strings. 
5. Check the allowed values in vartype column (int, float, str, datetime). 
6.  Additional context concepts are only allowed if the variable concept exists.
7.  Unit concepts are only allowed if 'units' is non-empty
8.  Visit concepts are only allowed if 'visits' is non-empty 
9.  The number of additional context names, codes, and OMOP IDs (joined with '|') must match.
10.  Variable concept code and OMOP ID must each have at most one value (no '|').
11.  Same single-concept rule as 10 for unit and visit concept code and OMOP ID.
12. For each categorical value, one concept must be present in
    - Categorical Value Concept Name
    - Categorical Value Concept Code
    - Categorical Value OMOP ID
13. For each concept triple the allowed vocabulary prefix and  validate correctness of triple (name, code, OMOP ID), via OHDSI Athena API. 
    -  Check all concept code columns have a vocabulary prefix (e.g., 'atc:AC902', 'snomed:39020002').
    -  One code and omop id for each concept name
 """

ATHENA_ENDPOINT = "https://athena.ohdsi.org/api/v1/concepts"
TIMEOUT = 25
COLUMN_CHECK_LIST  = [
        "variable name","variable label","domain", "categorical", "units","visits", "vartype", "missing", "count","na","min","max","formula",
        "categorical value concept name","categorical value concept code","categorical value omop id",
        "variable concept name","variable concept code","variable omop id",
        "additional context concept name","additional context concept code","additional context omop id",
        "unit concept name","unit concept code","unit omop id",
        "visit concept name","visit concept code","visit omop id",
    ]

CACHE_ATHENA: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = {}
VOCAB_CHECK_LIST = ["SNOMEDCT", "SNOMED-CT", "SNOMED", "RXNORM", "RXNORMEXTENSION", "RXNORM_EXTENSION", "OMOP", "OMOPGENOMIC", "UKBIOBANK", "ATC", "CDISC", "UCUM", "ICARE", "LOINC", "MESH"]
_INT   = re.compile(r"^\s*-?\d+\s*$")
_FLOAT = re.compile(r"^\s*-?(?:\d+(?:\.\d+)?|\.\d+)\s*$")
_PAIR  = re.compile(r"\s*=\s*")

_QUOTES = "'\u2018\u2019\"\u201C\u201D"

def _looks_datetime(s: str) -> bool:
    try:
        return pd.to_datetime(_strip(s), errors="raise") is not None
    except Exception:
        return False
def _looks_str(s: str) -> bool:
    ss = _strip(s)
    return not (_INT.match(ss) or _FLOAT.match(ss))

def _strip(s: str) -> str:
    return s.strip(_QUOTES + " ").strip()

def _as_int(s: str):
    s = _strip(s)
    return int(s) if _INT.match(s) else None

def _as_float(s: str):
    s = _strip(s)
    return float(s) if _FLOAT.match(s) else None

def _as_datetime(s: str):
    s = _strip(s)
    try:
        return pd.to_datetime(s, errors="raise")
    except Exception:
        return None
    
    
_OMOP_GRAPH: Optional[OmopGraphNX] = None

def get_omop_graph() -> OmopGraphNX:
    global _OMOP_GRAPH
    if _OMOP_GRAPH is None:
        _OMOP_GRAPH = OmopGraphNX(settings.concepts_file_path)   # builds/loads the 300MB graph once
    return _OMOP_GRAPH
    
def load_json(filepath: str) -> dict:
    """Loads a JSON file and returns its content as a dictionary."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

# def load_data( filepath=None) -> pd.DataFrame:
#         """Loads the input dataset."""
#         if filepath.endswith('.sav'):
#             df_input = pd.read_spss(filepath)
#             # Optionally save to Excel if needed
         
#         elif filepath.endswith('.csv'):
#             df_input = pd.read_csv(filepath, low_memory=False)
#         elif filepath.endswith('.xlsx'):
#             df_input = pd.read_excel(filepath, sheet_name=0)
#         else:
#             raise ValueError("Unsupported file format.")
#         if not df_input.empty:
#             df_input.columns = [col.lower().strip() for col in df_input.columns]
#             return df_input
#         else:
#             return None
        
QUOTE_CHARS = r"'\u2018\u2019\"\u201C\u201D"  # ' ’ “ ” "
def strip_end_quotes(s):
    if not isinstance(s, str):
        return s
    s = s.strip()
    # remove any run of quotes at start or end
    return re.sub(fr"^[{QUOTE_CHARS}]+|[{QUOTE_CHARS}]+$", "", s).strip()


def normalize_value(value): 
    """Normalize a value for comparison."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return value
def is_missing(x):
    return pd.isna(x) or (isinstance(x, str) and x.strip() == "")

# def safe_get(meta_df, var_name, col_name):
#             subset = meta_df.loc[meta_df["variable name"].str.lower() == var_name.lower(), col_name]
#             if subset.empty:
#                 return None
#             val = subset.values[0]
#             return None if pd.isna(val) else str(val).strip()

def make_standardized_csv_path(p: str, tag: str = "_standardized") -> str:
    """
    Append `_standardized` to the filename (once) and force .csv extension.
    Examples:
      foo.sav   -> foo_standardized.csv
      foo.xlsx  -> foo_standardized.csv
      foo.csv   -> foo_standardized.csv
      foo_standardized.csv -> foo_standardized.csv
    """
    path = Path(p)
    stem = path.stem
    # avoid double-appending (case-insensitive)
    if stem.lower().endswith(tag.lower()):
        new_name = f"{stem}.csv"
    else:
        new_name = f"{stem}{tag}.csv"
    return str(path.with_name(new_name))


def report_misordered_categories(categories: str, data_type: str) -> list[str]:
    """
    Returns a list of tokens (e.g., '1=Yes') that look 'mis-ordered'
    relative to the expected data_type:
      - str:  prefer 'text=number'  -> flag 'number=text'
      - int:  prefer 'number=text'  -> flag 'text=number'
      - float:prefer 'number=text'  -> flag 'text=number'
      - datetime: prefer 'datetime=text' -> flag 'text=datetime'
    If there's no '=', nothing to report.
    """
    if not isinstance(categories, str) or not categories.strip():
        return []

    dt = (data_type or "str").lower()
    issues = []

    for raw in categories.split("|"):
        item = raw.strip()
        if not item:
            continue
        parts = _PAIR.split(item, maxsplit=1)
        if len(parts) != 2:
            # single token (no '='), nothing to check
            continue

        left, right = parts[0], parts[1]

        if dt == "str":
            # want left=string (preferred). If left is numeric and right is string, flag.
            if not _looks_str(left) and _looks_str(right):
                issues.append(item)

        elif dt == "int":
            # want left=int. If left is string and right is int, flag.
            if _looks_str(left) and _as_int(right):
                issues.append(item)
            # also accept float that is numeric-looking for the right side
            elif _looks_str(left) and _as_float(right):
                issues.append(item)

        elif dt == "float":
            # want left=float (or int). If left is string and right is float/int, flag.
            if _looks_str(left) and (_as_float(right) or _as_int(right)):
                issues.append(item)

        elif dt == "datetime":
            # want left=datetime. If left is non-datetime and right is datetime, flag.
            if (not _looks_datetime(left)) and _looks_datetime(right):
                issues.append(item)

        # else: unknown dt -> do nothing

    # if issues:
    #     print(f"⚠️ Mis-ordered categories for type '{dt}': {issues}")
    return issues



def parse_category_keys(categories, data_type: str):
    """
    '1=Yes|0=No' with data_type='int'  -> [1, 0]
    '1=Yes|0=No' with data_type='str'  -> ['yes','no']
    'Red|Green|Blue' (str)             -> ['red','green','blue']
    """
    if not isinstance(categories, str):
        items = categories or []
    else:
        items = [t for t in categories.split("|") if t.strip()]

    out = []
    dt = (data_type or "str").lower()

    for item in items:
        tok = item.strip()
        parts = _PAIR.split(tok, maxsplit=1)

        # choose candidate(s) to test for the requested type
        cand = parts if len(parts) == 2 else [tok]

        chosen = None
        if dt == "int":
            for c in cand:
                v = _as_int(c)
                if v is not None: chosen = v; break
        elif dt == "float":
            for c in cand:
                v = _as_float(c)
                if v is not None: chosen = v; break
            if chosen is None:
                # ints are valid floats too
                for c in cand:
                    v = _as_int(c)
                    if v is not None: chosen = float(v); break
        elif dt == "datetime":
            for c in cand:
                v = _as_datetime(c)
                if v is not None: chosen = v; break
        else:  # dt == 'str' (or anything else)
            for c in cand:
                cs = _strip(c).lower()
                # prefer non-numeric-looking strings
                if not _INT.match(cs) and not _FLOAT.match(cs):
                    chosen = cs; break
            if chosen is None:
                chosen = _strip(cand[0]).lower()

        # final fallback if still None
        if chosen is None:
            chosen = _strip(cand[0]).lower()

        out.append(chosen)
    # print(f"parse_category_keys({categories}, {data_type}) -> {out}")
    return out


# ---------------------- helpers ----------------------

def split_code_prefixed(val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """'loinc:8302-2' -> ('LOINC','8302-2')"""
    if not val or ":" not in str(val):
        return (None, None)
    pre, post = str(val).split(":", 1)
    # print(f"split_code_prefixed: pre='{pre}', post='{post}'")
    vocab = pre.strip().upper()
    if vocab not in VOCAB_CHECK_LIST:
        return  None, post.strip()
    else:
        if vocab in {"SNOMEDCT", "SNOMED-CT", 'SNOMED'}:
            vocab = ["SNOMED", "SNOMED Veterinary"]
        elif vocab == 'OMOPGENOMIC':
            vocab = ['OMOP Genomic']
        elif vocab == "OMOP":
            vocab = ["OMOP Extension"]
        elif vocab == "RXNORM":
            vocab = ["RxNorm"]
        elif vocab == 'UKBIOBANK':
            vocab = ['UK Biobank']

        elif vocab in  ('RXNORMEXTENSION', 'RXNORM_EXTENSION'):
            vocab = ['RxNorm Extension']
        else:
            vocab = [vocab]
        return vocab, post.strip()

def to_int_or_none(x: Optional[str]) -> Optional[int]:
    
    if x is None or str(x).strip() == "" or str(x).strip().lower() == "nan":
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def athena_headers() -> Dict[str, str]:
    # Athena-Auth-Token = eyJhbGciOiJIUzI1NiJ9.eyIkaW50X3Blcm1zIjpbXSwic3ViIjoib3JnLnBhYzRqLnNhbWwucHJvZmlsZS5TQU1MMlByb2ZpbGUja29tYWwuZ2lsYW5pQG1hYXN0cmljaHR1bml2ZXJzaXR5Lm5sIiwibGFzdE5hbWUiOlsiR2lsYW5pIl0sImlzRnJvbU5ld0xvZ2luIjpbInRydWUiXSwiYXV0aGVudGljYXRpb25EYXRlIjpbIjIwMjUtMTEtMjVUMTU6MjE6NTQuODE3WltVVENdIl0sInN1Y2Nlc3NmdWxBdXRoZW50aWNhdGlvbkhhbmRsZXJzIjpbIkFyYWNobmUiXSwibm90QmVmb3JlIjp7fSwic2FtbEF1dGhlbnRpY2F0aW9uU3RhdGVtZW50QXV0aE1ldGhvZCI6WyJ1cm46b2FzaXM6bmFtZXM6dGM6U0FNTDoxLjA6YW06cGFzc3dvcmQiXSwiZmlyc3ROYW1lIjpbIktvbWFsIl0sImF1dGhlbnRpY2F0aW9uTWV0aG9kIjpbIkFyYWNobmUiXSwib3JnYW5pemF0aW9uIjpbIk1hYXN0cmljaHQgVW5pdmVyc2l0eSJdLCJub3RPbk9yQWZ0ZXIiOnt9LCIkaW50X3JvbGVzIjpbXSwibG9uZ1Rlcm1BdXRoZW50aWNhdGlvblJlcXVlc3RUb2tlblVzZWQiOlsiZmFsc2UiXSwiaWF0IjoxNzY0MDg0MTE3LCJzZXNzaW9uaW5kZXgiOiJfMjYyMTAyMTIzMTI2NjMxMzU0OCIsImVtYWlsIjoia29tYWwuZ2lsYW5pQG1hYXN0cmljaHR1bml2ZXJzaXR5Lm5sIn0.dA3ee3mxnJqP4bAVxOkpN_oIqhSqAf4hU2q_x2RjJNM
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://athena.ohdsi.org/"    
    }

def fetch_athena(label: str, vocabularies: List[str], include_classification: bool = True) -> List[Dict[str, Any]]:
    """
    Query Athena for a given code/label within one or more vocabularies.
    Athena allows repeated query params, so we build a list of (key, value) pairs.
    """
    # normalize key so that ["SNOMED", "LOINC"] and ["LOINC", "SNOMED"] are treated the same
    norm_label = (label or "").strip()
    cache_key = (norm_label, tuple(sorted(vocabularies)), include_classification)

    # 1) Check cache first
    if cache_key in CACHE_ATHENA:
        print(f"Using cached Athena results for label='{label}', vocabularies={vocabularies}")
        return CACHE_ATHENA[cache_key]

    # 2) Build params
    params: List[Tuple[str, Any]] = [
        ("query", norm_label),
        ("invalidReason", "Valid"),
        ("page", 1),
        ("pageSize", 5),
    ]

    for v in vocabularies:
        params.append(("vocabulary", v))

    # if you really want to include standard + classification concepts
    # if include_classification:
    #     params.append(("standardConcept", "Standard"))
    #     params.append(("standardConcept", "Classification"))

    try:
        r = requests.get(
            ATHENA_ENDPOINT,
            params=params,
            headers=athena_headers(),
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "content" in data:
            result: List[Dict[str, Any]] = data["content"]
        elif isinstance(data, list):
            result = data
        else:
            result = []

        # 3) Store in cache (including empty result, so we don’t re-query)
        CACHE_ATHENA[cache_key] = result
        return result

    except Exception as e:
        print(f"⚠️ Error querying Athena: {e}")
        return []



def rows_to_simple(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            out.append({
                "concept_id": int(r.get("id") or r.get("conceptId")),
                "name": r.get("name") or r.get("conceptName") or "",
                "code": r.get("code") or r.get("conceptCode") or "",
                "vocabulary": r.get("vocabulary") or r.get("vocabularyId") or "",
                "domain": r.get("domain") or r.get("domainId") or "",
                "standard": r.get("standardConcept"),
            })
         
        except Exception:
            continue
    return out

@dataclass
class ValidationLog:
    status: str                 # PASS | FAIL | N/A
    description: str
    vocabulary: Optional[str] = None
    label: Optional[str] = None
    assigned_concept_id: Optional[int] = None
    assigned_concept_code: Optional[str] = None
    matched_concept_id: Optional[int] = None
    matched_concept_code: Optional[str] = None
    matched_name: Optional[str] = None
    matched_domain: Optional[str] = None
    matched_standard: Optional[str] = None



def validate_component(label: str, code_prefixed: Optional[str], omop_id_str: Optional[str], omop_graph_obj:Any, context:str=None) -> ValidationLog:
    lab = " ".join((label or "").strip().split())
    vocab, code = split_code_prefixed(code_prefixed)
    cid = to_int_or_none(omop_id_str)
    if vocab and "ICARE" in (vocab or []):
        return ValidationLog(status="PASS", description="its custom vocabulary.")
    if not vocab and not code and cid is None:
        return ValidationLog(status="N/A", description="No code/ID provided for this component.")
    if not lab:
        return ValidationLog(status="FAIL", description="Missing label; cannot query Athena.", vocabulary=vocab)
    if not vocab:
        return ValidationLog(status="FAIL", description="Vocabulary prefix missing/unknown in code.", label=lab)
    if omop_graph_obj:
        flag, status = omop_graph_obj.concept_exists(cid, str(code), vocab) # it returns tuple (bool, str)
       
        if flag and status == "correct":
            return ValidationLog(status="PASS", description="concept_id and concept_code are the same.", vocabulary=vocab[0], label=lab, assigned_concept_id=cid, assigned_concept_code=code, matched_concept_id=cid, matched_concept_code=code)
        elif not flag and status == "incorrect":
            
            return ValidationLog(status="FAIL", description="concept_id and concept_code do not match.", vocabulary=vocab[0], label=lab, assigned_concept_id=cid, assigned_concept_code=code, matched_concept_id=None, matched_concept_code=None)
        
        else:
    
            try:
                    print(f"validate_component: label='{lab}', code='{code}', cid={cid} -> flag={flag}, status={status}, context={context}")
                    rows = fetch_athena(code, vocab, include_classification=True)
                    concepts = rows_to_simple(rows)
                    if not concepts:
                        return ValidationLog(
                            status="N/A",
                            description="Athena returned no candidates.",
                            vocabulary=vocab,
                            label=lab,
                        )
                    

                    id_hits = {c["concept_id"]: c for c in concepts}
                    code_hits = {c["code"]: c for c in concepts}

                    if cid is not None and code is not None:
                        c1, c2 = id_hits.get(cid), code_hits.get(code)
                        if c1 and c2 and c1["concept_id"] == c2["concept_id"]:
                            c = c1
                            return ValidationLog("PASS", "concept_id and concept_code are the same.", vocab, lab, cid, code,
                                                c["concept_id"], c["code"], c["name"], c["domain"], c["standard"])
                        if c1 and not c2:
                            return ValidationLog("FAIL", "concept_id matched but concept_code not found for this label/vocabulary.",
                                                vocab, lab, cid, code, c1["concept_id"], None, c1["name"], c1["domain"], c1["standard"])
                        if c2 and not c1:
                            return ValidationLog("FAIL", "concept_code matched but concept_id not found for this label/vocabulary.",
                                                vocab, lab, cid, code, None, c2["code"], c2["name"], c2["domain"], c2["standard"])
                        return ValidationLog("FAIL", "Neither concept_id nor concept_code matched Athena candidates.", vocab, lab, cid, code)

                    else:
                        return ValidationLog("N/A", "No code/id provided.")
            except Exception as e:
                    # <--  this is where your NameResolutionError ends up
                    print(f"⚠️ Error: {e}")
                    return ValidationLog(
                        status="N/A",
                        description=f"Athena API not reachable: {e}",
                        vocabulary=vocab,
                        label=lab,
                    )

def overall_status(*components: ValidationLog) -> str:
    present = [c.status for c in components if c.status != "N/A"]
    if not present:
        return "N/A"
    return "FAIL" if any(s == "FAIL" for s in present) else "PASS"

# Optional mapper: if your CSV puts OMOP **table** names in domain, normalize to OMOP **Domain** strings
# DOMAIN_MAP = {
#     "measurement": "Measurement",
#     "condition_occurrence": "Condition",
#     "observation": "Observation",
#     "procedure_occurrence": "Procedure",
#     "person": "Person",
#     "observation_period": "Observation Period",
#     # add more if needed
# }

# ---------------------- main ----------------------


def is_missing(x):
    return pd.isna(x) or (isinstance(x, str) and x.strip() == "")   

def structure_sanity_check(df: pd.DataFrame, required_columns: List[str], out_csv: str) -> List[str]:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        #raise ValueError(f"Missing required columns: {missing}")
        # return on row stating that columns are missing
        print(f"⚠️ Missing required columns: {missing}")
        out_rows = []
        v_log = ValidationLog(status="FAIL", description=f"Missing required columns: {missing}")
        out_rows.append({**asdict(v_log)})
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(out_csv, index=False)
        return False
    # check if vartype has allowed values
    # allowed_vartypes = {"int", "float", "str", "datetime"}
    # # invalid_vartypes = df[~df['vartype'].isin(allowed_vartypes)]

    # # check if all concept name columns have  strings or empty
    # concept_name_columns = [
    #     "variable concept name", 'additional context concept name', 'unit concept name', 'visit concept name']
    # for col in concept_name_columns:
    #     invalid_names = df[~df[col].isna() & (df[col].apply(lambda x: not isinstance(x, str)))]
    #     if not invalid_names.empty:
    #         print(f"⚠️ Invalid entries found in column '{col}': Non-string values present.")
    #         out_rows = []
    #         for _, r in invalid_names.iterrows():
    #             v_log = ValidationLog(status="FAIL", description=f"Invalid entry in column '{col}': Non-string value '{r[col]}' present.")
    #             out_rows.append({**asdict(v_log)})
    #         out_df = pd.DataFrame(out_rows)
    #         out_df.to_csv(out_csv, index=False)
    #         return False
    # for 
    return True

def validate_variable(r: pd.Series, omop_graph_obj: OmopGraphNX) -> ValidationLog:
    
    varlabel = r.get("variable label", "")
    # normalize domain (optional, safe to pass raw)
    
    v_log = ValidationLog(status="N/A", description="Variable description not available.")
    # variable
    if varlabel != "":
        if is_missing(r.get("variable concept name",None)) or is_missing(r.get("variable concept code",None)) or is_missing(r.get("variable omop id",None)):
            v_log = ValidationLog(status="N/A", description="Variable concept details missing/incomplete for non-empty variable label.")
            if not is_missing(r.get("additional context concept name",None)) or  not is_missing(r.get("additional context concept code",None)) or not is_missing(r.get("additional context omop id",None)):
                v_log = ValidationLog(status="FAIL", description="Additional context concept details exist but variable concept details are missing. Additional context depends on variable concept.")
        else:
            if '|' in str(r.get("variable concept code",None)) or '|' in str(r.get("variable omop id",None)):
                v_log = ValidationLog(status="FAIL", description="Multiple variable concept codes/ids found for single variable label.")
            else:
                
                # print(f"variable validation='{varlabel}'")
                v_log = validate_component(
                label=r.get("variable concept name"),
                code_prefixed=r.get("variable concept code"),
                omop_id_str=r.get("variable omop id"),
                omop_graph_obj=omop_graph_obj,
                context="variable",
                # domain_hint=domain,,
            )
                
    else:
        if not is_missing(r.get("variable concept name",None)) or not is_missing(r.get("variable concept code",None)) or not is_missing(r.get("variable omop id",None)):
            v_log = ValidationLog(status="FAIL", description="Variable label is empty but variable concept details exist.")
    return v_log

def validate_additional_context(r: pd.Series, omop_graph_obj: OmopGraphNX) -> ValidationLog:
    # normalize domain (optional, safe to pass raw)
    
    ac_logs = ValidationLog(status="N/A", description="Additional Context not available.")

    if not is_missing(r.get("additional context concept name",None)) and  not is_missing(r.get("additional context concept code",None)) and not is_missing(r.get("additional context omop id",None)):
            if is_missing(r.get("variable concept name",None)) or is_missing(r.get("variable concept code",None)) or is_missing(r.get("variable omop id",None)):
                ac_logs = ValidationLog(status="FAIL", description="Additional context concept details exist but variable concept details are missing/incomplete. Additional context depends on complete variable concept.")
            else:
                list_names = str(r.get("additional context concept name")).split("|")
                list_codes = str(r.get("additional context concept code")).split("|")
                list_ids = str(r.get("additional context omop id")).split("|")
                if not len(list_names) == len(list_codes) == len(list_ids):
                    ac_logs = ValidationLog(status="FAIL", description=f"Mismatch in number of additional context concept details: names({len(list_names)}), codes({len(list_codes)}), ids({len(list_ids)}).")
                elif not isinstance(r.get("additional context concept name"), str) or not isinstance(r.get("additional context concept code"), str) or not isinstance(r.get("additional context omop id"), str):
                    ac_logs = ValidationLog(status="FAIL", description="Additional context concept details must be strings.")
                else:
                    ac_log_list = []    
                    for idx in range(len(list_names)):
                        label, code_prefixed, omop_id_str = (
                                list_names[idx] if idx < len(list_names) else "",
                                list_codes[idx] if idx < len(list_codes) else "",
                                list_ids[idx] if idx < len(list_ids) else "",
                                )
                        if '|' in str(code_prefixed) or '|' in str(omop_id_str):
                            ac_log = ValidationLog(status="FAIL", description="Multiple additional context concept codes/ids found for single additional context.")
                            
                        else:
                            
                            print(f"Validating additional context={label} for variable='{r.get('variable label','')}'")
                            ac_log = validate_component(
                            label=label,
                            code_prefixed=code_prefixed,    
                            omop_id_str=omop_id_str,
                            omop_graph_obj=omop_graph_obj,
                            context="additional context",
                            )
                        ac_log_list.append(ac_log)
                    ac_logs_str = "; ".join([f"{ac_log.label}= {ac_log.status}: {ac_log.description} " for ac_log in ac_log_list])
                    ac_logs = ValidationLog(status="PASS" if all(ac_log.status in {"PASS", "N/A"} for ac_log in ac_log_list) else "FAIL", description=ac_logs_str)
                    # print(f"Validated additional context for variable '{r.get('variable name','')}' - Status: {ac_logs_str}")
                    
    # elif  is_missing(r.get("additional context concept name",None)) or is_missing(r.get("additional context concept code",None)) or  is_missing(r.get("additional context omop id",None)):
    #         ac_logs = ValidationLog(status="FAIL", description="Additional context concept details incomplete.")    
    return ac_logs
    
    
def validate_categorical_values(r: pd.Series, omop_graph_obj: OmopGraphNX) -> ValidationLog:
    cv_logs = ValidationLog(status="N/A", description="Categorical value not available.")
    if not is_missing(r.get("categorical")):  
            categories  = r.get("categorical")
            if is_missing(r.get("categorical value concept name",None)) or is_missing(r.get("categorical value concept code",None)) or is_missing(r.get("categorical value omop id",None)):
                cv_logs = ValidationLog(status="N/A", description="Categorical value concept details missing/incomplete for categorical variable.")
            else:
                # split the categories and check each one and its corresponding code and id existance
                categories_issues = report_misordered_categories(categories, data_type=r.get('vartype','str'))
                categories_list = categories.split("|")
                cat_name_list = str(r.get("categorical value concept name")).split("|")
                cat_code_list = str(r.get("categorical value concept code")).split("|")
                cat_id_list = str(r.get("categorical value omop id")).split("|")
                # report check for mismatch between number of categories and number of concept details
                if not len(categories_list) == len(cat_name_list):
                    cv_logs = ValidationLog(status="FAIL", description=f"Mismatch in number {len(categories_list)} of categorical values and their {len(cat_name_list)} concept details.")
                    if len(categories_issues) > 0:
                        cv_logs = ValidationLog(status="FAIL", description=f"Invalid categorical format (original data value=interpreted data value). Expected: <value of vartype>=<label> (e.g., int: 1=Yes|0=No, str: Yes=positive|No=Negative). Additionally Mismatch in number {len(categories_list)} of categorical values and their {len(cat_name_list)} concept details.")
                # report check for mismatch between number of names, codes and ids
                elif not len(cat_name_list) == len(cat_code_list) == len(cat_id_list):
                    cv_logs = ValidationLog(status="FAIL", description=f"Mismatch in number of categorical value concept details: names({len(cat_name_list)}), codes({len(cat_code_list)}), ids({len(cat_id_list)}).")
                    if len(categories_issues) > 0:
                        cv_logs = ValidationLog(status="FAIL", description=f"Invalid categorical format. Expected: <value of vartype>=<label> (e.g., int: 1=Yes|0=No, str: Yes=positive|No=Negative). Additionally Mismatch in number of categorical value concept details: names({len(cat_name_list)}), codes({len(cat_code_list)}), ids({len(cat_id_list)})")
                # validate each category concept for each categorical value
                else:
                    cv_log_list = []    
                    for idx in range(len(cat_name_list)):
                        
                        label, code_prefixed, omop_id_str = (
                            cat_name_list[idx] if idx < len(cat_name_list) else "",
                            cat_code_list[idx] if idx < len(cat_code_list) else "",
                            cat_id_list[idx] if idx < len(cat_id_list) else "",
                        )
                        # print(f"label={label}, code_prefixed={code_prefixed}, omop_id_str={omop_id_str}")
                        if '|' in str(code_prefixed) or '|' in str(omop_id_str):
                            cv_log = ValidationLog(status="FAIL", description="Multiple categorical value concept codes/ids found for single categorical value.")
                        else:
                           
                            cv_log = validate_component(
                            label=label,
                            code_prefixed=code_prefixed,
                            omop_id_str=omop_id_str,
                            omop_graph_obj=omop_graph_obj,
                            context="categorical value",
                            )
                           
                            # domain_hint=domain,
                        
                        cv_log_list.append(cv_log)
                    cv_logs_str = "; ".join([f"{cv_log.label}= {cv_log.status}: {cv_log.description} " for cv_log in cv_log_list])
                    cv_logs = ValidationLog(status="PASS" if all(cv_log.status in {"PASS", "N/A"} for cv_log in cv_log_list) else "FAIL", description=cv_logs_str)
                    
                    if len(categories_issues) > 0:
                        cv_logs = ValidationLog(status="FAIL", description=f"Invalid categorical format. Expected: <value of vartype>=<label> (e.g., int: 1=Yes|0=No, str: Yes=positive|No=Negative). Additionally {cv_logs_str}")
                    # print(f"Validated categorical values for variable '{r.get('variablename','')}' - Status: {cv_logs_str}")
    else:
        if not is_missing(r.get("categorical value concept name",None)) or not is_missing(r.get("categorical value concept code",None)) or not is_missing(r.get("categorical value omop id",None)):
            cv_logs = ValidationLog(status="FAIL", description="Categorical value concept details exist but categorical variable is empty.")
        
    return cv_logs
   

def validate_unit(r: pd.Series, omop_graph_obj: OmopGraphNX) -> ValidationLog:
    u_log = ValidationLog(status="N/A", description="Unit not available.")
    if not is_missing(r.get("units")):  
            if is_missing(r.get("unit concept name",None)) or is_missing(r.get("unit concept code",None)) or is_missing(r.get("unit omop id",None)):
                u_log = ValidationLog(status="N/A", description="Unit concept details missing/incomplete for non-empty units.")
            else:
                if '|' in str(r.get("unit concept code",None)) or '|' in str(r.get("unit omop id",None)):
                    u_log = ValidationLog(status="FAIL", description="Multiple unit concept codes/ids found for single unit.")
                else:
                    u_log = validate_component(
                    label=r.get("unit concept name") or r.get("units"),
                    code_prefixed=r.get("unit concept code"),
                    omop_id_str=r.get("unit omop id"),
                    omop_graph_obj=omop_graph_obj,
                    context="unit",
                    # domain_hint=domain,
                )
                # print(f"Validated unit for variable '{r.get('variable name','')}' - Status: {u_log.status}: {u_log.description}")
    else:
        if not is_missing(r.get("unit concept name",None)) or not is_missing(r.get("unit concept code",None)) or not is_missing(r.get("unit omop id",None)):
            u_log = ValidationLog(status="FAIL", description="Units is empty but unit concept details exist.")
    return u_log


def validate_var_type(r: pd.Series, omop_graph_obj: OmopGraphNX) -> ValidationLog:
    vt_log = ValidationLog(status="N/A", description="Variable type not available.")
    if not is_missing(r.get("vartype")):  
            var_type = r.get("vartype").strip().lower()
            allowed_vartypes = {"int", "float", "str", "datetime"}
            if var_type not in allowed_vartypes:
                vt_log = ValidationLog(status="FAIL", description=f"Invalid vartype '{var_type}'. Allowed vartypes are: {allowed_vartypes}.")
            else:
                vt_log = ValidationLog(status="PASS", description=f"Valid vartype '{var_type}'.")
    else:
        vt_log = ValidationLog(status="N/A", description="Vartype is missing.")
    return vt_log


def validate_timepoint(r: pd.Series, omop_graph_obj: OmopGraphNX) -> ValidationLog:  
    vi_log = ValidationLog(status="N/A", description="Visit not available.")
    if not is_missing(r.get("visits")):  
            if is_missing(r.get("visit concept name",None)) or is_missing(r.get("visit concept code",None)) or is_missing(r.get("visit omop id",None)):
                vi_log = ValidationLog(status="N/A", description="Visit concept details missing/incomplete for non-empty visits.")
            else:
                if '|' in str(r.get("visit concept code",None)) or '|' in str(r.get("visit omop id",None)):
                    vi_log = ValidationLog(status="FAIL", description="Multiple visit concept codes/ids found for single visit.")
                elif not isinstance(r.get("visit concept code",None), str) or not isinstance(r.get("visit omop id",None), int|float) or not isinstance(r.get("visit concept name",None), str):
                    vi_log = ValidationLog(status="FAIL", description="Invalid data type for visit concept columns details.")
                else:
                    vi_log = validate_component(
                    label=r.get("visit concept name") or r.get("visits"),
                    code_prefixed=r.get("visit concept code"),
                    omop_id_str=r.get("visit omop id"),
                    omop_graph_obj=omop_graph_obj,
                    context="visit",
                    # domain_hint=domain,
                )
                # print(f"Validated visit for variable '{r.get('variable name','')}' - Status: {vi_log.status}: {vi_log.description}")
    else:
        if not is_missing(r.get("visit concept name",None)) or not is_missing(r.get("visit concept code",None)) or not is_missing(r.get("visit omop id",None)):
            vi_log = ValidationLog(status="FAIL", description="Visits is empty but visit concept details exist.")
    return vi_log



def validate_dictionary(in_csv: str, out_csv: str=None) -> None:
    start_time = time.time()
    df = load_dictionary(in_csv)
    df = df.applymap(lambda x: None if pd.isna(x) or (isinstance(x, str) and x.strip() == "") else x)
    omop_graph_obj = get_omop_graph()  
    # print(df.head(2))
    # lower-case all column names
    df.columns = [c.lower() for c in df.columns]
    # convert all values to lower-case for required columns
    # expected columns (lowercase)
    
    if "variablelabel" in df.columns:
        df = df.rename(columns={"variablelabel": "variable label"})
    if "variablename" in df.columns:
        df = df.rename(columns={"variablename": "variable name"})
    if not structure_sanity_check(df, required_columns=COLUMN_CHECK_LIST, out_csv=out_csv):
        return
    out_rows = []
    for _, r in df.iterrows():
            # normalize variablelabel to variable label in row
            varlabel = r.get("variable label", "")

            raw_domain = r.get("domain", "").strip()
            vt_log = validate_var_type(r, omop_graph_obj)
            v_log = validate_variable(r, omop_graph_obj)
            ac_logs = validate_additional_context(r, omop_graph_obj)
            cv_logs = validate_categorical_values(r, omop_graph_obj)        
            u_log = validate_unit(r, omop_graph_obj)
            vi_log = validate_timepoint(r, omop_graph_obj)
            out_rows.append({
                "variable name": r.get("variable name",""),
                "variable label": varlabel,
         
                "domain": raw_domain,
                "vartype_status": vt_log.status,
                "vartype_reason": vt_log.description,
                
                "categorical_value_status": cv_logs.status,
                "categorical_value_reason": cv_logs.description,
                "variable_status": v_log.status, 
                "variable_reason": v_log.description,
                "additional_context_status": ac_logs,
                "unit_status": u_log.status, "unit_reason": u_log.description,
                "visit_status": vi_log.status, "visit_reason": vi_log.description,
                "overall_status": overall_status(cv_logs, v_log, ac_logs, u_log, vi_log),
                # 'validation_status': 'PASS' if complete_passing else 'CHECK AGAIN'
            })
    out_df = pd.DataFrame(out_rows)
    if out_csv:
        out_df.to_csv(out_csv, index=False)
    # print(f"Saved results to: {out_csv}")
    print(f"Validation completed in {time.time() - start_time:.2f} seconds")
    if out_df.empty:
        return True
    else:
        return False
   

if __name__ == "__main__":
    if len(sys.argv) != 3:
        # print("Usage: python validate cde_against athena system, params: input.csv output.csv")
        sys.exit(1)
    validate_dictionary(sys.argv[1], sys.argv[2])
    # # run the athena api for simple example

    # vocab, code = split_code_prefixed("loinc:LP6800-9")
    # print(fetch_athena(code, vocab))
