"""graph_similarity.py — Component-wise embedding context scoring.

Cascade: Structural (Step 1) > Extensional (Step 2) > Category Guard (Step 2.5) > Linguistic (Step 3a/3b/3c)
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import settings
from .data_model import MappingType, ContextMatchType
from .omop_graph_nx import OmopGraphNX
from .utils import parse_post_cordinating_concepts_ids, parse_post_cordinating_concepts_labels, build_concept_parts

_GENERIC = {"yes","no","true","false","present","absent","positive","negative",
            "normal","abnormal","unknown","none","na","n/a"}

# =====================================================================
# Parsing
# =====================================================================


# =====================================================================
# Step 1: Bijective graph matching
# =====================================================================

_EMBED_CACHE = {}
# def clear_embed_cache():
#     """Clear the embedding cache (e.g. when the embedding model changes)."""
#     _EMBED_CACHE.clear()
def _cached_embed(texts, embed_model):
    prefix = embed_model.model_name
    uncached = [(i, t) for i, t in enumerate(texts)
                if f"{prefix}::{t}" not in _EMBED_CACHE]
    if uncached:
        idxs, raw = zip(*uncached)
        vecs = embed_model.embed_batch(list(raw))
        for i, vec in zip(idxs, vecs):
            _EMBED_CACHE[f"{prefix}::{texts[i]}"] = vec
    return np.vstack([_EMBED_CACHE[f"{prefix}::{t}"] for t in texts])

# def _cached_embed(texts, embed_model):
#     """Batch-embed with per-text caching. Returns (n, dim) array."""
#     uncached = [(i, t) for i, t in enumerate(texts) if t not in _EMBED_CACHE]
#     if uncached:
#         idxs, raw = zip(*uncached)
#         vecs = embed_model.embed_batch(list(raw))
#         for i, vec in zip(idxs, vecs):
#             _EMBED_CACHE[texts[i]] = vec
#     return np.vstack([_EMBED_CACHE[t] for t in texts])

def check_concept_equivalence(codes_a, codes_b, graph, max_depth=1):
    if not codes_a and not codes_b: return ContextMatchType.EXACT.value
    if len(codes_a) != len(codes_b): return ContextMatchType.NOT_APPLICABLE.value
    if sorted(codes_a) == sorted(codes_b): return ContextMatchType.EXACT.value
    sa, sb = set(codes_a), set(codes_b)
    

    fwd = all(s in sb or graph.source_to_targets_paths(s, list(sb), max_depth=max_depth) for s in codes_a)
    rev = all(s in sa or graph.source_to_targets_paths(s, list(sa), max_depth=max_depth) for s in codes_b)
    if fwd and rev:
        # Check if main concepts are siblings (share parent but distinct)
        return ContextMatchType.EXACT.value
    return ContextMatchType.NOT_APPLICABLE.value

# =====================================================================
# Step 2: Value overlap
# =====================================================================

def check_value_overlap(codes_a: List[int], values_b: List[int],
                        codes_b: List[int], values_a: List[int],
                        graph: OmopGraphNX = None, max_depth: int = 1) -> bool:
    if codes_a and values_b and set(codes_a) & set(values_b): return True
    if codes_b and values_a and set(codes_b) & set(values_a): return True
    return False

# =====================================================================
# Step 2.5: Category structure guard
# =====================================================================

def is_trivial_value_set(labels: frozenset) -> bool:
    """Trivial if all labels are generic qualifiers (yes/no/present/absent/...)."""
    if not labels: return True
    if len(labels) == 2:
        from .fuzz_match import FuzzyMatcher
        pair = list(labels)
        is_neg, _, _ = FuzzyMatcher._is_negation_pair(pair[0], pair[1])
        if is_neg: return True
    return all(l in _GENERIC for l in labels)


def _category_guard(src_val_labels: List[str], tgt_val_labels: List[str]) -> bool:
    """Return True if category structure asymmetry blocks the match.

    Fires when concept sizes differ AND:
      a) One side has categories, other doesn't  (one-sided)
      b) One side trivial (Yes/No), other non-trivial (dose adjustments)
    """
    src_has = bool(src_val_labels)
    tgt_has = bool(tgt_val_labels)
    # a) One-sided: one has categories, other doesn't
    if src_has != tgt_has:
        return True
    # b) Trivial vs non-trivial asymmetry
    if src_has and tgt_has:
        src_triv = is_trivial_value_set(frozenset(l.strip().lower() for l in src_val_labels))
        tgt_triv = is_trivial_value_set(frozenset(l.strip().lower() for l in tgt_val_labels))
        if src_triv != tgt_triv:
            return True
    return False

# =====================================================================
# Step 3: Component-wise embedding
# =====================================================================


def _sanitize_labels(labels: List[str]) -> List[str]:
    """Strip empty/NaN/None values that break API embedding calls."""
    if not labels:
        return []
    out = []
    for l in labels:
        if l is None or (isinstance(l, float) and np.isnan(l)):
            continue
        s = str(l).strip()
        if s and s.lower() not in ("nan", "none", "null"):
            out.append(s)
    return out

# def check_embedding_match(
#     src_labels, tgt_labels, src_value_labels, tgt_value_labels,
#     embed_model, threshold=0.8, value_threshold=0.7,
# ):
#     if embed_model is None: return False, 0.0, None
#     sl = _sanitize_labels(src_labels)
#     tl = _sanitize_labels(tgt_labels)
#     if not sl or not tl: 
#         # print(f"no labels for {sl} and {tl}")
#         return False, 0.0, None
#     svl = [l.strip().lower() for l in (src_value_labels or []) if l.strip()]
#     tvl = [l.strip().lower() for l in (tgt_value_labels or []) if l.strip()]

#     if not svl or not tvl:
#         return False, 0.0, None

#     # 3a: Component match WITH unmatched validation
#     ok, score, reason = _component_match(sl, tl, embed_model, threshold, svl, tvl)
#     # print(f"component match for {sl} and {tl} and {svl} and {tvl} is {ok} and {score} and {reason}")
#     if ok: 
#         return True, score, f"3a: {reason}"
#     if _value_sets_agree(src_value_labels, tgt_value_labels):
#         ok, score, reason = _component_match(sl, tl, embed_model, value_threshold)
#         # print(f"value agreement for {sl} and {tl} is {ok} and {score} and {reason}")
#         if ok: 
#             return True, score, f"3b: {reason}"

#     # 3c: Concept ↔ value cross-match (unchanged)
#     # for concepts, values, tag in [(sl, tvl, "src→tgt"), (tl, svl, "tgt→src")]:
#     for concepts, values, tag in [(sl, tvl, "src→tgt"), (tl, svl, "tgt→src")]:
#         if not concepts or not values:
#             continue
#         if is_trivial_value_set(frozenset(values)):
#             continue
#         all_t = concepts + values
#         embs = _cached_embed(all_t, embed_model)
#         sims = cosine_similarity(embs[:len(concepts)], embs[len(concepts):])
#         mx = float(np.clip(sims.max(), 0.0, 1.0))
#         if mx >= threshold:
#             return True, mx, f"3c: {tag}"
#     return False, 0.0, None



# def _component_match(labels_a, labels_b, embed_model, threshold,
#                      val_labels_a=None, val_labels_b=None):
#     """Greedy bipartite match. When sizes differ, validates unmatched labels
#     against other side's value labels (if provided)."""
#     # print(f"labels_a: {labels_a}")
#     # print(f"labels_b: {labels_b}")
#     all_embs = _cached_embed(labels_a + labels_b, embed_model)   
#     # all_embs = embed_model.embed_batch(labels_a + labels_b)
#     sim = cosine_similarity(all_embs[:len(labels_a)], all_embs[len(labels_a):])

#     # Orient so rows = smaller side
#     if len(labels_a) <= len(labels_b):
#         smaller, larger, mat = labels_a, labels_b, sim
#         other_vals = val_labels_a   # smaller side's values
#     else:
#         smaller, larger, mat = labels_b, labels_a, sim.T
#         other_vals = val_labels_b   # smaller side's values

#     used, pairs = set(), []
#     for i in range(len(smaller)):
#         row = mat[i].copy()
#         row[list(used)] = -1.0
#         j = int(row.argmax())
#         pairs.append((i, j, float(row[j])))
#         used.add(j)

#     if len(pairs) < len(smaller): return False, 0.0, None
#     ms = min(s for _, _, s in pairs)
#     ms = float(np.clip(ms, 0.0, 1.0))
#     if ms < threshold: return False, 0.0, None



#     if len(smaller) < len(larger) and other_vals is not None:
#         clean_vals = [v.strip().lower() for v in other_vals if v.strip()]
#         if clean_vals:
#             unmatched = [larger[j] for j in range(len(larger)) if j not in used]
#             if unmatched:
#                 all_texts = unmatched + clean_vals
#                 embs = _cached_embed(all_texts, embed_model)
#                 um_embs = embs[:len(unmatched)]
#                 val_embs = embs[len(unmatched):]
#                 sims = cosine_similarity(um_embs, val_embs)
#                 if sims.max(axis=1).min() < threshold:
#                     return False, 0.0, None

#     detail = ", ".join(f"'{smaller[i]}'↔'{larger[j]}'={s:.2f}" for i, j, s in pairs)
#     return True, ms, f"min={ms:.2f}: {detail}"
    
# def _value_sets_agree(src_vals: List[str], tgt_vals: List[str]) -> bool:
#     """Non-trivial identical/high-overlap value sets confirm equivalence."""
#     if not src_vals or not tgt_vals: return False
#     a = frozenset(l.strip().lower() for l in src_vals)
#     b = frozenset(l.strip().lower() for l in tgt_vals)
#     if is_trivial_value_set(a) or is_trivial_value_set(b): return False
#     if a == b: return True
#     if a.issubset(b) or b.issubset(a): return True
#     u = a | b
#     return bool(u and len(a & b) / len(u) >= 0.8)


# def check_embedding_match(src_labels, tgt_labels, src_value_labels=None, tgt_value_labels=None,
#                           embed_model=None, threshold=0.8, value_threshold=0.7):
#     if embed_model is None: return False, 0.0, None
#     sl, tl = _sanitize_labels(src_labels), _sanitize_labels(tgt_labels)
#     if not sl or not tl: return False, 0.0, None
#     embs = _cached_embed([", ".join(sl), ", ".join(tl)], embed_model)
#     score = float(np.clip(cosine_similarity(embs[0:1], embs[1:2])[0, 0], 0.0, 1.0))
#     return score >= threshold, score, f"sim:{score:.3f}"

def check_embedding_match(src_labels, tgt_labels, src_value_labels=None, tgt_value_labels=None,
                          embed_model=None, threshold=0.8, value_threshold=0.7):
    if embed_model is None: return False, 0.0, None
    sl, tl = _sanitize_labels(src_labels), _sanitize_labels(tgt_labels)
    if not sl or not tl: return False, 0.0, None
    embs = _cached_embed([" ".join(sl), " ".join(tl)], embed_model)
    score = float(np.clip(cosine_similarity(embs[0:1], embs[1:2])[0, 0], 0.0, 1.0))
    return score >= threshold, score, f"sim:{score:.3f}"

def score_context(
    src_codes: List[int], tgt_codes: List[int],
    src_values: List[int], tgt_values: List[int],
    src_labels: List[str], tgt_labels: List[str],
    src_val_labels: List[str], tgt_val_labels: List[str],
    graph: OmopGraphNX = None,
    embed_model=None,
    mapping_mode: str = MappingType.OEH.value,
    max_depth: int = 1,
    threshold: float = settings.ADAPTIVE_THRESHOLD,
) -> Tuple[str, float]:
    """
            Mode semantics:
        OO       : graph only
        OEC/OEH  : graph first; embedding fallback iff graph is inconclusive
        NE       : not normally used here, because NE mode has no additional context
    """
    # print(f"src concept(s) label = {src_labels}, tgt concept(s) label = {tgt_labels}")
    # ── Step 1 & 2: Graph-based ──────────────────────────
    if graph is not None and mapping_mode != MappingType.NE.value:
        equiv = check_concept_equivalence(src_codes, tgt_codes, graph, max_depth)
        if equiv == ContextMatchType.EXACT.value:
            return ContextMatchType.EXACT.value, 1.0

        if check_value_overlap(src_codes, tgt_values, tgt_codes, src_values):
            return ContextMatchType.SUBSUMED.value, 1.0

        if (len(src_codes) <= 2 and len(tgt_codes) <= 2):  # one might have an extra concept -- still acceptable as partial? 
                return ContextMatchType.PENDING.value, 0.65 
        # ── Step 2.5: Category structure guard ────────────────
        if len(src_codes) != len(tgt_codes) and _category_guard(src_val_labels, tgt_val_labels):
            atomic_vs_post = (
                bool(src_codes) and bool(tgt_codes)
                and (len(src_codes) == 1 or len(tgt_codes) == 1)
                and (set(src_codes).issubset(set(tgt_codes)) or set(tgt_codes).issubset(set(src_codes)))
            )
            if atomic_vs_post:
                return ContextMatchType.SUBSUMED.value, 1.0
         
            return ContextMatchType.NOT_APPLICABLE.value, 0.0

    # ── Step 3: Component-wise embedding ─────────────────
    if embed_model is not None and mapping_mode != MappingType.OO.value:
        ok, score, _ = check_embedding_match(
            src_labels, tgt_labels, src_val_labels, tgt_val_labels,
            embed_model, threshold=threshold)
        if ok:
            return ContextMatchType.COMPATIBLE.value, score

    return ContextMatchType.NOT_APPLICABLE.value, 0.0


# =====================================================================
# DataFrame Adapter
# =====================================================================

def _parse_codes_with_main(row, codes_col: str, main_col: str) -> List[int]:
    codes = parse_post_cordinating_concepts_ids(row.get(codes_col))
    main = row.get(main_col)
    if main is not None and not pd.isna(main):
        m = int(main)
        if m not in codes: 
            codes.insert(0, m)
    return codes


def compute_context_scores(
    df: pd.DataFrame, graph: OmopGraphNX, embed_model=None,
    mapping_mode: str = MappingType.OEH.value,
    max_depth: int = None, threshold: float = settings.ADAPTIVE_THRESHOLD,
    llm:bool=False
) -> pd.DataFrame:
    
    """
    Compute additional-context consistency scores for symbolic/ontology-based mappings.

    Mode behavior:
      NE  : skip context scoring because neural-only candidates do not contain
            decomposed/post-coordinated additional context.
      OO  : graph-only context scoring.
      OEC/OEH : graph-based context scoring with optional embedding fallback.
    """
    if mapping_mode == MappingType.NE.value:  # no concepts mapping/context available in NE model. relies entirely on variable label which is already computed in neuro_matcher 
        return df
   
    if max_depth is None: 
        max_depth = settings.DEFAULT_GRAPH_DEPTH
    # use_emb = embed_model is not None and mapping_mode != MappingType.OO.value
    is_symbolic = df["mapping_relation"].str.lower().str.contains("symbolic", na=False)
    embed_model = embed_model if mapping_mode != MappingType.OO.value else None
    for idx in df[is_symbolic].index:
        row = df.loc[idx]
        ctx_type, score = score_context(
            src_codes=_parse_codes_with_main(row, "source_composite_code_omop_ids", "somop_id"),
            tgt_codes=_parse_codes_with_main(row, "target_composite_code_omop_ids", "tomop_id"),
            src_values=parse_post_cordinating_concepts_ids(row.get("source_categories_omop_ids")),
            tgt_values=parse_post_cordinating_concepts_ids(row.get("target_categories_omop_ids")),
            src_labels=build_concept_parts(row.get("slabel"), row.get("source_composite_code_labels")),
            tgt_labels=build_concept_parts(row.get("tlabel"), row.get("target_composite_code_labels")),
            src_val_labels=parse_post_cordinating_concepts_labels(row.get("source_categories_labels")),
            tgt_val_labels=parse_post_cordinating_concepts_labels(row.get("target_categories_labels")),
            graph=graph, embed_model=embed_model,
            mapping_mode=mapping_mode, max_depth=max_depth, threshold=threshold,
        )
        df.at[idx, "context_match_type"], df.at[idx, "sim_score"] = ctx_type, score
        
        # 2. NEW: Overwrite the mapping_relation based on context reality
        if ctx_type == ContextMatchType.SUBSUMED.value:
            df.at[idx, "mapping_relation"] = "Symbolic:closeMatch"
        elif ctx_type == ContextMatchType.COMPATIBLE.value:
            df.at[idx, "mapping_relation"] = "Symbolic:closeMatch"
        elif ctx_type == ContextMatchType.NOT_APPLICABLE.value:
            # Optional: Demote to Unmatched so it gets filtered out later
            df.at[idx, "mapping_relation"] = "Unmatched"

    # ── Neural / unresolved rows ─────────────────────────
    if not embed_model: 
        return df
   
    if not llm:
        pending = df["context_match_type"] == ContextMatchType.PENDING.value
        s = df["sim_score"].fillna(0.0)

        df.loc[pending & (s >= 0.85),                "context_match_type"] = ContextMatchType.EXACT.value
        df.loc[pending & (s >= 0.75) & (s < 0.85),   "context_match_type"] = ContextMatchType.COMPATIBLE.value
        df.loc[pending & (s >= 0.65) & (s < 0.75),   "context_match_type"] = ContextMatchType.PARTIAL.value
        df.loc[pending & (s < 0.65),                 "context_match_type"] = ContextMatchType.NOT_APPLICABLE.value



    return df
