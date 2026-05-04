import networkx as nx
import pandas as pd
import pickle, os, gzip, zlib, time
from typing import List, Tuple
from collections import deque, OrderedDict
from llm.data_model import MappingRelation

LOINC_REQUIRED_AXES = ['component', 'specimen', 'time_aspect']
LOINC_IGNORABLE_AXES = ['system', 'property', 'method', 'scale_type']
VOCAB_ALIASES = {
    "snomed": "snomed", "snomedct": "snomed",
    "snomed_veterinary": "snomed veterinary", "snomedct_veterinary": "snomed veterinary",
    "loinc": "loinc", "rxnorm": "rxnorm",
    "rxnorm extension": "rxnorm extension", "rxnorm_extension": "rxnorm extension",
    "omop extension": "omop extension", "omop_extension": "omop extension",
    "omop genomic": "omop genomic", "omop_genomic": "omop genomic",
    "atc": "atc", "icd10": "icd10", "icd10cm": "icd10cm", "icd9": "icd9cm",
    "ucum": "ucum", "cpt4": "cpt4", "ndfrt": "ndfrt", "mesh": "mesh",
    "ukbiobank": "uk biobank", "cdisc": "cdisc",
    "cancer modifier": "cancer modifier", "cancer_modifier": "cancer modifier",
    "icd9proc": "icd9proc", "visit": "visit",
    "visit type": "visit type", "visit_type": "visit type",
    "icare4cvd": "icare4cvd", "condition status": "condition status",
}
EQ_RELS = {
    "rxnorm - atc pr lat": "atc - rxnorm pr lat", "atc - rxnorm pr lat": "rxnorm - atc pr lat",
    "atc - rxnorm": "rxnorm - atc", "rxnorm - atc": "atc - rxnorm",
    "snomed - rxnorm eq": "rxnorm - snomed eq", "rxnorm - snomed eq": "snomed - rxnorm eq",
    "atc - snomed eq": "snomed - atc eq", "snomed - atc eq": "atc - snomed eq",
    "mapped from": "maps to", "maps to": "mapped from",
    "cpt4 - snomed eq": "snomed - cpt4 eq", "snomed - cpt4 eq": "cpt4 - snomed eq",
    "cpt4 - loinc eq": "loinc - cpt4 eq", "loinc - cpt4 eq": "cpt4 - loinc eq",
}
DIR_RELS = {"is a": "subsumes", "subsumes": "is a", "has answer": "answer of", "answer of": "has answer"}
LOINC_AXIS_RELS = {
    'has component': 'component', 'has property': 'property', 'has time aspect': 'time_aspect',
    'has system': 'system', 'has scale type': 'scale_type', 'has method': 'method', 'has specimen': 'specimen',
}
EQUIV_REL_NAMES = frozenset({
    "maps to", "mapped from", "rxnorm - atc pr lat", "atc - rxnorm pr lat",
    "atc - rxnorm", "rxnorm - atc", "snomed - rxnorm eq", "rxnorm - snomed eq",
    "atc - snomed eq", "snomed - atc eq", "cpt4 - snomed eq", "snomed - cpt4 eq",
    "cpt4 - loinc eq", "loinc - cpt4 eq",
})

# ── Multi-target mapper detection ──
# OMOP's ``maps_to`` is a MAPPING relationship, not a semantic equivalence.
# For single-ingredient concepts it behaves like 1:1 equivalence, but
# combination/multi-ingredient products (e.g. "furosemide + spironolactone")
# ``maps_to`` EACH ingredient separately.  Naively chaining through them
# merges unrelated drug classes.
#
# We detect these at index-build time: any concept with ≥2 outgoing
# ``maps_to`` edges is a "multi-target mapper" (combination product).
# During equivalence closure BFS the node is included but equivalence is
# NOT propagated through it, breaking the false bridge.


# class BlockingFilter:
#     __slots__ = ('_check_fn', 'blocked', 'passed', 'equiv_class', 'source')

#     def __init__(self, check_fn, source=None, equiv_class=None):
#         self._check_fn, self.source = check_fn, source
#         self.blocked, self.passed, self.equiv_class = {}, set(), equiv_class or set()

#     def __call__(self, tid: int) -> bool:
#         result = self._check_fn(tid)
#         if result:
#             self.blocked[tid] = "hierarchically related"
#         else:
#             self.passed.add(tid)
#         return result

#     def summary(self, graph=None):
#         lines = [
#             f"Source:{self.source}" + (f" ({graph.get_node_attr(self.source, 'name')})" if graph else ""),
#             f"Blocked:{len(self.blocked)}, Passed:{len(self.passed)}",
#         ]
#         for tid, reason in self.blocked.items():
#             name = graph.get_node_attr(tid, 'name') if graph else str(tid)
#             lines.append(f"  ✗ {tid} ({name}) — {reason}")
#         return "\n".join(lines)


class OmopGraphNX:
    def __init__(self, csv_file_path=None, output_file='graph_nx.pkl.gz'):
        self.csv_file_path, self.output_file = csv_file_path, output_file
        self.graph = nx.DiGraph()

        # ── Caches ──
        self._sssp_cache, self._sssp_cache_max = OrderedDict(), 100_000
        self._sibling_cache = {}
        self._ancestor_cache = {}
        self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None

        # ── Pre-indexed typed adjacency for O(1) typed-neighbor lookups ──
        self._isa_succ = {}       # node → frozenset of "is a" successors
        self._subs_succ = {}      # node → frozenset of "subsumes" successors
        self._equiv_bidir = {}    # node → frozenset of equiv neighbors (both dirs)
        self._multi_target_mappers = frozenset()  # nodes with ≥2 outgoing maps_to

        # ── Equiv closure cache ──
        self._equiv_cache = {}
        self._equiv_cache_max = 50_000

        # ── Per-source gate pre-computation cache ──
        self._gate_precomp_cache = OrderedDict()
        self._gate_precomp_cache_max = 30_000

        if os.path.exists(output_file):
            self.load_graph(output_file)
        elif csv_file_path:
            self.build_graph(csv_file_path)

    # ══════════════════════════════════════════════════════════════════
    # Typed adjacency index — built once and reusable 
    # ══════════════════════════════════════════════════════════════════

    def _build_typed_adjacency(self):
        """Pre-index edges by relationship type for O(1) typed-neighbor lookups.

        Also detects multi-target mappers (combination products) — concepts
        with ≥2 outgoing ``maps_to`` edges.
        """
        IS_A, SUBS, EQUIV = self._rel_ints()
        g = self.graph

        # Resolve the "maps to" int for multi-target detection
        rm_fwd = {r: i for i, r in g.graph.get('rel_map_rev', {}).items()}
        print(f"forward rel = {rm_fwd}")
        MAPTO_INT = rm_fwd.get("maps to", -1)
        print(f"mapto edges: {MAPTO_INT}")
        # isa_succ, subs_succ, equiv_bidir = {}, {}, {}
        # maps_to_count = {}  # node → count of outgoing maps_to edges

        # t0 = time.time()
        # for u, v, data in g.edges(data=True):
 
        #     rels = data.get('all_r') or frozenset({data.get('r', 0)})
        #     if IS_A in rels:
        #         isa_succ.setdefault(u, set()).add(v)
        #     if SUBS in rels:
        #         subs_succ.setdefault(u, set()).add(v)
        #     if rels & EQUIV:
        #         equiv_bidir.setdefault(u, set()).add(v)
        #         equiv_bidir.setdefault(v, set()).add(u)
        #     if MAPTO_INT in rels:
        #         maps_to_count[u] = maps_to_count.get(u, 0) + 1

        # self._isa_succ = {k: frozenset(v) for k, v in isa_succ.items()}
        # self._subs_succ = {k: frozenset(v) for k, v in subs_succ.items()}
        # self._equiv_bidir = {k: frozenset(v) for k, v in equiv_bidir.items()}
        # self._multi_target_mappers = frozenset(
        #     u for u, c in maps_to_count.items() if c > 1)

        isa_succ, subs_succ, equiv_bidir = {}, {}, {}
        equiv_targets = {}  # Track fan-out for ALL equivalence types

        t0 = time.time()
        for u, v, data in g.edges(data=True):
            rels = data.get('all_r') or frozenset({data.get('r', 0)})
            
            if IS_A in rels:
                isa_succ.setdefault(u, set()).add(v)
            if SUBS in rels:
                subs_succ.setdefault(u, set()).add(v)
            
            # If ANY equivalence relationship exists between u and v
            if rels & EQUIV:
                equiv_bidir.setdefault(u, set()).add(v)
                equiv_bidir.setdefault(v, set()).add(u)
                # Track the outgoing equivalence to detect hubs
                equiv_targets.setdefault(u, set()).add(v)

        self._isa_succ = {k: frozenset(v) for k, v in isa_succ.items()}
        self._subs_succ = {k: frozenset(v) for k, v in subs_succ.items()}
        self._equiv_bidir = {k: frozenset(v) for k, v in equiv_bidir.items()}
        
        # A multi-target mapper is any concept with equivalence edges to >1 distinct concept
        self._multi_target_mappers = frozenset(
            u for u, targets in equiv_targets.items() if len(targets) > 1
        )

        elapsed = time.time() - t0
        print(f"[INFO] Built typed adjacency index in {elapsed:.2f}s "
              f"(is_a:{sum(len(v) for v in self._isa_succ.values()):,}, "
              f"subsumes:{sum(len(v) for v in self._subs_succ.values()):,}, "
              f"equiv:{sum(len(v) for v in self._equiv_bidir.values()):,}, "
              f"multi_target_mappers:{len(self._multi_target_mappers):,})")

    # ══════════════════════════════════════════════════════════════════
    # Shared helpers
    # ══════════════════════════════════════════════════════════════════

    def get_edge_rel(self, u, v):
        """Return the primary relationship name for edge (u, v), or ''."""
        if not self.graph.has_edge(u, v):
            return ""
        return self.graph.graph.get('rel_map_rev', {}).get(
            self.graph.get_edge_data(u, v).get('r', 0), "")

    def get_edge_rels(self, u, v) -> List[str]:
        """Return ALL relationship names stored on edge (u, v)."""
        if not self.graph.has_edge(u, v):
            return []
        data = self.graph.get_edge_data(u, v)
        rev = self.graph.graph.get('rel_map_rev', {})
        rels = data.get('all_r') or frozenset({data.get('r', 0)})
        return [rev[r] for r in rels if r in rev]

    def _rel_ints(self):
        if self._IS_A is not None:
            return self._IS_A, self._SUBSUMES, self._EQUIV_INTS
        m = {r: i for i, r in self.graph.graph.get('rel_map_rev', {}).items()}
        self._IS_A = m.get("is a", -1)
        self._SUBSUMES = m.get("subsumes", -1)
        self._EQUIV_INTS = frozenset(x for x in (m.get(r) for r in EQUIV_REL_NAMES) if x and x > 0)
        return self._IS_A, self._SUBSUMES, self._EQUIV_INTS

    # ──────────────────────────────────────────────────────────────────
    # Hub-safe equivalence closure
    # ──────────────────────────────────────────────────────────────────

    def _equiv_closure(self, seed: int, depth=2) -> frozenset:
        """All nodes reachable from *seed* via equivalence edges.

        **Combination-product guard** — Concepts with ≥2 outgoing ``maps_to``
        edges (pre-indexed as ``_multi_target_mappers``) are combination/
        multi-ingredient products.  They are included in the closure but
        equivalence is NOT propagated *through* them, preventing false
        bridges like ``spironolactone ≡ combo ≡ furosemide``.

        CACHED + cross-cached: every member of a closure maps to the same
        result.
        """
        if seed in self._equiv_cache:
            return self._equiv_cache[seed]

        eq = {seed}
        q = deque([seed])
        while q:
            node = q.popleft()
            # If this non-seed node is a combination product, include it
            # but do NOT propagate equivalence through it.
            if node != seed and node in self._multi_target_mappers:
                continue
            for nb in self._equiv_bidir.get(node, ()):
                if nb not in eq:
                    eq.add(nb)
                    q.append(nb)

        result = frozenset(eq)
        for member in result:
            self._equiv_cache[member] = result
        if len(self._equiv_cache) > self._equiv_cache_max:
            keys = list(self._equiv_cache.keys())
            for k in keys[:self._equiv_cache_max // 5]:
                self._equiv_cache.pop(k, None)
        return result

    # ──────────────────────────────────────────────────────────────────
    # Directed BFS along is_a / subsumes
    # ──────────────────────────────────────────────────────────────────

    def _bfs_dir(self, seeds, max_h: int, direction: str = 'up') -> dict:
        """BFS along 'is a' (up) or 'subsumes' (down) from seed set.

        Returns dict  {reached_node: hop_distance}.
        """
        adj = self._isa_succ if direction == 'up' else self._subs_succ
        visited = set(seeds)
        result = {}
        q = deque((n, 0) for n in seeds)
        while q:
            node, d = q.popleft()
            if d >= max_h:
                continue
            for s in adj.get(node, ()):
                if s not in visited:
                    visited.add(s)
                    result[s] = d + 1
                    q.append((s, d + 1))
        return result

    def _allowed_depth(self, vs: str, vg: str, k: int) -> int:
        if vs not in {"rxnorm", "atc"} and vg not in {"rxnorm", "atc"}:
            return k
        if vs != vg:
            pair = {vs, vg}
            if pair in ({"snomed", "rxnorm"}, {"snomed", "loinc"}):
                return k + 1
            if pair in ({"atc", "rxnorm"}, {"snomed", "atc"}):
                return k + 2
            return k + 1
        return k + 1 if vs == "atc" else k

    def _resolve_targets(self, tids, exclude=None) -> set:
        out = set()
        for t in tids:
            r = self.resolve_id(t)
            if r is not None and r in self.graph and (exclude is None or r != exclude):
                out.add(r)
        return out

    # ══════════════════════════════════════════════════════════════════
    # Core accessors
    # ══════════════════════════════════════════════════════════════════

    def resolve_id(self, identifier) -> int | None:
        if isinstance(identifier, (int, float)):
            c = int(identifier)
            return c if c in self.graph else None
        s = str(identifier).strip()
        if s.isdigit():
            c = int(s)
            return c if c in self.graph else None
        if ':' in s:
            v, c = s.split(':', 1)
            return self.graph.graph.get('code_index', {}).get(
                (v.strip().lower(), c.strip().lower()))
        return None

    def get_node_attr(self, nid, attr):
        try:
            if attr in ('vocabulary', 'concept_name', 'name', 'concept_class'):
                col = {'name': 'concept_name', 'vocabulary': 'concept_vocabulary'}.get(
                    attr, f'concept_{attr}' if 'concept' not in attr else attr)
                meta = self.graph.graph.get('meta')
                if meta is None or col not in meta.columns:
                    return ""
                val = meta.at[nid, col]
                return str(val) if pd.notna(val) else ""
            if attr in ('synonyms', 'concept_synonym'):
                z = self.graph.graph.get('syn_map_z', {}).get(nid)
                return zlib.decompress(z).decode('utf-8') if z else ""
        except Exception:
            pass
        return ""

    def get_parents(self, cid: int) -> List[int]:
        if cid not in self.graph:
            return []
        return list(self._isa_succ.get(cid, ()))

    def _sssp_lengths(self, start: int, cutoff: int = 3) -> dict:
        key = (start, cutoff)
        if key in self._sssp_cache:
            self._sssp_cache.move_to_end(key)
            return self._sssp_cache[key]
        d = dict(nx.single_source_shortest_path_length(self.graph, start, cutoff=cutoff))
        self._sssp_cache[key] = d
        if len(self._sssp_cache) > self._sssp_cache_max:
            self._sssp_cache.popitem(last=False)
        return d

    # ══════════════════════════════════════════════════════════════════
    # Sibling detection
    # ══════════════════════════════════════════════════════════════════

    def is_sibling_path(self, src: int, tgt: int, max_hops: int = 2) -> bool:
        """Detect whether two concepts are distinct members of the same
        pharmacological or clinical class (shared ancestor, neither is
        ancestor of the other).
        """
        if src == tgt:
            return False
        key = (min(src, tgt), max(src, tgt))
        if key in self._sibling_cache:
            return self._sibling_cache[key]

        src_eq = self._equiv_closure(src)
        if tgt in src_eq:
            self._sibling_cache[key] = False
            return False

        tgt_eq = self._equiv_closure(tgt)

        src_anc = self._bfs_dir(src_eq, max_hops, 'up')
        tgt_anc = self._bfs_dir(tgt_eq, max_hops, 'up')

        # Parent–child check — exclude multi-target mappers (combo products)
        # which appear in both closures but don't represent hierarchy
        mtm = self._multi_target_mappers
        if (tgt_eq - mtm) & (set(src_anc) | (src_eq - mtm)):
            self._sibling_cache[key] = False
            return False
        if (src_eq - mtm) & (set(tgt_anc) | (tgt_eq - mtm)):
            self._sibling_cache[key] = False
            return False

        # Same-vocab ancestor overlap
        if set(src_anc) & set(tgt_anc):
            self._sibling_cache[key] = True
            return True

        # Cross-vocab bridge via equiv-normalised ancestors
        src_anc_closures = {id(self._equiv_closure(a)) for a in src_anc}
        for a in tgt_anc:
            if id(self._equiv_closure(a)) in src_anc_closures:
                self._sibling_cache[key] = True
                return True

        self._sibling_cache[key] = False
        return False

    # ══════════════════════════════════════════════════════════════════
    # LOINC axis comparison
    # ══════════════════════════════════════════════════════════════════

    def get_loinc_axes(self, cid: int) -> dict:
        try:
            return self.graph.graph.get('loinc_axes', {}).get(int(cid), {})
        except (ValueError, TypeError):
            return {}

    def compare_loinc_axes(self, sid: int, tid: int) -> dict:
        try:
            sid, tid = int(sid), int(tid)
        except (ValueError, TypeError):
            return {'is_match': False, 'reason': 'invalid IDs'}
        sa, ta = self.get_loinc_axes(sid), self.get_loinc_axes(tid)
        if 'component' not in sa or 'component' not in ta:
            return {'is_match': False, 'reason': 'component missing',
                    'source_axes': sa, 'target_axes': ta}
        matched, mismatched = [], []
        for ax in LOINC_REQUIRED_AXES:
            s, t = sa.get(ax), ta.get(ax)
            if s and t:
                (matched if s[0] == t[0] else mismatched).append(
                    (ax, s[1]) if s[0] == t[0] else (ax, s[1], t[1]))
            elif ax == 'component':
                mismatched.append((ax, s, t))
        ignored = [(a, sa[a][1], ta[a][1]) for a in LOINC_IGNORABLE_AXES
                   if a in sa and a in ta and sa[a][0] != ta[a][0]]
        return {'is_match': not mismatched,
                'matched': matched, 'mismatched': mismatched or ignored}

    # ══════════════════════════════════════════════════════════════════
    # Matching methods
    # ══════════════════════════════════════════════════════════════════

    def check_exact_term_match(self, start, target_ids):
        exact = set()
        start_name = self.get_node_attr(start, "concept_name").lower()
        for tid in target_ids:
            if self.get_node_attr(tid, "concept_name").lower() == start_name:
                exact.add(tid)
        return exact

    def source_to_targets_paths(self, start, target_ids, max_depth=3, checking_method='omop'):
        try:
            start = self.resolve_id(start) or start
        except Exception:
            return []
        if start not in self.graph:
            return []

        exact_term_match_targets = self.check_exact_term_match(start, target_ids)
        if len(exact_term_match_targets) == len(target_ids):
            return [(tid, MappingRelation.SymbolicExactMatch.value) for tid in exact_term_match_targets]

        results = [(tid, MappingRelation.SymbolicExactMatch.value) for tid in exact_term_match_targets]
        target_ids = list(set(target_ids) - exact_term_match_targets)

        vs = self.get_node_attr(start, "vocabulary").lower()
        cc = self.get_node_attr(start, "concept_class").lower()
        resolved = self._resolve_targets(target_ids, exclude=start)
        if not resolved:
            return results

        # Equivalence check
        eq = self._equiv_closure(start)
        eq_matched = resolved & eq
        for tid in eq_matched:
            results.append((tid, MappingRelation.SymbolicExactMatch.value))
        resolved -= eq_matched

        # LOINC-specific handling
        handled = set()
        if vs == 'loinc' and cc == 'lab test':
            for tid in resolved:
                if self.get_node_attr(tid, "vocabulary").lower() == 'loinc':
                    handled.add(tid)
                    if self.graph.has_edge(start, tid) or self.graph.has_edge(tid, start):
                        results.append((tid, MappingRelation.SymbolicCloseMatch.value))
                    elif self.compare_loinc_axes(start, tid)['is_match']:
                        results.append((tid, MappingRelation.SymbolicCloseMatch.value))

        remaining = resolved - handled
        if remaining:
       

            ancestors   = self._bfs_dir(eq, max_depth + 2, 'up')
            descendants = self._bfs_dir(eq, max_depth + 2, 'down')

            for tid in remaining:
                vg = self.get_node_attr(tid, "vocabulary").lower()
                allowed = self._allowed_depth(vs, vg, max_depth)

                # Forward: from src, walk hierarchy
                if tid in ancestors and ancestors[tid] <= allowed:
                    results.append((tid, MappingRelation.SymbolicBroadMatch.value))
                    continue
                if tid in descendants and descendants[tid] <= allowed:
                    results.append((tid, MappingRelation.SymbolicNarrowMatch.value))
                    continue

                # Reverse: from tgt, walk hierarchy back into src's equiv closure
                tgt_eq = self._equiv_closure(tid)
                tgt_anc = self._bfs_dir(tgt_eq, allowed, 'up')      # tgt is descendant of src?
                hit = next(((n, tgt_anc[n]) for n in tgt_anc if n in eq), None)
                if hit is not None:
                    results.append((tid, MappingRelation.SymbolicNarrowMatch.value))
                    continue

                tgt_desc = self._bfs_dir(tgt_eq, allowed, 'down')   # tgt is ancestor of src?
                hit = next(((n, tgt_desc[n]) for n in tgt_desc if n in eq), None)
                if hit is not None:
                    results.append((tid, MappingRelation.SymbolicBroadMatch.value))
                    continue

                # Fallback (mixed paths, sibling-via-pivot, etc.)
                dists = self._sssp_lengths(start, cutoff=max_depth + 2)
                if tid in dists and dists[tid] <= allowed and not self.is_sibling_path(start, tid):
                    results.append((tid, MappingRelation.SymbolicCloseMatch.value))
        return results

    # ══════════════════════════════════════════════════════════════════
    # Gate pre-computation cache
    # ══════════════════════════════════════════════════════════════════

    def _get_gate_precomp(self, source: int, k_hop: int, max_hop: int):
        """Cache the expensive per-source pre-computation.

        Returns: (eq, ancestors, descendants, src_ext_anc, vs, cc)
        """
        key = (source, k_hop, max_hop)
        if key in self._gate_precomp_cache:
            self._gate_precomp_cache.move_to_end(key)
            return self._gate_precomp_cache[key]

        vs = self.get_node_attr(source, "vocabulary").lower()
        cc = self.get_node_attr(source, "concept_class").lower()
        eq = self._equiv_closure(source)
        ancestors = self._bfs_dir(eq, k_hop, 'up')
        descendants = self._bfs_dir(eq, k_hop, 'down')
        src_ext_anc = frozenset(self._bfs_dir(eq, k_hop + max_hop, 'up'))

        result = (eq, ancestors, descendants, src_ext_anc, vs, cc)
        self._gate_precomp_cache[key] = result
        if len(self._gate_precomp_cache) > self._gate_precomp_cache_max:
            self._gate_precomp_cache.popitem(last=False)
        return result

    def check_concept_match(self, sid: int, tid: int, max_depth: int = 1) -> Tuple[bool, str]:
        try:
            sid, tid = int(sid), int(tid)
        except (ValueError, TypeError):
            return False, "invalid IDs"
        sv = self.get_node_attr(sid, 'vocabulary').lower()
        tv = self.get_node_attr(tid, 'vocabulary').lower()
        if sv == 'loinc' and tv == 'loinc':
            r = self.compare_loinc_axes(sid, tid)
            if r['is_match']:
                return True, "LOINC Axis Match"
            return False, f"LOINC Mismatch:{[m[0] for m in r['mismatched']]}"
        results = self.source_to_targets_paths(sid, [tid], max_depth=max_depth)
        return (True, results[0][1]) if results else (False, "No match")

    # ══════════════════════════════════════════════════════════════════
    # Ancestor / sibling utilities
    # ══════════════════════════════════════════════════════════════════

    def get_ancestors_fast(self, cid: int, max_hops: int = 3) -> frozenset:
        key = (cid, max_hops)
        if key in self._ancestor_cache:
            return self._ancestor_cache[key]
        if cid not in self.graph:
            self._ancestor_cache[key] = frozenset()
            return frozenset()
        eq = self._equiv_closure(cid)
        result = frozenset(self._bfs_dir(eq, max_hops, 'up'))
        self._ancestor_cache[key] = result
        return result

    def filter_sibling_matches(self, source_id: int, cands: set, max_hops: int = 3) -> set:
        try:
            source_id = int(source_id)
        except (ValueError, TypeError):
            return cands
        sa = self.get_ancestors_fast(source_id, max_hops)
        if not sa:
            return cands
        safe = set()
        for tid in cands:
            try:
                ti = int(tid)
            except (ValueError, TypeError):
                safe.add(tid)
                continue
            if ti in sa or not (sa & self.get_ancestors_fast(ti, max_hops)):
                safe.add(tid)
        return safe

    def concept_exists(self, cid: int, code: str, vocabulary: List[str]) -> Tuple[bool, str]:
        vocabulary, code = [v.lower() for v in vocabulary], code.lower()
        if cid not in self.graph:
            return False, "not found"
        meta = self.graph.graph.get('meta')
        if meta is None or cid not in meta.index:
            return False, "not found"
        row = meta.loc[cid]
        nv = str(row.get('concept_vocabulary', '')).strip().lower()
        nc = str(row.get('concept_code', '')).strip().lower()
        if not nv and not nc:
            return False, "not found"
        return (True, "correct") if nv in vocabulary and nc == code else (False, "incorrect")

    def get_vocabulary_stats(self):
        meta = self.graph.graph.get('meta')
        if meta is not None and 'concept_vocabulary' in meta.columns:
            c = meta['concept_vocabulary'].value_counts()
            print(f"Total unique vocabularies:{len(c)}\n\nVocabulary distribution:\n{c}")
            return c
        return None

    # ══════════════════════════════════════════════════════════════════
    # Build / Save / Load
    # ══════════════════════════════════════════════════════════════════

    def build_graph(self, csv_file_path=None):
        csv_file_path = csv_file_path or self.csv_file_path
        if not csv_file_path:
            raise ValueError("No CSV file path provided.")

        print("Reading CSV...")
        use_cols = [
            "concept_id_1", "concept_id_2", "relationship_id",
            "concept_vocabulary_1", "concept_vocabulary_2",
            "concept_name_1", "concept_name_2",
            "concept_code_1", "concept_code_2",
            "concept_synonym_1", "concept_synonym_2",
            "concept_class_1", "concept_class_2",
        ]
        header = pd.read_csv(csv_file_path, nrows=0)
        actual = [c for c in use_cols if c in header.columns]
        df = pd.read_csv(csv_file_path, usecols=actual, dtype=str)
        df['relationship_id'] = df['relationship_id'].str.lower()

        for col in ['concept_vocabulary_1', 'concept_vocabulary_2']:
            if col in df.columns:
                print(f"Unique vocabs:{sorted(df[col].dropna().unique())}")

        # ── Separate LOINC axis rows from edge rows ──
        loinc_df = df[df['relationship_id'].isin(LOINC_AXIS_RELS)].copy()
        df_e = df[~df['relationship_id'].isin(LOINC_AXIS_RELS)].copy()
        df_e = df_e[df_e['relationship_id'].isin(set(EQ_RELS) | set[str](DIR_RELS))].copy()
        print(f"df_e head\n{df_e.head()}")
        print(f"LOINC axis:{len(loinc_df):,}, Edge rows:{len(df_e):,}")

        # ── Build node metadata ──
        c1 = {c: c[:-2] for c in actual if c.endswith('_1')}
        print(f"c1= {c1}")
        c2 = {c: c[:-2] for c in actual if c.endswith('_2')}
        print(f"c2= {c2}")
        all_df = pd.concat([df_e, loinc_df], ignore_index=True)
        fm = pd.concat([
            all_df[list[str](c1)].rename(columns=c1),
            all_df[list(c2)].rename(columns=c2),
        ], ignore_index=True)
        fm['concept_id'] = pd.to_numeric(fm['concept_id'], errors='coerce').fillna(0).astype('int64')
        fm = fm[fm['concept_id'] != 0]

        mcols = ['concept_id', 'concept_vocabulary', 'concept_name', 'concept_synonym', 'concept_code']
        if 'concept_class' in fm.columns:
            mcols.append('concept_class')
        meta = fm.drop_duplicates(subset=['concept_id'])[
            [c for c in mcols if c in fm.columns]].copy()
        meta.set_index('concept_id', inplace=True)
        meta['concept_vocabulary'] = meta['concept_vocabulary'].fillna("").astype('category')
        self.graph.graph['meta'] = meta

        # ── Code index ──
        ci = {}
        for cid, row in meta.iterrows():
            code = str(row.get('concept_code', '')).strip().lower()
            if code and code != 'nan':
                ci[(str(row.get('concept_vocabulary', '')).strip().lower(), code)] = int(cid)
        self.graph.graph['code_index'] = ci

        # ── LOINC axis map ──
        lam = {}
        loinc_df['u'] = pd.to_numeric(loinc_df['concept_id_1'], errors='coerce').fillna(0).astype('int64')
        loinc_df['v'] = pd.to_numeric(loinc_df['concept_id_2'], errors='coerce').fillna(0).astype('int64')
        for r in loinc_df.itertuples(index=False):
            if r.u and r.v: # concept_id_1 and concept_id_2
                ax = LOINC_AXIS_RELS.get(r.relationship_id)
                if ax:
                    lam.setdefault(r.u, {})[ax] = (r.v, getattr(r, 'concept_name_2', ""))
        self.graph.graph['loinc_axes'] = lam

        # ── Add nodes ──
        self.graph.add_nodes_from(meta.index)

        # ── Relationship integer map ──
        all_r = sorted(set(EQ_RELS) | set(EQ_RELS.values()) | set(DIR_RELS) | set(DIR_RELS.values()))
        rm = {r: i + 1 for i, r in enumerate(all_r)}
        self.graph.graph['rel_map_rev'] = {i: r for r, i in rm.items()}
        self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None

        # ── Build edges — collect ALL rel-ints per (u,v) pair ──
        df_e['u'] = pd.to_numeric(df_e['concept_id_1'], errors='coerce').fillna(0).astype('int64')
        df_e['v'] = pd.to_numeric(df_e['concept_id_2'], errors='coerce').fillna(0).astype('int64')

        edge_rels = {}  # (u, v) → set of rel-ints
        for r in df_e.itertuples(index=False):
            rel, u, v = r.relationship_id, r.u, r.v
            if not u or not v:
                continue
            r1 = rm.get(rel, 0)
            if rel in EQ_RELS:
                r2 = rm.get(EQ_RELS[rel], 0)
                if r1:
                    edge_rels.setdefault((u, v), set()).add(r1)
                if r2:
                    edge_rels.setdefault((v, u), set()).add(r2)
            elif rel in DIR_RELS and r1:
                edge_rels.setdefault((u, v), set()).add(r1)

        # Store primary rel (deterministic: smallest int) + full set
        IS_A_INT = rm.get("is a", -1)
        SUBS_INT = rm.get("subsumes", -1)
        dir_ints = {IS_A_INT, SUBS_INT} - {-1}
        edges = []
        for (u, v), rels in edge_rels.items():
            # Prefer hierarchical as primary, then smallest
            primary = min(rels & dir_ints) if (rels & dir_ints) else min(rels)
            edges.append((u, v, {'r': primary, 'all_r': frozenset(rels)}))

        self.graph.add_edges_from(edges)

        # ── Build typed adjacency AFTER edges are added ──
        self._build_typed_adjacency()

        self.save_graph(self.output_file)
        print(f"[INFO] Done. Nodes:{self.graph.number_of_nodes():,}, Edges:{self.graph.number_of_edges():,}")

    def save_graph(self, path):
        out = path if path.endswith(".gz") else path + ".gz"
        bundle = {
            'graph': self.graph,
            'isa_succ': self._isa_succ,
            'subs_succ': self._subs_succ,
            'equiv_bidir': self._equiv_bidir,
            'multi_target_mappers': self._multi_target_mappers,
        }
        with gzip.open(out, "wb", compresslevel=6) as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved to {out}")

    def load_graph(self, path):
        if not path.endswith(".gz") and os.path.exists(path + ".gz"):
            path += ".gz"
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'graph' in data:
            self.graph = data['graph']
            self._isa_succ = data.get('isa_succ', {})
            self._subs_succ = data.get('subs_succ', {})
            self._equiv_bidir = data.get('equiv_bidir', {})
            self._multi_target_mappers = data.get('multi_target_mappers', frozenset())
        else:
            self.graph = data  # backward compat with old pickles
            self._build_typed_adjacency()
        self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None
        print(f"[INFO] Loaded {path}. Nodes:{self.graph.number_of_nodes()} Edges:{self.graph.number_of_edges()}")

    def clear_caches(self):
        """Clear all caches."""
        self._sssp_cache.clear()
        self._sibling_cache.clear()
        self._ancestor_cache.clear()
        self._equiv_cache.clear()
        self._gate_precomp_cache.clear()

    # ══════════════════════════════════════════════════════════════════
    # Explainable path
    # ══════════════════════════════════════════════════════════════════

    def explain_path(self, source_id: int, target_id: int, max_depth: int = 5) -> dict:
        """Return an explainable path between two OMOP concepts.

        Returns dict with keys:
            path_type:  'exact_match' | 'equivalence' | 'loinc_axis' |
                        'ancestor' | 'descendant' | 'sibling' |
                        'graph_traversal' | 'no_path'
            distance:   int (-1 if no path)
            path:       list of (concept_id, concept_name, vocabulary)
            edges:      list of (source_id, target_id, relationship)
            explanation: str
        """
        src = self.resolve_id(source_id)
        tgt = self.resolve_id(target_id)

        def _node_info(cid):
            return (cid, self.get_node_attr(cid, 'name'),
                    self.get_node_attr(cid, 'vocabulary'))

        def _no_path(reason):
            return {"path_type": "no_path", "distance": -1,
                    "path": [], "edges": [], "explanation": reason}

        def _annotate_path(node_ids):
            path = [_node_info(n) for n in node_ids]
            edges = []
            for i in range(len(node_ids) - 1):
                u, v = node_ids[i], node_ids[i + 1]
                rel = self.get_edge_rel(u, v)
                if not rel:
                    rel = self.get_edge_rel(v, u)
                if not rel:
                    if (v in self._equiv_bidir.get(u, ())
                            or u in self._equiv_bidir.get(v, ())):
                        rel = "equivalent"
                    else:
                        rel = "unknown"
                edges.append((u, v, rel))
            return path, edges

        if (src is None or tgt is None
                or src not in self.graph or tgt not in self.graph):
            return _no_path(
                f"Concept(s) not in graph:src={source_id} tgt={target_id}")

        s_info, t_info = _node_info(src), _node_info(tgt)

        # 1. Identity
        if src == tgt:
            return {"path_type": "exact_match", "distance": 0,
                    "path": [s_info], "edges": [],
                    "explanation": f"Same concept:{s_info[1]}"}

        # 2. Equivalence closure
        eq = self._equiv_closure(src)
        if tgt in eq:
            visited, parent = {src}, {src: None}
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in self._equiv_bidir.get(node, ()):
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = node
                        if nb == tgt:
                            break
                        q.append(nb)
                if tgt in parent:
                    break
            chain = []
            cur = tgt
            while cur is not None:
                chain.append(cur)
                cur = parent.get(cur)
            chain.reverse()
            path, edges = _annotate_path(chain)
            rels = " → ".join(r for _, _, r in edges)
            return {"path_type": "equivalence", "distance": len(chain) - 1,
                    "path": path, "edges": edges,
                    "explanation": f"Equivalent via:{s_info[1]} →[{rels}]→ {t_info[1]}"}

        # 3. LOINC axis match
        sv = self.get_node_attr(src, 'vocabulary').lower()
        tv = self.get_node_attr(tgt, 'vocabulary').lower()
        sc = self.get_node_attr(src, 'concept_class').lower()
        if sv == 'loinc' and tv == 'loinc' and sc == 'lab test':
            lr = self.compare_loinc_axes(src, tgt)
            if lr.get('is_match'):
                axes = [ax for ax, _ in lr.get('matched', [])]
                return {"path_type": "loinc_axis", "distance": 1,
                        "path": [s_info, t_info], "edges": [],
                        "explanation": (f"LOINC axis match on "
                                        f"[{', '.join(axes)}]:"
                                        f"{s_info[1]} ↔ {t_info[1]}"),
                        "loinc_details": lr}

        # 4. Ancestor (is-a chain upward)
        ancestors = self._bfs_dir(eq, max_depth, 'up')
        if tgt in ancestors:
            chain = self._trace_directed(src, tgt, 'up', max_depth)
            path, edges = _annotate_path(chain) if chain else ([s_info, t_info], [])
            return {"path_type": "ancestor", "distance": ancestors[tgt],
                    "path": path, "edges": edges,
                    "explanation": (f"{s_info[1]} ──is a──▷ "
                                    f"{t_info[1]} ({ancestors[tgt]} hops)")}

        # 5. Descendant (subsumes chain downward)
        descendants = self._bfs_dir(eq, max_depth, 'down')
        if tgt in descendants:
            chain = self._trace_directed(src, tgt, 'down', max_depth)
            path, edges = _annotate_path(chain) if chain else ([s_info, t_info], [])
            return {"path_type": "descendant", "distance": descendants[tgt],
                    "path": path, "edges": edges,
                    "explanation": (f"{t_info[1]} ◁──subsumes── "
                                    f"{s_info[1]} ({descendants[tgt]} hops)")}

        # 6. General shortest path (catches sibling / mixed)
        try:
            raw = nx.shortest_path(self.graph, src, tgt)
            if len(raw) - 1 <= max_depth:
                path, edges = _annotate_path(raw)
                has_up = any(r == "is a" for _, _, r in edges)
                has_down = any(r == "subsumes" for _, _, r in edges)
                ptype = "sibling" if (has_up and has_down) else "graph_traversal"
                pivot = ""
                if ptype == "sibling":
                    for i, (_, v, r) in enumerate(edges):
                        if (r == "is a" and i + 1 < len(edges)
                                and edges[i + 1][2] == "subsumes"):
                            pivot = f" via [{self.get_node_attr(v, 'name')}]"
                            break
                rels = " → ".join(r for _, _, r in edges)
                label = 'Sibling' if ptype == 'sibling' else 'Connected'
                return {"path_type": ptype, "distance": len(raw) - 1,
                        "path": path, "edges": edges,
                        "explanation": (f"{label}{pivot}:"
                                        f"{s_info[1]} →[{rels}]→ "
                                        f"{t_info[1]} ({len(raw)-1} hops)")}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return _no_path(
            f"No path within {max_depth} hops:{s_info[1]} ↔ {t_info[1]}")

    # ──────────────────────────────────────────────────────────────────
    # Directed path tracing (with equiv-hop visibility)
    # ──────────────────────────────────────────────────────────────────

    def _trace_directed(self, src: int, tgt: int,
                        direction: str, max_depth: int) -> List[int]:
        """BFS to recover the actual node chain along is-a/subsumes edges,
        correctly showing equivalence hops instead of collapsing them.

        Phase 1 — BFS through is_a/subsumes from every equiv member,
                  tracking the entry-point (which eq member reached tgt).
        Phase 2 — Reconstruct hierarchical chain from tgt back to the
                  equiv entry-point.
        Phase 3 — If entry-point ≠ src, prepend the equiv BFS chain
                  from src to the entry-point so the full cross-vocab
                  path is visible.
        """
        adj = self._isa_succ if direction == 'up' else self._subs_succ
        eq = self._equiv_closure(src)

        # Phase 1: directed BFS from all equiv members
        visited = set(eq)
        parent = {n: None for n in eq}
        entry_point = {n: n for n in eq}
        q = deque((n, 0) for n in eq)

        found = False
        while q:
            node, d = q.popleft()
            if d >= max_depth:
                continue
            for s in adj.get(node, ()):
                if s not in visited:
                    visited.add(s)
                    parent[s] = node
                    entry_point[s] = entry_point[node]
                    if s == tgt:
                        found = True
                        break
                    q.append((s, d + 1))
            if found:
                break

        if not found:
            return [src, tgt]

        # Phase 2: reconstruct hierarchical chain
        hier_chain = []
        cur = tgt
        while cur is not None:
            hier_chain.append(cur)
            cur = parent.get(cur)
        hier_chain.reverse()

        # Phase 3: prepend equiv chain if entry ≠ src
        ep = hier_chain[0]
        if ep == src:
            return hier_chain

        eq_visited, eq_parent = {src}, {src: None}
        eq_q = deque([src])
        while eq_q:
            node = eq_q.popleft()
            if node == ep:
                break
            for nb in self._equiv_bidir.get(node, ()):
                if nb not in eq_visited:
                    eq_visited.add(nb)
                    eq_parent[nb] = node
                    eq_q.append(nb)
                    if nb == ep:
                        break
            if ep in eq_parent:
                break

        equiv_chain = []
        cur = ep
        while cur is not None:
            equiv_chain.append(cur)
            cur = eq_parent.get(cur)
        equiv_chain.reverse()

        return equiv_chain + hier_chain[1:]

def run_pair_tests(omop_nx):
    """Test source_to_targets_paths against curated concept pairs."""

    cases = [
        # (src, tgt, should_match, description)
        # ── LOINC hierarchy: parent-child should NOT match ──
        (4248525, 4060832, True,
         "lying systolic BP vs systolic BP (parent-child)"),
        (4248525, 4326744, False,
         "lying systolic BP vs blood pressure (grandparent-child)"),

        # ── Drug cross-vocab: unrelated vs related ──
        (4306892, 21601810, False,
         "furosemide vs cilazapril+diuretics (unrelated combo)"),
        (4306892, 21601516, True,
         "furosemide vs HIGH-CEILING DIURETICS (ancestor class)"),

        # ── Combination ingredient: should NOT match single ingredient ──
        (21035025, 956874, False,
         "amiloride/furosemide oral soln vs furosemide (combo→ingredient)"),

        # ── Maps-to equivalence ──
        (4151548, 3020399, False,
         "Glucose measurement, body fluid vs glucose [mass/volume] urine (maps_to)"),

        # ── LOINC loose match (same component, different property) ──
        (3020399, 3005570, True,
         "glucose [mass/vol] urine vs glucose [mol/vol] urine (LOINC axes)"),

        # ── ATC hierarchy ──
        (21601517, 21601520, True,
         "sulfonamides plain vs piretanide (parent→child)"),
        (21601517, 21601521, True,
         "sulfonamides plain vs torasemide (parent→child)"),
        (21601517, 942350, True,
         "sulfonamides plain vs torsemide (cross-vocab descendant)"),

        # ── ATC siblings: should NOT match ──
        (942350, 21601520, False,
         "torsemide vs piretanide (siblings under same parent)"),

        # ── Disease hierarchy ──
        (312327, 4329847, True,
         "acute MI vs myocardial infarction (child→parent)"),
        (4173632, 312327, False,
         "microinfarct of heart vs acute MI (siblings)"),
         (4242997, 4336464, False,
         "Cholecystectomy vsC oronary artery bypass graft"),
         (21600961,4146455,False,
         "ANTITHROMBOTIC AGENTS Antihypertensive therapy"),
         (21600961, 3655005, False,
         "antithrombotic agents vs Platelet aggregation inhibitor therapy"),
         (4029066, 21601521, True,
         "Torsemide vs torasemide; oral, parenteral"),
         (21601665, 1314002, True,
         "BETA BLOCKING AGENTS vs Atenolol"),
          (1314002, 21601665, True,
         "Atenolol vs BETA BLOCKING AGENTS")
    ]

    passed = failed = 0
    for src, tgt, expected, desc in cases:
        results = omop_nx.source_to_targets_paths(src, [tgt], max_depth=1)
        matched = len(results) > 0
        ok = matched == expected

        # Gather diagnostics
        src_name = omop_nx.get_node_attr(src, 'name')
        tgt_name = omop_nx.get_node_attr(tgt, 'name')
        src_vocab = omop_nx.get_node_attr(src, 'vocabulary')
        tgt_vocab = omop_nx.get_node_attr(tgt, 'vocabulary')
        match_label = results[0][1] if results else "none"

        status = "✓" if ok else "✗ FAIL"
        expect_str = "match" if expected else "no match"
        got_str = f"matched ({match_label})" if matched else "no match"

        print(f"  {status}  [{src_vocab}] {src_name} ({src}) "
              f"→ [{tgt_vocab}] {tgt_name} ({tgt})")
        print(f"         expected: {expect_str}, got: {got_str}")

        if not ok:
            failed += 1
            path = omop_nx.explain_path(src, tgt)
            print(f"         explain:  {path['path_type']} — {path['explanation']}")
            eq_src = omop_nx._equiv_closure(src)
            print(f"equv({src}): {sorted(eq_src)[:8]}{'...' if len(eq_src) > 8 else ''}")
        else:
            passed += 1

    print(f"\n  Pair tests: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0
