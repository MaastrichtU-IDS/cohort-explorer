#new version of the module. From Komal (komi786) 
# added 05.03.2026

import numpy
if not hasattr(numpy, '_core'):
    numpy._core = numpy.core

import networkx as nx
import pandas as pd
import pickle, os, gzip, zlib, time
from typing import List, Tuple
from collections import deque, OrderedDict

LOINC_REQUIRED_AXES = ['component', 'scale_type', 'time_aspect']
LOINC_IGNORABLE_AXES = ['system', 'property', 'method', 'specimen']
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
    'has dose form': 'dose form of', 'dose form of': 'has dose form',
    'value mapped from': 'maps to value', 'maps to value': 'value mapped from',
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
    "has dose form", "dose form of",
    "value mapped from", "maps to value",
})


class BlockingFilter:
    __slots__ = ('_check_fn', 'blocked', 'passed', 'equiv_class', 'source')
    def __init__(self, check_fn, source=None, equiv_class=None):
        self._check_fn, self.source = check_fn, source
        self.blocked, self.passed, self.equiv_class = {}, set(), equiv_class or set()
    def __call__(self, tid: int) -> bool:
        result = self._check_fn(tid)
        if result: self.blocked[tid] = "hierarchically related"
        else: self.passed.add(tid)
        return result
    def summary(self, graph=None):
        lines = [f"Source: {self.source}" + (f" ({graph.get_node_attr(self.source, 'name')})" if graph else ""),
                 f"Blocked: {len(self.blocked)}, Passed: {len(self.passed)}"]
        for tid, reason in self.blocked.items():
            name = graph.get_node_attr(tid, 'name') if graph else str(tid)
            lines.append(f"  ✗ {tid} ({name}) — {reason}")
        return "\n".join(lines)


class OmopGraphNX:
    def __init__(self, csv_file_path=None, output_file='graph_nx.pkl.gz'):
        from CohortVarLinker.src.config import settings
        self.csv_file_path = csv_file_path
        # Use data_folder for the default location of graph file
        if output_file is None:
            output_file = os.path.join(settings.data_folder, "graph_nx.pkl.gz")
        elif not os.path.isabs(output_file):
            output_file = os.path.join(settings.data_folder, output_file)
        self.output_file = output_file
        self.graph = nx.DiGraph()

        # ── Caches ──
        self._sssp_cache, self._sssp_cache_max = OrderedDict(), 100_000  # up from 20K
        self._sibling_cache = {}
        self._ancestor_cache = {}
        self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None

        # ── NEW: Pre-indexed typed adjacency for O(1) typed-neighbor lookups ──
        # ROOT CAUSE FIX: The original _bfs_dir scanned ALL g.successors(node)
        # then filtered by edge type.  Hub nodes (common SNOMED/LOINC concepts)
        # have thousands of successors.  Scanning all of them to find the handful
        # of "is a" edges caused the >30 min hang.
        # Pre-indexing turns _bfs_dir from O(total_degree) to O(typed_degree).
        self._isa_succ = {}       # node -> frozenset of "is a" successors
        self._subs_succ = {}      # node -> frozenset of "subsumes" successors
        self._equiv_bidir = {}    # node -> frozenset of equiv neighbors (both dirs)

        # ── NEW: Equiv closure cache ──
        # The same node's equiv closure was recomputed in resolve_matches,
        # again in compute_context_scores, again in _all_codes_reach_target.
        self._equiv_cache = {}
        self._equiv_cache_max = 50_000

        # ── NEW: Per-source gate pre-computation cache ──
        # Stores (eq, ancestors, descendants, src_ext_anc, vs, cc) so that the
        # same source node doesn't repeat Phase 1-2-4 across many candidate sets.
        self._gate_precomp_cache = OrderedDict()
        self._gate_precomp_cache_max = 20_000

        if os.path.exists(output_file):
            self.load_graph(output_file)
        elif csv_file_path:
            self.build_graph(csv_file_path)

    # ══════════════════════════════════════════════════════════════════
    # Typed adjacency index — built once, used everywhere
    # ══════════════════════════════════════════════════════════════════

    def _build_typed_adjacency(self):
        """Pre-index edges by relationship type for O(1) typed-neighbor lookups."""
        IS_A, SUBS, EQUIV = self._rel_ints()
        g = self.graph

        isa_succ = {}
        subs_succ = {}
        equiv_bidir = {}

        t0 = time.time()
        for u, v, data in g.edges(data=True):
            r = data.get('r', 0)
            if r == IS_A:
                isa_succ.setdefault(u, set()).add(v)
            elif r == SUBS:
                subs_succ.setdefault(u, set()).add(v)
            elif r in EQUIV:
                equiv_bidir.setdefault(u, set()).add(v)
                equiv_bidir.setdefault(v, set()).add(u)

        self._isa_succ = {k: frozenset(v) for k, v in isa_succ.items()}
        self._subs_succ = {k: frozenset(v) for k, v in subs_succ.items()}
        self._equiv_bidir = {k: frozenset(v) for k, v in equiv_bidir.items()}

        elapsed = time.time() - t0
        print(f"[INFO] Built typed adjacency index in {elapsed:.2f}s "
              f"(is_a: {sum(len(v) for v in self._isa_succ.values()):,}, "
              f"subsumes: {sum(len(v) for v in self._subs_succ.values()):,}, "
              f"equiv: {sum(len(v) for v in self._equiv_bidir.values()):,})")

    # ══════════════════════════════════════════════════════════════════
    # Shared helpers
    # ══════════════════════════════════════════════════════════════════

    def _rel_ints(self):
        if self._IS_A is not None:
            return self._IS_A, self._SUBSUMES, self._EQUIV_INTS
        m = {r: i for i, r in self.graph.graph.get('rel_map_rev', {}).items()}
        self._IS_A, self._SUBSUMES = m.get("is a", -1), m.get("subsumes", -1)
        self._EQUIV_INTS = frozenset(x for x in (m.get(r) for r in EQUIV_REL_NAMES) if x and x > 0)
        return self._IS_A, self._SUBSUMES, self._EQUIV_INTS

    def _equiv_closure(self, seed: int) -> frozenset:
        """All nodes reachable from `seed` via equivalence edges.
        CACHED + cross-cached: every member of a closure maps to the same result.
        """
        if seed in self._equiv_cache:
            return self._equiv_cache[seed]

        eq = {seed}
        q = deque([seed])
        while q:
            node = q.popleft()
            for nb in self._equiv_bidir.get(node, ()):
                if nb not in eq:
                    eq.add(nb)
                    q.append(nb)

        result = frozenset(eq)

        # Cross-cache: every member points to the same frozenset
        for member in result:
            self._equiv_cache[member] = result

        if len(self._equiv_cache) > self._equiv_cache_max:
            keys = list(self._equiv_cache.keys())
            for k in keys[:self._equiv_cache_max // 5]:
                self._equiv_cache.pop(k, None)

        return result

    def _bfs_dir(self, seeds, max_h: int, direction: str = 'up') -> dict:
        """BFS along 'is a' (up) or 'subsumes' (down) from seed set.
        Uses pre-indexed typed adjacency instead of scanning all successors.
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
            if pair in ({"atc", "rxnorm"}, {"snomed", "atc"}):
                return k + 2
            if pair == {"snomed", "rxnorm"}:
                return k + 1
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
            return self.graph.graph.get('code_index', {}).get((v.strip().lower(), c.strip().lower()))
        return None

    def get_node_attr(self, nid, attr):
        try:
            if attr in ('vocabulary', 'concept_name', 'name', 'concept_class'):
                col = {'name': 'concept_name', 'vocabulary': 'concept_vocabulary'}.get(attr,
                       f'concept_{attr}' if 'concept' not in attr else attr)
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

    def get_edge_rel(self, u, v):
        if not self.graph.has_edge(u, v):
            return ""
        return self.graph.graph.get('rel_map_rev', {}).get(self.graph.get_edge_data(u, v).get('r', 0), "")

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

    def is_sibling_path(self, src: int, tgt: int, max_hops: int = 3) -> bool:
        """Shared-ancestor check via typed adjacency. O(typed_degree × hops), cached."""
        if src == tgt:
            return False
        if tgt in self._equiv_closure(src):
            return False                          # same equiv class, not sibling
        src_anc = self.get_ancestors_fast(src, max_hops)  # cached, uses equiv closure + is_a BFS
        if tgt in src_anc:
            return False                          # tgt is direct ancestor
        tgt_anc = self.get_ancestors_fast(tgt, max_hops)
        if src in tgt_anc:
            return False                          # src is direct ancestor
        return bool(src_anc & tgt_anc)            # shared ancestor = sibling
    # def is_sibling_path(self, src: int, tgt: int) -> bool:
    #     if src == tgt:
    #         return False
    #     sp = set( self.get_parents(src)) if src in self.graph else set()
    #     tp = set(self.get_parents(tgt)) if tgt in self.graph else set()
    #     if sp and tp and (sp & tp):
    #         return True
    #     return False
    
        # key = (src, tgt)
        # if key in self._sibling_cache: 
        #     return self._sibling_cache[key]

        # # 1. Hierarchy Check: If one is an ancestor of the other, they ARE NOT siblings.
        # # We use a generous hop limit to ensure we catch the vertical link.
        # src_ancestors = self.get_ancestors_fast(src, max_hops=3)
        # if tgt in src_ancestors:
        #     self._sibling_cache[key] = False
        #     return False
            
        # tgt_ancestors = self.get_ancestors_fast(tgt, max_hops=3)
        # if src in tgt_ancestors:
        #     self._sibling_cache[key] = False
        #     return False

        # # 2. Path Analysis: Only run this if no direct vertical relationship exists.
        # try: 
        #     path = nx.shortest_path(self.graph, src, tgt)
        # except (nx.NetworkXNoPath, nx.NodeNotFound): 
        #     self._sibling_cache[key] = False
        #     return False

        # # If the path is direct (length 2), it's not a sibling path.
        # if len(path) < 3: 
        #     self._sibling_cache[key] = False
        #     return False

        # up = down = False
        # for i in range(len(path) - 1):
        #     rel = self.get_edge_rel(path[i], path[i + 1])
        #     # We only care about direction switches in the hierarchical axes
        #     if rel == "is a": 
        #         up = True
        #     elif rel == "subsumes": 
        #         down = True
            
        #     # If we've gone both UP and DOWN, it's a sibling (horizontal) path.
        #     if up and down: 
        #         self._sibling_cache[key] = True
        #         return True

        # self._sibling_cache[key] = False
        # return False

    # ══════════════════════════════════════════════════════════════════
    # LOINC
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
            return {'is_match': False, 'reason': 'component missing', 'source_axes': sa, 'target_axes': ta}
        matched, mismatched = [], []
        for ax in LOINC_REQUIRED_AXES:
            s, t = sa.get(ax), ta.get(ax)
            if s and t:
                if s[0] == t[0]:
                    matched.append((ax, s[1]))
                else:
                    mismatched.append((ax, s[1], t[1]))
            elif ax == 'component':
                mismatched.append((ax, s, t))
        ignored = [(a, sa[a][1], ta[a][1]) for a in LOINC_IGNORABLE_AXES
                   if a in sa and a in ta and sa[a][0] != ta[a][0]]
        return {'is_match': not mismatched, 'matched': matched, 'mismatched': mismatched or ignored}

    # ══════════════════════════════════════════════════════════════════
    # Matching methods
    # ══════════════════════════════════════════════════════════════════

    def source_to_targets_paths(self, start, target_ids, max_depth=1, checking_method='omop'):
        try:
            start = self.resolve_id(start) or start
        except:
            return []
        if start not in self.graph:
            return []
        vs = self.get_node_attr(start, "vocabulary").lower()
        cc = self.get_node_attr(start, "concept_class").lower()
        resolved = self._resolve_targets(target_ids, exclude=start)
        if not resolved:
            return []
        results, handled = [], set()
        if vs == 'loinc' and cc == 'lab test':
            for tid in resolved:
                if self.get_node_attr(tid, "vocabulary").lower() == 'loinc':
                    handled.add(tid)
                    if self.graph.has_edge(start, tid) or self.graph.has_edge(tid, start):
                        results.append((tid, "Symbolic Match: Explicit Mapping"))
                    elif self.compare_loinc_axes(start, tid)['is_match']:
                        results.append((tid, "Symbolic Match: LabTest Properties Overlap"))
        remaining = resolved - handled
        if remaining:
            dists = self._sssp_lengths(start, cutoff=max_depth + 3)
            for tid, dist in dists.items():
                if tid in remaining and dist <= self._allowed_depth(vs, self.get_node_attr(tid, "vocabulary").lower(), max_depth) and not self.is_sibling_path(start, tid):
                    results.append((tid, "Symbolic Match: Graph Traversal"))
        return results

    # def source_to_targets_paths(self, start, target_ids, max_depth=1, checking_method='omop'):
    #     try:
    #         start = self.resolve_id(start) or start
    #     except:
    #         return [], set()
    #     if start not in self.graph:
    #         return [], set()
    #     vs = self.get_node_attr(start, "vocabulary").lower()
    #     cc = self.get_node_attr(start, "concept_class").lower()
    #     resolved = self._resolve_targets(target_ids, exclude=start)
    #     if not resolved:
    #         return [], set()
    #     results, handled, blocked = [], set(), set()
    #     if vs == 'loinc' and cc == 'lab test':
    #         for tid in resolved:
    #             if self.get_node_attr(tid, "vocabulary").lower() != 'loinc':
    #                 continue
    #             handled.add(tid)
    #             if self.graph.has_edge(start, tid) or self.graph.has_edge(tid, start):
    #                 results.append((tid, "Symbolic Match: Explicit Mapping"))
    #             else:
    #                 lr = self.compare_loinc_axes(start, tid)
    #                 if lr['is_match']:
    #                     results.append((tid, "Symbolic Match: LabTest Properties Overlap"))
    #                 elif lr.get('matched'):
    #                     blocked.add(tid)
    #     remaining = resolved - handled
    #     if remaining:
    #         dists = self._sssp_lengths(start, cutoff=max_depth + 4)
    #         for tid, dist in dists.items():
    #             if tid not in remaining:
    #                 continue
    #             allowed = self._allowed_depth(vs, self.get_node_attr(tid, "vocabulary").lower(), max_depth)
    #             if dist <= allowed and not self.is_sibling_path(start, tid):
    #                 results.append((tid, "Symbolic Match: Graph Traversal"))
    #                 handled.add(tid)
    #             else:
    #                 blocked.add(tid)
    #     return results, blocked

    # ══════════════════════════════════════════════════════════════════
    # Hierarchical Evidence Gate
    # ══════════════════════════════════════════════════════════════════

    def _get_gate_precomp(self, source: int, k_hop: int, max_hop: int):
        """Cache the expensive per-source pre-computation.

        The same source node appears across many rows (one source variable
        matched to many targets).  The eq closure, ancestors, descendants,
        and extended ancestors are identical for all of them.
        Computing once and reusing saves massive redundant BFS.

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

    def hierarchical_evidence_gate(self, source, candidates, k_hop=1, max_hop=4, allow_related=False) -> tuple[set[tuple[int, str]], set[int], set[int]]:
        try:
            source = self.resolve_id(source) or source
        except:
            return set(), set(), set(candidates)
        if source not in self.graph:
            return set(), set(), set(candidates)

        resolved = {}
        for t in candidates:
            r = self.resolve_id(t)
            if r is not None and r in self.graph and r != source:
                resolved[r] = t
        if not resolved:
            return set(), set(), set(candidates)

        g = self.graph

        # ── Cached pre-computation (eq, ancestors, descendants, ext anc) ──
        eq, ancestors, descendants, src_ext_anc, vs, cc = self._get_gate_precomp(source, k_hop, max_hop)

        # ── Phase 3: Classify candidates ──
        matched, handled = set(), set()
        if vs == 'loinc' and cc == 'lab test':
            for tid in resolved:
                if self.get_node_attr(tid, "vocabulary").lower() != 'loinc':
                    continue
                handled.add(tid)
                if tid in eq:
                    matched.add((tid, "Symbolic Match: Equivalent Concept"))
                elif g.has_edge(source, tid) or g.has_edge(tid, source):
                    matched.add((tid, "Symbolic Match: Explicit Mapping"))
                elif self.compare_loinc_axes(source, tid)['is_match']:
                    matched.add((tid, "Symbolic Match: LabTest Properties Overlap"))

        for tid in resolved:
            if tid not in handled and tid in eq:
                matched.add((tid, "Symbolic Match: Equivalent Concept"))
                handled.add(tid)

        for tid in resolved:
            if tid in handled:
                continue
            a = self._allowed_depth(vs, self.get_node_attr(tid, "vocabulary").lower(), k_hop)
            if tid in ancestors and ancestors[tid] <= a:
                matched.add((tid, "Symbolic Match: Ancestor"))
                handled.add(tid)
            elif tid in descendants and descendants[tid] <= a:
                matched.add((tid, "Symbolic Match: Descendant"))
                handled.add(tid)

        # ── Phase 4: Blocking for remaining candidates ──
        ext_depth = k_hop + max_hop
        _tgt_ext_cache = {}

        def _get_tgt_ext_anc(cid):
            if cid in _tgt_ext_cache:
                return _tgt_ext_cache[cid]
            cache_key = (cid, ext_depth)
            if cache_key in self._ancestor_cache:
                _tgt_ext_cache[cid] = self._ancestor_cache[cache_key]
                return _tgt_ext_cache[cid]
            r = frozenset(self._bfs_dir(self._equiv_closure(cid), ext_depth, 'up'))
            self._ancestor_cache[cache_key] = r
            _tgt_ext_cache[cid] = r
            return r

        def is_blocked(tid: int) -> bool:
            try:
                tid = int(tid)
            except (ValueError, TypeError):
                return False
            if tid in eq:
                return False
            if tid in src_ext_anc:
                return True
            ta = _get_tgt_ext_anc(tid)
            return source in ta or bool(eq & ta) or bool(src_ext_anc & ta)

        remaining = set(resolved) - {t for t, _ in matched} - handled
        open_set = set()
        blocked = set()
        for t in remaining:
            if is_blocked(t):
                blocked.add(t)
            else:
                open_set.add(t)

        return matched, blocked, open_set

    def check_concept_match(self, sid: int, tid: int, max_depth: int = 1) -> Tuple[bool, str]:
        try:
            sid, tid = int(sid), int(tid)
        except (ValueError, TypeError):
            return False, "invalid IDs"
        sv, tv = self.get_node_attr(sid, 'vocabulary').lower(), self.get_node_attr(tid, 'vocabulary').lower()
        if sv == 'loinc' and tv == 'loinc':
            r = self.compare_loinc_axes(sid, tid)
            return (True, "LOINC Axis Match") if r['is_match'] else (False, f"LOINC Mismatch: {[m[0] for m in r['mismatched']]}")
        results = self.source_to_targets_paths(sid, [tid], max_depth=max_depth)
        return (True, results[0][1]) if results else (False, "No match")

    # ── Ancestor / sibling utilities ──
    def get_ancestors_fast(self, cid: int, max_hops: int = 3) -> frozenset:
        key = (cid, max_hops)
        if key in self._ancestor_cache:
            return self._ancestor_cache[key]
        if cid not in self.graph:
            self._ancestor_cache[key] = frozenset()
            return frozenset()
        eq = self._equiv_closure(cid)
        anc = self._bfs_dir(eq, max_hops, 'up')
        result = frozenset(anc)
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
        nv, nc = str(row.get('concept_vocabulary', '')).strip().lower(), str(row.get('concept_code', '')).strip().lower()
        if not nv and not nc:
            return False, "not found"
        return (True, "correct") if nv in vocabulary and nc == code else (False, "incorrect")

    def get_vocabulary_stats(self):
        meta = self.graph.graph.get('meta')
        if meta is not None and 'concept_vocabulary' in meta.columns:
            c = meta['concept_vocabulary'].value_counts()
            print(f"Total unique vocabularies: {len(c)}\n\nVocabulary distribution:\n{c}")
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
        use_cols = ["concept_id_1", "concept_id_2", "relationship_id", "concept_vocabulary_1",
                    "concept_vocabulary_2", "concept_name_1", "concept_name_2", "concept_code_1",
                    "concept_code_2", "concept_synonym_1", "concept_synonym_2", "concept_class_1", "concept_class_2"]
        header = pd.read_csv(csv_file_path, nrows=0)
        actual = [c for c in use_cols if c in header.columns]
        df = pd.read_csv(csv_file_path, usecols=actual, dtype=str)
        df['relationship_id'] = df['relationship_id'].str.lower()
        for col in ['concept_vocabulary_1', 'concept_vocabulary_2']:
            if col in df.columns:
                print(f"Unique vocabs: {sorted(df[col].dropna().unique())}")
        loinc_df = df[df['relationship_id'].isin(LOINC_AXIS_RELS)].copy()
        df_e = df[~df['relationship_id'].isin(LOINC_AXIS_RELS)].copy()
        df_e = df_e[df_e['relationship_id'].isin(set(EQ_RELS) | set(DIR_RELS))].copy()
        print(f"LOINC axis: {len(loinc_df):,}, Edge rows: {len(df_e):,}")
        c1 = {c: c[:-2] for c in actual if c.endswith('_1')}
        c2 = {c: c[:-2] for c in actual if c.endswith('_2')}
        all_df = pd.concat([df_e, loinc_df], ignore_index=True)
        fm = pd.concat([all_df[list(c1)].rename(columns=c1), all_df[list(c2)].rename(columns=c2)], ignore_index=True)
        fm['concept_id'] = pd.to_numeric(fm['concept_id'], errors='coerce').fillna(0).astype('int64')
        fm = fm[fm['concept_id'] != 0]
        mcols = ['concept_id', 'concept_vocabulary', 'concept_name', 'concept_synonym', 'concept_code']
        if 'concept_class' in fm.columns:
            mcols.append('concept_class')
        meta = fm.drop_duplicates(subset=['concept_id'])[[c for c in mcols if c in fm.columns]].copy()
        meta.set_index('concept_id', inplace=True)
        meta['concept_vocabulary'] = meta['concept_vocabulary'].fillna("").astype('category')
        self.graph.graph['meta'] = meta
        ci = {}
        for cid, row in meta.iterrows():
            code = str(row.get('concept_code', '')).strip().lower()
            if code and code != 'nan':
                ci[(str(row.get('concept_vocabulary', '')).strip().lower(), code)] = int(cid)
        self.graph.graph['code_index'] = ci
        lam = {}
        loinc_df['u'] = pd.to_numeric(loinc_df['concept_id_1'], errors='coerce').fillna(0).astype('int64')
        loinc_df['v'] = pd.to_numeric(loinc_df['concept_id_2'], errors='coerce').fillna(0).astype('int64')
        for r in loinc_df.itertuples(index=False):
            if r.u and r.v:
                ax = LOINC_AXIS_RELS.get(r.relationship_id)
                if ax:
                    lam.setdefault(r.u, {})[ax] = (r.v, getattr(r, 'concept_name_2', ""))
        self.graph.graph['loinc_axes'] = lam
        self.graph.add_nodes_from(meta.index)
        all_r = sorted(set(EQ_RELS) | set(EQ_RELS.values()) | set(DIR_RELS) | set(DIR_RELS.values()))
        rm = {r: i + 1 for i, r in enumerate(all_r)}
        self.graph.graph['rel_map_rev'] = {i: r for r, i in rm.items()}
        self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None
        df_e['u'] = pd.to_numeric(df_e['concept_id_1'], errors='coerce').fillna(0).astype('int64')
        df_e['v'] = pd.to_numeric(df_e['concept_id_2'], errors='coerce').fillna(0).astype('int64')
        seen, edges = set(), []
        for r in df_e.itertuples(index=False):
            rel, u, v = r.relationship_id, r.u, r.v
            if not u or not v:
                continue
            r1 = rm.get(rel, 0)
            if rel in EQ_RELS:
                r2 = rm.get(EQ_RELS[rel], 0)
                for a, b, ri in [(u, v, r1), (v, u, r2)]:
                    if ri and (a, b, ri) not in seen:
                        edges.append((a, b, {'r': ri}))
                        seen.add((a, b, ri))
            elif rel in DIR_RELS and r1 and (u, v, r1) not in seen:
                edges.append((u, v, {'r': r1}))
                seen.add((u, v, r1))
        self.graph.add_edges_from(edges)

        # ── Build typed adjacency AFTER edges are added ──
        self._build_typed_adjacency()

        self.save_graph(self.output_file)
        print(f"[INFO] Done. Nodes: {self.graph.number_of_nodes():,}, Edges: {self.graph.number_of_edges():,}")

    # def save_graph(self, path):
    #     out = path if path.endswith(".gz") else path + ".gz"
    #     with gzip.open(out, "wb", compresslevel=6) as f:
    #         pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"[INFO] Saved to {out}")
    def save_graph(self, path):
        out = path if path.endswith(".gz") else path + ".gz"
        bundle = {
            'graph': self.graph,
            'isa_succ': self._isa_succ,
            'subs_succ': self._subs_succ,
            'equiv_bidir': self._equiv_bidir,
        }
        with gzip.open(out, "wb", compresslevel=6) as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved to {out}")
    # def load_graph(self, path):
    #         if not path.endswith(".gz") and os.path.exists(path + ".gz"):
    #             path += ".gz"
    #         with gzip.open(path, "rb") as f:
    #             self.graph = pickle.load(f)
    #         self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None
    #         print(f"[INFO] Loaded {path}. Nodes: {self.graph.number_of_nodes()} Edges: {self.graph.number_of_edges()}")
    #         # ── Build typed adjacency on load ──
    #         self._build_typed_adjacency()
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
        else:
            self.graph = data  # backward compat with old pickles
            self._build_typed_adjacency()
        self._IS_A = self._SUBSUMES = self._EQUIV_INTS = None
        print(f"[INFO] Loaded {path}. Nodes: {self.graph.number_of_nodes()} Edges: {self.graph.number_of_edges()}")
    def clear_caches(self):
        """Clear all caches. Useful between experiments to free memory."""
        self._sssp_cache.clear()
        self._sibling_cache.clear()
        self._ancestor_cache.clear()
        self._equiv_cache.clear()
        self._gate_precomp_cache.clear()

    def explain_path(self, source_id: int, target_id: int, max_depth: int = 5) -> dict:
        """
        Return an explainable path between two OMOP concepts.
        
        Returns dict with keys:
            path_type: 'exact_match' | 'equivalence' | 'loinc_axis' | 'ancestor' | 
                    'descendant' | 'sibling' | 'graph_traversal' | 'no_path'
            distance: int (-1 if no path)
            path: list of (concept_id, concept_name, vocabulary) tuples
            edges: list of (source_id, target_id, relationship) tuples
            explanation: str
        """
        src = self.resolve_id(source_id)
        tgt = self.resolve_id(target_id)
        
        def _node_info(cid):
            return (cid, self.get_node_attr(cid, 'name'), self.get_node_attr(cid, 'vocabulary'))
        
        def _no_path(reason):
            return {"path_type": "no_path", "distance": -1, "path": [], "edges": [], "explanation": reason}
        
        def _annotate_path(node_ids):
            path = [_node_info(n) for n in node_ids]
            edges = []
            for i in range(len(node_ids) - 1):
                u, v = node_ids[i], node_ids[i + 1]
                rel = self.get_edge_rel(u, v) or self.get_edge_rel(v, u) or "unknown"
                edges.append((u, v, rel))
            return path, edges
        
        if src is None or tgt is None or src not in self.graph or tgt not in self.graph:
            return _no_path(f"Concept(s) not in graph: src={source_id} tgt={target_id}")
        
        s_info, t_info = _node_info(src), _node_info(tgt)
        
        # 1. Identity
        if src == tgt:
            return {"path_type": "exact_match", "distance": 0, "path": [s_info],
                    "edges": [], "explanation": f"Same concept: {s_info[1]}"}
        
        # 2. Equivalence closure
        eq = self._equiv_closure(src)
        if tgt in eq:
            # BFS through equiv edges to find actual chain
            visited, parent = {src}, {src: None}
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in self._equiv_bidir.get(node, ()):
                    if nb not in visited:
                        visited.add(nb); parent[nb] = node
                        if nb == tgt: break
                        q.append(nb)
                if tgt in parent: break
            chain = []
            cur = tgt
            while cur is not None:
                chain.append(cur); cur = parent.get(cur)
            chain.reverse()
            path, edges = _annotate_path(chain)
            rels = " → ".join(r for _, _, r in edges)
            return {"path_type": "equivalence", "distance": len(chain) - 1,
                    "path": path, "edges": edges,
                    "explanation": f"Equivalent via: {s_info[1]} →[{rels}]→ {t_info[1]}"}
        
        # 3. LOINC axis match
        sv, tv = self.get_node_attr(src, 'vocabulary').lower(), self.get_node_attr(tgt, 'vocabulary').lower()
        sc = self.get_node_attr(src, 'concept_class').lower()
        if sv == 'loinc' and tv == 'loinc' and sc == 'lab test':
            lr = self.compare_loinc_axes(src, tgt)
            if lr.get('is_match'):
                axes = [ax for ax, _ in lr.get('matched', [])]
                return {"path_type": "loinc_axis", "distance": 1,
                        "path": [s_info, t_info], "edges": [],
                        "explanation": f"LOINC axis match on [{', '.join(axes)}]: {s_info[1]} ↔ {t_info[1]}",
                        "loinc_details": lr}
        
        # 4. Ancestor (is-a chain upward)
        ancestors = self._bfs_dir(eq, max_depth, 'up')
        if tgt in ancestors:
            chain = self._trace_directed(src, tgt, 'up', max_depth)
            path, edges = _annotate_path(chain) if chain else ([s_info, t_info], [])
            return {"path_type": "ancestor", "distance": ancestors[tgt],
                    "path": path, "edges": edges,
                    "explanation": f"{s_info[1]} ──is a──▷ {t_info[1]} ({ancestors[tgt]} hops)"}
        
        # 5. Descendant (subsumes chain downward)
        descendants = self._bfs_dir(eq, max_depth, 'down')
        if tgt in descendants:
            chain = self._trace_directed(src, tgt, 'down', max_depth)
            path, edges = _annotate_path(chain) if chain else ([s_info, t_info], [])
            return {"path_type": "descendant", "distance": descendants[tgt],
                    "path": path, "edges": edges,
                    "explanation": f"{t_info[1]} ◁──subsumes── {s_info[1]} ({descendants[tgt]} hops)"}
        
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
                        if r == "is a" and i + 1 < len(edges) and edges[i + 1][2] == "subsumes":
                            pivot = f" via [{self.get_node_attr(v, 'name')}]"
                            break
                rels = " → ".join(r for _, _, r in edges)
                return {"path_type": ptype, "distance": len(raw) - 1,
                        "path": path, "edges": edges,
                        "explanation": f"{'Sibling' if ptype == 'sibling' else 'Connected'}{pivot}: "
                                    f"{s_info[1]} →[{rels}]→ {t_info[1]} ({len(raw)-1} hops)"}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        
        return _no_path(f"No path within {max_depth} hops: {s_info[1]} ↔ {t_info[1]}")


    def _trace_directed(self, src: int, tgt: int, direction: str, max_depth: int) -> List[int]:
        """BFS to recover the actual node chain along is-a or subsumes edges."""
        adj = self._isa_succ if direction == 'up' else self._subs_succ
        eq = self._equiv_closure(src)
        visited = set(eq)
        parent = {n: (src if n != src else None) for n in eq}
        q = deque((n, 0) for n in eq)
        while q:
            node, d = q.popleft()
            if d >= max_depth: continue
            for s in adj.get(node, ()):
                if s not in visited:
                    visited.add(s); parent[s] = node
                    if s == tgt:
                        chain = []
                        cur = tgt
                        while cur is not None:
                            chain.append(cur); cur = parent.get(cur)
                        return chain[::-1]
                    q.append((s, d + 1))
        return [src, tgt]

if __name__ == "__main__":
    start_time = time.time()
    csv_path = "/Users/komalgilani/phd_projects/CohortVarLinker/data/concept_relationship_enriched.csv"
    
    omop_nx = OmopGraphNX(csv_path, output_file='graph_nx.pkl.gz')
    
    target_found  = omop_nx.source_to_targets_paths(3004249, [4236281])
    print(f"target found: {target_found}")
    print(f"path found: {omop_nx.explain_path(3004249, 3004249)}")
    
#     is_sibling_path  = omop_nx.is_sibling_path(3004249, 3004249)
#     print(f"is sibling path: {is_sibling_path}\n\n\n")


#     target_found = omop_nx.source_to_targets_paths(21601784, [42536050], max_depth=1)
#     print(f"target found: {target_found}")
#     print(f"path found: {omop_nx.explain_path(21601784, 42536050)}")
#     is_sibling_path  = omop_nx.is_sibling_path(21601784, 42536050)
#     print(f"is sibling path: {is_sibling_path}")