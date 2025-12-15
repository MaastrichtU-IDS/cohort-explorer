import time
import networkx as nx
import pandas as pd
import pickle
import os
from typing import List,Set, Tuple

from collections import deque, OrderedDict

class OmopGraphNX:
    """
    Builds a bidirectional graph from an OMOP concept_relationship CSV using networkx.
    Provides fast path searches and supports adding direct (inferred) shortcut edges.
    """
    def __init__(self, csv_file_path=None, output_file='graph_nx.pkl'):
        """
        :param csv_file_path: Path to the OMOP concept_relationship CSV.
        :param output_file: Filename to save/load the networkx graph. Defaults to 'data/graph_nx.pkl'.
        """
        from CohortVarLinker.src.config import settings
        self.csv_file_path = csv_file_path
        # Use data_folder for the default location of graph_nx.pkl
        if output_file is None:
            output_file = os.path.join(settings.data_folder, "graph_nx.pkl")
        elif not os.path.isabs(output_file):
            output_file = os.path.join(settings.data_folder, output_file)
        self.output_file = output_file
        self.graph = nx.DiGraph()
        if os.path.exists(self.output_file):
            print(f"[INFO] Loading graph from {self.output_file}.")
            self.load_graph(self.output_file)
        else:
            print(f"graph file does not exist at {self.output_file}")
            self.build_graph(csv_file_path, force_rebuild=True)

        self._sssp_cache = OrderedDict()
        self._sssp_cache_max = 20000   # tune (500–20000 typically OK)

    def build_graph(self, csv_file_path=None, force_rebuild=False):
        """
        Reads the CSV and builds a bidirectional networkx graph.
        Concept IDs are stored as integers.
        """
        if self.graph and not force_rebuild and self.graph.number_of_nodes() > 0:
            print("[INFO] Graph is already loaded. Skipping rebuild.")
            return

        csv_file_path = csv_file_path or self.csv_file_path
        if not csv_file_path:
            raise ValueError("No CSV file path provided.")
        
        # Read CSV into DataFrame
        df = pd.read_csv(csv_file_path, dtype=str)
        print(df.head())
        print("Unique relationship_ids:", df['relationship_id'].unique())
        
        # Convert concept IDs to integers
        df['concept_id_1'] = pd.to_numeric(df['concept_id_1'], errors='coerce', downcast='integer')
        df['concept_id_2'] = pd.to_numeric(df['concept_id_2'], errors='coerce', downcast='integer')
        df = df.dropna(subset=['concept_id_1', 'concept_id_2'])
        df['concept_id_1'] = df['concept_id_1'].astype(int)
        df['concept_id_2'] = df['concept_id_2'].astype(int)
       
        # Define relationship types
        eq_relationships = {
            'rxnorm - atc pr lat': 'atc - rxnorm pr lat',
            'atc - rxnorm pr lat': 'rxnorm - atc pr lat',
            'atc - rxnorm': 'rxnorm - atc',
            'rxnorm - atc': 'atc - rxnorm',
            'snomed - rxnorm eq': 'rxnorm - snomed eq',
            'rxnorm - snomed eq': 'snomed - rxnorm eq',
            'mapped from': 'maps to',
            'maps to': 'mapped from',
            'component of': 'has component',
            'has component': 'component of',
            
            # 'cpt4 - loinc eq': 'loinc - cpt4 eq',
            # 'loinc - cpt4 eq': 'cpt4 - loinc eq',
            # 'cpt4 - snomed eq': 'snomed - cpt4 eq',
            # 'snomed - cpt4 eq': 'cpt4 - snomed eq',
            # 'atc - snomed eq': 'snomed - atc eq',
            # 'snomed - atc eq': 'atc - snomed eq',
        }
   
        directed_hierarchical = {
            'is a': 'subsumes',
            'subsumes': 'is a',
            'has answer': 'answer of',
            'answer of':'has answer'
        }
        # directed_others_ = {
        #     'has disposition': 'is disposition of',
        #     'is disposition of': 'has disposition'
        # } concept_id_1,concept_id_2,relationship_id,valid_start_date,valid_end_date,invalid_reason,concept_name_1,concept_1_vocabulary,concept_1_domain,concept_1_concept_class,concept_name_2,concept_2_vocabulary,concept_2_domain,concept_2_concept_class

        for _, row in df.iterrows():
            c1 = row['concept_id_1']
            c2 = row['concept_id_2']
            concept_1_name = (row['concept_name_1']).strip().lower() if pd.notna(row['concept_name_1']) else ""
            concept_2_name = (row['concept_name_2']).strip().lower() if pd.notna(row['concept_name_2']) else ""
            rel_id = (row['relationship_id']).strip().lower() if pd.notna(row['relationship_id']) else ""
            vocab1 = (row['concept_1_vocabulary']).strip().lower() if pd.notna(row['concept_1_vocabulary']) else ""
            vocab2 = (row['concept_2_vocabulary']).strip().lower() if pd.notna(row['concept_2_vocabulary']) else ""
            concept_code1 = (row['concept_code_1']).strip().lower() if pd.notna(row['concept_code_1']) else ""
            concept_code2 = (row['concept_code_2']).strip().lower() if pd.notna(row['concept_code_2']) else ""
            domain1 = (row['concept_1_domain']).strip().lower() if pd.notna(row['concept_1_domain']) else ""
            domain2 = (row['concept_2_domain']).strip().lower() if pd.notna(row['concept_2_domain']) else ""
            concept_class1 = (row['concept_1_concept_class']).strip().lower() if pd.notna(row['concept_1_concept_class']) else ""
            concept_class2 = (row['concept_2_concept_class']).strip().lower() if pd.notna(row['concept_2_concept_class']) else ""
            if rel_id in eq_relationships:
                rel_id_ = f"{rel_id} eq"
                inv_rel_ = f"{eq_relationships[rel_id]} eq"
                if not self.graph.has_edge(c1, c2):
                    self.graph.add_edge(c1, c2, relation=rel_id_)
                if not self.graph.has_edge(c2, c1):
                    self.graph.add_edge(c2, c1, relation=inv_rel_)
            elif rel_id in directed_hierarchical:
                inv_rel = directed_hierarchical[rel_id]
                if not self.graph.has_edge(c1, c2):
                    self.graph.add_edge(c1, c2, relation=rel_id)
                if not self.graph.has_edge(c2, c1):
                        self.graph.add_edge(c2, c1, relation=inv_rel)
            self.graph.add_node(c1,
                    concept_id=c1,
                    concept_name=concept_1_name,
                    concept_code=concept_code1,
                    domain=domain1,
                    vocabulary=vocab1,
                    
                    concept_class=concept_class1
                )

            self.graph.add_node(c2,
                    concept_id=c2,
                    concept_name=concept_2_name,
                    concept_code=concept_code2,
                    domain=domain2,
                    vocabulary=vocab2,
                    concept_class=concept_class2
                )
        self.save_graph(self.output_file)

        print(f"[INFO] Graph built successfully with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _sssp_lengths(self, start: int, cutoff: int = 3) -> dict[int, int]:
        """
        Distances in number of edges from start to nodes within cutoff.
        Cached by (start, cutoff).
        """
        key = (start, cutoff)
        hit = self._sssp_cache.get(key)
        if hit is not None:
            self._sssp_cache.move_to_end(key)
            return hit

        dist = dict(nx.single_source_shortest_path_length(self.graph, start, cutoff=cutoff))
        self._sssp_cache[key] = dist
        if len(self._sssp_cache) > self._sssp_cache_max:
            self._sssp_cache.popitem(last=False)
        return dist
    def source_to_targets_paths( self, start, target_ids,  max_depth: int = 1, domain: str = "drug",
    ) -> List[Tuple[int, str]]:
        try:
            start = int(start)
        except (TypeError, ValueError):
            return []

        if start not in self.graph:
            return []

        # target_ids coming from your pipeline are already ints; keep conversion cheap anyway
        targets = set()
        for ti in target_ids:
            try:
                ti = int(ti)
            except (TypeError, ValueError):
                continue
            if ti in self.graph and ti != start:
                targets.add(ti)

        if not targets:
            return []

        start_node = self.graph.nodes[start]
        vocab_start = (start_node.get("vocabulary") or "").lower()
        domain_lower = (domain or "").lower()

        # IMPORTANT: cutoff must cover your largest "allowed" (you sometimes allow +3)
        cutoff = max_depth + 2
        dists = self._sssp_lengths(start, cutoff=cutoff)

        results: List[Tuple[int, str]] = []

        # Iterate reached nodes (small) instead of iterating targets (huge)
        for tid, dist_edges in dists.items():
            if tid not in targets:
                continue

            node_t = self.graph.nodes[tid]
            vocab_goal = (node_t.get("vocabulary") or "").lower()

            # --- allowed depth in EDGES ---
            allowed = max_depth
            if "drug" not in domain_lower:
                allowed = max_depth
            else:
                if vocab_start != vocab_goal:
                    if (vocab_start, vocab_goal) in {("atc", "rxnorm"), ("rxnorm", "atc")}:
                        allowed = max_depth + 1
                    elif (vocab_start, vocab_goal) in {("snomed", "atc"), ("atc", "snomed")}:
                        allowed = max_depth + 2
                    elif (vocab_start, vocab_goal) in {("snomed", "rxnorm"), ("rxnorm", "snomed")}:
                        allowed = max_depth + 3
                    else:
                        allowed = max_depth + 1
                else:
                    allowed = max_depth + 1

            if dist_edges > allowed:
                continue

            # --- mapping_relation ---
            # For multi-hop matches, you currently treat as related anyway.
            match_relation = "skos:relatedMatch"
            
            if dist_edges == 1:
                edge_data = self.graph.get_edge_data(start, tid) or {}
                rel = (edge_data.get("relation") or "").lower()
                if "subsumes" in rel:
                    match_relation = "skos:narrowMatch"
                elif "is a" in rel:
                    match_relation = "skos:broadMatch"
                elif "eq" in rel:
                    match_relation = "skos:exactMatch"
                
            else:
                if vocab_goal in {"rxnorm", "atc", "snomed"} and vocab_start in {"rxnorm", "atc", "snomed"}:
                    match_relation = "skos:closeMatch"

            results.append((tid, match_relation))

        return results

    def bfs_all_reachable_subsumes(self, start, max_depth=5):
        """
        Return all nodes reachable from `start` in self.graph by following only downward edges 
        that are labeled "subsumes". These nodes are those that are subsumed by the start node 
        (e.g. its children, grandchildren, etc.).

        Parameters:
        -----------
        start : int
            The starting concept ID.
        max_depth : int, optional (default=5)
            The maximum depth for the search.

        Returns:
        --------
        Set[int]
            A set of nodes reachable from the start via "is a" edges.
        """
        if start not in self.graph:
            # print(f"[WARN] Node {start} does not exist in the graph.")
            return set()

        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # For downward traversal, use predecessors (children)
            for neighbor in self.graph.predecessors(current):
                rel = self.graph.get_edge_data(neighbor, current).get("relation", "")
                if rel in {"is a"} and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Optionally, remove the start node from the returned set if you only want its subsumed nodes
        visited.remove(start)
        # print(f"len of nodes (subsumed by {start}) is {len(visited)}")
        return visited

    def bfs_all_reachable(self, start):
        """
        Return all nodes reachable from `start` in self.graph (a NetworkX Graph/DiGraph).
        """
        if start not in self.graph:
            # print(f"[WARN] Node {start} does not exist in the graph.")
            return set()
        # nx.bfs_tree(...) returns a subgraph (tree) containing nodes reachable from `start`.
        bfs_tree = nx.bfs_tree(self.graph, start, depth_limit=5)
        nodes = set(bfs_tree.nodes())
        # print(f"len of nodes is {len(nodes)}")
        return nodes
    


    
    
   
    def bfs_upward_reachable(self, start, target_ids, max_depth=5):
        """
        Traverse only upward (child -> parent) edges from `start`, and return reachable targets.
        """
        visited = set()
        queue = deque([(start, 0)])
        reachable_targets = set()
        target_ids = set(target_ids)

        while queue:
            current, depth = queue.popleft()
            if depth > max_depth or current in visited:
                continue

            visited.add(current)
            if current in target_ids:
                reachable_targets.add(current)

            # Follow outgoing edges: child -> parent
            if current in self.graph:
                for neighbor in self.graph.successors(current):
                    rel = self.graph.get_edge_data(current, neighbor).get("relation", "")
                    if rel in {"is a"}:  # or just "is a"
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))

        return list(reachable_targets)

    def bfs_downward_reachable(self, start, target_ids, max_depth=5):
        visited = set()
        queue = deque([(start, 0)])
        reachable_targets = set()
        target_ids = set(target_ids)
        # print(f"target ids are {target_ids}")
        while queue:
            current, depth = queue.popleft()
            if depth > max_depth or current in visited:
                continue

            visited.add(current)

            if current in target_ids and current != start:
                # print(f"[MATCH] Found target: {current} at depth {depth}")
                reachable_targets.add(current)

            if current in self.graph:
                for neighbor in self.graph.predecessors(current):  # children
                    rel = self.graph.get_edge_data(current, neighbor).get("relation", "")
                    # print(f"rel is {rel}")
                    if rel in {"subsumes"}:  # or "subsumes"
                        if neighbor not in visited:
                            # if neighbor == 42539487:
                            #     # print(f"adding neighbor {neighbor} to queue")
                            queue.append((neighbor, depth + 1))

        return list(reachable_targets)

    def only_upward_or_downward(self, start, target_ids, max_depth=5) -> set:
        upward = self.bfs_upward_reachable(start, target_ids, max_depth)
        downward = self.bfs_downward_reachable(start, target_ids, max_depth)
        equivalents = self.get_equivalent_nodes(start, max_depth=1)
        siblings = self.get_share_components(start, target_ids, max_depth=max_depth)
        # print(f"upward is {upward} and downward is {downward} and equivalents are {equivalents}")
        reachable_target_ids = set(upward) | set(downward) | set(equivalents) | set(siblings)
        reachable_target_ids = reachable_target_ids.intersection(target_ids)
        return reachable_target_ids
    
    def get_equivalent_nodes(self, node, max_depth=3):
        """
        Recursively find all equivalent nodes to the given node via 'eq' relationships (multi-hop).
        """
        try:
            node = int(node)
        except:
            # print(f"[ERROR] Node {node} is not an integer.")
            return set()

        if node not in self.graph:
            return set()

        visited = set()
        queue = deque([(node, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth > max_depth or current in visited:
                continue
            visited.add(current)

            neighbors = list(self.graph.successors(current)) + list(self.graph.predecessors(current))
            for neighbor in neighbors:
                if neighbor not in visited:
                    rel_data = self.graph.get_edge_data(current, neighbor) or self.graph.get_edge_data(neighbor, current)
                    if rel_data and "eq" in rel_data.get("relation", "").lower():  
                        
                        # further check if domain is drug of node or target than check they have atc as vocabulary if atc is the vocab then it should not have concept_class lower than 4th for atleast one of them 
                        
                        
                        queue.append((neighbor, depth + 1))

        visited.discard(node)  # Remove the original node
        return visited

    
    def get_direct_equivalents(self, node: int) -> set[int]:
        """Direct neighbors connected via an 'eq' relationship."""
        if node not in self.graph:
            return set()

        eq_neighbors = set()

        # outgoing
        for nb in self.graph.successors(node):
            rel = (self.graph.get_edge_data(node, nb) or {}).get("relation", "").lower()
            if "eq" in rel:
                eq_neighbors.add(nb)

        # incoming
        for nb in self.graph.predecessors(node):
            rel = (self.graph.get_edge_data(nb, node) or {}).get("relation", "").lower()
            if "eq" in rel:
                eq_neighbors.add(nb)

        return eq_neighbors
            
                        
        
       


    def bfs_upward_with_equivalences(self, start, candidate_ids, max_depth=5):
        """
        Like bfs_upward_reachable, but at each level we also consider equivalent nodes
        (via 'eq' relationships) so that cross-vocabulary concepts can be treated as parents too.
        
        Returns a set of candidate_ids that were reached by traversing upward 
        (or their equivalents).
        """
        if start not in self.graph:
            return set()
        
        candidate_ids = set(candidate_ids)
        visited = set()
        queue = deque()
        
        # Start from 'start' plus its equivalents
        start_equivs = self.get_equivalent_nodes(start, max_depth=1) | {start}
        for eqnode in start_equivs:
            queue.append((eqnode, 0))
            visited.add(eqnode)
        
        
        found_parents = set()
        
        while queue:
            current, depth = queue.popleft()
            if depth > max_depth:
                continue
            
            # If current is in candidate_ids (and not the original start), record it
            #   as an upward 'parent' (broadly speaking).
            if current in candidate_ids and current != start:
                found_parents.add(current)
            
            if current in self.graph:
                # Move upward: child -> parent edges
                for parent_node in self.graph.successors(current):
                    edge_rel = self.graph.get_edge_data(current, parent_node).get("relation", "")
                    if edge_rel in {"is a"}:  # or "is_a"/"isa" if your data uses that
                        if parent_node not in visited:
                            visited.add(parent_node)
                            queue.append((parent_node, depth + 1))

                        # Also consider the parent's equivalents
                        parent_equivs = self.get_equivalent_nodes(parent_node, max_depth=1)
                        for peq in parent_equivs:
                            if peq not in visited:
                                visited.add(peq)
                                queue.append((peq, depth + 1))
        
        return found_parents


    def bfs_downward_with_equivalences(self, start, candidate_ids, max_depth=5):
        """
        Like bfs_downward_reachable, but also expands each node by its equivalent concepts 
        at each step.
        """
        if start not in self.graph:
            return set()
        
        candidate_ids = set(candidate_ids)
        visited = set()
        queue = deque()

        # Start with 'start' plus its equivalents
        start_equivs = self.get_equivalent_nodes(start, max_depth=1) | {start}
        for eqnode in start_equivs:
            queue.append((eqnode, 0))
            visited.add(eqnode)

        found_children = set()
        
        while queue:
            current, depth = queue.popleft()
            if depth > max_depth:
                continue
            
            if current in candidate_ids and current != start:
                found_children.add(current)

            if current in self.graph:
                # Move downward: parent -> child edges
                for child_node in self.graph.predecessors(current):
                    edge_rel = self.graph.get_edge_data(child_node, current).get("relation", "")
                    if edge_rel in {"subsumes"}:  # or your chosen relationship
                        if child_node not in visited:
                            visited.add(child_node)
                            queue.append((child_node, depth + 1))
                        
                        # Also consider child's equivalents
                        child_equivs = self.get_equivalent_nodes(child_node, max_depth=1)
                        for ceq in child_equivs:
                            if ceq not in visited:
                                visited.add(ceq)
                                queue.append((ceq, depth + 1))
        
        return found_children

    def bfs_bidirectional_reachable(self, start, target_ids, max_depth=3, domain='drug'):
        """
        Enhanced BFS that traverses both directions and uses equivalence relationships to reach targets.
        """
        if start not in self.graph:
            # print(f"[WARN] Node {start} does not exist in the graph.")
            return []
        reachable_target_ids = set()
        target_ids = set(target_ids)

        # Include target equivalents
        target_equiv_map = {
            tid: self.get_direct_equivalents(tid) | {tid}
            for tid in target_ids
        }
        all_target_equivs = set().union(*target_equiv_map.values())
        # print(f"all target equivalents are {all_target_equivs}")
        # Step 1: BFS Upward from start
        upward = self.bfs_upward_reachable(start, all_target_equivs, max_depth)
        # print(f"upward reachable nodes are {upward}")
        reachable_target_ids.update(upward)

        # Step 2: BFS Downward from start
        downward = self.bfs_downward_reachable(start, all_target_equivs, max_depth)
        # print(f"downward reachable nodes are {downward}")
        reachable_target_ids.update(downward)
        
        # ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
        # STEP 2 b – *sibling* search via common ancestors  ◄─ NEW CODE
        # ▒▒▒▒ find every ancestor of `start` (≤ max_depth)
        ancestors = self.get_all_parents(start, max_depth=max_depth)
        # print(f"len of ancestors is {len(ancestors)} and ancestors are {ancestors}")
        for anc in ancestors:
            # print(f"anc is {anc}")
            for child in self.graph.successors(anc):        # direct children = siblings/cousins
                # print(f"child is {child}")
                if child == start or self.all_below_atc_4th(start, child, anc):
                    #  print(f"skipping {child} as it is start node or below atc 4th level")
                     continue
                elif child in all_target_equivs:
                    # print(f"[SIBLING-MATCH] target {child} reachable via sibling {anc}")
                    reachable_target_ids.add(child)
                else:
                    # try 1-hop equivalents of that child
                    equivs = self.get_equivalent_nodes(child, max_depth=1)
                    common = equivs & all_target_equivs
                    # if common:
                        # print(f"[SIBLING-MATCH] target {child} reachable via sibling {anc} and its equivalents")
                        # print(f"common is {common}")
                    reachable_target_ids.update(common)
                
        # Step 3: Try equivalents of target_ids
        for tid, equivs in target_equiv_map.items():
            for eq in equivs:
                if eq == tid:
                    continue
                # Can we reach this equivalent from start?
                up = self.bfs_upward_reachable(start, [eq], max_depth)
                down = self.bfs_downward_reachable(start, [eq], max_depth)
                if up or down:
                    # print(f"[EQUIV-MATCH] target {tid} reachable via equivalent {eq}")
                    reachable_target_ids.add(tid)
                    break

        # Step 4: Try equivalents of the start node
        start_equivs = self.get_equivalent_nodes(start, max_depth=1)
        for equiv in start_equivs:
            up = self.bfs_upward_reachable(equiv, all_target_equivs, max_depth)
            down = self.bfs_downward_reachable(equiv, all_target_equivs, max_depth)
            reachable_target_ids.update(up)
            reachable_target_ids.update(down)

        if reachable_target_ids == target_ids:
            return list(reachable_target_ids.intersection(target_ids))
        
        # Step 5: Traverse downward from start and check for equivalence to targets
        downward_children = self.bfs_all_reachable_subsumes(start, max_depth=max_depth)
        # print(f"downward children are {downward_children}")
        for child in downward_children:
            equivalents = self.get_equivalent_nodes(child, max_depth=1)
            if equivalents.intersection(target_ids):
                # print(f"[DESC-EQUIV-MATCH] {child} maps to one of {target_ids} via equivalence")
                reachable_target_ids.update(equivalents.intersection(target_ids))
        
        # only return target_ids that are reachable
        reachable_target_ids = reachable_target_ids.intersection(target_ids)
        
        
        if start in self.graph and domain == 'drug':
            filtered = set()
            for tid in reachable_target_ids:
                if self._is_valid_drug_graph_match(start, tid):
                    filtered.add(tid)
            reachable_target_ids = filtered

        # return list(reachable_target_ids)
        return list(reachable_target_ids)  # Remove the start node if present

    def concept_exists(self, concept_id: int, concept_code:str, vocabulary:List[str]) -> Tuple[bool, str]:
        """
        Check if a concept_id exists in the graph.
        """
        # given the concept code and vocabulary check if it exists in the graph
        vocabulary = [v.lower() for v in vocabulary]
        concept_code = concept_code.lower()
        # print(f"value of concept_id is {concept_id}, concept_code is {concept_code} and vocabulary is {vocabulary}")
        if concept_id in self.graph:
            node_data = self.graph.nodes[concept_id]
            # print(f"node data is {node_data}")
            node_vocab = node_data.get("vocabulary", "").strip().lower()
            node_concept_code = node_data.get("concept_code", "").strip().lower()
          
            if node_vocab == '' and node_concept_code == '': 
                return False, "not found"
            elif node_vocab in vocabulary and node_concept_code == concept_code:
                return True, "correct"
            else:
                return False, "incorrect"

        return False, "not found"
        

    def all_below_atc_4th(self, node1: int, node2: int, common_parent: int) -> bool:
        """
        Returns True if both node1 and node2 are ATC siblings at levels 1st, 2nd, 3rd,
        and should therefore be skipped for matching (too general).
        """
        edge1 = self.graph.nodes[node1]
        edge2 = self.graph.nodes[node2]
       # print(f"edge1 is {edge1} and edge2 is {edge2}")
        if not edge1 or not edge2:
            return False  # Not enough data to assess

        cc1 = edge1.get("concept_class", "").lower() if edge1 else ""
        cc2 = edge2.get("concept_class", "").lower()
        vocab1 = edge1.get("vocabulary", "").lower()
        vocab2 = edge2.get("vocabulary", "").lower()
        
        atc_list = {"atc 1st", "atc 2nd", "atc 3rd", "atc 4th"}
       #print(f"cc1 is {cc1} and cc2 is {cc2}") 
        
        flag = (vocab1 == "atc" and vocab2 == "atc" and cc1 in atc_list and cc2 in atc_list)
        flag_2 =  cc1 == cc2 and cc1 == "atc 4th" 
       # print(f"is {node1} and {node2} below atc 4th level? {flag} and {flag_2}")
        return flag or flag_2

    # def bfs_path(self, start, target_ids, max_depth=15):
    #     """
    #     Return all nodes reachable from `start` in self.graph and check if target_ids are reachable.
    #     """
    #     if start not in self.graph:
    #         # print(f"[WARN] Node {start} does not exist in the graph.")
    #         return []
        
    #     # Get all reachable nodes using shortest path lengths
    #     reachable_nodes = nx.single_source_shortest_path_length(self.graph, start,cutoff=max_depth)
    #     # reachable_targets = {target for target in target_ids 
    #     #                  if target in reachable_nodes and reachable_nodes[target] < max_depth}
    #     reachable_targets = {target for target in target_ids 
    #                      if target in reachable_nodes}
    #     # Find which target IDs are reachable
    #     # reachable_targets = target_ids.intersection(reachable_nodes)

    #     # print(f"len of reachable nodes is {len(reachable_nodes)}")
    #     # print(f"reachable target ids via graph path is {len(reachable_targets)}")
    #     # reachable_targets_with_depth = {
    #     #         target: reachable_nodes[target] 
    #     #         for target in target_ids 
    #     #         if target in reachable_nodes
    #     #     }
    #     return list(reachable_targets)
    #     # return reachable_targets_with_depth

    
    def bfs_path_exists(self, start, target_ids,  max_depth=3):
        """
        Uses networkx to check if a path exists between start and goal.
        Optionally, adds an inferred edge (shortcut) if a path is found.
        
        :param start: integer or string concept_id.
        :param goal: integer or string concept_id.
        :param add_inferred: if True, adds a direct edge when a path is found.
        :return: True if a path exists, else False.
        """
        start = int(start)
        # target_ids = [int(t) for t in target_ids]
        path = None
        reachable_targets = dict()
        # is_match = False
        if start not in self.graph:
            # print(f"[WARN] One of the nodes {start} or {target_ids} does not exist in the graph.")
            return []
        else:
            if start not in target_ids:
                try:
                    # networkx.shortest_path uses efficient algorithms.
                    for goal in target_ids:
                        if goal  in self.graph:
                            if (start, goal) in self._path_cache and self._path_cache[(start, goal)] is not None:
                                print(f"[CACHE HIT] Using cached path for {start} -> {goal}")
                                path = self._path_cache[(start, goal)]
                            else:
                                
                                path = nx.shortest_path(self.graph, source=start, target=goal)
                                print(f"[PATH FOUND] Path from {start} to {goal}: {path}")
                                self._path_cache[(start, goal)] = path
                            if len(path) <= max_depth:
                                reachable_targets[goal] = path
                                    
                           
                except nx.NetworkXNoPath:
                    pass
        return reachable_targets



    def get_share_components(self, start: int, target_ids: List[int], max_depth: int = 3):
        """
        Identifies if `start` and any of the `target_ids` share a component based on
        'has component' or 'component of' relationships in the graph.

        Parameters:
        -----------
        start : int
            The source concept_id (typically a lab test or similar).
        target_ids : List[int]
            List of concept_ids to check for shared components.
        max_depth : int
            Maximum depth to traverse for components.

        Returns:
        --------
        List[dict]
            List of dictionaries for each target with shared component info:
            [{'target': 123, 'shared_component': 456}, ...]
        """
        if start not in self.graph:
            # print(f"[WARN] Start node {start} not in graph.")
            return []

        target_ids = set(target_ids)
        shared_results = []

        def get_components(node: int, max_depth=3) -> Set[int]:
            visited = set()
            queue = deque([(node, 0)])
            components = set()

            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                for neighbor in self.graph.successors(current):  # e.g., has component
                    rel = self.graph.get_edge_data(current, neighbor).get("relation", "")
                    if rel in {"has component"} and neighbor not in visited:
                        visited.add(neighbor)
                        components.add(neighbor)
                        queue.append((neighbor, depth + 1))

                for neighbor in self.graph.predecessors(current):  # e.g., component of
                    rel = self.graph.get_edge_data(neighbor, current).get("relation", "")
                    if rel in {"component of"} and neighbor not in visited:
                        visited.add(neighbor)
                        components.add(neighbor)
                        queue.append((neighbor, depth + 1))

            return components

        # Get components of the start node
        start_components = get_components(start, max_depth=max_depth)

        for tid in target_ids:
            if tid not in self.graph:
                continue
            target_components = get_components(tid, max_depth=max_depth)
            common = start_components & target_components
            for c in common:
                shared_results.append({"target": tid, "shared_component": c})

        return shared_results

    
    def get_all_parents(self, concept_id: int, max_depth: int = 5) -> Set[int]:
        """
        Return the set of all ancestors/parents of `concept_id` in the `graph`,
        within `max_depth` levels, following "is a" edges upward.
        
        Assumes the graph is a DiGraph where an edge (child -> parent)
        has edge attribute "relation" == "is a".
        """
        if concept_id not in self.graph:
            return set()

        visited = set()
        queue = deque([(concept_id, 0)])
        
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # For "upward" edges in a DiGraph: child -> parent
            # => we look at graph.successors(child) to find its parents in your data model
            for parent in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, parent) or {}
                if edge_data.get("relation") == "is a":
                    if parent not in visited:
                        visited.add(parent)
                        queue.append((parent, depth + 1))
        
        return visited


    def find_sibling_targets(
        self, 
        start_concept: int, 
        target_ids: List[int], 
        max_depth: int = 2
    ) -> List[int]:
        """
        Returns all target IDs that share at least one parent
        with the `start_concept` (up to `max_depth` levels up).
        
        :param omop_nx: an instance of OmopGraphNX or similar with a `graph` attribute (networkx DiGraph).
        :param start_concept: integer OMOP ID of the "start" concept
        :param target_ids: list of integer OMOP IDs
        :param max_depth: how many levels upward to search
        :return: list of target IDs that are siblings (i.e., share ≥1 parent with `start_concept`)
        """
        # 1. Gather all parents of start_concept
        start_parents = self.get_all_parents(start_concept, max_depth=max_depth)

        siblings = []
        for tid in target_ids:
            # 2. Gather all parents for the target concept
            t_parents = self.get_all_parents(tid, max_depth=max_depth)
            # 3. Check if there's any overlap
            same_parents = start_parents.intersection(t_parents)
            if same_parents:
                siblings.append({"sibling":tid, "parents":same_parents})
                # siblings.append(tid)
        
        return siblings
  
    def add_inferred_edge(self, start, goal):
        """
        Adds a direct (inferred) edge between start and goal.
        """
        self.graph.add_edge(start, goal)
        # print(f"[INFO] Inferred edge added: {start} <-> {goal}")
        # Optionally, you can persist this update:
        self.save_graph(self.output_file)

    def save_graph(self, pickle_file='graph_nx.pkl'):
        """
        Save the networkx graph to disk.
        """
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Graph saved to {pickle_file}.")

    def load_graph(self, pickle_file='graph_nx.pkl'):
        """
        Load the networkx graph from disk.
        """
        with open(pickle_file, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"[INFO] Graph loaded from {pickle_file}. Contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def a_star_search(self, start, goal):
        try:
            path = nx.astar_path(self.graph, start, goal, heuristic=self.heuristic)
            return True, path
        except nx.NetworkXNoPath:
            return False, []
        
        
    

    def _is_valid_drug_graph_match(self, start: int, target: int) -> bool:
        """
        Extra guardrail for drug-domain concepts.

        - For ATC–ATC: only allow if same ATC level (same concept_class) and
          not a very broad level (1st–3rd).
        - For ATC–RxNorm: only allow if they are directly equivalent via an 'eq' edge.
        - For other domains / vocabularies: keep old behaviour (return True).
        """
        if start not in self.graph or target not in self.graph:
            return False

        s = self.graph.nodes[start]
        t = self.graph.nodes[target]

        domain1 = (s.get("domain") or "").lower()
        domain2 = (t.get("domain") or "").lower()
        vocab1 = (s.get("vocabulary") or "").lower()
        vocab2 = (t.get("vocabulary") or "").lower()
        cc1 = (s.get("concept_class") or "").lower()
        cc2 = (t.get("concept_class") or "").lower()

        # Only constrain when both are drug-domain concepts.
        if  "drug" not in domain1 or "drug" not in domain2:
            return True

        # ---------- ATC–ATC ----------
        if vocab1 == "atc" and vocab2 == "atc":
            print(f"cc1 is {cc1} and cc2 is {cc2}")
            broad_levels = {"atc 1st", "atc 2nd", "atc 3rd"}
            # Don't match very broad ATC levels at all.
            if cc1 in broad_levels or cc2 in broad_levels:
                return False

            # Only allow same ATC level (no parent–child matches).
            if cc1 != cc2:
                return False

            # At this point: both ATC, same (non-broad) level.
            return True

        # ---------- ATC–RxNorm (cross-vocabulary) ----------
        elif (vocab1 == "atc" and vocab2 == "rxnorm") or (vocab1 == "rxnorm" and vocab2 == "atc"):
            print(f"checking ATC–RxNorm match between {start} and {target} and eq notes are {self.get_equivalent_nodes(start, max_depth=1)} and {self.get_equivalent_nodes(target, max_depth=1)}")
       
            return True
        return False


# if __name__ == "__main__":
#     csv_path = "/Users/komalgilani/Documents/GitHub/CohortVarLinker/data/concept_relationship_enriched.csv"
    
#     omop_nx = OmopGraphNX(csv_path)
#     omop_nx.build_graph()
    
#     start_time = time.time()
#     start_concept = 3027018   
#     end_concept = [40771525] 
#     reachable_targets = omop_nx.source_to_targets_paths(21601855, [42536050], max_depth=1, domain='drug')
#     print(f"Reachable targets: {reachable_targets}")
#     reachable_targets = omop_nx.source_to_targets_paths(4306037, [21601682], max_depth=1, domain='drug')
#     print(f"Reachable targets: {reachable_targets}") 
#     reachable_targets = omop_nx.source_to_targets_paths(4306037, [21601682], max_depth=1, domain='drug')
#     print(f"Reachable targets: {reachable_targets}") 
#     reachable_targets = omop_nx.source_to_targets_paths(21601855, [42536050], max_depth=1, domain='drug')
#     print(f"Reachable targets: {reachable_targets}") 
 
