import time
import networkx as nx
import pandas as pd
import pickle
import os
from typing import List,Set

from collections import deque

class OmopGraphNX:
    """
    Builds a bidirectional graph from an OMOP concept_relationship CSV using networkx.
    Provides fast path searches and supports adding direct (inferred) shortcut edges.
    """
    def __init__(self, csv_file_path=None, output_file='data/graph_nx.pkl'):
        """
        :param csv_file_path: Path to the OMOP concept_relationship CSV.
        :param output_file: Filename to save/load the networkx graph. Defaults to 'data/graph_nx.pkl'.
        """
        self.csv_file_path = csv_file_path
        self.output_file = output_file
        self.graph = nx.DiGraph() # networkx Graph is undirected (bidirectional by default)
        # Always check for the graph in the data folder by default
        if os.path.exists(self.output_file):
            print(f"[INFO] Loading graph from {self.output_file}.")
            self.load_graph(self.output_file)
        else:
            print(f"graph file does not exist at {self.output_file}")
            self.build_graph(csv_file_path, force_rebuild=True)

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
            'mapped from': 'maps to',
            'maps to': 'mapped from',
            'component of': 'has component',
            'has component': 'component of',
            # 'atc - snomed eq': 'snomed - atc eq',
            # 'snomed - atc eq': 'atc - snomed eq',
        }
   
        directed_child_to_parent = {
            'is a': 'subsumes',
            'subsumes': 'is a'
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
            elif rel_id in directed_child_to_parent:
                inv_rel = directed_child_to_parent[rel_id]
                if not self.graph.has_edge(c1, c2):
                    self.graph.add_edge(c1, c2, relation=rel_id)
                if not self.graph.has_edge(c2, c1):
                        self.graph.add_edge(c2, c1, relation=inv_rel)
            self.graph.add_node(c1,
                    concept_id=c1,
                    concept_name=concept_1_name,
                    domain=domain1,
                    vocabulary=vocab1,
                    concept_class=concept_class1
                )

            self.graph.add_node(c2,
                    concept_id=c2,
                    concept_name=concept_2_name,
                    domain=domain2,
                    vocabulary=vocab2,
                    concept_class=concept_class2
                )
        self.save_graph(self.output_file)

        print(f"[INFO] Graph built successfully with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")


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
            print(f"[WARN] Node {start} does not exist in the graph.")
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
            print(f"[WARN] Node {start} does not exist in the graph.")
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
                    print(f"rel is {rel}")
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
        print(f"upward is {upward} and downward is {downward} and equivalents are {equivalents}")
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
            print(f"[ERROR] Node {node} is not an integer.")
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

    def bfs_bidirectional_reachable(self, start, target_ids, max_depth=3):
        """
        Enhanced BFS that traverses both directions and uses equivalence relationships to reach targets.
        """
        if start not in self.graph:
            print(f"[WARN] Node {start} does not exist in the graph.")
            return []
        reachable_target_ids = set()
        target_ids = set(target_ids)

        # Include target equivalents
        target_equiv_map = {
            tid: self.get_equivalent_nodes(tid, max_depth=1) | {tid}
            for tid in target_ids
        }
        all_target_equivs = set().union(*target_equiv_map.values())
        # print(f"all target equivalents are {all_target_equivs}")
        # Step 1: BFS Upward from start
        upward = self.bfs_upward_reachable(start, all_target_equivs, max_depth)
        print(f"upward reachable nodes are {upward}")
        reachable_target_ids.update(upward)

        # Step 2: BFS Downward from start
        downward = self.bfs_downward_reachable(start, all_target_equivs, max_depth)
        print(f"downward reachable nodes are {downward}")
        reachable_target_ids.update(downward)
        
        # ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
        # STEP 2 b – *sibling* search via common ancestors  ◄─ NEW CODE
        # ▒▒▒▒ find every ancestor of `start` (≤ max_depth)
        ancestors = self.get_all_parents(start, max_depth=max_depth)
        # print(f"len of ancestors is {len(ancestors)} and ancestors are {ancestors}")
        for anc in ancestors:
            print(f"anc is {anc}")
            for child in self.graph.successors(anc):        # direct children = siblings/cousins
                print(f"child is {child}")
                if child == start or self.all_below_atc_4th(start, child, anc):
                     print(f"skipping {child} as it is start node or below atc 4th level")
                     continue
                elif child in all_target_equivs:
                    print(f"[SIBLING-MATCH] target {child} reachable via sibling {anc}")
                    reachable_target_ids.add(child)
                else:
                    # try 1-hop equivalents of that child
                    equivs = self.get_equivalent_nodes(child, max_depth=1)
                    common = equivs & all_target_equivs
                    if common:
                        print(f"[SIBLING-MATCH] target {child} reachable via sibling {anc} and its equivalents")
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
                    print(f"[EQUIV-MATCH] target {tid} reachable via equivalent {eq}")
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
        print(f"downward children are {downward_children}")
        for child in downward_children:
            equivalents = self.get_equivalent_nodes(child, max_depth=1)
            if equivalents.intersection(target_ids):
                print(f"[DESC-EQUIV-MATCH] {child} maps to one of {target_ids} via equivalence")
                reachable_target_ids.update(equivalents.intersection(target_ids))
        
        # only return target_ids that are reachable
        reachable_target_ids = reachable_target_ids.intersection(target_ids)
        return list(reachable_target_ids)  # Remove the start node if present

    

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

    def bfs_path(self, start, target_ids, max_depth=15):
        """
        Return all nodes reachable from `start` in self.graph and check if target_ids are reachable.
        """
        if start not in self.graph:
            print(f"[WARN] Node {start} does not exist in the graph.")
            return []
        
        # Get all reachable nodes using shortest path lengths
        reachable_nodes = nx.single_source_shortest_path_length(self.graph, start,cutoff=max_depth)
        # reachable_targets = {target for target in target_ids 
        #                  if target in reachable_nodes and reachable_nodes[target] < max_depth}
        reachable_targets = {target for target in target_ids 
                         if target in reachable_nodes}
        # Find which target IDs are reachable
        # reachable_targets = target_ids.intersection(reachable_nodes)

        # print(f"len of reachable nodes is {len(reachable_nodes)}")
        # print(f"reachable target ids via graph path is {len(reachable_targets)}")
        # reachable_targets_with_depth = {
        #         target: reachable_nodes[target] 
        #         for target in target_ids 
        #         if target in reachable_nodes
        #     }
        return list(reachable_targets)
        # return reachable_targets_with_depth

    def bfs_path_exists(self, start, goal, add_inferred=True):
        """
        Uses networkx to check if a path exists between start and goal.
        Optionally, adds an inferred edge (shortcut) if a path is found.
        
        :param start: integer or string concept_id.
        :param goal: integer or string concept_id.
        :param add_inferred: if True, adds a direct edge when a path is found.
        :return: True if a path exists, else False.
        """
        start = int(start)
        goal = int(goal)
        path = None
        is_match = False
        if start not in self.graph or goal not in self.graph:
            print(f"[WARN] One of the nodes {start} or {goal} does not exist in the graph.")
            is_match = False
        else:
            if start == goal:
                is_match = True
            else:
                try:
                    # networkx.shortest_path uses efficient algorithms.
                    path = nx.shortest_path(self.graph, source=start, target=goal)
                    # print (f"for {start} to {goal} path is {path}")
                    if path and len(path) < 6:
                        is_match = True
                    elif len(path) >=6:
                         is_match = False
                         
                    # if add_inferred and not self.graph.has_edge(start, goal):
                        # self.add_inferred_edge(start, goal)
                        
                except nx.NetworkXNoPath:
                    is_match = False
        return is_match, path



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
            print(f"[WARN] Start node {start} not in graph.")
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
        print(f"[INFO] Inferred edge added: {start} <-> {goal}")
        # Optionally, you can persist this update:
        self.save_graph(self.output_file)

    def save_graph(self, pickle_file='graph_nx.pkl'):
        """
        Save the networkx graph to disk.
        """
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Graph saved to {pickle_file}.")

    def load_graph(self, pickle_file='data/graph_nx.pkl'):
        """
        Load the networkx graph from disk.
        By default, looks for the file in the 'data' folder relative to the project root.
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
# # Example usage

def print_downward_path(graph, start, target, max_depth=5):
        try:
            paths = nx.all_simple_paths(graph.reverse(copy=False), source=target, target=start, cutoff=max_depth)
            paths = list(paths)
            if paths:
                for p in paths:
                    print(f"[PATH from {start} to {target} (downward)]: {list(reversed(p))}")
            else:
                print("No downward path found.")
        except Exception as e:
            print(f"Error in path tracing: {e}")
if __name__ == "__main__":
    csv_path = "/Users/komalgilani/Desktop/cmh/data/concept_relationship.csv"
    
    omop_nx = OmopGraphNX(csv_path)
    omop_nx.build_graph()
    
    start_time = time.time()
    start_concept = 3005456   
    end_concept = [3023103] 
    if start_concept not in omop_nx.graph:
        print(f"{start_concept} does not exist")
    if end_concept[0] not in omop_nx.graph: 
        print(f"{end_concept[0]} does not exist")
    # ,1340128,1341927,1334456,1342439,21601822
    # print(omop_nx.bfs_bidirectional_reachable(start_concept, end_concept, max_depth=5))
    # print(omop_nx.find_sibling_targets(start_concept=start_concept, target_ids=end_concept, max_depth=5))
    
    # print(omop_nx.find_sibling_targets(21601783, [21601822], max_depth=1))
    # print(omop_nx.get_all_parents(21601783, max_depth=1))
    # print(omop_nx.get_equivalent_nodes(1338005))
    # is_match, path = omop_nx.bfs_path_exists(21601689, 21601665)
    # print(f"Match: {is_match}, path: {path}") 
    # reachable_parents = omop_nx.bfs_upward_reachable(21601783,[], max_depth=5)
    # print(reachable_parents)  # should include 21601665

    # children = omop_nx.bfs_downward_reachable(21601689, target_ids=[21601665], max_depth=5)
    # print(21601665 in children)  # should be True
   # print(omop_nx.bfs_bidirectional_reachable(21601784, [1308216], max_depth=2))
   # print(omop_nx.bfs_bidirectional_reachable(4329847, [312327], max_depth=2))
    #print(omop_nx.bfs_downward_reachable(21601784, target_ids=[1308216], max_depth=2))
    # print(omop_nx.bfs_bidirectional_reachable(21601665, target_ids=[1338005], max_depth=2))
    # print(omop_nx.bfs_bidirectional_reachable(956874, target_ids=[4186998], max_depth=2))    
    #print(omop_nx.bfs_bidirectional_reachable(21601782, target_ids=[1308216], max_depth=3))
    # print(omop_nx.bfs_bidirectional_reachable(2000000057, target_ids=[763968], max_depth=2))
    print(omop_nx.only_upward_or_downward(3028437, target_ids=[3001308], max_depth=1))
    # print(3000285 in omop_nx.graph)
    
    # find relationships between two nodes
    # print(omop_nx.bfs_path(3020491, [4156660], max_depth=1))
    # print(omop_nx.find_sibling_targets(3028437, [44817294], max_depth=1))
