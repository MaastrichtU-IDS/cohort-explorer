import argparse
import os
import pickle

import networkx as nx
import pandas as pd
from tqdm import tqdm

from .param import DATA_DIR
from .utils import normalize

STOP_WORDS = {
    "stop",
    "start",
    "combinations",
    "combination",
    "various combinations",
    "various",
    "left",
    "right",
    "blood",
    "finding",
    "finding status",
    "status",
    "extra",
    "point in time",
    "pnt",
    "oral",
    "product",
    "oral product",
    "several",
    "types",
    "several types",
    "random",
    "nominal",
    "p time",
    "quan",
    "qual",
    "quantitative",
    "qualitative",
    "ql",
    "qn",
    "quant",
    "anti",
    "antibodies",
    "wb",
    "whole blood",
    "serum",
    "plasma",
    "diseases",
    "disorders",
    "disorder",
    "disease",
    "lab test",
    "measurements",
    "lab tests",
    "meas value",
    "measurement",
    "procedure",
    "procedures",
    "panel",
    "ordinal",
    "after",
    "before",
    "survey",
    "level",
    "levels",
    "others",
    "other",
    "p dose",
    "dose",
    "dosage",
    "frequency",
    "calc",
    "calculation",
    "qnt",
    "cell",
    "avg",
    "average",
    "qal",
    "qul",
    "plt",
    "pln",
    "test",
    "tests",
    "thrb",
    "combinations combinations",
    "function",
    "fn",
    "fnc",
    "pst",
    "panl",
    "ob",
    "gyn",
    "p birth",
    "birth",
    "pan",
    "panel.",
    "proc",
    "device",
    "stnd",
    "standard",
    ";;",
    ";",
    "+",
    "-",
    "_",
    '"',
}


# import nltk
# nltk.download('punkt')
class Omop:
    def __init__(self, concept_dir, taxonomy=True):
        """ "
        concept_dir :  it should contain concept.txt, uml_synonyms.txt , chv_synonyms.txt, relationships.txt umls_defintion.txt

        """
        print(f"concept dir = {concept_dir}")
        self.omop_path = os.path.join(concept_dir, "output/concepts.csv")
        self.omop_rel_path = os.path.join(
            concept_dir, "output/relationships.csv"
        )  # change it with ancestor_concept.csv which include ancestor_concept_id, decendant_concept_id, min_levels_of_seperation, max_levels_of_seperation, consider min_level which could be 0 or 1 : 0 in cases where ancestor_concept_id can be its own ancestor
        self.graph_path = os.path.join(concept_dir, "output/omop_bi_graph.pkl")  # Path for the serialized graph
        print(f"graph path = {self.graph_path}")
        self.data_dir = concept_dir
        self.graph = None
        self.taxonomy = taxonomy

    def filter_synonyms(self, entity, vocab, entity_synonyms, stop_words=STOP_WORDS):
        entity = entity.lower()
        cleaned_synonyms = set()

        for synonym in entity_synonyms:
            cleaned = synonym.lower().replace(",", " ").strip()
            # words = set(cleaned.split())  # Convert to set directly here
            # Skip synonyms that are the same as the entity
            if cleaned == entity:
                continue

            # Skip synonyms containing any stop words
            if cleaned in STOP_WORDS:  # Checking each word individually
                continue
            if len(cleaned) < 2 and vocab != "ucum":
                continue

            # # Skip synonyms that are a substring of another synonym
            # if any(cleaned in s for s in cleaned_synonyms) or any(s in cleaned for s in cleaned_synonyms):
            #     continue

            cleaned_synonyms.add(synonym)

        return cleaned_synonyms

    def save_graph(self):
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        print("Graph saved to disk.")

    def load_graph(self):
        with open(self.graph_path, "rb") as f:
            self.graph = pickle.load(f)
        print("Graph loaded from disk.")

    def load_concepts(self):
        if os.path.exists(self.graph_path):
            self.load_graph()
        else:
            self.build_graph_for_synonym_pairs()

    def build_graph_for_synonym_pairs(self):
        # Initialize the graph based on taxonomy preference.
        self.graph = nx.DiGraph() if self.taxonomy else nx.Graph()
        # self.load_indexes()
        count_ent = 0
        ancestor_not_found = 0
        data = pd.read_csv(self.omop_path, dtype=str)
        print(data["vocabulary_id"].unique())
        print(f"Total entities={len(data['concept_name'].tolist())}")
        print(f"Total entities with synonyms ={len(data['concept_synonym_name'].tolist())}")
        # concept_ids = set(data['concept_id'].tolist())
        vocab_synonym_counts = {}
        for _, row in tqdm(data.iterrows()):
            cid = row["concept_id"]
            if cid not in self.graph:
                if not pd.isna(row["concept_name"]) and row["concept_name"] != "nan":
                    cname = row["concept_name"].strip().lower()
                    if not pd.isna(row["vocabulary_id"]):
                        ontology_ = row["vocabulary_id"].strip().lower()
                    else:
                        print(f"Ontology not found for {cname}")
                        ontology_ = None  # or you can set a default value
                    code_ = row["concept_code"]
                    synonyms = set()  # Use a set to avoid duplicate synonyms
                    if not pd.isna(row["concept_synonym_name"]):
                        if ";;" in row["concept_synonym_name"]:
                            synonyms = self.filter_synonyms(
                                cname,
                                vocab=ontology_,
                                entity_synonyms=set(row["concept_synonym_name"].split(";;")),
                                stop_words=STOP_WORDS,
                            )
                        else:
                            synonyms.add(normalize(row["concept_synonym_name"]))
                    if ontology_ == "ucum" or ontology_ == "UCUM":
                        synonyms.update([code_])
                        print(f"synonyms  = {synonyms} for {cname}")
                    synonyms = list(synonyms)  # Convert back to a list
                    # print(f"synonyms  = {synonyms} for {cname}")
                    desc = cname
                    concept_class = f"{row['concept_class_id']}"
                    domain_ = f"{row['domain_id']}"
                    hier_synonyms = []
                    aligned_synonyms = []
                    maps_to_syn = []
                    _answer_of = ""
                    type_standard = str(row["standard_concept"])
                    if desc:
                        self.graph.add_node(
                            cid,
                            code=code_,
                            desc=desc,
                            synonyms=synonyms,
                            hier_syn=hier_synonyms,
                            maps_to_syn=maps_to_syn,
                            aligned_synonyms=aligned_synonyms,
                            domain=domain_,
                            concept_class=concept_class,
                            ontology=ontology_,
                            type_standard=type_standard,
                            has_answers=[],
                            answer_of=_answer_of,
                        )
                        if ontology_ not in vocab_synonym_counts:
                            vocab_synonym_counts[ontology_] = 0
                        if len(synonyms) > 0:
                            vocab_synonym_counts[ontology_] += 1
                    else:
                        count_ent += 1
        print(f"curent synonm counts = {vocab_synonym_counts}")
        drug_not_found = 0
        ns_to_s_not_found = 0
        answer_not_found = 0
        # relations = pd.read_csv(self.omop_rel_path, dtype=str)
        # relations = relations[relations['relationship_id'].isin(['Is a','RxNorm - SNOMED eq','SNOMED - RxNorm eq','ATC - SNOMED eq','SNOMED - ATC eq','Has tradename','Tradename of','ATC - RxNorm pr lat','RxNorm - ATC pr lat','Maps to','Mapped from'])]
        # self.omop_rel_path = self.data_dir  + "/output/filtered_relationship.csv"
        # relations.to_csv(self.omop_rel_path, index=False)
        inverse_relationships = {
            "RxNorm - SNOMED eq": "SNOMED - RxNorm eq",
            "SNOMED - LOINC eq": "LOINC - SNOMED eq",
            "SNOMED - RxNorm eq": "RxNorm - SNOMED eq",
            "ATC - RxNorm pr lat": "RxNorm - ATC pr lat",
            "ATC - SNOMED eq": "SNOMED - ATC eq",
            "RxNorm - ATC pr lat": "ATC - RxNorm pr lat",
            "ATC - RxNorm name": "RxNorm - ATC name",
            "SNOMED - ICD9P eq": "ICD9P - SNOMED eq",
            "RxNorm - ATC": "ATC - RxNorm",
            # 'Maps to': 'Mapped from',
            # 'Mapped from': 'Maps to',
            "Has tradename": "Tradename of",
            "Tradename of": "Has tradename",
        }
        # relations.to_csv(self.omop_rel_path, index=False)
        for chunk in pd.read_csv(self.omop_rel_path, dtype=str, chunksize=10000):
            for _, row in tqdm(chunk.iterrows()):
                concept1 = str(row["concept_id_1"])
                concept2 = str(row["concept_id_2"])
                relationship = row["relationship_id"]

                if relationship == "Is a" or relationship == "Component of":
                    # Handle hierarchical relationships
                    if self.graph.has_node(concept1) and self.graph.has_node(concept2):
                        vocab_concept_1 = self.graph.nodes[concept1]["ontology"]
                        vocab_concept_2 = self.graph.nodes[concept2]["ontology"]
                        if vocab_concept_1 != vocab_concept_2 and relationship == "Component of":
                            parent_desc = self.graph.nodes[concept2]["desc"]
                            child_syn = self.graph.nodes[concept1]["aligned_synonyms"]
                            child_syn = [parent_desc] + child_syn  # Append the desc to the first index in the list
                            self.graph.nodes[concept1]["aligned_synonyms"] = list(set(child_syn))
                            self.graph.add_edge(concept2, concept1)  # Parent to child edge.
                        else:
                            parent_desc = self.graph.nodes[concept2]["desc"]
                            child_syn = self.graph.nodes[concept1]["hier_syn"]
                            child_syn = [parent_desc] + child_syn  # Append the desc to the first index in the list
                            self.graph.nodes[concept1]["hier_syn"] = list(set(child_syn))
                            self.graph.add_edge(concept2, concept1)  # Parent to child edge.
                    else:
                        ancestor_not_found += 1

                if relationship == "Has Answer":
                    # Handle hierarchical relationships
                    if self.graph.has_node(concept1) and self.graph.has_node(concept2):
                        main_concept_answers = set(self.graph.nodes[concept1]["has_answers"])
                        parent_desc = self.graph.nodes[concept2]["desc"]
                        main_concept_answers.add(parent_desc)  # Enrich child with parent information
                        self.graph.nodes[concept1]["has_answers"] = list(main_concept_answers)
                        self.graph.add_edge(concept1, concept2)  # question to answer
                    else:
                        answer_not_found += 1
                # if relationship == 'Answer of':
                #     # Handle hierarchical relationships  concept2 is question and concept1 is answer, we need to add concept2 description as has_answers in concept1
                #     if self.graph.has_node(concept1) and self.graph.has_node(concept2):
                #         main_concept_answers = set(self.graph.nodes[concept1]['answer_of'])
                #         parent_desc = self.graph.nodes[concept2]['desc']
                #         main_concept_answers.add(parent_desc)  # Enrich child with parent information
                #         self.graph.nodes[concept1]['answer_of'] =
                #         self.graph.add_edge(concept1, concept2)  # answer to question
                #     else:
                #         answer_not_found += 1
                if relationship == "Maps to":
                    # Handle hierarchical relationships concept1 maps to concept2 than concept2 is standard concept so add concept2 synonyms to concept1
                    if self.graph.has_node(concept1) and self.graph.has_node(concept2):
                        child_syn = set(self.graph.nodes[concept1]["maps_to_syn"])
                        parent_desc = self.graph.nodes[concept2]["desc"]
                        child_syn.add(parent_desc)  # Enrich non-standard concept with standard_concept information
                        self.graph.nodes[concept1]["maps_to_syn"] = list(child_syn)
                        self.graph.add_edge(concept1, concept2)  # non-standard to standard edge.
                    else:
                        ns_to_s_not_found += 1

                elif relationship in inverse_relationships:
                    # Handle relationships and their inverses uniformly
                    if self.graph.has_node(concept1) and self.graph.has_node(concept2):
                        concept1_syn = set(self.graph.nodes[concept1]["aligned_synonyms"])
                        concept2_syn = set(self.graph.nodes[concept2]["aligned_synonyms"])

                        concept1_syn.add(self.graph.nodes[concept2]["desc"])
                        concept2_syn.add(self.graph.nodes[concept1]["desc"])

                        self.graph.nodes[concept1]["aligned_synonyms"] = list(concept1_syn)
                        self.graph.nodes[concept2]["aligned_synonyms"] = list(concept2_syn)

                        self.graph.add_edge(concept1, concept2)  # Concept1 to Concept2 edge.

                    else:
                        drug_not_found += 1
        # Print the updated counts after adding synonyms
        updated_vocab_synonym_counts = {}
        for cid in self.graph.nodes():
            node = self.graph.nodes[cid]
            ontology_ = node["ontology"]
            if ontology_ not in updated_vocab_synonym_counts:
                updated_vocab_synonym_counts[ontology_] = 0
            if len(node["synonyms"]) > 0 or len(node["aligned_synonyms"]) > 0 or len(node["hier_syn"]) > 0:
                updated_vocab_synonym_counts[ontology_] += 1

        print("\nUpdated counts after adding synonyms:")
        for vocab, count in updated_vocab_synonym_counts.items():
            print(f"{vocab}: {count} concepts with synonyms")

        print(f"Entity not in graph = {count_ent}")
        print(f"Entity not found in graph for hierarchy check = {ancestor_not_found}")
        print(f"Entity not found in graph for Drugs check = {drug_not_found}")
        print(f"Answer Entity not found in graph for LOINC CODES = {answer_not_found}")
        self.save_graph()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a synonym graph")
    parser.add_argument("--concept_dir", type=str, default=DATA_DIR, help="Path to the concept directory")
    parser.add_argument("--taxonomy", action="store_true", help="Whether to use taxonomy (default: True)")
    args = parser.parse_args()

    omop = Omop(args.concept_dir, args.taxonomy)
    omop.build_graph_for_synonym_pairs()
