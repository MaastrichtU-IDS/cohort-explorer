# experiment/modes.py
from enum import Enum

class MappingType(str, Enum):
    OO = "ontology_only"
    OED = "ontology+embedding(description)"
    OEC = "ontology+embedding(concept)"
    OEH = "ontology+embedding(hybrid)"


class EmbeddingType(str, Enum):
    ED = "embedding(description)"
    EC = "embedding(concept)"
    EH = "embedding(hybrid)"

