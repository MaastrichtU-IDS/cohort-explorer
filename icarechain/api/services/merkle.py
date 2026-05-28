import hashlib
from typing import Optional
from dataclasses import dataclass, field
from eth_abi import encode

@dataclass
class MerkleTree:
    \
\
\
\
\
    leaves: list[bytes] = field(default_factory=list)
    layers: list[list[bytes]] = field(default_factory=list)

    def __post_init__(self):
        if self.leaves:
            self._build_tree()

    def _keccak256(self, data: bytes) -> bytes:
        \
        from web3 import Web3
        return Web3.keccak(data)

    def _build_tree(self):
        \
        if not self.leaves:
            return

        sorted_leaves = sorted(self.leaves)

        self.layers = [sorted_leaves]

        current_layer = sorted_leaves
        while len(current_layer) > 1:
            next_layer = []
            for i in range(0, len(current_layer), 2):
                if i + 1 < len(current_layer):

                    left, right = current_layer[i], current_layer[i + 1]
                    if left > right:
                        left, right = right, left
                    next_layer.append(self._keccak256(left + right))
                else:

                    next_layer.append(current_layer[i])

            self.layers.append(next_layer)
            current_layer = next_layer

    @property
    def root(self) -> bytes:
        \
        if not self.layers:
            return bytes(32)
        return self.layers[-1][0] if self.layers[-1] else bytes(32)

    def get_proof(self, leaf: bytes) -> list[bytes]:
        \
\
\
\
\
        if leaf not in self.leaves:
            return []

        proof = []

        sorted_leaves = sorted(self.leaves)
        index = sorted_leaves.index(leaf)

        for layer in self.layers[:-1]:
            sibling_index = index ^ 1

            if sibling_index < len(layer):
                proof.append(layer[sibling_index])

            index //= 2

        return proof

    def verify(self, leaf: bytes, proof: list[bytes]) -> bool:
        \
        computed_hash = leaf

        for proof_element in proof:

            if computed_hash < proof_element:
                combined = computed_hash + proof_element
            else:
                combined = proof_element + computed_hash

            computed_hash = self._keccak256(combined)

        return computed_hash == self.root

@dataclass
class CountryMerkleTree(MerkleTree):
    \
\
\
\
\
\
    countries: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.countries:
            self.leaves = [self._country_leaf(c) for c in self.countries]
            self._build_tree()

    def _country_leaf(self, country_code: str) -> bytes:
\

        country_bytes = country_code.encode('ascii')[:2].ljust(2, b'\x00')
        return self._keccak256(country_bytes)

    def get_country_proof(self, country_code: str) -> list[bytes]:
        \
        leaf = self._country_leaf(country_code)
        return self.get_proof(leaf)

    def get_country_leaf(self, country_code: str) -> bytes:
        \
        return self._country_leaf(country_code)

@dataclass
class InstitutionMerkleTree(MerkleTree):
    \
\
\
\
\
\
    institutions: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.institutions:
            self.leaves = [self._institution_leaf(i) for i in self.institutions]
            self._build_tree()

    def _institution_id_hash(self, ror_id: str) -> bytes:
        \
        return self._keccak256(ror_id.encode('utf-8'))

    def _institution_leaf(self, ror_id: str) -> bytes:
        \
        inst_id = self._institution_id_hash(ror_id)
        return self._keccak256(inst_id)

    def get_institution_proof(self, ror_id: str) -> list[bytes]:
        \
        leaf = self._institution_leaf(ror_id)
        return self.get_proof(leaf)

    def get_institution_leaf(self, ror_id: str) -> bytes:
        \
        return self._institution_leaf(ror_id)

    def get_institution_id_hash(self, ror_id: str) -> bytes:
        \
        return self._institution_id_hash(ror_id)

class MerkleService:
    \
\
\
\
\
\
\
\
\
    def __init__(self):
        self._country_trees: dict[str, CountryMerkleTree] = {}
        self._institution_trees: dict[str, InstitutionMerkleTree] = {}

    def create_country_tree(self, cohort_hash: str, countries: list[str]) -> bytes:
        \
\
\
\
\
\
\
\
\
\
        tree = CountryMerkleTree(countries=countries)
        self._country_trees[cohort_hash] = tree
        return tree.root

    def create_institution_tree(self, cohort_hash: str, institutions: list[str]) -> bytes:
        \
\
\
\
\
\
\
\
\
\
        tree = InstitutionMerkleTree(institutions=institutions)
        self._institution_trees[cohort_hash] = tree
        return tree.root

    def get_country_proof(self, cohort_hash: str, country_code: str) -> Optional[dict]:
        \
\
\
\
\
\
\
\
\
\
        tree = self._country_trees.get(cohort_hash)
        if not tree:
            return None

        proof = tree.get_country_proof(country_code)
        if not proof:
            return None

        return {
            "proof": [p.hex() for p in proof],
            "leaf": tree.get_country_leaf(country_code).hex(),
            "root": tree.root.hex(),
            "country_code": country_code
        }

    def get_institution_proof(self, cohort_hash: str, ror_id: str) -> Optional[dict]:
        \
\
\
\
\
\
\
\
\
\
        tree = self._institution_trees.get(cohort_hash)
        if not tree:
            return None

        proof = tree.get_institution_proof(ror_id)
        if not proof:
            return None

        return {
            "proof": [p.hex() for p in proof],
            "leaf": tree.get_institution_leaf(ror_id).hex(),
            "institution_id": tree.get_institution_id_hash(ror_id).hex(),
            "root": tree.root.hex(),
            "ror_id": ror_id
        }

    def verify_country(self, cohort_hash: str, country_code: str, proof: list[str]) -> bool:
        \
        tree = self._country_trees.get(cohort_hash)
        if not tree:
            return False

        leaf = tree.get_country_leaf(country_code)
        proof_bytes = [bytes.fromhex(p) for p in proof]
        return tree.verify(leaf, proof_bytes)

    def verify_institution(self, cohort_hash: str, ror_id: str, proof: list[str]) -> bool:
        \
        tree = self._institution_trees.get(cohort_hash)
        if not tree:
            return False

        leaf = tree.get_institution_leaf(ror_id)
        proof_bytes = [bytes.fromhex(p) for p in proof]
        return tree.verify(leaf, proof_bytes)

    def get_country_root(self, cohort_hash: str) -> Optional[str]:
        \
        tree = self._country_trees.get(cohort_hash)
        return tree.root.hex() if tree else None

    def get_institution_root(self, cohort_hash: str) -> Optional[str]:
        \
        tree = self._institution_trees.get(cohort_hash)
        return tree.root.hex() if tree else None

    def list_countries(self, cohort_hash: str) -> list[str]:
        \
        tree = self._country_trees.get(cohort_hash)
        return tree.countries if tree else []

    def list_institutions(self, cohort_hash: str) -> list[str]:
        \
        tree = self._institution_trees.get(cohort_hash)
        return tree.institutions if tree else []

_merkle_service: Optional[MerkleService] = None

def get_merkle_service() -> MerkleService:
    \
    global _merkle_service
    if _merkle_service is None:
        _merkle_service = MerkleService()
    return _merkle_service

def generate_country_merkle_root(countries: list[str]) -> str:
    \
\
\
\
\
    tree = CountryMerkleTree(countries=countries)
    return tree.root.hex()

def generate_institution_merkle_root(institutions: list[str]) -> str:
    \
\
\
\
\
    tree = InstitutionMerkleTree(institutions=institutions)
    return tree.root.hex()

def generate_country_proof(countries: list[str], country_code: str) -> Optional[dict]:
    \
\
\
\
\
    tree = CountryMerkleTree(countries=countries)
    proof = tree.get_country_proof(country_code)

    if not proof:
        return None

    return {
        "proof": [p.hex() for p in proof],
        "leaf": tree.get_country_leaf(country_code).hex(),
        "root": tree.root.hex()
    }

def generate_institution_proof(institutions: list[str], ror_id: str) -> Optional[dict]:
    \
\
\
\
\
    tree = InstitutionMerkleTree(institutions=institutions)
    proof = tree.get_institution_proof(ror_id)

    if not proof:
        return None

    return {
        "proof": [p.hex() for p in proof],
        "leaf": tree.get_institution_leaf(ror_id).hex(),
        "institution_id": tree.get_institution_id_hash(ror_id).hex(),
        "root": tree.root.hex()
    }
