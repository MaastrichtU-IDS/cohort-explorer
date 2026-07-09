import hashlib
import hmac
import os
import json
import subprocess
import tempfile
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from api.config import get_settings

FIELD_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617

@dataclass
class DerivedIdentity:
    identity_secret: int
    nullifier_secret: int
    commitment: int
    email_hash: str
    created_at: datetime
    groups: list[str]

@dataclass
class IdentityRecord:
    commitment: str
    created_at: str
    groups: list[str]

class IdentityVault:
    def __init__(self, redis_client=None, master_key: Optional[str] = None):

        settings = get_settings()

        from api.services.wallet import _salt_bytes
        if master_key:
            self._master_key = bytes.fromhex(master_key)
        elif hasattr(settings, 'identity_master_key') and settings.identity_master_key:
            self._master_key = bytes.fromhex(settings.identity_master_key)
        else:
            self._master_key = _salt_bytes() or b"\x00" * 32

        self._encryption_key = self._derive_encryption_key()
        self._fernet = Fernet(self._encryption_key)
        self._redis = redis_client
        self._cache: dict[str, DerivedIdentity] = {}

    def _derive_encryption_key(self) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"duo_consent_vault_v1",
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self._master_key))

    def _normalize_email(self, email: str) -> str:
        return email.lower().strip()

    def _compute_email_hash(self, email: str) -> str:
        normalized = self._normalize_email(email)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _derive_secrets(self, email: str) -> tuple[int, int]:

        from api.services.wallet import derive_identity_secret, derive_nullifier_secret
        identity_bytes = derive_identity_secret(email)
        nullifier_bytes = derive_nullifier_secret(email)
        identity_secret = int.from_bytes(identity_bytes, 'big') % FIELD_ORDER
        nullifier_secret = int.from_bytes(nullifier_bytes, 'big') % FIELD_ORDER
        return identity_secret, nullifier_secret

    def _compute_commitment(self, identity_secret: int, nullifier_secret: int) -> int:

        secret_bytes = identity_secret.to_bytes(32, 'big')
        commitment_bytes = hashlib.sha256(secret_bytes).digest()
        return int.from_bytes(commitment_bytes, 'big') % FIELD_ORDER

    def derive_identity(self, email: str) -> DerivedIdentity:
        email_hash = self._compute_email_hash(email)

        if email_hash in self._cache:
            return self._cache[email_hash]

        identity_secret, nullifier_secret = self._derive_secrets(email)
        commitment = self._compute_commitment(identity_secret, nullifier_secret)

        identity = DerivedIdentity(
            identity_secret=identity_secret,
            nullifier_secret=nullifier_secret,
            commitment=commitment,
            email_hash=email_hash,
            created_at=datetime.utcnow(),
            groups=[]
        )

        self._cache[email_hash] = identity
        return identity

    async def store_identity(self, identity: DerivedIdentity) -> bool:
        if not self._redis:
            return False

        record = IdentityRecord(
            commitment=hex(identity.commitment),
            created_at=identity.created_at.isoformat(),
            groups=identity.groups
        )

        encrypted = self._fernet.encrypt(
            json.dumps(asdict(record)).encode()
        )

        key = f"identity:{identity.email_hash}"
        await self._redis.set(key, encrypted)
        return True

    async def load_identity(self, email: str) -> Optional[DerivedIdentity]:
        if not self._redis:
            return None

        email_hash = self._compute_email_hash(email)
        key = f"identity:{email_hash}"

        encrypted = await self._redis.get(key)
        if not encrypted:
            return None

        decrypted = self._fernet.decrypt(encrypted)
        record_dict = json.loads(decrypted)

        identity_secret, nullifier_secret = self._derive_secrets(email)

        identity = DerivedIdentity(
            identity_secret=identity_secret,
            nullifier_secret=nullifier_secret,
            commitment=int(record_dict['commitment'], 16),
            email_hash=email_hash,
            created_at=datetime.fromisoformat(record_dict['created_at']),
            groups=record_dict['groups']
        )

        self._cache[email_hash] = identity
        return identity

    async def add_to_group(self, email: str, cohort_hash: str) -> bool:
        identity = await self.load_identity(email)
        if not identity:
            identity = self.derive_identity(email)

        if cohort_hash not in identity.groups:
            identity.groups.append(cohort_hash)
            await self.store_identity(identity)

        return True

    async def remove_from_group(self, email: str, cohort_hash: str) -> bool:
        identity = await self.load_identity(email)
        if not identity:
            return False

        if cohort_hash in identity.groups:
            identity.groups.remove(cohort_hash)
            await self.store_identity(identity)

        return True

    async def get_groups(self, email: str) -> list[str]:
        identity = await self.load_identity(email)
        if not identity:
            return []
        return identity.groups

    def get_commitment(self, email: str) -> int:
        identity = self.derive_identity(email)
        return identity.commitment

    def clear_cache(self):
        self._cache.clear()

_vault_instance: Optional[IdentityVault] = None

def get_identity_vault() -> IdentityVault:
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = IdentityVault()
    return _vault_instance

async def get_identity_vault_async(redis_client=None) -> IdentityVault:
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = IdentityVault(redis_client=redis_client)
    return _vault_instance

@dataclass
class MerkleNode:
    value: int
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None

@dataclass
class MerkleProof:
    root: int
    path: list[int]
    indices: list[int]

@dataclass
class SemaphoreGroup:
    cohort_hash: str
    depth: int
    members: list[int] = field(default_factory=list)
    merkle_root: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    synced_at: Optional[datetime] = None

class GroupManager:
    DEFAULT_DEPTH = 16
    ZEROS: list[int] = []

    def __init__(self):
        self._vault = get_identity_vault()
        self._groups: dict[str, SemaphoreGroup] = {}
        self._init_zeros()

    def _init_zeros(self):
        zero = 0
        self.ZEROS = [zero]
        for _ in range(self.DEFAULT_DEPTH):
            zero = self._hash_pair(zero, zero)
            self.ZEROS.append(zero)

    def _hash(self, value: int) -> int:
        data = str(value).encode()
        hash_bytes = hashlib.sha256(data).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    def _hash_pair(self, left: int, right: int) -> int:
        combined = f"{left}:{right}".encode()
        hash_bytes = hashlib.sha256(combined).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    def _compute_cohort_hash(self, cohort_id: str) -> str:
        hash_bytes = hashlib.sha256(cohort_id.encode()).digest()
        return "0x" + hash_bytes.hex()

    def create_group(self, cohort_id: str, depth: int = None) -> SemaphoreGroup:
        cohort_hash = self._compute_cohort_hash(cohort_id)

        if cohort_hash in self._groups:
            return self._groups[cohort_hash]

        group = SemaphoreGroup(
            cohort_hash=cohort_hash,
            depth=depth or self.DEFAULT_DEPTH,
            members=[],
            merkle_root=self.ZEROS[self.DEFAULT_DEPTH],
            created_at=datetime.utcnow()
        )

        self._groups[cohort_hash] = group
        return group

    def get_group(self, cohort_id: str) -> Optional[SemaphoreGroup]:
        cohort_hash = self._compute_cohort_hash(cohort_id)
        return self._groups.get(cohort_hash)

    def add_member(self, cohort_id: str, commitment: int) -> tuple[int, MerkleProof]:
        cohort_hash = self._compute_cohort_hash(cohort_id)

        if cohort_hash not in self._groups:
            self.create_group(cohort_id)

        group = self._groups[cohort_hash]

        if commitment in group.members:
            return group.merkle_root, self.get_membership_proof(cohort_id, commitment)

        group.members.append(commitment)
        group.merkle_root = self._compute_merkle_root(group.members, group.depth)
        proof = self.get_membership_proof(cohort_id, commitment)

        return group.merkle_root, proof

    def add_member_by_email(self, cohort_id: str, email: str) -> tuple[int, MerkleProof]:
        identity = self._vault.derive_identity(email)
        return self.add_member(cohort_id, identity.commitment)

    def remove_member(self, cohort_id: str, commitment: int) -> int:
        cohort_hash = self._compute_cohort_hash(cohort_id)

        if cohort_hash not in self._groups:
            raise ValueError(f"Group not found: {cohort_id}")

        group = self._groups[cohort_hash]

        if commitment not in group.members:
            raise ValueError(f"Member not found: {commitment}")

        group.members.remove(commitment)
        group.merkle_root = self._compute_merkle_root(group.members, group.depth)
        return group.merkle_root

    def _compute_merkle_root(self, members: list[int], depth: int) -> int:
        if not members:
            return self.ZEROS[depth]

        layer = members.copy()
        layer_size = 2 ** depth

        while len(layer) < layer_size:
            layer.append(0)

        for level in range(depth):
            next_layer = []
            for i in range(0, len(layer), 2):
                left = layer[i] if i < len(layer) else self.ZEROS[level]
                right = layer[i + 1] if i + 1 < len(layer) else self.ZEROS[level]
                next_layer.append(self._hash_pair(left, right))
            layer = next_layer

        return layer[0]

    def get_membership_proof(self, cohort_id: str, commitment: int) -> MerkleProof:
        cohort_hash = self._compute_cohort_hash(cohort_id)

        if cohort_hash not in self._groups:
            raise ValueError(f"Group not found: {cohort_id}")

        group = self._groups[cohort_hash]

        if commitment not in group.members:
            raise ValueError(f"Member not found: {commitment}")

        index = group.members.index(commitment)
        depth = group.depth

        layer = group.members.copy()
        layer_size = 2 ** depth

        while len(layer) < layer_size:
            layer.append(0)

        path = []
        indices = []

        for level in range(depth):
            sibling_index = index ^ 1
            sibling = layer[sibling_index] if sibling_index < len(layer) else self.ZEROS[level]
            path.append(sibling)
            indices.append(index & 1)

            next_layer = []
            for i in range(0, len(layer), 2):
                left = layer[i] if i < len(layer) else self.ZEROS[level]
                right = layer[i + 1] if i + 1 < len(layer) else self.ZEROS[level]
                next_layer.append(self._hash_pair(left, right))

            layer = next_layer
            index = index // 2

        return MerkleProof(root=group.merkle_root, path=path, indices=indices)

    def verify_membership(self, commitment: int, proof: MerkleProof) -> bool:
        current = commitment
        for sibling, index in zip(proof.path, proof.indices):
            if index == 0:
                current = self._hash_pair(current, sibling)
            else:
                current = self._hash_pair(sibling, current)
        return current == proof.root

    async def sync_with_chain(self, cohort_id: str) -> bool:
        try:
            from api.services.blockchain import get_blockchain_service
            blockchain = get_blockchain_service()
            cohort_hash = self._compute_cohort_hash(cohort_id)

            group_data = await blockchain.call_contract(
                'IdentityRegistry',
                'getGroup',
                [bytes.fromhex(cohort_hash[2:])]
            )

            if group_data and group_data[3]:
                if cohort_hash not in self._groups:
                    self.create_group(cohort_id)

                group = self._groups[cohort_hash]
                group.merkle_root = group_data[1]
                group.synced_at = datetime.utcnow()
                return True

        except Exception as e:
            print(f"Failed to sync group {cohort_id}: {e}")

        return False

    async def push_to_chain(self, cohort_id: str) -> Optional[str]:
        try:
            from api.services.blockchain import get_blockchain_service
            blockchain = get_blockchain_service()
            cohort_hash = self._compute_cohort_hash(cohort_id)

            if cohort_hash not in self._groups:
                raise ValueError(f"Group not found: {cohort_id}")

            group = self._groups[cohort_hash]

            tx_hash = await blockchain.send_transaction(
                'IdentityRegistry',
                'updateMerkleRoot',
                [bytes.fromhex(cohort_hash[2:]), group.merkle_root]
            )

            group.synced_at = datetime.utcnow()
            return tx_hash

        except Exception as e:
            print(f"Failed to push group {cohort_id}: {e}")
            return None

    def get_all_groups(self) -> list[SemaphoreGroup]:
        return list(self._groups.values())

    def get_group_stats(self) -> dict:
        return {
            "total_groups": len(self._groups),
            "total_members": sum(len(g.members) for g in self._groups.values()),
            "groups": [
                {
                    "cohort_hash": g.cohort_hash,
                    "member_count": len(g.members),
                    "depth": g.depth,
                    "synced_at": g.synced_at.isoformat() if g.synced_at else None
                }
                for g in self._groups.values()
            ]
        }

_manager_instance: Optional[GroupManager] = None

def get_group_manager() -> GroupManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = GroupManager()
    return _manager_instance

class ActionType(IntEnum):
    CONSENT = 0
    ACCESS = 1
    REVOKE = 2

class DUOPermission(IntEnum):
    NRES = 0
    GRU = 1
    HMB = 2
    DS = 3
    POA = 4

class RequesterType(IntEnum):
    UNKNOWN = 0
    ACADEMIC = 1
    NONPROFIT = 2
    GOVERNMENT = 3
    COMMERCIAL = 4

@dataclass
class SemaphoreProofData:
    merkle_root: int
    nullifier_hash: int
    signal: int
    proof: list[int]

@dataclass
class NoirProofData:
    consent_commitment: bytes
    permission_requested: int
    score_threshold: int
    proof: bytes

@dataclass
class CombinedProof:
    semaphore_proof: SemaphoreProofData
    noir_proof: NoirProofData
    cohort_hash: bytes
    epoch: int

class ProofGenerator:
    def __init__(self):
        self._vault = get_identity_vault()
        self._circuits_dir = Path(__file__).parent.parent.parent / "circuits"
        self._semaphore_dir = self._circuits_dir / "semaphore"
        self._noir_consent_dir = self._circuits_dir / "noir" / "consent_compliance"

    def _compute_cohort_hash(self, cohort_id: str) -> int:
        hash_bytes = hashlib.sha256(cohort_id.encode()).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    def _compute_external_nullifier(self, cohort_id: str, epoch: int, nonce: str = "") -> int:
        cohort_hash = self._compute_cohort_hash(cohort_id)
        combined = f"{cohort_hash}:{epoch}:{nonce}".encode()
        hash_bytes = hashlib.sha256(combined).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    def _compute_nullifier_hash(self, nullifier_secret: int, external_nullifier: int) -> int:
        combined = f"{nullifier_secret}:{external_nullifier}".encode()
        hash_bytes = hashlib.sha256(combined).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    def _compute_signal_hash(self, signal: str) -> int:
        hash_bytes = hashlib.sha256(signal.encode()).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    def _compute_consent_commitment(self, permission: int, modifiers: int, disease_code: int, expiry: int) -> int:
        combined = f"{permission}:{modifiers}:{disease_code}:{expiry}".encode()
        hash_bytes = hashlib.sha256(combined).digest()
        return int.from_bytes(hash_bytes, 'big') % FIELD_ORDER

    async def generate_semaphore_proof(
        self, email: str, cohort_id: str,
        epoch: int, signal: str, merkle_root: int,
        merkle_path: list[int], merkle_indices: list[int],
        nonce: str = ""
    ) -> SemaphoreProofData:
        identity = self._vault.derive_identity(email)
        external_nullifier = self._compute_external_nullifier(cohort_id, epoch, nonce)
        nullifier_hash = self._compute_nullifier_hash(identity.nullifier_secret, external_nullifier)
        signal_hash = self._compute_signal_hash(signal)

        settings = get_settings()
        if not hasattr(settings, 'zkp_production_mode') or not settings.zkp_production_mode:
            return SemaphoreProofData(
                merkle_root=merkle_root,
                nullifier_hash=nullifier_hash,
                signal=signal_hash,
                proof=[0] * 8
            )

        return await self._generate_semaphore_proof_production(
            identity=identity, external_nullifier=external_nullifier,
            signal_hash=signal_hash, merkle_root=merkle_root,
            merkle_path=merkle_path, merkle_indices=merkle_indices
        )

    async def _generate_semaphore_proof_production(
        self, identity: DerivedIdentity, external_nullifier: int,
        signal_hash: int, merkle_root: int,
        merkle_path: list[int], merkle_indices: list[int]
    ) -> SemaphoreProofData:
        input_data = {
            "identitySecret": str(identity.identity_secret),
            "nullifierSecret": str(identity.nullifier_secret),
            "externalNullifier": str(external_nullifier),
            "signalHash": str(signal_hash),
            "merkleRoot": str(merkle_root),
            "merklePath": [str(p) for p in merkle_path],
            "merkleIndices": [str(i) for i in merkle_indices],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name

        try:
            result = subprocess.run(
                [
                    "npx", "snarkjs", "groth16", "fullprove",
                    input_file,
                    str(self._semaphore_dir / "semaphore.wasm"),
                    str(self._semaphore_dir / "semaphore.zkey"),
                    "/dev/stdout"
                ],
                capture_output=True, text=True,
                cwd=str(self._semaphore_dir)
            )

            if result.returncode != 0:
                raise Exception(f"Semaphore proof generation failed: {result.stderr}")

            proof_data = json.loads(result.stdout)

            return SemaphoreProofData(
                merkle_root=merkle_root,
                nullifier_hash=int(proof_data['publicSignals'][0]),
                signal=signal_hash,
                proof=[int(p) for p in proof_data['proof']]
            )
        finally:
            os.unlink(input_file)

    async def generate_noir_proof(
        self, consent_commitment: int, permission_requested: int,
        score_threshold: int, consent_permission: int,
        consent_modifiers: int, consent_disease_code: int,
        consent_expiry: int, requester_type: RequesterType,
        attestations: int, permission_score: int,
        modifier_score: int, attestation_score: int, trust_score: int
    ) -> NoirProofData:
        settings = get_settings()
        if not hasattr(settings, 'zkp_production_mode') or not settings.zkp_production_mode:
            return NoirProofData(
                consent_commitment=consent_commitment.to_bytes(32, 'big'),
                permission_requested=permission_requested,
                score_threshold=score_threshold,
                proof=b'\x00' * 128
            )

        return await self._generate_noir_proof_production(
            consent_commitment=consent_commitment,
            permission_requested=permission_requested,
            score_threshold=score_threshold,
            consent_permission=consent_permission,
            consent_modifiers=consent_modifiers,
            consent_disease_code=consent_disease_code,
            consent_expiry=consent_expiry,
            requester_type=requester_type,
            attestations=attestations,
            permission_score=permission_score,
            modifier_score=modifier_score,
            attestation_score=attestation_score,
            trust_score=trust_score
        )

    async def _generate_noir_proof_production(
        self, consent_commitment: int, permission_requested: int,
        score_threshold: int, consent_permission: int,
        consent_modifiers: int, consent_disease_code: int,
        consent_expiry: int, requester_type: RequesterType,
        attestations: int, permission_score: int,
        modifier_score: int, attestation_score: int, trust_score: int
    ) -> NoirProofData:
        prover_toml = f"""
# Public inputs
consent_commitment = "{consent_commitment}"
permission_requested = "{permission_requested}"
score_threshold = {score_threshold}

# Private inputs
consent_permission = "{consent_permission}"
consent_modifiers = {consent_modifiers}
consent_disease_code = "{consent_disease_code}"
consent_expiry = {consent_expiry}
requester_type = "{requester_type.value}"
attestations = {attestations}
permission_score = {permission_score}
modifier_score = {modifier_score}
attestation_score = {attestation_score}
trust_score = {trust_score}
"""

        prover_path = self._noir_consent_dir / "Prover.toml"
        with open(prover_path, 'w') as f:
            f.write(prover_toml)

        try:
            subprocess.run(
                ["nargo", "compile"],
                cwd=str(self._noir_consent_dir),
                capture_output=True
            )

            result = subprocess.run(
                ["nargo", "prove"],
                cwd=str(self._noir_consent_dir),
                capture_output=True, text=True
            )

            if result.returncode != 0:
                raise Exception(f"Noir proof generation failed: {result.stderr}")

            proof_path = self._noir_consent_dir / "proofs" / "consent_compliance.proof"
            with open(proof_path, 'rb') as f:
                proof_bytes = f.read()

            return NoirProofData(
                consent_commitment=consent_commitment.to_bytes(32, 'big'),
                permission_requested=permission_requested,
                score_threshold=score_threshold,
                proof=proof_bytes
            )
        finally:
            if prover_path.exists():
                os.unlink(prover_path)

    async def generate_access_proof(
        self, email: str, cohort_id: str, intended_use: str,
        consent: dict, requester_profile: dict, attestations: list[dict],
        score_components: dict, epoch: int, merkle_root: int,
        merkle_path: list[int], merkle_indices: list[int]
    ) -> CombinedProof:
        permission_map = {
            'NRES': DUOPermission.NRES, 'GRU': DUOPermission.GRU,
            'HMB': DUOPermission.HMB, 'DS': DUOPermission.DS, 'POA': DUOPermission.POA
        }
        permission_requested = permission_map.get(intended_use.upper(), DUOPermission.GRU)
        consent_permission = permission_map.get(consent.get('permission', 'GRU'), DUOPermission.GRU)

        type_map = {
            'academic': RequesterType.ACADEMIC, 'nonprofit': RequesterType.NONPROFIT,
            'government': RequesterType.GOVERNMENT, 'commercial': RequesterType.COMMERCIAL,
        }
        requester_type = type_map.get(requester_profile.get('type', 'unknown'), RequesterType.UNKNOWN)

        modifier_map = {
            'NPU': 1, 'NCU': 2, 'IRB': 4, 'PUB': 8, 'COL': 16,
            'TS': 32, 'RTN': 64, 'GS': 128, 'IS': 256
        }
        consent_modifiers = 0
        for mod in consent.get('modifiers', []):
            consent_modifiers |= modifier_map.get(mod, 0)

        att_map = {
            'IRB_APPROVAL': 1, 'COLLABORATION': 2, 'PUBLICATION': 4,
            'RETURN_DATA': 8, 'GEOGRAPHIC': 16, 'INSTITUTION': 32
        }
        attestation_bitmap = 0
        for att in attestations:
            if att.get('valid', False):
                attestation_bitmap |= att_map.get(att.get('type', ''), 0)

        consent_commitment = self._compute_consent_commitment(
            consent_permission.value, consent_modifiers,
            0, consent.get('expires_at', 0)
        )

        signal = f"access:{cohort_id}:{intended_use}:{epoch}"

        semaphore_proof = await self.generate_semaphore_proof(
            email=email, cohort_id=cohort_id,
            epoch=epoch, signal=signal, merkle_root=merkle_root,
            merkle_path=merkle_path, merkle_indices=merkle_indices,
        )

        noir_proof = await self.generate_noir_proof(
            consent_commitment=consent_commitment,
            permission_requested=permission_requested.value,
            score_threshold=score_components.get('threshold', 800),
            consent_permission=consent_permission.value,
            consent_modifiers=consent_modifiers,
            consent_disease_code=0,
            consent_expiry=consent.get('expires_at', 0),
            requester_type=requester_type,
            attestations=attestation_bitmap,
            permission_score=score_components.get('permission_score', 0),
            modifier_score=score_components.get('modifier_score', 0),
            attestation_score=score_components.get('attestation_score', 0),
            trust_score=score_components.get('trust_score', 0)
        )

        cohort_hash = self._compute_cohort_hash(cohort_id)

        return CombinedProof(
            semaphore_proof=semaphore_proof,
            noir_proof=noir_proof,
            cohort_hash=cohort_hash.to_bytes(32, 'big'),
            epoch=epoch
        )

_generator_instance: Optional[ProofGenerator] = None

def get_proof_generator() -> ProofGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ProofGenerator()
    return _generator_instance
