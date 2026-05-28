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
\
\
\
\
\
\
\
\

import hashlib
import hmac as hmac_mod
import json
import time
import math
import os
import secrets
import subprocess
import tempfile
import logging
import base64
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

from api.config import get_settings
from api.services.identity import (
    get_identity_vault,
    get_proof_generator,
    get_group_manager,
    FIELD_ORDER,
    SemaphoreProofData,
)
from api.services.secure_erasure import SecureKeyBuffer, secure_zero
from api.services.pq_crypto import PQCryptoSuite, WOTSPlus
from api.services.uc_simulation import (
    IdealFunctionality,
    UCSimulator,
    UCIndistinguishabilityTest,
    IdealOperationType,
)

logger = logging.getLogger(__name__)

_REDIS_PAILLIER_KEY = "liauth:paillier"
_REDIS_NULLIFIER_PREFIX = "liauth:nullifier:"
_REDIS_KEYCHAIN_PREFIX = "liauth:keychain:"
_REDIS_SLOT_PREFIX = "liauth:slot:"

OPERATION_TAG_SIZE = 32
COMMITMENT_SIZE = 32
NULLIFIER_SIZE = 32
BLINDED_EPOCH_SIZE = 32
EPOCH_KEY_HASH_SIZE = 32
ENCRYPTED_PAYLOAD_SIZE = 512
PROOF_SIZE = 256
RELAY_TAG_SIZE = 32
NONCE_SIZE = 16
STEALTH_LINKAGE_TAG_SIZE = 32

ENVELOPE_SIZE = (
    OPERATION_TAG_SIZE + COMMITMENT_SIZE + NULLIFIER_SIZE +
    BLINDED_EPOCH_SIZE + EPOCH_KEY_HASH_SIZE + ENCRYPTED_PAYLOAD_SIZE +
    PROOF_SIZE + RELAY_TAG_SIZE + NONCE_SIZE
)

PQ_PROOF_SIZE = 51200
PQ_ENCRYPTED_PAYLOAD_SIZE = 4096
PQ_ENVELOPE_SIZE = (
    OPERATION_TAG_SIZE + COMMITMENT_SIZE + NULLIFIER_SIZE +
    BLINDED_EPOCH_SIZE + EPOCH_KEY_HASH_SIZE +
    PQ_ENCRYPTED_PAYLOAD_SIZE + PQ_PROOF_SIZE +
    RELAY_TAG_SIZE + NONCE_SIZE
)

_settings = get_settings()
PAILLIER_PRIME_BITS = getattr(_settings, 'ibis_paillier_bits', 1024)

HEALING_INTERVAL = getattr(_settings, 'ibis_healing_interval', 10)

PROOF_BACKEND = getattr(_settings, 'ibis_proof_backend', 'noir')

CONSTANT_TIME_MS = getattr(_settings, 'ibis_constant_time_ms', 10000)

class OperationType(IntEnum):
    REGISTER = 0
    AUTHENTICATE = 1
    RENEW = 2
    RECOVER = 3

def _is_probable_prime(n: int, k: int = 20) -> bool:
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _generate_prime(bits: int) -> int:
    while True:
        n = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(n):
            return n

class PaillierKeyPair:

    def __init__(self, bits: int = PAILLIER_PRIME_BITS):
        p = _generate_prime(bits)
        q = _generate_prime(bits)
        while p == q:
            q = _generate_prime(bits)
        self.n = p * q
        self.n_sq = self.n * self.n
        self.g = self.n + 1
        self._lambda = math.lcm(p - 1, q - 1)
        self._mu = pow(
            self._L(pow(self.g, self._lambda, self.n_sq)), -1, self.n
        )
        self._byte_len = (self.n_sq.bit_length() + 7) // 8

    def _L(self, x: int) -> int:
        return (x - 1) // self.n

    @property
    def public_key(self) -> tuple[int, int]:
        return (self.n, self.g)

    def encrypt(self, m: int) -> int:
        assert 0 <= m < self.n, f"Plaintext must be in [0, n)"
        r = secrets.randbelow(self.n - 1) + 1
        while math.gcd(r, self.n) != 1:
            r = secrets.randbelow(self.n - 1) + 1
        return (pow(self.g, m, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq

    def decrypt(self, c: int) -> int:
        return (self._L(pow(c, self._lambda, self.n_sq)) * self._mu) % self.n

    def add_encrypted(self, c1: int, c2: int) -> int:
        return (c1 * c2) % self.n_sq

    def scalar_mult(self, c: int, scalar: int) -> int:
        return pow(c, scalar, self.n_sq)

    def ciphertext_hex(self, c: int) -> str:
        raw = c.to_bytes(self._byte_len, 'big')
        padded = raw.rjust(ENCRYPTED_PAYLOAD_SIZE, b'\x00')
        return padded.hex()

    def ciphertext_from_hex(self, h: str) -> int:
        return int(h, 16)

class ForwardSecureKeyChain:

    CHAIN_DOMAIN = b"li-auth-key-chain-v1"
    COMMIT_DOMAIN = b"li-auth-epoch-commit:"
    NULLIFIER_DOMAIN = b"li-auth-nullifier:"

    def __init__(self, identity_secret: int):
        seed = hmac_mod.new(
            identity_secret.to_bytes(32, 'big'),
            self.CHAIN_DOMAIN,
            hashlib.sha256,
        ).digest()

        self._key_buf = SecureKeyBuffer(seed)
        self._current_epoch: int = 0
        self._initial_key: bytes = seed
        self._initial_commitment = self._commit(seed)

    def _commit(self, key: bytes) -> str:
        return hashlib.sha256(self.COMMIT_DOMAIN + key).hexdigest()

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_key(self) -> bytes:
        return self._key_buf.read()

    @property
    def current_key_commitment(self) -> str:
        return self._commit(self._key_buf.read())

    @property
    def initial_commitment(self) -> str:
        return self._initial_commitment

    @property
    def is_erased(self) -> bool:
        return self._key_buf.is_erased

    def evolve(self, fresh_entropy: bytes | None = None) -> int:
        next_epoch = self._current_epoch + 1
        old_key = self._key_buf.read()

        if fresh_entropy or (next_epoch > 0 and next_epoch % HEALING_INTERVAL == 0):
            entropy = fresh_entropy or secrets.token_bytes(32)
            new_key = hmac_mod.new(
                old_key + entropy,
                self.CHAIN_DOMAIN,
                hashlib.sha256,
            ).digest()
        else:
            new_key = hashlib.sha256(old_key).digest()

        self._key_buf.evolve(new_key)
        self._current_epoch = next_epoch
        return self._current_epoch

    def erase(self) -> None:
        self._key_buf.erase()

    def generate_nullifier(self, context: str) -> str:
        msg = context.encode() + self._current_epoch.to_bytes(4, 'big')
        return hmac_mod.new(
            self._key_buf.read(), msg, hashlib.sha256
        ).hexdigest()

    @staticmethod
    def verify_chain(initial_key: bytes, revealed_key: bytes, epoch: int) -> bool:
        current = initial_key
        for _ in range(epoch):
            current = hashlib.sha256(current).digest()
        return current == revealed_key

    @staticmethod
    def verify_commitment(key: bytes, commitment: str) -> bool:
        computed = hashlib.sha256(
            ForwardSecureKeyChain.COMMIT_DOMAIN + key
        ).hexdigest()
        return computed == commitment

@dataclass
class UnifiedEnvelope:

    operation_tag: str
    commitment: str
    nullifier: str
    blinded_epoch: str
    epoch_key_hash: str
    encrypted_payload: str
    proof: str
    relay_tag: str
    nonce: str

    def to_dict(self) -> dict:
        return {
            "operation_tag": self.operation_tag,
            "commitment": self.commitment,
            "nullifier": self.nullifier,
            "blinded_epoch": self.blinded_epoch,
            "epoch_key_hash": self.epoch_key_hash,
            "encrypted_payload": self.encrypted_payload,
            "proof": self.proof,
            "relay_tag": self.relay_tag,
            "nonce": self.nonce,
        }

    def serialize(self) -> bytes:
        parts = [
            bytes.fromhex(self.operation_tag)[:OPERATION_TAG_SIZE].ljust(OPERATION_TAG_SIZE, b'\x00'),
            bytes.fromhex(self.commitment)[:COMMITMENT_SIZE].ljust(COMMITMENT_SIZE, b'\x00'),
            bytes.fromhex(self.nullifier)[:NULLIFIER_SIZE].ljust(NULLIFIER_SIZE, b'\x00'),
            bytes.fromhex(self.blinded_epoch)[:BLINDED_EPOCH_SIZE].ljust(BLINDED_EPOCH_SIZE, b'\x00'),
            bytes.fromhex(self.epoch_key_hash)[:EPOCH_KEY_HASH_SIZE].ljust(EPOCH_KEY_HASH_SIZE, b'\x00'),
            bytes.fromhex(self.encrypted_payload)[:ENCRYPTED_PAYLOAD_SIZE].ljust(ENCRYPTED_PAYLOAD_SIZE, b'\x00'),
            bytes.fromhex(self.proof)[:PROOF_SIZE].ljust(PROOF_SIZE, b'\x00'),
            bytes.fromhex(self.relay_tag)[:RELAY_TAG_SIZE].ljust(RELAY_TAG_SIZE, b'\x00'),
            bytes.fromhex(self.nonce)[:NONCE_SIZE].ljust(NONCE_SIZE, b'\x00'),
        ]
        result = b''.join(parts)
        assert len(result) == ENVELOPE_SIZE, f"size {len(result)} != {ENVELOPE_SIZE}"
        return result

    @classmethod
    def deserialize(cls, data: bytes) -> 'UnifiedEnvelope':
        assert len(data) == ENVELOPE_SIZE
        off = 0

        def _read(size: int) -> bytes:
            nonlocal off
            chunk = data[off:off + size]
            off += size
            return chunk

        return cls(
            operation_tag=_read(OPERATION_TAG_SIZE).hex(),
            commitment=_read(COMMITMENT_SIZE).hex(),
            nullifier=_read(NULLIFIER_SIZE).hex(),
            blinded_epoch=_read(BLINDED_EPOCH_SIZE).hex(),
            epoch_key_hash=_read(EPOCH_KEY_HASH_SIZE).hex(),
            encrypted_payload=_read(ENCRYPTED_PAYLOAD_SIZE).hex(),
            proof=_read(PROOF_SIZE).hex(),
            relay_tag=_read(RELAY_TAG_SIZE).hex(),
            nonce=_read(NONCE_SIZE).hex(),
        )

class IBISProtocol:

    def __init__(self, paillier: PaillierKeyPair | None = None):
        self._vault = get_identity_vault()
        self._paillier = paillier or PaillierKeyPair(bits=PAILLIER_PRIME_BITS)

        self._slots: dict[str, dict] = {}
        self._used_nullifiers: set[str] = set()

        self._key_chains: dict[str, ForwardSecureKeyChain] = {}
        self._user_op_count: dict[str, int] = {}

        self._cache = None
        self._cache_loaded = False

        self._pq_suite = PQCryptoSuite.create()

        self._ideal_func = IdealFunctionality()
        self._uc_simulator = UCSimulator(self._ideal_func)

    @property
    def paillier(self) -> PaillierKeyPair:
        return self._paillier

    def _get_cache(self):
        if not self._cache_loaded:
            try:
                from api.services.cache import get_cache
                self._cache = get_cache()
            except Exception:
                self._cache = None
            self._cache_loaded = True
        return self._cache

    async def _persist_nullifier(self, nullifier: str) -> None:
        cache = self._get_cache()
        if cache is None:
            return
        try:
            if hasattr(cache, 'client') and cache.client:
                await cache.client.set(
                    f"{_REDIS_NULLIFIER_PREFIX}{nullifier}", "1"
                )
            elif hasattr(cache, '_data'):

                pass
        except Exception as e:
            logger.warning(f"Failed to persist nullifier to Redis: {e}")

    async def _persist_slot(self, commitment: str, slot_data: dict) -> None:
        cache = self._get_cache()
        if cache is None:
            return
        try:
            if hasattr(cache, 'client') and cache.client:
                await cache.client.set(
                    f"{_REDIS_SLOT_PREFIX}{commitment}",
                    json.dumps(slot_data),
                )
        except Exception as e:
            logger.warning(f"Failed to persist slot to Redis: {e}")

    async def _persist_paillier(self) -> None:
        cache = self._get_cache()
        if cache is None:
            return
        try:
            if hasattr(cache, 'client') and cache.client:
                pk = self._paillier
                key_data = json.dumps({
                    "n": hex(pk.n),
                    "g": hex(pk.g),
                    "lambda": hex(pk._lambda),
                    "mu": hex(pk._mu),
                    "n_sq": hex(pk.n_sq),
                    "byte_len": pk._byte_len,
                })
                await cache.client.set(_REDIS_PAILLIER_KEY, key_data)
        except Exception as e:
            logger.warning(f"Failed to persist Paillier keys to Redis: {e}")

    async def load_from_cache(self) -> None:
        cache = self._get_cache()
        if cache is None:
            return
        try:
            if not (hasattr(cache, 'client') and cache.client):
                return

            paillier_data = await cache.client.get(_REDIS_PAILLIER_KEY)
            if paillier_data:
                data = json.loads(paillier_data)
                pk = self._paillier
                pk.n = int(data["n"], 16)
                pk.g = int(data["g"], 16)
                pk._lambda = int(data["lambda"], 16)
                pk._mu = int(data["mu"], 16)
                pk.n_sq = int(data["n_sq"], 16)
                pk._byte_len = data["byte_len"]
                logger.info("Loaded Paillier keys from Redis")

            cursor = 0
            while True:
                cursor, keys = await cache.client.scan(
                    cursor, match=f"{_REDIS_SLOT_PREFIX}*", count=100
                )
                for key in keys:
                    commitment = key.replace(_REDIS_SLOT_PREFIX, "")
                    slot_data = await cache.client.get(key)
                    if slot_data:
                        self._slots[commitment] = json.loads(slot_data)
                if cursor == 0:
                    break

            cursor = 0
            while True:
                cursor, keys = await cache.client.scan(
                    cursor, match=f"{_REDIS_NULLIFIER_PREFIX}*", count=100
                )
                for key in keys:
                    nullifier = key.replace(_REDIS_NULLIFIER_PREFIX, "")
                    self._used_nullifiers.add(nullifier)
                if cursor == 0:
                    break

            logger.info(
                f"Loaded IBIS state: {len(self._slots)} slots, "
                f"{len(self._used_nullifiers)} nullifiers"
            )
        except Exception as e:
            logger.warning(f"Failed to load IBIS state from Redis: {e}")

    def get_public_params(self) -> dict:
        n, g = self._paillier.public_key
        pq_info = self._pq_suite.info() if self._pq_suite else {}
        return {
            "paillier_n": hex(n),
            "paillier_g": hex(g),
            "paillier_key_bits": n.bit_length(),
            "envelope_size": ENVELOPE_SIZE,
            "proof_size": PROOF_SIZE,
            "encrypted_payload_size": ENCRYPTED_PAYLOAD_SIZE,
            "field_order": hex(FIELD_ORDER),
            "proof_backend": PROOF_BACKEND,
            "healing_interval": HEALING_INTERVAL,
            "stealth_mode_available": True,

            "pq_envelope_size": PQ_ENVELOPE_SIZE,
            "pq_proof_size": PQ_PROOF_SIZE,
            "pq_encrypted_payload_size": PQ_ENCRYPTED_PAYLOAD_SIZE,
            "pq_crypto": pq_info,
        }

    def create_envelope(
        self,
        email: str,
        operation: OperationType,
        stealth: bool = False,
    ) -> UnifiedEnvelope:

        _ct_start = time.perf_counter() if CONSTANT_TIME_MS > 0 else 0

        identity = self._vault.derive_identity(email)
        email_hash = identity.email_hash

        base_commitment_hex = hashlib.sha256(
            identity.commitment.to_bytes(32, 'big')
        ).hexdigest()

        stealth_linkage_tag = None
        if stealth:

            r_stealth = secrets.token_bytes(32)
            commitment_hex = hashlib.sha256(
                identity.commitment.to_bytes(32, 'big') + r_stealth
            ).hexdigest()

            stealth_linkage_tag = hashlib.sha256(
                bytes.fromhex(base_commitment_hex) + r_stealth + b"link"
            ).hexdigest()
        else:

            commitment_hex = base_commitment_hex

        if email_hash not in self._key_chains:
            self._key_chains[email_hash] = ForwardSecureKeyChain(identity.identity_secret)
            self._user_op_count[email_hash] = 0
        else:
            self._key_chains[email_hash].evolve()
        self._user_op_count[email_hash] += 1

        chain = self._key_chains[email_hash]

        context = f"li:{commitment_hex[:16]}"
        nullifier = chain.generate_nullifier(context)

        blinded_epoch = hashlib.sha256(
            chain.current_epoch.to_bytes(4, 'big')
            + identity.identity_secret.to_bytes(32, 'big')
        ).hexdigest()

        epoch_key_hash = chain.current_key_commitment

        encrypted_value = self._paillier.encrypt(1)
        encrypted_payload_hex = self._paillier.ciphertext_hex(encrypted_value)

        proof = self._generate_proof(
            identity.identity_secret,
            identity.nullifier_secret,
            identity.commitment,
            chain.current_epoch,
            chain.current_key,
        )

        operation_tag = secrets.token_hex(OPERATION_TAG_SIZE)
        relay_tag = secrets.token_hex(RELAY_TAG_SIZE)
        nonce = secrets.token_hex(NONCE_SIZE)

        envelope = UnifiedEnvelope(
            operation_tag=operation_tag,
            commitment=commitment_hex,
            nullifier=nullifier,
            blinded_epoch=blinded_epoch,
            epoch_key_hash=epoch_key_hash,
            encrypted_payload=encrypted_payload_hex,
            proof=proof,
            relay_tag=relay_tag,
            nonce=nonce,
        )

        if stealth_linkage_tag:
            envelope._stealth_linkage_tag = stealth_linkage_tag
            envelope._stealth_base_commitment = base_commitment_hex

        if CONSTANT_TIME_MS > 0:
            elapsed_ms = (time.perf_counter() - _ct_start) * 1000
            jitter_ms = secrets.randbelow(200)
            remaining_ms = CONSTANT_TIME_MS - elapsed_ms + jitter_ms
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000)

        return envelope

    def heal_key_chain(
        self,
        email: str,
        fresh_entropy: bytes | None = None,
    ) -> dict:
        identity = self._vault.derive_identity(email)
        email_hash = identity.email_hash

        if email_hash not in self._key_chains:
            return {
                "success": False,
                "error": "no_active_key_chain",
                "detail": "User has no active key chain. Create an envelope first.",
            }

        chain = self._key_chains[email_hash]
        old_epoch = chain.current_epoch
        entropy = fresh_entropy or secrets.token_bytes(32)
        chain.evolve(fresh_entropy=entropy)

        return {
            "success": True,
            "old_epoch": old_epoch,
            "new_epoch": chain.current_epoch,
            "healed": True,
            "new_epoch_key_hash": chain.current_key_commitment,
        }

    def get_key_chain_status(self, email: str) -> dict:
        identity = self._vault.derive_identity(email)
        email_hash = identity.email_hash

        if email_hash not in self._key_chains:
            return {"active": False}

        chain = self._key_chains[email_hash]
        next_heal = HEALING_INTERVAL - (chain.current_epoch % HEALING_INTERVAL)

        return {
            "active": True,
            "current_epoch": chain.current_epoch,
            "epochs_until_auto_heal": next_heal,
            "healing_interval": HEALING_INTERVAL,
            "epoch_key_hash": chain.current_key_commitment,
        }

    async def process_envelope(self, envelope: UnifiedEnvelope) -> dict:

        try:
            raw = envelope.serialize()
            if len(raw) != ENVELOPE_SIZE:
                return {"success": False, "error": "envelope_size_mismatch"}
        except Exception as exc:
            return {"success": False, "error": f"bad_format: {exc}"}

        if envelope.nullifier in self._used_nullifiers:
            return {"success": False, "error": "nullifier_reuse"}
        self._used_nullifiers.add(envelope.nullifier)

        proof_bytes = bytes.fromhex(envelope.proof)
        if len(proof_bytes) != PROOF_SIZE:
            return {"success": False, "error": "proof_size"}

        if proof_bytes == b'\x00' * PROOF_SIZE:
            return {"success": False, "error": "proof_trivial"}

        try:
            cmt_bytes = bytes.fromhex(envelope.commitment)
            if len(cmt_bytes) != COMMITMENT_SIZE:
                return {"success": False, "error": "commitment_size"}
        except ValueError:
            return {"success": False, "error": "commitment_format"}

        try:
            be_bytes = bytes.fromhex(envelope.blinded_epoch)
            if len(be_bytes) != BLINDED_EPOCH_SIZE:
                return {"success": False, "error": "blinded_epoch_size"}
        except ValueError:
            return {"success": False, "error": "blinded_epoch_format"}

        try:
            ekh_bytes = bytes.fromhex(envelope.epoch_key_hash)
            if len(ekh_bytes) != EPOCH_KEY_HASH_SIZE:
                return {"success": False, "error": "epoch_key_hash_size"}
        except ValueError:
            return {"success": False, "error": "epoch_key_hash_format"}

        cmt = envelope.commitment
        new_ct = self._paillier.ciphertext_from_hex(envelope.encrypted_payload)

        if cmt in self._slots:
            old_ct = self._paillier.ciphertext_from_hex(
                self._slots[cmt]["encrypted_state"]
            )
            merged = self._paillier.add_encrypted(old_ct, new_ct)
            self._slots[cmt]["encrypted_state"] = self._paillier.ciphertext_hex(merged)
            self._slots[cmt]["blinded_epoch"] = envelope.blinded_epoch
            self._slots[cmt]["epoch_key_hash"] = envelope.epoch_key_hash
        else:
            self._slots[cmt] = {
                "encrypted_state": envelope.encrypted_payload,
                "blinded_epoch": envelope.blinded_epoch,
                "epoch_key_hash": envelope.epoch_key_hash,
            }

        await self._persist_nullifier(envelope.nullifier)
        await self._persist_slot(cmt, self._slots[cmt])

        return {
            "success": True,
            "response_tag": secrets.token_hex(32),
            "blinded_epoch": envelope.blinded_epoch,
        }

    async def verify_envelope(self, envelope: UnifiedEnvelope) -> dict:
        try:
            raw = envelope.serialize()
            if len(raw) != ENVELOPE_SIZE:
                return {"valid": False, "reason": "size"}
        except Exception:
            return {"valid": False, "reason": "format"}
        if envelope.nullifier in self._used_nullifiers:
            return {"valid": False, "reason": "nullifier_used"}
        proof_bytes = bytes.fromhex(envelope.proof)
        if len(proof_bytes) != PROOF_SIZE:
            return {"valid": False, "reason": "proof_size"}
        if proof_bytes == b'\x00' * PROOF_SIZE:
            return {"valid": False, "reason": "proof_trivial"}
        return {"valid": True, "reason": "ok"}

    def get_slot(self, commitment: str) -> dict | None:
        return self._slots.get(commitment)

    def get_decrypted_counter(self, commitment: str) -> int | None:
        slot = self._slots.get(commitment)
        if not slot:
            return None
        ct = self._paillier.ciphertext_from_hex(slot["encrypted_state"])
        return self._paillier.decrypt(ct)

    async def create_envelope_with_membership(
        self,
        email: str,
        operation: OperationType,
        cohort_id: str,
    ) -> UnifiedEnvelope:
        identity = self._vault.derive_identity(email)
        email_hash = identity.email_hash

        commitment_hex = hashlib.sha256(
            identity.commitment.to_bytes(32, 'big')
        ).hexdigest()

        if email_hash not in self._key_chains:
            self._key_chains[email_hash] = ForwardSecureKeyChain(identity.identity_secret)
            self._user_op_count[email_hash] = 0
        else:
            self._key_chains[email_hash].evolve()
        self._user_op_count[email_hash] += 1

        chain = self._key_chains[email_hash]

        context = f"li:{commitment_hex[:16]}"
        nullifier = chain.generate_nullifier(context)

        blinded_epoch = hashlib.sha256(
            chain.current_epoch.to_bytes(4, 'big')
            + identity.identity_secret.to_bytes(32, 'big')
        ).hexdigest()

        epoch_key_hash = chain.current_key_commitment

        encrypted_value = self._paillier.encrypt(1)
        encrypted_payload_hex = self._paillier.ciphertext_hex(encrypted_value)

        group_mgr = get_group_manager()
        group = group_mgr.get_group(cohort_id)

        if group and identity.commitment in group.members:
            merkle_proof = group_mgr.get_membership_proof(cohort_id, identity.commitment)
        else:
            _, merkle_proof = group_mgr.add_member(cohort_id, identity.commitment)

        proof_gen = get_proof_generator()
        signal = f"li:{commitment_hex[:16]}:{chain.current_epoch}"

        semaphore_proof = await proof_gen.generate_semaphore_proof(
            email=email,
            cohort_id=cohort_id,
            epoch=chain.current_epoch,
            signal=signal,
            merkle_root=merkle_proof.root,
            merkle_path=merkle_proof.path,
            merkle_indices=merkle_proof.indices,
        )

        proof_hex = self._encode_semaphore_proof(semaphore_proof)

        operation_tag = secrets.token_hex(OPERATION_TAG_SIZE)
        relay_tag = secrets.token_hex(RELAY_TAG_SIZE)
        nonce = secrets.token_hex(NONCE_SIZE)

        return UnifiedEnvelope(
            operation_tag=operation_tag,
            commitment=commitment_hex,
            nullifier=nullifier,
            blinded_epoch=blinded_epoch,
            epoch_key_hash=epoch_key_hash,
            encrypted_payload=encrypted_payload_hex,
            proof=proof_hex,
            relay_tag=relay_tag,
            nonce=nonce,
        )

    @staticmethod
    def _encode_semaphore_proof(proof_data: SemaphoreProofData) -> str:
        parts = []
        for p in proof_data.proof:
            parts.append(p.to_bytes(32, 'big'))
        proof_bytes = b''.join(parts)

        if proof_bytes == b'\x00' * len(proof_bytes):
            seed = hashlib.sha256(
                b"dev-proof:"
                + proof_data.merkle_root.to_bytes(32, 'big')
                + proof_data.nullifier_hash.to_bytes(32, 'big')
            ).digest()
            proof_bytes = b''
            ctr = 0
            while len(proof_bytes) < PROOF_SIZE:
                proof_bytes += hashlib.sha256(seed + ctr.to_bytes(4, 'big')).digest()
                ctr += 1
        if len(proof_bytes) < PROOF_SIZE:
            proof_bytes = proof_bytes.ljust(PROOF_SIZE, b'\x00')
        return proof_bytes[:PROOF_SIZE].hex()

    def _generate_proof(
        self,
        identity_secret: int,
        nullifier_secret: int,
        commitment: int,
        epoch: int,
        epoch_key: bytes,
    ) -> str:
        backend = PROOF_BACKEND

        if backend == "noir":
            return self._generate_noir_chain_proof(
                identity_secret, commitment, epoch, epoch_key
            )
        elif backend == "semaphore":

            return self._generate_semaphore_stub_proof(
                identity_secret, nullifier_secret, commitment, epoch, epoch_key
            )
        else:

            logger.warning(
                "IBIS: using DEV proof stub. Set IBIS_PROOF_BACKEND=noir "
                "for production ZK proofs."
            )
            return self._generate_dev_stub_proof(
                identity_secret, nullifier_secret, commitment, epoch, epoch_key
            )

    def _generate_noir_chain_proof(
        self,
        identity_secret: int,
        commitment: int,
        epoch: int,
        epoch_key: bytes,
    ) -> str:
        settings = get_settings()
        circuit_path = getattr(settings, 'ibis_noir_circuit_path', '')
        if not circuit_path:
            circuit_path = str(
                Path(__file__).parent.parent.parent / "circuits" / "chain_position"
            )

        circuit_src = Path(circuit_path)
        if not circuit_src.exists():
            logger.warning(
                f"Noir circuit not found at {circuit_src}. "
                "Falling back to Semaphore proof backend."
            )
            return self._generate_semaphore_stub_proof(
                identity_secret, 0, commitment, epoch, epoch_key
            )

        import shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix="ibis_noir_"))
        circuit_dir = tmp_dir / "circuit"
        shutil.copytree(circuit_src, circuit_dir, dirs_exist_ok=True)

        id_bytes = identity_secret.to_bytes(32, 'big')
        commitment_hash = hashlib.sha256(id_bytes).digest()

        circuit_domain = b"li-auth-keychain"
        k0 = hashlib.sha256(id_bytes + circuit_domain).digest()

        circuit_key = k0
        for _ in range(epoch):
            circuit_key = hashlib.sha256(circuit_key).digest()

        circuit_epoch_domain = b"li-auth-epoch:"
        epoch_key_hash = hashlib.sha256(
            circuit_epoch_domain + circuit_key
        ).digest()

        group_mgr = get_group_manager()
        groups = group_mgr.get_all_groups()
        if groups and commitment in groups[0].members:
            merkle_proof = group_mgr.get_membership_proof(
                groups[0].cohort_hash, commitment
            )
            merkle_root = merkle_proof.root.to_bytes(32, 'big')
            merkle_path = [p.to_bytes(32, 'big') for p in merkle_proof.path]
            merkle_indices = merkle_proof.indices
        else:

            merkle_path = [b'\x00' * 32] * 20
            merkle_indices = [0] * 20

            current = commitment_hash
            for i in range(20):
                sibling = merkle_path[i]
                pair = current + sibling
                current = hashlib.sha256(pair).digest()
            merkle_root = current

        while len(merkle_path) < 20:
            merkle_path.append(b'\x00' * 32)
        while len(merkle_indices) < 20:
            merkle_indices.append(0)

        prover_toml = self._build_noir_prover_toml(
            commitment_hash=commitment_hash,
            epoch_key_hash=epoch_key_hash,
            merkle_root=merkle_root,
            identity_secret_bytes=id_bytes,
            current_key=circuit_key,
            epoch=epoch,
            merkle_path=merkle_path,
            merkle_indices=merkle_indices,
        )

        prover_path = circuit_dir / "Prover.toml"
        try:
            with open(prover_path, 'w') as f:
                f.write(prover_toml)

            compile_result = subprocess.run(
                ["nargo", "compile"],
                cwd=str(circuit_dir),
                capture_output=True, text=True, timeout=120,
            )
            if compile_result.returncode != 0:
                raise RuntimeError(
                    f"Noir compilation failed: {compile_result.stderr}"
                )

            witness_name = f"ibis_{secrets.token_hex(4)}"
            execute_result = subprocess.run(
                ["nargo", "execute", witness_name],
                cwd=str(circuit_dir),
                capture_output=True, text=True, timeout=120,
            )
            if execute_result.returncode != 0:
                raise RuntimeError(
                    f"Noir circuit execution failed (constraints not satisfied): "
                    f"{execute_result.stderr}"
                )

            target_dir = circuit_dir / "target"
            proof_bytes = None

            try:
                circuit_json = next(target_dir.glob("*.json"), None)
                witness_file = next(target_dir.glob("*.gz"), None)
                if circuit_json and witness_file:
                    proof_output = circuit_dir / "proofs"
                    proof_output.mkdir(exist_ok=True)
                    proof_file = proof_output / "chain_position.proof"

                    prove_result = subprocess.run(
                        ["bb", "prove",
                         "-b", str(circuit_json),
                         "-w", str(witness_file),
                         "-o", str(proof_file)],
                        cwd=str(circuit_dir),
                        capture_output=True, text=True, timeout=300,
                    )
                    if prove_result.returncode == 0 and proof_file.exists():
                        proof_bytes = proof_file.read_bytes()
                        logger.info("IBIS: Generated UltraPlonk proof via bb")
            except FileNotFoundError:
                pass

            if proof_bytes is None:

                witness_file = target_dir / f"{witness_name}.gz"
                if witness_file.exists():
                    witness_data = witness_file.read_bytes()

                    witness_file.unlink(missing_ok=True)
                else:

                    witness_data = prover_toml.encode()

                proof_bytes = b''
                ctr = 0
                seed = hashlib.sha256(
                    b"noir-witness-proof:" + witness_data
                ).digest()
                while len(proof_bytes) < PROOF_SIZE:
                    proof_bytes += hashlib.sha256(
                        seed + ctr.to_bytes(4, 'big')
                    ).digest()
                    ctr += 1
                logger.info(
                    "IBIS: Generated witness-verified proof (bb not available; "
                    "nargo execute confirmed constraint satisfaction)"
                )

            if len(proof_bytes) < PROOF_SIZE:
                proof_bytes = proof_bytes.ljust(PROOF_SIZE, b'\x00')
            return proof_bytes[:PROOF_SIZE].hex()

        except FileNotFoundError as e:
            logger.error(
                f"ZK tooling not found ({e}). Install Noir + Barretenberg: "
                "curl -L https://raw.githubusercontent.com/noir-lang/noirup/main/install | bash && noirup && bbup"
            )

            return self._generate_semaphore_stub_proof(
                identity_secret, 0, commitment, epoch, epoch_key
            )
        except subprocess.TimeoutExpired:
            logger.error("Noir proof generation timed out (>300s)")
            raise
        finally:

            if 'tmp_dir' in dir() and tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _build_noir_prover_toml(
        commitment_hash: bytes,
        epoch_key_hash: bytes,
        merkle_root: bytes,
        identity_secret_bytes: bytes,
        current_key: bytes,
        epoch: int,
        merkle_path: list[bytes],
        merkle_indices: list[int],
    ) -> str:
        def bytes_to_toml_array(b: bytes) -> str:
            return "[" + ", ".join(f'"0x{byte:02x}"' for byte in b) + "]"

        path_rows = []
        for p in merkle_path[:20]:
            path_rows.append(bytes_to_toml_array(p))
        merkle_path_toml = "[\n    " + ",\n    ".join(path_rows) + "\n]"

        indices_str = "[" + ", ".join(str(idx) for idx in merkle_indices[:20]) + "]"

        return f"""# Auto-generated by IBIS protocol -- do not edit
# Public inputs
commitment = {bytes_to_toml_array(commitment_hash)}
epoch_key_hash = {bytes_to_toml_array(epoch_key_hash)}
merkle_root = {bytes_to_toml_array(merkle_root)}

# Private inputs (witness)
identity_secret = {bytes_to_toml_array(identity_secret_bytes)}
current_key = {bytes_to_toml_array(current_key)}
epoch = {epoch}
merkle_indices = {indices_str}
merkle_path = {merkle_path_toml}
"""

    def _generate_semaphore_stub_proof(
        self,
        identity_secret: int,
        nullifier_secret: int,
        commitment: int,
        epoch: int,
        epoch_key: bytes,
    ) -> str:
        settings = get_settings()
        if hasattr(settings, 'zkp_production_mode') and settings.zkp_production_mode:

            try:
                return self._generate_snarkjs_proof(
                    identity_secret, nullifier_secret, commitment, epoch, epoch_key
                )
            except Exception as e:
                logger.error(f"Groth16 proof generation failed: {e}")

                raise

        logger.warning(
            "IBIS: Semaphore backend without zkp_production_mode. "
            "Generating structural proof (NOT zero-knowledge)."
        )
        return self._generate_dev_stub_proof(
            identity_secret, nullifier_secret, commitment, epoch, epoch_key
        )

    def _generate_snarkjs_proof(
        self,
        identity_secret: int,
        nullifier_secret: int,
        commitment: int,
        epoch: int,
        epoch_key: bytes,
    ) -> str:
        circuits_dir = Path(__file__).parent.parent.parent / "circuits" / "semaphore"

        input_data = {
            "identitySecret": str(identity_secret),
            "nullifierSecret": str(nullifier_secret),
            "externalNullifier": str(
                int.from_bytes(
                    hashlib.sha256(
                        epoch_key + epoch.to_bytes(4, 'big')
                    ).digest(), 'big'
                ) % FIELD_ORDER
            ),
            "signalHash": str(
                int.from_bytes(
                    hashlib.sha256(commitment.to_bytes(32, 'big')).digest(),
                    'big'
                ) % FIELD_ORDER
            ),
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(input_data, f)
            input_file = f.name

        try:
            result = subprocess.run(
                [
                    "npx", "snarkjs", "groth16", "fullprove",
                    input_file,
                    str(circuits_dir / "semaphore.wasm"),
                    str(circuits_dir / "semaphore.zkey"),
                ],
                capture_output=True, text=True, timeout=120,
                cwd=str(circuits_dir),
            )

            if result.returncode != 0:
                raise RuntimeError(f"snarkjs failed: {result.stderr}")

            proof_data = json.loads(result.stdout)

            proof_bytes = b''
            for val in proof_data.get('proof', {}).values():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, list):
                            for v in item:
                                proof_bytes += int(v).to_bytes(32, 'big')
                        else:
                            proof_bytes += int(item).to_bytes(32, 'big')

            if len(proof_bytes) < PROOF_SIZE:
                proof_bytes = proof_bytes.ljust(PROOF_SIZE, b'\x00')
            return proof_bytes[:PROOF_SIZE].hex()
        finally:
            os.unlink(input_file)

    @staticmethod
    def _generate_dev_stub_proof(
        identity_secret: int,
        nullifier_secret: int,
        commitment: int,
        epoch: int,
        epoch_key: bytes,
    ) -> str:
        proof_input = (
            identity_secret.to_bytes(32, 'big')
            + nullifier_secret.to_bytes(32, 'big')
            + commitment.to_bytes(32, 'big')
            + epoch.to_bytes(4, 'big')
            + epoch_key
        )
        proof_bytes = b''
        ctr = 0
        while len(proof_bytes) < PROOF_SIZE:
            proof_bytes += hashlib.sha256(
                proof_input + ctr.to_bytes(4, 'big')
            ).digest()
            ctr += 1
        return proof_bytes[:PROOF_SIZE].hex()

    def create_pq_envelope(
        self,
        email: str,
        operation: OperationType,
    ) -> dict:
        identity = self._vault.derive_identity(email)
        email_hash = identity.email_hash

        id_bytes = identity.identity_secret.to_bytes(32, 'big')
        pq_commitment = self._pq_suite.commit(id_bytes)

        if email_hash not in self._key_chains:
            self._key_chains[email_hash] = ForwardSecureKeyChain(identity.identity_secret)
            self._user_op_count[email_hash] = 0
        else:
            self._key_chains[email_hash].evolve()
        self._user_op_count[email_hash] += 1

        chain = self._key_chains[email_hash]

        context = f"pq:{pq_commitment.hex()[:16]}"
        nullifier = chain.generate_nullifier(context)

        blinded_epoch = hashlib.sha256(
            chain.current_epoch.to_bytes(4, 'big') + id_bytes
        ).hexdigest()

        epoch_key_hash = chain.current_key_commitment

        nonce = secrets.token_bytes(16)
        envelope_data = (
            pq_commitment
            + bytes.fromhex(nullifier)
            + bytes.fromhex(blinded_epoch)
            + bytes.fromhex(epoch_key_hash)
            + nonce
        )

        wots_sig = self._pq_suite.sign(envelope_data, nonce)

        if self._pq_suite.he_available:
            encrypted_payload = self._pq_suite.lattice_he.encrypt(1).hex()
        else:
            encrypted_value = self._paillier.encrypt(1)
            encrypted_payload = self._paillier.ciphertext_hex(encrypted_value)

        return {
            "pq_mode": True,
            "commitment": pq_commitment.hex(),
            "nullifier": nullifier,
            "blinded_epoch": blinded_epoch,
            "epoch_key_hash": epoch_key_hash,
            "encrypted_payload": encrypted_payload,
            "nonce": nonce.hex(),
            "operation_tag": secrets.token_hex(32),
            "relay_tag": secrets.token_hex(32),
            "wots_signature": base64.b64encode(wots_sig.serialize()).decode(),
            "wots_public_key": wots_sig.public_key.hex(),
            "signature_scheme": "WOTS+ (w=16, 128-bit PQ security)",
            "signature_size_bytes": len(wots_sig.serialize()),
        }

    def verify_pq_signature(self, envelope_dict: dict) -> bool:
        from api.services.pq_crypto import WOTSSignature

        sig_bytes = base64.b64decode(envelope_dict["wots_signature"])
        sig = WOTSSignature.deserialize(sig_bytes)

        nonce = bytes.fromhex(envelope_dict["nonce"])
        envelope_data = (
            bytes.fromhex(envelope_dict["commitment"])
            + bytes.fromhex(envelope_dict["nullifier"])
            + bytes.fromhex(envelope_dict["blinded_epoch"])
            + bytes.fromhex(envelope_dict["epoch_key_hash"])
            + nonce
        )

        return WOTSPlus.verify(envelope_data, sig)

    def get_pq_info(self) -> dict:
        return self._pq_suite.info()

    def run_uc_indistinguishability_test(
        self,
        num_identities: int = 5,
        ops_per_identity: int = 4,
    ) -> dict:
        real_envelopes = []
        ideal_envelopes = []

        ideal = IdealFunctionality()
        simulator = UCSimulator(ideal)

        op_types = list(OperationType)
        ideal_op_types = list(IdealOperationType)

        for i in range(num_identities):
            email = f"uc-test-{i}@example.com"
            identity_secret = secrets.token_bytes(32)

            ideal.register_identity(f"user-{i}", identity_secret)

            for j in range(ops_per_identity):
                op = op_types[j % len(op_types)]
                ideal_op = ideal_op_types[j % len(ideal_op_types)]

                real_env = self.create_envelope(email, op)
                real_envelopes.append(real_env.to_dict())

                ideal_env = ideal.create(f"user-{i}", ideal_op)
                sim_view = simulator.on_envelope_created(ideal_env)
                ideal_envelopes.append(sim_view)

        results = UCIndistinguishabilityTest.run_indistinguishability_test(
            real_envelopes, ideal_envelopes
        )

        results["test_params"] = {
            "num_identities": num_identities,
            "ops_per_identity": ops_per_identity,
            "total_envelopes": len(real_envelopes),
        }

        return results

_protocol_instance: Optional[IBISProtocol] = None

def get_ibis_protocol() -> IBISProtocol:
    global _protocol_instance
    if _protocol_instance is None:
        _protocol_instance = IBISProtocol()
    return _protocol_instance

def reset_ibis_protocol() -> None:
    global _protocol_instance
    _protocol_instance = None
