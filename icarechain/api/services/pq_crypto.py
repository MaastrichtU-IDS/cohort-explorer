import hashlib
import hmac as hmac_mod
import secrets
import struct
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

class PQCommitment:
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
    DOMAIN = b"pq-commit-v1:"

    @staticmethod
    def commit(secret: bytes, randomness: bytes = b"") -> bytes:
        \
        return hashlib.sha256(
            PQCommitment.DOMAIN + secret + randomness
        ).digest()

    @staticmethod
    def verify(secret: bytes, commitment: bytes, randomness: bytes = b"") -> bool:
        \
        return PQCommitment.commit(secret, randomness) == commitment

@dataclass
class WOTSSignature:
    \
    signature_chains: list[bytes]
    public_key: bytes

    def serialize(self) -> bytes:
        \
        return b''.join(self.signature_chains) + self.public_key

    @classmethod
    def deserialize(cls, data: bytes, chain_count: int = 67) -> 'WOTSSignature':
        chains = []
        for i in range(chain_count):
            chains.append(data[i * 32:(i + 1) * 32])
        pk = data[chain_count * 32:chain_count * 32 + 32]
        return cls(signature_chains=chains, public_key=pk)

class WOTSPlus:
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
\
\
\
\
\
    N = 32
    W = 16
    LOG_W = 4
    L1 = 64
    L2 = 3
    L = 67

    def __init__(self, seed: bytes | None = None):
        \
        self._seed = seed or secrets.token_bytes(self.N)
        self._private_chains = self._generate_private_key()
        self._public_key = self._compute_public_key()
        self._used = False

    def _hash_chain(self, start: bytes, steps: int, addr: int) -> bytes:
        \
        current = start
        for i in range(steps):
            current = hashlib.sha256(
                current + struct.pack(">I", addr) + struct.pack(">I", i)
            ).digest()
        return current

    def _generate_private_key(self) -> list[bytes]:
        \
        chains = []
        for i in range(self.L):
            ki = hmac_mod.new(
                self._seed,
                struct.pack(">I", i) + b"wots-sk",
                hashlib.sha256,
            ).digest()
            chains.append(ki)
        return chains

    def _compute_public_key(self) -> bytes:
        \
        pk_elements = []
        for i in range(self.L):
            pk_i = self._hash_chain(self._private_chains[i], self.W - 1, i)
            pk_elements.append(pk_i)

        return hashlib.sha256(b''.join(pk_elements)).digest()

    @property
    def public_key(self) -> bytes:
        return self._public_key

    def sign(self, message: bytes) -> WOTSSignature:
        \
\
\
\
\
\
\
        if self._used:
            raise RuntimeError(
                "WOTS+ key already used. One-time signatures must not be reused."
            )
        self._used = True

        msg_hash = hashlib.sha256(message).digest()

        msg_digits = self._bytes_to_base_w(msg_hash, self.L1)

        checksum = sum(self.W - 1 - d for d in msg_digits)
        checksum_bytes = struct.pack(">I", checksum)
        checksum_digits = self._bytes_to_base_w(checksum_bytes, self.L2)

        all_digits = msg_digits + checksum_digits

        sig_chains = []
        for i in range(self.L):
            sig_i = self._hash_chain(self._private_chains[i], all_digits[i], i)
            sig_chains.append(sig_i)

        return WOTSSignature(
            signature_chains=sig_chains,
            public_key=self._public_key,
        )

    @staticmethod
    def verify(message: bytes, signature: WOTSSignature) -> bool:
        \
\
\
\
\
\
        msg_hash = hashlib.sha256(message).digest()
        msg_digits = WOTSPlus._bytes_to_base_w(msg_hash, WOTSPlus.L1)

        checksum = sum(WOTSPlus.W - 1 - d for d in msg_digits)
        checksum_bytes = struct.pack(">I", checksum)
        checksum_digits = WOTSPlus._bytes_to_base_w(checksum_bytes, WOTSPlus.L2)

        all_digits = msg_digits + checksum_digits

        pk_elements = []
        for i in range(WOTSPlus.L):
            remaining = WOTSPlus.W - 1 - all_digits[i]
            pk_i = signature.signature_chains[i]
            for j in range(remaining):
                pk_i = hashlib.sha256(
                    pk_i
                    + struct.pack(">I", i)
                    + struct.pack(">I", all_digits[i] + j)
                ).digest()
            pk_elements.append(pk_i)

        computed_pk = hashlib.sha256(b''.join(pk_elements)).digest()
        return computed_pk == signature.public_key

    @staticmethod
    def _bytes_to_base_w(data: bytes, out_len: int) -> list[int]:
        \
        digits = []
        for byte in data:
            digits.append((byte >> 4) & 0x0F)
            digits.append(byte & 0x0F)
            if len(digits) >= out_len:
                break
        return digits[:out_len]

class PQEnvelopeSigner:
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
    def __init__(self, master_seed: bytes | None = None):
        self._master_seed = master_seed or secrets.token_bytes(32)

    def sign_envelope(self, envelope_bytes: bytes, nonce: bytes) -> WOTSSignature:
        \
\
\
\
        derived_seed = hmac_mod.new(
            self._master_seed,
            nonce + b"wots-envelope-key",
            hashlib.sha256,
        ).digest()
        wots = WOTSPlus(seed=derived_seed)
        return wots.sign(envelope_bytes)

    def get_verifier_key(self, nonce: bytes) -> bytes:
        \
        derived_seed = hmac_mod.new(
            self._master_seed,
            nonce + b"wots-envelope-key",
            hashlib.sha256,
        ).digest()
        wots = WOTSPlus(seed=derived_seed)
        return wots.public_key

    @staticmethod
    def verify_envelope(
        envelope_bytes: bytes, signature: WOTSSignature
    ) -> bool:
        \
        return WOTSPlus.verify(envelope_bytes, signature)

class LatticeHEAdapter:
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
    def __init__(self):
        self._tenseal = None
        self._context = None
        try:
            import tenseal
            self._tenseal = tenseal
            self._context = tenseal.context(
                tenseal.SCHEME_TYPE.BFV,
                poly_modulus_degree=4096,
                plain_modulus=1032193,
                coeff_mod_bit_sizes=[40, 20, 40],
            )
            self._context.generate_galois_keys()
            logger.info("Lattice HE (BFV via tenseal) initialized")
        except ImportError:
            logger.info(
                "tenseal not available. Lattice HE adapter in interface-only mode. "
                "Install: pip install tenseal"
            )

    @property
    def available(self) -> bool:
        return self._tenseal is not None

    def encrypt(self, value: int) -> bytes:
        \
        if not self._tenseal:
            raise RuntimeError(
                "Lattice HE requires tenseal. Install: pip install tenseal"
            )
        encrypted = self._tenseal.bfv_vector(self._context, [value])
        return encrypted.serialize()

    def decrypt(self, ciphertext: bytes) -> int:
        \
        if not self._tenseal:
            raise RuntimeError("Lattice HE requires tenseal")
        encrypted = self._tenseal.bfv_vector_from(self._context, ciphertext)
        return encrypted.decrypt()[0]

    def add_encrypted(self, ct1: bytes, ct2: bytes) -> bytes:
        \
        if not self._tenseal:
            raise RuntimeError("Lattice HE requires tenseal")
        e1 = self._tenseal.bfv_vector_from(self._context, ct1)
        e2 = self._tenseal.bfv_vector_from(self._context, ct2)
        result = e1 + e2
        return result.serialize()

    def ciphertext_size(self) -> int:
        \
        return 4096

@dataclass
class PQCryptoSuite:
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
    commitment: PQCommitment = field(default_factory=PQCommitment)
    signer: PQEnvelopeSigner = field(default_factory=PQEnvelopeSigner)
    lattice_he: LatticeHEAdapter = field(default_factory=LatticeHEAdapter)

    @classmethod
    def create(cls, master_seed: bytes | None = None) -> 'PQCryptoSuite':
        return cls(
            commitment=PQCommitment(),
            signer=PQEnvelopeSigner(master_seed=master_seed),
            lattice_he=LatticeHEAdapter(),
        )

    def commit(self, secret: bytes, randomness: bytes = b"") -> bytes:
        return self.commitment.commit(secret, randomness)

    def sign(self, data: bytes, nonce: bytes) -> WOTSSignature:
        return self.signer.sign_envelope(data, nonce)

    def verify_sig(self, data: bytes, sig: WOTSSignature) -> bool:
        return self.signer.verify_envelope(data, sig)

    @property
    def he_available(self) -> bool:
        return self.lattice_he.available

    def info(self) -> dict:
        return {
            "commitment_scheme": "SHA-256 (128-bit post-quantum)",
            "signature_scheme": "WOTS+ (w=16, 128-bit post-quantum)",
            "signature_size_bytes": WOTSPlus.L * WOTSPlus.N + WOTSPlus.N,
            "he_scheme": "BFV (lattice-based)" if self.he_available else "interface-only (install tenseal)",
            "he_ciphertext_size_bytes": self.lattice_he.ciphertext_size(),
            "quantum_safe_components": [
                "SHA-256 key chain (Grover: 128-bit)",
                "HMAC-SHA256 nullifiers (quantum-safe)",
                "SHA-256 commitments (quantum-safe)",
                "WOTS+ signatures (hash-based, quantum-safe)",
            ],
            "migration_status": {
                "commitments": "DONE (SHA-256)",
                "signatures": "DONE (WOTS+)",
                "he": "READY" if self.he_available else "INTERFACE_DEFINED",
                "proofs": "PLANNED (STARK via hash-based FRI)",
            },
        }
