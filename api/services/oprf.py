import hashlib
import hmac as hmac_mod
import secrets
import logging
from dataclasses import dataclass
from typing import Optional

from ecpy.curves import Curve, Point

logger = logging.getLogger(__name__)

_CURVE = Curve.get_curve('secp256r1')
_G = _CURVE.generator
_ORDER = _CURVE.order
_FIELD_PRIME = _CURVE.field
_CURVE_A = _CURVE.a
_CURVE_B = _CURVE.b

_HASH_TO_CURVE_DST = b"IBIS-OPRF-P256-SHA256-SSWU-RO-V1"

_FINALIZE_DST = b"IBIS-OPRF-P256-Finalize-V1"

_KEY_DERIVE_DST = b"IBIS-OPRF-P256-KeyDerive-V1"

def hash_to_curve(data: bytes, dst: bytes = _HASH_TO_CURVE_DST) -> Point:
    for counter in range(256):
        hash_input = dst + data + counter.to_bytes(4, 'big')
        h = hashlib.sha256(hash_input).digest()
        x = int.from_bytes(h, 'big') % _FIELD_PRIME

        rhs = (pow(x, 3, _FIELD_PRIME) + _CURVE_A * x + _CURVE_B) % _FIELD_PRIME

        if pow(rhs, (_FIELD_PRIME - 1) // 2, _FIELD_PRIME) != 1:
            continue

        y = pow(rhs, (_FIELD_PRIME + 1) // 4, _FIELD_PRIME)

        if (y * y) % _FIELD_PRIME != rhs:
            continue

        if y % 2 != 0:
            y = _FIELD_PRIME - y

        point = Point(x, y, _CURVE)
        if not _CURVE.is_on_curve(point):
            continue

        return point

    raise ValueError(f"hash_to_curve failed after 256 iterations for input of length {len(data)}")

def _serialize_point(point: Point) -> bytes:
    x_bytes = point.x.to_bytes(32, 'big')
    y_bytes = point.y.to_bytes(32, 'big')
    return b'\x04' + x_bytes + y_bytes

def _deserialize_point(data: bytes) -> Point:
    if len(data) != 65 or data[0] != 0x04:
        raise ValueError(f"Invalid uncompressed point: expected 65 bytes starting with 0x04, got {len(data)} bytes")
    x = int.from_bytes(data[1:33], 'big')
    y = int.from_bytes(data[33:65], 'big')
    point = Point(x, y, _CURVE)
    if not _CURVE.is_on_curve(point):
        raise ValueError("Deserialized point is not on P-256")
    return point

def _finalize(output_element: Point, input_data: bytes) -> bytes:
    element_bytes = _serialize_point(output_element)
    input_len = len(input_data).to_bytes(2, 'big')
    element_len = len(element_bytes).to_bytes(2, 'big')

    return hashlib.sha256(
        input_len + input_data + element_len + element_bytes + _FINALIZE_DST
    ).digest()

@dataclass
class OPRFKeyPair:
    scalar: int
    public_key: Point

    @classmethod
    def generate(cls) -> 'OPRFKeyPair':
        k = secrets.randbelow(_ORDER - 1) + 1
        K = k * _G
        return cls(scalar=k, public_key=K)

    @classmethod
    def from_master_key(cls, master_key: bytes) -> 'OPRFKeyPair':
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes as crypto_hashes

        hkdf = HKDF(
            algorithm=crypto_hashes.SHA256(),
            length=32,
            salt=_KEY_DERIVE_DST,
            info=b"IBIS-OPRF-Scalar",
        )
        k_bytes = hkdf.derive(master_key)
        k = int.from_bytes(k_bytes, 'big') % _ORDER
        if k == 0:
            k = 1

        K = k * _G
        return cls(scalar=k, public_key=K)

    def serialize_public_key(self) -> bytes:
        return _serialize_point(self.public_key)

class OPRFServer:
    def __init__(self, key_pair: OPRFKeyPair):
        self._key = key_pair

    @property
    def public_key(self) -> Point:
        return self._key.public_key

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.serialize_public_key()

    def blind_evaluate(self, blinded_element: Point) -> Point:
        if not _CURVE.is_on_curve(blinded_element):
            raise ValueError("Blinded element is not on P-256")
        return self._key.scalar * blinded_element

    def full_evaluate(self, input_data: bytes) -> bytes:
        H_input = hash_to_curve(input_data)
        output_element = self._key.scalar * H_input
        return _finalize(output_element, input_data)

class OPRFClient:
    def __init__(self):
        self._r: Optional[int] = None
        self._input_data: Optional[bytes] = None

    def blind(self, input_data: bytes) -> Point:
        self._input_data = input_data
        self._r = secrets.randbelow(_ORDER - 1) + 1

        P = hash_to_curve(input_data)
        blinded_element = self._r * P

        return blinded_element

    def finalize(self, evaluated_element: Point) -> bytes:
        if self._r is None or self._input_data is None:
            raise ValueError("Must call blind() before finalize()")

        r_inv = pow(self._r, _ORDER - 2, _ORDER)

        output_element = r_inv * evaluated_element

        output = _finalize(output_element, self._input_data)

        self._r = None
        self._input_data = None

        return output

class OPRFIdentityDeriver:
    def __init__(self, server: OPRFServer):
        self._server = server

    def derive_server_side(self, email: str) -> tuple[int, int]:
        from api.services.identity import FIELD_ORDER
        normalized = email.lower().strip()

        id_output = self._server.full_evaluate(
            b"identity:" + normalized.encode()
        )
        null_output = self._server.full_evaluate(
            b"nullifier:" + normalized.encode()
        )

        identity_secret = int.from_bytes(id_output, 'big') % FIELD_ORDER
        nullifier_secret = int.from_bytes(null_output, 'big') % FIELD_ORDER

        return identity_secret, nullifier_secret

    def create_blind_request(self, email: str) -> dict:
        normalized = email.lower().strip()

        client_id = OPRFClient()
        blinded_id = client_id.blind(b"identity:" + normalized.encode())

        client_null = OPRFClient()
        blinded_null = client_null.blind(b"nullifier:" + normalized.encode())

        return {
            "blinded_identity": _serialize_point(blinded_id).hex(),
            "blinded_nullifier": _serialize_point(blinded_null).hex(),
            "_client_id": client_id,
            "_client_null": client_null,
        }

    def server_evaluate(self, blinded_identity_hex: str, blinded_nullifier_hex: str) -> dict:
        blinded_id = _deserialize_point(bytes.fromhex(blinded_identity_hex))
        blinded_null = _deserialize_point(bytes.fromhex(blinded_nullifier_hex))

        evaluated_id = self._server.blind_evaluate(blinded_id)
        evaluated_null = self._server.blind_evaluate(blinded_null)

        return {
            "evaluated_identity": _serialize_point(evaluated_id).hex(),
            "evaluated_nullifier": _serialize_point(evaluated_null).hex(),
        }

    def client_finalize(
        self,
        client_id: OPRFClient,
        client_null: OPRFClient,
        evaluated_identity_hex: str,
        evaluated_nullifier_hex: str,
    ) -> tuple[int, int]:
        from api.services.identity import FIELD_ORDER

        evaluated_id = _deserialize_point(bytes.fromhex(evaluated_identity_hex))
        evaluated_null = _deserialize_point(bytes.fromhex(evaluated_nullifier_hex))

        id_output = client_id.finalize(evaluated_id)
        null_output = client_null.finalize(evaluated_null)

        identity_secret = int.from_bytes(id_output, 'big') % FIELD_ORDER
        nullifier_secret = int.from_bytes(null_output, 'big') % FIELD_ORDER

        return identity_secret, nullifier_secret

_oprf_server: Optional[OPRFServer] = None

def get_oprf_server(master_key: Optional[bytes] = None) -> OPRFServer:
    global _oprf_server
    if _oprf_server is None:
        if master_key is None:
            from api.config import get_settings
            settings = get_settings()
            mk = getattr(settings, 'identity_master_key', None)
            if mk:
                master_key = bytes.fromhex(mk)
            else:
                master_key = secrets.token_bytes(32)
                logger.warning("OPRF: using random master key (no identity_master_key configured)")

        key_pair = OPRFKeyPair.from_master_key(master_key)
        _oprf_server = OPRFServer(key_pair)
        logger.info(
            f"OPRF server initialized on P-256, "
            f"public key: {_oprf_server.public_key_bytes.hex()[:32]}..."
        )
    return _oprf_server

def get_oprf_deriver(master_key: Optional[bytes] = None) -> OPRFIdentityDeriver:
    server = get_oprf_server(master_key)
    return OPRFIdentityDeriver(server)
