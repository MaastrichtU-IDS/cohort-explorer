import hashlib
import hmac
import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from eth_account import Account
from web3 import Web3

from api.config import get_settings

logger = logging.getLogger(__name__)

BN254_FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def _salt_bytes() -> bytes:
    raw = (get_settings().derivation_salt or "").strip()
    if not raw:
        return b""
    stripped = raw[2:] if raw.startswith("0x") else raw
    try:
        return bytes.fromhex(stripped)
    except ValueError:
        return Web3.keccak(text=raw)

_EOA_KEY_DOMAIN = b"icare-eoa-key|"
_ROLE_COMMIT_DOMAIN = b"icare-role-commit|"
_IDENTITY_DOMAIN = b"icare-identity|"
_NULLIFIER_DOMAIN = b"icare-nullifier|"

def _hmac(domain: bytes, msg: bytes) -> bytes:
    return hmac.new(_salt_bytes(), domain + msg, hashlib.sha256).digest()

def derive_private_key_from_email(email: str) -> bytes:
    return _hmac(_EOA_KEY_DOMAIN, email.lower().strip().encode())

def derive_role_commitment(email: str, role: str) -> bytes:
    msg = role.upper().encode() + b"|" + email.lower().strip().encode()
    return _hmac(_ROLE_COMMIT_DOMAIN, msg)

def derive_identity_secret(email: str) -> bytes:
    return _hmac(_IDENTITY_DOMAIN, email.lower().strip().encode())

def derive_nullifier_secret(email: str) -> bytes:
    return _hmac(_NULLIFIER_DOMAIN, email.lower().strip().encode())

def derive_identity_field(email: str) -> int:

    return int.from_bytes(derive_identity_secret(email), "big") % BN254_FIELD_PRIME

def _circuits_dir() -> Path:
    return Path(__file__).parent.parent.parent / "circuits"

def _ensure_helper_compiled() -> Path:

    tmp_root = Path(os.environ.get("ICARE_NARGO_TMP", "/tmp/icare_circuits"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    dest = tmp_root / "identity_helper"
    src = _circuits_dir() / "identity_helper"
    if not (dest / "target" / "identity_helper.json").exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        env = os.environ.copy()
        env["PATH"] = "/root/.nargo/bin:/root/.bb:" + env.get("PATH", "")
        result = subprocess.run(
            ["nargo", "compile"],
            cwd=str(dest), env=env, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"identity_helper compile failed: {result.stderr}")
    return dest

@lru_cache(maxsize=4096)
def _poseidon2_commit_field(secret_field: int) -> tuple[int, int]:

    dest = _ensure_helper_compiled()
    env = os.environ.copy()
    env["PATH"] = "/root/.nargo/bin:/root/.bb:" + env.get("PATH", "")
    with tempfile.TemporaryDirectory() as wd:
        run_dir = Path(wd) / "run"
        shutil.copytree(dest, run_dir)
        prover = run_dir / "Prover.toml"
        prover.write_text(f'secret = "{secret_field}"\n')
        result = subprocess.run(
            ["nargo", "execute", "w"],
            cwd=str(run_dir), env=env, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"identity_helper execute failed: {result.stderr}")

        out = result.stdout
        marker = "Circuit output: ["
        idx = out.find(marker)
        if idx < 0:
            raise RuntimeError(f"unexpected nargo output: {out}")
        rest = out[idx + len(marker):]
        end = rest.find("]")
        if end < 0:
            raise RuntimeError(f"unparseable nargo output: {out}")
        parts = [p.strip() for p in rest[:end].split(",")]
        if len(parts) != 2:
            raise RuntimeError(f"expected 2 outputs, got {parts}")
        return int(parts[0], 16), int(parts[1], 16)

def derive_user_commitment(email: str) -> bytes:

    secret_field = derive_identity_field(email)
    commitment, _k0 = _poseidon2_commit_field(secret_field)
    return commitment.to_bytes(32, "big")

def derive_user_chain_k0(email: str) -> bytes:
    secret_field = derive_identity_field(email)
    _commitment, k0 = _poseidon2_commit_field(secret_field)
    return k0.to_bytes(32, "big")

def derive_account_from_email(email: str):
    return Account.from_key(derive_private_key_from_email(email))

def derive_address_from_email(email: str) -> str:
    return Web3.to_checksum_address(derive_account_from_email(email).address)

def get_cohort_hash(cohort_id: str) -> bytes:
    return Web3.keccak(text=cohort_id)
