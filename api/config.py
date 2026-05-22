import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

_CONTRACT_FIELDS = [
    ("duo_ontology_address", "duoOntology"),
    ("institution_registry_address", "institutionRegistry"),
    ("attestation_registry_address", "attestationRegistry"),
    ("duo_consent_vault_v2_address", "duoConsentVaultV2"),
    ("duo_consent_token_address", "duoConsentToken"),
    ("access_credential_nft_address", "accessCredentialNFT"),
    ("duo_attestation_resolver_address", "duoAttestationResolver"),
    ("zk_consent_verifier_address", "zkConsentVerifier"),
    ("nullifier_registry_address", "nullifierRegistry"),
    ("identity_registry_address", "identityRegistry"),
    ("ibis_verifier_address", "ibisVerifier"),
    ("user_identity_registry_address", "userIdentityRegistry"),
    ("role_group_registry_address", "roleGroupRegistry"),
    ("role_account_factory_address", "roleAccountFactory"),
    ("gas_sponsor_address", "gasSponsor"),
    ("commitment_tracker_address", "commitmentTracker"),
    ("trusted_forwarder_address", "trustedForwarder"),
]

def load_deployments() -> dict:
    paths = [
        Path("/app/deployments/deployments.json"),
        Path(__file__).parent.parent / "deployments" / "deployments.json",
        Path(__file__).parent.parent / "deployments.json",
        Path("deployments.json"),
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
                return data if "contracts" in data else {"contracts": data}
    return {"contracts": {}}

class Settings(BaseSettings):
    rpc_url: str = "http://127.0.0.1:8545"
    chain_id: int = 31337
    relayer_private_key: str = ""
    derivation_salt: str = ""

    duo_ontology_address: str = ""
    institution_registry_address: str = ""
    attestation_registry_address: str = ""
    duo_consent_vault_v2_address: str = ""
    duo_consent_token_address: str = ""
    access_credential_nft_address: str = ""
    duo_attestation_resolver_address: str = ""
    zk_consent_verifier_address: str = ""
    nullifier_registry_address: str = ""
    identity_registry_address: str = ""
    ibis_verifier_address: str = ""
    user_identity_registry_address: str = ""
    role_group_registry_address: str = ""
    role_account_factory_address: str = ""
    gas_sponsor_address: str = ""
    commitment_tracker_address: str = ""
    trusted_forwarder_address: str = ""

    ibis_paillier_bits: int = 1024
    ibis_proof_backend: str = "noir"
    ibis_noir_circuit_path: str = ""
    ibis_healing_interval: int = 10
    ibis_constant_time_ms: int = 10000
    zkp_production_mode: bool = True

    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_max_connections: int = 50
    cache_backend: str = "auto"
    cache_sync_interval_seconds: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        deployments = load_deployments()
        contracts = deployments.get("contracts", {})
        for attr, key in _CONTRACT_FIELDS:
            if not getattr(self, attr) and contracts.get(key):
                setattr(self, attr, contracts[key])
        if not self.derivation_salt and deployments.get("derivationSalt"):
            self.derivation_salt = deployments["derivationSalt"]

@lru_cache()
def get_settings() -> Settings:
    return Settings()
