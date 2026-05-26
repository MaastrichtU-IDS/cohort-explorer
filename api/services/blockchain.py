\
\
\
\
\
\
\

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_typed_data

from api.config import get_settings
from api.services.wallet import (
    derive_account_from_email,
    derive_address_from_email,
    derive_role_commitment,
    derive_user_commitment,
    get_cohort_hash,
)
from api.models.duo import (
    DUOPermission,
    PERMISSION_VALUES,
    PERMISSION_CODES,
    PERMISSION_LABELS,
    get_modifiers_bitmask,
    bitmask_to_modifiers,
    get_modifier_details,
)

logger = logging.getLogger(__name__)

class BlockchainService:

    def __init__(self):
        self.settings = get_settings()
        self.w3 = Web3(Web3.HTTPProvider(self.settings.rpc_url))

        if self.settings.relayer_private_key:
            self.relayer_account = Account.from_key(self.settings.relayer_private_key)
        else:
            self.relayer_account = None

        self._load_contracts()

    def _load_contracts(self):
        abi_path = Path(__file__).parent.parent / "contracts" / "abis"

        self.ontology = self._load_contract(
            abi_path / "DUOOntology.json",
            self.settings.duo_ontology_address
        )
        self.institution_registry = self._load_contract(
            abi_path / "InstitutionRegistry.json",
            self.settings.institution_registry_address
        )
        self.attestation_registry = self._load_contract(
            abi_path / "AttestationRegistry.json",
            self.settings.attestation_registry_address
        )

        self.consent_vault_v2 = self._load_contract(
            abi_path / "DUOConsentVaultV2.json",
            self.settings.duo_consent_vault_v2_address
        )

        self.consent_vault = self.consent_vault_v2

        self.consent_token = self._load_contract(
            abi_path / "DUOConsentToken.json",
            self.settings.duo_consent_token_address
        )
        self.access_credential_nft = self._load_contract(
            abi_path / "AccessCredentialNFT.json",
            self.settings.access_credential_nft_address
        )
        self.attestation_resolver = self._load_contract(
            abi_path / "DUOAttestationResolver.json",
            self.settings.duo_attestation_resolver_address
        )

        self.zk_verifier = self._load_contract(
            abi_path / "ZKConsentVerifier.json",
            self.settings.zk_consent_verifier_address
        )
        self.nullifier_registry = self._load_contract(
            abi_path / "NullifierRegistry.json",
            self.settings.nullifier_registry_address
        )
        self.identity_registry = self._load_contract(
            abi_path / "IdentityRegistry.json",
            self.settings.identity_registry_address
        )

        self.commitment_tracker = self._load_contract(
            abi_path / "CommitmentTracker.json",
            self.settings.commitment_tracker_address
        )

        self.ibis_verifier = self._load_contract(
            abi_path / "IBISVerifier.json",
            self.settings.ibis_verifier_address
        )

        self.gas_sponsor = self._load_contract(
            abi_path / "GasSponsor.json",
            self.settings.gas_sponsor_address
        )

        self.trusted_forwarder = self._load_contract(
            abi_path / "TrustedForwarder.json",
            self.settings.trusted_forwarder_address
        )

        self.role_group_registry = self._load_contract(
            abi_path / "RoleGroupRegistry.json",
            self.settings.role_group_registry_address
        )
        if self.role_group_registry:
            self.ROLE_PROVIDER = Web3.keccak(text="ROLE_PROVIDER")
            self.ROLE_REQUESTER = Web3.keccak(text="ROLE_REQUESTER")

        self.user_identity_registry = self._load_contract(
            abi_path / "UserIdentityRegistry.json",
            self.settings.user_identity_registry_address
        )

        self.role_account_factory = self._load_contract(
            abi_path / "RoleAccountFactory.json",
            self.settings.role_account_factory_address
        )

    def _load_contract(self, abi_path: Path, address: str):
        if not address or not abi_path.exists():
            return None
        try:
            with open(abi_path) as f:
                data = json.load(f)
            return self.w3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=data["abi"]
            )
        except Exception as e:
            logger.error(f"Failed to load contract {abi_path}: {e}")
            return None

    def is_connected(self) -> bool:
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def get_chain_id(self) -> int | None:
        try:
            return self.w3.eth.chain_id
        except Exception:
            return None

    def get_latest_block(self) -> int | None:
        try:
            return self.w3.eth.block_number
        except Exception:
            return None

    def get_eip712_domain(self) -> dict:
        return {
            "name": "DUOConsentVaultV2",
            "version": "1",
            "chainId": self.settings.chain_id,
            "verifyingContract": self.settings.duo_consent_vault_v2_address
        }

    def prepare_consent_typed_data(
        self,
        cohort_hash: bytes,
        permission: int,
        modifiers: int,
        disease_code: bytes,
        valid_until: int,
        nonce: int
    ) -> dict:
        domain = self.get_eip712_domain()

        types = {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "RecordConsent": [
                {"name": "cohortHash", "type": "bytes32"},
                {"name": "permission", "type": "bytes4"},
                {"name": "modifiers", "type": "uint16"},
                {"name": "diseaseCode", "type": "bytes32"},
                {"name": "validUntil", "type": "uint256"},
                {"name": "nonce", "type": "uint256"},
            ]
        }

        message = {
            "cohortHash": cohort_hash.hex() if isinstance(cohort_hash, bytes) else cohort_hash,
            "permission": f"0x{permission:08x}",
            "modifiers": modifiers,
            "diseaseCode": disease_code.hex() if isinstance(disease_code, bytes) else disease_code,
            "validUntil": valid_until,
            "nonce": nonce,
        }

        typed_data = {
            "types": types,
            "primaryType": "RecordConsent",
            "domain": domain,
            "message": message
        }

        message_hash = self._hash_typed_data(typed_data)

        return {
            "typed_data": typed_data,
            "message_hash": message_hash
        }

    def _hash_typed_data(self, typed_data: dict) -> str:
        try:
            signable = encode_typed_data(full_message=typed_data)
            return signable.body.hex()
        except Exception as e:
            logger.error(f"Failed to hash typed data: {e}")
            return ""

    async def get_nonce(self, owner_email: str) -> int:
        if not self.consent_vault:
            return 0

        try:
            owner_address = derive_address_from_email(owner_email)
            return self.consent_vault.functions.getNonce(owner_address).call()
        except Exception as e:
            logger.error(f"Failed to get nonce: {e}")
            return 0

    async def record_consent(
        self,
        owner_email: str,
        cohort_id: str,
        permission: str,
        modifiers: list[str],
        disease_code: str | None = None,
        allowed_countries: list[str] | None = None,
        allowed_institutions: list[str] | None = None,
        allowed_projects: list[str] | None = None,
        allowed_users: list[str] | None = None,
        moratorium_months: int = 0,
        publication_deadline_days: int = 0,
        expiration_days: int = 0,
        metadata: dict | None = None,
        signature: str | None = None,
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"success": False, "error": "Contract not initialized"}
        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        from api.services.iso3166 import country_bitset

        try:
            owner = derive_account_from_email(owner_email)
            owner_address = owner.address

            provider_info = await self.ensure_role_active(owner_email, "PROVIDER")
            if not provider_info.get("success"):
                return {"success": False, "error": f"PROVIDER activation failed: {provider_info.get('error')}"}
            principal_address = provider_info["account"]

            cohort_hash = get_cohort_hash(cohort_id)
            perm_bytes4 = PERMISSION_VALUES.get(permission.upper(), DUOPermission.GRU)
            mod_bitmask = int(get_modifiers_bitmask(modifiers))
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)

            metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")).encode()
            metadata_hash = self.w3.keccak(metadata_json) if metadata else bytes(32)

            bitset = country_bitset(allowed_countries) if allowed_countries else 0

            inst_ids = [self.w3.keccak(text=i) for i in (allowed_institutions or [])]
            proj_ids = [self.w3.keccak(text=p) for p in (allowed_projects or [])]
            user_addrs = [
                Web3.to_checksum_address(u) if u.startswith("0x") else derive_address_from_email(u)
                for u in (allowed_users or [])
            ]

            args = (
                cohort_hash,
                perm_bytes4.to_bytes(4, "big"),
                mod_bitmask,
                disease_bytes,
                metadata_hash,
                bitset,
                int(expiration_days),
                int(moratorium_months) & 0xFFFF,
                int(publication_deadline_days) & 0xFFFF,
                bytes(32),
            )

            nonce = self.consent_vault.functions.getNonce(owner_address).call()
            inst_hash = self.w3.keccak(b"".join(inst_ids)) if inst_ids else self.w3.keccak(b"")
            proj_hash = self.w3.keccak(b"".join(proj_ids)) if proj_ids else self.w3.keccak(b"")
            addr_hash = self.w3.keccak(
                b"".join(b"\x00" * 12 + bytes.fromhex(a[2:]) for a in user_addrs)
            ) if user_addrs else self.w3.keccak(b"")

            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    "RecordConsent": [
                        {"name": "cohortHash", "type": "bytes32"},
                        {"name": "permission", "type": "bytes4"},
                        {"name": "modifiers", "type": "uint256"},
                        {"name": "diseaseCode", "type": "bytes32"},
                        {"name": "metadataHash", "type": "bytes32"},
                        {"name": "countryBitset", "type": "uint256"},
                        {"name": "validDays", "type": "uint256"},
                        {"name": "moratoriumMonths", "type": "uint256"},
                        {"name": "publicationDeadlineDays", "type": "uint256"},
                        {"name": "institutionsRoot", "type": "bytes32"},
                        {"name": "institutionIdsHash", "type": "bytes32"},
                        {"name": "projectIdsHash", "type": "bytes32"},
                        {"name": "userAddressesHash", "type": "bytes32"},
                        {"name": "nonce", "type": "uint256"},
                    ],
                },
                "primaryType": "RecordConsent",
                "domain": {
                    "name": "DUOConsentVaultV2",
                    "version": "2",
                    "chainId": int(self.w3.eth.chain_id),
                    "verifyingContract": Web3.to_checksum_address(self.consent_vault.address),
                },
                "message": {
                    "cohortHash": cohort_hash,
                    "permission": perm_bytes4.to_bytes(4, "big"),
                    "modifiers": mod_bitmask,
                    "diseaseCode": disease_bytes,
                    "metadataHash": metadata_hash,
                    "countryBitset": bitset,
                    "validDays": int(expiration_days),
                    "moratoriumMonths": int(moratorium_months) & 0xFFFF,
                    "publicationDeadlineDays": int(publication_deadline_days) & 0xFFFF,
                    "institutionsRoot": bytes(32),
                    "institutionIdsHash": inst_hash,
                    "projectIdsHash": proj_hash,
                    "userAddressesHash": addr_hash,
                    "nonce": nonce,
                },
            }

            signable = encode_typed_data(full_message=typed_data)
            signed = owner.sign_message(signable)
            signature = signed.signature

            tx = self.consent_vault.functions.recordConsentWithSignature(
                args, nonce, inst_ids, proj_ids, user_addrs, signature
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 1_500_000,
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt["status"] != 1:
                return {"success": False, "error": "tx reverted"}

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "cohort_hash": cohort_hash.hex(),
                "owner_address": principal_address,
                "owner_eoa": owner_address,
                "permission": permission.upper(),
                "permission_label": PERMISSION_LABELS.get(permission.upper(), permission),
                "modifiers": modifiers,
                "modifier_details": get_modifier_details(modifiers),
                "metadata_hash": "0x" + metadata_hash.hex(),
                "country_bitset": hex(bitset),
                "via_signature": True,
                "nonce": nonce,
                "provider_commitment": provider_info.get("commitment"),
                "provider_account": provider_info.get("account"),
            }

        except Exception as e:
            logger.error(f"Failed to record consent: {e}")
            return {"success": False, "error": str(e)}

    async def consent_exists_on_chain(self, cohort_id: str) -> bool:
        if not self.consent_vault:
            return False
        try:
            cohort_hash = get_cohort_hash(cohort_id)
            result = self.consent_vault.functions.getConsent(cohort_hash).call()
            valid_from = result[4]
            return valid_from > 0
        except Exception as e:
            logger.error(f"consent_exists_on_chain failed: {e}")
            return False

    async def revoke_consent(
        self,
        owner_email: str,
        cohort_id: str
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"success": False, "error": "Contract not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)

            tx = self.consent_vault.functions.revokeConsent(
                cohort_hash
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 300000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            revoked_count = 0
            for log in receipt.get("logs", []):

                if len(log.get("topics", [])) >= 1:
                    revoked_count += 1

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "revoked_access_count": revoked_count
            }

        except Exception as e:
            logger.error(f"Failed to revoke consent: {e}")
            return {"success": False, "error": str(e)}

    async def update_consent(
        self,
        owner_email: str,
        cohort_id: str,
        permission: str,
        modifiers: list[str],
        disease_code: str | None = None
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"success": False, "error": "Contract not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)

            perm_bytes4 = PERMISSION_VALUES.get(permission.upper(), DUOPermission.GRU)

            mod_bitmask = int(get_modifiers_bitmask(modifiers))
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)

            tx = self.consent_vault.functions.updateConsent(
                cohort_hash,
                perm_bytes4.to_bytes(4, 'big'),
                mod_bitmask,
                disease_bytes
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 400000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            revoked = []
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 3:

                    revoked.append(log["topics"][2].hex())

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "revalidation": {
                    "revoked": len(revoked),
                    "revoked_requesters": revoked
                }
            }

        except Exception as e:
            logger.error(f"Failed to update consent: {e}")
            return {"success": False, "error": str(e)}

    async def add_owner(
        self,
        owner_email: str,
        cohort_id: str,
        new_owner_email: str
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"success": False, "error": "Contract not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            new_owner_address = derive_address_from_email(new_owner_email)

            tx = self.consent_vault.functions.addOwner(
                cohort_hash,
                new_owner_address
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "new_owner_address": new_owner_address
            }

        except Exception as e:
            logger.error(f"Failed to add owner: {e}")
            return {"success": False, "error": str(e)}

    _REQ_TYPES = {"ACADEMIC": 1, "NONPROFIT": 2, "PROFIT": 3, "COMMERCIAL": 3, "INDIVIDUAL": 3, "CLINICAL": 4, "GOVERNMENT": 5}

    async def set_requester_type(
        self,
        user_email: str,
        requester_type: str,
        country_code: str | None = None,
    ) -> dict[str, Any]:
        if not self.institution_registry or not self.relayer_account:
            return {"success": False, "error": "registry/relayer not initialized"}
        try:
            t = self._REQ_TYPES.get(requester_type.upper(), 0)
            if t == 0:
                return {"success": False, "error": f"unknown requesterType {requester_type}"}
            user_addr = Web3.to_checksum_address(derive_address_from_email(user_email))
            cc = (country_code or "").upper().encode()[:2].ljust(2, b"\x00")
            tx = self.institution_registry.functions.setRequesterType(user_addr, t, cc).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return {"success": receipt["status"] == 1, "tx_hash": receipt["transactionHash"].hex()}
        except Exception as e:
            logger.error(f"Failed to set requester type: {e}")
            return {"success": False, "error": str(e)}

    async def request_access(
        self,
        requester_email: str,
        cohort_id: str,
        intended_use: str,
        purpose: int = 0,
        disease_code: str | None = None,
        project_id: str | None = None,
        country_code: str | None = None,
        institution_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"success": False, "error": "Contract not initialized"}
        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        from api.services.iso3166 import country_index

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            requester = derive_account_from_email(requester_email)
            requester_eoa = requester.address

            requester_info = await self.ensure_role_active(requester_email, "REQUESTER")
            if not requester_info.get("success"):
                return {"success": False, "error": f"REQUESTER activation failed: {requester_info.get('error')}"}
            requester_address = requester_info["account"]

            use_bytes4 = PERMISSION_VALUES.get(intended_use.upper(), DUOPermission.HMB)
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)
            project_bytes = self.w3.keccak(text=project_id) if project_id else bytes(32)
            inst_bytes = self.w3.keccak(text=institution_id) if institution_id else bytes(32)
            c_idx = country_index(country_code) if country_code else 0

            args = (
                cohort_hash,
                requester_address,
                use_bytes4.to_bytes(4, "big"),
                int(purpose) & 0xFF,
                disease_bytes,
                project_bytes,
                c_idx & 0xFF,
                inst_bytes,
            )

            nonce = self.consent_vault.functions.getNonce(requester_eoa).call()
            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    "RequestAccess": [
                        {"name": "cohortHash", "type": "bytes32"},
                        {"name": "requester", "type": "address"},
                        {"name": "intendedUse", "type": "bytes4"},
                        {"name": "purpose", "type": "uint8"},
                        {"name": "diseaseCode", "type": "bytes32"},
                        {"name": "projectId", "type": "bytes32"},
                        {"name": "countryIndex", "type": "uint8"},
                        {"name": "institutionId", "type": "bytes32"},
                        {"name": "nonce", "type": "uint256"},
                    ],
                },
                "primaryType": "RequestAccess",
                "domain": {
                    "name": "DUOConsentVaultV2",
                    "version": "2",
                    "chainId": int(self.w3.eth.chain_id),
                    "verifyingContract": Web3.to_checksum_address(self.consent_vault.address),
                },
                "message": {
                    "cohortHash": cohort_hash,
                    "requester": requester_address,
                    "intendedUse": use_bytes4.to_bytes(4, "big"),
                    "purpose": int(purpose) & 0xFF,
                    "diseaseCode": disease_bytes,
                    "projectId": project_bytes,
                    "countryIndex": c_idx & 0xFF,
                    "institutionId": inst_bytes,
                    "nonce": nonce,
                },
            }

            signable = encode_typed_data(full_message=typed_data)
            signed = requester.sign_message(signable)
            signature = signed.signature

            tx = self.consent_vault.functions.requestAccessWithSignature(
                args, nonce, signature
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 1_500_000,
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            request_id = None
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 3:
                    request_id = log["topics"][1].hex()

            return {
                "success": receipt["status"] == 1,
                "tx_hash": receipt["transactionHash"].hex(),
                "request_id": request_id,
                "auto_approved": receipt["status"] == 1,
                "requester_address": requester_address,
                "requester_eoa": requester_eoa,
                "via_signature": True,
                "nonce": nonce,
                "requester_commitment": requester_info.get("commitment"),
            }

        except Exception as e:
            logger.error(f"Failed to request access: {e}")
            return {"success": False, "error": str(e)}

    async def check_access(
        self,
        cohort_id: str,
        requester_email: str
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"has_access": False, "error": "Contract not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            requester_address = derive_address_from_email(requester_email)

            has_access = self.consent_vault.functions.hasAccess(
                cohort_hash,
                requester_address
            ).call()

            return {
                "has_access": has_access,
                "requester_address": requester_address
            }

        except Exception as e:
            logger.error(f"Failed to check access: {e}")
            return {"has_access": False, "error": str(e)}

    def is_user_registered_onchain(self, email: str) -> bool:
        if not self.user_identity_registry:
            return False
        try:
            eoa = derive_address_from_email(email)
            return self.user_identity_registry.functions.isRegistered(eoa).call()
        except Exception as e:
            logger.warning(f"isRegistered failed: {e}")
            return False

    async def register_user_identity(self, email: str) -> dict[str, Any]:
        if not (self.user_identity_registry and self.ibis_verifier and self.relayer_account):
            return {"success": False, "error": "identity infra not initialized"}
        try:
            eoa = derive_address_from_email(email)
            commitment = derive_user_commitment(email)

            existing = self.user_identity_registry.functions.identityOf(eoa).call()
            if int.from_bytes(existing, "big") != 0:
                if existing != commitment:
                    return {"success": False, "error": "EOA bound to a different commitment"}
                return {"success": True, "eoa": eoa, "commitment": "0x" + commitment.hex(), "already": True}

            slot_ok = False
            try:
                tx = self.ibis_verifier.functions.initializeSlot(commitment).build_transaction({
                    "from": self.relayer_account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                    "gas": 800_000,
                    "gasPrice": self.w3.eth.gas_price,
                })
                signed = self.relayer_account.sign_transaction(tx)
                rh = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                rec = self.w3.eth.wait_for_transaction_receipt(rh)
                slot_ok = rec["status"] == 1
            except Exception as e:
                logger.warning(f"initializeSlot failed: {e}")
            if not slot_ok:
                return {"success": False, "error": "IBIS slot init failed"}

            tx = self.user_identity_registry.functions.register(eoa, commitment).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 200_000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.relayer_account.sign_transaction(tx)
            rh = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            rec = self.w3.eth.wait_for_transaction_receipt(rh)
            if rec["status"] != 1:
                return {"success": False, "error": "register reverted"}
            return {
                "success": True,
                "eoa": eoa,
                "commitment": "0x" + commitment.hex(),
                "tx_hash": rec["transactionHash"].hex(),
                "already": False,
            }
        except Exception as e:
            logger.error(f"register_user_identity failed: {e}")
            return {"success": False, "error": str(e)}

    def _role_id(self, role: str) -> bytes:
        if role == "PROVIDER":
            return self.ROLE_PROVIDER
        if role == "REQUESTER":
            return self.ROLE_REQUESTER
        raise ValueError(f"unknown role {role}")

    def get_role_account_if_attached(self, eoa: str, role: str) -> dict | None:
        if not self.role_group_registry or not self.role_account_factory:
            return None
        try:
            role_id = self._role_id(role)
            eoa_cs = Web3.to_checksum_address(eoa)
            commitment = self.role_group_registry.functions.commitmentOf(role_id, eoa_cs).call()
            if int.from_bytes(commitment, "big") == 0:
                return None
            account = self.role_account_factory.functions.computeAddress(eoa_cs, commitment).call()
            return {"commitment": "0x" + commitment.hex(), "account": account}
        except Exception as e:
            logger.warning(f"get_role_account_if_attached failed: {e}")
            return None

    async def activate_role(self, email: str, role: str) -> dict[str, Any]:
        if not (self.role_group_registry and self.role_account_factory and self.ibis_verifier and self.relayer_account):
            return {"success": False, "error": "role infra not initialized"}
        try:
            role_id = self._role_id(role)
            eoa = derive_account_from_email(email).address
            commitment = derive_role_commitment(email, role)

            existing = self.role_group_registry.functions.commitmentOf(role_id, eoa).call()
            if int.from_bytes(existing, "big") != 0:
                if existing != commitment:
                    return {"success": False, "error": "EOA bound to a different commitment (refuse to overwrite)"}
            else:
                tx = self.role_group_registry.functions.attachRole(
                    role_id, eoa, commitment
                ).build_transaction({
                    "from": self.relayer_account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                    "gas": 250_000,
                    "gasPrice": self.w3.eth.gas_price,
                })
                signed = self.relayer_account.sign_transaction(tx)
                rh = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                rec = self.w3.eth.wait_for_transaction_receipt(rh)
                if rec["status"] != 1:
                    return {"success": False, "error": "attachRole reverted"}

            account = self.role_account_factory.functions.computeAddress(eoa, commitment).call()
            code = self.w3.eth.get_code(account)
            if len(code) <= 2:
                tx = self.role_account_factory.functions.deploy(eoa, commitment).build_transaction({
                    "from": self.relayer_account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                    "gas": 600_000,
                    "gasPrice": self.w3.eth.gas_price,
                })
                signed = self.relayer_account.sign_transaction(tx)
                rh = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                rec = self.w3.eth.wait_for_transaction_receipt(rh)
                if rec["status"] != 1:
                    return {"success": False, "error": "RoleAccount deploy reverted"}

            try:
                tx = self.ibis_verifier.functions.initializeSlot(commitment).build_transaction({
                    "from": self.relayer_account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                    "gas": 800_000,
                    "gasPrice": self.w3.eth.gas_price,
                })
                signed = self.relayer_account.sign_transaction(tx)
                rh = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                self.w3.eth.wait_for_transaction_receipt(rh)
            except Exception as e:
                logger.warning(f"ibis initializeSlot failed (non-fatal): {e}")

            return {"success": True, "commitment": "0x" + commitment.hex(), "account": account}
        except Exception as e:
            logger.error(f"activate_role failed: {e}")
            return {"success": False, "error": str(e)}

    async def ensure_role_active(self, email: str, role: str) -> dict[str, Any]:
        eoa = derive_account_from_email(email).address
        existing = self.get_role_account_if_attached(eoa, role)
        if existing:
            return {"success": True, **existing, "deployed": False}
        result = await self.activate_role(email, role)
        return result

    def compute_role_account(self, email: str, role: str) -> str | None:
        if not self.role_account_factory:
            return None
        try:
            owner = derive_account_from_email(email).address
            commitment = derive_role_commitment(email, role)
            return self.role_account_factory.functions.computeAddress(owner, commitment).call()
        except Exception as e:
            logger.warning(f"computeRoleAccount failed: {e}")
            return None

    async def deploy_role_account(self, email: str, role: str) -> dict[str, Any]:
        if not self.role_account_factory or not self.relayer_account:
            return {"success": False, "error": "factory/relayer not initialized"}
        try:
            owner = derive_account_from_email(email).address
            commitment = derive_role_commitment(email, role)
            expected = self.role_account_factory.functions.computeAddress(owner, commitment).call()
            code = self.w3.eth.get_code(expected)
            if len(code) > 2:
                return {"success": True, "account": expected, "deployed": False}
            tx = self.role_account_factory.functions.deploy(owner, commitment).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500_000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return {
                "success": receipt["status"] == 1,
                "account": expected,
                "deployed": True,
                "tx_hash": receipt["transactionHash"].hex(),
            }
        except Exception as e:
            logger.error(f"deployRoleAccount failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_consent_status(self, cohort_id: str) -> dict[str, Any]:
        if not self.consent_vault:
            return {"active": False, "error": "Contract not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)

            result = self.consent_vault.functions.getConsent(cohort_hash).call()
            (
                primary_owner,
                permission,
                modifiers_bitmask,
                disease_code,
                valid_from,
                valid_until,
                active,
                metadata_uri
            ) = result

            owners = self.consent_vault.functions.getOwners(cohort_hash).call()

            perm_int = int.from_bytes(permission, 'big') if isinstance(permission, bytes) else permission
            perm_code = PERMISSION_CODES.get(perm_int, "UNKNOWN")

            mod_list = bitmask_to_modifiers(modifiers_bitmask)

            return {
                "active": active,
                "owners": owners,
                "duo_permission": perm_code,
                "duo_permission_label": PERMISSION_LABELS.get(perm_code, perm_code),
                "duo_modifiers": mod_list,
                "modifier_details": get_modifier_details(mod_list),
                "disease_code": disease_code.hex() if disease_code != bytes(32) else None,
                "valid_from": datetime.fromtimestamp(valid_from) if valid_from > 0 else None,
                "valid_until": datetime.fromtimestamp(valid_until) if valid_until > 0 else None,
                "metadata_uri": metadata_uri,
                "cohort_hash": cohort_hash.hex()
            }

        except Exception as e:
            logger.error(f"Failed to get consent status: {e}")
            return {"active": False, "error": str(e)}

    async def get_cohort_requests(self, cohort_id: str) -> dict[str, Any]:
        if not self.consent_vault:
            return {"requests": [], "error": "Contract not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            count = self.consent_vault.functions.getCohortRequestCount(cohort_hash).call()

            return {
                "cohort_id": cohort_id,
                "request_count": count
            }

        except Exception as e:
            logger.error(f"Failed to get cohort requests: {e}")
            return {"requests": [], "error": str(e)}

    async def revoke_access(
        self,
        owner_email: str,
        cohort_id: str,
        requester_email: str
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"success": False, "error": "Contract not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            requester_address = derive_address_from_email(requester_email)

            tx = self.consent_vault.functions.revokeAccess(
                cohort_hash,
                requester_address
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex()
            }

        except Exception as e:
            logger.error(f"Failed to revoke access: {e}")
            return {"success": False, "error": str(e)}

    async def submit_attestation(
        self,
        subject_email: str,
        attestation_type: str,
        scope: str,
        data_hash: str | None = None,
        valid_days: int = 365,
        metadata: dict | None = None
    ) -> dict[str, Any]:
        if not self.attestation_registry:
            return {"success": False, "error": "AttestationRegistry not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:

            subject_address = derive_address_from_email(subject_email)
            try:
                requester_info = self.get_role_account_if_attached(subject_address, "REQUESTER")
                if requester_info and requester_info.get("account"):
                    subject_address = requester_info["account"]
            except Exception:
                pass

            type_hash = self.w3.keccak(text=attestation_type)
            scope_bytes = bytes.fromhex(scope) if scope.startswith("0x") else bytes.fromhex(scope)
            data_bytes = self.w3.keccak(text=data_hash) if data_hash else bytes(32)

            tx = self.attestation_registry.functions.createAttestation(
                type_hash,
                subject_address,
                scope_bytes,
                data_bytes,
                valid_days,
                json.dumps(metadata) if metadata else ""
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 600_000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            attestation_id = None
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 2:
                    attestation_id = log["topics"][1].hex()
                    break

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "attestation_id": attestation_id
            }

        except Exception as e:
            logger.error(f"Failed to submit attestation: {e}")
            return {"success": False, "error": str(e)}

    async def revoke_attestation(
        self,
        subject_email: str,
        attestation_type: str,
        scope: str
    ) -> dict[str, Any]:
        if not self.attestation_registry:
            return {"success": False, "error": "AttestationRegistry not initialized"}
        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}
        try:
            subject_address = Web3.to_checksum_address(derive_address_from_email(subject_email))
            type_hash = self.w3.keccak(text=attestation_type)
            scope_bytes = bytes.fromhex(scope[2:]) if scope.startswith("0x") else bytes.fromhex(scope)

            valid, attestation_id = self.attestation_registry.functions.hasValidAttestation(
                subject_address, type_hash, scope_bytes,
            ).call()
            if not valid or attestation_id == b"\x00" * 32:
                return {"success": False, "error": "No active attestation found to revoke"}

            tx = self.attestation_registry.functions.revokeAttestation(
                attestation_id
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return {"success": receipt["status"] == 1, "tx_hash": receipt["transactionHash"].hex()}
        except Exception as e:
            logger.error(f"Failed to revoke attestation: {e}")
            return {"success": False, "error": str(e)}

    async def establish_collaboration(
        self,
        owner_email: str,
        cohort_id: str,
        requester_email: str,
        terms_hash: str | None = None,
        valid_days: int = 365
    ) -> dict[str, Any]:
        if not self.attestation_registry:
            return {"success": False, "error": "AttestationRegistry not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            requester_address = derive_address_from_email(requester_email)
            cohort_hash = get_cohort_hash(cohort_id)

            project_id = bytes(32)
            terms_uri = terms_hash or ""

            tx = self.attestation_registry.functions.establishCollaboration(
                cohort_hash,
                requester_address,
                project_id,
                valid_days,
                terms_uri
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            collaboration_id = None
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 2:
                    collaboration_id = log["topics"][1].hex()
                    break

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "collaboration_id": collaboration_id
            }

        except Exception as e:
            logger.error(f"Failed to establish collaboration: {e}")
            return {"success": False, "error": str(e)}

    async def revoke_collaboration(
        self,
        owner_email: str,
        cohort_id: str,
        requester_email: str
    ) -> dict[str, Any]:
        if not self.attestation_registry:
            return {"success": False, "error": "AttestationRegistry not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            requester_address = derive_address_from_email(requester_email)
            cohort_hash = get_cohort_hash(cohort_id)
            type_hash = self.w3.keccak(text="COLLABORATION")

            agreement_id = self.w3.keccak(cohort_hash + requester_address.encode())

            tx = self.attestation_registry.functions.revokeCollaboration(
                agreement_id
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex()
            }

        except Exception as e:
            logger.error(f"Failed to revoke collaboration: {e}")
            return {"success": False, "error": str(e)}

    async def record_consent_v2(
        self,
        owner_email: str,
        cohort_id: str,
        permission: str,
        modifiers: list[str],
        disease_code: str | None = None,
        countries_merkle_root: str | None = None,
        institutions_merkle_root: str | None = None,
        expiration_days: int = 0,
        metadata_uri: str = "",
        signature: str | None = None
    ) -> dict[str, Any]:
        if not self.consent_vault_v2:
            return {"success": False, "error": "ConsentVaultV2 not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            owner_address = derive_address_from_email(owner_email)
            cohort_hash = get_cohort_hash(cohort_id)

            perm_bytes4 = PERMISSION_VALUES.get(permission.upper(), DUOPermission.GRU)
            mod_bitmask = int(get_modifiers_bitmask(modifiers))
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)

            countries_root = bytes.fromhex(countries_merkle_root.replace("0x", "")) if countries_merkle_root else bytes(32)
            institutions_root = bytes.fromhex(institutions_merkle_root.replace("0x", "")) if institutions_merkle_root else bytes(32)

            if signature:

                tx = self.consent_vault_v2.functions.recordConsentWithSignature(
                    cohort_hash,
                    perm_bytes4.to_bytes(4, 'big'),
                    mod_bitmask,
                    disease_bytes,
                    countries_root,
                    institutions_root,
                    expiration_days,
                    metadata_uri,
                    bytes.fromhex(signature.replace("0x", ""))
                ).build_transaction({
                    "from": self.relayer_account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                    "gas": 500000,
                    "gasPrice": self.w3.eth.gas_price
                })
                via_sig = True
            else:

                tx = self.consent_vault_v2.functions.recordConsent(
                    cohort_hash,
                    perm_bytes4.to_bytes(4, 'big'),
                    mod_bitmask,
                    disease_bytes,
                    countries_root,
                    institutions_root,
                    expiration_days,
                    metadata_uri
                ).build_transaction({
                    "from": self.relayer_account.address,
                    "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                    "gas": 500000,
                    "gasPrice": self.w3.eth.gas_price
                })
                via_sig = False

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "cohort_hash": cohort_hash.hex(),
                "owner_address": owner_address,
                "permission": permission.upper(),
                "permission_label": PERMISSION_LABELS.get(permission.upper(), permission),
                "modifiers": modifiers,
                "modifier_details": get_modifier_details(modifiers),
                "countries_merkle_root": countries_merkle_root,
                "institutions_merkle_root": institutions_merkle_root,
                "contract_version": "v2",
                "via_signature": via_sig
            }

        except Exception as e:
            logger.error(f"Failed to record consent V2: {e}")
            return {"success": False, "error": str(e)}

    async def update_merkle_roots(
        self,
        owner_email: str,
        cohort_id: str,
        countries_merkle_root: str | None = None,
        institutions_merkle_root: str | None = None
    ) -> dict[str, Any]:
        if not self.consent_vault_v2:
            return {"success": False, "error": "ConsentVaultV2 not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            countries_root = bytes.fromhex(countries_merkle_root.replace("0x", "")) if countries_merkle_root else bytes(32)
            institutions_root = bytes.fromhex(institutions_merkle_root.replace("0x", "")) if institutions_merkle_root else bytes(32)

            tx = self.consent_vault_v2.functions.updateMerkleRoots(
                cohort_hash,
                countries_root,
                institutions_root
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "countries_merkle_root": countries_merkle_root,
                "institutions_merkle_root": institutions_merkle_root
            }

        except Exception as e:
            logger.error(f"Failed to update Merkle roots: {e}")
            return {"success": False, "error": str(e)}

    async def request_access_with_proofs(
        self,
        requester_email: str,
        cohort_id: str,
        intended_use: str,
        purpose: int = 0,
        disease_code: str | None = None,
        country_proof: list[str] | None = None,
        country_leaf_hash: str | None = None,
        institution_proof: list[str] | None = None,
        institution_leaf_hash: str | None = None
    ) -> dict[str, Any]:
        if not self.consent_vault_v2:
            return {"success": False, "error": "ConsentVaultV2 not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            requester_address = derive_address_from_email(requester_email)

            use_bytes4 = PERMISSION_VALUES.get(intended_use.upper(), DUOPermission.HMB)
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)

            country_proof_bytes = [bytes.fromhex(p.replace("0x", "")) for p in (country_proof or [])]
            country_leaf_bytes = bytes.fromhex(country_leaf_hash.replace("0x", "")) if country_leaf_hash else bytes(32)
            institution_proof_bytes = [bytes.fromhex(p.replace("0x", "")) for p in (institution_proof or [])]
            institution_leaf_bytes = bytes.fromhex(institution_leaf_hash.replace("0x", "")) if institution_leaf_hash else bytes(32)

            tx = self.consent_vault_v2.functions.requestAccessWithProofs(
                cohort_hash,
                use_bytes4.to_bytes(4, 'big'),
                purpose,
                disease_bytes,
                country_proof_bytes,
                country_leaf_bytes,
                institution_proof_bytes,
                institution_leaf_bytes
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            approved = False
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 2:
                    approved = True
                    break

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "requester_address": requester_address,
                "approved": approved,
                "contract_version": "v2"
            }

        except Exception as e:
            logger.error(f"Failed to request access with proofs: {e}")
            return {"success": False, "error": str(e)}

    async def get_consent_status_v2(self, cohort_id: str) -> dict[str, Any]:
        if not self.consent_vault_v2:
            return {"active": False, "error": "ConsentVaultV2 not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)

            result = self.consent_vault_v2.functions.getConsent(cohort_hash).call()
            (
                primary_owner,
                permission,
                modifiers_bitmask,
                disease_code,
                valid_from,
                valid_until,
                active,
                countries_root,
                institutions_root
            ) = result

            owners = self.consent_vault_v2.functions.getOwners(cohort_hash).call()

            perm_int = int.from_bytes(permission, 'big') if isinstance(permission, bytes) else permission
            perm_code = PERMISSION_CODES.get(perm_int, "UNKNOWN")

            mod_list = bitmask_to_modifiers(modifiers_bitmask)

            return {
                "active": active,
                "owners": owners,
                "duo_permission": perm_code,
                "duo_permission_label": PERMISSION_LABELS.get(perm_code, perm_code),
                "duo_modifiers": mod_list,
                "modifier_details": get_modifier_details(mod_list),
                "disease_code": disease_code.hex() if disease_code != bytes(32) else None,
                "valid_from": valid_from if valid_from > 0 else None,
                "valid_until": valid_until if valid_until > 0 else None,
                "countries_merkle_root": countries_root.hex() if countries_root != bytes(32) else None,
                "institutions_merkle_root": institutions_root.hex() if institutions_root != bytes(32) else None,
                "cohort_hash": cohort_hash.hex(),
                "contract_version": "v2"
            }

        except Exception as e:
            logger.error(f"Failed to get consent status V2: {e}")
            return {"active": False, "error": str(e)}

    async def verify_country_on_chain(
        self,
        cohort_id: str,
        country_code: str,
        proof: list[str]
    ) -> dict[str, Any]:
        if not self.consent_vault_v2:
            return {"verified": False, "error": "ConsentVaultV2 not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            country_bytes = country_code.encode('ascii')[:2]
            proof_bytes = [bytes.fromhex(p.replace("0x", "")) for p in proof]

            verified = self.consent_vault_v2.functions.verifyCountry(
                cohort_hash,
                country_bytes,
                proof_bytes
            ).call()

            return {
                "verified": verified,
                "country_code": country_code,
                "cohort_hash": cohort_hash.hex()
            }

        except Exception as e:
            logger.error(f"Failed to verify country: {e}")
            return {"verified": False, "error": str(e)}

    async def verify_institution_on_chain(
        self,
        cohort_id: str,
        institution_id: str,
        proof: list[str]
    ) -> dict[str, Any]:
        if not self.consent_vault_v2:
            return {"verified": False, "error": "ConsentVaultV2 not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            institution_bytes = self.w3.keccak(text=institution_id)
            proof_bytes = [bytes.fromhex(p.replace("0x", "")) for p in proof]

            verified = self.consent_vault_v2.functions.verifyInstitution(
                cohort_hash,
                institution_bytes,
                proof_bytes
            ).call()

            return {
                "verified": verified,
                "institution_id": institution_id,
                "cohort_hash": cohort_hash.hex()
            }

        except Exception as e:
            logger.error(f"Failed to verify institution: {e}")
            return {"verified": False, "error": str(e)}

    async def create_commitment(
        self,
        researcher_email: str,
        cohort_id: str,
        commitment_type: str,
        deadline_days: int = 0,
        description: str = ""
    ) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"success": False, "error": "CommitmentTracker not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            researcher_address = derive_address_from_email(researcher_email)
            cohort_hash = get_cohort_hash(cohort_id)

            type_hash = self.w3.keccak(text=commitment_type.upper())

            tx = self.commitment_tracker.functions.createCommitment(
                cohort_hash,
                researcher_address,
                type_hash,
                deadline_days,
                description
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            commitment_id = None
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 2:
                    commitment_id = log["topics"][1].hex()
                    break

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "commitment_id": commitment_id,
                "researcher_address": researcher_address,
                "commitment_type": commitment_type
            }

        except Exception as e:
            logger.error(f"Failed to create commitment: {e}")
            return {"success": False, "error": str(e)}

    async def fulfill_commitment(
        self,
        commitment_id: str,
        evidence_hash: str,
        evidence_uri: str = ""
    ) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"success": False, "error": "CommitmentTracker not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            commitment_bytes = bytes.fromhex(commitment_id.replace("0x", ""))
            evidence_bytes = self.w3.keccak(text=evidence_hash) if not evidence_hash.startswith("0x") else bytes.fromhex(evidence_hash.replace("0x", ""))

            tx = self.commitment_tracker.functions.fulfillCommitment(
                commitment_bytes,
                evidence_bytes,
                evidence_uri
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "commitment_id": commitment_id
            }

        except Exception as e:
            logger.error(f"Failed to fulfill commitment: {e}")
            return {"success": False, "error": str(e)}

    async def get_commitment(self, commitment_id: str) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"error": "CommitmentTracker not initialized"}

        try:
            commitment_bytes = bytes.fromhex(commitment_id.replace("0x", ""))
            c = self.commitment_tracker.functions.getCommitment(commitment_bytes).call()

            status_map = {0: "ACTIVE", 1: "FULFILLED", 2: "EXPIRED", 3: "CANCELLED"}

            return {
                "commitment_id": c[0].hex(),
                "cohort_hash": c[1].hex(),
                "researcher": c[2],
                "commitment_type": c[3].hex(),
                "created_at": c[4],
                "deadline": c[5],
                "status": status_map.get(c[6], "UNKNOWN"),
                "evidence_hash": c[7].hex() if c[7] != bytes(32) else None,
                "evidence_uri": c[8],
                "fulfilled_at": c[9] if c[9] > 0 else None,
                "description": c[10]
            }

        except Exception as e:
            logger.error(f"Failed to get commitment: {e}")
            return {"error": str(e)}

    async def get_researcher_commitments(self, researcher_email: str) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"error": "CommitmentTracker not initialized", "commitments": []}

        try:
            researcher_address = derive_address_from_email(researcher_email)
            commitment_ids = self.commitment_tracker.functions.getResearcherCommitments(researcher_address).call()

            commitments = []
            for cid in commitment_ids:
                c = self.commitment_tracker.functions.getCommitment(cid).call()
                status_map = {0: "ACTIVE", 1: "FULFILLED", 2: "EXPIRED", 3: "CANCELLED"}
                commitments.append({
                    "commitment_id": c[0].hex(),
                    "cohort_hash": c[1].hex(),
                    "commitment_type": c[3].hex(),
                    "deadline": c[5],
                    "status": status_map.get(c[6], "UNKNOWN"),
                    "description": c[10]
                })

            return {
                "researcher_address": researcher_address,
                "commitments": commitments,
                "total": len(commitments)
            }

        except Exception as e:
            logger.error(f"Failed to get researcher commitments: {e}")
            return {"error": str(e), "commitments": []}

    async def record_commitment_promise(
        self,
        requester_email: str,
        cohort_id: str,
        att_type: str,
    ) -> dict[str, Any]:
        if not self.attestation_registry:
            return {"success": False, "error": "AttestationRegistry not initialized"}
        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}
        try:

            role_info = await self.ensure_role_active(requester_email, "REQUESTER")
            if not role_info.get("success"):
                return {"success": False, "error": f"REQUESTER activation failed: {role_info.get('error')}"}

            requester = derive_account_from_email(requester_email)
            cohort_hash = get_cohort_hash(cohort_id)

            type_const = att_type.upper()
            if type_const == "PUBLICATION_PROMISE":
                type_hash = self.attestation_registry.functions.ATT_PUB().call()
            elif type_const == "RETURN_DATA_PROMISE":
                type_hash = self.attestation_registry.functions.ATT_RTN().call()
            else:
                return {"success": False, "error": f"Unsupported promise type: {att_type}"}

            details = json.dumps({"source_modifier": "PUB" if type_const == "PUBLICATION_PROMISE" else "RTN", "auto": True})
            nonce = self.attestation_registry.functions.commitmentNonces(requester.address).call()

            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"},
                    ],
                    "RecordCommitment": [
                        {"name": "commitmentType", "type": "bytes32"},
                        {"name": "signerEOA", "type": "address"},
                        {"name": "cohortHash", "type": "bytes32"},
                        {"name": "details", "type": "string"},
                        {"name": "nonce", "type": "uint256"},
                    ],
                },
                "primaryType": "RecordCommitment",
                "domain": {
                    "name": "AttestationRegistry",
                    "version": "1",
                    "chainId": int(self.w3.eth.chain_id),
                    "verifyingContract": Web3.to_checksum_address(self.attestation_registry.address),
                },
                "message": {
                    "commitmentType": type_hash,
                    "signerEOA": Web3.to_checksum_address(requester.address),
                    "cohortHash": cohort_hash,
                    "details": details,
                    "nonce": nonce,
                },
            }

            signable = encode_typed_data(full_message=typed_data)
            signed = requester.sign_message(signable)

            tx = self.attestation_registry.functions.recordCommitmentWithSignature(
                type_hash,
                Web3.to_checksum_address(requester.address),
                cohort_hash,
                details,
                nonce,
                signed.signature,
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500_000,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt["status"] != 1:
                return {"success": False, "error": "tx reverted", "tx_hash": receipt["transactionHash"].hex()}

            attestation_id = None
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 2:
                    attestation_id = log["topics"][1].hex()
                    break

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "attestation_id": attestation_id,
                "principal": role_info.get("account"),
                "nonce": nonce,
            }
        except Exception as e:
            logger.error(f"record_commitment_promise failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_researcher_commitments_by_address(self, researcher_address: str) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"error": "CommitmentTracker not initialized", "commitments": []}
        try:
            checksum = self.w3.to_checksum_address(researcher_address)
            commitment_ids = self.commitment_tracker.functions.getResearcherCommitments(checksum).call()
            status_map = {0: "ACTIVE", 1: "FULFILLED", 2: "EXPIRED", 3: "CANCELLED"}
            commitments = []
            for cid in commitment_ids:
                c = self.commitment_tracker.functions.getCommitment(cid).call()
                commitments.append({
                    "commitment_id": c[0].hex(),
                    "cohort_hash": c[1].hex(),
                    "commitment_type": c[3].hex(),
                    "deadline": c[5],
                    "status": status_map.get(c[6], "UNKNOWN"),
                    "description": c[10],
                })
            return {"researcher_address": checksum, "commitments": commitments, "total": len(commitments)}
        except Exception as e:
            logger.error(f"Failed to get researcher commitments by address: {e}")
            return {"error": str(e), "commitments": []}

    async def get_pending_commitments(self, researcher_email: str) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"error": "CommitmentTracker not initialized", "commitments": []}

        try:
            researcher_address = derive_address_from_email(researcher_email)
            pending_ids = self.commitment_tracker.functions.getPendingCommitments(researcher_address).call()

            commitments = []
            for cid in pending_ids:
                c = self.commitment_tracker.functions.getCommitment(cid).call()
                days_remaining = self.commitment_tracker.functions.getDaysRemaining(cid).call()
                commitments.append({
                    "commitment_id": c[0].hex(),
                    "cohort_hash": c[1].hex(),
                    "commitment_type": c[3].hex(),
                    "deadline": c[5],
                    "days_remaining": days_remaining,
                    "description": c[10]
                })

            return {
                "researcher_address": researcher_address,
                "pending_commitments": commitments,
                "total_pending": len(commitments)
            }

        except Exception as e:
            logger.error(f"Failed to get pending commitments: {e}")
            return {"error": str(e), "commitments": []}

    async def get_commitment_summary(self, researcher_email: str) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"error": "CommitmentTracker not initialized"}

        try:
            researcher_address = derive_address_from_email(researcher_email)
            summary = self.commitment_tracker.functions.getResearcherSummary(researcher_address).call()

            return {
                "researcher_address": researcher_address,
                "total": summary[0],
                "active": summary[1],
                "fulfilled": summary[2],
                "expired": summary[3],
                "cancelled": summary[4],
                "fulfillment_rate": (summary[2] / summary[0] * 100) if summary[0] > 0 else 100
            }

        except Exception as e:
            logger.error(f"Failed to get commitment summary: {e}")
            return {"error": str(e)}

    async def get_cohort_commitments(self, cohort_id: str) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"error": "CommitmentTracker not initialized", "commitments": []}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            commitment_ids = self.commitment_tracker.functions.getCohortCommitments(cohort_hash).call()

            status_map = {0: "ACTIVE", 1: "FULFILLED", 2: "EXPIRED", 3: "CANCELLED"}
            commitments = []
            for cid in commitment_ids:
                c = self.commitment_tracker.functions.getCommitment(cid).call()
                commitments.append({
                    "commitment_id": c[0].hex(),
                    "cohort_hash": c[1].hex(),
                    "researcher": c[2],
                    "commitment_type": c[3].hex(),
                    "deadline": c[5],
                    "status": status_map.get(c[6], "UNKNOWN"),
                    "description": c[10],
                })

            return {
                "cohort_id": cohort_id,
                "commitments": commitments,
                "total": len(commitments),
            }

        except Exception as e:
            logger.error(f"Failed to get cohort commitments: {e}")
            return {"error": str(e), "commitments": []}

    async def batch_mark_expired(self, commitment_ids: list[str]) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"success": False, "error": "CommitmentTracker not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            id_bytes = [bytes.fromhex(cid.replace("0x", "")) for cid in commitment_ids]

            tx = self.commitment_tracker.functions.batchMarkExpired(
                id_bytes
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000 + 50000 * len(id_bytes),
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "marked_count": len(id_bytes),
                "commitment_ids": commitment_ids,
            }

        except Exception as e:
            logger.error(f"Failed to batch mark expired: {e}")
            return {"success": False, "error": str(e)}

    async def cancel_commitment(self, commitment_id: str, reason: str = "") -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"success": False, "error": "CommitmentTracker not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            commitment_bytes = bytes.fromhex(commitment_id.replace("0x", ""))

            tx = self.commitment_tracker.functions.cancelCommitment(
                commitment_bytes,
                reason,
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "commitment_id": commitment_id,
                "action": "cancelled",
            }

        except Exception as e:
            logger.error(f"Failed to cancel commitment: {e}")
            return {"success": False, "error": str(e)}

    async def extend_commitment_deadline(self, commitment_id: str, additional_days: int) -> dict[str, Any]:
        if not self.commitment_tracker:
            return {"success": False, "error": "CommitmentTracker not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            commitment_bytes = bytes.fromhex(commitment_id.replace("0x", ""))

            tx = self.commitment_tracker.functions.extendDeadline(
                commitment_bytes,
                additional_days,
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "commitment_id": commitment_id,
                "additional_days": additional_days,
            }

        except Exception as e:
            logger.error(f"Failed to extend commitment deadline: {e}")
            return {"success": False, "error": str(e)}

    def get_forwarder_domain(self) -> dict[str, Any]:
        if not self.trusted_forwarder:
            return {"error": "TrustedForwarder not initialized"}

        try:

            chain_id = self.w3.eth.chain_id
            forwarder_address = self.trusted_forwarder.address

            return {
                "name": "DUOConsentForwarder",
                "version": "1",
                "chainId": chain_id,
                "verifyingContract": forwarder_address
            }
        except Exception as e:
            logger.error(f"Failed to get forwarder domain: {e}")
            return {"error": str(e)}

    def get_forward_request_types(self) -> dict:
        return {
            "ForwardRequest": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "gas", "type": "uint256"},
                {"name": "nonce", "type": "uint256"},
                {"name": "deadline", "type": "uint48"},
                {"name": "data", "type": "bytes"}
            ]
        }

    async def get_user_nonce(self, user_address: str) -> int:
        if not self.trusted_forwarder:
            return 0

        try:
            nonce = self.trusted_forwarder.functions.nonces(
                self.w3.to_checksum_address(user_address)
            ).call()
            return nonce
        except Exception as e:
            logger.error(f"Failed to get user nonce: {e}")
            return 0

    async def prepare_meta_transaction(
        self,
        user_email: str,
        target_contract: str,
        function_data: bytes,
        gas_limit: int = 500000,
        deadline_minutes: int = 30
    ) -> dict[str, Any]:
        if not self.trusted_forwarder:
            return {"error": "TrustedForwarder not initialized"}

        try:
            user_address = derive_address_from_email(user_email)
            nonce = await self.get_user_nonce(user_address)

            import time
            deadline = int(time.time()) + (deadline_minutes * 60)

            forward_request = {
                "from": user_address,
                "to": self.w3.to_checksum_address(target_contract),
                "value": 0,
                "gas": gas_limit,
                "nonce": nonce,
                "deadline": deadline,
                "data": function_data.hex() if isinstance(function_data, bytes) else function_data
            }

            domain = self.get_forwarder_domain()
            types = self.get_forward_request_types()

            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"}
                    ],
                    **types
                },
                "primaryType": "ForwardRequest",
                "domain": domain,
                "message": forward_request
            }

            return {
                "success": True,
                "typed_data": typed_data,
                "forward_request": forward_request,
                "user_address": user_address,
                "forwarder_address": self.trusted_forwarder.address
            }

        except Exception as e:
            logger.error(f"Failed to prepare meta-transaction: {e}")
            return {"error": str(e)}

    async def execute_meta_transaction(
        self,
        forward_request: dict,
        signature: str
    ) -> dict[str, Any]:
        if not self.trusted_forwarder:
            return {"success": False, "error": "TrustedForwarder not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:

            sig_bytes = bytes.fromhex(signature.replace("0x", ""))

            data = forward_request["data"]
            if isinstance(data, str):
                data = bytes.fromhex(data.replace("0x", ""))

            request_tuple = (
                self.w3.to_checksum_address(forward_request["from"]),
                self.w3.to_checksum_address(forward_request["to"]),
                forward_request["value"],
                forward_request["gas"],
                forward_request["nonce"],
                forward_request["deadline"],
                data
            )

            tx = self.trusted_forwarder.functions.execute(
                request_tuple,
                sig_bytes
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": forward_request["gas"] + 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": receipt["status"] == 1,
                "tx_hash": receipt["transactionHash"].hex(),
                "gas_used": receipt["gasUsed"],
                "from_address": forward_request["from"],
                "to_address": forward_request["to"]
            }

        except Exception as e:
            logger.error(f"Failed to execute meta-transaction: {e}")
            return {"success": False, "error": str(e)}

    async def prepare_gasless_consent_record(
        self,
        owner_email: str,
        cohort_id: str,
        permission: str,
        modifiers: list[str],
        disease_code: str | None = None,
        expiration_days: int = 0
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"error": "ConsentVault not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            perm_bytes4 = PERMISSION_VALUES.get(permission.upper(), DUOPermission.GRU)
            mod_bitmask = int(get_modifiers_bitmask(modifiers))
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)

            function_data = self.consent_vault.encode_abi(
                abi_element_identifier="recordConsent",
                args=[
                    cohort_hash,
                    perm_bytes4.to_bytes(4, 'big'),
                    mod_bitmask,
                    disease_bytes,
                    [],
                    [],
                    expiration_days,
                    ""
                ]
            )

            result = await self.prepare_meta_transaction(
                user_email=owner_email,
                target_contract=self.consent_vault.address,
                function_data=bytes.fromhex(function_data[2:]),
                gas_limit=500000
            )

            if "error" in result:
                return result

            return {
                **result,
                "consent_details": {
                    "cohort_id": cohort_id,
                    "cohort_hash": cohort_hash.hex(),
                    "permission": permission.upper(),
                    "modifiers": modifiers
                }
            }

        except Exception as e:
            logger.error(f"Failed to prepare gasless consent: {e}")
            return {"error": str(e)}

    async def prepare_gasless_access_request(
        self,
        requester_email: str,
        cohort_id: str,
        intended_use: str,
        purpose: int = 0
    ) -> dict[str, Any]:
        if not self.consent_vault:
            return {"error": "ConsentVault not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            use_bytes4 = PERMISSION_VALUES.get(intended_use.upper(), DUOPermission.HMB)

            function_data = self.consent_vault.encode_abi(
                abi_element_identifier="requestAccess",
                args=[
                    cohort_hash,
                    use_bytes4.to_bytes(4, 'big'),
                    purpose,
                    bytes(32),
                    bytes(32),
                ]
            )

            result = await self.prepare_meta_transaction(
                user_email=requester_email,
                target_contract=self.consent_vault.address,
                function_data=bytes.fromhex(function_data[2:]),
                gas_limit=300000
            )

            if "error" in result:
                return result

            return {
                **result,
                "access_details": {
                    "cohort_id": cohort_id,
                    "cohort_hash": cohort_hash.hex(),
                    "intended_use": intended_use.upper()
                }
            }

        except Exception as e:
            logger.error(f"Failed to prepare gasless access request: {e}")
            return {"error": str(e)}

    def verify_meta_transaction_signature(
        self,
        forward_request: dict,
        signature: str
    ) -> dict[str, Any]:
        try:
            from eth_account.messages import encode_typed_data
            from eth_account import Account

            domain = self.get_forwarder_domain()
            types = self.get_forward_request_types()

            typed_data = {
                "types": {
                    "EIP712Domain": [
                        {"name": "name", "type": "string"},
                        {"name": "version", "type": "string"},
                        {"name": "chainId", "type": "uint256"},
                        {"name": "verifyingContract", "type": "address"}
                    ],
                    **types
                },
                "primaryType": "ForwardRequest",
                "domain": domain,
                "message": forward_request
            }

            encoded = encode_typed_data(full_message=typed_data)
            recovered = Account.recover_message(encoded, signature=signature)

            expected_signer = forward_request["from"]
            is_valid = recovered.lower() == expected_signer.lower()

            return {
                "valid": is_valid,
                "recovered_signer": recovered,
                "expected_signer": expected_signer
            }

        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return {"valid": False, "error": str(e)}

    async def mint_consent_nft(
        self,
        owner_email: str,
        cohort_id: str,
        permission: str,
        modifiers: list[str],
        disease_code: str | None = None,
        countries_merkle_root: str | None = None,
        institutions_merkle_root: str | None = None,
        valid_days: int = 0,
        metadata_uri: str = ""
    ) -> dict[str, Any]:
        if not self.consent_token:
            return {"success": False, "error": "DUOConsentToken not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            owner_address = derive_address_from_email(owner_email)
            cohort_hash = get_cohort_hash(cohort_id)
            perm_bytes4 = PERMISSION_VALUES.get(permission.upper(), DUOPermission.GRU)
            mod_bitmask = int(get_modifiers_bitmask(modifiers))
            disease_bytes = self.w3.keccak(text=disease_code) if disease_code else bytes(32)
            countries_root = bytes.fromhex(countries_merkle_root.replace("0x", "")) if countries_merkle_root else bytes(32)
            institutions_root = bytes.fromhex(institutions_merkle_root.replace("0x", "")) if institutions_merkle_root else bytes(32)

            tx = self.consent_token.functions.mintConsent(
                owner_address,
                cohort_hash,
                perm_bytes4.to_bytes(4, 'big'),
                mod_bitmask,
                disease_bytes,
                countries_root,
                institutions_root,
                valid_days,
                metadata_uri
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 600000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            token_id = int.from_bytes(cohort_hash, 'big')

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "token_id": hex(token_id),
                "cohort_hash": cohort_hash.hex(),
                "owner_address": owner_address,
                "permission": permission.upper(),
                "modifiers": modifiers,
                "soulbound": True
            }

        except Exception as e:
            logger.error(f"Failed to mint consent NFT: {e}")
            return {"success": False, "error": str(e)}

    async def get_consent_nft(self, cohort_id: str) -> dict[str, Any]:
        if not self.consent_token:
            return {"error": "DUOConsentToken not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            token_id = int.from_bytes(cohort_hash, 'big')

            consent = self.consent_token.functions.getConsentData(token_id).call()

            owner = self.consent_token.functions.ownerOf(token_id).call()

            owners = self.consent_token.functions.getOwners(token_id).call()

            is_valid = self.consent_token.functions.isConsentValid(token_id).call()

            try:
                token_uri = self.consent_token.functions.tokenURI(token_id).call()
            except:
                token_uri = None

            perm_int = int.from_bytes(consent[1], 'big') if isinstance(consent[1], bytes) else consent[1]
            perm_code = PERMISSION_CODES.get(perm_int, "UNKNOWN")
            mod_list = bitmask_to_modifiers(consent[2])

            return {
                "token_id": hex(token_id),
                "cohort_hash": consent[0].hex(),
                "permission": perm_code,
                "permission_label": PERMISSION_LABELS.get(perm_code, perm_code),
                "modifiers": mod_list,
                "modifier_details": get_modifier_details(mod_list),
                "disease_code": consent[3].hex() if consent[3] != bytes(32) else None,
                "countries_merkle_root": consent[4].hex() if consent[4] != bytes(32) else None,
                "institutions_merkle_root": consent[5].hex() if consent[5] != bytes(32) else None,
                "valid_from": consent[6],
                "valid_until": consent[7] if consent[7] > 0 else None,
                "owner_count": consent[8],
                "active": consent[9],
                "primary_owner": owner,
                "all_owners": owners,
                "is_valid": is_valid,
                "token_uri": token_uri,
                "soulbound": True
            }

        except Exception as e:
            logger.error(f"Failed to get consent NFT: {e}")
            return {"error": str(e)}

    async def deactivate_consent_nft(self, cohort_id: str, owner_email: str) -> dict[str, Any]:
        if not self.consent_token:
            return {"success": False, "error": "DUOConsentToken not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            token_id = int.from_bytes(cohort_hash, 'big')

            tx = self.consent_token.functions.deactivateConsent(token_id).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "token_id": hex(token_id),
                "action": "deactivated"
            }

        except Exception as e:
            logger.error(f"Failed to deactivate consent NFT: {e}")
            return {"success": False, "error": str(e)}

    async def burn_consent_nft(self, cohort_id: str, owner_email: str) -> dict[str, Any]:
        if not self.consent_token:
            return {"success": False, "error": "DUOConsentToken not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            token_id = int.from_bytes(cohort_hash, 'big')

            tx = self.consent_token.functions.burnConsent(token_id).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "token_id": hex(token_id),
                "action": "burned"
            }

        except Exception as e:
            logger.error(f"Failed to burn consent NFT: {e}")
            return {"success": False, "error": str(e)}

    async def calculate_compliance_score_nft(
        self,
        cohort_id: str,
        requester_email: str,
        intended_use: str,
        requester_type: int = 1
    ) -> dict[str, Any]:
        if not self.consent_token:
            return {"error": "DUOConsentToken not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            token_id = int.from_bytes(cohort_hash, 'big')
            requester_address = derive_address_from_email(requester_email)
            use_bytes4 = PERMISSION_VALUES.get(intended_use.upper(), DUOPermission.HMB)

            score = self.consent_token.functions.calculateComplianceScore(
                token_id,
                requester_address,
                use_bytes4.to_bytes(4, 'big'),
                requester_type
            ).call()

            risk_levels = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

            return {
                "total_score": score[0],
                "permission_score": score[1],
                "modifier_score": score[2],
                "attestation_score": score[3],
                "trust_score": score[4],
                "risk_level": risk_levels.get(score[5], "UNKNOWN"),
                "auto_approve": score[0] >= 800,
                "requester_address": requester_address
            }

        except Exception as e:
            logger.error(f"Failed to calculate compliance score: {e}")
            return {"error": str(e)}

    async def issue_access_credential(
        self,
        requester_email: str,
        cohort_id: str,
        granted_use: str,
        granted_purpose: int = 0,
        compliance_score: int = 500,
        risk_level: int = 1,
        expires_at: int = 0,
        conditions: str = "",
        metadata_uri: str = ""
    ) -> dict[str, Any]:
        if not self.access_credential_nft:
            return {"success": False, "error": "AccessCredentialNFT not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            requester_address = derive_address_from_email(requester_email)
            cohort_hash = get_cohort_hash(cohort_id)
            consent_token_id = int.from_bytes(cohort_hash, 'big')
            use_bytes4 = PERMISSION_VALUES.get(granted_use.upper(), DUOPermission.HMB)

            tx = self.access_credential_nft.functions.issueCredential(
                requester_address,
                consent_token_id,
                use_bytes4.to_bytes(4, 'big'),
                granted_purpose,
                compliance_score,
                risk_level,
                expires_at,
                conditions,
                metadata_uri
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 400000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            credential_id = None
            for log in receipt.get("logs", []):
                if len(log.get("topics", [])) >= 2:
                    credential_id = int(log["topics"][1].hex(), 16)
                    break

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "credential_id": credential_id,
                "requester_address": requester_address,
                "consent_token_id": hex(consent_token_id),
                "granted_use": granted_use.upper(),
                "compliance_score": compliance_score
            }

        except Exception as e:
            logger.error(f"Failed to issue access credential: {e}")
            return {"success": False, "error": str(e)}

    async def get_access_credential(self, credential_id: int) -> dict[str, Any]:
        if not self.access_credential_nft:
            return {"error": "AccessCredentialNFT not initialized"}

        try:
            cred = self.access_credential_nft.functions.getCredential(credential_id).call()
            is_valid = self.access_credential_nft.functions.isCredentialValid(credential_id).call()

            try:
                owner = self.access_credential_nft.functions.ownerOf(credential_id).call()
            except:
                owner = None

            use_bytes = cred[3]
            use_int = int.from_bytes(use_bytes, 'big') if isinstance(use_bytes, bytes) else use_bytes
            use_code = PERMISSION_CODES.get(use_int, "UNKNOWN")

            risk_levels = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

            return {
                "credential_id": credential_id,
                "consent_token_id": hex(cred[0]),
                "consent_tba": cred[1],
                "issuer": cred[2],
                "granted_use": use_code,
                "granted_purpose": cred[4],
                "granted_at": cred[5],
                "expires_at": cred[6] if cred[6] > 0 else None,
                "compliance_score": cred[7],
                "risk_level": risk_levels.get(cred[8], "UNKNOWN"),
                "revoked": cred[9],
                "conditions": cred[10],
                "is_valid": is_valid,
                "current_owner": owner
            }

        except Exception as e:
            logger.error(f"Failed to get access credential: {e}")
            return {"error": str(e)}

    async def get_requester_credentials(self, requester_email: str) -> dict[str, Any]:
        if not self.access_credential_nft:
            return {"error": "AccessCredentialNFT not initialized", "credentials": []}

        try:
            requester_address = derive_address_from_email(requester_email)

            cred_ids = self.access_credential_nft.functions.getCredentialsForRequester(
                requester_address
            ).call()

            valid_ids = self.access_credential_nft.functions.getValidCredentials(
                requester_address
            ).call()

            credentials = []
            for cid in cred_ids:
                cred = await self.get_access_credential(cid)
                if "error" not in cred:
                    credentials.append(cred)

            return {
                "requester_address": requester_address,
                "total": len(credentials),
                "valid_count": len(valid_ids),
                "credentials": credentials
            }

        except Exception as e:
            logger.error(f"Failed to get requester credentials: {e}")
            return {"error": str(e), "credentials": []}

    async def revoke_access_credential(
        self,
        credential_id: int,
        reason: str = ""
    ) -> dict[str, Any]:
        if not self.access_credential_nft:
            return {"success": False, "error": "AccessCredentialNFT not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            tx = self.access_credential_nft.functions.revokeCredential(
                credential_id,
                reason
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "credential_id": credential_id,
                "action": "revoked"
            }

        except Exception as e:
            logger.error(f"Failed to revoke access credential: {e}")
            return {"success": False, "error": str(e)}

    async def check_valid_access(self, requester_email: str, cohort_id: str) -> dict[str, Any]:
        if not self.access_credential_nft:
            return {"has_access": False, "error": "AccessCredentialNFT not initialized"}

        try:
            requester_address = derive_address_from_email(requester_email)
            cohort_hash = get_cohort_hash(cohort_id)
            consent_token_id = int.from_bytes(cohort_hash, 'big')

            has_access = self.access_credential_nft.functions.hasValidAccess(
                requester_address,
                consent_token_id
            ).call()

            return {
                "has_access": has_access,
                "requester_address": requester_address,
                "consent_token_id": hex(consent_token_id)
            }

        except Exception as e:
            logger.error(f"Failed to check valid access: {e}")
            return {"has_access": False, "error": str(e)}

    async def create_semaphore_group(
        self,
        cohort_id: str,
        group_depth: int = 20
    ) -> dict[str, Any]:
        if not self.identity_registry:
            return {"success": False, "error": "IdentityRegistry not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            tx = self.identity_registry.functions.createGroup(
                group_id,
                group_depth,
                self.relayer_account.address
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "group_id": str(group_id),
                "cohort_id": cohort_id,
                "depth": group_depth
            }

        except Exception as e:
            logger.error(f"Failed to create Semaphore group: {e}")
            return {"success": False, "error": str(e)}

    async def add_identity_to_group(
        self,
        cohort_id: str,
        identity_commitment: str
    ) -> dict[str, Any]:
        if not self.identity_registry:
            return {"success": False, "error": "IdentityRegistry not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            commitment = int(identity_commitment, 16) if identity_commitment.startswith("0x") else int(identity_commitment)

            tx = self.identity_registry.functions.addMember(
                group_id,
                commitment
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 150000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "group_id": str(group_id),
                "identity_commitment": identity_commitment
            }

        except Exception as e:
            logger.error(f"Failed to add identity to group: {e}")
            return {"success": False, "error": str(e)}

    async def batch_add_identities(
        self,
        cohort_id: str,
        identity_commitments: list[str]
    ) -> dict[str, Any]:
        if not self.identity_registry:
            return {"success": False, "error": "IdentityRegistry not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            commitments = [
                int(c, 16) if c.startswith("0x") else int(c)
                for c in identity_commitments
            ]

            tx = self.identity_registry.functions.addMembers(
                group_id,
                commitments
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 50000 + 50000 * len(commitments),
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "group_id": str(group_id),
                "added_count": len(commitments)
            }

        except Exception as e:
            logger.error(f"Failed to batch add identities: {e}")
            return {"success": False, "error": str(e)}

    async def get_group_info(self, cohort_id: str) -> dict[str, Any]:
        if not self.identity_registry:
            return {"error": "IdentityRegistry not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            depth = self.identity_registry.functions.getDepth(group_id).call()
            size = self.identity_registry.functions.getSize(group_id).call()
            root = self.identity_registry.functions.getRoot(group_id).call()

            return {
                "group_id": str(group_id),
                "cohort_id": cohort_id,
                "depth": depth,
                "member_count": size,
                "merkle_root": hex(root) if root else "0x0",
                "exists": depth > 0
            }

        except Exception as e:
            logger.error(f"Failed to get group info: {e}")
            return {"error": str(e)}

    async def is_member_of_group(
        self,
        cohort_id: str,
        identity_commitment: str
    ) -> dict[str, Any]:
        if not self.identity_registry:
            return {"is_member": False, "error": "IdentityRegistry not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            commitment = int(identity_commitment, 16) if identity_commitment.startswith("0x") else int(identity_commitment)

            is_member = self.identity_registry.functions.isMember(
                group_id,
                commitment
            ).call()

            return {
                "is_member": is_member,
                "group_id": str(group_id),
                "identity_commitment": identity_commitment
            }

        except Exception as e:
            logger.error(f"Failed to check membership: {e}")
            return {"is_member": False, "error": str(e)}

    async def check_nullifier(self, nullifier_hash: str) -> dict[str, Any]:
        if not self.nullifier_registry:
            return {"error": "NullifierRegistry not initialized"}

        try:
            nullifier = int(nullifier_hash, 16) if nullifier_hash.startswith("0x") else int(nullifier_hash)

            is_used = self.nullifier_registry.functions.isNullifierUsed(nullifier).call()

            result = {
                "nullifier_hash": nullifier_hash,
                "is_used": is_used
            }

            if is_used:

                usage = self.nullifier_registry.functions.nullifierUsage(nullifier).call()
                result["used_at"] = datetime.fromtimestamp(usage[0]).isoformat() if usage[0] else None
                result["used_for_cohort"] = usage[1].hex() if usage[1] else None

            return result

        except Exception as e:
            logger.error(f"Failed to check nullifier: {e}")
            return {"error": str(e)}

    async def get_nullifier_stats(self) -> dict[str, Any]:
        if not self.nullifier_registry:
            return {"error": "NullifierRegistry not initialized"}

        try:
            total_used = self.nullifier_registry.functions.totalNullifiersUsed().call()

            return {
                "total_nullifiers_used": total_used,
                "contract_address": self.settings.nullifier_registry_address
            }

        except Exception as e:
            logger.error(f"Failed to get nullifier stats: {e}")
            return {"error": str(e)}

    async def verify_and_grant_zk_access(
        self,
        cohort_id: str,
        signal_hash: str,
        nullifier_hash: str,
        proof: list[int],
        intended_use: str = "HMB"
    ) -> dict[str, Any]:
        if not self.zk_verifier:
            return {"success": False, "error": "ZKConsentVerifier not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            signal = int(signal_hash, 16) if signal_hash.startswith("0x") else int(signal_hash)
            nullifier = int(nullifier_hash, 16) if nullifier_hash.startswith("0x") else int(nullifier_hash)

            merkle_root = self.identity_registry.functions.getRoot(group_id).call()

            tx = self.zk_verifier.functions.verifyAndGrant(
                group_id,
                merkle_root,
                signal,
                nullifier,
                proof
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 500000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "group_id": str(group_id),
                "nullifier_hash": nullifier_hash,
                "cohort_id": cohort_id,
                "privacy": "Identity not revealed - ZK proof verified"
            }

        except Exception as e:
            logger.error(f"Failed to verify ZK proof: {e}")
            return {"success": False, "error": str(e)}

    async def verify_membership_proof(
        self,
        cohort_id: str,
        merkle_root: str,
        signal_hash: str,
        nullifier_hash: str,
        proof: list[int]
    ) -> dict[str, Any]:
        if not self.zk_verifier:
            return {"valid": False, "error": "ZKConsentVerifier not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            group_id = int.from_bytes(cohort_hash, 'big') % (2**128)

            root = int(merkle_root, 16) if merkle_root.startswith("0x") else int(merkle_root)
            signal = int(signal_hash, 16) if signal_hash.startswith("0x") else int(signal_hash)
            nullifier = int(nullifier_hash, 16) if nullifier_hash.startswith("0x") else int(nullifier_hash)

            is_valid = self.zk_verifier.functions.verifyProof(
                group_id,
                root,
                signal,
                nullifier,
                proof
            ).call()

            return {
                "valid": is_valid,
                "group_id": str(group_id),
                "cohort_id": cohort_id
            }

        except Exception as e:
            logger.error(f"Failed to verify membership proof: {e}")
            return {"valid": False, "error": str(e)}

    async def get_zk_access_grant(self, nullifier_hash: str) -> dict[str, Any]:
        if not self.zk_verifier:
            return {"error": "ZKConsentVerifier not initialized"}

        try:
            nullifier = int(nullifier_hash, 16) if nullifier_hash.startswith("0x") else int(nullifier_hash)

            grant = self.zk_verifier.functions.accessGrants(nullifier).call()

            if not grant[0]:
                return {"error": "No access grant found for this nullifier"}

            return {
                "granted": True,
                "nullifier_hash": nullifier_hash,
                "group_id": str(grant[1]),
                "signal_hash": hex(grant[2]),
                "granted_at": datetime.fromtimestamp(grant[3]).isoformat() if grant[3] else None
            }

        except Exception as e:
            logger.error(f"Failed to get ZK access grant: {e}")
            return {"error": str(e)}

    async def get_zk_verifier_stats(self) -> dict[str, Any]:
        if not self.zk_verifier:
            return {"error": "ZKConsentVerifier not initialized"}

        try:
            total_grants = self.zk_verifier.functions.totalAccessGrants().call()

            return {
                "total_zk_access_grants": total_grants,
                "verifier_address": self.settings.zk_consent_verifier_address,
                "nullifier_registry": self.settings.nullifier_registry_address,
                "identity_registry": self.settings.identity_registry_address
            }

        except Exception as e:
            logger.error(f"Failed to get ZK verifier stats: {e}")
            return {"error": str(e)}

    def get_eas_schema_ids(self) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"error": "DUOAttestationResolver not initialized"}

        try:
            return {
                "irb_approval": self.attestation_resolver.functions.schemaIRBApproval().call().hex(),
                "collaboration": self.attestation_resolver.functions.schemaCollaboration().call().hex(),
                "publication": self.attestation_resolver.functions.schemaPublication().call().hex(),
                "return_data": self.attestation_resolver.functions.schemaReturnData().call().hex(),
                "geographic": self.attestation_resolver.functions.schemaGeographic().call().hex(),
                "institution": self.attestation_resolver.functions.schemaInstitution().call().hex(),
            }
        except Exception as e:
            logger.error(f"Failed to get EAS schema IDs: {e}")
            return {"error": str(e)}

    async def check_eas_irb(
        self,
        cohort_id: str,
        subject_email: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"has_irb": False, "error": "DUOAttestationResolver not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            subject_address = derive_address_from_email(subject_email)

            result = self.attestation_resolver.functions.hasValidIRB(
                cohort_hash,
                subject_address
            ).call()

            return {
                "has_irb": result[0],
                "attestation_uid": result[1].hex() if result[1] else None,
                "cohort_id": cohort_id,
                "subject_address": subject_address
            }

        except Exception as e:
            logger.error(f"Failed to check EAS IRB: {e}")
            return {"has_irb": False, "error": str(e)}

    async def check_eas_collaboration(
        self,
        cohort_id: str,
        partner_email: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"has_collaboration": False, "error": "DUOAttestationResolver not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            partner_address = derive_address_from_email(partner_email)

            result = self.attestation_resolver.functions.hasValidCollaboration(
                cohort_hash,
                partner_address
            ).call()

            return {
                "has_collaboration": result[0],
                "attestation_uid": result[1].hex() if result[1] else None,
                "cohort_id": cohort_id,
                "partner_address": partner_address
            }

        except Exception as e:
            logger.error(f"Failed to check EAS collaboration: {e}")
            return {"has_collaboration": False, "error": str(e)}

    async def check_eas_publication_commitment(
        self,
        cohort_id: str,
        subject_email: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"has_commitment": False, "error": "DUOAttestationResolver not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            subject_address = derive_address_from_email(subject_email)

            result = self.attestation_resolver.functions.hasPublicationCommitment(
                cohort_hash,
                subject_address
            ).call()

            return {
                "has_commitment": result[0],
                "attestation_uid": result[1].hex() if result[1] else None,
                "cohort_id": cohort_id,
                "subject_address": subject_address
            }

        except Exception as e:
            logger.error(f"Failed to check EAS publication commitment: {e}")
            return {"has_commitment": False, "error": str(e)}

    async def check_eas_return_data_commitment(
        self,
        cohort_id: str,
        subject_email: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"has_commitment": False, "error": "DUOAttestationResolver not initialized"}

        try:
            cohort_hash = get_cohort_hash(cohort_id)
            subject_address = derive_address_from_email(subject_email)

            result = self.attestation_resolver.functions.hasReturnDataCommitment(
                cohort_hash,
                subject_address
            ).call()

            return {
                "has_commitment": result[0],
                "attestation_uid": result[1].hex() if result[1] else None,
                "cohort_id": cohort_id,
                "subject_address": subject_address
            }

        except Exception as e:
            logger.error(f"Failed to check EAS return data commitment: {e}")
            return {"has_commitment": False, "error": str(e)}

    async def get_eas_geographic_attestation(
        self,
        subject_email: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"error": "DUOAttestationResolver not initialized"}

        try:
            subject_address = derive_address_from_email(subject_email)

            result = self.attestation_resolver.functions.getGeographicAttestation(
                subject_address
            ).call()

            if not result[0]:
                return {
                    "has_attestation": False,
                    "subject_address": subject_address
                }

            return {
                "has_attestation": result[0],
                "attestation_uid": result[1].hex() if result[1] else None,
                "country_code": result[2].decode('utf-8', errors='ignore').strip('\x00') if result[2] else None,
                "subject_address": subject_address
            }

        except Exception as e:
            logger.error(f"Failed to get EAS geographic attestation: {e}")
            return {"error": str(e)}

    async def get_eas_institution_attestation(
        self,
        subject_email: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"error": "DUOAttestationResolver not initialized"}

        try:
            subject_address = derive_address_from_email(subject_email)

            result = self.attestation_resolver.functions.getInstitutionAttestation(
                subject_address
            ).call()

            if not result[0]:
                return {
                    "has_attestation": False,
                    "subject_address": subject_address
                }

            return {
                "has_attestation": result[0],
                "attestation_uid": result[1].hex() if result[1] else None,
                "ror_id": result[2] if result[2] else None,
                "institution_name": result[3] if result[3] else None,
                "subject_address": subject_address
            }

        except Exception as e:
            logger.error(f"Failed to get EAS institution attestation: {e}")
            return {"error": str(e)}

    async def add_trusted_attester(
        self,
        schema_type: str,
        attester_address: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"success": False, "error": "DUOAttestationResolver not initialized"}

        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:

            schema_map = {
                "irb": self.attestation_resolver.functions.schemaIRBApproval().call(),
                "collaboration": self.attestation_resolver.functions.schemaCollaboration().call(),
                "publication": self.attestation_resolver.functions.schemaPublication().call(),
                "return_data": self.attestation_resolver.functions.schemaReturnData().call(),
                "geographic": self.attestation_resolver.functions.schemaGeographic().call(),
                "institution": self.attestation_resolver.functions.schemaInstitution().call(),
            }

            schema_id = schema_map.get(schema_type.lower())
            if not schema_id:
                return {"success": False, "error": f"Unknown schema type: {schema_type}"}

            tx = self.attestation_resolver.functions.addTrustedAttester(
                schema_id,
                Web3.to_checksum_address(attester_address)
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000,
                "gasPrice": self.w3.eth.gas_price
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": True,
                "tx_hash": receipt["transactionHash"].hex(),
                "schema_type": schema_type,
                "attester": attester_address
            }

        except Exception as e:
            logger.error(f"Failed to add trusted attester: {e}")
            return {"success": False, "error": str(e)}

    async def is_trusted_attester(
        self,
        schema_type: str,
        attester_address: str
    ) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"error": "DUOAttestationResolver not initialized"}

        try:
            schema_map = {
                "irb": self.attestation_resolver.functions.schemaIRBApproval().call(),
                "collaboration": self.attestation_resolver.functions.schemaCollaboration().call(),
                "publication": self.attestation_resolver.functions.schemaPublication().call(),
                "return_data": self.attestation_resolver.functions.schemaReturnData().call(),
                "geographic": self.attestation_resolver.functions.schemaGeographic().call(),
                "institution": self.attestation_resolver.functions.schemaInstitution().call(),
            }

            schema_id = schema_map.get(schema_type.lower())
            if not schema_id:
                return {"error": f"Unknown schema type: {schema_type}"}

            is_trusted = self.attestation_resolver.functions.isTrustedAttester(
                schema_id,
                Web3.to_checksum_address(attester_address)
            ).call()

            return {
                "is_trusted": is_trusted,
                "schema_type": schema_type,
                "attester": attester_address
            }

        except Exception as e:
            logger.error(f"Failed to check trusted attester: {e}")
            return {"error": str(e)}

    async def get_eas_stats(self) -> dict[str, Any]:
        if not self.attestation_resolver:
            return {"error": "DUOAttestationResolver not initialized"}

        try:
            total = self.attestation_resolver.functions.getTotalAttestations().call()
            schemas = self.get_eas_schema_ids()

            return {
                "total_attestations": total,
                "resolver_address": self.settings.duo_attestation_resolver_address,
                "schemas": schemas if "error" not in schemas else {}
            }

        except Exception as e:
            logger.error(f"Failed to get EAS stats: {e}")
            return {"error": str(e)}

    async def submit_ibis_envelope(
        self,
        commitment: bytes,
        nullifier: bytes,
        blinded_epoch: bytes,
        epoch_key_hash: bytes,
        encrypted_payload: bytes,
        proof: bytes,
    ) -> dict[str, Any]:
        if not self.ibis_verifier:
            return {"success": False, "error": "IBISVerifier not initialized"}
        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:

            paused = self.ibis_verifier.functions.paused().call()
            if paused:
                return {"success": False, "error": "IBISVerifier is paused"}

            nullifier_used = self.ibis_verifier.functions.isNullifierUsed(
                nullifier
            ).call()
            if nullifier_used:
                return {"success": False, "error": "nullifier_already_used_onchain"}

            tx = self.ibis_verifier.functions.processEnvelope(
                commitment,
                nullifier,
                blinded_epoch,
                epoch_key_hash,
                encrypted_payload,
                proof,
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 1500000,
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": receipt["status"] == 1,
                "tx_hash": receipt["transactionHash"].hex(),
                "block_number": receipt["blockNumber"],
            }

        except Exception as e:
            logger.error(f"Failed to submit IBIS envelope on-chain: {e}")
            return {"success": False, "error": str(e)}

    async def initialize_ibis_slots(
        self, commitments: list[bytes]
    ) -> dict[str, Any]:
        if not self.ibis_verifier:
            return {"success": False, "error": "IBISVerifier not initialized"}
        if not self.relayer_account:
            return {"success": False, "error": "Relayer not configured"}

        try:

            commitment_bytes32 = [c.ljust(32, b'\x00')[:32] for c in commitments]

            tx = self.ibis_verifier.functions.batchInitializeSlots(
                commitment_bytes32
            ).build_transaction({
                "from": self.relayer_account.address,
                "nonce": self.w3.eth.get_transaction_count(self.relayer_account.address),
                "gas": 100000 + 60000 * len(commitments),
                "gasPrice": self.w3.eth.gas_price,
            })

            signed_tx = self.relayer_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return {
                "success": receipt["status"] == 1,
                "tx_hash": receipt["transactionHash"].hex(),
                "slots_initialized": len(commitments),
            }

        except Exception as e:
            logger.error(f"Failed to initialize IBIS slots: {e}")
            return {"success": False, "error": str(e)}

    async def get_ibis_state(self, commitment: bytes) -> dict[str, Any]:
        if not self.ibis_verifier:
            return {"error": "IBISVerifier not initialized"}

        try:
            encrypted_state, epoch_key_hash, blinded_epoch = (
                self.ibis_verifier.functions.getState(commitment).call()
            )
            return {
                "encrypted_state": encrypted_state.hex(),
                "epoch_key_hash": epoch_key_hash.hex(),
                "blinded_epoch": blinded_epoch.hex(),
                "exists": encrypted_state != b'\x00' * len(encrypted_state),
            }
        except Exception as e:
            logger.error(f"Failed to get IBIS state: {e}")
            return {"error": str(e)}

_blockchain_service: BlockchainService | None = None

def get_blockchain_service() -> BlockchainService:
    global _blockchain_service
    if _blockchain_service is None:
        _blockchain_service = BlockchainService()
    return _blockchain_service
