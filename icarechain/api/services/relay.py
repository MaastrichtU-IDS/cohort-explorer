import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from web3 import Web3
from eth_account import Account

from api.config import get_settings
from api.services.wallet import derive_address_from_email

logger = logging.getLogger(__name__)

FIXED_PAYLOAD_SIZE = 256

class TxType(str, Enum):
    IBIS_ENVELOPE = "ibis_envelope"
    IBIS_SLOT_INIT = "ibis_slot_init"
    CONSENT_RECORD = "consent_record"
    CONSENT_REVOKE = "consent_revoke"
    ACCESS_REQUEST = "access_request"
    IDENTITY_JOIN = "identity_join"
    ATTESTATION = "attestation"

GAS_ESTIMATES: dict[TxType, int] = {
    TxType.IBIS_ENVELOPE: 120_000,
    TxType.IBIS_SLOT_INIT: 80_000,
    TxType.CONSENT_RECORD: 500_000,
    TxType.CONSENT_REVOKE: 300_000,
    TxType.ACCESS_REQUEST: 1_000_000,
    TxType.IDENTITY_JOIN: 150_000,
    TxType.ATTESTATION: 200_000,
}

@dataclass
class RelayResult:
    success: bool
    tx_hash: str = ""
    gas_used: int = 0
    gas_cost_wei: int = 0
    gas_sponsored: bool = False
    institution_id: str = ""
    error: str = ""

@dataclass
class OnboardingResult:
    success: bool
    derived_address: str = ""
    institution_id: str = ""
    sponsored: bool = False
    daily_budget_wei: int = 0
    error: str = ""

class RelayService:
    def __init__(self, blockchain_service=None):
        self.settings = get_settings()
        self._blockchain = blockchain_service
        self._relay_pool: list = []
        self._init_relay_pool()

    def _init_relay_pool(self):

        if self.settings.relayer_private_key:
            primary = Account.from_key(self.settings.relayer_private_key)
            self._relay_pool.append(primary)

        extra_keys = getattr(self.settings, 'relay_pool_keys', '')
        if extra_keys:
            for key in extra_keys.split(','):
                key = key.strip()
                if key:
                    self._relay_pool.append(Account.from_key(key))

    @property
    def blockchain(self):
        if self._blockchain is None:
            from api.services.blockchain import BlockchainService
            self._blockchain = BlockchainService()
        return self._blockchain

    @property
    def w3(self) -> Web3:
        return self.blockchain.w3

    @property
    def relayer(self):
        if not self._relay_pool:
            return self.blockchain.relayer_account
        return secrets.choice(self._relay_pool)

    async def onboard_user(
        self,
        email: str,
        institution_id: str,
        role: str = "researcher",
    ) -> OnboardingResult:
        try:
            derived_address = derive_address_from_email(email)
            institution_bytes = Web3.keccak(text=institution_id)

            gas_sponsor = self.blockchain.gas_sponsor
            if gas_sponsor is None:
                return OnboardingResult(
                    success=True,
                    derived_address=derived_address,
                    institution_id=institution_id,
                    sponsored=False,
                    error="GasSponsor contract not deployed"
                )

            pool = gas_sponsor.functions.getPool(institution_bytes).call()
            pool_balance, _, daily_budget, _, _, pool_active = pool

            if not pool_active or pool_balance == 0:
                return OnboardingResult(
                    success=True,
                    derived_address=derived_address,
                    institution_id=institution_id,
                    sponsored=False,
                    error="Institution gas pool not funded or inactive"
                )

            tx = gas_sponsor.functions.addSponsoredMember(
                institution_bytes,
                Web3.to_checksum_address(derived_address)
            ).build_transaction(self._tx_params())

            receipt = await self._send_tx(tx)

            return OnboardingResult(
                success=receipt["status"] == 1,
                derived_address=derived_address,
                institution_id=institution_id,
                sponsored=True,
                daily_budget_wei=daily_budget,
            )

        except Exception as e:
            logger.error(f"Onboarding failed for {email}: {e}")
            return OnboardingResult(success=False, error=str(e))

    async def relay_transaction(
        self,
        user_email: str,
        tx_type: TxType,
        contract_call,
        gas_limit: int | None = None,
    ) -> RelayResult:
        if not self.relayer:
            return RelayResult(success=False, error="Relayer not configured")

        try:
            derived_address = derive_address_from_email(user_email)
            gas_estimate = gas_limit or GAS_ESTIMATES.get(tx_type, 200_000)

            selected_relay = self.relayer

            sponsorship = await self._check_sponsorship(
                derived_address, gas_estimate
            )

            tx = contract_call.build_transaction(
                self._tx_params(gas=gas_estimate, relay_account=selected_relay)
            )

            receipt = await self._send_tx(tx, relay_account=selected_relay)

            if receipt["status"] != 1:
                return RelayResult(
                    success=False,
                    tx_hash=receipt["transactionHash"].hex(),
                    error="Transaction reverted"
                )

            actual_gas = receipt["gasUsed"]
            gas_price = receipt.get("effectiveGasPrice", self.w3.eth.gas_price)
            gas_cost = actual_gas * gas_price

            if sponsorship["sponsored"]:
                await self._record_reimbursement(
                    derived_address, actual_gas, gas_price, tx_type
                )

            return RelayResult(
                success=True,
                tx_hash=receipt["transactionHash"].hex(),
                gas_used=actual_gas,
                gas_cost_wei=gas_cost,
                gas_sponsored=sponsorship["sponsored"],
                institution_id=sponsorship.get("institution_id", ""),
            )

        except Exception as e:
            logger.error(f"Relay failed for {user_email} ({tx_type}): {e}")
            return RelayResult(success=False, error=str(e))

    async def relay_ibis_envelope(
        self,
        user_email: str,
        commitment: bytes,
        nullifier: bytes,
        blinded_epoch: bytes,
        epoch_key_hash: bytes,
        encrypted_payload: bytes,
        proof: bytes,
    ) -> RelayResult:
        verifier = self.blockchain.ibis_verifier
        if verifier is None:
            return RelayResult(success=False, error="IBISVerifier not deployed")

        if len(encrypted_payload) < FIXED_PAYLOAD_SIZE:
            encrypted_payload = encrypted_payload + bytes(
                FIXED_PAYLOAD_SIZE - len(encrypted_payload)
            )

        contract_call = verifier.functions.processEnvelope(
            commitment, nullifier, blinded_epoch,
            epoch_key_hash, encrypted_payload, proof
        )

        return await self.relay_transaction(
            user_email, TxType.IBIS_ENVELOPE, contract_call
        )

    async def _check_sponsorship(
        self, member_address: str, estimated_gas: int
    ) -> dict[str, Any]:
        gas_sponsor = self.blockchain.gas_sponsor
        if gas_sponsor is None:
            return {"sponsored": False, "reason": "GasSponsor not deployed"}

        try:
            gas_price = self.w3.eth.gas_price
            estimated_cost = estimated_gas * gas_price

            can_sponsor, reason = gas_sponsor.functions.canSponsor(
                Web3.to_checksum_address(member_address),
                estimated_cost
            ).call()

            if can_sponsor:
                institution_id = gas_sponsor.functions.memberInstitution(
                    Web3.to_checksum_address(member_address)
                ).call()
                return {
                    "sponsored": True,
                    "institution_id": institution_id.hex(),
                }

            return {"sponsored": False, "reason": reason}

        except Exception as e:
            logger.warning(f"Sponsorship check failed: {e}")
            return {"sponsored": False, "reason": str(e)}

    async def _record_reimbursement(
        self,
        member_address: str,
        gas_used: int,
        gas_price: int,
        tx_type: TxType,
    ) -> None:
        gas_sponsor = self.blockchain.gas_sponsor
        if gas_sponsor is None:
            return

        try:
            tx_type_bytes = Web3.keccak(text=tx_type.value)
            tx = gas_sponsor.functions.reimburse(
                Web3.to_checksum_address(member_address),
                gas_used,
                gas_price,
                tx_type_bytes,
            ).build_transaction(self._tx_params())

            await self._send_tx(tx)

        except Exception as e:

            logger.error(f"Reimbursement recording failed: {e}")

    async def get_institution_gas_report(
        self, institution_id: str
    ) -> dict[str, Any]:
        gas_sponsor = self.blockchain.gas_sponsor
        if gas_sponsor is None:
            return {"error": "GasSponsor not deployed"}

        try:
            institution_bytes = Web3.keccak(text=institution_id)
            pool = gas_sponsor.functions.getPool(institution_bytes).call()

            balance, total_spent, daily_budget, max_members, member_count, active = pool

            return {
                "institution_id": institution_id,
                "balance_wei": balance,
                "balance_eth": Web3.from_wei(balance, "ether"),
                "total_spent_wei": total_spent,
                "total_spent_eth": Web3.from_wei(total_spent, "ether"),
                "daily_budget_per_member_wei": daily_budget,
                "max_members": max_members,
                "current_members": member_count,
                "active": active,
                "estimated_txs_remaining": balance // (60_000 * self.w3.eth.gas_price)
                if self.w3.eth.gas_price > 0 else 0,
            }

        except Exception as e:
            return {"error": str(e)}

    async def get_member_gas_usage(
        self, email: str, institution_id: str
    ) -> dict[str, Any]:
        gas_sponsor = self.blockchain.gas_sponsor
        if gas_sponsor is None:
            return {"error": "GasSponsor not deployed"}

        try:
            member_address = derive_address_from_email(email)
            institution_bytes = Web3.keccak(text=institution_id)

            usage = gas_sponsor.functions.getMemberUsage(
                institution_bytes,
                Web3.to_checksum_address(member_address)
            ).call()

            daily_spent, daily_tx_count, total_spent, total_tx_count, active = usage

            return {
                "email": email,
                "derived_address": member_address,
                "daily_spent_wei": daily_spent,
                "daily_tx_count": daily_tx_count,
                "total_spent_wei": total_spent,
                "total_spent_eth": Web3.from_wei(total_spent, "ether"),
                "total_tx_count": total_tx_count,
                "active": active,
            }

        except Exception as e:
            return {"error": str(e)}

    def _tx_params(self, gas: int = 200_000, relay_account=None) -> dict:
        account = relay_account or self.relayer
        return {
            "from": account.address,
            "nonce": self.w3.eth.get_transaction_count(account.address),
            "gas": gas,
            "gasPrice": self.w3.eth.gas_price,
        }

    async def _send_tx(self, tx: dict, relay_account=None) -> dict:
        account = relay_account or self.relayer
        signed = account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
