# icarechain

Blockchain-backed [GA4GH DUO](https://github.com/EBISPOT/DUO) consent management. A FastAPI service over a Solidity contract suite. Deterministic per-role smart accounts. IBIS lifecycle envelopes for unlinkable on-chain operations.

## Quick start

```bash
docker compose up -d --build
```

Service ports: API `8020`, Hardhat RPC `8559`, Redis `6391`. Swagger at `http://localhost:8020/docs`.

## Identity model

| Object | Created at | Lifetime |
|---|---|---|
| EOA | `derive_address_from_email(email)` — deterministic | forever |
| Role smart account (per role) | `/auth/roles/{role}/activate` or first state-changing call | forever |
| Session token | `/auth/verify` | 24h |
| Access credential NFT | Successful access request | until revoked |
| Consent NFT (ERC-5192 soulbound) | `/providers/consents` | until revoked |

The vault treats the SCA as the on-chain principal. The EOA only signs. The relayer pays gas and submits every transaction via EIP-712 + ERC-1271.

## Access-request cascade

`POST /api/requesters/access-requests` is one HTTP call. When the cohort carries `PUB` / `RTN` / `MOR` modifiers, the API:

1. Resolves the requester profile (404 if absent).
2. Fires an IBIS lifecycle envelope.
3. Submits required self-attestations via `AttestationRegistry.recordCommitmentWithSignature` (EIP-712 meta-tx).
4. Opens matching `CommitmentTracker` entries.
5. Submits `requestAccessWithSignature` to the vault.
6. On success, mints the `AccessCredentialNFT`.

Response carries `obligations[]`, `attestations[]`, `pendingObligations[]`.

## API

29 endpoints, OpenAPI 3.0.3, every route carries a Swagger `summary`. Tags:

`auth` · `providers` · `requesters` · `cohorts` · `attestations` · `commitments` · `ontology` · `health`

Open `/docs` for the full surface.

## License

See `LICENSE.txt`.
