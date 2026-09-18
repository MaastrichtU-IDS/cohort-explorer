#!/bin/sh
set -e

export RPC_URL=${RPC_URL:-http://hardhat:8545}
DEPLOYMENTS=/app/deployments/deployments.json

rpc() {
    curl -sf "$RPC_URL" -X POST -H "Content-Type: application/json" -d "$1"
}

echo "Waiting for Hardhat node at $RPC_URL..."
until rpc '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' > /dev/null 2>&1; do
    echo "Waiting..."
    sleep 2
done

echo "Hardhat node is ready"

# Deploy only onto a node that does not already hold this deployment. docker compose re-runs one-shot
# services on every `up`; redeploying onto a live node yields a second set of contracts at new
# addresses, orphans everything cached against the first set, and invalidates the reset snapshot.
ALREADY_DEPLOYED=0
if [ -f "$DEPLOYMENTS" ]; then
    VAULT=$(jq -r '.contracts.duoConsentVaultV2 // empty' "$DEPLOYMENTS")
    if [ -n "$VAULT" ]; then
        CODE=$(rpc "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getCode\",\"params\":[\"$VAULT\",\"latest\"],\"id\":1}" | jq -r '.result // "0x"')
        if [ "$CODE" != "0x" ] && [ -n "$CODE" ]; then
            ALREADY_DEPLOYED=1
        fi
    fi
fi

if [ "$ALREADY_DEPLOYED" = "1" ]; then
    echo "Contracts from $DEPLOYMENTS are already on this node (vault $VAULT); skipping deployment."
else
    echo "Deploying contracts..."
    npx hardhat run scripts/deploy.ts --network localhost
fi

echo "Generating ABIs..."
npx hardhat run scripts/generate-abis.ts

echo "Deployment complete!"
echo "Deployments:"
cat "$DEPLOYMENTS"

echo "ABIs generated in /app/api/contracts/abis/"
ls -la /app/api/contracts/abis/
