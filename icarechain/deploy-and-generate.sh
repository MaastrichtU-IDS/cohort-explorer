#!/bin/sh
set -e

export RPC_URL=${RPC_URL:-http://hardhat:8545}

echo "Waiting for Hardhat node at $RPC_URL..."
until curl -sf $RPC_URL -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' > /dev/null 2>&1; do
    echo "Waiting..."
    sleep 2
done

echo "Hardhat node is ready"

echo "Deploying contracts..."
npx hardhat run scripts/deploy.ts --network localhost

echo "Generating ABIs..."
npx hardhat run scripts/generate-abis.ts

echo "Deployment complete!"
echo "Deployments:"
cat /app/deployments/deployments.json

echo "ABIs generated in /app/api/contracts/abis/"
ls -la /app/api/contracts/abis/
