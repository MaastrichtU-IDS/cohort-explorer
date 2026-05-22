#!/usr/bin/env bash
# Build Semaphore Groth16 circuit artifacts (semaphore.wasm + semaphore.zkey)
#
# This script downloads the Semaphore v4 circuit, compiles it with circom,
# performs the trusted setup (powers of tau + circuit-specific), and outputs
# the artifacts needed by the Python proof backend (snarkjs groth16 fullprove).
#
# Prerequisites:
#   - Node.js >= 18
#   - circom (https://docs.circom.io/getting-started/installation/)
#   - snarkjs: npm install -g snarkjs
#
# Usage:
#   cd circuits/semaphore && bash build-circuits.sh
#
# Output:
#   semaphore.wasm  — WASM witness generator
#   semaphore.zkey  — Groth16 proving key (after phase 2 contribution)
#   verification_key.json — Verification key (for on-chain verifier generation)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PTAU_FILE="${BUILD_DIR}/pot16_final.ptau"
CIRCUIT_DIR="${BUILD_DIR}/circuit"

echo "=== IBIS Semaphore Circuit Build ==="
echo ""

# Check prerequisites
for cmd in node circom snarkjs; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found. Please install it first."
        case "$cmd" in
            circom)
                echo "  Install: https://docs.circom.io/getting-started/installation/"
                ;;
            snarkjs)
                echo "  Install: npm install -g snarkjs"
                ;;
        esac
        exit 1
    fi
done

echo "Prerequisites OK: node=$(node --version), circom=$(circom --version 2>&1 | head -1), snarkjs=$(snarkjs --version 2>&1 | head -1)"
echo ""

mkdir -p "$BUILD_DIR" "$CIRCUIT_DIR"

# Step 1: Get Semaphore v4 circuit
SEMAPHORE_CIRCOM="${CIRCUIT_DIR}/semaphore.circom"
if [ ! -f "$SEMAPHORE_CIRCOM" ]; then
    echo "--- Step 1: Downloading Semaphore v4 circuit ---"
    # Install Semaphore circom circuits package
    cd "$CIRCUIT_DIR"
    npm init -y --silent 2>/dev/null
    npm install --save @semaphore-protocol/circuits@^4.0.0 circomlib 2>/dev/null
    # Create wrapper circuit that sets tree depth = 20 (matching IBIS GroupManager)
    cat > "$SEMAPHORE_CIRCOM" <<'CIRCOM'
pragma circom 2.1.0;

include "node_modules/@semaphore-protocol/circuits/semaphore.circom";

// Instantiate Semaphore with tree depth 20 (supports up to ~1M members)
// This matches IBIS GroupManager.DEFAULT_DEPTH and chain_position.TREE_DEPTH.
component main {public [merkleTreeDepth, merkleTreeRoot, externalNullifier, signal]} = Semaphore(20);
CIRCOM
    cd "$SCRIPT_DIR"
else
    echo "--- Step 1: Semaphore circuit already exists, skipping download ---"
fi

# Step 2: Compile circuit
WASM_DIR="${BUILD_DIR}/semaphore_js"
R1CS_FILE="${BUILD_DIR}/semaphore.r1cs"
if [ ! -f "$R1CS_FILE" ]; then
    echo "--- Step 2: Compiling circuit with circom ---"
    circom "$SEMAPHORE_CIRCOM" \
        --r1cs --wasm --sym \
        -o "$BUILD_DIR" \
        -l "${CIRCUIT_DIR}/node_modules"
    echo "  R1CS constraints: $(snarkjs r1cs info "$R1CS_FILE" 2>&1 | grep -o '[0-9]* constraints' || echo 'see above')"
else
    echo "--- Step 2: Circuit already compiled, skipping ---"
fi

# Step 3: Powers of Tau ceremony (phase 1)
if [ ! -f "$PTAU_FILE" ]; then
    echo "--- Step 3: Powers of Tau phase 1 (this may take a few minutes) ---"
    # Use 2^16 = 65536 constraints capacity (sufficient for Semaphore depth 20)
    snarkjs powersoftau new bn128 16 "${BUILD_DIR}/pot16_0000.ptau" -v
    # Contribute (deterministic for reproducibility — use random entropy in production)
    snarkjs powersoftau contribute "${BUILD_DIR}/pot16_0000.ptau" "${BUILD_DIR}/pot16_0001.ptau" \
        --name="IBIS build contribution" -v -e="ibis-semaphore-build-entropy"
    # Prepare phase 2
    snarkjs powersoftau prepare phase2 "${BUILD_DIR}/pot16_0001.ptau" "$PTAU_FILE" -v
    # Clean up intermediate files
    rm -f "${BUILD_DIR}/pot16_0000.ptau" "${BUILD_DIR}/pot16_0001.ptau"
else
    echo "--- Step 3: Powers of Tau already exists, skipping ---"
fi

# Step 4: Circuit-specific setup (phase 2)
ZKEY_FILE="${BUILD_DIR}/semaphore.zkey"
if [ ! -f "$ZKEY_FILE" ]; then
    echo "--- Step 4: Groth16 circuit-specific setup (phase 2) ---"
    snarkjs groth16 setup "$R1CS_FILE" "$PTAU_FILE" "${BUILD_DIR}/semaphore_0000.zkey"
    # Contribute to phase 2
    snarkjs zkey contribute "${BUILD_DIR}/semaphore_0000.zkey" "$ZKEY_FILE" \
        --name="IBIS phase2 contribution" -v -e="ibis-phase2-entropy"
    rm -f "${BUILD_DIR}/semaphore_0000.zkey"
else
    echo "--- Step 4: Proving key already exists, skipping ---"
fi

# Step 5: Export verification key
VKEY_FILE="${BUILD_DIR}/verification_key.json"
if [ ! -f "$VKEY_FILE" ]; then
    echo "--- Step 5: Exporting verification key ---"
    snarkjs zkey export verificationkey "$ZKEY_FILE" "$VKEY_FILE"
fi

# Step 6: Copy artifacts to the expected locations
echo "--- Step 6: Installing artifacts ---"
cp "${WASM_DIR}/semaphore.wasm" "${SCRIPT_DIR}/semaphore.wasm"
cp "$ZKEY_FILE" "${SCRIPT_DIR}/semaphore.zkey"
cp "$VKEY_FILE" "${SCRIPT_DIR}/verification_key.json"

echo ""
echo "=== Build complete ==="
echo "  semaphore.wasm:         ${SCRIPT_DIR}/semaphore.wasm"
echo "  semaphore.zkey:         ${SCRIPT_DIR}/semaphore.zkey"
echo "  verification_key.json:  ${SCRIPT_DIR}/verification_key.json"
echo ""
echo "To generate the Solidity verifier for on-chain deployment:"
echo "  snarkjs zkey export solidityverifier ${SCRIPT_DIR}/semaphore.zkey ${SCRIPT_DIR}/SemaphoreVerifier.sol"
echo ""
echo "To run a test proof:"
echo "  npx snarkjs groth16 fullprove input.json ${SCRIPT_DIR}/semaphore.wasm ${SCRIPT_DIR}/semaphore.zkey"
