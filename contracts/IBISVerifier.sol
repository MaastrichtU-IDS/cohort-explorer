// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IHonkVerifier {
    function verify(bytes calldata proof, bytes32[] calldata publicInputs) external view returns (bool);
}

contract IBISVerifier {

    uint256 public constant ENVELOPE_SIZE = 976;

    bytes32 private constant SENTINEL = keccak256("LI_AUTH_SENTINEL_V1");

    IHonkVerifier public chainPositionVerifier;

    event EnvelopeProcessed(
        bytes32 indexed commitment,
        bytes32 nullifier,
        bytes32 blindedEpoch
    );

    mapping(bytes32 => bytes32[16]) public encryptedStates;

    mapping(bytes32 => bytes32) public epochKeyHashes;

    mapping(bytes32 => bytes32) public blindedEpochs;

    mapping(bytes32 => bool) public usedNullifiers;

    address public admin;
    bool public paused;

    constructor() {
        admin = msg.sender;
    }

    function setChainPositionVerifier(address verifier) external onlyAdmin {
        chainPositionVerifier = IHonkVerifier(verifier);
    }

    modifier whenNotPaused() {
        require(!paused, "IBIS: paused");
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "IBIS: not admin");
        _;
    }

    function processEnvelopeZK(
        bytes32 commitment,
        bytes32 nullifier,
        bytes32 blindedEpoch,
        bytes32 epochKeyHash,
        bytes calldata encryptedPayload,
        bytes calldata proof,
        bytes32[] calldata publicInputs
    ) external whenNotPaused {
        require(address(chainPositionVerifier) != address(0), "IBIS: verifier not set");
        require(!usedNullifiers[nullifier], "IBIS: nullifier reuse");
        require(encryptedPayload.length == 512, "IBIS: payload must be 512 bytes");
        require(publicInputs.length >= 2, "IBIS: missing public inputs");
        require(publicInputs[0] == commitment, "IBIS: pub[0] != commitment");
        require(publicInputs[1] == epochKeyHash, "IBIS: pub[1] != epochKeyHash");
        require(chainPositionVerifier.verify(proof, publicInputs), "IBIS: proof rejected");

        usedNullifiers[nullifier] = true;
        bytes32[16] memory payload;
        for (uint256 i = 0; i < 16; i++) {
            payload[i] = bytes32(encryptedPayload[i * 32:(i + 1) * 32]);
        }
        _writeState(commitment, epochKeyHash, blindedEpoch, payload);
        emit EnvelopeProcessed(commitment, nullifier, blindedEpoch);
    }

    function processEnvelope(
        bytes32 commitment,
        bytes32 nullifier,
        bytes32 blindedEpoch,
        bytes32 epochKeyHash,
        bytes calldata encryptedPayload,
        bytes calldata proof
    ) external whenNotPaused {

        require(!usedNullifiers[nullifier], "IBIS: nullifier reuse");
        usedNullifiers[nullifier] = true;

        require(proof.length == 256, "IBIS: bad proof size");

        require(encryptedPayload.length == 512, "IBIS: payload must be 512 bytes");

        bytes32[16] memory payload;
        for (uint256 i = 0; i < 16; i++) {
            payload[i] = bytes32(encryptedPayload[i * 32:(i + 1) * 32]);
        }
        _writeState(commitment, epochKeyHash, blindedEpoch, payload);

        emit EnvelopeProcessed(commitment, nullifier, blindedEpoch);
    }

    function _writeState(
        bytes32 commitment,
        bytes32 _epochKeyHash,
        bytes32 _blindedEpoch,
        bytes32[16] memory payload
    ) internal {
        encryptedStates[commitment] = payload;
        epochKeyHashes[commitment] = _epochKeyHash;
        blindedEpochs[commitment] = _blindedEpoch;
    }

    function initializeSlot(bytes32 commitment) external whenNotPaused {
        if (epochKeyHashes[commitment] == bytes32(0)) {

            bytes32[16] memory sentinelPayload;
            for (uint256 i = 0; i < 16; i++) {
                sentinelPayload[i] = SENTINEL;
            }
            _writeState(commitment, SENTINEL, SENTINEL, sentinelPayload);
        }
    }

    function batchInitializeSlots(bytes32[] calldata commitments) external whenNotPaused {
        bytes32[16] memory sentinelPayload;
        for (uint256 j = 0; j < 16; j++) {
            sentinelPayload[j] = SENTINEL;
        }
        for (uint256 i = 0; i < commitments.length; i++) {
            if (epochKeyHashes[commitments[i]] == bytes32(0)) {
                _writeState(commitments[i], SENTINEL, SENTINEL, sentinelPayload);
            }
        }
    }

    function isNullifierUsed(bytes32 nullifier) external view returns (bool) {
        return usedNullifiers[nullifier];
    }

    function isSlotInitialized(bytes32 commitment) external view returns (bool) {
        return epochKeyHashes[commitment] != bytes32(0);
    }

    function getState(bytes32 commitment)
        external
        view
        returns (
            bytes32[16] memory encryptedState,
            bytes32 epochKeyHash,
            bytes32 blindedEpoch
        )
    {
        return (
            encryptedStates[commitment],
            epochKeyHashes[commitment],
            blindedEpochs[commitment]
        );
    }

    function pause() external onlyAdmin {
        paused = true;
    }

    function unpause() external onlyAdmin {
        paused = false;
    }

    function transferAdmin(address newAdmin) external onlyAdmin {
        require(newAdmin != address(0), "IBIS: zero address");
        admin = newAdmin;
    }
}
