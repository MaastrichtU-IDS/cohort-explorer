// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./NullifierRegistry.sol";
import "./IdentityRegistry.sol";

contract ZKConsentVerifier is AccessControl, Pausable, ReentrancyGuard {

    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    uint8 public constant ACTION_CONSENT = 0;
    uint8 public constant ACTION_ACCESS = 1;
    uint8 public constant ACTION_REVOKE = 2;

    uint32 public constant MIN_SCORE_AUTO_APPROVE = 800;

    NullifierRegistry public nullifierRegistry;
    IdentityRegistry public identityRegistry;

    address public semaphoreVerifier;

    address public noirVerifier;

    struct SemaphoreProof {
        uint256 merkleTreeRoot;
        uint256 nullifierHash;
        uint256 signal;
        uint256[8] proof;
    }

    struct NoirProof {
        bytes32 consentCommitment;
        uint32 permissionRequested;
        uint32 scoreThreshold;
        bytes proof;
    }

    struct AccessGrant {
        bytes32 nullifierHash;
        bytes32 consentCommitment;
        uint256 grantedAt;
        uint256 epoch;
        bool valid;
    }

    mapping(bytes32 => AccessGrant) public accessGrants;

    uint256 public totalAccessGrants;

    uint256 public currentEpoch;

    event AccessGranted(
        bytes32 indexed nullifierHash,
        bytes32 indexed consentCommitment,
        bytes32 indexed cohortHash,
        uint256 epoch,
        uint256 timestamp
    );

    event AccessDenied(
        bytes32 indexed cohortHash,
        string reason,
        uint256 timestamp
    );

    event EpochUpdated(
        uint256 oldEpoch,
        uint256 newEpoch,
        uint256 timestamp
    );

    event VerifierUpdated(
        string verifierType,
        address oldAddress,
        address newAddress
    );

    error InvalidSemaphoreProof();
    error InvalidNoirProof();
    error InvalidMerkleRoot(bytes32 cohortHash, uint256 root);
    error NullifierAlreadyUsed(bytes32 nullifierHash);
    error ScoreBelowThreshold(uint32 score, uint32 threshold);
    error InvalidVerifierAddress();
    error InvalidCohort(bytes32 cohortHash);
    error ProofVerificationFailed(string proofType);

    constructor(
        address admin,
        address _nullifierRegistry,
        address _identityRegistry,
        address _semaphoreVerifier,
        address _noirVerifier
    ) {
        if (_semaphoreVerifier == address(0)) revert InvalidVerifierAddress();
        if (_noirVerifier == address(0)) revert InvalidVerifierAddress();

        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
        _grantRole(OPERATOR_ROLE, admin);

        nullifierRegistry = NullifierRegistry(_nullifierRegistry);
        identityRegistry = IdentityRegistry(_identityRegistry);
        semaphoreVerifier = _semaphoreVerifier;
        noirVerifier = _noirVerifier;

        currentEpoch = block.timestamp / 1 days;
    }

    function requestAccess(
        bytes32 cohortHash,
        SemaphoreProof calldata semaphoreProof,
        NoirProof calldata noirProof
    ) external nonReentrant whenNotPaused returns (bool success) {

        (,,, bool active,) = identityRegistry.getGroup(cohortHash);
        if (!active) revert InvalidCohort(cohortHash);

        if (!identityRegistry.isValidRoot(cohortHash, semaphoreProof.merkleTreeRoot)) {
            revert InvalidMerkleRoot(cohortHash, semaphoreProof.merkleTreeRoot);
        }

        bytes32 nullifierHash = bytes32(semaphoreProof.nullifierHash);
        if (nullifierRegistry.isNullifierUsed(nullifierHash)) {
            revert NullifierAlreadyUsed(nullifierHash);
        }

        if (!_verifySemaphoreProof(semaphoreProof)) {
            revert ProofVerificationFailed("Semaphore");
        }

        if (!_verifyNoirProof(noirProof)) {
            revert ProofVerificationFailed("Noir");
        }

        nullifierRegistry.checkAndUseNullifier(
            nullifierHash,
            cohortHash,
            ACTION_ACCESS,
            currentEpoch
        );

        accessGrants[nullifierHash] = AccessGrant({
            nullifierHash: nullifierHash,
            consentCommitment: noirProof.consentCommitment,
            grantedAt: block.timestamp,
            epoch: currentEpoch,
            valid: true
        });
        totalAccessGrants++;

        emit AccessGranted(
            nullifierHash,
            noirProof.consentCommitment,
            cohortHash,
            currentEpoch,
            block.timestamp
        );

        return true;
    }

    function verifyMembership(
        bytes32 cohortHash,
        SemaphoreProof calldata semaphoreProof
    ) external view returns (bool success) {

        (,,, bool active,) = identityRegistry.getGroup(cohortHash);
        if (!active) return false;

        if (!identityRegistry.isValidRoot(cohortHash, semaphoreProof.merkleTreeRoot)) {
            return false;
        }

        return _verifySemaphoreProof(semaphoreProof);
    }

    function simulateCompliance(
        NoirProof calldata noirProof
    ) external view returns (bool valid) {
        return _verifyNoirProof(noirProof);
    }

    function _verifySemaphoreProof(
        SemaphoreProof calldata proof
    ) internal view returns (bool) {
        if (semaphoreVerifier == address(0)) {
            revert InvalidVerifierAddress();
        }

        (bool success, bytes memory result) = semaphoreVerifier.staticcall(
            abi.encodeWithSignature(
                "verifyProof(uint256[8],uint256[3])",
                proof.proof,
                [proof.merkleTreeRoot, proof.nullifierHash, proof.signal]
            )
        );

        if (!success) return false;
        return abi.decode(result, (bool));
    }

    function _verifyNoirProof(
        NoirProof calldata proof
    ) internal view returns (bool) {
        if (noirVerifier == address(0)) {
            revert InvalidVerifierAddress();
        }

        bytes32[] memory publicInputs = new bytes32[](3);
        publicInputs[0] = proof.consentCommitment;
        publicInputs[1] = bytes32(uint256(proof.permissionRequested));
        publicInputs[2] = bytes32(uint256(proof.scoreThreshold));

        (bool success, bytes memory result) = noirVerifier.staticcall(
            abi.encodeWithSignature(
                "verify(bytes,bytes32[])",
                proof.proof,
                publicInputs
            )
        );

        if (!success) return false;
        return abi.decode(result, (bool));
    }

    function hasAccess(bytes32 nullifierHash) external view returns (bool) {
        return accessGrants[nullifierHash].valid;
    }

    function getAccessGrant(bytes32 nullifierHash) external view returns (
        bytes32 consentCommitment,
        uint256 grantedAt,
        uint256 epoch,
        bool valid
    ) {
        AccessGrant storage grant = accessGrants[nullifierHash];
        return (grant.consentCommitment, grant.grantedAt, grant.epoch, grant.valid);
    }

    function updateEpoch() external onlyRole(OPERATOR_ROLE) {
        uint256 oldEpoch = currentEpoch;
        currentEpoch = block.timestamp / 1 days;

        emit EpochUpdated(oldEpoch, currentEpoch, block.timestamp);
    }

    function setSemaphoreVerifier(address _verifier) external onlyRole(ADMIN_ROLE) {
        if (_verifier == address(0)) revert InvalidVerifierAddress();
        address old = semaphoreVerifier;
        semaphoreVerifier = _verifier;
        emit VerifierUpdated("Semaphore", old, _verifier);
    }

    function setNoirVerifier(address _verifier) external onlyRole(ADMIN_ROLE) {
        if (_verifier == address(0)) revert InvalidVerifierAddress();
        address old = noirVerifier;
        noirVerifier = _verifier;
        emit VerifierUpdated("Noir", old, _verifier);
    }

    function setRegistries(
        address _nullifierRegistry,
        address _identityRegistry
    ) external onlyRole(ADMIN_ROLE) {
        nullifierRegistry = NullifierRegistry(_nullifierRegistry);
        identityRegistry = IdentityRegistry(_identityRegistry);
    }

    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
}
