// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

contract IdentityRegistry is AccessControl, Pausable {

    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    uint8 public constant MAX_DEPTH = 20;

    uint8 public constant DEFAULT_DEPTH = 16;

    struct Group {

        uint8 depth;

        uint256 merkleRoot;

        uint256 memberCount;

        bool active;

        uint256 createdAt;

        address creator;
    }

    mapping(bytes32 => Group) public groups;

    mapping(bytes32 => mapping(uint256 => bool)) public commitmentExists;

    mapping(bytes32 => mapping(uint256 => bool)) public historicalRoots;

    bytes32[] public allCohorts;

    event GroupCreated(
        bytes32 indexed cohortHash,
        uint8 depth,
        address indexed creator,
        uint256 timestamp
    );

    event MemberAdded(
        bytes32 indexed cohortHash,
        uint256 indexed commitment,
        uint256 newMerkleRoot,
        uint256 memberIndex
    );

    event MemberRemoved(
        bytes32 indexed cohortHash,
        uint256 indexed commitment,
        uint256 newMerkleRoot
    );

    event GroupDeactivated(
        bytes32 indexed cohortHash,
        address indexed admin,
        uint256 timestamp
    );

    event MerkleRootUpdated(
        bytes32 indexed cohortHash,
        uint256 oldRoot,
        uint256 newRoot
    );

    error GroupNotFound(bytes32 cohortHash);
    error GroupAlreadyExists(bytes32 cohortHash);
    error GroupNotActive(bytes32 cohortHash);
    error CommitmentAlreadyExists(bytes32 cohortHash, uint256 commitment);
    error CommitmentNotFound(bytes32 cohortHash, uint256 commitment);
    error InvalidCommitment();
    error InvalidMerkleRoot();
    error InvalidDepth(uint8 depth);
    error GroupFull(bytes32 cohortHash);

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
        _grantRole(OPERATOR_ROLE, admin);
    }

    function getGroup(bytes32 cohortHash) external view returns (
        uint8 depth,
        uint256 merkleRoot,
        uint256 memberCount,
        bool active,
        uint256 createdAt
    ) {
        Group storage g = groups[cohortHash];
        return (g.depth, g.merkleRoot, g.memberCount, g.active, g.createdAt);
    }

    function groupExists(bytes32 cohortHash) external view returns (bool) {
        return groups[cohortHash].createdAt > 0;
    }

    function isMember(bytes32 cohortHash, uint256 commitment) external view returns (bool) {
        return commitmentExists[cohortHash][commitment];
    }

    function getMerkleRoot(bytes32 cohortHash) external view returns (uint256) {
        return groups[cohortHash].merkleRoot;
    }

    function isValidRoot(bytes32 cohortHash, uint256 root) external view returns (bool) {
        return groups[cohortHash].merkleRoot == root || historicalRoots[cohortHash][root];
    }

    function getCohortCount() external view returns (uint256) {
        return allCohorts.length;
    }

    function createGroup(
        bytes32 cohortHash,
        uint8 depth
    ) external onlyRole(OPERATOR_ROLE) whenNotPaused {
        if (groups[cohortHash].createdAt > 0) revert GroupAlreadyExists(cohortHash);

        uint8 actualDepth = depth == 0 ? DEFAULT_DEPTH : depth;
        if (actualDepth > MAX_DEPTH) revert InvalidDepth(actualDepth);

        groups[cohortHash] = Group({
            depth: actualDepth,
            merkleRoot: 0,
            memberCount: 0,
            active: true,
            createdAt: block.timestamp,
            creator: msg.sender
        });

        allCohorts.push(cohortHash);

        emit GroupCreated(cohortHash, actualDepth, msg.sender, block.timestamp);
    }

    function addMember(
        bytes32 cohortHash,
        uint256 commitment,
        uint256 newMerkleRoot
    ) external onlyRole(OPERATOR_ROLE) whenNotPaused {
        Group storage g = groups[cohortHash];

        if (g.createdAt == 0) revert GroupNotFound(cohortHash);
        if (!g.active) revert GroupNotActive(cohortHash);
        if (commitment == 0) revert InvalidCommitment();
        if (commitmentExists[cohortHash][commitment]) revert CommitmentAlreadyExists(cohortHash, commitment);

        uint256 maxMembers = 2 ** g.depth;
        if (g.memberCount >= maxMembers) revert GroupFull(cohortHash);

        commitmentExists[cohortHash][commitment] = true;

        uint256 oldRoot = g.merkleRoot;
        historicalRoots[cohortHash][oldRoot] = true;
        g.merkleRoot = newMerkleRoot;
        g.memberCount++;

        emit MemberAdded(cohortHash, commitment, newMerkleRoot, g.memberCount - 1);
        emit MerkleRootUpdated(cohortHash, oldRoot, newMerkleRoot);
    }

    function batchAddMembers(
        bytes32 cohortHash,
        uint256[] calldata commitments,
        uint256 finalMerkleRoot
    ) external onlyRole(OPERATOR_ROLE) whenNotPaused {
        Group storage g = groups[cohortHash];

        if (g.createdAt == 0) revert GroupNotFound(cohortHash);
        if (!g.active) revert GroupNotActive(cohortHash);

        uint256 maxMembers = 2 ** g.depth;
        if (g.memberCount + commitments.length > maxMembers) revert GroupFull(cohortHash);

        uint256 oldRoot = g.merkleRoot;

        for (uint256 i = 0; i < commitments.length; i++) {
            uint256 commitment = commitments[i];
            if (commitment == 0) revert InvalidCommitment();
            if (commitmentExists[cohortHash][commitment]) revert CommitmentAlreadyExists(cohortHash, commitment);

            commitmentExists[cohortHash][commitment] = true;
            g.memberCount++;

            emit MemberAdded(cohortHash, commitment, 0, g.memberCount - 1);
        }

        historicalRoots[cohortHash][oldRoot] = true;
        g.merkleRoot = finalMerkleRoot;

        emit MerkleRootUpdated(cohortHash, oldRoot, finalMerkleRoot);
    }

    function updateMerkleRoot(
        bytes32 cohortHash,
        uint256 newMerkleRoot
    ) external onlyRole(OPERATOR_ROLE) {
        Group storage g = groups[cohortHash];
        if (g.createdAt == 0) revert GroupNotFound(cohortHash);
        if (newMerkleRoot == 0) revert InvalidMerkleRoot();

        uint256 oldRoot = g.merkleRoot;
        historicalRoots[cohortHash][oldRoot] = true;
        g.merkleRoot = newMerkleRoot;

        emit MerkleRootUpdated(cohortHash, oldRoot, newMerkleRoot);
    }

    function deactivateGroup(bytes32 cohortHash) external onlyRole(ADMIN_ROLE) {
        Group storage g = groups[cohortHash];
        if (g.createdAt == 0) revert GroupNotFound(cohortHash);

        g.active = false;

        emit GroupDeactivated(cohortHash, msg.sender, block.timestamp);
    }

    function reactivateGroup(bytes32 cohortHash) external onlyRole(ADMIN_ROLE) {
        Group storage g = groups[cohortHash];
        if (g.createdAt == 0) revert GroupNotFound(cohortHash);

        g.active = true;
    }

    function addOperator(address operator) external onlyRole(ADMIN_ROLE) {
        _grantRole(OPERATOR_ROLE, operator);
    }

    function removeOperator(address operator) external onlyRole(ADMIN_ROLE) {
        _revokeRole(OPERATOR_ROLE, operator);
    }

    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
}
