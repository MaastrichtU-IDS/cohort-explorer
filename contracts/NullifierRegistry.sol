// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

contract NullifierRegistry is AccessControl, Pausable {

    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    mapping(bytes32 => bool) public usedNullifiers;

    mapping(bytes32 => uint256) public nullifierBlock;

    mapping(bytes32 => bytes32) public nullifierCohort;

    uint256 public totalNullifiersUsed;

    event NullifierUsed(
        bytes32 indexed nullifierHash,
        bytes32 indexed cohortHash,
        uint8 actionType,
        uint256 epoch,
        uint256 timestamp
    );

    event NullifierInvalidated(
        bytes32 indexed nullifierHash,
        address indexed admin,
        string reason
    );

    error NullifierAlreadyUsed(bytes32 nullifierHash);
    error NullifierNotUsed(bytes32 nullifierHash);
    error InvalidNullifier();

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
    }

    function isNullifierUsed(bytes32 nullifierHash) external view returns (bool) {
        return usedNullifiers[nullifierHash];
    }

    function getNullifierBlock(bytes32 nullifierHash) external view returns (uint256) {
        return nullifierBlock[nullifierHash];
    }

    function batchCheckNullifiers(bytes32[] calldata nullifierHashes) external view returns (bool[] memory) {
        bool[] memory results = new bool[](nullifierHashes.length);
        for (uint256 i = 0; i < nullifierHashes.length; i++) {
            results[i] = usedNullifiers[nullifierHashes[i]];
        }
        return results;
    }

    function useNullifier(
        bytes32 nullifierHash,
        bytes32 cohortHash,
        uint8 actionType,
        uint256 epoch
    ) external onlyRole(VERIFIER_ROLE) whenNotPaused {
        if (nullifierHash == bytes32(0)) revert InvalidNullifier();
        if (usedNullifiers[nullifierHash]) revert NullifierAlreadyUsed(nullifierHash);

        usedNullifiers[nullifierHash] = true;
        nullifierBlock[nullifierHash] = block.number;
        nullifierCohort[nullifierHash] = cohortHash;
        totalNullifiersUsed++;

        emit NullifierUsed(nullifierHash, cohortHash, actionType, epoch, block.timestamp);
    }

    function checkAndUseNullifier(
        bytes32 nullifierHash,
        bytes32 cohortHash,
        uint8 actionType,
        uint256 epoch
    ) external onlyRole(VERIFIER_ROLE) whenNotPaused {
        if (nullifierHash == bytes32(0)) revert InvalidNullifier();
        if (usedNullifiers[nullifierHash]) revert NullifierAlreadyUsed(nullifierHash);

        usedNullifiers[nullifierHash] = true;
        nullifierBlock[nullifierHash] = block.number;
        nullifierCohort[nullifierHash] = cohortHash;
        totalNullifiersUsed++;

        emit NullifierUsed(nullifierHash, cohortHash, actionType, epoch, block.timestamp);
    }

    function invalidateNullifier(
        bytes32 nullifierHash,
        string calldata reason
    ) external onlyRole(ADMIN_ROLE) {
        if (!usedNullifiers[nullifierHash]) revert NullifierNotUsed(nullifierHash);

        usedNullifiers[nullifierHash] = false;
        nullifierBlock[nullifierHash] = 0;

        emit NullifierInvalidated(nullifierHash, msg.sender, reason);
    }

    function addVerifier(address verifier) external onlyRole(ADMIN_ROLE) {
        _grantRole(VERIFIER_ROLE, verifier);
    }

    function removeVerifier(address verifier) external onlyRole(ADMIN_ROLE) {
        _revokeRole(VERIFIER_ROLE, verifier);
    }

    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
}
