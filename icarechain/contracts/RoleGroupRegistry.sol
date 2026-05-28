// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

interface IUserIdentityRegistry {
    function isRegistered(address eoa) external view returns (bool);
}

contract RoleGroupRegistry is AccessControl {
    bytes32 public constant ROLE_OPERATOR = keccak256("ROLE_OPERATOR");

    bytes32 public constant ROLE_PROVIDER = keccak256("ROLE_PROVIDER");
    bytes32 public constant ROLE_REQUESTER = keccak256("ROLE_REQUESTER");

    mapping(bytes32 => mapping(bytes32 => bool)) public isMember;
    mapping(bytes32 => bytes32[]) internal _members;
    mapping(bytes32 => bytes32) public groupRoot;

    mapping(bytes32 => mapping(address => bytes32)) public commitmentOf;
    mapping(bytes32 => mapping(bytes32 => address)) public ownerOf;

    IUserIdentityRegistry public userIdentityRegistry;

    event RoleAttached(bytes32 indexed roleId, address indexed ownerEOA, bytes32 commitment, uint256 index);
    event RoleRootUpdated(bytes32 indexed roleId, bytes32 root);
    event UserIdentityRegistrySet(address registry);

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ROLE_OPERATOR, admin);
    }

    function setUserIdentityRegistry(address registry) external onlyRole(DEFAULT_ADMIN_ROLE) {
        userIdentityRegistry = IUserIdentityRegistry(registry);
        emit UserIdentityRegistrySet(registry);
    }

    function attachRole(
        bytes32 roleId,
        address ownerEOA,
        bytes32 commitment
    ) external onlyRole(ROLE_OPERATOR) {
        require(roleId != bytes32(0), "roleId=0");
        require(ownerEOA != address(0), "owner=0");
        require(commitment != bytes32(0), "commitment=0");
        require(commitmentOf[roleId][ownerEOA] == bytes32(0), "EOA already attached");
        require(ownerOf[roleId][commitment] == address(0), "commitment claimed");

        require(
            address(userIdentityRegistry) != address(0)
                && userIdentityRegistry.isRegistered(ownerEOA),
            "identity not registered"
        );

        commitmentOf[roleId][ownerEOA] = commitment;
        ownerOf[roleId][commitment] = ownerEOA;
        isMember[roleId][commitment] = true;
        _members[roleId].push(commitment);

        bytes32 newRoot = keccak256(abi.encodePacked(groupRoot[roleId], commitment));
        groupRoot[roleId] = newRoot;

        emit RoleAttached(roleId, ownerEOA, commitment, _members[roleId].length - 1);
        emit RoleRootUpdated(roleId, newRoot);
    }

    function isAttached(bytes32 roleId, address ownerEOA) external view returns (bool) {
        return commitmentOf[roleId][ownerEOA] != bytes32(0);
    }

    function memberCount(bytes32 roleId) external view returns (uint256) {
        return _members[roleId].length;
    }

    function memberAt(bytes32 roleId, uint256 idx) external view returns (bytes32) {
        return _members[roleId][idx];
    }
}
