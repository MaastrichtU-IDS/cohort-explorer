// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "./RoleAccount.sol";

contract RoleAccountFactory {
    event AccountCreated(address indexed account, address indexed owner, bytes32 indexed roleCommitment);

    function deploy(address owner, bytes32 roleCommitment) external returns (address) {
        bytes32 salt = roleCommitment;
        bytes memory bytecode = abi.encodePacked(
            type(RoleAccount).creationCode,
            abi.encode(owner, roleCommitment)
        );
        address acct;
        assembly {
            acct := create2(0, add(bytecode, 0x20), mload(bytecode), salt)
        }
        require(acct != address(0), "create2 failed");
        emit AccountCreated(acct, owner, roleCommitment);
        return acct;
    }

    function computeAddress(address owner, bytes32 roleCommitment) external view returns (address) {
        bytes memory bytecode = abi.encodePacked(
            type(RoleAccount).creationCode,
            abi.encode(owner, roleCommitment)
        );
        bytes32 hash = keccak256(abi.encodePacked(
            bytes1(0xff), address(this), roleCommitment, keccak256(bytecode)
        ));
        return address(uint160(uint256(hash)));
    }
}
