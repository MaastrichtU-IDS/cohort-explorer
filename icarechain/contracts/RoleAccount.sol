// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

contract RoleAccount {
    using ECDSA for bytes32;

    address public immutable owner;
    bytes32 public immutable roleCommitment;

    bytes4 internal constant MAGICVALUE = 0x1626ba7e;
    bytes4 internal constant INVALID_SIG = 0xffffffff;

    event Executed(address indexed target, uint256 value, bytes data);

    constructor(address _owner, bytes32 _roleCommitment) {
        require(_owner != address(0), "owner=0");
        require(_roleCommitment != bytes32(0), "commitment=0");
        owner = _owner;
        roleCommitment = _roleCommitment;
    }

    receive() external payable {}

    function execute(address target, uint256 value, bytes calldata data) external returns (bytes memory) {
        require(msg.sender == owner, "not owner");
        (bool ok, bytes memory ret) = target.call{value: value}(data);
        require(ok, "exec failed");
        emit Executed(target, value, data);
        return ret;
    }

    function isValidSignature(bytes32 hash, bytes calldata signature)
        external
        view
        returns (bytes4)
    {
        if (signature.length != 65) return INVALID_SIG;
        address signer = ECDSA.recover(hash, signature);
        if (signer == address(0)) return INVALID_SIG;
        if (signer != owner) return INVALID_SIG;
        return MAGICVALUE;
    }
}
