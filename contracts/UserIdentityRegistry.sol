// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract UserIdentityRegistry is AccessControl {
    bytes32 public constant REGISTRAR_ROLE = keccak256("REGISTRAR_ROLE");

    mapping(address => bytes32) public identityOf;
    mapping(bytes32 => address) public eoaOf;

    address public ibisVerifier;

    event IdentityRegistered(address indexed eoa, bytes32 indexed commitment, uint256 timestamp);
    event IBISVerifierSet(address verifier);

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(REGISTRAR_ROLE, admin);
    }

    function setIBISVerifier(address verifier) external onlyRole(DEFAULT_ADMIN_ROLE) {
        ibisVerifier = verifier;
        emit IBISVerifierSet(verifier);
    }

    function register(address eoa, bytes32 commitment) external onlyRole(REGISTRAR_ROLE) {
        require(eoa != address(0), "eoa=0");
        require(commitment != bytes32(0), "commitment=0");
        require(identityOf[eoa] == bytes32(0), "EOA already registered");
        require(eoaOf[commitment] == address(0), "commitment claimed");

        if (ibisVerifier != address(0)) {
            (bool ok, bytes memory ret) = ibisVerifier.staticcall(
                abi.encodeWithSignature("isSlotInitialized(bytes32)", commitment)
            );
            require(ok && ret.length == 32 && abi.decode(ret, (bool)),
                "no IBIS state for commitment");
        }

        identityOf[eoa] = commitment;
        eoaOf[commitment] = eoa;
        emit IdentityRegistered(eoa, commitment, block.timestamp);
    }

    function isRegistered(address eoa) external view returns (bool) {
        return identityOf[eoa] != bytes32(0);
    }
}
