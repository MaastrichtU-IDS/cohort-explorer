// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract DUOOntology is AccessControl {
    bytes32 public constant ONTOLOGY_ADMIN = keccak256("ONTOLOGY_ADMIN");

    bytes4 public constant NRES = 0x00000004;
    bytes4 public constant GRU  = 0x00000042;
    bytes4 public constant HMB  = 0x00000006;
    bytes4 public constant DS   = 0x00000007;
    bytes4 public constant POA  = 0x00000011;

    uint32 public constant MOD_NONE   = 0;
    uint32 public constant MOD_NPU    = 1 << 0;
    uint32 public constant MOD_NCU    = 1 << 1;
    uint32 public constant MOD_NPUNCU = 1 << 2;
    uint32 public constant MOD_PUB    = 1 << 3;
    uint32 public constant MOD_COL    = 1 << 4;
    uint32 public constant MOD_IRB    = 1 << 5;
    uint32 public constant MOD_GS     = 1 << 6;
    uint32 public constant MOD_MOR    = 1 << 7;
    uint32 public constant MOD_TS     = 1 << 8;
    uint32 public constant MOD_US     = 1 << 9;
    uint32 public constant MOD_PS     = 1 << 10;
    uint32 public constant MOD_IS     = 1 << 11;
    uint32 public constant MOD_RTN    = 1 << 12;
    uint32 public constant MOD_CC     = 1 << 13;
    uint32 public constant MOD_NPOA   = 1 << 14;
    uint32 public constant MOD_GSO    = 1 << 15;
    uint32 public constant MOD_RS     = 1 << 16;
    uint32 public constant MOD_NMDS   = 1 << 17;

    uint8 public constant REQ_ACADEMIC   = 1;
    uint8 public constant REQ_NONPROFIT  = 2;
    uint8 public constant REQ_COMMERCIAL = 3;
    uint8 public constant REQ_CLINICAL   = 4;
    uint8 public constant REQ_GOVERNMENT = 5;

    uint8 public constant PURPOSE_GENERAL     = 1;
    uint8 public constant PURPOSE_HEALTH      = 2;
    uint8 public constant PURPOSE_DISEASE     = 3;
    uint8 public constant PURPOSE_POPULATION  = 4;
    uint8 public constant PURPOSE_METHODS     = 5;
    uint8 public constant PURPOSE_GENETICS    = 6;
    uint8 public constant PURPOSE_CLINICAL    = 7;

    mapping(bytes4 => bytes4) public permissionParent;

    mapping(bytes32 => bytes32) public diseaseParent;

    mapping(bytes4 => bool) public validPermission;

    event PermissionHierarchySet(bytes4 indexed child, bytes4 indexed parent);
    event DiseaseHierarchySet(bytes32 indexed child, bytes32 indexed parent);

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ONTOLOGY_ADMIN, msg.sender);

        _initializePermissionHierarchy();
    }

    function _initializePermissionHierarchy() internal {

        validPermission[NRES] = true;
        validPermission[GRU] = true;
        validPermission[HMB] = true;
        validPermission[DS] = true;
        validPermission[POA] = true;

        permissionParent[DS] = HMB;
        permissionParent[HMB] = GRU;
        permissionParent[GRU] = NRES;

    }

    function isPermissionCompatible(
        bytes4 consentedPermission,
        bytes4 requestedUse
    ) public view returns (bool compatible) {

        if (consentedPermission == requestedUse) return true;

        if (consentedPermission == NRES) return true;

        if (consentedPermission == POA || requestedUse == POA) {
            return consentedPermission == requestedUse;
        }

        bytes4 current = requestedUse;
        uint8 maxDepth = 10;

        for (uint8 i = 0; i < maxDepth; i++) {
            current = permissionParent[current];
            if (current == bytes4(0)) break;
            if (current == consentedPermission) return true;
        }

        return false;
    }

    function isDiseaseCompatible(
        bytes32 consentedDisease,
        bytes32 requestedDisease
    ) public view returns (bool) {

        if (consentedDisease == bytes32(0)) return true;

        if (consentedDisease == requestedDisease) return true;

        bytes32 current = requestedDisease;
        uint8 maxDepth = 15;

        for (uint8 i = 0; i < maxDepth; i++) {
            current = diseaseParent[current];
            if (current == bytes32(0)) break;
            if (current == consentedDisease) return true;
        }

        return false;
    }

    function checkRequesterTypeConstraints(
        uint16 modifiers,
        uint8 requesterType
    ) public pure returns (bool allowed, string memory reason) {

        if ((modifiers & MOD_NPUNCU) != 0) {
            if (requesterType == REQ_COMMERCIAL) {
                return (false, "NPUNCU: Commercial organizations not allowed");
            }
        }

        if ((modifiers & MOD_NPU) != 0) {
            if (requesterType != REQ_ACADEMIC && requesterType != REQ_NONPROFIT) {
                return (false, "NPU: Only non-profit organizations allowed");
            }
        }

        if ((modifiers & MOD_NCU) != 0) {
            if (requesterType == REQ_COMMERCIAL) {
                return (false, "NCU: Commercial use not allowed");
            }
        }

        return (true, "");
    }

    function checkPurposeConstraints(
        uint16 modifiers,
        uint8 purpose
    ) public pure returns (bool allowed, string memory reason) {

        if ((modifiers & MOD_GSO) != 0) {
            if (purpose != PURPOSE_GENETICS) {
                return (false, "GSO: Only genetic studies allowed");
            }
        }

        if ((modifiers & MOD_NPOA) != 0) {
            if (purpose == PURPOSE_POPULATION) {
                return (false, "NPOA: Population/ancestry research prohibited");
            }
        }

        return (true, "");
    }

    function getRequiredAttestations(
        uint16 modifiers
    ) public pure returns (bytes32[] memory attestationTypes) {

        uint8 count = 0;
        if ((modifiers & MOD_IRB) != 0) count++;
        if ((modifiers & MOD_COL) != 0) count++;
        if ((modifiers & MOD_GS) != 0) count++;
        if ((modifiers & MOD_IS) != 0) count++;
        if ((modifiers & MOD_PS) != 0) count++;
        if ((modifiers & MOD_US) != 0) count++;
        if ((modifiers & MOD_PUB) != 0) count++;
        if ((modifiers & MOD_RTN) != 0) count++;

        attestationTypes = new bytes32[](count);
        uint8 idx = 0;

        if ((modifiers & MOD_IRB) != 0) attestationTypes[idx++] = keccak256("IRB_APPROVAL");
        if ((modifiers & MOD_COL) != 0) attestationTypes[idx++] = keccak256("COLLABORATION");
        if ((modifiers & MOD_GS) != 0) attestationTypes[idx++] = keccak256("GEOGRAPHIC");
        if ((modifiers & MOD_IS) != 0) attestationTypes[idx++] = keccak256("INSTITUTION");
        if ((modifiers & MOD_PS) != 0) attestationTypes[idx++] = keccak256("PROJECT");
        if ((modifiers & MOD_US) != 0) attestationTypes[idx++] = keccak256("USER");
        if ((modifiers & MOD_PUB) != 0) attestationTypes[idx++] = keccak256("PUBLICATION_PROMISE");
        if ((modifiers & MOD_RTN) != 0) attestationTypes[idx++] = keccak256("RETURN_DATA_PROMISE");

        return attestationTypes;
    }

    function setDiseaseHierarchy(
        bytes32 childDisease,
        bytes32 parentDisease
    ) external onlyRole(ONTOLOGY_ADMIN) {
        diseaseParent[childDisease] = parentDisease;
        emit DiseaseHierarchySet(childDisease, parentDisease);
    }

    function setDiseaseHierarchyBatch(
        bytes32[] calldata children,
        bytes32[] calldata parents
    ) external onlyRole(ONTOLOGY_ADMIN) {
        require(children.length == parents.length, "Length mismatch");
        for (uint i = 0; i < children.length; i++) {
            diseaseParent[children[i]] = parents[i];
            emit DiseaseHierarchySet(children[i], parents[i]);
        }
    }

    function getModifierCodes(uint16 modifiers) public pure returns (string[] memory codes) {
        uint8 count = 0;
        for (uint8 i = 0; i < 16; i++) {
            if ((modifiers & (1 << i)) != 0) count++;
        }

        codes = new string[](count);
        uint8 idx = 0;

        if ((modifiers & MOD_NPU) != 0) codes[idx++] = "NPU";
        if ((modifiers & MOD_NCU) != 0) codes[idx++] = "NCU";
        if ((modifiers & MOD_NPUNCU) != 0) codes[idx++] = "NPUNCU";
        if ((modifiers & MOD_PUB) != 0) codes[idx++] = "PUB";
        if ((modifiers & MOD_COL) != 0) codes[idx++] = "COL";
        if ((modifiers & MOD_IRB) != 0) codes[idx++] = "IRB";
        if ((modifiers & MOD_GS) != 0) codes[idx++] = "GS";
        if ((modifiers & MOD_MOR) != 0) codes[idx++] = "MOR";
        if ((modifiers & MOD_TS) != 0) codes[idx++] = "TS";
        if ((modifiers & MOD_US) != 0) codes[idx++] = "US";
        if ((modifiers & MOD_PS) != 0) codes[idx++] = "PS";
        if ((modifiers & MOD_IS) != 0) codes[idx++] = "IS";
        if ((modifiers & MOD_RTN) != 0) codes[idx++] = "RTN";
        if ((modifiers & MOD_CC) != 0) codes[idx++] = "CC";
        if ((modifiers & MOD_NPOA) != 0) codes[idx++] = "NPOA";
        if ((modifiers & MOD_GSO) != 0) codes[idx++] = "GSO";

        return codes;
    }

    function getPermissionLabel(bytes4 permission) public pure returns (string memory) {
        if (permission == NRES) return "No Restriction";
        if (permission == GRU) return "General Research Use";
        if (permission == HMB) return "Health/Medical/Biomedical";
        if (permission == DS) return "Disease Specific";
        if (permission == POA) return "Population Origins/Ancestry";
        return "Unknown";
    }
}
