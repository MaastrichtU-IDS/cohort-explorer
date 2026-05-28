// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

interface IEAS {
    struct Attestation {
        bytes32 uid;
        bytes32 schema;
        uint64 time;
        uint64 expirationTime;
        uint64 revocationTime;
        bytes32 refUID;
        address recipient;
        address attester;
        bool revocable;
        bytes data;
    }

    function getAttestation(bytes32 uid) external view returns (Attestation memory);
    function isAttestationValid(bytes32 uid) external view returns (bool);
}

interface ISchemaResolver {
    function isPayable() external pure returns (bool);

    function attest(
        Attestation calldata attestation
    ) external payable returns (bool);

    function revoke(
        Attestation calldata attestation
    ) external payable returns (bool);

    function multiAttest(
        Attestation[] calldata attestations,
        uint256[] calldata values
    ) external payable returns (bool);

    function multiRevoke(
        Attestation[] calldata attestations,
        uint256[] calldata values
    ) external payable returns (bool);

    struct Attestation {
        bytes32 uid;
        bytes32 schema;
        uint64 time;
        uint64 expirationTime;
        uint64 revocationTime;
        bytes32 refUID;
        address recipient;
        address attester;
        bool revocable;
        bytes data;
    }
}

contract DUOAttestationResolver is AccessControl, ISchemaResolver {

    bytes32 public constant TRUSTED_ATTESTER = keccak256("TRUSTED_ATTESTER");

    bytes32 public schemaIRBApproval;
    bytes32 public schemaCollaboration;
    bytes32 public schemaPublication;
    bytes32 public schemaReturnData;
    bytes32 public schemaGeographic;
    bytes32 public schemaInstitution;

    IEAS public eas;

    mapping(bytes32 => mapping(address => bytes32)) public irbAttestations;

    mapping(bytes32 => mapping(address => bytes32)) public collaborationAttestations;

    mapping(bytes32 => mapping(address => bytes32)) public publicationAttestations;

    mapping(bytes32 => mapping(address => bytes32)) public returnDataAttestations;

    mapping(address => bytes32) public geographicAttestations;

    mapping(address => bytes32) public institutionAttestations;

    mapping(bytes32 => mapping(address => bool)) public trustedAttesters;

    bytes32[] public allAttestations;

    event SchemaRegistered(
        string schemaType,
        bytes32 schemaId
    );

    event TrustedAttesterAdded(
        bytes32 indexed schemaId,
        address indexed attester
    );

    event TrustedAttesterRemoved(
        bytes32 indexed schemaId,
        address indexed attester
    );

    event AttestationRecorded(
        bytes32 indexed uid,
        bytes32 indexed schemaId,
        address indexed recipient,
        bytes32 cohortHash
    );

    event AttestationRevoked(
        bytes32 indexed uid,
        bytes32 indexed schemaId
    );

    constructor(address _eas) {
        eas = IEAS(_eas);

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(TRUSTED_ATTESTER, msg.sender);
    }

    function setSchemaIRBApproval(bytes32 schemaId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        schemaIRBApproval = schemaId;
        emit SchemaRegistered("IRB_APPROVAL", schemaId);
    }

    function setSchemaCollaboration(bytes32 schemaId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        schemaCollaboration = schemaId;
        emit SchemaRegistered("COLLABORATION", schemaId);
    }

    function setSchemaPublication(bytes32 schemaId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        schemaPublication = schemaId;
        emit SchemaRegistered("PUBLICATION", schemaId);
    }

    function setSchemaReturnData(bytes32 schemaId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        schemaReturnData = schemaId;
        emit SchemaRegistered("RETURN_DATA", schemaId);
    }

    function setSchemaGeographic(bytes32 schemaId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        schemaGeographic = schemaId;
        emit SchemaRegistered("GEOGRAPHIC", schemaId);
    }

    function setSchemaInstitution(bytes32 schemaId) external onlyRole(DEFAULT_ADMIN_ROLE) {
        schemaInstitution = schemaId;
        emit SchemaRegistered("INSTITUTION", schemaId);
    }

    function addTrustedAttester(
        bytes32 schemaId,
        address attester
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        trustedAttesters[schemaId][attester] = true;
        emit TrustedAttesterAdded(schemaId, attester);
    }

    function removeTrustedAttester(
        bytes32 schemaId,
        address attester
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        trustedAttesters[schemaId][attester] = false;
        emit TrustedAttesterRemoved(schemaId, attester);
    }

    function isTrustedAttester(
        bytes32 schemaId,
        address attester
    ) public view returns (bool) {
        return trustedAttesters[schemaId][attester] ||
               hasRole(TRUSTED_ATTESTER, attester);
    }

    function isPayable() external pure override returns (bool) {
        return false;
    }

    function attest(
        Attestation calldata attestation
    ) external payable override returns (bool) {

        require(
            isTrustedAttester(attestation.schema, attestation.attester),
            "Attester not trusted for this schema"
        );

        if (attestation.schema == schemaIRBApproval) {
            return _processIRBAttestation(attestation);
        } else if (attestation.schema == schemaCollaboration) {
            return _processCollaborationAttestation(attestation);
        } else if (attestation.schema == schemaPublication) {
            return _processPublicationAttestation(attestation);
        } else if (attestation.schema == schemaReturnData) {
            return _processReturnDataAttestation(attestation);
        } else if (attestation.schema == schemaGeographic) {
            return _processGeographicAttestation(attestation);
        } else if (attestation.schema == schemaInstitution) {
            return _processInstitutionAttestation(attestation);
        }

        return false;
    }

    function revoke(
        Attestation calldata attestation
    ) external payable override returns (bool) {

        if (attestation.schema == schemaIRBApproval) {
            return _revokeIRBAttestation(attestation);
        } else if (attestation.schema == schemaCollaboration) {
            return _revokeCollaborationAttestation(attestation);
        } else if (attestation.schema == schemaPublication) {
            return _revokePublicationAttestation(attestation);
        } else if (attestation.schema == schemaReturnData) {
            return _revokeReturnDataAttestation(attestation);
        }

        emit AttestationRevoked(attestation.uid, attestation.schema);
        return true;
    }

    function multiAttest(
        Attestation[] calldata attestations,
        uint256[] calldata
    ) external payable override returns (bool) {
        for (uint256 i = 0; i < attestations.length; i++) {
            if (!this.attest(attestations[i])) {
                return false;
            }
        }
        return true;
    }

    function multiRevoke(
        Attestation[] calldata attestations,
        uint256[] calldata
    ) external payable override returns (bool) {
        for (uint256 i = 0; i < attestations.length; i++) {
            if (!this.revoke(attestations[i])) {
                return false;
            }
        }
        return true;
    }

    function _processIRBAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {

        (bytes32 cohortHash, , , uint64 validUntil, ) = abi.decode(
            attestation.data,
            (bytes32, string, string, uint64, bytes32)
        );

        require(cohortHash != bytes32(0), "Invalid cohort hash");
        require(validUntil > block.timestamp, "IRB approval expired");

        irbAttestations[cohortHash][attestation.recipient] = attestation.uid;
        allAttestations.push(attestation.uid);

        emit AttestationRecorded(
            attestation.uid,
            attestation.schema,
            attestation.recipient,
            cohortHash
        );

        return true;
    }

    function _processCollaborationAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, address partner, , uint64 validUntil) = abi.decode(
            attestation.data,
            (bytes32, address, string, uint64)
        );

        require(cohortHash != bytes32(0), "Invalid cohort hash");
        require(partner != address(0), "Invalid partner");
        require(validUntil == 0 || validUntil > block.timestamp, "Collaboration expired");

        collaborationAttestations[cohortHash][partner] = attestation.uid;
        allAttestations.push(attestation.uid);

        emit AttestationRecorded(
            attestation.uid,
            attestation.schema,
            partner,
            cohortHash
        );

        return true;
    }

    function _processPublicationAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, , ) = abi.decode(
            attestation.data,
            (bytes32, string, uint64)
        );

        require(cohortHash != bytes32(0), "Invalid cohort hash");

        publicationAttestations[cohortHash][attestation.recipient] = attestation.uid;
        allAttestations.push(attestation.uid);

        emit AttestationRecorded(
            attestation.uid,
            attestation.schema,
            attestation.recipient,
            cohortHash
        );

        return true;
    }

    function _processReturnDataAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, , , ) = abi.decode(
            attestation.data,
            (bytes32, string, string, uint64)
        );

        require(cohortHash != bytes32(0), "Invalid cohort hash");

        returnDataAttestations[cohortHash][attestation.recipient] = attestation.uid;
        allAttestations.push(attestation.uid);

        emit AttestationRecorded(
            attestation.uid,
            attestation.schema,
            attestation.recipient,
            cohortHash
        );

        return true;
    }

    function _processGeographicAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (address subject, bytes2 countryCode, ) = abi.decode(
            attestation.data,
            (address, bytes2, string)
        );

        require(subject != address(0), "Invalid subject");
        require(countryCode != bytes2(0), "Invalid country code");

        geographicAttestations[subject] = attestation.uid;
        allAttestations.push(attestation.uid);

        emit AttestationRecorded(
            attestation.uid,
            attestation.schema,
            subject,
            bytes32(0)
        );

        return true;
    }

    function _processInstitutionAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (address subject, string memory rorId, , ) = abi.decode(
            attestation.data,
            (address, string, string, string)
        );

        require(subject != address(0), "Invalid subject");
        require(bytes(rorId).length > 0, "Invalid ROR ID");

        institutionAttestations[subject] = attestation.uid;
        allAttestations.push(attestation.uid);

        emit AttestationRecorded(
            attestation.uid,
            attestation.schema,
            subject,
            bytes32(0)
        );

        return true;
    }

    function _revokeIRBAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, , , , ) = abi.decode(
            attestation.data,
            (bytes32, string, string, uint64, bytes32)
        );

        delete irbAttestations[cohortHash][attestation.recipient];
        emit AttestationRevoked(attestation.uid, attestation.schema);
        return true;
    }

    function _revokeCollaborationAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, address partner, , ) = abi.decode(
            attestation.data,
            (bytes32, address, string, uint64)
        );

        delete collaborationAttestations[cohortHash][partner];
        emit AttestationRevoked(attestation.uid, attestation.schema);
        return true;
    }

    function _revokePublicationAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, , ) = abi.decode(
            attestation.data,
            (bytes32, string, uint64)
        );

        delete publicationAttestations[cohortHash][attestation.recipient];
        emit AttestationRevoked(attestation.uid, attestation.schema);
        return true;
    }

    function _revokeReturnDataAttestation(
        Attestation calldata attestation
    ) internal returns (bool) {
        (bytes32 cohortHash, , , ) = abi.decode(
            attestation.data,
            (bytes32, string, string, uint64)
        );

        delete returnDataAttestations[cohortHash][attestation.recipient];
        emit AttestationRevoked(attestation.uid, attestation.schema);
        return true;
    }

    function hasValidIRB(
        bytes32 cohortHash,
        address subject
    ) external view returns (bool, bytes32) {
        bytes32 uid = irbAttestations[cohortHash][subject];
        if (uid == bytes32(0)) {
            return (false, bytes32(0));
        }

        if (address(eas) != address(0)) {
            if (!eas.isAttestationValid(uid)) {
                return (false, uid);
            }

            IEAS.Attestation memory att = eas.getAttestation(uid);
            if (att.expirationTime > 0 && block.timestamp > att.expirationTime) {
                return (false, uid);
            }
            if (att.revocationTime > 0) {
                return (false, uid);
            }
        }

        return (true, uid);
    }

    function hasValidCollaboration(
        bytes32 cohortHash,
        address partner
    ) external view returns (bool, bytes32) {
        bytes32 uid = collaborationAttestations[cohortHash][partner];
        if (uid == bytes32(0)) {
            return (false, bytes32(0));
        }

        if (address(eas) != address(0)) {
            if (!eas.isAttestationValid(uid)) {
                return (false, uid);
            }
        }

        return (true, uid);
    }

    function hasPublicationCommitment(
        bytes32 cohortHash,
        address subject
    ) external view returns (bool, bytes32) {
        bytes32 uid = publicationAttestations[cohortHash][subject];
        if (uid == bytes32(0)) {
            return (false, bytes32(0));
        }

        if (address(eas) != address(0)) {
            if (!eas.isAttestationValid(uid)) {
                return (false, uid);
            }
        }

        return (true, uid);
    }

    function hasReturnDataCommitment(
        bytes32 cohortHash,
        address subject
    ) external view returns (bool, bytes32) {
        bytes32 uid = returnDataAttestations[cohortHash][subject];
        if (uid == bytes32(0)) {
            return (false, bytes32(0));
        }

        if (address(eas) != address(0)) {
            if (!eas.isAttestationValid(uid)) {
                return (false, uid);
            }
        }

        return (true, uid);
    }

    function getGeographicAttestation(
        address subject
    ) external view returns (bool, bytes32, bytes2) {
        bytes32 uid = geographicAttestations[subject];
        if (uid == bytes32(0)) {
            return (false, bytes32(0), bytes2(0));
        }

        if (address(eas) != address(0)) {
            if (!eas.isAttestationValid(uid)) {
                return (false, uid, bytes2(0));
            }

            IEAS.Attestation memory att = eas.getAttestation(uid);
            (, bytes2 countryCode, ) = abi.decode(
                att.data,
                (address, bytes2, string)
            );
            return (true, uid, countryCode);
        }

        return (true, uid, bytes2(0));
    }

    function getInstitutionAttestation(
        address subject
    ) external view returns (bool, bytes32, string memory, string memory) {
        bytes32 uid = institutionAttestations[subject];
        if (uid == bytes32(0)) {
            return (false, bytes32(0), "", "");
        }

        if (address(eas) != address(0)) {
            if (!eas.isAttestationValid(uid)) {
                return (false, uid, "", "");
            }

            IEAS.Attestation memory att = eas.getAttestation(uid);
            (, string memory rorId, string memory instName, ) = abi.decode(
                att.data,
                (address, string, string, string)
            );
            return (true, uid, rorId, instName);
        }

        return (true, uid, "", "");
    }

    function getTotalAttestations() external view returns (uint256) {
        return allAttestations.length;
    }

    function setEAS(address _eas) external onlyRole(DEFAULT_ADMIN_ROLE) {
        eas = IEAS(_eas);
    }
}
