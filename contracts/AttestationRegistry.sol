// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

interface IRoleGroupRegistryAR {
    function commitmentOf(bytes32 roleId, address ownerEOA) external view returns (bytes32);
    function ROLE_REQUESTER() external view returns (bytes32);
}

interface IRoleAccountFactoryAR {
    function computeAddress(address eoa, bytes32 commitment) external view returns (address);
}

interface IERC1271AR {
    function isValidSignature(bytes32 hash, bytes calldata signature) external view returns (bytes4);
}

contract AttestationRegistry is AccessControl, EIP712 {
    bytes4 internal constant ERC1271_MAGIC_AR = 0x1626ba7e;

    bytes32 public constant RECORD_COMMITMENT_TYPEHASH = keccak256(
        "RecordCommitment(bytes32 commitmentType,address signerEOA,bytes32 cohortHash,string details,uint256 nonce)"
    );

    IRoleGroupRegistryAR public roleGroupRegistry;
    IRoleAccountFactoryAR public roleAccountFactory;
    mapping(address => uint256) public commitmentNonces;

    bytes32 public constant REGISTRY_ADMIN = keccak256("REGISTRY_ADMIN");

    bytes32 public constant ATT_IRB = keccak256("IRB_APPROVAL");
    bytes32 public constant ATT_COL = keccak256("COLLABORATION");
    bytes32 public constant ATT_PROJECT = keccak256("PROJECT");
    bytes32 public constant ATT_USER = keccak256("USER_WHITELIST");
    bytes32 public constant ATT_PUB = keccak256("PUBLICATION_PROMISE");
    bytes32 public constant ATT_RTN = keccak256("RETURN_DATA_PROMISE");
    bytes32 public constant ATT_GEO = keccak256("GEOGRAPHIC");

    struct Attestation {
        bytes32 attestationId;
        bytes32 attestationType;
        address subject;
        bytes32 scope;
        address attestor;
        bytes32 evidenceHash;
        uint256 issuedAt;
        uint256 validUntil;
        bool revoked;
        string metadata;
    }

    struct Project {
        bytes32 projectId;
        string name;
        address owner;
        bytes32 institutionId;
        bytes32[] cohortAccess;
        uint256 registeredAt;
        uint256 validUntil;
        bool active;
        string descriptionURI;
    }

    struct CollaborationAgreement {
        bytes32 agreementId;
        bytes32 cohortHash;
        address dataOwner;
        address requester;
        bytes32 projectId;
        uint256 agreedAt;
        uint256 validUntil;
        bool active;
        string termsURI;
    }

    mapping(bytes32 => Attestation) public attestations;

    mapping(address => mapping(bytes32 => mapping(bytes32 => bytes32))) public activeAttestation;

    mapping(bytes32 => mapping(address => bool)) public trustedAttestors;

    mapping(bytes32 => Project) public projects;
    mapping(address => bytes32[]) public userProjects;

    mapping(bytes32 => CollaborationAgreement) public collaborations;

    mapping(bytes32 => mapping(address => bytes32)) public cohortCollaborations;

    mapping(bytes32 => mapping(address => bool)) public userWhitelist;

    mapping(bytes32 => uint256) public publicationMoratorium;

    event AttestorAdded(bytes32 indexed attestationType, address indexed attestor);
    event AttestorRemoved(bytes32 indexed attestationType, address indexed attestor);

    event AttestationCreated(
        bytes32 indexed attestationId,
        bytes32 indexed attestationType,
        address indexed subject,
        bytes32 scope,
        address attestor
    );

    event AttestationRevoked(bytes32 indexed attestationId, address revokedBy);

    event ProjectRegistered(
        bytes32 indexed projectId,
        string name,
        address indexed owner,
        bytes32 indexed institutionId
    );

    event ProjectDeactivated(bytes32 indexed projectId);

    event CollaborationEstablished(
        bytes32 indexed agreementId,
        bytes32 indexed cohortHash,
        address indexed requester,
        address dataOwner
    );

    event CollaborationRevoked(bytes32 indexed agreementId);

    event UserWhitelisted(bytes32 indexed cohortHash, address indexed user, bool allowed);
    event MoratoriumSet(bytes32 indexed cohortHash, uint256 until);

    constructor() EIP712("AttestationRegistry", "1") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(REGISTRY_ADMIN, msg.sender);
    }

    function setRoleInfra(address registry, address factory) external onlyRole(DEFAULT_ADMIN_ROLE) {
        roleGroupRegistry = IRoleGroupRegistryAR(registry);
        roleAccountFactory = IRoleAccountFactoryAR(factory);
    }

    function domainSeparator() external view returns (bytes32) {
        return _domainSeparatorV4();
    }

    function addTrustedAttestor(
        bytes32 attestationType,
        address attestor
    ) external onlyRole(REGISTRY_ADMIN) {
        require(attestor != address(0), "Invalid attestor");
        trustedAttestors[attestationType][attestor] = true;
        emit AttestorAdded(attestationType, attestor);
    }

    function removeTrustedAttestor(
        bytes32 attestationType,
        address attestor
    ) external onlyRole(REGISTRY_ADMIN) {
        trustedAttestors[attestationType][attestor] = false;
        emit AttestorRemoved(attestationType, attestor);
    }

    function isTrustedAttestor(bytes32 attestationType, address attestor) public view returns (bool) {
        return trustedAttestors[attestationType][attestor] || hasRole(REGISTRY_ADMIN, attestor);
    }

    function createAttestation(
        bytes32 attestationType,
        address subject,
        bytes32 scope,
        bytes32 evidenceHash,
        uint256 validDays,
        string calldata metadata
    ) external returns (bytes32 attestationId) {
        require(isTrustedAttestor(attestationType, msg.sender), "Not authorized attestor");
        require(subject != address(0), "Invalid subject");

        attestationId = keccak256(abi.encodePacked(
            attestationType,
            subject,
            scope,
            msg.sender,
            block.timestamp
        ));

        uint256 validUntil = validDays > 0 ? block.timestamp + (validDays * 1 days) : 0;

        attestations[attestationId] = Attestation({
            attestationId: attestationId,
            attestationType: attestationType,
            subject: subject,
            scope: scope,
            attestor: msg.sender,
            evidenceHash: evidenceHash,
            issuedAt: block.timestamp,
            validUntil: validUntil,
            revoked: false,
            metadata: metadata
        });

        activeAttestation[subject][attestationType][scope] = attestationId;

        emit AttestationCreated(attestationId, attestationType, subject, scope, msg.sender);

        return attestationId;
    }

    function revokeAttestation(bytes32 attestationId) external {
        Attestation storage att = attestations[attestationId];
        require(att.issuedAt > 0, "Attestation not found");
        require(
            msg.sender == att.attestor || hasRole(REGISTRY_ADMIN, msg.sender),
            "Not authorized"
        );

        att.revoked = true;

        if (activeAttestation[att.subject][att.attestationType][att.scope] == attestationId) {
            activeAttestation[att.subject][att.attestationType][att.scope] = bytes32(0);
        }

        emit AttestationRevoked(attestationId, msg.sender);
    }

    function hasValidAttestation(
        address subject,
        bytes32 attestationType,
        bytes32 scope
    ) public view returns (bool valid, bytes32 attestationId) {

        attestationId = activeAttestation[subject][attestationType][scope];
        if (_isAttestationValid(attestationId)) {
            return (true, attestationId);
        }

        if (scope != bytes32(0)) {
            attestationId = activeAttestation[subject][attestationType][bytes32(0)];
            if (_isAttestationValid(attestationId)) {
                return (true, attestationId);
            }
        }

        return (false, bytes32(0));
    }

    function _isAttestationValid(bytes32 attestationId) internal view returns (bool) {
        if (attestationId == bytes32(0)) return false;
        Attestation storage att = attestations[attestationId];
        if (att.revoked) return false;
        if (att.validUntil > 0 && block.timestamp > att.validUntil) return false;
        return true;
    }

    function registerProject(
        string calldata name,
        bytes32 institutionId,
        uint256 validDays,
        string calldata descriptionURI
    ) external returns (bytes32 projectId) {
        require(bytes(name).length > 0, "Name required");

        projectId = keccak256(abi.encodePacked(name, msg.sender, block.timestamp));

        uint256 validUntil = validDays > 0 ? block.timestamp + (validDays * 1 days) : 0;

        projects[projectId] = Project({
            projectId: projectId,
            name: name,
            owner: msg.sender,
            institutionId: institutionId,
            cohortAccess: new bytes32[](0),
            registeredAt: block.timestamp,
            validUntil: validUntil,
            active: true,
            descriptionURI: descriptionURI
        });

        userProjects[msg.sender].push(projectId);

        emit ProjectRegistered(projectId, name, msg.sender, institutionId);

        return projectId;
    }

    function grantProjectCohortAccess(
        bytes32 projectId,
        bytes32 cohortHash
    ) external {

        require(hasRole(REGISTRY_ADMIN, msg.sender), "Not authorized");
        require(projects[projectId].active, "Project not active");

        projects[projectId].cohortAccess.push(cohortHash);
    }

    function hasProjectAccess(bytes32 projectId, bytes32 cohortHash) public view returns (bool) {
        Project storage p = projects[projectId];
        if (!p.active) return false;
        if (p.validUntil > 0 && block.timestamp > p.validUntil) return false;

        for (uint i = 0; i < p.cohortAccess.length; i++) {
            if (p.cohortAccess[i] == cohortHash) return true;
        }
        return false;
    }

    function isProjectMember(bytes32 projectId, address user) public view returns (bool) {

        return projects[projectId].owner == user && projects[projectId].active;
    }

    function establishCollaboration(
        bytes32 cohortHash,
        address requester,
        bytes32 projectId,
        uint256 validDays,
        string calldata termsURI
    ) external returns (bytes32 agreementId) {

        agreementId = keccak256(abi.encodePacked(
            cohortHash,
            msg.sender,
            requester,
            block.timestamp
        ));

        uint256 validUntil = validDays > 0 ? block.timestamp + (validDays * 1 days) : 0;

        collaborations[agreementId] = CollaborationAgreement({
            agreementId: agreementId,
            cohortHash: cohortHash,
            dataOwner: msg.sender,
            requester: requester,
            projectId: projectId,
            agreedAt: block.timestamp,
            validUntil: validUntil,
            active: true,
            termsURI: termsURI
        });

        cohortCollaborations[cohortHash][requester] = agreementId;

        emit CollaborationEstablished(agreementId, cohortHash, requester, msg.sender);

        return agreementId;
    }

    function revokeCollaboration(bytes32 agreementId) external {
        CollaborationAgreement storage c = collaborations[agreementId];
        require(c.agreedAt > 0, "Agreement not found");
        require(
            msg.sender == c.dataOwner || hasRole(REGISTRY_ADMIN, msg.sender),
            "Not authorized"
        );

        c.active = false;
        cohortCollaborations[c.cohortHash][c.requester] = bytes32(0);

        emit CollaborationRevoked(agreementId);
    }

    function hasCollaboration(bytes32 cohortHash, address requester) public view returns (bool) {
        bytes32 agreementId = cohortCollaborations[cohortHash][requester];
        if (agreementId == bytes32(0)) return false;

        CollaborationAgreement storage c = collaborations[agreementId];
        if (!c.active) return false;
        if (c.validUntil > 0 && block.timestamp > c.validUntil) return false;
        return true;
    }

    function setUserWhitelist(
        bytes32 cohortHash,
        address user,
        bool allowed
    ) external {

        userWhitelist[cohortHash][user] = allowed;
        emit UserWhitelisted(cohortHash, user, allowed);
    }

    function setUserWhitelistBatch(
        bytes32 cohortHash,
        address[] calldata users,
        bool allowed
    ) external {
        for (uint i = 0; i < users.length; i++) {
            userWhitelist[cohortHash][users[i]] = allowed;
            emit UserWhitelisted(cohortHash, users[i], allowed);
        }
    }

    function isUserWhitelisted(bytes32 cohortHash, address user) public view returns (bool) {
        return userWhitelist[cohortHash][user];
    }

    function setMoratorium(bytes32 cohortHash, uint256 until) external {

        publicationMoratorium[cohortHash] = until;
        emit MoratoriumSet(cohortHash, until);
    }

    function isMoratoriumActive(bytes32 cohortHash) public view returns (bool active, uint256 until) {
        until = publicationMoratorium[cohortHash];
        active = until > 0 && block.timestamp < until;
        return (active, until);
    }

    function recordCommitment(
        bytes32 commitmentType,
        bytes32 cohortHash,
        string calldata details
    ) external returns (bytes32 attestationId) {
        require(
            commitmentType == ATT_PUB || commitmentType == ATT_RTN,
            "Invalid commitment type"
        );

        attestationId = keccak256(abi.encodePacked(
            commitmentType,
            msg.sender,
            cohortHash,
            block.timestamp
        ));

        attestations[attestationId] = Attestation({
            attestationId: attestationId,
            attestationType: commitmentType,
            subject: msg.sender,
            scope: cohortHash,
            attestor: msg.sender,
            evidenceHash: bytes32(0),
            issuedAt: block.timestamp,
            validUntil: 0,
            revoked: false,
            metadata: details
        });

        activeAttestation[msg.sender][commitmentType][cohortHash] = attestationId;

        emit AttestationCreated(attestationId, commitmentType, msg.sender, cohortHash, msg.sender);

        return attestationId;
    }

    function recordCommitmentWithSignature(
        bytes32 commitmentType,
        address signerEOA,
        bytes32 cohortHash,
        string calldata details,
        uint256 nonce,
        bytes calldata signature
    ) external returns (bytes32 attestationId) {
        require(
            commitmentType == ATT_PUB || commitmentType == ATT_RTN,
            "Invalid commitment type"
        );
        require(address(roleGroupRegistry) != address(0), "registry not set");
        require(address(roleAccountFactory) != address(0), "factory not set");
        require(nonce == commitmentNonces[signerEOA], "Bad nonce");

        bytes32 structHash = keccak256(abi.encode(
            RECORD_COMMITMENT_TYPEHASH,
            commitmentType,
            signerEOA,
            cohortHash,
            keccak256(bytes(details)),
            nonce
        ));
        bytes32 digest = _hashTypedDataV4(structHash);

        bytes32 commitment = roleGroupRegistry.commitmentOf(
            roleGroupRegistry.ROLE_REQUESTER(),
            signerEOA
        );
        require(commitment != bytes32(0), "role not activated");
        address principal = roleAccountFactory.computeAddress(signerEOA, commitment);
        require(principal.code.length > 0, "role account not deployed");
        require(
            IERC1271AR(principal).isValidSignature(digest, signature) == ERC1271_MAGIC_AR,
            "ERC1271 reject"
        );

        commitmentNonces[signerEOA] = nonce + 1;

        attestationId = keccak256(abi.encodePacked(
            commitmentType,
            principal,
            cohortHash,
            block.timestamp
        ));

        attestations[attestationId] = Attestation({
            attestationId: attestationId,
            attestationType: commitmentType,
            subject: principal,
            scope: cohortHash,
            attestor: principal,
            evidenceHash: bytes32(0),
            issuedAt: block.timestamp,
            validUntil: 0,
            revoked: false,
            metadata: details
        });

        activeAttestation[principal][commitmentType][cohortHash] = attestationId;

        emit AttestationCreated(attestationId, commitmentType, principal, cohortHash, principal);

        return attestationId;
    }

    function getAttestation(bytes32 attestationId) external view returns (Attestation memory) {
        return attestations[attestationId];
    }

    function getProject(bytes32 projectId) external view returns (Project memory) {
        return projects[projectId];
    }

    function getCollaboration(bytes32 agreementId) external view returns (CollaborationAgreement memory) {
        return collaborations[agreementId];
    }

    function getUserProjectCount(address user) external view returns (uint256) {
        return userProjects[user].length;
    }
}
