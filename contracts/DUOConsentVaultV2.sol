// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "./DUOOntology.sol";
import "./AttestationRegistry.sol";
import "./InstitutionRegistry.sol";

interface IIBISVerifier {
    function processEnvelope(
        bytes32 commitment,
        bytes32 nullifier,
        bytes32 blindedEpoch,
        bytes32 epochKeyHash,
        bytes calldata encryptedPayload,
        bytes calldata proof
    ) external;
}

interface IRoleGroupRegistry {
    function commitmentOf(bytes32 roleId, address ownerEOA) external view returns (bytes32);
    function ownerOf(bytes32 roleId, bytes32 commitment) external view returns (address);
    function ROLE_PROVIDER() external view returns (bytes32);
    function ROLE_REQUESTER() external view returns (bytes32);
}

interface IRoleAccountFactory {
    function computeAddress(address owner, bytes32 roleCommitment) external view returns (address);
}

interface IERC1271 {
    function isValidSignature(bytes32 hash, bytes calldata signature) external view returns (bytes4);
}

contract DUOConsentVaultV2 is AccessControl, ReentrancyGuard, EIP712 {
    using ECDSA for bytes32;

    bytes32 public constant CONSENT_ADMIN = keccak256("CONSENT_ADMIN");
    bytes32 public constant RELAYER_ROLE = keccak256("RELAYER_ROLE");

    bytes32 public constant CONSENT_TYPEHASH = keccak256(
        "RecordConsent(bytes32 cohortHash,bytes4 permission,uint256 modifiers,bytes32[] diseaseCodes,bytes32 metadataHash,uint256 countryBitset,uint256 validDays,uint256 moratoriumMonths,uint256 publicationDeadlineDays,bytes32 institutionsRoot,bytes32 institutionIdsHash,bytes32 projectIdsHash,bytes32 userAddressesHash,uint256 nonce)"
    );

    bytes32 public constant ACCESS_TYPEHASH = keccak256(
        "RequestAccess(bytes32 cohortHash,address requester,bytes4 intendedUse,uint8 purpose,bytes32[] diseaseCodes,bytes32 projectId,uint8 countryIndex,bytes32 institutionId,uint256 nonce)"
    );

    bytes32 public constant REVOKE_TYPEHASH = keccak256(
        "RevokeConsent(bytes32 cohortHash,uint256 nonce)"
    );

    DUOOntology public immutable ontology;
    AttestationRegistry public immutable attestationRegistry;

    struct ConsentCore {

        bytes32 cohortHash;

        bytes4 permission;
        uint32 modifiersLow;
        uint64 validFrom;
        uint64 validUntil;
        uint16 ownerCount;
        bool active;
        uint8 flags;
        uint16 moratoriumMonths;
        uint16 publicationDeadlineDays;

        bytes32 metadataHash;
    }

    struct MerkleRoots {
        bytes32 countriesRoot;
        bytes32 institutionsRoot;
    }

    struct ExtendedModifiers {
        uint256 modifiersHigh;
    }

    struct AccessGrant {
        uint64 grantedAt;
        uint64 expiresAt;
        bytes4 grantedUse;
        uint8 status;
        uint8 flags;
        uint8 reason;
    }

    mapping(bytes32 => ConsentCore) public consents;

    mapping(bytes32 => bytes32[]) internal _consentDiseases;

    mapping(bytes32 => MerkleRoots) public merkleRoots;

    mapping(bytes32 => ExtendedModifiers) public extendedModifiers;

    mapping(bytes32 => address[]) internal _owners;

    mapping(bytes32 => mapping(address => bool)) public isOwner;

    mapping(bytes32 => mapping(address => AccessGrant)) public accessGrants;

    mapping(address => uint256) public nonces;

    mapping(bytes32 => uint256) public countryBitset;

    mapping(bytes32 => mapping(bytes32 => bool)) public allowedInstitutions;
    mapping(bytes32 => mapping(bytes32 => bool)) public allowedProjects;
    mapping(bytes32 => mapping(address => bool)) public allowedUsers;

    bytes32[] public allCohorts;

    mapping(bytes32 => uint256) public cohortIndex;

    uint8 public constant STATUS_PENDING = 0;
    uint8 public constant STATUS_APPROVED = 1;
    uint8 public constant STATUS_REVOKED = 2;
    uint8 public constant STATUS_REJECTED = 3;

    uint8 public constant R_OK = 0;
    uint8 public constant R_CONSENT_INACTIVE = 1;
    uint8 public constant R_CONSENT_EXPIRED = 2;
    uint8 public constant R_PERM = 3;
    uint8 public constant R_DISEASE = 4;
    uint8 public constant R_NPOA = 5;
    uint8 public constant R_GSO = 6;
    uint8 public constant R_NMDS = 7;
    uint8 public constant R_GS = 8;
    uint8 public constant R_IS = 9;
    uint8 public constant R_PS = 10;
    uint8 public constant R_US = 11;
    uint8 public constant R_NPU = 12;
    uint8 public constant R_NCU = 13;
    uint8 public constant R_PUB = 14;
    uint8 public constant R_RTN = 15;
    uint8 public constant R_COL = 16;

    event ConsentRecorded(
        bytes32 indexed cohortHash,
        address indexed primaryOwner,
        uint64 validUntil,
        bool viaSig
    );

    event ConsentUpdated(
        bytes32 indexed cohortHash,
        address indexed updatedBy,
        bytes4 newPermission,
        uint32 newModifiers
    );

    event ConsentRevoked(
        bytes32 indexed cohortHash,
        address indexed revokedBy,
        uint256 accessGrantsRevoked
    );

    event OwnershipChanged(
        bytes32 indexed cohortHash,
        address indexed owner,
        bool added
    );

    event AccessGranted(
        bytes32 indexed cohortHash,
        address indexed requester,
        bytes4 grantedUse,
        uint64 expiresAt
    );

    event AccessDecision(
        bytes32 indexed cohortHash,
        address indexed requester,
        bool approved,
        uint8 reason,
        uint64 expiresAt
    );

    event AccessRevoked(
        bytes32 indexed cohortHash,
        address indexed requester,
        string reason
    );

    event MerkleRootsUpdated(
        bytes32 indexed cohortHash,
        bytes32 countriesRoot,
        bytes32 institutionsRoot
    );

    InstitutionRegistry public immutable institutionRegistry;
    IIBISVerifier public ibisVerifier;
    IRoleGroupRegistry public roleGroupRegistry;
    IRoleAccountFactory public roleAccountFactory;

    bytes4 internal constant ERC1271_MAGIC = 0x1626ba7e;

    constructor(
        address _ontology,
        address _attestationRegistry,
        address _institutionRegistry
    ) EIP712("DUOConsentVaultV2", "2") {
        require(_ontology != address(0), "Invalid ontology");
        require(_attestationRegistry != address(0), "Invalid attestation registry");
        require(_institutionRegistry != address(0), "Invalid institution registry");

        ontology = DUOOntology(_ontology);
        attestationRegistry = AttestationRegistry(_attestationRegistry);
        institutionRegistry = InstitutionRegistry(_institutionRegistry);

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(CONSENT_ADMIN, msg.sender);
        _grantRole(RELAYER_ROLE, msg.sender);
    }

    function setIBISVerifier(address verifier) external onlyRole(DEFAULT_ADMIN_ROLE) {
        ibisVerifier = IIBISVerifier(verifier);
    }

    function setRoleInfra(address registry, address factory) external onlyRole(DEFAULT_ADMIN_ROLE) {
        roleGroupRegistry = IRoleGroupRegistry(registry);
        roleAccountFactory = IRoleAccountFactory(factory);
    }

    function _resolvePrincipal(address signerEOA, bytes32 roleId, bytes32 digest, bytes calldata signature)
        internal
        view
        returns (address)
    {
        require(address(roleGroupRegistry) != address(0), "registry not set");
        require(address(roleAccountFactory) != address(0), "factory not set");

        bytes32 commitment = roleGroupRegistry.commitmentOf(roleId, signerEOA);
        require(commitment != bytes32(0), "role not activated");

        address principal = roleAccountFactory.computeAddress(signerEOA, commitment);
        require(principal.code.length > 0, "role account not deployed");
        require(
            IERC1271(principal).isValidSignature(digest, signature) == ERC1271_MAGIC,
            "ERC1271 reject"
        );
        return principal;
    }

    struct RecordArgs {
        bytes32 cohortHash;
        bytes4 permission;
        uint32 modifiers;
        bytes32[] diseaseCodes;
        bytes32 metadataHash;
        uint256 countryBitset;
        uint256 validDays;
        uint16 moratoriumMonths;
        uint16 publicationDeadlineDays;
        bytes32 institutionsRoot;
    }

    function recordConsent(
        RecordArgs calldata args,
        bytes32[] calldata institutionIds,
        bytes32[] calldata projectIds,
        address[] calldata userAddresses
    ) external returns (bool) {
        return _recordConsentInternal(args, msg.sender, institutionIds, projectIds, userAddresses, false);
    }

    function recordConsentWithSignature(
        RecordArgs calldata args,
        uint256 nonce,
        bytes32[] calldata institutionIds,
        bytes32[] calldata projectIds,
        address[] calldata userAddresses,
        bytes calldata signature
    ) external onlyRole(RELAYER_ROLE) returns (bool) {
        bytes32 structHash = keccak256(abi.encode(
            CONSENT_TYPEHASH,
            args.cohortHash,
            args.permission,
            uint256(args.modifiers),
            keccak256(abi.encodePacked(args.diseaseCodes)),
            args.metadataHash,
            args.countryBitset,
            args.validDays,
            uint256(args.moratoriumMonths),
            uint256(args.publicationDeadlineDays),
            args.institutionsRoot,
            keccak256(abi.encodePacked(institutionIds)),
            keccak256(abi.encodePacked(projectIds)),
            keccak256(abi.encodePacked(userAddresses)),
            nonce
        ));

        bytes32 digest = _hashTypedDataV4(structHash);
        address signerEOA = ECDSA.recover(digest, signature);
        require(signerEOA != address(0), "Invalid signature");
        require(nonces[signerEOA] == nonce, "Bad nonce");
        nonces[signerEOA] = nonce + 1;

        address principal = _resolvePrincipal(
            signerEOA,
            roleGroupRegistry.ROLE_PROVIDER(),
            digest,
            signature
        );
        return _recordConsentInternal(args, principal, institutionIds, projectIds, userAddresses, true);
    }

    function _recordConsentInternal(
        RecordArgs calldata args,
        address owner,
        bytes32[] calldata institutionIds,
        bytes32[] calldata projectIds,
        address[] calldata userAddresses,
        bool viaSig
    ) internal returns (bool) {
        require(args.cohortHash != bytes32(0), "cohort=0");
        require(ontology.validPermission(args.permission), "bad permission");

        uint32 mods = args.modifiers;
        if ((mods & ontology.MOD_GS()) != 0) {
            require(args.countryBitset != 0, "GS needs countryBitset");
        }
        if ((mods & ontology.MOD_IS()) != 0) {
            require(institutionIds.length > 0 || args.institutionsRoot != bytes32(0), "IS needs institutions");
        }
        if ((mods & ontology.MOD_PS()) != 0) {
            require(projectIds.length > 0, "PS needs projects");
        }
        if ((mods & ontology.MOD_US()) != 0) {
            require(userAddresses.length > 0, "US needs users");
        }
        if ((mods & ontology.MOD_MOR()) != 0) {
            require(args.moratoriumMonths > 0, "MOR needs months");
        }
        if ((mods & ontology.MOD_TS()) != 0) {
            require(args.validDays > 0, "TS needs validDays");
        }
        require(args.diseaseCodes.length <= ontology.MAX_DISEASE_CODES(), "too many disease codes");
        for (uint256 i = 0; i < args.diseaseCodes.length; i++) {
            require(args.diseaseCodes[i] != bytes32(0), "zero disease code");
        }
        if (args.permission == ontology.DS()) {
            require(args.diseaseCodes.length > 0, "DS needs disease");
        }

        bool isNew = consents[args.cohortHash].validFrom == 0;
        if (!isNew) {
            require(isOwner[args.cohortHash][owner], "not owner");
        }

        uint64 validFrom = uint64(block.timestamp);
        uint64 validUntil = args.validDays > 0
            ? uint64(block.timestamp + (args.validDays * 1 days))
            : 0;

        delete _consentDiseases[args.cohortHash];
        for (uint256 i = 0; i < args.diseaseCodes.length; i++) {
            _consentDiseases[args.cohortHash].push(args.diseaseCodes[i]);
        }

        consents[args.cohortHash] = ConsentCore({
            cohortHash: args.cohortHash,
            permission: args.permission,
            modifiersLow: args.modifiers,
            validFrom: validFrom,
            validUntil: validUntil,
            ownerCount: isNew ? 1 : consents[args.cohortHash].ownerCount,
            active: true,
            flags: 0,
            moratoriumMonths: args.moratoriumMonths,
            publicationDeadlineDays: args.publicationDeadlineDays,
            metadataHash: args.metadataHash
        });

        merkleRoots[args.cohortHash] = MerkleRoots({
            countriesRoot: bytes32(0),
            institutionsRoot: args.institutionsRoot
        });

        countryBitset[args.cohortHash] = args.countryBitset;
        for (uint256 i = 0; i < institutionIds.length; i++) {
            allowedInstitutions[args.cohortHash][institutionIds[i]] = true;
        }
        for (uint256 i = 0; i < projectIds.length; i++) {
            allowedProjects[args.cohortHash][projectIds[i]] = true;
        }
        for (uint256 i = 0; i < userAddresses.length; i++) {
            allowedUsers[args.cohortHash][userAddresses[i]] = true;
        }

        if (isNew) {
            _owners[args.cohortHash].push(owner);
            isOwner[args.cohortHash][owner] = true;
            cohortIndex[args.cohortHash] = allCohorts.length;
            allCohorts.push(args.cohortHash);
        }

        emit ConsentRecorded(args.cohortHash, owner, validUntil, viaSig);
        return true;
    }

    function updateMerkleRoots(
        bytes32 cohortHash,
        bytes32 newCountriesRoot,
        bytes32 newInstitutionsRoot
    ) external {
        require(isOwner[cohortHash][msg.sender], "Not an owner");

        merkleRoots[cohortHash] = MerkleRoots({
            countriesRoot: newCountriesRoot,
            institutionsRoot: newInstitutionsRoot
        });

        emit MerkleRootsUpdated(cohortHash, newCountriesRoot, newInstitutionsRoot);
    }

    function revokeConsent(bytes32 cohortHash) external {
        require(
            isOwner[cohortHash][msg.sender] || hasRole(CONSENT_ADMIN, msg.sender),
            "only consent owner"
        );

        consents[cohortHash].active = false;
        emit ConsentRevoked(cohortHash, msg.sender, 0);
    }

    function revokeConsentWithSignature(
        bytes32 cohortHash,
        uint256 nonce,
        bytes calldata signature
    ) external onlyRole(RELAYER_ROLE) {
        bytes32 structHash = keccak256(abi.encode(REVOKE_TYPEHASH, cohortHash, nonce));
        bytes32 digest = _hashTypedDataV4(structHash);
        address signerEOA = ECDSA.recover(digest, signature);
        require(signerEOA != address(0), "Invalid signature");
        require(nonces[signerEOA] == nonce, "Bad nonce");
        nonces[signerEOA] = nonce + 1;

        address principal = _resolvePrincipal(
            signerEOA,
            roleGroupRegistry.ROLE_PROVIDER(),
            digest,
            signature
        );
        require(isOwner[cohortHash][principal], "only consent owner");

        consents[cohortHash].active = false;
        emit ConsentRevoked(cohortHash, principal, 0);
    }

    function addOwner(bytes32 cohortHash, address newOwner) external {
        require(isOwner[cohortHash][msg.sender], "Not an owner");
        require(newOwner != address(0), "Invalid owner");
        require(!isOwner[cohortHash][newOwner], "Already an owner");

        _owners[cohortHash].push(newOwner);
        isOwner[cohortHash][newOwner] = true;
        consents[cohortHash].ownerCount++;

        emit OwnershipChanged(cohortHash, newOwner, true);
    }

    function removeOwner(bytes32 cohortHash, address ownerToRemove) external {
        require(isOwner[cohortHash][msg.sender], "Not an owner");
        require(consents[cohortHash].ownerCount > 1, "Cannot remove last owner");

        address[] storage owners = _owners[cohortHash];
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == ownerToRemove) {
                owners[i] = owners[owners.length - 1];
                owners.pop();
                isOwner[cohortHash][ownerToRemove] = false;
                consents[cohortHash].ownerCount--;
                emit OwnershipChanged(cohortHash, ownerToRemove, false);
                return;
            }
        }
        revert("Owner not found");
    }

    struct AccessArgs {
        bytes32 cohortHash;
        address requester;
        bytes4 intendedUse;
        uint8 purpose;
        bytes32[] diseaseCodes;
        bytes32 projectId;
        uint8 countryIndex;
        bytes32 institutionId;
    }

    function requestAccess(AccessArgs calldata args) external nonReentrant returns (bool approved) {
        require(args.requester == msg.sender, "requester != sender");
        return _requestAccessInternal(args);
    }

    function requestAccessWithSignature(
        AccessArgs calldata args,
        uint256 nonce,
        bytes calldata signature
    ) external nonReentrant onlyRole(RELAYER_ROLE) returns (bool approved) {
        bytes32 structHash = keccak256(abi.encode(
            ACCESS_TYPEHASH,
            args.cohortHash,
            args.requester,
            args.intendedUse,
            args.purpose,
            keccak256(abi.encodePacked(args.diseaseCodes)),
            args.projectId,
            args.countryIndex,
            args.institutionId,
            nonce
        ));
        bytes32 digest = _hashTypedDataV4(structHash);
        address signerEOA = ECDSA.recover(digest, signature);
        require(signerEOA != address(0), "Invalid signature");
        require(nonces[signerEOA] == nonce, "Bad nonce");
        nonces[signerEOA] = nonce + 1;

        address principal = _resolvePrincipal(
            signerEOA,
            roleGroupRegistry.ROLE_REQUESTER(),
            digest,
            signature
        );
        require(principal == args.requester, "args.requester != role account");
        return _requestAccessInternal(args);
    }

    function _requestAccessInternal(AccessArgs calldata args) internal returns (bool) {
        ConsentCore storage consent = consents[args.cohortHash];
        require(consent.validFrom != 0, "unknown cohort");
        require(args.diseaseCodes.length <= ontology.MAX_DISEASE_CODES(), "too many disease codes");

        uint8 reason = _evaluate(consent, args);
        bool approved = (reason == R_OK);
        uint64 expiresAt = approved ? consent.validUntil : 0;

        accessGrants[args.cohortHash][args.requester] = AccessGrant({
            grantedAt: uint64(block.timestamp),
            expiresAt: expiresAt,
            grantedUse: args.intendedUse,
            status: approved ? STATUS_APPROVED : STATUS_REJECTED,
            flags: 0,
            reason: reason
        });

        if (approved) {
            emit AccessGranted(args.cohortHash, args.requester, args.intendedUse, expiresAt);
        }
        emit AccessDecision(args.cohortHash, args.requester, approved, reason, expiresAt);
        return approved;
    }

    function _evaluate(ConsentCore storage consent, AccessArgs calldata args) internal view returns (uint8) {
        if (!consent.active) return R_CONSENT_INACTIVE;
        if (!_isConsentValid(consent)) return R_CONSENT_EXPIRED;
        if (!ontology.isPermissionCompatible(consent.permission, args.intendedUse)) return R_PERM;

        if (consent.permission == ontology.DS() || args.intendedUse == ontology.DS()) {
            if (!ontology.areDiseasesCompatible(_consentDiseases[args.cohortHash], args.diseaseCodes)) {
                return R_DISEASE;
            }
        }

        uint8 cr = _evaluateConstraints(consent.modifiersLow, args);
        if (cr != R_OK) return cr;

        return _evaluateAttestations(args.cohortHash, args.requester, consent.modifiersLow);
    }

    function _evaluateConstraints(uint32 mods, AccessArgs calldata args) internal view returns (uint8) {
        bytes4 useCode = args.intendedUse;
        bytes32 c = args.cohortHash;
        uint8 purpose = args.purpose;

        address requesterEOA = args.requester;
        if (address(roleGroupRegistry) != address(0) && args.requester.code.length > 0) {
            (bool ok, bytes memory ret) = args.requester.staticcall(
                abi.encodeWithSignature("roleCommitment()")
            );
            if (ok && ret.length == 32) {
                bytes32 cmt = abi.decode(ret, (bytes32));
                address resolved = roleGroupRegistry.ownerOf(
                    roleGroupRegistry.ROLE_REQUESTER(), cmt
                );
                if (resolved != address(0)) requesterEOA = resolved;
            }
        }

        if ((mods & ontology.MOD_NPOA()) != 0) {
            if (useCode == ontology.POA() || purpose == ontology.PURPOSE_POPULATION()) return R_NPOA;
        }
        if ((mods & ontology.MOD_GSO()) != 0) {
            if (purpose != ontology.PURPOSE_GENETICS()) return R_GSO;
        }
        if ((mods & ontology.MOD_NMDS()) != 0) {
            if (purpose == ontology.PURPOSE_METHODS()) return R_NMDS;
        }
        if ((mods & ontology.MOD_GS()) != 0) {
            if (((countryBitset[c] >> args.countryIndex) & 1) != 1) return R_GS;
        }
        if ((mods & ontology.MOD_IS()) != 0) {
            if (!allowedInstitutions[c][args.institutionId]) return R_IS;
        }
        if ((mods & ontology.MOD_PS()) != 0) {
            if (!allowedProjects[c][args.projectId]) return R_PS;
        }
        if ((mods & ontology.MOD_US()) != 0) {
            if (!(allowedUsers[c][requesterEOA] || allowedUsers[c][args.requester])) return R_US;
        }
        if ((mods & ontology.MOD_NPU()) != 0) {
            uint8 t = institutionRegistry.getRequesterType(requesterEOA);
            if (!(t == ontology.REQ_ACADEMIC() || t == ontology.REQ_NONPROFIT() || t == ontology.REQ_GOVERNMENT())) return R_NPU;
        }
        if ((mods & ontology.MOD_NCU()) != 0) {
            uint8 t = institutionRegistry.getRequesterType(requesterEOA);
            if (t == ontology.REQ_COMMERCIAL()) return R_NCU;
        }
        if ((mods & ontology.MOD_NPUNCU()) != 0) {
            uint8 t = institutionRegistry.getRequesterType(requesterEOA);
            if (!(t == ontology.REQ_ACADEMIC() || t == ontology.REQ_NONPROFIT() || t == ontology.REQ_GOVERNMENT())) return R_NPU;
        }
        return R_OK;
    }

    function _evaluateAttestations(
        bytes32 cohortHash,
        address requester,
        uint32 modifiers
    ) internal view returns (uint8) {

        if ((modifiers & uint32(ontology.MOD_COL())) != 0) {
            (bool hasCol, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_COL(),
                cohortHash
            );
            if (!hasCol) return R_COL;
        }

        if ((modifiers & uint32(ontology.MOD_PUB())) != 0) {
            (bool hasPub, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_PUB(),
                cohortHash
            );
            if (!hasPub) return R_PUB;
        }

        if ((modifiers & uint32(ontology.MOD_RTN())) != 0) {
            (bool hasRtn, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_RTN(),
                cohortHash
            );
            if (!hasRtn) return R_RTN;
        }

        return R_OK;
    }

    function hasAccess(bytes32 cohortHash, address requester) external view returns (bool) {
        ConsentCore storage consent = consents[cohortHash];
        if (!consent.active) return false;
        if (!_isConsentValid(consent)) return false;

        AccessGrant storage grant = accessGrants[cohortHash][requester];
        if (grant.status != STATUS_APPROVED) return false;
        if (grant.expiresAt > 0 && block.timestamp > grant.expiresAt) return false;

        return true;
    }

    function getConsent(bytes32 cohortHash) external view returns (
        address primaryOwner,
        bytes4 permission,
        uint32 modifiers,
        bytes32[] memory diseaseCodes,
        uint64 validFrom,
        uint64 validUntil,
        bool active,
        bytes32 countriesRoot,
        bytes32 institutionsRoot
    ) {
        ConsentCore storage c = consents[cohortHash];
        MerkleRoots storage m = merkleRoots[cohortHash];
        address owner = _owners[cohortHash].length > 0 ? _owners[cohortHash][0] : address(0);

        return (
            owner,
            c.permission,
            c.modifiersLow,
            _consentDiseases[cohortHash],
            c.validFrom,
            c.validUntil,
            c.active,
            m.countriesRoot,
            m.institutionsRoot
        );
    }

    function getConsentDiseases(bytes32 cohortHash) external view returns (bytes32[] memory) {
        return _consentDiseases[cohortHash];
    }

    function getOwners(bytes32 cohortHash) external view returns (address[] memory) {
        return _owners[cohortHash];
    }

    function getAccessGrant(
        bytes32 cohortHash,
        address requester
    ) external view returns (AccessGrant memory) {
        return accessGrants[cohortHash][requester];
    }

    function getNonce(address account) external view returns (uint256) {
        return nonces[account];
    }

    function getDomainSeparator() external view returns (bytes32) {
        return _domainSeparatorV4();
    }

    function getTotalCohorts() external view returns (uint256) {
        return allCohorts.length;
    }

    function getCohortAt(uint256 index) external view returns (bytes32) {
        require(index < allCohorts.length, "Index out of bounds");
        return allCohorts[index];
    }

    function _isConsentValid(ConsentCore storage consent) internal view returns (bool) {
        if (consent.validUntil > 0 && block.timestamp > consent.validUntil) {
            return false;
        }
        return true;
    }

    function verifyCountry(
        bytes32 cohortHash,
        bytes2 countryCode,
        bytes32[] calldata proof
    ) external view returns (bool) {
        bytes32 leaf = keccak256(abi.encodePacked(countryCode));
        return MerkleProof.verify(proof, merkleRoots[cohortHash].countriesRoot, leaf);
    }

    function verifyInstitution(
        bytes32 cohortHash,
        bytes32 institutionId,
        bytes32[] calldata proof
    ) external view returns (bool) {
        bytes32 leaf = keccak256(abi.encodePacked(institutionId));
        return MerkleProof.verify(proof, merkleRoots[cohortHash].institutionsRoot, leaf);
    }

    function countryLeaf(bytes2 countryCode) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(countryCode));
    }

    function institutionLeaf(bytes32 institutionId) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(institutionId));
    }
}
