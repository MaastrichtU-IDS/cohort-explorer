// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "./DUOOntology.sol";
import "./AttestationRegistry.sol";

interface IERC5192 {

    event Locked(uint256 tokenId);
    event Unlocked(uint256 tokenId);

    function locked(uint256 tokenId) external view returns (bool);
}

contract DUOConsentToken is
    ERC721,
    ERC721URIStorage,
    IERC5192,
    AccessControl,
    ReentrancyGuard,
    EIP712
{
    using ECDSA for bytes32;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant RELAYER_ROLE = keccak256("RELAYER_ROLE");

    bytes32 public constant MINT_CONSENT_TYPEHASH = keccak256(
        "MintConsent(bytes32 cohortHash,bytes4 permission,uint256 modifiers,bytes32 diseaseCode,bytes32 countriesMerkle,bytes32 institutionsMerkle,uint64 validUntil,uint256 nonce)"
    );

    bytes32 public constant BURN_CONSENT_TYPEHASH = keccak256(
        "BurnConsent(uint256 tokenId,uint256 nonce)"
    );

    DUOOntology public immutable ontology;
    AttestationRegistry public immutable attestationRegistry;

    struct ConsentData {
        bytes32 cohortHash;
        bytes4 permission;
        uint256 modifiers;
        bytes32 diseaseCode;
        bytes32 countriesMerkle;
        bytes32 institutionsMerkle;
        uint64 validFrom;
        uint64 validUntil;
        uint16 ownerCount;
        bool active;
    }

    struct ComplianceScore {
        uint16 totalScore;
        uint16 permissionScore;
        uint16 modifierScore;
        uint16 attestationScore;
        uint16 trustScore;
        RiskLevel riskLevel;
    }

    enum RiskLevel {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }

    mapping(uint256 => ConsentData) public consents;

    mapping(uint256 => address[]) internal _additionalOwners;

    mapping(uint256 => mapping(address => bool)) public isTokenOwner;

    mapping(address => uint256) public nonces;

    mapping(uint256 => address) public tokenBoundAccounts;

    uint256 public totalMinted;

    event ConsentMinted(
        uint256 indexed tokenId,
        address indexed owner,
        bytes32 cohortHash,
        bytes4 permission,
        uint256 modifiers,
        bool viaSig
    );

    event ConsentUpdated(
        uint256 indexed tokenId,
        bytes4 newPermission,
        uint256 newModifiers
    );

    event ConsentDeactivated(
        uint256 indexed tokenId,
        address indexed deactivatedBy
    );

    event ConsentReactivated(
        uint256 indexed tokenId,
        address indexed reactivatedBy
    );

    event OwnerAdded(
        uint256 indexed tokenId,
        address indexed newOwner
    );

    event OwnerRemoved(
        uint256 indexed tokenId,
        address indexed removedOwner
    );

    event TokenBoundAccountCreated(
        uint256 indexed tokenId,
        address indexed tba
    );

    event ComplianceScoreCalculated(
        uint256 indexed tokenId,
        address indexed requester,
        uint16 totalScore,
        RiskLevel riskLevel
    );

    constructor(
        address _ontology,
        address _attestationRegistry
    )
        ERC721("DUO Consent Token", "DUO-CONSENT")
        EIP712("DUOConsentToken", "1")
    {
        require(_ontology != address(0), "Invalid ontology");
        require(_attestationRegistry != address(0), "Invalid attestation registry");

        ontology = DUOOntology(_ontology);
        attestationRegistry = AttestationRegistry(_attestationRegistry);

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(RELAYER_ROLE, msg.sender);
    }

    function locked(uint256 tokenId) external view override returns (bool) {

        if (_ownerOf(tokenId) == address(0)) {
            return false;
        }
        return true;
    }

    function _update(
        address to,
        uint256 tokenId,
        address auth
    ) internal virtual override(ERC721) returns (address) {
        address from = _ownerOf(tokenId);

        if (from != address(0) && to != address(0)) {
            revert("Soulbound: non-transferable");
        }

        return super._update(to, tokenId, auth);
    }

    function mintConsent(
        address owner,
        bytes32 cohortHash,
        bytes4 permission,
        uint256 modifiers,
        bytes32 diseaseCode,
        bytes32 countriesMerkle,
        bytes32 institutionsMerkle,
        uint256 validDays,
        string calldata metadataURI
    ) external onlyRole(MINTER_ROLE) returns (uint256 tokenId) {
        return _mintConsentInternal(
            owner,
            cohortHash,
            permission,
            modifiers,
            diseaseCode,
            countriesMerkle,
            institutionsMerkle,
            validDays,
            metadataURI,
            false
        );
    }

    function mintConsentWithSignature(
        address owner,
        bytes32 cohortHash,
        bytes4 permission,
        uint256 modifiers,
        bytes32 diseaseCode,
        bytes32 countriesMerkle,
        bytes32 institutionsMerkle,
        uint256 validDays,
        string calldata metadataURI,
        bytes calldata signature
    ) external onlyRole(RELAYER_ROLE) returns (uint256 tokenId) {

        uint64 validUntil = validDays > 0
            ? uint64(block.timestamp + (validDays * 1 days))
            : 0;

        bytes32 structHash = keccak256(abi.encode(
            MINT_CONSENT_TYPEHASH,
            cohortHash,
            permission,
            modifiers,
            diseaseCode,
            countriesMerkle,
            institutionsMerkle,
            validUntil,
            nonces[owner]
        ));

        bytes32 digest = _hashTypedDataV4(structHash);
        address signer = ECDSA.recover(digest, signature);

        require(signer != address(0), "Invalid signature");
        require(signer == owner, "Signer must be owner");

        nonces[owner]++;

        return _mintConsentInternal(
            owner,
            cohortHash,
            permission,
            modifiers,
            diseaseCode,
            countriesMerkle,
            institutionsMerkle,
            validDays,
            metadataURI,
            true
        );
    }

    function _mintConsentInternal(
        address owner,
        bytes32 cohortHash,
        bytes4 permission,
        uint256 modifiers,
        bytes32 diseaseCode,
        bytes32 countriesMerkle,
        bytes32 institutionsMerkle,
        uint256 validDays,
        string calldata metadataURI,
        bool viaSig
    ) internal returns (uint256 tokenId) {
        require(owner != address(0), "Invalid owner");
        require(cohortHash != bytes32(0), "Invalid cohort hash");
        require(ontology.validPermission(permission), "Invalid permission");

        tokenId = uint256(cohortHash);

        require(_ownerOf(tokenId) == address(0), "Already minted");

        if ((modifiers & uint256(ontology.MOD_GS())) != 0) {
            require(countriesMerkle != bytes32(0), "GS requires countries merkle root");
        }
        if ((modifiers & uint256(ontology.MOD_IS())) != 0) {
            require(institutionsMerkle != bytes32(0), "IS requires institutions merkle root");
        }
        if (permission == ontology.DS()) {
            require(diseaseCode != bytes32(0), "DS requires disease code");
        }

        uint64 validFrom = uint64(block.timestamp);
        uint64 validUntil = validDays > 0
            ? uint64(block.timestamp + (validDays * 1 days))
            : 0;

        consents[tokenId] = ConsentData({
            cohortHash: cohortHash,
            permission: permission,
            modifiers: modifiers,
            diseaseCode: diseaseCode,
            countriesMerkle: countriesMerkle,
            institutionsMerkle: institutionsMerkle,
            validFrom: validFrom,
            validUntil: validUntil,
            ownerCount: 1,
            active: true
        });

        isTokenOwner[tokenId][owner] = true;

        _safeMint(owner, tokenId);

        if (bytes(metadataURI).length > 0) {
            _setTokenURI(tokenId, metadataURI);
        }

        emit Locked(tokenId);

        totalMinted++;

        emit ConsentMinted(
            tokenId,
            owner,
            cohortHash,
            permission,
            modifiers,
            viaSig
        );

        return tokenId;
    }

    function updateConsent(
        uint256 tokenId,
        bytes4 newPermission,
        uint256 newModifiers,
        bytes32 newDiseaseCode,
        bytes32 newCountriesMerkle,
        bytes32 newInstitutionsMerkle
    ) external {
        require(isTokenOwner[tokenId][msg.sender], "Not an owner");
        require(ontology.validPermission(newPermission), "Invalid permission");

        ConsentData storage consent = consents[tokenId];

        if ((newModifiers & uint256(ontology.MOD_GS())) != 0) {
            require(newCountriesMerkle != bytes32(0), "GS requires countries merkle root");
        }
        if ((newModifiers & uint256(ontology.MOD_IS())) != 0) {
            require(newInstitutionsMerkle != bytes32(0), "IS requires institutions merkle root");
        }
        if (newPermission == ontology.DS()) {
            require(newDiseaseCode != bytes32(0), "DS requires disease code");
        }

        consent.permission = newPermission;
        consent.modifiers = newModifiers;
        consent.diseaseCode = newDiseaseCode;
        consent.countriesMerkle = newCountriesMerkle;
        consent.institutionsMerkle = newInstitutionsMerkle;

        emit ConsentUpdated(tokenId, newPermission, newModifiers);
    }

    function deactivateConsent(uint256 tokenId) external {
        require(isTokenOwner[tokenId][msg.sender], "Not an owner");
        require(consents[tokenId].active, "Already inactive");

        consents[tokenId].active = false;
        emit ConsentDeactivated(tokenId, msg.sender);
    }

    function reactivateConsent(uint256 tokenId) external {
        require(isTokenOwner[tokenId][msg.sender], "Not an owner");
        require(!consents[tokenId].active, "Already active");

        consents[tokenId].active = true;
        emit ConsentReactivated(tokenId, msg.sender);
    }

    function burnConsent(uint256 tokenId) external {
        require(isTokenOwner[tokenId][msg.sender], "Not an owner");

        delete consents[tokenId];

        address[] storage additionalOwners = _additionalOwners[tokenId];
        for (uint256 i = 0; i < additionalOwners.length; i++) {
            isTokenOwner[tokenId][additionalOwners[i]] = false;
        }
        delete _additionalOwners[tokenId];
        isTokenOwner[tokenId][ownerOf(tokenId)] = false;

        _burn(tokenId);

        emit Unlocked(tokenId);
    }

    function addOwner(uint256 tokenId, address newOwner) external {
        require(isTokenOwner[tokenId][msg.sender], "Not an owner");
        require(newOwner != address(0), "Invalid owner");
        require(!isTokenOwner[tokenId][newOwner], "Already an owner");

        _additionalOwners[tokenId].push(newOwner);
        isTokenOwner[tokenId][newOwner] = true;
        consents[tokenId].ownerCount++;

        emit OwnerAdded(tokenId, newOwner);
    }

    function removeOwner(uint256 tokenId, address ownerToRemove) external {
        require(isTokenOwner[tokenId][msg.sender], "Not an owner");
        require(ownerToRemove != ownerOf(tokenId), "Cannot remove primary owner");
        require(isTokenOwner[tokenId][ownerToRemove], "Not an owner");

        address[] storage owners = _additionalOwners[tokenId];
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == ownerToRemove) {
                owners[i] = owners[owners.length - 1];
                owners.pop();
                isTokenOwner[tokenId][ownerToRemove] = false;
                consents[tokenId].ownerCount--;
                emit OwnerRemoved(tokenId, ownerToRemove);
                return;
            }
        }
        revert("Owner not found in additional owners");
    }

    function calculateComplianceScore(
        uint256 tokenId,
        address requester,
        bytes4 intendedUse,
        uint8 requesterType
    ) external view returns (ComplianceScore memory score) {
        ConsentData storage consent = consents[tokenId];

        score.permissionScore = _calculatePermissionScore(
            consent.permission,
            intendedUse,
            consent.diseaseCode
        );

        score.modifierScore = _calculateModifierScore(
            tokenId,
            consent.modifiers,
            requester,
            requesterType
        );

        score.attestationScore = _calculateAttestationScore(
            tokenId,
            consent.modifiers,
            requester
        );

        score.trustScore = _calculateTrustScore(requester);

        score.totalScore = score.permissionScore +
            score.modifierScore +
            score.attestationScore +
            score.trustScore;

        if (score.totalScore >= 800) {
            score.riskLevel = RiskLevel.LOW;
        } else if (score.totalScore >= 600) {
            score.riskLevel = RiskLevel.MEDIUM;
        } else if (score.totalScore >= 400) {
            score.riskLevel = RiskLevel.HIGH;
        } else {
            score.riskLevel = RiskLevel.CRITICAL;
        }

        if (score.permissionScore == 0) {
            score.riskLevel = RiskLevel.CRITICAL;
        }

        return score;
    }

    function _calculatePermissionScore(
        bytes4 consentPerm,
        bytes4 requestPerm,
        bytes32
    ) internal view returns (uint16) {

        if (consentPerm == requestPerm) {
            return 300;
        }

        if (consentPerm == ontology.NRES()) {
            return 285;
        }

        if (ontology.isPermissionCompatible(consentPerm, requestPerm)) {

            return 270;
        }

        return 0;
    }

    function _calculateModifierScore(
        uint256 tokenId,
        uint256 modifiers,
        address requester,
        uint8 requesterType
    ) internal view returns (uint16) {
        if (modifiers == 0) {
            return 400;
        }

        uint16 totalWeight = 0;
        uint16 achievedWeight = 0;

        if ((modifiers & uint256(ontology.MOD_NPU())) != 0) {
            totalWeight += 8;

            if (requesterType >= 1 && requesterType <= 3) {
                achievedWeight += 8;
            }
        }

        if ((modifiers & uint256(ontology.MOD_NCU())) != 0) {
            totalWeight += 7;
            if (requesterType >= 1 && requesterType <= 3) {
                achievedWeight += 7;
            }
        }

        if ((modifiers & uint256(ontology.MOD_IRB())) != 0) {
            totalWeight += 10;
            (bool hasIRB, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_IRB(),
                consents[tokenId].cohortHash
            );
            if (hasIRB) {
                achievedWeight += 10;
            }
        }

        if ((modifiers & uint256(ontology.MOD_COL())) != 0) {
            totalWeight += 9;
            if (attestationRegistry.hasCollaboration(
                consents[tokenId].cohortHash,
                requester
            )) {
                achievedWeight += 9;
            }
        }

        if ((modifiers & uint256(ontology.MOD_PUB())) != 0) {
            totalWeight += 5;
            (bool hasPub, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_PUB(),
                consents[tokenId].cohortHash
            );
            if (hasPub) {
                achievedWeight += 5;
            }
        }

        if ((modifiers & uint256(ontology.MOD_TS())) != 0) {
            totalWeight += 6;

            if (consents[tokenId].validUntil == 0 ||
                block.timestamp <= consents[tokenId].validUntil) {
                achievedWeight += 6;
            }
        }

        if (totalWeight == 0) {
            return 400;
        }

        return uint16((uint256(achievedWeight) * 400) / uint256(totalWeight));
    }

    function _calculateAttestationScore(
        uint256 tokenId,
        uint256 modifiers,
        address requester
    ) internal view returns (uint16) {
        uint16 score = 0;
        bytes32 cohortHash = consents[tokenId].cohortHash;

        if ((modifiers & uint256(ontology.MOD_IRB())) != 0) {
            (bool hasIRB, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_IRB(),
                cohortHash
            );
            if (hasIRB) {
                score += 70;
            }
        }

        if ((modifiers & uint256(ontology.MOD_COL())) != 0) {
            if (attestationRegistry.hasCollaboration(cohortHash, requester)) {
                score += 50;
            }
        }

        if ((modifiers & uint256(ontology.MOD_PUB())) != 0) {
            (bool hasPub, ) = attestationRegistry.hasValidAttestation(
                requester,
                attestationRegistry.ATT_PUB(),
                cohortHash
            );
            if (hasPub) {
                score += 40;
            }
        }

        if (modifiers == 0 ||
            ((modifiers & uint256(ontology.MOD_IRB())) == 0 &&
             (modifiers & uint256(ontology.MOD_COL())) == 0 &&
             (modifiers & uint256(ontology.MOD_PUB())) == 0)) {
            score = 200;
        }

        return score > 200 ? 200 : score;
    }

    function _calculateTrustScore(
        address
    ) internal pure returns (uint16) {

        return 50;
    }

    function getConsentData(uint256 tokenId) external view returns (ConsentData memory) {
        return consents[tokenId];
    }

    function isConsentValid(uint256 tokenId) external view returns (bool) {
        ConsentData storage consent = consents[tokenId];
        if (!consent.active) return false;
        if (consent.validUntil > 0 && block.timestamp > consent.validUntil) return false;
        return true;
    }

    function getOwners(uint256 tokenId) external view returns (address[] memory) {
        address primaryOwner = ownerOf(tokenId);
        address[] storage additionalOwners = _additionalOwners[tokenId];

        address[] memory allOwners = new address[](additionalOwners.length + 1);
        allOwners[0] = primaryOwner;

        for (uint256 i = 0; i < additionalOwners.length; i++) {
            allOwners[i + 1] = additionalOwners[i];
        }

        return allOwners;
    }

    function getNonce(address account) external view returns (uint256) {
        return nonces[account];
    }

    function getDomainSeparator() external view returns (bytes32) {
        return _domainSeparatorV4();
    }

    function getTokenIdFromCohort(bytes32 cohortHash) external pure returns (uint256) {
        return uint256(cohortHash);
    }

    function consentExists(bytes32 cohortHash) external view returns (bool) {
        return _ownerOf(uint256(cohortHash)) != address(0);
    }

    function setTokenBoundAccount(
        uint256 tokenId,
        address tba
    ) external onlyRole(MINTER_ROLE) {
        require(_ownerOf(tokenId) != address(0), "Token does not exist");
        require(tokenBoundAccounts[tokenId] == address(0), "TBA already set");

        tokenBoundAccounts[tokenId] = tba;
        emit TokenBoundAccountCreated(tokenId, tba);
    }

    function getTokenBoundAccount(uint256 tokenId) external view returns (address) {
        return tokenBoundAccounts[tokenId];
    }

    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721URIStorage, AccessControl)
        returns (bool)
    {

        return interfaceId == type(IERC5192).interfaceId ||
            super.supportsInterface(interfaceId);
    }
}
