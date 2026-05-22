// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./DUOConsentToken.sol";

contract AccessCredentialNFT is
    ERC721,
    ERC721URIStorage,
    AccessControl,
    ReentrancyGuard
{

    bytes32 public constant ISSUER_ROLE = keccak256("ISSUER_ROLE");

    struct Credential {
        uint256 consentTokenId;
        address consentTBA;
        address issuer;
        bytes4 grantedUse;
        uint8 grantedPurpose;
        uint64 grantedAt;
        uint64 expiresAt;
        uint16 complianceScore;
        DUOConsentToken.RiskLevel riskLevel;
        bool revoked;
        string conditions;
    }

    DUOConsentToken public immutable consentToken;

    mapping(uint256 => Credential) public credentials;

    mapping(uint256 => uint256[]) public credentialsByConsent;

    mapping(address => uint256[]) public credentialsByRequester;

    uint256 public totalCredentials;

    event CredentialIssued(
        uint256 indexed credentialId,
        uint256 indexed consentTokenId,
        address indexed requester,
        bytes4 grantedUse,
        uint16 complianceScore,
        uint64 expiresAt
    );

    event CredentialRevoked(
        uint256 indexed credentialId,
        address indexed revokedBy,
        string reason
    );

    event CredentialExpired(
        uint256 indexed credentialId
    );

    event CredentialTransferredForRevocation(
        uint256 indexed credentialId,
        address indexed from,
        address indexed to
    );

    constructor(
        address _consentToken
    ) ERC721("DUO Access Credential", "DUO-ACCESS") {
        require(_consentToken != address(0), "Invalid consent token");
        consentToken = DUOConsentToken(_consentToken);

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ISSUER_ROLE, msg.sender);
    }

    function _update(
        address to,
        uint256 tokenId,
        address auth
    ) internal virtual override(ERC721) returns (address) {
        address from = _ownerOf(tokenId);

        if (from == address(0) || to == address(0)) {
            return super._update(to, tokenId, auth);
        }

        Credential storage cred = credentials[tokenId];
        require(
            to == cred.issuer || to == cred.consentTBA,
            "Can only transfer to issuer for revocation"
        );

        emit CredentialTransferredForRevocation(tokenId, from, to);

        return super._update(to, tokenId, auth);
    }

    function issueCredential(
        address requester,
        uint256 consentTokenId,
        bytes4 grantedUse,
        uint8 grantedPurpose,
        uint16 complianceScore,
        DUOConsentToken.RiskLevel riskLevel,
        uint64 expiresAt,
        string calldata conditions,
        string calldata metadataURI
    ) external onlyRole(ISSUER_ROLE) returns (uint256 credentialId) {
        require(requester != address(0), "Invalid requester");

        DUOConsentToken.ConsentData memory consent = consentToken.getConsentData(consentTokenId);
        require(consent.active, "Consent not active");
        require(
            consent.validUntil == 0 || block.timestamp <= consent.validUntil,
            "Consent expired"
        );

        credentialId = ++totalCredentials;

        uint64 effectiveExpiry = expiresAt;
        if (effectiveExpiry == 0 && consent.validUntil > 0) {
            effectiveExpiry = consent.validUntil;
        }

        credentials[credentialId] = Credential({
            consentTokenId: consentTokenId,
            consentTBA: consentToken.getTokenBoundAccount(consentTokenId),
            issuer: msg.sender,
            grantedUse: grantedUse,
            grantedPurpose: grantedPurpose,
            grantedAt: uint64(block.timestamp),
            expiresAt: effectiveExpiry,
            complianceScore: complianceScore,
            riskLevel: riskLevel,
            revoked: false,
            conditions: conditions
        });

        credentialsByConsent[consentTokenId].push(credentialId);
        credentialsByRequester[requester].push(credentialId);

        _safeMint(requester, credentialId);

        if (bytes(metadataURI).length > 0) {
            _setTokenURI(credentialId, metadataURI);
        }

        emit CredentialIssued(
            credentialId,
            consentTokenId,
            requester,
            grantedUse,
            complianceScore,
            effectiveExpiry
        );

        return credentialId;
    }

    function issueCredentialFromTBA(
        address requester,
        uint256 consentTokenId,
        bytes4 grantedUse,
        uint8 grantedPurpose,
        uint16 complianceScore,
        DUOConsentToken.RiskLevel riskLevel,
        uint64 expiresAt,
        string calldata conditions,
        string calldata metadataURI
    ) external returns (uint256 credentialId) {

        address tba = consentToken.getTokenBoundAccount(consentTokenId);
        require(msg.sender == tba, "Caller is not the consent TBA");

        return _issueCredentialInternal(
            requester,
            consentTokenId,
            tba,
            grantedUse,
            grantedPurpose,
            complianceScore,
            riskLevel,
            expiresAt,
            conditions,
            metadataURI
        );
    }

    function _issueCredentialInternal(
        address requester,
        uint256 consentTokenId,
        address issuer,
        bytes4 grantedUse,
        uint8 grantedPurpose,
        uint16 complianceScore,
        DUOConsentToken.RiskLevel riskLevel,
        uint64 expiresAt,
        string calldata conditions,
        string calldata metadataURI
    ) internal returns (uint256 credentialId) {
        require(requester != address(0), "Invalid requester");

        DUOConsentToken.ConsentData memory consent = consentToken.getConsentData(consentTokenId);
        require(consent.active, "Consent not active");
        require(
            consent.validUntil == 0 || block.timestamp <= consent.validUntil,
            "Consent expired"
        );

        credentialId = ++totalCredentials;

        uint64 effectiveExpiry = expiresAt;
        if (effectiveExpiry == 0 && consent.validUntil > 0) {
            effectiveExpiry = consent.validUntil;
        }

        credentials[credentialId] = Credential({
            consentTokenId: consentTokenId,
            consentTBA: consentToken.getTokenBoundAccount(consentTokenId),
            issuer: issuer,
            grantedUse: grantedUse,
            grantedPurpose: grantedPurpose,
            grantedAt: uint64(block.timestamp),
            expiresAt: effectiveExpiry,
            complianceScore: complianceScore,
            riskLevel: riskLevel,
            revoked: false,
            conditions: conditions
        });

        credentialsByConsent[consentTokenId].push(credentialId);
        credentialsByRequester[requester].push(credentialId);

        _safeMint(requester, credentialId);

        if (bytes(metadataURI).length > 0) {
            _setTokenURI(credentialId, metadataURI);
        }

        emit CredentialIssued(
            credentialId,
            consentTokenId,
            requester,
            grantedUse,
            complianceScore,
            effectiveExpiry
        );

        return credentialId;
    }

    function revokeCredential(
        uint256 credentialId,
        string calldata reason
    ) external {
        Credential storage cred = credentials[credentialId];
        require(!cred.revoked, "Already revoked");

        bool authorized = false;

        if (msg.sender == cred.issuer) {
            authorized = true;
        }

        if (msg.sender == cred.consentTBA) {
            authorized = true;
        }

        if (consentToken.isTokenOwner(cred.consentTokenId, msg.sender)) {
            authorized = true;
        }

        if (hasRole(DEFAULT_ADMIN_ROLE, msg.sender)) {
            authorized = true;
        }

        require(authorized, "Not authorized to revoke");

        cred.revoked = true;

        emit CredentialRevoked(credentialId, msg.sender, reason);
    }

    function revokeAllForConsent(
        uint256 consentTokenId,
        string calldata reason
    ) external {

        require(
            consentToken.isTokenOwner(consentTokenId, msg.sender) ||
            hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Not authorized"
        );

        uint256[] storage credIds = credentialsByConsent[consentTokenId];
        for (uint256 i = 0; i < credIds.length; i++) {
            if (!credentials[credIds[i]].revoked) {
                credentials[credIds[i]].revoked = true;
                emit CredentialRevoked(credIds[i], msg.sender, reason);
            }
        }
    }

    function isCredentialValid(uint256 credentialId) public view returns (bool) {
        if (_ownerOf(credentialId) == address(0)) {
            return false;
        }

        Credential storage cred = credentials[credentialId];

        if (cred.revoked) {
            return false;
        }

        if (cred.expiresAt > 0 && block.timestamp > cred.expiresAt) {
            return false;
        }

        try consentToken.isConsentValid(cred.consentTokenId) returns (bool valid) {
            if (!valid) {
                return false;
            }
        } catch {
            return false;
        }

        return true;
    }

    function hasValidAccess(
        address requester,
        uint256 consentTokenId
    ) external view returns (bool) {
        uint256[] storage credIds = credentialsByRequester[requester];

        for (uint256 i = 0; i < credIds.length; i++) {
            Credential storage cred = credentials[credIds[i]];

            if (cred.consentTokenId != consentTokenId) {
                continue;
            }

            if (isCredentialValid(credIds[i])) {

                if (ownerOf(credIds[i]) == requester) {
                    return true;
                }
            }
        }

        return false;
    }

    function getValidCredentials(
        address requester
    ) external view returns (uint256[] memory) {
        uint256[] storage allCredIds = credentialsByRequester[requester];
        uint256 validCount = 0;

        for (uint256 i = 0; i < allCredIds.length; i++) {
            if (isCredentialValid(allCredIds[i]) &&
                ownerOf(allCredIds[i]) == requester) {
                validCount++;
            }
        }

        uint256[] memory validIds = new uint256[](validCount);
        uint256 idx = 0;

        for (uint256 i = 0; i < allCredIds.length; i++) {
            if (isCredentialValid(allCredIds[i]) &&
                ownerOf(allCredIds[i]) == requester) {
                validIds[idx++] = allCredIds[i];
            }
        }

        return validIds;
    }

    function getCredential(
        uint256 credentialId
    ) external view returns (Credential memory) {
        return credentials[credentialId];
    }

    function getCredentialsForConsent(
        uint256 consentTokenId
    ) external view returns (uint256[] memory) {
        return credentialsByConsent[consentTokenId];
    }

    function getCredentialCountForConsent(
        uint256 consentTokenId
    ) external view returns (uint256) {
        return credentialsByConsent[consentTokenId].length;
    }

    function getCredentialsForRequester(
        address requester
    ) external view returns (uint256[] memory) {
        return credentialsByRequester[requester];
    }

    function getRemainingValidity(
        uint256 credentialId
    ) external view returns (int256) {
        Credential storage cred = credentials[credentialId];

        if (cred.revoked) {
            return -1;
        }

        if (cred.expiresAt == 0) {
            return int256(type(uint256).max);
        }

        if (block.timestamp > cred.expiresAt) {
            return 0;
        }

        return int256(uint256(cred.expiresAt) - block.timestamp);
    }

    function verifyCredentialForUse(
        uint256 credentialId,
        bytes4 requiredUse,
        uint16 minScore
    ) external view returns (bool, string memory) {
        if (!isCredentialValid(credentialId)) {
            return (false, "Credential not valid");
        }

        Credential storage cred = credentials[credentialId];

        if (cred.grantedUse != requiredUse) {

            return (false, "Permission mismatch");
        }

        if (cred.complianceScore < minScore) {
            return (false, "Score below minimum");
        }

        return (true, "");
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
        return super.supportsInterface(interfaceId);
    }
}
