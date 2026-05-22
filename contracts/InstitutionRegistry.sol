// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract InstitutionRegistry is AccessControl {
    bytes32 public constant REGISTRY_ADMIN = keccak256("REGISTRY_ADMIN");

    uint8 public constant TYPE_ACADEMIC = 1;
    uint8 public constant TYPE_NONPROFIT = 2;
    uint8 public constant TYPE_COMMERCIAL = 3;
    uint8 public constant TYPE_CLINICAL = 4;
    uint8 public constant TYPE_GOVERNMENT = 5;

    struct Institution {
        bytes32 institutionId;
        string name;
        bytes2 countryCode;
        uint8 institutionType;
        address admin;
        bool verified;
        bool active;
        uint256 registeredAt;
        string metadataURI;
    }

    struct Membership {
        bytes32 institutionId;
        address member;
        string role;
        uint256 validFrom;
        uint256 validUntil;
        bool active;
    }

    mapping(bytes32 => Institution) public institutions;

    mapping(address => bytes32) public primaryAffiliation;

    mapping(address => mapping(bytes32 => Membership)) public memberships;

    mapping(address => bytes32[]) public memberInstitutions;

    mapping(bytes2 => bytes32[]) public institutionsByCountry;

    uint256 public totalInstitutions;
    bytes32[] public allInstitutionIds;

    mapping(address => uint8) public requesterTypes;
    mapping(address => bytes2) public requesterCountries;

    event InstitutionRegistered(
        bytes32 indexed institutionId,
        string name,
        bytes2 countryCode,
        uint8 institutionType,
        address admin
    );

    event InstitutionVerified(bytes32 indexed institutionId, bool verified);
    event InstitutionDeactivated(bytes32 indexed institutionId);
    event InstitutionAdminChanged(bytes32 indexed institutionId, address oldAdmin, address newAdmin);

    event MemberAdded(
        bytes32 indexed institutionId,
        address indexed member,
        string role,
        uint256 validUntil
    );

    event MemberRemoved(bytes32 indexed institutionId, address indexed member);
    event MemberRoleUpdated(bytes32 indexed institutionId, address indexed member, string newRole);
    event PrimaryAffiliationSet(address indexed member, bytes32 indexed institutionId);
    event RequesterTypeSet(address indexed user, uint8 requesterType, bytes2 countryCode);

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(REGISTRY_ADMIN, msg.sender);
    }

    function registerInstitution(
        string calldata name,
        bytes2 countryCode,
        uint8 institutionType,
        string calldata metadataURI
    ) external returns (bytes32 institutionId) {
        require(bytes(name).length > 0, "Name required");
        require(countryCode != bytes2(0), "Country required");
        require(institutionType >= TYPE_ACADEMIC && institutionType <= TYPE_GOVERNMENT, "Invalid type");

        institutionId = keccak256(abi.encodePacked(name, countryCode, block.chainid));
        require(institutions[institutionId].registeredAt == 0, "Already registered");

        institutions[institutionId] = Institution({
            institutionId: institutionId,
            name: name,
            countryCode: countryCode,
            institutionType: institutionType,
            admin: msg.sender,
            verified: false,
            active: true,
            registeredAt: block.timestamp,
            metadataURI: metadataURI
        });

        allInstitutionIds.push(institutionId);
        institutionsByCountry[countryCode].push(institutionId);
        totalInstitutions++;

        emit InstitutionRegistered(institutionId, name, countryCode, institutionType, msg.sender);

        return institutionId;
    }

    function verifyInstitution(bytes32 institutionId, bool verified) external onlyRole(REGISTRY_ADMIN) {
        require(institutions[institutionId].registeredAt > 0, "Not registered");
        institutions[institutionId].verified = verified;
        emit InstitutionVerified(institutionId, verified);
    }

    function deactivateInstitution(bytes32 institutionId) external {
        Institution storage inst = institutions[institutionId];
        require(inst.registeredAt > 0, "Not registered");
        require(
            hasRole(REGISTRY_ADMIN, msg.sender) || msg.sender == inst.admin,
            "Not authorized"
        );

        inst.active = false;
        emit InstitutionDeactivated(institutionId);
    }

    function changeInstitutionAdmin(bytes32 institutionId, address newAdmin) external {
        Institution storage inst = institutions[institutionId];
        require(inst.registeredAt > 0, "Not registered");
        require(
            hasRole(REGISTRY_ADMIN, msg.sender) || msg.sender == inst.admin,
            "Not authorized"
        );
        require(newAdmin != address(0), "Invalid admin");

        address oldAdmin = inst.admin;
        inst.admin = newAdmin;
        emit InstitutionAdminChanged(institutionId, oldAdmin, newAdmin);
    }

    function addMember(
        bytes32 institutionId,
        address member,
        string calldata role,
        uint256 validDays
    ) external {
        Institution storage inst = institutions[institutionId];
        require(inst.active, "Institution not active");
        require(msg.sender == inst.admin || hasRole(REGISTRY_ADMIN, msg.sender), "Not authorized");
        require(member != address(0), "Invalid member");

        uint256 validUntil = validDays > 0 ? block.timestamp + (validDays * 1 days) : 0;

        Membership storage existing = memberships[member][institutionId];
        if (existing.validFrom > 0) {

            existing.role = role;
            existing.validUntil = validUntil;
            existing.active = true;
        } else {

            memberships[member][institutionId] = Membership({
                institutionId: institutionId,
                member: member,
                role: role,
                validFrom: block.timestamp,
                validUntil: validUntil,
                active: true
            });
            memberInstitutions[member].push(institutionId);
        }

        if (primaryAffiliation[member] == bytes32(0)) {
            primaryAffiliation[member] = institutionId;
            emit PrimaryAffiliationSet(member, institutionId);
        }

        emit MemberAdded(institutionId, member, role, validUntil);
    }

    function removeMember(bytes32 institutionId, address member) external {
        Institution storage inst = institutions[institutionId];
        require(msg.sender == inst.admin || hasRole(REGISTRY_ADMIN, msg.sender), "Not authorized");

        Membership storage m = memberships[member][institutionId];
        require(m.validFrom > 0, "Not a member");

        m.active = false;

        if (primaryAffiliation[member] == institutionId) {
            primaryAffiliation[member] = bytes32(0);

            bytes32[] storage insts = memberInstitutions[member];
            for (uint i = 0; i < insts.length; i++) {
                if (insts[i] != institutionId && memberships[member][insts[i]].active) {
                    primaryAffiliation[member] = insts[i];
                    emit PrimaryAffiliationSet(member, insts[i]);
                    break;
                }
            }
        }

        emit MemberRemoved(institutionId, member);
    }

    function setPrimaryAffiliation(bytes32 institutionId) external {
        require(isMemberOf(msg.sender, institutionId), "Not a member");
        primaryAffiliation[msg.sender] = institutionId;
        emit PrimaryAffiliationSet(msg.sender, institutionId);
    }

    function isMemberOf(address member, bytes32 institutionId) public view returns (bool) {
        Membership storage m = memberships[member][institutionId];
        if (!m.active) return false;
        if (m.validUntil > 0 && block.timestamp > m.validUntil) return false;
        if (!institutions[institutionId].active) return false;
        return true;
    }

    function isInAllowedCountry(
        address member,
        bytes2[] calldata allowedCountries
    ) external view returns (bool allowed, bytes2 memberCountry) {
        bytes32 instId = primaryAffiliation[member];
        if (instId == bytes32(0)) return (false, bytes2(0));
        if (!isMemberOf(member, instId)) return (false, bytes2(0));

        Institution storage inst = institutions[instId];
        memberCountry = inst.countryCode;

        for (uint i = 0; i < allowedCountries.length; i++) {
            if (allowedCountries[i] == memberCountry) {
                return (true, memberCountry);
            }
        }

        return (false, memberCountry);
    }

    function isInAllowedInstitution(
        address member,
        bytes32[] calldata allowedInstitutions
    ) external view returns (bool allowed, bytes32 memberInstitution) {
        bytes32 instId = primaryAffiliation[member];
        if (instId == bytes32(0)) return (false, bytes32(0));
        if (!isMemberOf(member, instId)) return (false, bytes32(0));

        memberInstitution = instId;

        for (uint i = 0; i < allowedInstitutions.length; i++) {
            if (allowedInstitutions[i] == instId) {
                return (true, instId);
            }
        }

        return (false, instId);
    }

    function getMemberInstitutionType(address member) external view returns (
        bool hasMembership,
        uint8 institutionType,
        bool isVerified
    ) {
        bytes32 instId = primaryAffiliation[member];
        if (instId == bytes32(0)) return (false, 0, false);
        if (!isMemberOf(member, instId)) return (false, 0, false);

        Institution storage inst = institutions[instId];
        return (true, inst.institutionType, inst.verified);
    }

    function getMemberProfile(address member) external view returns (
        bool hasAffiliation,
        bytes32 institutionId,
        string memory institutionName,
        bytes2 countryCode,
        uint8 institutionType,
        bool institutionVerified,
        string memory memberRole
    ) {
        bytes32 instId = primaryAffiliation[member];
        if (instId == bytes32(0)) return (false, bytes32(0), "", bytes2(0), 0, false, "");
        if (!isMemberOf(member, instId)) return (false, bytes32(0), "", bytes2(0), 0, false, "");

        Institution storage inst = institutions[instId];
        Membership storage m = memberships[member][instId];

        return (
            true,
            instId,
            inst.name,
            inst.countryCode,
            inst.institutionType,
            inst.verified,
            m.role
        );
    }

    function getInstitution(bytes32 institutionId) external view returns (Institution memory) {
        return institutions[institutionId];
    }

    function getMembership(address member, bytes32 institutionId) external view returns (Membership memory) {
        return memberships[member][institutionId];
    }

    function getMemberInstitutionCount(address member) external view returns (uint256) {
        return memberInstitutions[member].length;
    }

    function getInstitutionsByCountry(bytes2 countryCode) external view returns (bytes32[] memory) {
        return institutionsByCountry[countryCode];
    }

    function getAllInstitutions() external view returns (bytes32[] memory) {
        return allInstitutionIds;
    }

    function setRequesterType(address user, uint8 requesterType, bytes2 countryCode)
        external
        onlyRole(REGISTRY_ADMIN)
    {
        require(user != address(0), "user=0");
        require(requesterType >= 1 && requesterType <= 5, "bad type");
        requesterTypes[user] = requesterType;
        requesterCountries[user] = countryCode;
        emit RequesterTypeSet(user, requesterType, countryCode);
    }

    function getRequesterType(address user) external view returns (uint8) {
        return requesterTypes[user];
    }

    function getRequesterCountry(address user) external view returns (bytes2) {
        return requesterCountries[user];
    }
}
