// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract ReputationRegistry is AccessControl {
    bytes32 public constant REPUTATION_ADMIN = keccak256("REPUTATION_ADMIN");
    bytes32 public constant RECORDER_ROLE = keccak256("RECORDER_ROLE");

    struct ResearcherStats {
        uint256 totalAccessGrants;
        uint256 totalCommitmentsMade;
        uint256 totalCommitmentsFulfilled;
        uint256 totalPublications;
        uint256 violations;
        uint256 firstActivityAt;
        uint256 lastActivityAt;
        uint256 reputationScore;
    }

    struct InstitutionStats {
        uint256 totalMembers;
        uint256 totalMemberAccessGrants;
        uint256 totalMemberPublications;
        uint256 totalMemberViolations;
        uint256 institutionScore;
    }

    struct ActivityRecord {
        uint256 timestamp;
        bytes32 activityType;
        bytes32 cohortHash;
        string details;
    }

    bytes32 public constant ACT_ACCESS_GRANT = keccak256("ACCESS_GRANT");
    bytes32 public constant ACT_COMMITMENT_MADE = keccak256("COMMITMENT_MADE");
    bytes32 public constant ACT_COMMITMENT_FULFILLED = keccak256("COMMITMENT_FULFILLED");
    bytes32 public constant ACT_PUBLICATION = keccak256("PUBLICATION");
    bytes32 public constant ACT_VIOLATION = keccak256("VIOLATION");

    mapping(address => ResearcherStats) public researcherStats;

    mapping(address => ActivityRecord[]) public researcherHistory;

    mapping(bytes32 => InstitutionStats) public institutionStats;

    mapping(address => bytes32) public researcherInstitution;

    mapping(bytes32 => mapping(address => bool)) public cohortAccess;

    uint256 public totalResearchers;

    event AccessGrantRecorded(
        address indexed researcher,
        bytes32 indexed cohortHash,
        uint256 newScore
    );

    event CommitmentMade(
        address indexed researcher,
        bytes32 indexed commitmentId,
        bytes32 commitmentType
    );

    event CommitmentFulfilled(
        address indexed researcher,
        bytes32 indexed commitmentId,
        bytes32 evidenceHash,
        uint256 newScore
    );

    event PublicationRecorded(
        address indexed researcher,
        bytes32 indexed cohortHash,
        string publicationDOI,
        uint256 newScore
    );

    event ViolationRecorded(
        address indexed researcher,
        string reason,
        uint256 newScore
    );

    event ReputationUpdated(
        address indexed researcher,
        uint256 oldScore,
        uint256 newScore
    );

    event ResearcherInstitutionSet(
        address indexed researcher,
        bytes32 indexed institutionId
    );

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(REPUTATION_ADMIN, msg.sender);
        _grantRole(RECORDER_ROLE, msg.sender);
    }

    function recordAccessGrant(
        address researcher,
        bytes32 cohortHash
    ) external onlyRole(RECORDER_ROLE) {
        require(researcher != address(0), "Invalid researcher");
        require(cohortHash != bytes32(0), "Invalid cohort");

        _initializeResearcherIfNeeded(researcher);

        require(!cohortAccess[cohortHash][researcher], "Access already recorded");
        cohortAccess[cohortHash][researcher] = true;

        ResearcherStats storage stats = researcherStats[researcher];
        stats.totalAccessGrants++;
        stats.lastActivityAt = block.timestamp;

        researcherHistory[researcher].push(ActivityRecord({
            timestamp: block.timestamp,
            activityType: ACT_ACCESS_GRANT,
            cohortHash: cohortHash,
            details: ""
        }));

        uint256 oldScore = stats.reputationScore;
        _updateReputationScore(researcher);

        emit AccessGrantRecorded(researcher, cohortHash, stats.reputationScore);

        _updateInstitutionStats(researcher);
    }

    function recordCommitmentMade(
        address researcher,
        bytes32 commitmentId,
        bytes32 commitmentType
    ) external onlyRole(RECORDER_ROLE) {
        require(researcher != address(0), "Invalid researcher");

        _initializeResearcherIfNeeded(researcher);

        ResearcherStats storage stats = researcherStats[researcher];
        stats.totalCommitmentsMade++;
        stats.lastActivityAt = block.timestamp;

        researcherHistory[researcher].push(ActivityRecord({
            timestamp: block.timestamp,
            activityType: ACT_COMMITMENT_MADE,
            cohortHash: commitmentId,
            details: ""
        }));

        emit CommitmentMade(researcher, commitmentId, commitmentType);
    }

    function recordCommitmentFulfilled(
        address researcher,
        bytes32 commitmentId,
        bytes32 evidenceHash
    ) external onlyRole(RECORDER_ROLE) {
        require(researcher != address(0), "Invalid researcher");

        ResearcherStats storage stats = researcherStats[researcher];
        require(stats.firstActivityAt > 0, "Researcher not found");

        stats.totalCommitmentsFulfilled++;
        stats.lastActivityAt = block.timestamp;

        researcherHistory[researcher].push(ActivityRecord({
            timestamp: block.timestamp,
            activityType: ACT_COMMITMENT_FULFILLED,
            cohortHash: commitmentId,
            details: ""
        }));

        _updateReputationScore(researcher);

        emit CommitmentFulfilled(researcher, commitmentId, evidenceHash, stats.reputationScore);

        _updateInstitutionStats(researcher);
    }

    function recordPublication(
        address researcher,
        bytes32 cohortHash,
        string calldata publicationDOI
    ) external onlyRole(RECORDER_ROLE) {
        require(researcher != address(0), "Invalid researcher");
        require(bytes(publicationDOI).length > 0, "DOI required");

        ResearcherStats storage stats = researcherStats[researcher];
        require(stats.firstActivityAt > 0, "Researcher not found");

        stats.totalPublications++;
        stats.lastActivityAt = block.timestamp;

        researcherHistory[researcher].push(ActivityRecord({
            timestamp: block.timestamp,
            activityType: ACT_PUBLICATION,
            cohortHash: cohortHash,
            details: publicationDOI
        }));

        _updateReputationScore(researcher);

        emit PublicationRecorded(researcher, cohortHash, publicationDOI, stats.reputationScore);

        _updateInstitutionStats(researcher);
    }

    function recordViolation(
        address researcher,
        string calldata reason
    ) external onlyRole(REPUTATION_ADMIN) {
        require(researcher != address(0), "Invalid researcher");

        ResearcherStats storage stats = researcherStats[researcher];
        require(stats.firstActivityAt > 0, "Researcher not found");

        stats.violations++;
        stats.lastActivityAt = block.timestamp;

        researcherHistory[researcher].push(ActivityRecord({
            timestamp: block.timestamp,
            activityType: ACT_VIOLATION,
            cohortHash: bytes32(0),
            details: reason
        }));

        _updateReputationScore(researcher);

        emit ViolationRecorded(researcher, reason, stats.reputationScore);

        _updateInstitutionStats(researcher);
    }

    function _updateReputationScore(address researcher) internal {
        ResearcherStats storage stats = researcherStats[researcher];
        uint256 oldScore = stats.reputationScore;

        uint256 score = 500;

        score += _min(stats.totalAccessGrants * 10, 200);
        score += _min(stats.totalCommitmentsFulfilled * 20, 200);
        score += _min(stats.totalPublications * 50, 100);

        uint256 penalty = stats.violations * 100;
        if (penalty >= score) {
            score = 0;
        } else {
            score -= penalty;
        }

        stats.reputationScore = _min(score, 1000);

        if (oldScore != stats.reputationScore) {
            emit ReputationUpdated(researcher, oldScore, stats.reputationScore);
        }
    }

    function recalculateScore(address researcher) external {
        require(researcherStats[researcher].firstActivityAt > 0, "Researcher not found");
        _updateReputationScore(researcher);
    }

    function setResearcherInstitution(
        address researcher,
        bytes32 institutionId
    ) external onlyRole(RECORDER_ROLE) {
        bytes32 oldInstitution = researcherInstitution[researcher];

        if (oldInstitution != bytes32(0)) {
            institutionStats[oldInstitution].totalMembers--;
        }

        researcherInstitution[researcher] = institutionId;

        if (institutionId != bytes32(0)) {
            institutionStats[institutionId].totalMembers++;
            _updateInstitutionStats(researcher);
        }

        emit ResearcherInstitutionSet(researcher, institutionId);
    }

    function _updateInstitutionStats(address researcher) internal {
        bytes32 instId = researcherInstitution[researcher];
        if (instId == bytes32(0)) return;

        InstitutionStats storage instStats = institutionStats[instId];
        ResearcherStats storage resStats = researcherStats[researcher];

        instStats.totalMemberAccessGrants++;
        instStats.totalMemberPublications = resStats.totalPublications;

        if (instStats.totalMembers > 0) {
            instStats.institutionScore = _min(
                (instStats.totalMemberAccessGrants * 10) / instStats.totalMembers + 500,
                1000
            );
        }
    }

    function getReputationScore(address researcher) external view returns (uint256) {
        return researcherStats[researcher].reputationScore;
    }

    function getResearcherStats(address researcher) external view returns (ResearcherStats memory) {
        return researcherStats[researcher];
    }

    function getResearcherHistory(
        address researcher
    ) external view returns (ActivityRecord[] memory) {
        return researcherHistory[researcher];
    }

    function getResearcherActivityCount(address researcher) external view returns (uint256) {
        return researcherHistory[researcher].length;
    }

    function getInstitutionStats(bytes32 institutionId) external view returns (InstitutionStats memory) {
        return institutionStats[institutionId];
    }

    function hasAccessedCohort(
        address researcher,
        bytes32 cohortHash
    ) external view returns (bool) {
        return cohortAccess[cohortHash][researcher];
    }

    function getCommitmentFulfillmentRate(address researcher) external view returns (uint256) {
        ResearcherStats storage stats = researcherStats[researcher];
        if (stats.totalCommitmentsMade == 0) return 10000;
        return (stats.totalCommitmentsFulfilled * 10000) / stats.totalCommitmentsMade;
    }

    function _initializeResearcherIfNeeded(address researcher) internal {
        if (researcherStats[researcher].firstActivityAt == 0) {
            researcherStats[researcher].firstActivityAt = block.timestamp;
            researcherStats[researcher].reputationScore = 500;
            totalResearchers++;
        }
    }

    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}
