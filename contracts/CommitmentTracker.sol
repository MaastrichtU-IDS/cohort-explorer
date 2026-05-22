// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";

contract CommitmentTracker is AccessControl {
    bytes32 public constant TRACKER_ADMIN = keccak256("TRACKER_ADMIN");
    bytes32 public constant RECORDER_ROLE = keccak256("RECORDER_ROLE");

    bytes32 public constant TYPE_PUBLICATION = keccak256("PUBLICATION");
    bytes32 public constant TYPE_DATA_RETURN = keccak256("DATA_RETURN");
    bytes32 public constant TYPE_COLLABORATION = keccak256("COLLABORATION");
    bytes32 public constant TYPE_MORATORIUM = keccak256("MORATORIUM");

    enum CommitmentStatus {
        ACTIVE,
        FULFILLED,
        EXPIRED,
        CANCELLED
    }

    struct Commitment {
        bytes32 commitmentId;
        bytes32 cohortHash;
        address researcher;
        bytes32 commitmentType;
        uint256 createdAt;
        uint256 deadline;
        CommitmentStatus status;
        bytes32 evidenceHash;
        string evidenceURI;
        uint256 fulfilledAt;
        string description;
    }

    struct CommitmentSummary {
        uint256 total;
        uint256 active;
        uint256 fulfilled;
        uint256 expired;
        uint256 cancelled;
    }

    mapping(bytes32 => Commitment) public commitments;

    bytes32[] public allCommitmentIds;

    mapping(address => bytes32[]) public researcherCommitments;

    mapping(bytes32 => bytes32[]) public cohortCommitments;

    mapping(address => mapping(bytes32 => mapping(bytes32 => bytes32))) public activeCommitment;

    mapping(bytes32 => uint256) public defaultDeadlineDays;

    event CommitmentCreated(
        bytes32 indexed commitmentId,
        bytes32 indexed cohortHash,
        address indexed researcher,
        bytes32 commitmentType,
        uint256 deadline,
        string description
    );

    event CommitmentFulfilled(
        bytes32 indexed commitmentId,
        address indexed researcher,
        bytes32 evidenceHash,
        string evidenceURI
    );

    event CommitmentExpired(
        bytes32 indexed commitmentId,
        address indexed researcher,
        bytes32 indexed cohortHash
    );

    event CommitmentCancelled(
        bytes32 indexed commitmentId,
        address cancelledBy,
        string reason
    );

    event DeadlineExtended(
        bytes32 indexed commitmentId,
        uint256 oldDeadline,
        uint256 newDeadline
    );

    event DefaultDeadlineSet(
        bytes32 indexed commitmentType,
        uint256 days_
    );

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(TRACKER_ADMIN, msg.sender);
        _grantRole(RECORDER_ROLE, msg.sender);

        defaultDeadlineDays[TYPE_PUBLICATION] = 730;
        defaultDeadlineDays[TYPE_DATA_RETURN] = 365;
        defaultDeadlineDays[TYPE_COLLABORATION] = 0;
        defaultDeadlineDays[TYPE_MORATORIUM] = 180;
    }

    function createCommitment(
        bytes32 cohortHash,
        address researcher,
        bytes32 commitmentType,
        uint256 deadlineDays,
        string calldata description
    ) external onlyRole(RECORDER_ROLE) returns (bytes32 commitmentId) {
        require(cohortHash != bytes32(0), "Invalid cohort");
        require(researcher != address(0), "Invalid researcher");

        if (deadlineDays == 0) {
            deadlineDays = defaultDeadlineDays[commitmentType];
        }
        require(deadlineDays > 0, "Deadline required");

        commitmentId = keccak256(abi.encodePacked(
            cohortHash,
            researcher,
            commitmentType,
            block.timestamp,
            block.number
        ));

        uint256 deadline = block.timestamp + (deadlineDays * 1 days);

        commitments[commitmentId] = Commitment({
            commitmentId: commitmentId,
            cohortHash: cohortHash,
            researcher: researcher,
            commitmentType: commitmentType,
            createdAt: block.timestamp,
            deadline: deadline,
            status: CommitmentStatus.ACTIVE,
            evidenceHash: bytes32(0),
            evidenceURI: "",
            fulfilledAt: 0,
            description: description
        });

        allCommitmentIds.push(commitmentId);
        researcherCommitments[researcher].push(commitmentId);
        cohortCommitments[cohortHash].push(commitmentId);

        activeCommitment[researcher][cohortHash][commitmentType] = commitmentId;

        emit CommitmentCreated(
            commitmentId,
            cohortHash,
            researcher,
            commitmentType,
            deadline,
            description
        );

        return commitmentId;
    }

    function createMyCommitment(
        bytes32 cohortHash,
        bytes32 commitmentType,
        uint256 deadlineDays,
        string calldata description
    ) external returns (bytes32) {

        uint256 actualDeadline = deadlineDays > 0 ? deadlineDays : defaultDeadlineDays[commitmentType];
        require(actualDeadline > 0, "Deadline required");

        bytes32 commitmentId = keccak256(abi.encodePacked(
            cohortHash,
            msg.sender,
            commitmentType,
            block.timestamp,
            block.number
        ));

        uint256 deadline = block.timestamp + (actualDeadline * 1 days);

        commitments[commitmentId] = Commitment({
            commitmentId: commitmentId,
            cohortHash: cohortHash,
            researcher: msg.sender,
            commitmentType: commitmentType,
            createdAt: block.timestamp,
            deadline: deadline,
            status: CommitmentStatus.ACTIVE,
            evidenceHash: bytes32(0),
            evidenceURI: "",
            fulfilledAt: 0,
            description: description
        });

        allCommitmentIds.push(commitmentId);
        researcherCommitments[msg.sender].push(commitmentId);
        cohortCommitments[cohortHash].push(commitmentId);
        activeCommitment[msg.sender][cohortHash][commitmentType] = commitmentId;

        emit CommitmentCreated(
            commitmentId,
            cohortHash,
            msg.sender,
            commitmentType,
            deadline,
            description
        );

        return commitmentId;
    }

    function fulfillCommitment(
        bytes32 commitmentId,
        bytes32 evidenceHash,
        string calldata evidenceURI
    ) external {
        Commitment storage c = commitments[commitmentId];
        require(c.createdAt > 0, "Commitment not found");
        require(c.researcher == msg.sender || hasRole(RECORDER_ROLE, msg.sender), "Not authorized");
        require(c.status == CommitmentStatus.ACTIVE, "Commitment not active");
        require(evidenceHash != bytes32(0) || bytes(evidenceURI).length > 0, "Evidence required");

        c.status = CommitmentStatus.FULFILLED;
        c.evidenceHash = evidenceHash;
        c.evidenceURI = evidenceURI;
        c.fulfilledAt = block.timestamp;

        emit CommitmentFulfilled(commitmentId, c.researcher, evidenceHash, evidenceURI);
    }

    function markExpired(bytes32 commitmentId) external {
        Commitment storage c = commitments[commitmentId];
        require(c.createdAt > 0, "Commitment not found");
        require(c.status == CommitmentStatus.ACTIVE, "Commitment not active");
        require(block.timestamp > c.deadline, "Not expired yet");

        c.status = CommitmentStatus.EXPIRED;

        emit CommitmentExpired(commitmentId, c.researcher, c.cohortHash);
    }

    function batchMarkExpired(bytes32[] calldata commitmentIds) external {
        for (uint256 i = 0; i < commitmentIds.length; i++) {
            Commitment storage c = commitments[commitmentIds[i]];
            if (c.status == CommitmentStatus.ACTIVE && block.timestamp > c.deadline) {
                c.status = CommitmentStatus.EXPIRED;
                emit CommitmentExpired(commitmentIds[i], c.researcher, c.cohortHash);
            }
        }
    }

    function cancelCommitment(
        bytes32 commitmentId,
        string calldata reason
    ) external onlyRole(TRACKER_ADMIN) {
        Commitment storage c = commitments[commitmentId];
        require(c.createdAt > 0, "Commitment not found");
        require(c.status == CommitmentStatus.ACTIVE, "Commitment not active");

        c.status = CommitmentStatus.CANCELLED;

        emit CommitmentCancelled(commitmentId, msg.sender, reason);
    }

    function extendDeadline(
        bytes32 commitmentId,
        uint256 additionalDays
    ) external onlyRole(TRACKER_ADMIN) {
        Commitment storage c = commitments[commitmentId];
        require(c.createdAt > 0, "Commitment not found");
        require(c.status == CommitmentStatus.ACTIVE, "Commitment not active");

        uint256 oldDeadline = c.deadline;
        c.deadline += additionalDays * 1 days;

        emit DeadlineExtended(commitmentId, oldDeadline, c.deadline);
    }

    function setDefaultDeadline(
        bytes32 commitmentType,
        uint256 days_
    ) external onlyRole(TRACKER_ADMIN) {
        defaultDeadlineDays[commitmentType] = days_;
        emit DefaultDeadlineSet(commitmentType, days_);
    }

    function getCommitment(bytes32 commitmentId) external view returns (Commitment memory) {
        return commitments[commitmentId];
    }

    function getResearcherCommitments(
        address researcher
    ) external view returns (bytes32[] memory) {
        return researcherCommitments[researcher];
    }

    function getCohortCommitments(
        bytes32 cohortHash
    ) external view returns (bytes32[] memory) {
        return cohortCommitments[cohortHash];
    }

    function getPendingCommitments(
        address researcher
    ) external view returns (bytes32[] memory) {
        bytes32[] storage all = researcherCommitments[researcher];
        uint256 count = 0;

        for (uint256 i = 0; i < all.length; i++) {
            if (commitments[all[i]].status == CommitmentStatus.ACTIVE) {
                count++;
            }
        }

        bytes32[] memory pending = new bytes32[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < all.length; i++) {
            if (commitments[all[i]].status == CommitmentStatus.ACTIVE) {
                pending[idx++] = all[i];
            }
        }

        return pending;
    }

    function getUpcomingDeadlines(
        address researcher,
        uint256 withinDays
    ) external view returns (bytes32[] memory) {
        bytes32[] storage all = researcherCommitments[researcher];
        uint256 threshold = block.timestamp + (withinDays * 1 days);
        uint256 count = 0;

        for (uint256 i = 0; i < all.length; i++) {
            Commitment storage c = commitments[all[i]];
            if (c.status == CommitmentStatus.ACTIVE && c.deadline <= threshold) {
                count++;
            }
        }

        bytes32[] memory upcoming = new bytes32[](count);
        uint256 idx = 0;
        for (uint256 i = 0; i < all.length; i++) {
            Commitment storage c = commitments[all[i]];
            if (c.status == CommitmentStatus.ACTIVE && c.deadline <= threshold) {
                upcoming[idx++] = all[i];
            }
        }

        return upcoming;
    }

    function getResearcherSummary(
        address researcher
    ) external view returns (CommitmentSummary memory) {
        bytes32[] storage all = researcherCommitments[researcher];
        CommitmentSummary memory summary;

        summary.total = all.length;
        for (uint256 i = 0; i < all.length; i++) {
            CommitmentStatus s = commitments[all[i]].status;
            if (s == CommitmentStatus.ACTIVE) summary.active++;
            else if (s == CommitmentStatus.FULFILLED) summary.fulfilled++;
            else if (s == CommitmentStatus.EXPIRED) summary.expired++;
            else if (s == CommitmentStatus.CANCELLED) summary.cancelled++;
        }

        return summary;
    }

    function hasActiveCommitment(
        address researcher,
        bytes32 cohortHash,
        bytes32 commitmentType
    ) external view returns (bool, bytes32) {
        bytes32 commitmentId = activeCommitment[researcher][cohortHash][commitmentType];
        if (commitmentId == bytes32(0)) return (false, bytes32(0));

        Commitment storage c = commitments[commitmentId];
        return (c.status == CommitmentStatus.ACTIVE, commitmentId);
    }

    function getTotalCommitments() external view returns (uint256) {
        return allCommitmentIds.length;
    }

    function isExpired(bytes32 commitmentId) external view returns (bool) {
        Commitment storage c = commitments[commitmentId];
        return c.status == CommitmentStatus.ACTIVE && block.timestamp > c.deadline;
    }

    function getDaysRemaining(bytes32 commitmentId) external view returns (uint256) {
        Commitment storage c = commitments[commitmentId];
        if (c.status != CommitmentStatus.ACTIVE) return 0;
        if (block.timestamp >= c.deadline) return 0;
        return (c.deadline - block.timestamp) / 1 days;
    }
}
