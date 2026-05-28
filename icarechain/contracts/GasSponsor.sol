// SPDX-License-Identifier: MIT
pragma solidity ^0.8.30;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract GasSponsor is AccessControl, Pausable, ReentrancyGuard {

    bytes32 public constant SPONSOR_ADMIN = keccak256("SPONSOR_ADMIN");
    bytes32 public constant RELAY_ROLE = keccak256("RELAY_ROLE");

    uint256 public constant MAX_GAS_PER_TX = 1_000_000;

    uint256 public constant DEFAULT_DAILY_BUDGET = 0.01 ether;

    struct InstitutionPool {

        uint256 balance;

        uint256 totalSpent;

        uint256 dailyBudgetPerMember;

        uint256 maxMembers;

        uint256 memberCount;

        bool active;

        address poolAdmin;

        uint256 createdAt;
    }

    struct MemberUsage {

        uint256 dailySpent;

        uint256 lastDay;

        uint256 totalSpent;

        uint256 dailyTxCount;

        uint256 totalTxCount;

        bool active;
    }

    mapping(bytes32 => InstitutionPool) public pools;

    mapping(bytes32 => mapping(address => MemberUsage)) public memberUsage;

    mapping(address => bytes32) public memberInstitution;

    uint256 public totalPools;

    uint256 public totalGasSponsored;

    event PoolCreated(
        bytes32 indexed institutionId,
        address indexed admin,
        uint256 initialDeposit
    );

    event PoolFunded(
        bytes32 indexed institutionId,
        address indexed funder,
        uint256 amount,
        uint256 newBalance
    );

    event PoolWithdrawn(
        bytes32 indexed institutionId,
        address indexed admin,
        uint256 amount,
        uint256 newBalance
    );

    event MemberSponsored(
        bytes32 indexed institutionId,
        address indexed member
    );

    event MemberRemoved(
        bytes32 indexed institutionId,
        address indexed member
    );

    event GasReimbursed(
        bytes32 indexed institutionId,
        address indexed member,
        uint256 gasUsed,
        uint256 gasCost,
        bytes32 txType
    );

    event DailyBudgetUpdated(
        bytes32 indexed institutionId,
        uint256 oldBudget,
        uint256 newBudget
    );

    error PoolNotFound(bytes32 institutionId);
    error PoolNotActive(bytes32 institutionId);
    error PoolAlreadyExists(bytes32 institutionId);
    error InsufficientBalance(bytes32 institutionId, uint256 required, uint256 available);
    error DailyBudgetExceeded(address member, uint256 spent, uint256 budget);
    error MemberNotSponsored(address member);
    error NotPoolAdmin(bytes32 institutionId, address caller);
    error MaxMembersReached(bytes32 institutionId);
    error GasLimitExceeded(uint256 gasUsed);

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(SPONSOR_ADMIN, admin);
    }

    function createPool(
        bytes32 institutionId,
        uint256 maxMembers,
        uint256 dailyBudget
    ) external payable whenNotPaused {
        if (pools[institutionId].createdAt > 0) revert PoolAlreadyExists(institutionId);

        pools[institutionId] = InstitutionPool({
            balance: msg.value,
            totalSpent: 0,
            dailyBudgetPerMember: dailyBudget,
            maxMembers: maxMembers,
            memberCount: 0,
            active: true,
            poolAdmin: msg.sender,
            createdAt: block.timestamp
        });

        totalPools++;

        emit PoolCreated(institutionId, msg.sender, msg.value);
        if (msg.value > 0) {
            emit PoolFunded(institutionId, msg.sender, msg.value, msg.value);
        }
    }

    function fundPool(bytes32 institutionId) external payable whenNotPaused {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);

        pool.balance += msg.value;

        emit PoolFunded(institutionId, msg.sender, msg.value, pool.balance);
    }

    function withdrawFromPool(
        bytes32 institutionId,
        uint256 amount
    ) external nonReentrant whenNotPaused {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);
        if (msg.sender != pool.poolAdmin && !hasRole(SPONSOR_ADMIN, msg.sender))
            revert NotPoolAdmin(institutionId, msg.sender);
        if (pool.balance < amount)
            revert InsufficientBalance(institutionId, amount, pool.balance);

        pool.balance -= amount;

        (bool sent, ) = payable(msg.sender).call{value: amount}("");
        require(sent, "GasSponsor: transfer failed");

        emit PoolWithdrawn(institutionId, msg.sender, amount, pool.balance);
    }

    function addSponsoredMember(
        bytes32 institutionId,
        address member
    ) external whenNotPaused {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);
        if (!pool.active) revert PoolNotActive(institutionId);
        if (msg.sender != pool.poolAdmin && !hasRole(SPONSOR_ADMIN, msg.sender))
            revert NotPoolAdmin(institutionId, msg.sender);
        if (pool.maxMembers > 0 && pool.memberCount >= pool.maxMembers)
            revert MaxMembersReached(institutionId);

        MemberUsage storage usage = memberUsage[institutionId][member];
        if (!usage.active) {
            usage.active = true;
            pool.memberCount++;
            memberInstitution[member] = institutionId;
        }

        emit MemberSponsored(institutionId, member);
    }

    function removeSponsoredMember(
        bytes32 institutionId,
        address member
    ) external {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);
        if (msg.sender != pool.poolAdmin && !hasRole(SPONSOR_ADMIN, msg.sender))
            revert NotPoolAdmin(institutionId, msg.sender);

        MemberUsage storage usage = memberUsage[institutionId][member];
        if (usage.active) {
            usage.active = false;
            pool.memberCount--;
            if (memberInstitution[member] == institutionId) {
                memberInstitution[member] = bytes32(0);
            }
        }

        emit MemberRemoved(institutionId, member);
    }

    function reimburse(
        address member,
        uint256 gasUsed,
        uint256 gasPrice,
        bytes32 txType
    ) external onlyRole(RELAY_ROLE) whenNotPaused {
        if (gasUsed > MAX_GAS_PER_TX) revert GasLimitExceeded(gasUsed);

        bytes32 institutionId = memberInstitution[member];
        if (institutionId == bytes32(0)) revert MemberNotSponsored(member);

        InstitutionPool storage pool = pools[institutionId];
        if (!pool.active) revert PoolNotActive(institutionId);

        uint256 gasCost = gasUsed * gasPrice;
        if (pool.balance < gasCost)
            revert InsufficientBalance(institutionId, gasCost, pool.balance);

        MemberUsage storage usage = memberUsage[institutionId][member];
        if (!usage.active) revert MemberNotSponsored(member);

        uint256 currentDay = block.timestamp / 1 days;
        if (usage.lastDay != currentDay) {
            usage.dailySpent = 0;
            usage.dailyTxCount = 0;
            usage.lastDay = currentDay;
        }

        uint256 dailyBudget = pool.dailyBudgetPerMember > 0
            ? pool.dailyBudgetPerMember
            : DEFAULT_DAILY_BUDGET;

        if (usage.dailySpent + gasCost > dailyBudget)
            revert DailyBudgetExceeded(member, usage.dailySpent + gasCost, dailyBudget);

        pool.balance -= gasCost;
        pool.totalSpent += gasCost;
        usage.dailySpent += gasCost;
        usage.totalSpent += gasCost;
        usage.dailyTxCount++;
        usage.totalTxCount++;
        totalGasSponsored += gasCost;

        emit GasReimbursed(institutionId, member, gasUsed, gasCost, txType);
    }

    function canSponsor(
        address member,
        uint256 estimatedGasCost
    ) external view returns (bool canSponsor_, string memory reason) {
        bytes32 institutionId = memberInstitution[member];
        if (institutionId == bytes32(0)) return (false, "Not sponsored");

        InstitutionPool storage pool = pools[institutionId];
        if (!pool.active) return (false, "Pool inactive");
        if (pool.balance < estimatedGasCost) return (false, "Insufficient pool balance");

        MemberUsage storage usage = memberUsage[institutionId][member];
        if (!usage.active) return (false, "Member inactive");

        uint256 dailyBudget = pool.dailyBudgetPerMember > 0
            ? pool.dailyBudgetPerMember
            : DEFAULT_DAILY_BUDGET;

        uint256 currentDay = block.timestamp / 1 days;
        uint256 todaySpent = usage.lastDay == currentDay ? usage.dailySpent : 0;

        if (todaySpent + estimatedGasCost > dailyBudget)
            return (false, "Daily budget exceeded");

        return (true, "");
    }

    function getPool(bytes32 institutionId) external view returns (
        uint256 balance,
        uint256 totalSpent,
        uint256 dailyBudgetPerMember,
        uint256 maxMembers,
        uint256 memberCount,
        bool active
    ) {
        InstitutionPool storage pool = pools[institutionId];
        return (
            pool.balance,
            pool.totalSpent,
            pool.dailyBudgetPerMember > 0 ? pool.dailyBudgetPerMember : DEFAULT_DAILY_BUDGET,
            pool.maxMembers,
            pool.memberCount,
            pool.active
        );
    }

    function getMemberUsage(
        bytes32 institutionId,
        address member
    ) external view returns (
        uint256 dailySpent,
        uint256 dailyTxCount,
        uint256 totalSpent,
        uint256 totalTxCount,
        bool active
    ) {
        MemberUsage storage usage = memberUsage[institutionId][member];
        uint256 currentDay = block.timestamp / 1 days;

        return (
            usage.lastDay == currentDay ? usage.dailySpent : 0,
            usage.lastDay == currentDay ? usage.dailyTxCount : 0,
            usage.totalSpent,
            usage.totalTxCount,
            usage.active
        );
    }

    function setDailyBudget(
        bytes32 institutionId,
        uint256 newBudget
    ) external {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);
        if (msg.sender != pool.poolAdmin && !hasRole(SPONSOR_ADMIN, msg.sender))
            revert NotPoolAdmin(institutionId, msg.sender);

        uint256 oldBudget = pool.dailyBudgetPerMember;
        pool.dailyBudgetPerMember = newBudget;

        emit DailyBudgetUpdated(institutionId, oldBudget, newBudget);
    }

    function deactivatePool(bytes32 institutionId) external {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);
        if (msg.sender != pool.poolAdmin && !hasRole(SPONSOR_ADMIN, msg.sender))
            revert NotPoolAdmin(institutionId, msg.sender);

        pool.active = false;
    }

    function reactivatePool(bytes32 institutionId) external {
        InstitutionPool storage pool = pools[institutionId];
        if (pool.createdAt == 0) revert PoolNotFound(institutionId);
        if (msg.sender != pool.poolAdmin && !hasRole(SPONSOR_ADMIN, msg.sender))
            revert NotPoolAdmin(institutionId, msg.sender);

        pool.active = true;
    }

    function addRelay(address relay) external onlyRole(SPONSOR_ADMIN) {
        _grantRole(RELAY_ROLE, relay);
    }

    function removeRelay(address relay) external onlyRole(SPONSOR_ADMIN) {
        _revokeRole(RELAY_ROLE, relay);
    }

    function pause() external onlyRole(SPONSOR_ADMIN) {
        _pause();
    }

    function unpause() external onlyRole(SPONSOR_ADMIN) {
        _unpause();
    }

    receive() external payable {}
}
