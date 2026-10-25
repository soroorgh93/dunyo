// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Dunyo Energy Market (Track 2 — Renewable Energy & Energy Trading)
/// @notice Hourly (granular) renewable certificates, peer-to-peer settlement, same-hour claims and GREEN rewards.
contract DunyoEnergyMarket {
    address public operator;
    struct Device { address owner; uint32 capacityW; bool exists; }
    mapping(bytes32 => Device) public devices;
    mapping(bytes32 => mapping(uint64 => uint256)) public certWh;     // device => hour (unix/3600) => unsold/unclaimed Wh
    mapping(bytes32 => mapping(uint64 => bool)) public minted;
    mapping(address => uint256) public green;                           // GREEN reward token balance

    event Minted(bytes32 indexed device, uint64 hour, uint256 wh);
    event Trade(bytes32 indexed device, uint64 hour, address seller, address buyer, uint256 wh, uint256 priceWeiPerKWh);
    event Claimed(bytes32 indexed device, uint64 hour, address by, uint256 wh);
    event Reward(address indexed to, uint256 amount, string reason);

    constructor() { operator = msg.sender; }

    function registerDevice(bytes32 id, uint32 capacityW) external {
        require(!devices[id].exists, "device exists");
        devices[id] = Device(msg.sender, capacityW, true);
    }

    function mint(bytes32 id, uint64 hour, uint256 wh) external {
        Device storage d = devices[id];
        require(d.exists && d.owner == msg.sender, "only meter owner");
        require(wh <= d.capacityW, "physically impossible for this capacity");
        require(!minted[id][hour], "double issuance: hour already certified");
        minted[id][hour] = true;
        certWh[id][hour] = wh;
        emit Minted(id, hour, wh);
    }

    function settle(bytes32 id, uint64 hour, address seller, address buyer, uint256 wh, uint256 priceWeiPerKWh) external payable {
        require(msg.sender == operator, "only operator");
        require(certWh[id][hour] >= wh, "no certified energy left for that hour");
        require(msg.value == wh * priceWeiPerKWh / 1000, "wrong payment");
        certWh[id][hour] -= wh;
        payable(seller).transfer(msg.value);
        emit Trade(id, hour, seller, buyer, wh, priceWeiPerKWh);
    }

    function claim(bytes32 id, uint64 certHour, uint64 useHour, uint256 wh) external {
        require(certHour == useHour, "temporal mismatch: certificate hour must equal consumption hour");
        require(certWh[id][certHour] >= wh, "double claiming: certificate already used");
        certWh[id][certHour] -= wh;
        emit Claimed(id, certHour, msg.sender, wh);
    }

    function reward(address to, uint256 amount, string calldata reason) external {
        require(msg.sender == operator, "only operator");
        green[to] += amount;
        emit Reward(to, amount, reason);
    }
}
