// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Dunyo Data Market (Track 4 — Climate Data & Environmental Monitoring)
/// @notice IoT sensors submit signed data batches (Merkle roots) once; an AI validator oracle sets trust scores;
/// verified datasets are sold and 80% of revenue is shared among trusted sensor owners by trust weight.
contract DunyoDataMarket {
    address public validator;
    struct Sensor { address owner; uint16 trustBps; bool exists; }
    struct Dataset { address owner; uint256 priceWei; bytes32 contentHash; bytes32[] contributors; bool exists; }
    mapping(bytes32 => Sensor) public sensors;
    mapping(bytes32 => bool) public batchSeen;          // keccak(sensor, period)
    mapping(bytes32 => Dataset) internal datasets;
    mapping(address => uint256) public earnings;

    event SensorRegistered(bytes32 indexed sensor, address owner);
    event Batch(bytes32 indexed sensor, string period, bytes32 merkleRoot, uint32 readings);
    event Trust(bytes32 indexed sensor, uint16 trustBps, string reason);
    event Sale(bytes32 indexed dataset, address buyer, bytes32 accessToken);

    constructor() { validator = msg.sender; }

    function registerSensor(bytes32 id) external {
        require(!sensors[id].exists, "sensor exists");
        sensors[id] = Sensor(msg.sender, 5000, true);
        emit SensorRegistered(id, msg.sender);
    }

    function submitBatch(bytes32 id, string calldata period, bytes32 merkleRoot, uint32 readings) external {
        require(sensors[id].owner == msg.sender, "only the sensor's own key");
        bytes32 k = keccak256(abi.encode(id, period));
        require(!batchSeen[k], "replay: batch already submitted");
        batchSeen[k] = true;
        emit Batch(id, period, merkleRoot, readings);
    }

    function setTrust(bytes32 id, uint16 trustBps, string calldata reason) external {
        require(msg.sender == validator, "only validator oracle");
        require(sensors[id].exists && trustBps <= 10000, "bad trust");
        sensors[id].trustBps = trustBps;
        emit Trust(id, trustBps, reason);
    }

    function listDataset(bytes32 id, uint256 priceWei, bytes32 contentHash, bytes32[] calldata contributors) external {
        Dataset storage d = datasets[id];
        require(!d.exists, "dataset exists");
        d.owner = msg.sender;
        d.priceWei = priceWei;
        d.contentHash = contentHash;
        d.contributors = contributors;
        d.exists = true;
    }

    function buy(bytes32 id) external payable returns (bytes32 token) {
        Dataset storage d = datasets[id];
        require(d.exists && msg.value == d.priceWei, "wrong dataset or payment");
        uint256 total;
        for (uint256 i = 0; i < d.contributors.length; i++) {
            Sensor storage s = sensors[d.contributors[i]];
            if (s.trustBps >= 5000) total += s.trustBps;
        }
        uint256 pool = msg.value * 80 / 100;
        uint256 paid;
        if (total > 0) {
            for (uint256 i = 0; i < d.contributors.length; i++) {
                Sensor storage s = sensors[d.contributors[i]];
                if (s.trustBps >= 5000) { uint256 a = pool * s.trustBps / total; earnings[s.owner] += a; paid += a; }
            }
        }
        earnings[d.owner] += msg.value - paid;
        token = keccak256(abi.encode(id, msg.sender, block.number));
        emit Sale(id, msg.sender, token);
    }

    function withdraw() external { uint256 a = earnings[msg.sender]; earnings[msg.sender] = 0; payable(msg.sender).transfer(a); }
}
