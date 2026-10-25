// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ClimateEvidenceRegistry — tamper-evident anchoring of climate-anomaly investigations
/// @notice Dunyo's Notary agent anchors, for every WATCH/ALERT report:
///         • reportHash  = SHA-256 of the canonical JSON report
///         • merkleRoot  = Merkle root (SHA-256, sorted pairs) over every data payload, evidence item,
///                         model output and the report itself → any single input can later be proven
///                         to be part of the investigation with a short Merkle proof.
///         A report can be anchored only once (no double counting / silent re-issuing).
///         Independent parties (other stations, NGOs, agencies) can attest agreement on-chain.
contract ClimateEvidenceRegistry {
    struct Record {
        bytes32 merkleRoot;
        uint8 level;          // 0 = CLEAR, 1 = WATCH, 2 = ALERT
        uint64 timestamp;
        address notary;
        string location;
        string meta;          // compact JSON: period, event kinds, belief, model versions, IPFS/URL of report
        uint32 agree;
        uint32 disagree;
    }

    address public owner;
    mapping(address => bool) public notaries;
    mapping(bytes32 => Record) private records;     // key: reportHash
    bytes32[] public reportIds;
    mapping(bytes32 => mapping(address => bool)) public hasAttested;

    event NotaryChanged(address indexed notary, bool allowed);
    event EvidenceAnchored(bytes32 indexed reportHash, bytes32 merkleRoot, uint8 level, string location, address indexed notary);
    event Attested(bytes32 indexed reportHash, address indexed attester, bool agrees, string note);

    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }
    modifier onlyNotary() { require(notaries[msg.sender], "not a notary"); _; }

    constructor() {
        owner = msg.sender;
        notaries[msg.sender] = true;
        emit NotaryChanged(msg.sender, true);
    }

    function setNotary(address who, bool allowed) external onlyOwner {
        notaries[who] = allowed;
        emit NotaryChanged(who, allowed);
    }

    function anchor(bytes32 reportHash, bytes32 merkleRoot, uint8 level, string calldata location, string calldata meta)
        external onlyNotary
    {
        require(level <= 2, "bad level");
        require(records[reportHash].timestamp == 0, "already anchored");
        records[reportHash] = Record(merkleRoot, level, uint64(block.timestamp), msg.sender, location, meta, 0, 0);
        reportIds.push(reportHash);
        emit EvidenceAnchored(reportHash, merkleRoot, level, location, msg.sender);
    }

    function attest(bytes32 reportHash, bool agrees, string calldata note) external {
        require(records[reportHash].timestamp != 0, "unknown report");
        require(!hasAttested[reportHash][msg.sender], "already attested");
        hasAttested[reportHash][msg.sender] = true;
        if (agrees) { records[reportHash].agree += 1; } else { records[reportHash].disagree += 1; }
        emit Attested(reportHash, msg.sender, agrees, note);
    }

    function getRecord(bytes32 reportHash) external view returns (
        bytes32 merkleRoot, uint8 level, uint64 timestamp, address notary, string memory location, string memory meta,
        uint32 agree, uint32 disagree)
    {
        Record storage r = records[reportHash];
        return (r.merkleRoot, r.level, r.timestamp, r.notary, r.location, r.meta, r.agree, r.disagree);
    }

    /// @notice Verify that `leaf` (SHA-256 of one input) belongs to the anchored investigation.
    function verifyLeaf(bytes32 reportHash, bytes32 leaf, bytes32[] calldata proof) external view returns (bool) {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 p = proof[i];
            h = h < p ? sha256(abi.encodePacked(h, p)) : sha256(abi.encodePacked(p, h));
        }
        return h == records[reportHash].merkleRoot && records[reportHash].timestamp != 0;
    }

    function count() external view returns (uint256) { return reportIds.length; }
}
