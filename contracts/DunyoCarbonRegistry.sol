// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Dunyo Carbon Registry (Track 1 — Carbon Markets & Emissions Transparency)
/// @notice Emissions reports, verified credit issuance, transparent trading and final retirement.
/// Blocks the ICVCM double-counting cases: double issuance (same project+period, or same site+period across
/// registries), double use (retiring more than you hold) and double selling (selling credits you no longer hold).
contract DunyoCarbonRegistry {
    address public registry;
    mapping(address => bool) public verifier;

    struct Project { address owner; string name; int32 latE3; int32 lonE3; uint32 capacityKW; bool exists; }
    struct Batch { bytes32 project; string period; uint256 issued; uint256 retired; bytes32 evidence; }
    struct Listing { uint256 batch; address seller; uint256 tonnes; uint256 priceWei; bool open; }

    mapping(bytes32 => Project) public projects;
    mapping(bytes32 => bool) public issuedFor;          // keccak(project, period)
    mapping(bytes32 => bool) public siteCredited;       // keccak(lat, lon, period) — cross-registry guard
    Batch[] public batches;
    mapping(uint256 => mapping(address => uint256)) public balanceOf;   // batch => holder => credits
    Listing[] public listings;
    mapping(bytes32 => bool) public emissionReported;   // keccak(facility, period)

    event Accredited(address verifier);
    event ProjectRegistered(bytes32 indexed id, address owner);
    event Issued(uint256 indexed batch, bytes32 indexed project, string period, uint256 tonnes, bytes32 evidence);
    event Listed(uint256 indexed listing, uint256 batch, uint256 tonnes, uint256 priceWei);
    event Bought(uint256 indexed listing, address buyer);
    event Retired(uint256 indexed batch, address indexed by, uint256 tonnes, string beneficiary, bytes32 certificate);
    event EmissionsReported(bytes32 indexed facility, string period, uint256 tonnesE1, bool flagged, bytes32 evidence);

    constructor() { registry = msg.sender; }

    function accredit(address v) external { require(msg.sender == registry, "only registry"); verifier[v] = true; emit Accredited(v); }

    function registerProject(bytes32 id, string calldata name, int32 latE3, int32 lonE3, uint32 capacityKW) external {
        require(!projects[id].exists, "project exists");
        projects[id] = Project(msg.sender, name, latE3, lonE3, capacityKW, true);
        emit ProjectRegistered(id, msg.sender);
    }

    function issue(bytes32 project, string calldata period, uint256 tonnes, bytes32 evidence) external returns (uint256 id) {
        require(verifier[msg.sender], "only accredited verifier");
        Project storage p = projects[project];
        require(p.exists && tonnes > 0, "bad issue");
        bytes32 k = keccak256(abi.encode(project, period));
        require(!issuedFor[k], "double issuance: period already credited");
        bytes32 site = keccak256(abi.encode(p.latE3, p.lonE3, period));
        require(!siteCredited[site], "double counting: site and period already credited");
        issuedFor[k] = true; siteCredited[site] = true;
        batches.push(Batch(project, period, tonnes, 0, evidence));
        id = batches.length - 1;
        balanceOf[id][p.owner] = tonnes;
        emit Issued(id, project, period, tonnes, evidence);
    }

    function list(uint256 batch, uint256 tonnes, uint256 priceWei) external returns (uint256 id) {
        require(balanceOf[batch][msg.sender] >= tonnes, "not enough credits (sold or retired?)");
        listings.push(Listing(batch, msg.sender, tonnes, priceWei, true));
        id = listings.length - 1;
        emit Listed(id, batch, tonnes, priceWei);
    }

    function buy(uint256 id) external payable {
        Listing storage l = listings[id];
        require(l.open, "listing closed");
        require(balanceOf[l.batch][l.seller] >= l.tonnes, "double selling: seller no longer holds credits");
        require(msg.value == l.tonnes * l.priceWei, "wrong payment");
        l.open = false;
        balanceOf[l.batch][l.seller] -= l.tonnes;
        balanceOf[l.batch][msg.sender] += l.tonnes;
        payable(l.seller).transfer(msg.value);
        emit Bought(id, msg.sender);
    }

    function retire(uint256 batch, uint256 tonnes, string calldata beneficiary) external returns (bytes32 cert) {
        require(balanceOf[batch][msg.sender] >= tonnes, "double use: not enough active credits");
        balanceOf[batch][msg.sender] -= tonnes;
        batches[batch].retired += tonnes;
        cert = keccak256(abi.encode(batch, msg.sender, tonnes, beneficiary, block.number));
        emit Retired(batch, msg.sender, tonnes, beneficiary, cert);
    }

    function reportEmissions(bytes32 facility, string calldata period, uint256 tonnesE1, bool flagged, bytes32 evidence) external {
        bytes32 k = keccak256(abi.encode(facility, period));
        require(!emissionReported[k], "report exists; publish a new version");
        emissionReported[k] = true;
        emit EmissionsReported(facility, period, tonnesE1, flagged, evidence);
    }
}
