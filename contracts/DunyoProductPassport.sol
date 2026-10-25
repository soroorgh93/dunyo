// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Dunyo Product Passport (Track 3 — Sustainable Supply Chains)
/// @notice Digital product passports: role-checked, ordered, hash-linked lifecycle events; accredited certifiers;
/// recycling mints recycled-material credits and refunds the deposit exactly once.
contract DunyoProductPassport {
    enum Role { None, Supplier, Certifier, Manufacturer, Logistics, Retailer, Consumer, Recycler }
    enum Stage { Created, Sourcing, Certification, Manufacturing, Transport, Retail, Use, Return, Recycling }

    address public registry;
    mapping(address => Role) public role;
    struct Passport { bytes32 gtin; address issuer; Stage stage; bytes32 head; uint256 co2Grams; uint256 depositWei; bool recycled; bool exists; }
    mapping(bytes32 => Passport) public passports;
    mapping(address => uint256) public recycledMaterialGrams;

    event Created(bytes32 indexed id, bytes32 gtin);
    event Step(bytes32 indexed id, Stage stage, address by, bytes32 eventHash, uint256 co2Grams);
    event Recycled(bytes32 indexed id, address recycler, address returnedBy, uint256 grams, uint256 refundWei);

    constructor() { registry = msg.sender; }

    function setRole(address who, Role r) external { require(msg.sender == registry, "only registry"); role[who] = r; }

    function create(bytes32 id, bytes32 gtin, bytes32 dataHash) external payable {
        require(!passports[id].exists, "passport exists");
        passports[id] = Passport(gtin, msg.sender, Stage.Created, dataHash, 0, msg.value, false, true);
        emit Created(id, gtin);
    }

    function allowed(Stage from, Stage to) public pure returns (bool) {
        if (from == Stage.Created) return to == Stage.Sourcing;
        if (from == Stage.Sourcing) return to == Stage.Certification || to == Stage.Manufacturing || to == Stage.Transport;
        if (from == Stage.Certification) return to == Stage.Manufacturing || to == Stage.Transport;
        if (from == Stage.Manufacturing) return to == Stage.Transport || to == Stage.Certification;
        if (from == Stage.Transport) return to == Stage.Transport || to == Stage.Manufacturing || to == Stage.Retail;
        if (from == Stage.Retail) return to == Stage.Use;
        if (from == Stage.Use) return to == Stage.Return;
        return false;
    }

    function roleFor(Stage s) public pure returns (Role) {
        if (s == Stage.Sourcing) return Role.Supplier;
        if (s == Stage.Certification) return Role.Certifier;
        if (s == Stage.Manufacturing) return Role.Manufacturer;
        if (s == Stage.Transport) return Role.Logistics;
        if (s == Stage.Retail) return Role.Retailer;
        if (s == Stage.Use || s == Stage.Return) return Role.Consumer;
        return Role.Recycler;
    }

    function addStep(bytes32 id, Stage s, bytes32 dataHash, uint256 co2Grams) external {
        Passport storage p = passports[id];
        require(p.exists && !p.recycled, "unknown or closed passport");
        require(allowed(p.stage, s), "out-of-order lifecycle step");
        require(role[msg.sender] == roleFor(s), "wrong signer for this step");
        p.head = keccak256(abi.encode(p.head, s, msg.sender, dataHash, co2Grams));
        p.stage = s;
        p.co2Grams += co2Grams;
        emit Step(id, s, msg.sender, p.head, co2Grams);
    }

    function recycle(bytes32 id, uint256 grams, address payable returnedBy) external {
        Passport storage p = passports[id];
        require(role[msg.sender] == Role.Recycler, "only a recycler");
        require(p.exists && !p.recycled, "double return: already recycled");
        require(p.stage == Stage.Return, "must be returned first");
        p.recycled = true;
        p.stage = Stage.Recycling;
        recycledMaterialGrams[msg.sender] += grams;
        uint256 refund = p.depositWei;
        p.depositWei = 0;
        if (refund > 0) returnedBy.transfer(refund);
        emit Recycled(id, msg.sender, returnedBy, grams, refund);
    }
}
