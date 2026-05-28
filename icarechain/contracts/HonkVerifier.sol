// SPDX-License-Identifier: Apache-2.0

pragma solidity >=0.8.21;

uint256 constant N = 32768;
uint256 constant LOG_N = 15;
uint256 constant NUMBER_OF_PUBLIC_INPUTS = 10;
uint256 constant VK_HASH = 0x13deb59aa5400ce2592ca8736dc7782dc48306ab13c9dc7a405723d2a7f19984;
library HonkVerificationKey {
    function loadVerificationKey() internal pure returns (Honk.VerificationKey memory) {
        Honk.VerificationKey memory vk = Honk.VerificationKey({
            circuitSize: uint256(32768),
            logCircuitSize: uint256(15),
            publicInputsSize: uint256(10),
            ql: Honk.G1Point({
               x: uint256(0x26dfffaa71ae6af551e57b279bb6ccab7ffecab690fa9809cac312dd639ba550),
               y: uint256(0x2cdad22e78cb994fadabcedf42528c9acf92079ccc57394d9eaf8332bf439fcb)
            }),
            qr: Honk.G1Point({
               x: uint256(0x2c8ee6edef99c9373e0448816ec62ebd8a3f2f732edcc2f0a05cbb4240200206),
               y: uint256(0x2b269a43db704b1a47b6fb883e0e16f01cb46357d5662f8b472ce88e9ed0b46f)
            }),
            qo: Honk.G1Point({
               x: uint256(0x0926d35fba2ecf501582aaad58149f51eb324553dee8b53872c2f8e59e617080),
               y: uint256(0x06364200ba346df58be5c3af7609a93dd742e0d03f9a1ab01d0645b277e94da0)
            }),
            q4: Honk.G1Point({
               x: uint256(0x0ee7857102eb3e5495f8917db75e84e622113fa3fe2ce13f80c3a78e5efe9549),
               y: uint256(0x263ae33d4eebbd788d1e18ac20849026e2e5543dc37a0c78de3da2aca8d44110)
            }),
            qm: Honk.G1Point({
               x: uint256(0x1539333a8519ae9a6632e036cf00de5135465ce2009ad665ac6f54505f7c4f1c),
               y: uint256(0x13766afa64ba2c3fc947d4c0765338c0faebf4308feb0dbddc69c09981e4f192)
            }),
            qc: Honk.G1Point({
               x: uint256(0x0a813263ae3d5d3d9b66c78c4a39aa9010e894bd9b797be2458f85cbec12ef92),
               y: uint256(0x2c993841b508d44797844cf06d4be0ca2775a221d9a0b81240798ae0678b167f)
            }),
            qLookup: Honk.G1Point({
               x: uint256(0x1bb0476d951ac63700ce2c49a098f286cda6e48c58afad92e17fdca8597bc70c),
               y: uint256(0x1a2dd67195770a01dd12aa3dbf7cb7a068a95b5cca74583356c05da4b13b454b)
            }),
            qArith: Honk.G1Point({
               x: uint256(0x2fe66eee22c45b04c462e585410f6dba97d5289d550a2888f71a2d4024e7dc72),
               y: uint256(0x28768872300ba073142b4bc9a9468c56518db2fd9816758f1463e01941697c99)
            }),
            qDeltaRange: Honk.G1Point({
               x: uint256(0x244f8d2bd1b4c5f71be26dc227c2cbb5f7e13a95074b6bd4bfb50fe5b50e7752),
               y: uint256(0x1f3b06b9a85d5d9c98f54eb7c97dfda01dc0e4454842bc2924e96f00278fc516)
            }),
            qElliptic: Honk.G1Point({
               x: uint256(0x1a531bcb75d6b1c3c0667549e113a1367aeebffa049a051a4829cb3840b70cd7),
               y: uint256(0x008995af741d83d73a6f13380b51a82b75fc1b09740fa76dac811025775d741f)
            }),
            qMemory: Honk.G1Point({
               x: uint256(0x0f477dd5985db8e6d30071e64066dc471c5cc4607e368dc50d82f8f1e22c83a8),
               y: uint256(0x1f46167513c359c7b65a149202d72e5b6bb250fd9d3b9836d78f751cc67a33ac)
            }),
            qNnf: Honk.G1Point({
               x: uint256(0x01549d5d5d1095b852a705d95bd91af6a78eb1e1b10eac808f1aaf14cd89d310),
               y: uint256(0x12d41cf03cb5c45fd8257d5c0ba51a34145e8ab09e601af44924f338b7cac4cf)
            }),
            qPoseidon2External: Honk.G1Point({
               x: uint256(0x08e0c04022cac0bf380ec95753628993a699d63b556af7a0169f321c3c9bb79d),
               y: uint256(0x08850536b5e694bd6d0a57d17033a87399354d5bd48c32dca5494ef46d319703)
            }),
            qPoseidon2Internal: Honk.G1Point({
               x: uint256(0x03933dc820fc91f7f8da3b8bbcb5ba7f59048f5a46e96827490998b1f53aec5e),
               y: uint256(0x2617fec00dea76dd14692151140f327914d1f2d2f518999edd1ddfb70dc46fc9)
            }),
            s1: Honk.G1Point({
               x: uint256(0x061f91b321f1ef380ae5c6d0bef22f7e1e72b341ad2255cfb6e6f1338821c45b),
               y: uint256(0x02fab254fb0b62b60a431a1248a535f370ace49f8777d3dc221560f657754665)
            }),
            s2: Honk.G1Point({
               x: uint256(0x28745c6eee49b0a41a7ec7d40daad2ff01da5f3a04d6c2f77e2a1d12618d934e),
               y: uint256(0x143b24ff245d87cdbb728f1d21bda52b6de345779b66cde9aa30fde4dac13626)
            }),
            s3: Honk.G1Point({
               x: uint256(0x2d34bef1b15616f73ad45173facd45e518eccc1ba025ed87a62207a3a5355c1e),
               y: uint256(0x246ca2deb649e938dc6087fa7a77cedbef8496c7b3e3a57319790319d3844db8)
            }),
            s4: Honk.G1Point({
               x: uint256(0x0ec0725c2cd2f8b4d140cda003ab359a5356ddd1adfb896b4072a731038c5d80),
               y: uint256(0x01677c5e0bf9ce6ed27f884eebfa578e5e23612600ce6e222bf58a6e354a3244)
            }),
            t1: Honk.G1Point({
               x: uint256(0x099e3bd5a0a00ab7fe18040105b9b395b5d8b7b4a63b05df652b0d10ef146d26),
               y: uint256(0x0015b8d2515d76e2ccec99dcd194592129af3a637f5a622a32440f860d1e2a7f)
            }),
            t2: Honk.G1Point({
               x: uint256(0x1b917517920bad3d8bc01c9595092a222b888108dc25d1aa450e0b4bc212c37e),
               y: uint256(0x305e8992b148eedb22e6e992077a84482141c7ebe42000a1d58ccb74381f6d19)
            }),
            t3: Honk.G1Point({
               x: uint256(0x16465a5ccbb550cd2c63bd58116fe47c86847618681dc29d8a9363ab7c40e1c3),
               y: uint256(0x2e24d420fbf9508ed31de692db477b439973ac12d7ca796d6fe98ca40e6ca6b7)
            }),
            t4: Honk.G1Point({
               x: uint256(0x043d063b130adfb37342af45d0155a28edd1a7e46c840d9c943fdf45521c64ce),
               y: uint256(0x261522c4089330646aff96736194949330952ae74c573d1686d9cb4a00733854)
            }),
            id1: Honk.G1Point({
               x: uint256(0x27da968ccfbcac2e80e0625b1f5f9ce2585330e8810af95cdac97bd20d1efb63),
               y: uint256(0x212fea8e0bf4474a8182ebe8de8d0bfe5191b5f8c472ab979684ddcd9de27738)
            }),
            id2: Honk.G1Point({
               x: uint256(0x22eab38caabcca25342b87ec5ea7d75e42ef7992bd299f67853b400e48d9ccd9),
               y: uint256(0x193b1f9b385aac8ec301821db8b03a24faa805b143e9b5c1513a2f9296c37f0c)
            }),
            id3: Honk.G1Point({
               x: uint256(0x0d8680d7f60dadefa89ea1abfb2c608f9b3fdb14cdf6f0c6531ac714177d3bfc),
               y: uint256(0x06f707ded047ada607dda8fefdde1992fd1ff10a927239ed412894b8d0504c9f)
            }),
            id4: Honk.G1Point({
               x: uint256(0x0dc71b86bc3b5457c138fdf930206a3d8ad10ccd411fd7bb3a233143ef672c67),
               y: uint256(0x1a99a0e5658f0a6444f7ff5f6107344ba2661490dbbec681d4c8f110bc3bbdb5)
            }),
            lagrangeFirst: Honk.G1Point({
               x: uint256(0x0000000000000000000000000000000000000000000000000000000000000001),
               y: uint256(0x0000000000000000000000000000000000000000000000000000000000000002)
            }),
            lagrangeLast: Honk.G1Point({
               x: uint256(0x29a5c276a5bb58c6147fd311b6ed235c78a621a8c459ccaef4bd834bf85281fc),
               y: uint256(0x0fd17395ab67e6224d0fdc1bda9f16c9a0e7ef186aead6e3e9b23c22d6ec865d)
            })
        });
        return vk;
    }
}

pragma solidity ^0.8.27;

interface IVerifier {
    function verify(bytes calldata _proof, bytes32[] calldata _publicInputs) external view returns (bool);
}

library Errors {
    error ValueGeLimbMax();
    error ValueGeGroupOrder();
    error ValueGeFieldOrder();

    error InvertOfZero();
    error NotPowerOfTwo();
    error ModExpFailed();

    error ProofLengthWrong();
    error ProofLengthWrongWithLogN(uint256 logN, uint256 actualLength, uint256 expectedLength);
    error PublicInputsLengthWrong();
    error SumcheckFailed();
    error ShpleminiFailed();

    error PointAtInfinity();

    error ConsistencyCheckFailed();
    error GeminiChallengeInSubgroup();
}

type Fr is uint256;

using {add as +} for Fr global;
using {sub as -} for Fr global;
using {mul as *} for Fr global;

using {notEqual as !=} for Fr global;
using {equal as ==} for Fr global;

uint256 constant SUBGROUP_SIZE = 256;
uint256 constant MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
uint256 constant P = MODULUS;
Fr constant SUBGROUP_GENERATOR = Fr.wrap(0x07b0c561a6148404f086204a9f36ffb0617942546750f230c893619174a57a76);
Fr constant SUBGROUP_GENERATOR_INVERSE = Fr.wrap(0x204bd3277422fad364751ad938e2b5e6a54cf8c68712848a692c553d0329f5d6);
Fr constant MINUS_ONE = Fr.wrap(MODULUS - 1);
Fr constant ONE = Fr.wrap(1);
Fr constant ZERO = Fr.wrap(0);

library FrLib {
    bytes4 internal constant FRLIB_MODEXP_FAILED_SELECTOR = 0xf8d61709;

    function invert(Fr value) internal view returns (Fr) {
        uint256 v = Fr.unwrap(value);
        require(v != 0, Errors.InvertOfZero());

        uint256 result;

        assembly ("memory-safe") {
            let free := mload(0x40)
            mstore(free, 0x20)
            mstore(add(free, 0x20), 0x20)
            mstore(add(free, 0x40), 0x20)
            mstore(add(free, 0x60), v)
            mstore(add(free, 0x80), sub(MODULUS, 2))
            mstore(add(free, 0xa0), MODULUS)
            let success := staticcall(gas(), 0x05, free, 0xc0, 0x00, 0x20)
            if iszero(success) {
                mstore(0x00, FRLIB_MODEXP_FAILED_SELECTOR)
                revert(0, 0x04)
            }
            result := mload(0x00)
            mstore(0x40, add(free, 0xc0))
        }

        return Fr.wrap(result);
    }

    function pow(Fr base, uint256 v) internal view returns (Fr) {
        uint256 b = Fr.unwrap(base);

        require(v > 0 && (v & (v - 1)) == 0, Errors.NotPowerOfTwo());
        uint256 result;

        assembly ("memory-safe") {
            let free := mload(0x40)
            mstore(free, 0x20)
            mstore(add(free, 0x20), 0x20)
            mstore(add(free, 0x40), 0x20)
            mstore(add(free, 0x60), b)
            mstore(add(free, 0x80), v)
            mstore(add(free, 0xa0), MODULUS)
            let success := staticcall(gas(), 0x05, free, 0xc0, 0x00, 0x20)
            if iszero(success) {
                mstore(0x00, FRLIB_MODEXP_FAILED_SELECTOR)
                revert(0, 0x04)
            }
            result := mload(0x00)
            mstore(0x40, add(free, 0xc0))
        }

        return Fr.wrap(result);
    }

    function div(Fr numerator, Fr denominator) internal view returns (Fr) {
        unchecked {
            return numerator * invert(denominator);
        }
    }

    function sqr(Fr value) internal pure returns (Fr) {
        unchecked {
            return value * value;
        }
    }

    function unwrap(Fr value) internal pure returns (uint256) {
        unchecked {
            return Fr.unwrap(value);
        }
    }

    function neg(Fr value) internal pure returns (Fr) {
        unchecked {
            return Fr.wrap(MODULUS - Fr.unwrap(value));
        }
    }

    function from(uint256 value) internal pure returns (Fr) {
        unchecked {
            require(value < MODULUS, Errors.ValueGeFieldOrder());
            return Fr.wrap(value);
        }
    }

    function fromBytes32(bytes32 value) internal pure returns (Fr) {
        unchecked {
            uint256 v = uint256(value);
            require(v < MODULUS, Errors.ValueGeFieldOrder());
            return Fr.wrap(v);
        }
    }

    function toBytes32(Fr value) internal pure returns (bytes32) {
        unchecked {
            return bytes32(Fr.unwrap(value));
        }
    }
}

function add(Fr a, Fr b) pure returns (Fr) {
    unchecked {
        return Fr.wrap(addmod(Fr.unwrap(a), Fr.unwrap(b), MODULUS));
    }
}

function mul(Fr a, Fr b) pure returns (Fr) {
    unchecked {
        return Fr.wrap(mulmod(Fr.unwrap(a), Fr.unwrap(b), MODULUS));
    }
}

function sub(Fr a, Fr b) pure returns (Fr) {
    unchecked {
        return Fr.wrap(addmod(Fr.unwrap(a), MODULUS - Fr.unwrap(b), MODULUS));
    }
}

function notEqual(Fr a, Fr b) pure returns (bool) {
    unchecked {
        return Fr.unwrap(a) != Fr.unwrap(b);
    }
}

function equal(Fr a, Fr b) pure returns (bool) {
    unchecked {
        return Fr.unwrap(a) == Fr.unwrap(b);
    }
}

uint256 constant CONST_PROOF_SIZE_LOG_N = 25;

uint256 constant NUMBER_OF_SUBRELATIONS = 28;
uint256 constant BATCHED_RELATION_PARTIAL_LENGTH = 8;
uint256 constant ZK_BATCHED_RELATION_PARTIAL_LENGTH = 9;
uint256 constant NUMBER_OF_ENTITIES = 41;

uint256 constant NUM_MASKING_POLYNOMIALS = 1;
uint256 constant NUMBER_OF_ENTITIES_ZK = NUMBER_OF_ENTITIES + NUM_MASKING_POLYNOMIALS;
uint256 constant NUMBER_UNSHIFTED = 36;
uint256 constant NUMBER_UNSHIFTED_ZK = NUMBER_UNSHIFTED + NUM_MASKING_POLYNOMIALS;
uint256 constant NUMBER_TO_BE_SHIFTED = 5;
uint256 constant PAIRING_POINTS_SIZE = 8;

uint256 constant FIELD_ELEMENT_SIZE = 0x20;
uint256 constant GROUP_ELEMENT_SIZE = 0x40;

uint256 constant NUMBER_OF_ALPHAS = NUMBER_OF_SUBRELATIONS - 1;

enum WIRE {
    Q_M,
    Q_C,
    Q_L,
    Q_R,
    Q_O,
    Q_4,
    Q_LOOKUP,
    Q_ARITH,
    Q_RANGE,
    Q_ELLIPTIC,
    Q_MEMORY,
    Q_NNF,
    Q_POSEIDON2_EXTERNAL,
    Q_POSEIDON2_INTERNAL,
    SIGMA_1,
    SIGMA_2,
    SIGMA_3,
    SIGMA_4,
    ID_1,
    ID_2,
    ID_3,
    ID_4,
    TABLE_1,
    TABLE_2,
    TABLE_3,
    TABLE_4,
    LAGRANGE_FIRST,
    LAGRANGE_LAST,
    W_L,
    W_R,
    W_O,
    W_4,
    Z_PERM,
    LOOKUP_INVERSES,
    LOOKUP_READ_COUNTS,
    LOOKUP_READ_TAGS,
    W_L_SHIFT,
    W_R_SHIFT,
    W_O_SHIFT,
    W_4_SHIFT,
    Z_PERM_SHIFT
}

library Honk {
    struct G1Point {
        uint256 x;
        uint256 y;
    }

    struct VerificationKey {

        uint256 circuitSize;
        uint256 logCircuitSize;
        uint256 publicInputsSize;

        G1Point qm;
        G1Point qc;
        G1Point ql;
        G1Point qr;
        G1Point qo;
        G1Point q4;
        G1Point qLookup;
        G1Point qArith;
        G1Point qDeltaRange;
        G1Point qMemory;
        G1Point qNnf;
        G1Point qElliptic;
        G1Point qPoseidon2External;
        G1Point qPoseidon2Internal;

        G1Point s1;
        G1Point s2;
        G1Point s3;
        G1Point s4;

        G1Point id1;
        G1Point id2;
        G1Point id3;
        G1Point id4;

        G1Point t1;
        G1Point t2;
        G1Point t3;
        G1Point t4;

        G1Point lagrangeFirst;
        G1Point lagrangeLast;
    }

    struct RelationParameters {

        Fr eta;
        Fr beta;
        Fr gamma;

        Fr publicInputsDelta;
    }

    struct Proof {

        Fr[PAIRING_POINTS_SIZE] pairingPointObject;

        G1Point w1;
        G1Point w2;
        G1Point w3;
        G1Point w4;

        G1Point zPerm;

        G1Point lookupReadCounts;
        G1Point lookupReadTags;
        G1Point lookupInverses;

        Fr[BATCHED_RELATION_PARTIAL_LENGTH][CONST_PROOF_SIZE_LOG_N] sumcheckUnivariates;
        Fr[NUMBER_OF_ENTITIES] sumcheckEvaluations;

        G1Point[CONST_PROOF_SIZE_LOG_N - 1] geminiFoldComms;
        Fr[CONST_PROOF_SIZE_LOG_N] geminiAEvaluations;
        G1Point shplonkQ;
        G1Point kzgQuotient;
    }

    struct ZKProof {

        Fr[PAIRING_POINTS_SIZE] pairingPointObject;

        G1Point geminiMaskingPoly;

        G1Point w1;
        G1Point w2;
        G1Point w3;
        G1Point w4;

        G1Point lookupReadCounts;
        G1Point lookupReadTags;
        G1Point lookupInverses;

        G1Point zPerm;
        G1Point[3] libraCommitments;

        Fr libraSum;
        Fr[ZK_BATCHED_RELATION_PARTIAL_LENGTH][CONST_PROOF_SIZE_LOG_N] sumcheckUnivariates;
        Fr libraEvaluation;
        Fr[NUMBER_OF_ENTITIES_ZK] sumcheckEvaluations;

        G1Point[CONST_PROOF_SIZE_LOG_N - 1] geminiFoldComms;
        Fr[CONST_PROOF_SIZE_LOG_N] geminiAEvaluations;
        Fr[4] libraPolyEvals;
        G1Point shplonkQ;
        G1Point kzgQuotient;
    }
}

struct ZKTranscript {

    Honk.RelationParameters relationParameters;
    Fr[NUMBER_OF_ALPHAS] alphas;
    Fr[CONST_PROOF_SIZE_LOG_N] gateChallenges;

    Fr libraChallenge;
    Fr[CONST_PROOF_SIZE_LOG_N] sumCheckUChallenges;

    Fr rho;
    Fr geminiR;
    Fr shplonkNu;
    Fr shplonkZ;

    Fr publicInputsDelta;
}

library ZKTranscriptLib {
    function generateTranscript(
        Honk.ZKProof memory proof,
        bytes32[] calldata publicInputs,
        uint256 vkHash,
        uint256 publicInputsSize,
        uint256 logN
    ) external pure returns (ZKTranscript memory t) {
        Fr previousChallenge;
        (t.relationParameters, previousChallenge) =
            generateRelationParametersChallenges(proof, publicInputs, vkHash, publicInputsSize, previousChallenge);

        (t.alphas, previousChallenge) = generateAlphaChallenges(previousChallenge, proof);

        (t.gateChallenges, previousChallenge) = generateGateChallenges(previousChallenge, logN);
        (t.libraChallenge, previousChallenge) = generateLibraChallenge(previousChallenge, proof);
        (t.sumCheckUChallenges, previousChallenge) = generateSumcheckChallenges(proof, previousChallenge, logN);

        (t.rho, previousChallenge) = generateRhoChallenge(proof, previousChallenge);

        (t.geminiR, previousChallenge) = generateGeminiRChallenge(proof, previousChallenge, logN);

        (t.shplonkNu, previousChallenge) = generateShplonkNuChallenge(proof, previousChallenge, logN);

        (t.shplonkZ, previousChallenge) = generateShplonkZChallenge(proof, previousChallenge);
        return t;
    }

    function splitChallenge(Fr challenge) internal pure returns (Fr first, Fr second) {
        uint256 challengeU256 = uint256(Fr.unwrap(challenge));

        uint256 lo = challengeU256 & 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        uint256 hi = challengeU256 >> 127;
        first = FrLib.from(lo);
        second = FrLib.from(hi);
    }

    function generateRelationParametersChallenges(
        Honk.ZKProof memory proof,
        bytes32[] calldata publicInputs,
        uint256 vkHash,
        uint256 publicInputsSize,
        Fr previousChallenge
    ) internal pure returns (Honk.RelationParameters memory rp, Fr nextPreviousChallenge) {
        (rp.eta, previousChallenge) = generateEtaChallenge(proof, publicInputs, vkHash, publicInputsSize);

        (rp.beta, rp.gamma, nextPreviousChallenge) = generateBetaGammaChallenges(previousChallenge, proof);
    }

    function generateEtaChallenge(
        Honk.ZKProof memory proof,
        bytes32[] calldata publicInputs,
        uint256 vkHash,
        uint256 publicInputsSize
    ) internal pure returns (Fr eta, Fr previousChallenge) {

        bytes32[] memory round0 = new bytes32[](1 + publicInputsSize + 8);
        round0[0] = bytes32(vkHash);

        for (uint256 i = 0; i < publicInputsSize - PAIRING_POINTS_SIZE; i++) {
            require(uint256(publicInputs[i]) < P, Errors.ValueGeFieldOrder());
            round0[1 + i] = publicInputs[i];
        }
        for (uint256 i = 0; i < PAIRING_POINTS_SIZE; i++) {
            round0[1 + publicInputsSize - PAIRING_POINTS_SIZE + i] = FrLib.toBytes32(proof.pairingPointObject[i]);
        }

        round0[1 + publicInputsSize] = bytes32(proof.geminiMaskingPoly.x);
        round0[1 + publicInputsSize + 1] = bytes32(proof.geminiMaskingPoly.y);

        round0[1 + publicInputsSize + 2] = bytes32(proof.w1.x);
        round0[1 + publicInputsSize + 3] = bytes32(proof.w1.y);
        round0[1 + publicInputsSize + 4] = bytes32(proof.w2.x);
        round0[1 + publicInputsSize + 5] = bytes32(proof.w2.y);
        round0[1 + publicInputsSize + 6] = bytes32(proof.w3.x);
        round0[1 + publicInputsSize + 7] = bytes32(proof.w3.y);

        previousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(round0))) % P);
        (eta,) = splitChallenge(previousChallenge);
    }

    function generateBetaGammaChallenges(Fr previousChallenge, Honk.ZKProof memory proof)
        internal
        pure
        returns (Fr beta, Fr gamma, Fr nextPreviousChallenge)
    {
        bytes32[7] memory round1;
        round1[0] = FrLib.toBytes32(previousChallenge);
        round1[1] = bytes32(proof.lookupReadCounts.x);
        round1[2] = bytes32(proof.lookupReadCounts.y);
        round1[3] = bytes32(proof.lookupReadTags.x);
        round1[4] = bytes32(proof.lookupReadTags.y);
        round1[5] = bytes32(proof.w4.x);
        round1[6] = bytes32(proof.w4.y);

        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(round1))) % P);
        (beta, gamma) = splitChallenge(nextPreviousChallenge);
    }

    function generateAlphaChallenges(Fr previousChallenge, Honk.ZKProof memory proof)
        internal
        pure
        returns (Fr[NUMBER_OF_ALPHAS] memory alphas, Fr nextPreviousChallenge)
    {

        uint256[5] memory alpha0;
        alpha0[0] = Fr.unwrap(previousChallenge);
        alpha0[1] = proof.lookupInverses.x;
        alpha0[2] = proof.lookupInverses.y;
        alpha0[3] = proof.zPerm.x;
        alpha0[4] = proof.zPerm.y;

        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(alpha0))) % P);
        Fr alpha;
        (alpha,) = splitChallenge(nextPreviousChallenge);

        alphas[0] = alpha;
        for (uint256 i = 1; i < NUMBER_OF_ALPHAS; i++) {
            alphas[i] = alphas[i - 1] * alpha;
        }
    }

    function generateGateChallenges(Fr previousChallenge, uint256 logN)
        internal
        pure
        returns (Fr[CONST_PROOF_SIZE_LOG_N] memory gateChallenges, Fr nextPreviousChallenge)
    {
        previousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(Fr.unwrap(previousChallenge)))) % P);
        (gateChallenges[0],) = splitChallenge(previousChallenge);
        for (uint256 i = 1; i < logN; i++) {
            gateChallenges[i] = gateChallenges[i - 1] * gateChallenges[i - 1];
        }
        nextPreviousChallenge = previousChallenge;
    }

    function generateLibraChallenge(Fr previousChallenge, Honk.ZKProof memory proof)
        internal
        pure
        returns (Fr libraChallenge, Fr nextPreviousChallenge)
    {

        uint256[4] memory challengeData;
        challengeData[0] = Fr.unwrap(previousChallenge);
        challengeData[1] = proof.libraCommitments[0].x;
        challengeData[2] = proof.libraCommitments[0].y;
        challengeData[3] = Fr.unwrap(proof.libraSum);
        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(challengeData))) % P);
        (libraChallenge,) = splitChallenge(nextPreviousChallenge);
    }

    function generateSumcheckChallenges(Honk.ZKProof memory proof, Fr prevChallenge, uint256 logN)
        internal
        pure
        returns (Fr[CONST_PROOF_SIZE_LOG_N] memory sumcheckChallenges, Fr nextPreviousChallenge)
    {
        for (uint256 i = 0; i < logN; i++) {
            Fr[ZK_BATCHED_RELATION_PARTIAL_LENGTH + 1] memory univariateChal;
            univariateChal[0] = prevChallenge;

            for (uint256 j = 0; j < ZK_BATCHED_RELATION_PARTIAL_LENGTH; j++) {
                univariateChal[j + 1] = proof.sumcheckUnivariates[i][j];
            }
            prevChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(univariateChal))) % P);

            (sumcheckChallenges[i],) = splitChallenge(prevChallenge);
        }
        nextPreviousChallenge = prevChallenge;
    }

    function generateRhoChallenge(Honk.ZKProof memory proof, Fr prevChallenge)
        internal
        pure
        returns (Fr rho, Fr nextPreviousChallenge)
    {
        uint256[NUMBER_OF_ENTITIES_ZK + 6] memory rhoChallengeElements;
        rhoChallengeElements[0] = Fr.unwrap(prevChallenge);
        uint256 i;
        for (i = 1; i <= NUMBER_OF_ENTITIES_ZK; i++) {
            rhoChallengeElements[i] = Fr.unwrap(proof.sumcheckEvaluations[i - 1]);
        }
        rhoChallengeElements[i] = Fr.unwrap(proof.libraEvaluation);
        i += 1;
        rhoChallengeElements[i] = proof.libraCommitments[1].x;
        rhoChallengeElements[i + 1] = proof.libraCommitments[1].y;
        i += 2;
        rhoChallengeElements[i] = proof.libraCommitments[2].x;
        rhoChallengeElements[i + 1] = proof.libraCommitments[2].y;

        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(rhoChallengeElements))) % P);
        (rho,) = splitChallenge(nextPreviousChallenge);
    }

    function generateGeminiRChallenge(Honk.ZKProof memory proof, Fr prevChallenge, uint256 logN)
        internal
        pure
        returns (Fr geminiR, Fr nextPreviousChallenge)
    {
        uint256[] memory gR = new uint256[]((logN - 1) * 2 + 1);
        gR[0] = Fr.unwrap(prevChallenge);

        for (uint256 i = 0; i < logN - 1; i++) {
            gR[1 + i * 2] = proof.geminiFoldComms[i].x;
            gR[2 + i * 2] = proof.geminiFoldComms[i].y;
        }

        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(gR))) % P);

        (geminiR,) = splitChallenge(nextPreviousChallenge);
    }

    function generateShplonkNuChallenge(Honk.ZKProof memory proof, Fr prevChallenge, uint256 logN)
        internal
        pure
        returns (Fr shplonkNu, Fr nextPreviousChallenge)
    {
        uint256[] memory shplonkNuChallengeElements = new uint256[](logN + 1 + 4);
        shplonkNuChallengeElements[0] = Fr.unwrap(prevChallenge);

        for (uint256 i = 1; i <= logN; i++) {
            shplonkNuChallengeElements[i] = Fr.unwrap(proof.geminiAEvaluations[i - 1]);
        }

        uint256 libraIdx = 0;
        for (uint256 i = logN + 1; i <= logN + 4; i++) {
            shplonkNuChallengeElements[i] = Fr.unwrap(proof.libraPolyEvals[libraIdx]);
            libraIdx++;
        }

        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(shplonkNuChallengeElements))) % P);
        (shplonkNu,) = splitChallenge(nextPreviousChallenge);
    }

    function generateShplonkZChallenge(Honk.ZKProof memory proof, Fr prevChallenge)
        internal
        pure
        returns (Fr shplonkZ, Fr nextPreviousChallenge)
    {
        uint256[3] memory shplonkZChallengeElements;
        shplonkZChallengeElements[0] = Fr.unwrap(prevChallenge);

        shplonkZChallengeElements[1] = proof.shplonkQ.x;
        shplonkZChallengeElements[2] = proof.shplonkQ.y;

        nextPreviousChallenge = FrLib.from(uint256(keccak256(abi.encodePacked(shplonkZChallengeElements))) % P);
        (shplonkZ,) = splitChallenge(nextPreviousChallenge);
    }

    function loadProof(bytes calldata proof, uint256 logN) internal pure returns (Honk.ZKProof memory p) {
        uint256 boundary = 0x0;

        for (uint256 i = 0; i < PAIRING_POINTS_SIZE; i++) {
            uint256 limb = uint256(bytes32(proof[boundary:boundary + FIELD_ELEMENT_SIZE]));

            require(limb < 2 ** (i % 2 == 0 ? 136 : 120), Errors.ValueGeLimbMax());
            p.pairingPointObject[i] = FrLib.from(limb);
            boundary += FIELD_ELEMENT_SIZE;
        }

        p.geminiMaskingPoly = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;

        p.w1 = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.w2 = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.w3 = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;

        p.lookupReadCounts = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.lookupReadTags = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.w4 = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.lookupInverses = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.zPerm = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.libraCommitments[0] = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;

        p.libraSum = bytesToFr(proof[boundary:boundary + FIELD_ELEMENT_SIZE]);
        boundary += FIELD_ELEMENT_SIZE;

        for (uint256 i = 0; i < logN; i++) {
            for (uint256 j = 0; j < ZK_BATCHED_RELATION_PARTIAL_LENGTH; j++) {
                p.sumcheckUnivariates[i][j] = bytesToFr(proof[boundary:boundary + FIELD_ELEMENT_SIZE]);
                boundary += FIELD_ELEMENT_SIZE;
            }
        }

        for (uint256 i = 0; i < NUMBER_OF_ENTITIES_ZK; i++) {
            p.sumcheckEvaluations[i] = bytesToFr(proof[boundary:boundary + FIELD_ELEMENT_SIZE]);
            boundary += FIELD_ELEMENT_SIZE;
        }

        p.libraEvaluation = bytesToFr(proof[boundary:boundary + FIELD_ELEMENT_SIZE]);
        boundary += FIELD_ELEMENT_SIZE;

        p.libraCommitments[1] = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;
        p.libraCommitments[2] = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;

        for (uint256 i = 0; i < logN - 1; i++) {
            p.geminiFoldComms[i] = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
            boundary += GROUP_ELEMENT_SIZE;
        }

        for (uint256 i = 0; i < logN; i++) {
            p.geminiAEvaluations[i] = bytesToFr(proof[boundary:boundary + FIELD_ELEMENT_SIZE]);
            boundary += FIELD_ELEMENT_SIZE;
        }

        for (uint256 i = 0; i < 4; i++) {
            p.libraPolyEvals[i] = bytesToFr(proof[boundary:boundary + FIELD_ELEMENT_SIZE]);
            boundary += FIELD_ELEMENT_SIZE;
        }

        p.shplonkQ = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
        boundary += GROUP_ELEMENT_SIZE;

        p.kzgQuotient = bytesToG1Point(proof[boundary:boundary + GROUP_ELEMENT_SIZE]);
    }
}

library RelationsLib {
    struct EllipticParams {

        Fr x_1;
        Fr y_1;
        Fr x_2;
        Fr y_2;
        Fr y_3;
        Fr x_3;

        Fr x_double_identity;
    }

    struct MemParams {
        Fr memory_record_check;
        Fr partial_record_check;
        Fr next_gate_access_type;
        Fr record_delta;
        Fr index_delta;
        Fr adjacent_values_match_if_adjacent_indices_match;
        Fr adjacent_values_match_if_adjacent_indices_match_and_next_access_is_a_read_operation;
        Fr access_check;
        Fr next_gate_access_type_is_boolean;
        Fr ROM_consistency_check_identity;
        Fr RAM_consistency_check_identity;
        Fr timestamp_delta;
        Fr RAM_timestamp_check_identity;
        Fr memory_identity;
        Fr index_is_monotonically_increasing;
    }

    struct NnfParams {
        Fr limb_subproduct;
        Fr non_native_field_gate_1;
        Fr non_native_field_gate_2;
        Fr non_native_field_gate_3;
        Fr limb_accumulator_1;
        Fr limb_accumulator_2;
        Fr nnf_identity;
    }

    struct PoseidonExternalParams {
        Fr s1;
        Fr s2;
        Fr s3;
        Fr s4;
        Fr u1;
        Fr u2;
        Fr u3;
        Fr u4;
        Fr t0;
        Fr t1;
        Fr t2;
        Fr t3;
        Fr v1;
        Fr v2;
        Fr v3;
        Fr v4;
        Fr q_pos_by_scaling;
    }

    struct PoseidonInternalParams {
        Fr u1;
        Fr u2;
        Fr u3;
        Fr u4;
        Fr u_sum;
        Fr v1;
        Fr v2;
        Fr v3;
        Fr v4;
        Fr s1;
        Fr q_pos_by_scaling;
    }

    Fr internal constant GRUMPKIN_CURVE_B_PARAMETER_NEGATED = Fr.wrap(17);
    uint256 internal constant NEG_HALF_MODULO_P = 0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000000;

    Fr internal constant LIMB_SIZE = Fr.wrap(uint256(1) << 68);
    Fr internal constant SUBLIMB_SHIFT = Fr.wrap(uint256(1) << 14);

    function accumulateRelationEvaluations(
        Fr[NUMBER_OF_ENTITIES] memory purportedEvaluations,
        Honk.RelationParameters memory rp,
        Fr[NUMBER_OF_ALPHAS] memory subrelationChallenges,
        Fr powPartialEval
    ) internal pure returns (Fr accumulator) {
        Fr[NUMBER_OF_SUBRELATIONS] memory evaluations;

        accumulateArithmeticRelation(purportedEvaluations, evaluations, powPartialEval);
        accumulatePermutationRelation(purportedEvaluations, rp, evaluations, powPartialEval);
        accumulateLogDerivativeLookupRelation(purportedEvaluations, rp, evaluations, powPartialEval);
        accumulateDeltaRangeRelation(purportedEvaluations, evaluations, powPartialEval);
        accumulateEllipticRelation(purportedEvaluations, evaluations, powPartialEval);
        accumulateMemoryRelation(purportedEvaluations, rp, evaluations, powPartialEval);
        accumulateNnfRelation(purportedEvaluations, evaluations, powPartialEval);
        accumulatePoseidonExternalRelation(purportedEvaluations, evaluations, powPartialEval);
        accumulatePoseidonInternalRelation(purportedEvaluations, evaluations, powPartialEval);

        accumulator = scaleAndBatchSubrelations(evaluations, subrelationChallenges);
    }

    function wire(Fr[NUMBER_OF_ENTITIES] memory p, WIRE _wire) internal pure returns (Fr) {
        return p[uint256(_wire)];
    }

    function accumulateArithmeticRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {

        Fr q_arith = wire(p, WIRE.Q_ARITH);
        {
            Fr neg_half = Fr.wrap(NEG_HALF_MODULO_P);

            Fr accum = (q_arith - Fr.wrap(3)) * (wire(p, WIRE.Q_M) * wire(p, WIRE.W_R) * wire(p, WIRE.W_L)) * neg_half;
            accum = accum + (wire(p, WIRE.Q_L) * wire(p, WIRE.W_L)) + (wire(p, WIRE.Q_R) * wire(p, WIRE.W_R))
                + (wire(p, WIRE.Q_O) * wire(p, WIRE.W_O)) + (wire(p, WIRE.Q_4) * wire(p, WIRE.W_4)) + wire(p, WIRE.Q_C);
            accum = accum + (q_arith - ONE) * wire(p, WIRE.W_4_SHIFT);
            accum = accum * q_arith;
            accum = accum * domainSep;
            evals[0] = accum;
        }

        {
            Fr accum = wire(p, WIRE.W_L) + wire(p, WIRE.W_4) - wire(p, WIRE.W_L_SHIFT) + wire(p, WIRE.Q_M);
            accum = accum * (q_arith - Fr.wrap(2));
            accum = accum * (q_arith - ONE);
            accum = accum * q_arith;
            accum = accum * domainSep;
            evals[1] = accum;
        }
    }

    function accumulatePermutationRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Honk.RelationParameters memory rp,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        Fr grand_product_numerator;
        Fr grand_product_denominator;

        {
            Fr num = wire(p, WIRE.W_L) + wire(p, WIRE.ID_1) * rp.beta + rp.gamma;
            num = num * (wire(p, WIRE.W_R) + wire(p, WIRE.ID_2) * rp.beta + rp.gamma);
            num = num * (wire(p, WIRE.W_O) + wire(p, WIRE.ID_3) * rp.beta + rp.gamma);
            num = num * (wire(p, WIRE.W_4) + wire(p, WIRE.ID_4) * rp.beta + rp.gamma);

            grand_product_numerator = num;
        }
        {
            Fr den = wire(p, WIRE.W_L) + wire(p, WIRE.SIGMA_1) * rp.beta + rp.gamma;
            den = den * (wire(p, WIRE.W_R) + wire(p, WIRE.SIGMA_2) * rp.beta + rp.gamma);
            den = den * (wire(p, WIRE.W_O) + wire(p, WIRE.SIGMA_3) * rp.beta + rp.gamma);
            den = den * (wire(p, WIRE.W_4) + wire(p, WIRE.SIGMA_4) * rp.beta + rp.gamma);

            grand_product_denominator = den;
        }

        {
            Fr acc = (wire(p, WIRE.Z_PERM) + wire(p, WIRE.LAGRANGE_FIRST)) * grand_product_numerator;

            acc = acc
                - ((wire(p, WIRE.Z_PERM_SHIFT) + (wire(p, WIRE.LAGRANGE_LAST) * rp.publicInputsDelta))
                    * grand_product_denominator);
            acc = acc * domainSep;
            evals[2] = acc;
        }

        {
            Fr acc = (wire(p, WIRE.LAGRANGE_LAST) * wire(p, WIRE.Z_PERM_SHIFT)) * domainSep;
            evals[3] = acc;
        }
    }

    function accumulateLogDerivativeLookupRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Honk.RelationParameters memory rp,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        Fr table_term;
        Fr lookup_term;

        {
            Fr beta_sqr = rp.beta * rp.beta;
            table_term = wire(p, WIRE.TABLE_1) + rp.gamma + (wire(p, WIRE.TABLE_2) * rp.beta)
                + (wire(p, WIRE.TABLE_3) * beta_sqr) + (wire(p, WIRE.TABLE_4) * beta_sqr * rp.beta);
        }

        {
            Fr beta_sqr = rp.beta * rp.beta;
            Fr derived_entry_1 = wire(p, WIRE.W_L) + rp.gamma + (wire(p, WIRE.Q_R) * wire(p, WIRE.W_L_SHIFT));
            Fr derived_entry_2 = wire(p, WIRE.W_R) + wire(p, WIRE.Q_M) * wire(p, WIRE.W_R_SHIFT);
            Fr derived_entry_3 = wire(p, WIRE.W_O) + wire(p, WIRE.Q_C) * wire(p, WIRE.W_O_SHIFT);

            lookup_term = derived_entry_1 + (derived_entry_2 * rp.beta) + (derived_entry_3 * beta_sqr)
                + (wire(p, WIRE.Q_O) * beta_sqr * rp.beta);
        }

        Fr lookup_inverse = wire(p, WIRE.LOOKUP_INVERSES) * table_term;
        Fr table_inverse = wire(p, WIRE.LOOKUP_INVERSES) * lookup_term;

        Fr inverse_exists_xor =
        wire(p, WIRE.LOOKUP_READ_TAGS) + wire(p, WIRE.Q_LOOKUP)
            - (wire(p, WIRE.LOOKUP_READ_TAGS) * wire(p, WIRE.Q_LOOKUP));

        Fr accumulatorNone = lookup_term * table_term * wire(p, WIRE.LOOKUP_INVERSES) - inverse_exists_xor;
        accumulatorNone = accumulatorNone * domainSep;

        Fr accumulatorOne = wire(p, WIRE.Q_LOOKUP) * lookup_inverse - wire(p, WIRE.LOOKUP_READ_COUNTS) * table_inverse;

        Fr read_tag = wire(p, WIRE.LOOKUP_READ_TAGS);

        Fr read_tag_boolean_relation = read_tag * read_tag - read_tag;

        evals[4] = accumulatorNone;
        evals[5] = accumulatorOne;
        evals[6] = read_tag_boolean_relation * domainSep;
    }

    function accumulateDeltaRangeRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        Fr minus_one = ZERO - ONE;
        Fr minus_two = ZERO - Fr.wrap(2);
        Fr minus_three = ZERO - Fr.wrap(3);

        Fr delta_1 = wire(p, WIRE.W_R) - wire(p, WIRE.W_L);
        Fr delta_2 = wire(p, WIRE.W_O) - wire(p, WIRE.W_R);
        Fr delta_3 = wire(p, WIRE.W_4) - wire(p, WIRE.W_O);
        Fr delta_4 = wire(p, WIRE.W_L_SHIFT) - wire(p, WIRE.W_4);

        {
            Fr acc = delta_1;
            acc = acc * (delta_1 + minus_one);
            acc = acc * (delta_1 + minus_two);
            acc = acc * (delta_1 + minus_three);
            acc = acc * wire(p, WIRE.Q_RANGE);
            acc = acc * domainSep;
            evals[7] = acc;
        }

        {
            Fr acc = delta_2;
            acc = acc * (delta_2 + minus_one);
            acc = acc * (delta_2 + minus_two);
            acc = acc * (delta_2 + minus_three);
            acc = acc * wire(p, WIRE.Q_RANGE);
            acc = acc * domainSep;
            evals[8] = acc;
        }

        {
            Fr acc = delta_3;
            acc = acc * (delta_3 + minus_one);
            acc = acc * (delta_3 + minus_two);
            acc = acc * (delta_3 + minus_three);
            acc = acc * wire(p, WIRE.Q_RANGE);
            acc = acc * domainSep;
            evals[9] = acc;
        }

        {
            Fr acc = delta_4;
            acc = acc * (delta_4 + minus_one);
            acc = acc * (delta_4 + minus_two);
            acc = acc * (delta_4 + minus_three);
            acc = acc * wire(p, WIRE.Q_RANGE);
            acc = acc * domainSep;
            evals[10] = acc;
        }
    }

    function accumulateEllipticRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        EllipticParams memory ep;
        ep.x_1 = wire(p, WIRE.W_R);
        ep.y_1 = wire(p, WIRE.W_O);

        ep.x_2 = wire(p, WIRE.W_L_SHIFT);
        ep.y_2 = wire(p, WIRE.W_4_SHIFT);
        ep.y_3 = wire(p, WIRE.W_O_SHIFT);
        ep.x_3 = wire(p, WIRE.W_R_SHIFT);

        Fr q_sign = wire(p, WIRE.Q_L);
        Fr q_is_double = wire(p, WIRE.Q_M);

        Fr x_diff = (ep.x_2 - ep.x_1);
        Fr y1_sqr = (ep.y_1 * ep.y_1);
        {

            Fr partialEval = domainSep;

            Fr y2_sqr = (ep.y_2 * ep.y_2);
            Fr y1y2 = ep.y_1 * ep.y_2 * q_sign;
            Fr x_add_identity = (ep.x_3 + ep.x_2 + ep.x_1);
            x_add_identity = x_add_identity * x_diff * x_diff;
            x_add_identity = x_add_identity - y2_sqr - y1_sqr + y1y2 + y1y2;

            evals[11] = x_add_identity * partialEval * wire(p, WIRE.Q_ELLIPTIC) * (ONE - q_is_double);
        }

        {
            Fr y1_plus_y3 = ep.y_1 + ep.y_3;
            Fr y_diff = ep.y_2 * q_sign - ep.y_1;
            Fr y_add_identity = y1_plus_y3 * x_diff + (ep.x_3 - ep.x_1) * y_diff;
            evals[12] = y_add_identity * domainSep * wire(p, WIRE.Q_ELLIPTIC) * (ONE - q_is_double);
        }

        {
            Fr x_pow_4 = (y1_sqr + GRUMPKIN_CURVE_B_PARAMETER_NEGATED) * ep.x_1;
            Fr y1_sqr_mul_4 = y1_sqr + y1_sqr;
            y1_sqr_mul_4 = y1_sqr_mul_4 + y1_sqr_mul_4;
            Fr x1_pow_4_mul_9 = x_pow_4 * Fr.wrap(9);

            ep.x_double_identity = (ep.x_3 + ep.x_1 + ep.x_1) * y1_sqr_mul_4 - x1_pow_4_mul_9;

            Fr acc = ep.x_double_identity * domainSep * wire(p, WIRE.Q_ELLIPTIC) * q_is_double;
            evals[11] = evals[11] + acc;
        }

        {
            Fr x1_sqr_mul_3 = (ep.x_1 + ep.x_1 + ep.x_1) * ep.x_1;
            Fr y_double_identity = x1_sqr_mul_3 * (ep.x_1 - ep.x_3) - (ep.y_1 + ep.y_1) * (ep.y_1 + ep.y_3);
            evals[12] = evals[12] + y_double_identity * domainSep * wire(p, WIRE.Q_ELLIPTIC) * q_is_double;
        }
    }

    function accumulateMemoryRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Honk.RelationParameters memory rp,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        MemParams memory ap;

        Fr eta_two = rp.eta * rp.eta;
        Fr eta_three = eta_two * rp.eta;

        ap.memory_record_check = wire(p, WIRE.W_O) * eta_three;
        ap.memory_record_check = ap.memory_record_check + (wire(p, WIRE.W_R) * eta_two);
        ap.memory_record_check = ap.memory_record_check + (wire(p, WIRE.W_L) * rp.eta);
        ap.memory_record_check = ap.memory_record_check + wire(p, WIRE.Q_C);
        ap.partial_record_check = ap.memory_record_check;
        ap.memory_record_check = ap.memory_record_check - wire(p, WIRE.W_4);

        ap.index_delta = wire(p, WIRE.W_L_SHIFT) - wire(p, WIRE.W_L);
        ap.record_delta = wire(p, WIRE.W_4_SHIFT) - wire(p, WIRE.W_4);

        ap.index_is_monotonically_increasing = ap.index_delta * (ap.index_delta - Fr.wrap(1));

        ap.adjacent_values_match_if_adjacent_indices_match = (ap.index_delta * MINUS_ONE + ONE) * ap.record_delta;

        evals[14] = ap.adjacent_values_match_if_adjacent_indices_match * (wire(p, WIRE.Q_L) * wire(p, WIRE.Q_R))
            * (wire(p, WIRE.Q_MEMORY) * domainSep);
        evals[15] = ap.index_is_monotonically_increasing * (wire(p, WIRE.Q_L) * wire(p, WIRE.Q_R))
            * (wire(p, WIRE.Q_MEMORY) * domainSep);

        ap.ROM_consistency_check_identity = ap.memory_record_check * (wire(p, WIRE.Q_L) * wire(p, WIRE.Q_R));

        Fr access_type = (wire(p, WIRE.W_4) - ap.partial_record_check);
        ap.access_check = access_type * (access_type - Fr.wrap(1));

        ap.next_gate_access_type = wire(p, WIRE.W_O_SHIFT) * eta_three;
        ap.next_gate_access_type = ap.next_gate_access_type + (wire(p, WIRE.W_R_SHIFT) * eta_two);
        ap.next_gate_access_type = ap.next_gate_access_type + (wire(p, WIRE.W_L_SHIFT) * rp.eta);
        ap.next_gate_access_type = wire(p, WIRE.W_4_SHIFT) - ap.next_gate_access_type;

        Fr value_delta = wire(p, WIRE.W_O_SHIFT) - wire(p, WIRE.W_O);
        ap.adjacent_values_match_if_adjacent_indices_match_and_next_access_is_a_read_operation =
            (ap.index_delta * MINUS_ONE + ONE) * value_delta * (ap.next_gate_access_type * MINUS_ONE + ONE);

        ap.next_gate_access_type_is_boolean =
            ap.next_gate_access_type * ap.next_gate_access_type - ap.next_gate_access_type;

        evals[16] = ap.adjacent_values_match_if_adjacent_indices_match_and_next_access_is_a_read_operation
            * (wire(p, WIRE.Q_O)) * (wire(p, WIRE.Q_MEMORY) * domainSep);
        evals[17] = ap.index_is_monotonically_increasing * (wire(p, WIRE.Q_O)) * (wire(p, WIRE.Q_MEMORY) * domainSep);
        evals[18] = ap.next_gate_access_type_is_boolean * (wire(p, WIRE.Q_O)) * (wire(p, WIRE.Q_MEMORY) * domainSep);

        ap.RAM_consistency_check_identity = ap.access_check * (wire(p, WIRE.Q_O));

        ap.timestamp_delta = wire(p, WIRE.W_R_SHIFT) - wire(p, WIRE.W_R);
        ap.RAM_timestamp_check_identity = (ap.index_delta * MINUS_ONE + ONE) * ap.timestamp_delta - wire(p, WIRE.W_O);

        ap.memory_identity = ap.ROM_consistency_check_identity;
        ap.memory_identity =
            ap.memory_identity + ap.RAM_timestamp_check_identity * (wire(p, WIRE.Q_4) * wire(p, WIRE.Q_L));
        ap.memory_identity = ap.memory_identity + ap.memory_record_check * (wire(p, WIRE.Q_M) * wire(p, WIRE.Q_L));
        ap.memory_identity = ap.memory_identity + ap.RAM_consistency_check_identity;

        ap.memory_identity = ap.memory_identity * (wire(p, WIRE.Q_MEMORY) * domainSep);
        evals[13] = ap.memory_identity;
    }

    function accumulateNnfRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        NnfParams memory ap;

        ap.limb_subproduct = wire(p, WIRE.W_L) * wire(p, WIRE.W_R_SHIFT) + wire(p, WIRE.W_L_SHIFT) * wire(p, WIRE.W_R);
        ap.non_native_field_gate_2 =
            (wire(p, WIRE.W_L) * wire(p, WIRE.W_4) + wire(p, WIRE.W_R) * wire(p, WIRE.W_O) - wire(p, WIRE.W_O_SHIFT));
        ap.non_native_field_gate_2 = ap.non_native_field_gate_2 * LIMB_SIZE;
        ap.non_native_field_gate_2 = ap.non_native_field_gate_2 - wire(p, WIRE.W_4_SHIFT);
        ap.non_native_field_gate_2 = ap.non_native_field_gate_2 + ap.limb_subproduct;
        ap.non_native_field_gate_2 = ap.non_native_field_gate_2 * wire(p, WIRE.Q_4);

        ap.limb_subproduct = ap.limb_subproduct * LIMB_SIZE;
        ap.limb_subproduct = ap.limb_subproduct + (wire(p, WIRE.W_L_SHIFT) * wire(p, WIRE.W_R_SHIFT));
        ap.non_native_field_gate_1 = ap.limb_subproduct;
        ap.non_native_field_gate_1 = ap.non_native_field_gate_1 - (wire(p, WIRE.W_O) + wire(p, WIRE.W_4));
        ap.non_native_field_gate_1 = ap.non_native_field_gate_1 * wire(p, WIRE.Q_O);

        ap.non_native_field_gate_3 = ap.limb_subproduct;
        ap.non_native_field_gate_3 = ap.non_native_field_gate_3 + wire(p, WIRE.W_4);
        ap.non_native_field_gate_3 = ap.non_native_field_gate_3 - (wire(p, WIRE.W_O_SHIFT) + wire(p, WIRE.W_4_SHIFT));
        ap.non_native_field_gate_3 = ap.non_native_field_gate_3 * wire(p, WIRE.Q_M);

        Fr non_native_field_identity =
        ap.non_native_field_gate_1 + ap.non_native_field_gate_2 + ap.non_native_field_gate_3;
        non_native_field_identity = non_native_field_identity * wire(p, WIRE.Q_R);

        ap.limb_accumulator_1 = wire(p, WIRE.W_R_SHIFT) * SUBLIMB_SHIFT;
        ap.limb_accumulator_1 = ap.limb_accumulator_1 + wire(p, WIRE.W_L_SHIFT);
        ap.limb_accumulator_1 = ap.limb_accumulator_1 * SUBLIMB_SHIFT;
        ap.limb_accumulator_1 = ap.limb_accumulator_1 + wire(p, WIRE.W_O);
        ap.limb_accumulator_1 = ap.limb_accumulator_1 * SUBLIMB_SHIFT;
        ap.limb_accumulator_1 = ap.limb_accumulator_1 + wire(p, WIRE.W_R);
        ap.limb_accumulator_1 = ap.limb_accumulator_1 * SUBLIMB_SHIFT;
        ap.limb_accumulator_1 = ap.limb_accumulator_1 + wire(p, WIRE.W_L);
        ap.limb_accumulator_1 = ap.limb_accumulator_1 - wire(p, WIRE.W_4);
        ap.limb_accumulator_1 = ap.limb_accumulator_1 * wire(p, WIRE.Q_4);

        ap.limb_accumulator_2 = wire(p, WIRE.W_O_SHIFT) * SUBLIMB_SHIFT;
        ap.limb_accumulator_2 = ap.limb_accumulator_2 + wire(p, WIRE.W_R_SHIFT);
        ap.limb_accumulator_2 = ap.limb_accumulator_2 * SUBLIMB_SHIFT;
        ap.limb_accumulator_2 = ap.limb_accumulator_2 + wire(p, WIRE.W_L_SHIFT);
        ap.limb_accumulator_2 = ap.limb_accumulator_2 * SUBLIMB_SHIFT;
        ap.limb_accumulator_2 = ap.limb_accumulator_2 + wire(p, WIRE.W_4);
        ap.limb_accumulator_2 = ap.limb_accumulator_2 * SUBLIMB_SHIFT;
        ap.limb_accumulator_2 = ap.limb_accumulator_2 + wire(p, WIRE.W_O);
        ap.limb_accumulator_2 = ap.limb_accumulator_2 - wire(p, WIRE.W_4_SHIFT);
        ap.limb_accumulator_2 = ap.limb_accumulator_2 * wire(p, WIRE.Q_M);

        Fr limb_accumulator_identity = ap.limb_accumulator_1 + ap.limb_accumulator_2;
        limb_accumulator_identity = limb_accumulator_identity * wire(p, WIRE.Q_O);

        ap.nnf_identity = non_native_field_identity + limb_accumulator_identity;
        ap.nnf_identity = ap.nnf_identity * (wire(p, WIRE.Q_NNF) * domainSep);
        evals[19] = ap.nnf_identity;
    }

    function accumulatePoseidonExternalRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        PoseidonExternalParams memory ep;

        ep.s1 = wire(p, WIRE.W_L) + wire(p, WIRE.Q_L);
        ep.s2 = wire(p, WIRE.W_R) + wire(p, WIRE.Q_R);
        ep.s3 = wire(p, WIRE.W_O) + wire(p, WIRE.Q_O);
        ep.s4 = wire(p, WIRE.W_4) + wire(p, WIRE.Q_4);

        ep.u1 = ep.s1 * ep.s1 * ep.s1 * ep.s1 * ep.s1;
        ep.u2 = ep.s2 * ep.s2 * ep.s2 * ep.s2 * ep.s2;
        ep.u3 = ep.s3 * ep.s3 * ep.s3 * ep.s3 * ep.s3;
        ep.u4 = ep.s4 * ep.s4 * ep.s4 * ep.s4 * ep.s4;

        ep.t0 = ep.u1 + ep.u2;
        ep.t1 = ep.u3 + ep.u4;
        ep.t2 = ep.u2 + ep.u2 + ep.t1;

        ep.t3 = ep.u4 + ep.u4 + ep.t0;

        ep.v4 = ep.t1 + ep.t1;
        ep.v4 = ep.v4 + ep.v4 + ep.t3;

        ep.v2 = ep.t0 + ep.t0;
        ep.v2 = ep.v2 + ep.v2 + ep.t2;

        ep.v1 = ep.t3 + ep.v2;
        ep.v3 = ep.t2 + ep.v4;

        ep.q_pos_by_scaling = wire(p, WIRE.Q_POSEIDON2_EXTERNAL) * domainSep;
        evals[20] = evals[20] + ep.q_pos_by_scaling * (ep.v1 - wire(p, WIRE.W_L_SHIFT));

        evals[21] = evals[21] + ep.q_pos_by_scaling * (ep.v2 - wire(p, WIRE.W_R_SHIFT));

        evals[22] = evals[22] + ep.q_pos_by_scaling * (ep.v3 - wire(p, WIRE.W_O_SHIFT));

        evals[23] = evals[23] + ep.q_pos_by_scaling * (ep.v4 - wire(p, WIRE.W_4_SHIFT));
    }

    function accumulatePoseidonInternalRelation(
        Fr[NUMBER_OF_ENTITIES] memory p,
        Fr[NUMBER_OF_SUBRELATIONS] memory evals,
        Fr domainSep
    ) internal pure {
        PoseidonInternalParams memory ip;

        Fr[4] memory INTERNAL_MATRIX_DIAGONAL = [
            FrLib.from(0x10dc6e9c006ea38b04b1e03b4bd9490c0d03f98929ca1d7fb56821fd19d3b6e7),
            FrLib.from(0x0c28145b6a44df3e0149b3d0a30b3bb599df9756d4dd9b84a86b38cfb45a740b),
            FrLib.from(0x00544b8338791518b2c7645a50392798b21f75bb60e3596170067d00141cac15),
            FrLib.from(0x222c01175718386f2e2e82eb122789e352e105a3b8fa852613bc534433ee428b)
        ];

        ip.s1 = wire(p, WIRE.W_L) + wire(p, WIRE.Q_L);

        ip.u1 = ip.s1 * ip.s1 * ip.s1 * ip.s1 * ip.s1;
        ip.u2 = wire(p, WIRE.W_R);
        ip.u3 = wire(p, WIRE.W_O);
        ip.u4 = wire(p, WIRE.W_4);

        ip.u_sum = ip.u1 + ip.u2 + ip.u3 + ip.u4;

        ip.q_pos_by_scaling = wire(p, WIRE.Q_POSEIDON2_INTERNAL) * domainSep;

        ip.v1 = ip.u1 * INTERNAL_MATRIX_DIAGONAL[0] + ip.u_sum;
        evals[24] = evals[24] + ip.q_pos_by_scaling * (ip.v1 - wire(p, WIRE.W_L_SHIFT));

        ip.v2 = ip.u2 * INTERNAL_MATRIX_DIAGONAL[1] + ip.u_sum;
        evals[25] = evals[25] + ip.q_pos_by_scaling * (ip.v2 - wire(p, WIRE.W_R_SHIFT));

        ip.v3 = ip.u3 * INTERNAL_MATRIX_DIAGONAL[2] + ip.u_sum;
        evals[26] = evals[26] + ip.q_pos_by_scaling * (ip.v3 - wire(p, WIRE.W_O_SHIFT));

        ip.v4 = ip.u4 * INTERNAL_MATRIX_DIAGONAL[3] + ip.u_sum;
        evals[27] = evals[27] + ip.q_pos_by_scaling * (ip.v4 - wire(p, WIRE.W_4_SHIFT));
    }

    function scaleAndBatchSubrelations(
        Fr[NUMBER_OF_SUBRELATIONS] memory evaluations,
        Fr[NUMBER_OF_ALPHAS] memory subrelationChallenges
    ) internal pure returns (Fr accumulator) {
        accumulator = evaluations[0];

        for (uint256 i = 1; i < NUMBER_OF_SUBRELATIONS; ++i) {
            accumulator = accumulator + evaluations[i] * subrelationChallenges[i - 1];
        }
    }
}

library CommitmentSchemeLib {
    using FrLib for Fr;

    struct ShpleminiIntermediates {
        Fr unshiftedScalar;
        Fr shiftedScalar;
        Fr unshiftedScalarNeg;
        Fr shiftedScalarNeg;

        Fr constantTermAccumulator;

        Fr batchingChallenge;

        Fr batchedEvaluation;
        Fr[4] denominators;
        Fr[4] batchingScalars;

        Fr posInvertedDenominator;

        Fr negInvertedDenominator;

        Fr scalingFactorPos;

        Fr scalingFactorNeg;

        Fr[] foldPosEvaluations;
    }

    function computeFoldPosEvaluations(
        Fr[CONST_PROOF_SIZE_LOG_N] memory sumcheckUChallenges,
        Fr batchedEvalAccumulator,
        Fr[CONST_PROOF_SIZE_LOG_N] memory geminiEvaluations,
        Fr[] memory geminiEvalChallengePowers,
        uint256 logSize
    ) internal view returns (Fr[] memory) {
        Fr[] memory foldPosEvaluations = new Fr[](logSize);
        for (uint256 i = logSize; i > 0; --i) {
            Fr challengePower = geminiEvalChallengePowers[i - 1];
            Fr u = sumcheckUChallenges[i - 1];

            Fr batchedEvalRoundAcc = ((challengePower * batchedEvalAccumulator * Fr.wrap(2)) - geminiEvaluations[i - 1]
                    * (challengePower * (ONE - u) - u));

            batchedEvalRoundAcc = batchedEvalRoundAcc * (challengePower * (ONE - u) + u).invert();

            batchedEvalAccumulator = batchedEvalRoundAcc;
            foldPosEvaluations[i - 1] = batchedEvalRoundAcc;
        }
        return foldPosEvaluations;
    }

    function computeSquares(Fr r, uint256 logN) internal pure returns (Fr[] memory) {
        Fr[] memory squares = new Fr[](logN);
        squares[0] = r;
        for (uint256 i = 1; i < logN; ++i) {
            squares[i] = squares[i - 1].sqr();
        }
        return squares;
    }
}

uint256 constant Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

function bytesToFr(bytes calldata proofSection) pure returns (Fr scalar) {
    scalar = FrLib.fromBytes32(bytes32(proofSection));
}

function bytesToG1Point(bytes calldata proofSection) pure returns (Honk.G1Point memory point) {
    uint256 x = uint256(bytes32(proofSection[0x00:0x20]));
    uint256 y = uint256(bytes32(proofSection[0x20:0x40]));
    require(x < Q && y < Q, Errors.ValueGeGroupOrder());

    require((x | y) != 0, Errors.PointAtInfinity());

    point = Honk.G1Point({x: x, y: y});
}

function negateInplace(Honk.G1Point memory point) pure returns (Honk.G1Point memory) {

    if (point.y != 0) {
        point.y = Q - point.y;
    }
    return point;
}

function convertPairingPointsToG1(Fr[PAIRING_POINTS_SIZE] memory pairingPoints)
    pure
    returns (Honk.G1Point memory lhs, Honk.G1Point memory rhs)
{

    uint256 lhsX = Fr.unwrap(pairingPoints[0]);
    lhsX |= Fr.unwrap(pairingPoints[1]) << 136;

    uint256 lhsY = Fr.unwrap(pairingPoints[2]);
    lhsY |= Fr.unwrap(pairingPoints[3]) << 136;

    uint256 rhsX = Fr.unwrap(pairingPoints[4]);
    rhsX |= Fr.unwrap(pairingPoints[5]) << 136;

    uint256 rhsY = Fr.unwrap(pairingPoints[6]);
    rhsY |= Fr.unwrap(pairingPoints[7]) << 136;

    require(lhsX < Q && lhsY < Q && rhsX < Q && rhsY < Q, Errors.ValueGeGroupOrder());

    lhs.x = lhsX;
    lhs.y = lhsY;
    rhs.x = rhsX;
    rhs.y = rhsY;
}

function generateRecursionSeparator(
    Fr[PAIRING_POINTS_SIZE] memory proofPairingPoints,
    Honk.G1Point memory accLhs,
    Honk.G1Point memory accRhs
) pure returns (Fr recursionSeparator) {

    (Honk.G1Point memory proofLhs, Honk.G1Point memory proofRhs) = convertPairingPointsToG1(proofPairingPoints);

    uint256[8] memory recursionSeparatorElements;

    recursionSeparatorElements[0] = proofLhs.x;
    recursionSeparatorElements[1] = proofLhs.y;
    recursionSeparatorElements[2] = proofRhs.x;
    recursionSeparatorElements[3] = proofRhs.y;

    recursionSeparatorElements[4] = accLhs.x;
    recursionSeparatorElements[5] = accLhs.y;
    recursionSeparatorElements[6] = accRhs.x;
    recursionSeparatorElements[7] = accRhs.y;

    recursionSeparator = FrLib.from(uint256(keccak256(abi.encodePacked(recursionSeparatorElements))) % P);
}

function mulWithSeperator(Honk.G1Point memory basePoint, Honk.G1Point memory other, Fr recursionSeperator)
    view
    returns (Honk.G1Point memory)
{
    Honk.G1Point memory result;

    result = ecMul(recursionSeperator, basePoint);
    result = ecAdd(result, other);

    return result;
}

function ecMul(Fr value, Honk.G1Point memory point) view returns (Honk.G1Point memory) {
    Honk.G1Point memory result;

    assembly ("memory-safe") {
        let free := mload(0x40)

        mstore(free, mload(point))
        mstore(add(free, 0x20), mload(add(point, 0x20)))

        mstore(add(free, 0x40), value)

        let success := staticcall(gas(), 0x07, free, 0x60, free, 0x40)
        if iszero(success) {
            revert(0, 0)
        }

        mstore(result, mload(free))
        mstore(add(result, 0x20), mload(add(free, 0x20)))

        mstore(0x40, add(free, 0x60))
    }

    return result;
}

function ecAdd(Honk.G1Point memory lhs, Honk.G1Point memory rhs) view returns (Honk.G1Point memory) {
    Honk.G1Point memory result;

    assembly ("memory-safe") {
        let free := mload(0x40)

        mstore(free, mload(lhs))
        mstore(add(free, 0x20), mload(add(lhs, 0x20)))

        mstore(add(free, 0x40), mload(rhs))
        mstore(add(free, 0x60), mload(add(rhs, 0x20)))

        let success := staticcall(gas(), 0x06, free, 0x80, free, 0x40)
        if iszero(success) { revert(0, 0) }

        mstore(result, mload(free))
        mstore(add(result, 0x20), mload(add(free, 0x20)))

        mstore(0x40, add(free, 0x80))
    }

    return result;
}

function rejectPointAtInfinity(Honk.G1Point memory point) pure {
    require((point.x | point.y) != 0, Errors.PointAtInfinity());
}

function arePairingPointsDefault(Fr[PAIRING_POINTS_SIZE] memory pairingPoints) pure returns (bool) {
    uint256 acc = 0;
    for (uint256 i = 0; i < PAIRING_POINTS_SIZE; i++) {
        acc |= Fr.unwrap(pairingPoints[i]);
    }
    return acc == 0;
}

function pairing(Honk.G1Point memory rhs, Honk.G1Point memory lhs) view returns (bool decodedResult) {
    bytes memory input = abi.encodePacked(
        rhs.x,
        rhs.y,

        uint256(0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2),
        uint256(0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed),
        uint256(0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b),
        uint256(0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa),
        lhs.x,
        lhs.y,

        uint256(0x260e01b251f6f1c7e7ff4e580791dee8ea51d87a358e038b4efe30fac09383c1),
        uint256(0x0118c4d5b837bcc2bc89b5b398b5974e9f5944073b32078b7e231fec938883b0),
        uint256(0x04fc6369f7110fe3d25156c1bb9a72859cf2a04641f99ba4ee413c80da6a5fe4),
        uint256(0x22febda3c0c0632a56475b4214e5615e11e6dd3f96e6cea2854a87d4dacc5e55)
    );

    (bool success, bytes memory result) = address(0x08).staticcall(input);
    decodedResult = success && abi.decode(result, (bool));
}

abstract contract BaseZKHonkVerifier is IVerifier {
    using FrLib for Fr;

    struct PairingInputs {
        Honk.G1Point P_0;
        Honk.G1Point P_1;
    }

    struct SmallSubgroupIpaIntermediates {
        Fr[SUBGROUP_SIZE] challengePolyLagrange;
        Fr challengePolyEval;
        Fr lagrangeFirst;
        Fr lagrangeLast;
        Fr rootPower;
        Fr[SUBGROUP_SIZE] denominators;
        Fr diff;
    }

    uint256 internal constant NUM_WITNESS_ENTITIES = 8 + NUM_MASKING_POLYNOMIALS;
    uint256 internal constant NUM_ELEMENTS_COMM = 2;
    uint256 internal constant NUM_ELEMENTS_FR = 1;
    uint256 internal constant NUM_LIBRA_EVALUATIONS = 4;

    uint256 internal constant LIBRA_COMMITMENTS = 3;
    uint256 internal constant LIBRA_EVALUATIONS = 4;
    uint256 internal constant LIBRA_UNIVARIATES_LENGTH = 9;

    uint256 internal constant SHIFTED_COMMITMENTS_START = 30;
    uint256 internal constant PERMUTATION_ARGUMENT_VALUE_SEPARATOR = 1 << 28;

    uint256 internal immutable $N;
    uint256 internal immutable $LOG_N;
    uint256 internal immutable $VK_HASH;
    uint256 internal immutable $NUM_PUBLIC_INPUTS;
    uint256 internal immutable $MSMSize;

    constructor(uint256 _N, uint256 _logN, uint256 _vkHash, uint256 _numPublicInputs) {
        $N = _N;
        $LOG_N = _logN;
        $VK_HASH = _vkHash;
        $NUM_PUBLIC_INPUTS = _numPublicInputs;
        $MSMSize = NUMBER_UNSHIFTED_ZK + _logN + LIBRA_COMMITMENTS + 2;
    }

    function verify(bytes calldata proof, bytes32[] calldata publicInputs)
        public
        view
        override
        returns (bool verified)
    {

        uint256 expectedProofSize = calculateProofSize($LOG_N);

        require(
            proof.length == expectedProofSize, Errors.ProofLengthWrongWithLogN($LOG_N, proof.length, expectedProofSize)
        );

        Honk.VerificationKey memory vk = loadVerificationKey();
        Honk.ZKProof memory p = ZKTranscriptLib.loadProof(proof, $LOG_N);

        require(publicInputs.length == vk.publicInputsSize - PAIRING_POINTS_SIZE, Errors.PublicInputsLengthWrong());

        ZKTranscript memory t =
            ZKTranscriptLib.generateTranscript(p, publicInputs, $VK_HASH, $NUM_PUBLIC_INPUTS, $LOG_N);

        t.relationParameters.publicInputsDelta = computePublicInputDelta(
            publicInputs,
            p.pairingPointObject,
            t.relationParameters.beta,
            t.relationParameters.gamma,
            1
        );

        require(verifySumcheck(p, t), Errors.SumcheckFailed());
        require(verifyShplemini(p, vk, t), Errors.ShpleminiFailed());

        verified = true;
    }

    function computePublicInputDelta(
        bytes32[] memory publicInputs,
        Fr[PAIRING_POINTS_SIZE] memory pairingPointObject,
        Fr beta,
        Fr gamma,
        uint256 offset
    ) internal view returns (Fr publicInputDelta) {
        Fr numerator = Fr.wrap(1);
        Fr denominator = Fr.wrap(1);

        Fr numeratorAcc = gamma + (beta * FrLib.from(PERMUTATION_ARGUMENT_VALUE_SEPARATOR + offset));
        Fr denominatorAcc = gamma - (beta * FrLib.from(offset + 1));

        {
            for (uint256 i = 0; i < $NUM_PUBLIC_INPUTS - PAIRING_POINTS_SIZE; i++) {
                Fr pubInput = FrLib.fromBytes32(publicInputs[i]);

                numerator = numerator * (numeratorAcc + pubInput);
                denominator = denominator * (denominatorAcc + pubInput);

                numeratorAcc = numeratorAcc + beta;
                denominatorAcc = denominatorAcc - beta;
            }

            for (uint256 i = 0; i < PAIRING_POINTS_SIZE; i++) {
                Fr pubInput = pairingPointObject[i];

                numerator = numerator * (numeratorAcc + pubInput);
                denominator = denominator * (denominatorAcc + pubInput);

                numeratorAcc = numeratorAcc + beta;
                denominatorAcc = denominatorAcc - beta;
            }
        }

        publicInputDelta = FrLib.div(numerator, denominator);
    }

    function verifySumcheck(Honk.ZKProof memory proof, ZKTranscript memory tp) internal view returns (bool verified) {
        Fr roundTargetSum = tp.libraChallenge * proof.libraSum;
        Fr powPartialEvaluation = Fr.wrap(1);

        for (uint256 round; round < $LOG_N; ++round) {
            Fr[ZK_BATCHED_RELATION_PARTIAL_LENGTH] memory roundUnivariate = proof.sumcheckUnivariates[round];
            Fr totalSum = roundUnivariate[0] + roundUnivariate[1];
            require(totalSum == roundTargetSum, Errors.SumcheckFailed());

            Fr roundChallenge = tp.sumCheckUChallenges[round];

            roundTargetSum = computeNextTargetSum(roundUnivariate, roundChallenge);
            powPartialEvaluation =
                powPartialEvaluation * (Fr.wrap(1) + roundChallenge * (tp.gateChallenges[round] - Fr.wrap(1)));
        }

        Fr[NUMBER_OF_ENTITIES] memory relationsEvaluations;
        for (uint256 i = 0; i < NUMBER_OF_ENTITIES; i++) {
            relationsEvaluations[i] = proof.sumcheckEvaluations[i + NUM_MASKING_POLYNOMIALS];
        }
        Fr grandHonkRelationSum = RelationsLib.accumulateRelationEvaluations(
            relationsEvaluations, tp.relationParameters, tp.alphas, powPartialEvaluation
        );

        Fr evaluation = Fr.wrap(1);
        for (uint256 i = 2; i < $LOG_N; i++) {
            evaluation = evaluation * tp.sumCheckUChallenges[i];
        }

        grandHonkRelationSum =
            grandHonkRelationSum * (Fr.wrap(1) - evaluation) + proof.libraEvaluation * tp.libraChallenge;
        verified = (grandHonkRelationSum == roundTargetSum);
    }

    function computeNextTargetSum(Fr[ZK_BATCHED_RELATION_PARTIAL_LENGTH] memory roundUnivariates, Fr roundChallenge)
        internal
        view
        returns (Fr targetSum)
    {
        Fr[ZK_BATCHED_RELATION_PARTIAL_LENGTH] memory BARYCENTRIC_LAGRANGE_DENOMINATORS = [
            Fr.wrap(0x0000000000000000000000000000000000000000000000000000000000009d80),
            Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffec51),
            Fr.wrap(0x00000000000000000000000000000000000000000000000000000000000005a0),
            Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effffd31),
            Fr.wrap(0x0000000000000000000000000000000000000000000000000000000000000240),
            Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effffd31),
            Fr.wrap(0x00000000000000000000000000000000000000000000000000000000000005a0),
            Fr.wrap(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffec51),
            Fr.wrap(0x0000000000000000000000000000000000000000000000000000000000009d80)
        ];

        Fr numeratorValue = Fr.wrap(1);
        for (uint256 i = 0; i < ZK_BATCHED_RELATION_PARTIAL_LENGTH; ++i) {
            numeratorValue = numeratorValue * (roundChallenge - Fr.wrap(i));
        }

        Fr[ZK_BATCHED_RELATION_PARTIAL_LENGTH] memory denominatorInverses;
        for (uint256 i = 0; i < ZK_BATCHED_RELATION_PARTIAL_LENGTH; ++i) {
            denominatorInverses[i] = FrLib.invert(BARYCENTRIC_LAGRANGE_DENOMINATORS[i] * (roundChallenge - Fr.wrap(i)));
        }

        for (uint256 i = 0; i < ZK_BATCHED_RELATION_PARTIAL_LENGTH; ++i) {
            targetSum = targetSum + roundUnivariates[i] * denominatorInverses[i];
        }

        targetSum = targetSum * numeratorValue;
    }

    function verifyShplemini(Honk.ZKProof memory proof, Honk.VerificationKey memory vk, ZKTranscript memory tp)
        internal
        view
        returns (bool verified)
    {
        CommitmentSchemeLib.ShpleminiIntermediates memory mem;

        Fr[] memory powers_of_evaluation_challenge = CommitmentSchemeLib.computeSquares(tp.geminiR, $LOG_N);

        Fr[] memory scalars = new Fr[]($MSMSize);
        Honk.G1Point[] memory commitments = new Honk.G1Point[]($MSMSize);

        mem.posInvertedDenominator = (tp.shplonkZ - powers_of_evaluation_challenge[0]).invert();
        mem.negInvertedDenominator = (tp.shplonkZ + powers_of_evaluation_challenge[0]).invert();

        mem.unshiftedScalar = mem.posInvertedDenominator + (tp.shplonkNu * mem.negInvertedDenominator);
        mem.shiftedScalar =
            tp.geminiR.invert() * (mem.posInvertedDenominator - (tp.shplonkNu * mem.negInvertedDenominator));

        scalars[0] = Fr.wrap(1);
        commitments[0] = proof.shplonkQ;

        mem.batchingChallenge = Fr.wrap(1);
        mem.batchedEvaluation = Fr.wrap(0);

        mem.unshiftedScalarNeg = mem.unshiftedScalar.neg();
        mem.shiftedScalarNeg = mem.shiftedScalar.neg();

        for (uint256 i = 1; i <= NUMBER_UNSHIFTED_ZK; ++i) {
            scalars[i] = mem.unshiftedScalarNeg * mem.batchingChallenge;
            mem.batchedEvaluation = mem.batchedEvaluation
                + (proof.sumcheckEvaluations[i - NUM_MASKING_POLYNOMIALS] * mem.batchingChallenge);
            mem.batchingChallenge = mem.batchingChallenge * tp.rho;
        }

        for (uint256 i = 0; i < NUMBER_TO_BE_SHIFTED; ++i) {
            uint256 scalarOff = i + SHIFTED_COMMITMENTS_START;
            uint256 evaluationOff = i + NUMBER_UNSHIFTED_ZK;

            scalars[scalarOff] = scalars[scalarOff] + (mem.shiftedScalarNeg * mem.batchingChallenge);
            mem.batchedEvaluation =
                mem.batchedEvaluation + (proof.sumcheckEvaluations[evaluationOff] * mem.batchingChallenge);
            mem.batchingChallenge = mem.batchingChallenge * tp.rho;
        }

        commitments[1] = proof.geminiMaskingPoly;

        commitments[2] = vk.qm;
        commitments[3] = vk.qc;
        commitments[4] = vk.ql;
        commitments[5] = vk.qr;
        commitments[6] = vk.qo;
        commitments[7] = vk.q4;
        commitments[8] = vk.qLookup;
        commitments[9] = vk.qArith;
        commitments[10] = vk.qDeltaRange;
        commitments[11] = vk.qElliptic;
        commitments[12] = vk.qMemory;
        commitments[13] = vk.qNnf;
        commitments[14] = vk.qPoseidon2External;
        commitments[15] = vk.qPoseidon2Internal;
        commitments[16] = vk.s1;
        commitments[17] = vk.s2;
        commitments[18] = vk.s3;
        commitments[19] = vk.s4;
        commitments[20] = vk.id1;
        commitments[21] = vk.id2;
        commitments[22] = vk.id3;
        commitments[23] = vk.id4;
        commitments[24] = vk.t1;
        commitments[25] = vk.t2;
        commitments[26] = vk.t3;
        commitments[27] = vk.t4;
        commitments[28] = vk.lagrangeFirst;
        commitments[29] = vk.lagrangeLast;

        commitments[30] = proof.w1;
        commitments[31] = proof.w2;
        commitments[32] = proof.w3;
        commitments[33] = proof.w4;
        commitments[34] = proof.zPerm;
        commitments[35] = proof.lookupInverses;
        commitments[36] = proof.lookupReadCounts;
        commitments[37] = proof.lookupReadTags;

        Fr[] memory foldPosEvaluations = CommitmentSchemeLib.computeFoldPosEvaluations(
            tp.sumCheckUChallenges,
            mem.batchedEvaluation,
            proof.geminiAEvaluations,
            powers_of_evaluation_challenge,
            $LOG_N
        );

        mem.constantTermAccumulator = foldPosEvaluations[0] * mem.posInvertedDenominator;
        mem.constantTermAccumulator =
            mem.constantTermAccumulator + (proof.geminiAEvaluations[0] * tp.shplonkNu * mem.negInvertedDenominator);

        mem.batchingChallenge = tp.shplonkNu.sqr();
        uint256 boundary = NUMBER_UNSHIFTED_ZK + 1;

        for (uint256 i = 0; i < $LOG_N - 1; ++i) {
            bool dummy_round = i >= ($LOG_N - 1);

            if (!dummy_round) {

                mem.posInvertedDenominator = (tp.shplonkZ - powers_of_evaluation_challenge[i + 1]).invert();
                mem.negInvertedDenominator = (tp.shplonkZ + powers_of_evaluation_challenge[i + 1]).invert();

                mem.scalingFactorPos = mem.batchingChallenge * mem.posInvertedDenominator;
                mem.scalingFactorNeg = mem.batchingChallenge * tp.shplonkNu * mem.negInvertedDenominator;
                scalars[boundary + i] = mem.scalingFactorNeg.neg() + mem.scalingFactorPos.neg();

                Fr accumContribution = mem.scalingFactorNeg * proof.geminiAEvaluations[i + 1];
                accumContribution = accumContribution + mem.scalingFactorPos * foldPosEvaluations[i + 1];
                mem.constantTermAccumulator = mem.constantTermAccumulator + accumContribution;
            }

            mem.batchingChallenge = mem.batchingChallenge * tp.shplonkNu * tp.shplonkNu;

            commitments[boundary + i] = proof.geminiFoldComms[i];
        }

        boundary += $LOG_N - 1;

        mem.denominators[0] = Fr.wrap(1).div(tp.shplonkZ - tp.geminiR);
        mem.denominators[1] = Fr.wrap(1).div(tp.shplonkZ - SUBGROUP_GENERATOR * tp.geminiR);
        mem.denominators[2] = mem.denominators[0];
        mem.denominators[3] = mem.denominators[0];

        for (uint256 i = 0; i < LIBRA_EVALUATIONS; i++) {
            Fr scalingFactor = mem.denominators[i] * mem.batchingChallenge;
            mem.batchingScalars[i] = scalingFactor.neg();
            mem.batchingChallenge = mem.batchingChallenge * tp.shplonkNu;
            mem.constantTermAccumulator = mem.constantTermAccumulator + scalingFactor * proof.libraPolyEvals[i];
        }
        scalars[boundary] = mem.batchingScalars[0];
        scalars[boundary + 1] = mem.batchingScalars[1] + mem.batchingScalars[2];
        scalars[boundary + 2] = mem.batchingScalars[3];

        for (uint256 i = 0; i < LIBRA_COMMITMENTS; i++) {
            commitments[boundary++] = proof.libraCommitments[i];
        }

        commitments[boundary] = Honk.G1Point({x: 1, y: 2});
        scalars[boundary++] = mem.constantTermAccumulator;

        require(
            checkEvalsConsistency(proof.libraPolyEvals, tp.geminiR, tp.sumCheckUChallenges, proof.libraEvaluation),
            Errors.ConsistencyCheckFailed()
        );

        Honk.G1Point memory quotient_commitment = proof.kzgQuotient;

        commitments[boundary] = quotient_commitment;
        scalars[boundary] = tp.shplonkZ;

        PairingInputs memory pair;
        pair.P_0 = batchMul(commitments, scalars);
        pair.P_1 = negateInplace(quotient_commitment);

        if (!arePairingPointsDefault(proof.pairingPointObject)) {
            Fr recursionSeparator = generateRecursionSeparator(proof.pairingPointObject, pair.P_0, pair.P_1);
            (Honk.G1Point memory P_0_other, Honk.G1Point memory P_1_other) =
                convertPairingPointsToG1(proof.pairingPointObject);

            rejectPointAtInfinity(P_0_other);
            rejectPointAtInfinity(P_1_other);

            pair.P_0 = mulWithSeperator(pair.P_0, P_0_other, recursionSeparator);
            pair.P_1 = mulWithSeperator(pair.P_1, P_1_other, recursionSeparator);
        }

        return pairing(pair.P_0, pair.P_1);
    }

    function checkEvalsConsistency(
        Fr[LIBRA_EVALUATIONS] memory libraPolyEvals,
        Fr geminiR,
        Fr[CONST_PROOF_SIZE_LOG_N] memory uChallenges,
        Fr libraEval
    ) internal view returns (bool check) {
        Fr one = Fr.wrap(1);
        Fr vanishingPolyEval = geminiR.pow(SUBGROUP_SIZE) - one;
        require(vanishingPolyEval != Fr.wrap(0), Errors.GeminiChallengeInSubgroup());

        SmallSubgroupIpaIntermediates memory mem;
        mem.challengePolyLagrange[0] = one;
        for (uint256 round = 0; round < $LOG_N; round++) {
            uint256 currIdx = 1 + LIBRA_UNIVARIATES_LENGTH * round;
            mem.challengePolyLagrange[currIdx] = one;
            for (uint256 idx = currIdx + 1; idx < currIdx + LIBRA_UNIVARIATES_LENGTH; idx++) {
                mem.challengePolyLagrange[idx] = mem.challengePolyLagrange[idx - 1] * uChallenges[round];
            }
        }

        mem.rootPower = one;
        mem.challengePolyEval = Fr.wrap(0);
        for (uint256 idx = 0; idx < SUBGROUP_SIZE; idx++) {
            mem.denominators[idx] = mem.rootPower * geminiR - one;
            mem.denominators[idx] = mem.denominators[idx].invert();
            mem.challengePolyEval = mem.challengePolyEval + mem.challengePolyLagrange[idx] * mem.denominators[idx];
            mem.rootPower = mem.rootPower * SUBGROUP_GENERATOR_INVERSE;
        }

        Fr numerator = vanishingPolyEval * Fr.wrap(SUBGROUP_SIZE).invert();
        mem.challengePolyEval = mem.challengePolyEval * numerator;
        mem.lagrangeFirst = mem.denominators[0] * numerator;
        mem.lagrangeLast = mem.denominators[SUBGROUP_SIZE - 1] * numerator;

        mem.diff = mem.lagrangeFirst * libraPolyEvals[2];

        mem.diff = mem.diff + (geminiR - SUBGROUP_GENERATOR_INVERSE)
            * (libraPolyEvals[1] - libraPolyEvals[2] - libraPolyEvals[0] * mem.challengePolyEval);
        mem.diff = mem.diff + mem.lagrangeLast * (libraPolyEvals[2] - libraEval) - vanishingPolyEval * libraPolyEvals[3];

        check = mem.diff == Fr.wrap(0);
    }

    function batchMul(Honk.G1Point[] memory base, Fr[] memory scalars)
        internal
        view
        returns (Honk.G1Point memory result)
    {
        uint256 limit = $MSMSize;

        for (uint256 i = 0; i < limit; ++i) {
            rejectPointAtInfinity(base[i]);
        }

        bool success = true;
        assembly ("memory-safe") {
            let free := mload(0x40)

            let count := 0x01
            for {} lt(count, add(limit, 1)) { count := add(count, 1) } {

                let base_base := add(base, mul(count, 0x20))
                let scalar_base := add(scalars, mul(count, 0x20))

                mstore(add(free, 0x40), mload(mload(base_base)))
                mstore(add(free, 0x60), mload(add(0x20, mload(base_base))))

                mstore(add(free, 0x80), mload(scalar_base))

                success := and(success, staticcall(gas(), 7, add(free, 0x40), 0x60, add(free, 0x40), 0x40))

                success := and(success, staticcall(gas(), 6, free, 0x80, free, 0x40))
            }

            mstore(result, mload(free))
            mstore(add(result, 0x20), mload(add(free, 0x20)))
        }

        require(success, Errors.ShpleminiFailed());
    }

    function calculateProofSize(uint256 logN) internal pure returns (uint256) {

        uint256 proofLength = NUM_WITNESS_ENTITIES * NUM_ELEMENTS_COMM;
        proofLength += NUM_ELEMENTS_COMM * 3;

        proofLength += logN * ZK_BATCHED_RELATION_PARTIAL_LENGTH * NUM_ELEMENTS_FR;
        proofLength += NUMBER_OF_ENTITIES_ZK * NUM_ELEMENTS_FR;

        proofLength += NUM_ELEMENTS_FR * 2;
        proofLength += logN * NUM_ELEMENTS_FR;
        proofLength += NUM_LIBRA_EVALUATIONS * NUM_ELEMENTS_FR;

        proofLength += (logN - 1) * NUM_ELEMENTS_COMM;
        proofLength += NUM_ELEMENTS_COMM * 2;

        proofLength += PAIRING_POINTS_SIZE;

        return proofLength * 32;
    }

    function loadVerificationKey() internal pure virtual returns (Honk.VerificationKey memory);
}

contract HonkVerifier is BaseZKHonkVerifier(N, LOG_N, VK_HASH, NUMBER_OF_PUBLIC_INPUTS) {
     function loadVerificationKey() internal pure override returns (Honk.VerificationKey memory) {
       return HonkVerificationKey.loadVerificationKey();
    }
}
