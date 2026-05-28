// Off-chain helper: computes the canonical Poseidon2 commitment +
// initial chain key for a given identity_secret. The Python service runs
// `nargo execute` against this circuit and reads the pub return value to
// learn the canonical commitment (no Python Poseidon2 implementation
// required, which removes a cross-language correctness risk).
//
// Domain separator must match circuits/chain_position/src/main.nr.

use poseidon::poseidon2;

global DOMAIN: Field = 0x6c692d617574682d6b65796368696e;

fn main(secret: Field) -> pub [Field; 2] {
    let commitment = poseidon2::Poseidon2::hash([secret], 1);
    let k0 = poseidon2::Poseidon2::hash([secret, DOMAIN], 2);
    [commitment, k0]
}
