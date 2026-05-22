// ZK Circuit: Hash Chain Position Proof (Poseidon2-flavored, ZK-native).
//
// Proves "I know identity_secret whose Poseidon2 commitment is `commitment`,
// AND I have evolved a forward-secure key chain to position `epoch` whose
// commitment matches `epoch_key_hash`," WITHOUT revealing identity_secret,
// the current key, or the epoch.
//
// Public inputs:  (commitment, epoch_key_hash)
// Private inputs: (identity_secret, current_key, epoch)
//
// Constraints:
//   1. commitment = Poseidon2([identity_secret])             -- identity binding
//   2. K_0 = Poseidon2([identity_secret, DOMAIN])            -- initial key
//   3. iterate K_{i+1} = Poseidon2([K_i]); at i == epoch
//      assert K_i == current_key                             -- chain position
//   4. epoch_key_hash = Poseidon2([EPOCH_DOMAIN, current_key])

use poseidon::poseidon2;

global MAX_EPOCH: u32 = 256;

// Domain separators encoded as Field elements (same byte semantics as the
// off-chain HMAC domains, just packed into a single Field per tag).
global DOMAIN: Field = 0x6c692d617574682d6b65796368696e;
global EPOCH_DOMAIN: Field = 0x6c692d617574682d65706f63683a;

fn h1(x: Field) -> Field {
    poseidon2::Poseidon2::hash([x], 1)
}

fn h2(a: Field, b: Field) -> Field {
    poseidon2::Poseidon2::hash([a, b], 2)
}

fn main(
    // Public
    commitment: pub Field,
    epoch_key_hash: pub Field,

    // Private
    identity_secret: Field,
    current_key: Field,
    epoch: u32,
) {
    // 1. Identity binding.
    let computed_commitment = h1(identity_secret);
    assert(computed_commitment == commitment, "Identity commitment mismatch");

    // 2. Initial key.
    let k0 = h2(identity_secret, DOMAIN);

    // 3. Walk the chain MAX_EPOCH steps; at i == epoch assert match. The
    //    constant-time loop hides which iteration succeeded; observers
    //    cannot tell `epoch` from this circuit's gates.
    let mut k: Field = k0;
    for i in 0..MAX_EPOCH {
        if (i as u32) == epoch {
            assert(k == current_key, "Chain position key mismatch");
        }
        if (i as u32) < (MAX_EPOCH - 1) {
            k = h1(k);
        }
    }
    assert(epoch < MAX_EPOCH, "Epoch out of range");

    // 4. Epoch key commitment.
    let computed_epoch_hash = h2(EPOCH_DOMAIN, current_key);
    assert(computed_epoch_hash == epoch_key_hash, "Epoch key hash mismatch");
}

#[test]
fn test_identity_commitment() {
    let secret: Field = 42;
    let c = h1(secret);
    assert(c != 0);
    assert(c != secret);
}

#[test]
fn test_chain_step() {
    let secret: Field = 7;
    let k0 = h2(secret, DOMAIN);
    let k1 = h1(k0);
    let k2 = h1(k1);
    assert(k1 != k2);
}
