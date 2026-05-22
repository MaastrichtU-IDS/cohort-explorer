// DUO Permission Hierarchy Circuit
//
// Lightweight circuit for verifying permission compatibility
// without revealing the actual consent permission.
//
// This is useful for quick access checks before generating
// the full compliance proof.
//
// Public inputs:
// - permission_commitment: Hash of consent permission
// - requested_permission: The DUO permission being requested
// - compatible: Boolean result
//
// Private inputs:
// - consent_permission: The actual consent permission

// ============ DUO Permission Codes ============

global PERM_NRES: Field = 0;  // No restriction
global PERM_GRU: Field = 1;   // General research use
global PERM_HMB: Field = 2;   // Health/medical/biomedical
global PERM_DS: Field = 3;    // Disease specific
global PERM_POA: Field = 4;   // Population/ancestry only

// ============ Helper Functions ============

/// Compute commitment of permission using SHA-256.
/// SHA-256(perm || salt) reduced to a field element.
/// This is collision-resistant and pre-image resistant,
/// hiding the private permission from the verifier.
fn hash_permission(perm: Field, salt: Field) -> Field {
    // Pack permission (32 bytes) + salt (32 bytes) = 64 bytes
    let mut buf: [u8; 64] = [0; 64];
    let perm_bytes: [u8; 32] = perm.to_be_bytes();
    let salt_bytes: [u8; 32] = salt.to_be_bytes();
    for i in 0..32 {
        buf[i] = perm_bytes[i];
        buf[32 + i] = salt_bytes[i];
    }
    let hash = std::hash::sha256(buf);
    let mut result: Field = 0;
    for i in 0..32 {
        result = result * 256 + hash[i] as Field;
    }
    result
}

/// Check DUO permission hierarchy
///
/// Hierarchy:
/// NRES (0) allows everything
///   +-- GRU (1) allows HMB, DS
///         +-- HMB (2) allows DS
///               +-- DS (3) terminal
/// POA (4) is isolated (only allows POA)
fn is_compatible(consent: Field, requested: Field) -> bool {
    if consent == requested {
        // Same permission always compatible
        true
    } else if consent == PERM_NRES {
        // NRES allows anything
        true
    } else if consent == PERM_GRU {
        // GRU allows more restrictive uses: HMB, DS
        (requested == PERM_HMB) | (requested == PERM_DS)
    } else if consent == PERM_HMB {
        // HMB allows only DS
        requested == PERM_DS
    } else {
        // DS and POA don't allow anything else
        false
    }
}

/// Get permission level in hierarchy (0 = most permissive)
fn get_permission_level(perm: Field) -> u8 {
    if perm == PERM_NRES {
        0
    } else if perm == PERM_GRU {
        1
    } else if perm == PERM_HMB {
        2
    } else if perm == PERM_DS {
        3
    } else {
        // POA has its own level
        10
    }
}

// ============ Main Circuit ============

/// Verify permission compatibility
///
/// Proves: "I have a consent that allows the requested permission"
/// without revealing which specific permission the consent grants.
fn main(
    // Public inputs
    permission_commitment: pub Field,   // Hash of consent permission
    requested_permission: pub Field,    // DUO permission being requested
    salt: pub Field,                    // Salt used in commitment

    // Private inputs
    consent_permission: Field           // Actual consent permission
) -> pub bool {
    // 1. Verify the commitment matches the private permission
    let computed_commitment = hash_permission(consent_permission, salt);
    assert(computed_commitment == permission_commitment, "Permission commitment mismatch");

    // 2. Check compatibility
    let compatible = is_compatible(consent_permission, requested_permission);

    // 3. Return result (public output)
    compatible
}

/// Extended check including permission level
fn main_with_level(
    permission_commitment: pub Field,
    requested_permission: pub Field,
    salt: pub Field,
    min_level: pub u8,                  // Minimum permission level required
    consent_permission: Field
) -> pub bool {
    // Verify commitment
    let computed_commitment = hash_permission(consent_permission, salt);
    assert(computed_commitment == permission_commitment, "Permission commitment mismatch");

    // Check compatibility
    let compatible = is_compatible(consent_permission, requested_permission);

    // Check level (consent must be at or above min_level)
    let consent_level = get_permission_level(consent_permission);
    let level_ok = consent_level <= min_level;

    compatible & level_ok
}

// ============ Disease-Specific Extension ============

/// For DS (Disease-Specific) permissions, verify disease code match.
/// The requester's requested disease code is public; the consent's
/// disease code is private and verified via commitment.
fn check_disease_specific(
    consent_commitment: pub Field,
    requested_permission: pub Field,
    requested_disease_code: pub Field,
    disease_code_commitment: pub Field,
    salt: pub Field,

    consent_permission: Field,
    consent_disease_code: Field
) -> pub bool {
    // Verify permission commitment
    let computed_perm_commitment = hash_permission(consent_permission, salt);
    assert(computed_perm_commitment == consent_commitment, "Permission commitment mismatch");

    // Verify disease code commitment
    let computed_disease_commitment = hash_permission(consent_disease_code, salt);
    assert(computed_disease_commitment == disease_code_commitment, "Disease code commitment mismatch");

    // Check base compatibility
    let compatible = is_compatible(consent_permission, requested_permission);

    // For DS requests, disease codes must match (or consent is broader)
    if requested_permission == PERM_DS {
        if consent_permission == PERM_DS {
            // Both DS: consent disease code must match requested disease code
            compatible & (consent_disease_code == requested_disease_code)
        } else {
            // Consent is broader (HMB, GRU, NRES), any disease is allowed
            compatible
        }
    } else {
        compatible
    }
}

// ============ Tests ============

#[test]
fn test_permission_hierarchy() {
    // Test all hierarchy relationships
    assert(is_compatible(PERM_NRES, PERM_NRES));
    assert(is_compatible(PERM_NRES, PERM_GRU));
    assert(is_compatible(PERM_NRES, PERM_HMB));
    assert(is_compatible(PERM_NRES, PERM_DS));
    assert(is_compatible(PERM_NRES, PERM_POA));

    assert(is_compatible(PERM_GRU, PERM_GRU));
    assert(is_compatible(PERM_GRU, PERM_HMB));
    assert(is_compatible(PERM_GRU, PERM_DS));
    assert(!is_compatible(PERM_GRU, PERM_NRES));
    assert(!is_compatible(PERM_GRU, PERM_POA));

    assert(is_compatible(PERM_HMB, PERM_HMB));
    assert(is_compatible(PERM_HMB, PERM_DS));
    assert(!is_compatible(PERM_HMB, PERM_GRU));
    assert(!is_compatible(PERM_HMB, PERM_NRES));

    assert(is_compatible(PERM_DS, PERM_DS));
    assert(!is_compatible(PERM_DS, PERM_HMB));
    assert(!is_compatible(PERM_DS, PERM_GRU));

    assert(is_compatible(PERM_POA, PERM_POA));
    assert(!is_compatible(PERM_POA, PERM_GRU));
}

#[test]
fn test_permission_levels() {
    assert(get_permission_level(PERM_NRES) == 0);
    assert(get_permission_level(PERM_GRU) == 1);
    assert(get_permission_level(PERM_HMB) == 2);
    assert(get_permission_level(PERM_DS) == 3);
    assert(get_permission_level(PERM_POA) == 10);
}

#[test]
fn test_main_circuit() {
    let salt: Field = 12345;
    let consent = PERM_GRU;
    let commitment = hash_permission(consent, salt);

    // GRU consent should allow HMB request
    let result = main(commitment, PERM_HMB, salt, consent);
    assert(result);

    // GRU consent should not allow NRES request
    let result2 = main(commitment, PERM_NRES, salt, consent);
    assert(!result2);
}
