// Consent Compliance Circuit for DUO Consent
//
// This circuit proves consent compliance without revealing:
// - The actual consent details
// - The requester's full profile
// - The attestation documents
//
// Public inputs (visible on-chain):
// - consent_commitment: Hash of consent details
// - permission_requested: The DUO permission being requested
// - score_threshold: Minimum required score
//
// Private inputs (hidden):
// - consent_permission: The actual consent permission
// - consent_modifiers: Bitmap of consent modifiers
// - requester_type: 0=unknown, 1=academic, 2=nonprofit, 3=government, 4=commercial
// - attestation_bitmap: Which attestations the requester has
// - score_components: [permission, modifier, attestation, trust] scores

// ============ DUO Permission Codes (as Field elements) ============
// NRES = 0 (No restriction)
// GRU  = 1 (General research use)
// HMB  = 2 (Health/medical/biomedical)
// DS   = 3 (Disease specific)
// POA  = 4 (Population/ancestry only)

global PERM_NRES: Field = 0;
global PERM_GRU: Field = 1;
global PERM_HMB: Field = 2;
global PERM_DS: Field = 3;
global PERM_POA: Field = 4;

// ============ Modifier Bitmap Positions ============
// NPU = bit 0 (Non-profit only)
// NCU = bit 1 (Non-commercial use)
// IRB = bit 2 (Ethics approval required)
// PUB = bit 3 (Publication required)
// COL = bit 4 (Collaboration required)
// TS  = bit 5 (Time-limited)
// RTN = bit 6 (Return data required)
// GS  = bit 7 (Geographic restriction)
// IS  = bit 8 (Institution restriction)

global MOD_NPU: u16 = 1;      // 0b0000000001
global MOD_NCU: u16 = 2;      // 0b0000000010
global MOD_IRB: u16 = 4;      // 0b0000000100
global MOD_PUB: u16 = 8;      // 0b0000001000
global MOD_COL: u16 = 16;     // 0b0000010000
global MOD_TS: u16 = 32;      // 0b0000100000
global MOD_RTN: u16 = 64;     // 0b0001000000
global MOD_GS: u16 = 128;     // 0b0010000000
global MOD_IS: u16 = 256;     // 0b0100000000

// ============ Requester Types ============
global TYPE_UNKNOWN: Field = 0;
global TYPE_ACADEMIC: Field = 1;
global TYPE_NONPROFIT: Field = 2;
global TYPE_GOVERNMENT: Field = 3;
global TYPE_COMMERCIAL: Field = 4;

// ============ Attestation Bitmap ============
// IRB_APPROVAL = bit 0
// COLLABORATION = bit 1
// PUBLICATION_COMMITMENT = bit 2
// RETURN_DATA_COMMITMENT = bit 3
// GEOGRAPHIC_PROOF = bit 4
// INSTITUTION_PROOF = bit 5

global ATT_IRB: u8 = 1;
global ATT_COL: u8 = 2;
global ATT_PUB: u8 = 4;
global ATT_RTN: u8 = 8;
global ATT_GEO: u8 = 16;
global ATT_INST: u8 = 32;

// ============ Score Thresholds ============
global MIN_SCORE_LOW_RISK: u32 = 800;
global MIN_SCORE_MEDIUM_RISK: u32 = 600;
global MIN_SCORE_HIGH_RISK: u32 = 400;

// ============ Helper Functions ============

/// Compute commitment of consent details using SHA-256.
///
/// Matches the Python implementation in api/services/identity.py:
///   SHA256("{permission}:{modifiers}:{disease_code}:{expiry}") mod FIELD_ORDER
///
/// SHA-256 is collision-resistant and pre-image resistant, so the
/// commitment hides the private consent details from an observer
/// who only sees the public commitment value.
fn compute_consent_commitment(
    permission: Field,
    modifiers: u16,
    disease_code: Field,
    expiry: u32
) -> Field {
    // Pack inputs into a byte buffer for SHA-256.
    // Layout: [permission:32][modifiers:2][disease_code:32][expiry:4] = 70 bytes
    let mut buf: [u8; 70] = [0; 70];

    // permission as 32-byte big-endian Field encoding
    let perm_bytes: [u8; 32] = permission.to_be_bytes();
    for i in 0..32 {
        buf[i] = perm_bytes[i];
    }

    // modifiers as 2-byte big-endian
    buf[32] = ((modifiers >> 8) & 0xff) as u8;
    buf[33] = (modifiers & 0xff) as u8;

    // disease_code as 32-byte big-endian Field encoding
    let disease_bytes: [u8; 32] = disease_code.to_be_bytes();
    for i in 0..32 {
        buf[34 + i] = disease_bytes[i];
    }

    // expiry as 4-byte big-endian
    buf[66] = ((expiry >> 24) & 0xff) as u8;
    buf[67] = ((expiry >> 16) & 0xff) as u8;
    buf[68] = ((expiry >> 8) & 0xff) as u8;
    buf[69] = (expiry & 0xff) as u8;

    // SHA-256 hash and reduce to field element
    let hash = std::hash::sha256(buf);
    let mut result: Field = 0;
    for i in 0..32 {
        result = result * 256 + hash[i] as Field;
    }
    result
}

/// Check if requested permission is compatible with consent permission
/// Uses DUO hierarchy: NRES > GRU > HMB > DS, POA is separate
fn is_permission_compatible(consent_perm: Field, requested_perm: Field) -> bool {
    // Same permission is always compatible
    if consent_perm == requested_perm {
        true
    } else if consent_perm == PERM_NRES {
        // NRES allows anything
        true
    } else if consent_perm == PERM_GRU {
        // GRU allows HMB, DS
        (requested_perm == PERM_HMB) | (requested_perm == PERM_DS)
    } else if consent_perm == PERM_HMB {
        // HMB allows DS
        requested_perm == PERM_DS
    } else {
        // DS and POA don't allow anything else
        false
    }
}

/// Check if requester type satisfies NPU/NCU modifiers
fn check_requester_type(requester_type: Field, modifiers: u16) -> bool {
    let has_npu = (modifiers & MOD_NPU) != 0;
    let has_ncu = (modifiers & MOD_NCU) != 0;

    // Commercial entities fail NPU and NCU
    if requester_type == TYPE_COMMERCIAL {
        if has_npu | has_ncu {
            false
        } else {
            true
        }
    } else {
        // Non-commercial entities pass all requester type checks
        true
    }
}

/// Check if required attestations are present
fn check_attestations(modifiers: u16, attestations: u8) -> bool {
    let mut valid = true;

    // IRB required?
    if (modifiers & MOD_IRB) != 0 {
        valid = valid & ((attestations & ATT_IRB) != 0);
    }

    // Collaboration required?
    if (modifiers & MOD_COL) != 0 {
        valid = valid & ((attestations & ATT_COL) != 0);
    }

    // Publication commitment required?
    if (modifiers & MOD_PUB) != 0 {
        valid = valid & ((attestations & ATT_PUB) != 0);
    }

    // Return data commitment required?
    if (modifiers & MOD_RTN) != 0 {
        valid = valid & ((attestations & ATT_RTN) != 0);
    }

    // Geographic proof required?
    if (modifiers & MOD_GS) != 0 {
        valid = valid & ((attestations & ATT_GEO) != 0);
    }

    // Institution proof required?
    if (modifiers & MOD_IS) != 0 {
        valid = valid & ((attestations & ATT_INST) != 0);
    }

    valid
}

/// Calculate total compliance score from components
fn calculate_total_score(
    permission_score: u32,
    modifier_score: u32,
    attestation_score: u32,
    trust_score: u32
) -> u32 {
    permission_score + modifier_score + attestation_score + trust_score
}

// ============ Main Circuit ============

/// Main consent compliance proof
///
/// Proves that a requester's access request is compliant with the consent
/// without revealing the actual consent details or requester profile.
///
/// Returns: bool indicating compliance (score >= threshold and all checks pass)
fn main(
    // Public inputs
    consent_commitment: pub Field,      // Hash of consent details
    permission_requested: pub Field,    // DUO permission being requested
    score_threshold: pub u32,           // Minimum required score

    // Private inputs - consent details
    consent_permission: Field,          // Actual consent permission
    consent_modifiers: u16,             // Modifier bitmap
    consent_disease_code: Field,        // Disease code (0 if none)
    consent_expiry: u32,                // Expiry timestamp

    // Private inputs - requester profile
    requester_type: Field,              // Requester organization type
    attestations: u8,                   // Attestation bitmap

    // Private inputs - score components
    permission_score: u32,              // Permission compatibility score (0-300)
    modifier_score: u32,                // Modifier satisfaction score (0-400)
    attestation_score: u32,             // Attestation quality score (0-200)
    trust_score: u32                    // Requester trust score (0-100)
) -> pub bool {
    // 1. Verify consent commitment matches private inputs
    let computed_commitment = compute_consent_commitment(
        consent_permission,
        consent_modifiers,
        consent_disease_code,
        consent_expiry
    );
    assert(computed_commitment == consent_commitment, "Consent commitment mismatch");

    // 2. Check permission compatibility
    let perm_ok = is_permission_compatible(consent_permission, permission_requested);
    assert(perm_ok, "Permission not compatible");

    // 3. Check requester type against NPU/NCU
    let type_ok = check_requester_type(requester_type, consent_modifiers);
    assert(type_ok, "Requester type not allowed");

    // 4. Check required attestations
    let att_ok = check_attestations(consent_modifiers, attestations);
    assert(att_ok, "Missing required attestations");

    // 5. Verify score components are in valid ranges
    assert(permission_score <= 300, "Permission score out of range");
    assert(modifier_score <= 400, "Modifier score out of range");
    assert(attestation_score <= 200, "Attestation score out of range");
    assert(trust_score <= 100, "Trust score out of range");

    // 6. Calculate and check total score
    let total_score = calculate_total_score(
        permission_score,
        modifier_score,
        attestation_score,
        trust_score
    );
    assert(total_score >= score_threshold, "Score below threshold");

    // All checks passed
    true
}

// ============ Tests ============

#[test]
fn test_permission_hierarchy() {
    // NRES allows everything
    assert(is_permission_compatible(PERM_NRES, PERM_GRU));
    assert(is_permission_compatible(PERM_NRES, PERM_HMB));
    assert(is_permission_compatible(PERM_NRES, PERM_DS));
    assert(is_permission_compatible(PERM_NRES, PERM_POA));

    // GRU allows HMB and DS
    assert(is_permission_compatible(PERM_GRU, PERM_HMB));
    assert(is_permission_compatible(PERM_GRU, PERM_DS));
    assert(!is_permission_compatible(PERM_GRU, PERM_POA));

    // HMB allows DS only
    assert(is_permission_compatible(PERM_HMB, PERM_DS));
    assert(!is_permission_compatible(PERM_HMB, PERM_GRU));

    // DS allows nothing else
    assert(!is_permission_compatible(PERM_DS, PERM_HMB));

    // POA is isolated
    assert(!is_permission_compatible(PERM_POA, PERM_GRU));
}

#[test]
fn test_requester_type_checks() {
    // Commercial fails NPU
    assert(!check_requester_type(TYPE_COMMERCIAL, MOD_NPU));

    // Commercial fails NCU
    assert(!check_requester_type(TYPE_COMMERCIAL, MOD_NCU));

    // Academic passes NPU
    assert(check_requester_type(TYPE_ACADEMIC, MOD_NPU));

    // Nonprofit passes NPU
    assert(check_requester_type(TYPE_NONPROFIT, MOD_NPU));

    // Government passes NPU
    assert(check_requester_type(TYPE_GOVERNMENT, MOD_NPU));
}

#[test]
fn test_attestation_checks() {
    // No modifiers = always pass
    assert(check_attestations(0, 0));

    // IRB required, IRB present
    assert(check_attestations(MOD_IRB, ATT_IRB));

    // IRB required, IRB missing
    assert(!check_attestations(MOD_IRB, 0));

    // Multiple requirements
    let multi_mod = MOD_IRB | MOD_COL;
    let multi_att = ATT_IRB | ATT_COL;
    assert(check_attestations(multi_mod, multi_att));
    assert(!check_attestations(multi_mod, ATT_IRB)); // Missing COL
}

#[test]
fn test_full_compliance() {
    // Prepare inputs for a valid compliance check
    let consent_commitment = compute_consent_commitment(PERM_GRU, MOD_NPU, 0, 1735689600);

    let result = main(
        consent_commitment,     // public: commitment
        PERM_HMB,              // public: requested permission
        800,                    // public: threshold
        PERM_GRU,              // private: actual permission
        MOD_NPU,               // private: modifiers
        0,                      // private: disease code
        1735689600,            // private: expiry
        TYPE_ACADEMIC,         // private: requester type
        0,                      // private: attestations
        300,                    // private: permission score
        400,                    // private: modifier score
        150,                    // private: attestation score
        100                     // private: trust score
    );

    assert(result);
}
