//! Combined SHA-256 + ECDSA UAIR (side-by-side merge).
//!
//! Runs both `Sha256CompressionSliceUair` and `EcdsaUair` inside a
//! single UAIR on disjoint columns of the same trace. One proof
//! attests to both the SHA-256 compression round-trip **and** the
//! ECDSA Shamir scalar multiplication using complete ordinary-projective
//! addition and doubling.
//!
//! ## What this is (and isn't)
//!
//! Structural side-by-side merge: both sub-UAIRs' constraints live on
//! disjoint column ranges of one trace. There is **no** in-circuit
//! cross-binding constraint between SHA's chaining output and the ECDSA scalar
//! inputs. Instead, [`verify_sha_ecdsa_scalar_binding`] derives `u1` and `u2`
//! from public `(r, s)` and the proof-bound public chaining-output cells, then
//! checks the existing public Shamir bit columns exactly. The complete
//! [`verify_sha_ecdsa_application_binding`] composes that check with H6's
//! `R_x mod n = r` result binding.
//!
//! The H8 statement fixture is an honest seven-compression SHA-256 chain with
//! the canonical IV, exact public message, and FIPS padding bound before proof
//! verification. Legacy random-trace builders remain available for focused
//! tamper and structural tests, where their initial state and message words are
//! explicitly fixture supplied.
//!
//! ## Column layout
//!
//! Flat trace = `binary_poly || arbitrary_poly || int` (no
//! arbitrary_poly columns).
//!
//! **binary_poly section**: 20 SHA columns, 9 public and 11 witness.
//!
//! **int section**: 42 columns, 24 public and 18 witness. The ECDSA slice
//! contributes 18 public and 13 witness columns; SHA contributes 6 public and
//! 5 witness integer columns.
//!
//! Both halves' shifts and lookup specs are unioned. Lookup groups
//! come from the SHA half only (ECDSA has no range-checked carries
//! in the no-quotient F_p formulation).
//!
//! ## Selectors and trace length
//!
//! Both halves use row-0 init and end-of-trace final selectors on
//! disjoint columns. Trace length is bounded by ECDSA: needs >
//! 256 (`FINAL_ROW = NUM_SHAMIR_ROUNDS = 256`), so `num_vars >= 9`.
//! SHA needs >= 16 rows; satisfied.
//!
//! ## Quotient-witness convention
//!
//! ECDSA F_p constraints are direct (no quotients) — the proving field is the
//! secp256k1 base prime. SHA's five linear-constraint compensators are witness
//! integer columns whose active rows are pinned to zero in-circuit. Its five
//! bounded carries are packed into the binary-polynomial `W_MU_PACKED` column;
//! binary-column booleanity enforces their bit widths without the unavailable
//! arbitrary-lookup path. ECDSA has no quotient columns at all.

use core::{fmt, marker::PhantomData};

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::{ConstSemiring, semiring::boolean::Boolean};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    BitOp, BitOpSpec, ConstraintBuilder, LookupColumnSpec, PublicColumnLayout,
    PublicStructureError, ShiftSpec, ShiftedBitSliceSpec, TotalColumnLayout, TraceRow, Uair,
    UairSignature, UairTrace, VirtualBinaryPolySource, VirtualBinaryPolySpec,
    ideal::rotation::RotationIdeal,
};

use crate::{
    GenerateRandomTrace,
    ecdsa::{self},
    ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing},
    sha256::{self, Sha256CompressionSliceUair, Sha256Ideal},
};

use crypto_primitives::crypto_bigint_int::Int;

// Re-export for convenience.
pub use crate::ecdsa::{
    EcdsaBoundPublicKey, EcdsaPublicKeyBindingError, EcdsaResultBindingError,
    EcdsaResultBranch, FINAL_ROW,
};

/// Which ECDSA verification scalar owns a public Shamir bit column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdsaVerificationScalar {
    /// `u1 = e * s^-1 mod n`.
    U1,
    /// `u2 = r * s^-1 mod n`.
    U2,
}

/// Exact verifier-derived ECDSA values read from one combined public trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaEcdsaDerivedScalars {
    /// Eight SHA-256 chaining-output words serialized in big-endian order.
    pub digest: [u8; 32],
    /// SEC 1 message representative before reduction modulo `n`.
    pub message_representative: CbUint<EC_FP_INT_LIMBS>,
    /// `message_representative * s^-1 mod n`.
    pub u1: CbUint<EC_FP_INT_LIMBS>,
    /// `r * s^-1 mod n`.
    pub u2: CbUint<EC_FP_INT_LIMBS>,
}

/// Failure while binding the declared SHA output to the public ECDSA scalars.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaEcdsaScalarBindingError {
    /// Compact `r` must be exactly one 32-byte big-endian integer.
    SignatureREncoding { actual_bytes: usize },
    /// SEC 1 rejects `r = 0`.
    SignatureRZero,
    /// SEC 1 rejects `r >= n`.
    SignatureROutOfRange,
    /// Compact `s` must be exactly one 32-byte big-endian integer.
    SignatureSEncoding { actual_bytes: usize },
    /// SEC 1 rejects `s = 0`.
    SignatureSZero,
    /// SEC 1 rejects `s >= n`.
    SignatureSOutOfRange,
    /// A declared chaining-output word is absent from the public trace.
    MissingDigestWord { column: usize, row: usize },
    /// A required public SHA structure cell is absent.
    MissingShaStructureCell { column: &'static str, row: usize },
    /// A public SHA selector, round constant, or slack cell is wrong.
    ShaStructureMismatch { column: &'static str, row: usize },
    /// A required public scalar-selector cell is absent.
    MissingScalarBit { column: usize, row: usize },
    /// A public scalar-selector cell is neither canonical zero nor one.
    NonBooleanScalarBit { column: usize, row: usize },
    /// A public scalar-selector cell is not the exact verifier-derived bit.
    ScalarBitMismatch {
        scalar: EcdsaVerificationScalar,
        column: usize,
        row: usize,
        expected: bool,
    },
    /// A derived companion selector (`PA_B1B2` or `S_ADD`) is wrong.
    CompanionSelectorMismatch {
        column: usize,
        row: usize,
        expected: bool,
    },
}

impl fmt::Display for ShaEcdsaScalarBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureREncoding { actual_bytes } => {
                write!(
                    formatter,
                    "signature scalar r must be exactly 32 bytes, got {actual_bytes}"
                )
            }
            Self::SignatureRZero => formatter.write_str("signature scalar r is zero"),
            Self::SignatureROutOfRange => {
                formatter.write_str("signature scalar r is not below the secp256k1 order")
            }
            Self::SignatureSEncoding { actual_bytes } => {
                write!(
                    formatter,
                    "signature scalar s must be exactly 32 bytes, got {actual_bytes}"
                )
            }
            Self::SignatureSZero => formatter.write_str("signature scalar s is zero"),
            Self::SignatureSOutOfRange => {
                formatter.write_str("signature scalar s is not below the secp256k1 order")
            }
            Self::MissingDigestWord { column, row } => {
                write!(
                    formatter,
                    "missing SHA output word at column {column}, row {row}"
                )
            }
            Self::MissingShaStructureCell { column, row } => {
                write!(
                    formatter,
                    "missing SHA structure cell in {column} at row {row}"
                )
            }
            Self::ShaStructureMismatch { column, row } => {
                write!(formatter, "SHA structure mismatch in {column} at row {row}")
            }
            Self::MissingScalarBit { column, row } => {
                write!(
                    formatter,
                    "missing ECDSA scalar bit at column {column}, row {row}"
                )
            }
            Self::NonBooleanScalarBit { column, row } => write!(
                formatter,
                "non-boolean ECDSA scalar bit at column {column}, row {row}",
            ),
            Self::ScalarBitMismatch {
                scalar,
                column,
                row,
                expected,
            } => write!(
                formatter,
                "{scalar:?} bit mismatch at column {column}, row {row}: expected {expected}",
            ),
            Self::CompanionSelectorMismatch {
                column,
                row,
                expected,
            } => write!(
                formatter,
                "ECDSA companion selector mismatch at column {column}, row {row}: expected {expected}",
            ),
        }
    }
}

impl std::error::Error for ShaEcdsaScalarBindingError {}

/// Failure while binding the public SHA trace to the exact H8 message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaEcdsaHonestShaBindingError {
    /// The H8 statement uses one exact 400-byte message.
    MessageLength { actual_bytes: usize },
    /// The fixed selector/K/slack topology is malformed.
    Structure(ShaEcdsaScalarBindingError),
    /// A required public binary cell is absent.
    MissingCell { column: &'static str, row: usize },
    /// One canonical FIPS 180-4 IV word is wrong.
    IvMismatch { column: &'static str, row: usize },
    /// One of the 112 padded-message words is wrong.
    MessageWordMismatch { row: usize },
    /// A feed-forward addend is not a copy of that compression's input state.
    JunctionMismatch {
        column: &'static str,
        row: usize,
        input_row: usize,
    },
}

impl fmt::Display for ShaEcdsaHonestShaBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MessageLength { actual_bytes } => write!(
                formatter,
                "SHA statement message must be exactly {} bytes, got {actual_bytes}",
                sha256::SEVEN_BLOCK_MESSAGE_BYTES,
            ),
            Self::Structure(error) => write!(formatter, "SHA structure failed: {error}"),
            Self::MissingCell { column, row } => {
                write!(formatter, "missing SHA cell in {column} at row {row}")
            }
            Self::IvMismatch { column, row } => {
                write!(formatter, "SHA IV mismatch in {column} at row {row}")
            }
            Self::MessageWordMismatch { row } => {
                write!(formatter, "SHA padded-message mismatch at PA_M row {row}")
            }
            Self::JunctionMismatch {
                column,
                row,
                input_row,
            } => write!(
                formatter,
                "SHA feed-forward copy mismatch in {column} at row {row} (input row {input_row})",
            ),
        }
    }
}

impl std::error::Error for ShaEcdsaHonestShaBindingError {}

/// Failure from the complete public ECDSA application binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaEcdsaApplicationBindingError {
    /// Digest-to-scalar derivation or public-bit binding failed.
    Scalar(ShaEcdsaScalarBindingError),
    /// The H6 `R_x mod n = r` postcondition failed.
    Result(EcdsaResultBindingError),
}

impl fmt::Display for ShaEcdsaApplicationBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(error) => write!(formatter, "scalar binding failed: {error}"),
            Self::Result(error) => write!(formatter, "result binding failed: {error}"),
        }
    }
}

impl std::error::Error for ShaEcdsaApplicationBindingError {}

/// Successful composition of H7 scalar binding and H6 result binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaEcdsaApplicationBinding {
    /// Values derived from the proof-bound public SHA output and `(r, s)`.
    pub scalars: ShaEcdsaDerivedScalars,
    /// Which canonical `R_x` representative satisfied H6.
    pub result_branch: EcdsaResultBranch,
}

/// Failure from the complete H8 message/signature/public-key binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaEcdsaH8ApplicationBindingError {
    HonestSha(ShaEcdsaHonestShaBindingError),
    Scalar(ShaEcdsaScalarBindingError),
    PublicKey(EcdsaPublicKeyBindingError),
    Result(EcdsaResultBindingError),
}

impl fmt::Display for ShaEcdsaH8ApplicationBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HonestSha(error) => write!(formatter, "honest SHA binding failed: {error}"),
            Self::Scalar(error) => write!(formatter, "scalar binding failed: {error}"),
            Self::PublicKey(error) => write!(formatter, "public-key binding failed: {error}"),
            Self::Result(error) => write!(formatter, "result binding failed: {error}"),
        }
    }
}

impl std::error::Error for ShaEcdsaH8ApplicationBindingError {}

/// Successful H8 composition over the complete public statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaEcdsaH8ApplicationBinding {
    pub scalars: ShaEcdsaDerivedScalars,
    pub public_key: EcdsaBoundPublicKey,
    pub result_branch: EcdsaResultBranch,
}

/// Failure while constructing a combined trace from explicit statement data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaEcdsaTraceBuildError {
    ShaMessage(sha256::Sha256MessageTraceError),
    Scalar(ShaEcdsaScalarBindingError),
    PublicKey(EcdsaPublicKeyBindingError),
}

impl fmt::Display for ShaEcdsaTraceBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShaMessage(error) => write!(formatter, "SHA trace construction failed: {error}"),
            Self::Scalar(error) => write!(formatter, "scalar derivation failed: {error}"),
            Self::PublicKey(error) => write!(formatter, "public-key construction failed: {error}"),
        }
    }
}

impl std::error::Error for ShaEcdsaTraceBuildError {}

const SHA_OUTPUT_START_ROW: usize = sha256::cols::NUM_COMPRESSIONS * sha256::cols::ROWS_PER_COMP;

const SHA_OUTPUT_WORD_CELLS: [(usize, usize); 8] = [
    (cols::PA_A, SHA_OUTPUT_START_ROW + 3),
    (cols::PA_A, SHA_OUTPUT_START_ROW + 2),
    (cols::PA_A, SHA_OUTPUT_START_ROW + 1),
    (cols::PA_A, SHA_OUTPUT_START_ROW),
    (cols::PA_E, SHA_OUTPUT_START_ROW + 3),
    (cols::PA_E, SHA_OUTPUT_START_ROW + 2),
    (cols::PA_E, SHA_OUTPUT_START_ROW + 1),
    (cols::PA_E, SHA_OUTPUT_START_ROW),
];

fn binary_word(cell: &zinc_poly::univariate::binary::BinaryPoly<32>) -> u32 {
    let dense: DensePolynomial<Boolean, 32> = cell.clone().into();
    let mut value = 0_u32;
    for (index, coefficient) in dense.coeffs.iter().enumerate() {
        if coefficient.into_inner() {
            let shift = u32::try_from(index).expect("32-bit coefficient index fits u32");
            value |= 1_u32
                .checked_shl(shift)
                .expect("coefficient index is below 32");
        }
    }
    value
}

/// Read the eight declared chaining-output words from the combined public
/// trace.
///
/// This function does not claim that the trace uses the canonical SHA-256 IV
/// or message padding. It only decodes the output of the compression chain
/// already bound by the proof.
pub fn extract_sha256_output<PolyCoeff: Clone, IntT: Clone>(
    public_trace: &UairTrace<'_, PolyCoeff, IntT, 32>,
) -> Result<[u8; 32], ShaEcdsaScalarBindingError> {
    let mut digest = [0_u8; 32];
    for (word_index, &(column, row)) in SHA_OUTPUT_WORD_CELLS.iter().enumerate() {
        let cell = public_trace
            .binary_poly
            .get(column)
            .and_then(|public_column| public_column.evaluations.get(row))
            .ok_or(ShaEcdsaScalarBindingError::MissingDigestWord { column, row })?;
        let start = word_index * 4;
        digest[start..start + 4].copy_from_slice(&binary_word(cell).to_be_bytes());
    }
    Ok(digest)
}

fn parse_signature_scalar(
    bytes: &[u8],
    is_r: bool,
) -> Result<CbUint<EC_FP_INT_LIMBS>, ShaEcdsaScalarBindingError> {
    if bytes.len() != 32 {
        return Err(if is_r {
            ShaEcdsaScalarBindingError::SignatureREncoding {
                actual_bytes: bytes.len(),
            }
        } else {
            ShaEcdsaScalarBindingError::SignatureSEncoding {
                actual_bytes: bytes.len(),
            }
        });
    }
    let scalar = CbUint::<EC_FP_INT_LIMBS>::from_be_slice(bytes);
    if scalar == CbUint::ZERO {
        return Err(if is_r {
            ShaEcdsaScalarBindingError::SignatureRZero
        } else {
            ShaEcdsaScalarBindingError::SignatureSZero
        });
    }
    if scalar >= ecdsa::SECP256K1_N_UINT {
        return Err(if is_r {
            ShaEcdsaScalarBindingError::SignatureROutOfRange
        } else {
            ShaEcdsaScalarBindingError::SignatureSOutOfRange
        });
    }
    Ok(scalar)
}

fn mul_mod_n(
    left: &CbUint<EC_FP_INT_LIMBS>,
    right: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let product: CbUint<{ EC_FP_INT_LIMBS * 2 }> = left.widening_mul(right).into();
    let order: CbUint<{ EC_FP_INT_LIMBS * 2 }> = ecdsa::SECP256K1_N_UINT.resize();
    let order = NonZero::new(order).expect("secp256k1 order is nonzero");
    let (_, remainder) = product.div_rem_vartime(&order);
    remainder.resize()
}

#[cfg(test)]
fn add_mod_n(
    left: &CbUint<EC_FP_INT_LIMBS>,
    right: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let left: CbUint<{ EC_FP_INT_LIMBS * 2 }> = left.resize();
    let right: CbUint<{ EC_FP_INT_LIMBS * 2 }> = right.resize();
    let sum = left.wrapping_add(&right);
    let order: CbUint<{ EC_FP_INT_LIMBS * 2 }> = ecdsa::SECP256K1_N_UINT.resize();
    let order = NonZero::new(order).expect("secp256k1 order is nonzero");
    let (_, remainder) = sum.div_rem_vartime(&order);
    remainder.resize()
}

/// Derive the SEC 1 verification scalars from a 256-bit digest and compact
/// signature.
pub fn derive_ecdsa_verification_scalars(
    digest: [u8; 32],
    signature_r_be: &[u8],
    signature_s_be: &[u8],
) -> Result<ShaEcdsaDerivedScalars, ShaEcdsaScalarBindingError> {
    let signature_r = parse_signature_scalar(signature_r_be, true)?;
    let signature_s = parse_signature_scalar(signature_s_be, false)?;
    let message_representative = CbUint::<EC_FP_INT_LIMBS>::from_be_slice(&digest);
    let order = Odd::new(ecdsa::SECP256K1_N_UINT).expect("secp256k1 order is odd");
    let s_inverse = signature_s
        .invert_odd_mod(&order)
        .expect("every nonzero scalar below the prime order has an inverse");
    let u1 = mul_mod_n(&message_representative, &s_inverse);
    let u2 = mul_mod_n(&signature_r, &s_inverse);
    Ok(ShaEcdsaDerivedScalars {
        digest,
        message_representative,
        u1,
        u2,
    })
}

fn scalar_bit(value: &CbUint<EC_FP_INT_LIMBS>, row: usize) -> bool {
    let bit = ecdsa::NUM_SHAMIR_ROUNDS - 1 - row;
    let word = bit / 64;
    let offset = bit % 64;
    ((value.as_words()[word] >> offset) & 1) == 1
}

fn public_bit(
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
    column: usize,
    row: usize,
) -> Result<bool, ShaEcdsaScalarBindingError> {
    let value = public_trace
        .int
        .get(column)
        .and_then(|public_column| public_column.evaluations.get(row))
        .ok_or(ShaEcdsaScalarBindingError::MissingScalarBit { column, row })?;
    if *value == Int::from(0_u32) {
        Ok(false)
    } else if *value == Int::from(1_u32) {
        Ok(true)
    } else {
        Err(ShaEcdsaScalarBindingError::NonBooleanScalarBit { column, row })
    }
}

fn sha_int_cell<'a>(
    public_trace: &'a UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
    column: usize,
    name: &'static str,
    row: usize,
) -> Result<&'a Int<EC_FP_INT_LIMBS>, ShaEcdsaScalarBindingError> {
    public_trace
        .int
        .get(column)
        .and_then(|public_column| public_column.evaluations.get(row))
        .ok_or(ShaEcdsaScalarBindingError::MissingShaStructureCell { column: name, row })
}

fn selector_expected(row: usize, width: usize, offset: usize) -> bool {
    let compression = row / sha256::cols::ROWS_PER_COMP;
    let row_in_compression = row % sha256::cols::ROWS_PER_COMP;
    compression < sha256::cols::NUM_COMPRESSIONS
        && (offset..offset + width).contains(&row_in_compression)
}

/// Pin the fixed public layout of the seven-compression SHA research chain.
///
/// The initial state and 16 message words per compression remain arbitrary
/// public statement values. This check fixes the selectors that connect those
/// values to the witness, the canonical SHA-256 round constants, and zero-only
/// message slack. It deliberately does not claim canonical SHA-256 IV or
/// padding semantics.
pub fn verify_sha_ecdsa_chain_structure(
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
) -> Result<(), ShaEcdsaScalarBindingError> {
    let n = public_trace
        .binary_poly
        .get(cols::PA_A)
        .map(|column| column.evaluations.len())
        .ok_or(ShaEcdsaScalarBindingError::MissingShaStructureCell {
            column: "PA_A",
            row: 0,
        })?;

    for row in 0..n {
        let output_prefix = row / sha256::cols::ROWS_PER_COMP == sha256::cols::NUM_COMPRESSIONS
            && row % sha256::cols::ROWS_PER_COMP < 4;
        let selectors = [
            (
                cols::SHA_S_INIT_PREFIX,
                "SHA_S_INIT_PREFIX",
                selector_expected(row, 4, 0) || output_prefix,
            ),
            (
                cols::SHA_S_FEEDFORWARD,
                "SHA_S_FEEDFORWARD",
                selector_expected(row, 4, 64),
            ),
            (
                cols::SHA_S_MSG_INIT,
                "SHA_S_MSG_INIT",
                selector_expected(row, 16, 0),
            ),
            (
                cols::SHA_S_ACTIVE_SCHED,
                "SHA_S_ACTIVE_SCHED",
                selector_expected(row, 48, 0),
            ),
            (
                cols::SHA_S_ACTIVE_UPD,
                "SHA_S_ACTIVE_UPD",
                selector_expected(row, 64, 0),
            ),
        ];
        for (column, name, expected) in selectors {
            if *sha_int_cell(public_trace, column, name, row)? != Int::from(u32::from(expected)) {
                return Err(ShaEcdsaScalarBindingError::ShaStructureMismatch { column: name, row });
            }
        }

        let compression = row / sha256::cols::ROWS_PER_COMP;
        let row_in_compression = row % sha256::cols::ROWS_PER_COMP;
        let expected_k = if compression < sha256::cols::NUM_COMPRESSIONS
            && row_in_compression < sha256::cols::ROUNDS_PER_COMP
        {
            sha256::K_CANONICAL[row_in_compression]
        } else {
            0
        };
        if *sha_int_cell(public_trace, cols::SHA_PA_K, "SHA_PA_K", row)? != Int::from(expected_k) {
            return Err(ShaEcdsaScalarBindingError::ShaStructureMismatch {
                column: "SHA_PA_K",
                row,
            });
        }

        if !selector_expected(row, 16, 0) {
            let message = public_trace
                .binary_poly
                .get(cols::PA_M)
                .and_then(|public_column| public_column.evaluations.get(row))
                .ok_or(ShaEcdsaScalarBindingError::MissingShaStructureCell {
                    column: "PA_M",
                    row,
                })?;
            if binary_word(message) != 0 {
                return Err(ShaEcdsaScalarBindingError::ShaStructureMismatch {
                    column: "PA_M",
                    row,
                });
            }
        }
    }
    Ok(())
}

fn sha_binary_cell(
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
    column: usize,
    name: &'static str,
    row: usize,
) -> Result<u32, ShaEcdsaHonestShaBindingError> {
    public_trace
        .binary_poly
        .get(column)
        .and_then(|values| values.evaluations.get(row))
        .map(binary_word)
        .ok_or(ShaEcdsaHonestShaBindingError::MissingCell { column: name, row })
}

/// Bind the seven-compression trace to the canonical SHA-256 IV, one exact
/// 400-byte message and its canonical padding, and every feed-forward copy.
///
/// Digest correctness is not re-computed natively here. Once these public
/// values are pinned, the existing proof-enforced schedule, round, and
/// feed-forward constraints make the extracted output load-bearing.
pub fn verify_sha_ecdsa_honest_sha_binding(
    message: &[u8],
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
) -> Result<[u8; 32], ShaEcdsaHonestShaBindingError> {
    verify_sha_ecdsa_chain_structure(public_trace)
        .map_err(ShaEcdsaHonestShaBindingError::Structure)?;
    let padded = sha256::pad_seven_block_message(message).map_err(|error| match error {
        sha256::Sha256MessageTraceError::MessageLength { actual_bytes } => {
            ShaEcdsaHonestShaBindingError::MessageLength { actual_bytes }
        }
    })?;

    let iv_a = [
        sha256::IV_CANONICAL[3],
        sha256::IV_CANONICAL[2],
        sha256::IV_CANONICAL[1],
        sha256::IV_CANONICAL[0],
    ];
    let iv_e = [
        sha256::IV_CANONICAL[7],
        sha256::IV_CANONICAL[6],
        sha256::IV_CANONICAL[5],
        sha256::IV_CANONICAL[4],
    ];
    for row in 0..4 {
        for (column, name, expected) in [
            (cols::PA_A, "PA_A", iv_a[row]),
            (cols::PA_E, "PA_E", iv_e[row]),
        ] {
            if sha_binary_cell(public_trace, column, name, row)? != expected {
                return Err(ShaEcdsaHonestShaBindingError::IvMismatch {
                    column: name,
                    row,
                });
            }
        }
    }

    for (word_index, chunk) in padded.chunks_exact(4).enumerate() {
        let compression = word_index / 16;
        let row = compression * sha256::cols::ROWS_PER_COMP + word_index % 16;
        let expected = u32::from_be_bytes(chunk.try_into().expect("four-byte chunk"));
        if sha_binary_cell(public_trace, cols::PA_M, "PA_M", row)? != expected {
            return Err(ShaEcdsaHonestShaBindingError::MessageWordMismatch { row });
        }
    }

    for compression in 0..sha256::cols::NUM_COMPRESSIONS {
        let start = compression * sha256::cols::ROWS_PER_COMP;
        for offset in 0..4 {
            let input_row = start + offset;
            let row = start + sha256::cols::ROUNDS_PER_COMP + offset;
            for (column, name) in [(cols::PA_A, "PA_A"), (cols::PA_E, "PA_E")] {
                let input = sha_binary_cell(public_trace, column, name, input_row)?;
                let junction = sha_binary_cell(public_trace, column, name, row)?;
                if junction != input {
                    return Err(ShaEcdsaHonestShaBindingError::JunctionMismatch {
                        column: name,
                        row,
                        input_row,
                    });
                }
            }
        }
    }

    extract_sha256_output(public_trace).map_err(ShaEcdsaHonestShaBindingError::Structure)
}

/// Bind the declared SHA output to all public ECDSA Shamir selectors.
pub fn verify_sha_ecdsa_scalar_binding(
    signature_r_be: &[u8],
    signature_s_be: &[u8],
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
) -> Result<ShaEcdsaDerivedScalars, ShaEcdsaScalarBindingError> {
    verify_sha_ecdsa_chain_structure(public_trace)?;
    let digest = extract_sha256_output(public_trace)?;
    let scalars = derive_ecdsa_verification_scalars(digest, signature_r_be, signature_s_be)?;

    for row in 0..ecdsa::NUM_SHAMIR_ROUNDS {
        let expected_u1 = scalar_bit(&scalars.u1, row);
        let expected_u2 = scalar_bit(&scalars.u2, row);
        let actual_u1 = public_bit(public_trace, cols::ECDSA_PA_B1, row)?;
        let actual_u2 = public_bit(public_trace, cols::ECDSA_PA_B2, row)?;
        if actual_u1 != expected_u1 {
            return Err(ShaEcdsaScalarBindingError::ScalarBitMismatch {
                scalar: EcdsaVerificationScalar::U1,
                column: cols::ECDSA_PA_B1,
                row,
                expected: expected_u1,
            });
        }
        if actual_u2 != expected_u2 {
            return Err(ShaEcdsaScalarBindingError::ScalarBitMismatch {
                scalar: EcdsaVerificationScalar::U2,
                column: cols::ECDSA_PA_B2,
                row,
                expected: expected_u2,
            });
        }

        let expected_product = expected_u1 && expected_u2;
        let actual_product = public_bit(public_trace, cols::ECDSA_PA_B1B2, row)?;
        if actual_product != expected_product {
            return Err(ShaEcdsaScalarBindingError::CompanionSelectorMismatch {
                column: cols::ECDSA_PA_B1B2,
                row,
                expected: expected_product,
            });
        }

        let expected_add = expected_u1 || expected_u2;
        let actual_add = public_bit(public_trace, cols::ECDSA_S_ADD, row)?;
        if actual_add != expected_add {
            return Err(ShaEcdsaScalarBindingError::CompanionSelectorMismatch {
                column: cols::ECDSA_S_ADD,
                row,
                expected: expected_add,
            });
        }
    }
    Ok(scalars)
}

/// Compose H7 scalar binding with the H6 final-result postcondition.
pub fn verify_sha_ecdsa_application_binding(
    signature_r_be: &[u8],
    signature_s_be: &[u8],
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
) -> Result<ShaEcdsaApplicationBinding, ShaEcdsaApplicationBindingError> {
    let scalars = verify_sha_ecdsa_scalar_binding(signature_r_be, signature_s_be, public_trace)
        .map_err(ShaEcdsaApplicationBindingError::Scalar)?;
    let result_branch = verify_sha_ecdsa_result_binding(signature_r_be, public_trace)
        .map_err(ShaEcdsaApplicationBindingError::Result)?;
    Ok(ShaEcdsaApplicationBinding {
        scalars,
        result_branch,
    })
}

/// Bind the complete H8 public statement `(message, r, s, Q)` to one trace.
pub fn verify_sha_ecdsa_h8_application_binding(
    message: &[u8],
    signature_r_be: &[u8],
    signature_s_be: &[u8],
    q_x_be: &[u8],
    q_y_be: &[u8],
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
) -> Result<ShaEcdsaH8ApplicationBinding, ShaEcdsaH8ApplicationBindingError> {
    let digest = verify_sha_ecdsa_honest_sha_binding(message, public_trace)
        .map_err(ShaEcdsaH8ApplicationBindingError::HonestSha)?;
    let scalars = verify_sha_ecdsa_scalar_binding(signature_r_be, signature_s_be, public_trace)
        .map_err(ShaEcdsaH8ApplicationBindingError::Scalar)?;
    debug_assert_eq!(scalars.digest, digest);
    let public_key = ecdsa::verify_ecdsa_public_key_binding_in_columns(
        q_x_be,
        q_y_be,
        &public_trace.int,
        [
            cols::ECDSA_PA_QX,
            cols::ECDSA_PA_QY,
            cols::ECDSA_PA_QGX,
            cols::ECDSA_PA_QGY,
        ],
    )
    .map_err(ShaEcdsaH8ApplicationBindingError::PublicKey)?;
    let result_branch = verify_sha_ecdsa_result_binding(signature_r_be, public_trace)
        .map_err(ShaEcdsaH8ApplicationBindingError::Result)?;
    Ok(ShaEcdsaH8ApplicationBinding {
        scalars,
        public_key,
        result_branch,
    })
}

/// Bind the combined trace's proof-owned final x-coordinate to the public
/// compact ECDSA scalar `r`.
pub fn verify_sha_ecdsa_result_binding<const D: usize>(
    signature_r_be: &[u8],
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, D>,
) -> Result<EcdsaResultBranch, EcdsaResultBindingError> {
    ecdsa::verify_ecdsa_result_binding_in_column(
        signature_r_be,
        &public_trace.int,
        cols::ECDSA_PA_R_X,
    )
}

// ---------------------------------------------------------------------------
// Column layout for the merged trace.
// ---------------------------------------------------------------------------

pub mod cols {
    // ===== binary_poly (mirrors sha256.rs: 8 pub + 10 witness) =====
    // OV cols are public — overflow witnesses for the rotation-ideal
    // constraints (C1, C2, C4, C6) are verifier-derivable. PA_R_*_COMP
    // are public boundary compensators for the Ch (63) / Maj (64) virtual
    // binary_poly residuals (see sha256.rs doc).
    pub const PA_A: usize = 0;
    pub const PA_E: usize = 1;
    pub const PA_OV_SIG0: usize = 2;
    pub const PA_OV_SIG1: usize = 3;
    pub const PA_OV_LSIG0: usize = 4;
    pub const PA_OV_LSIG1: usize = 5;
    pub const PA_R_CH2_COMP: usize = 6;
    pub const PA_R_MAJ_COMP: usize = 7;
    // Public message-block words (Table 9 row 77). See sha256.rs cols
    // doc for the layout.
    pub const PA_M: usize = 8;
    pub const W_A: usize = 9;
    pub const W_SIG0: usize = 10;
    pub const W_E: usize = 11;
    pub const W_SIG1: usize = 12;
    pub const W_W: usize = 13;
    pub const W_LSIG0: usize = 14;
    pub const W_LSIG1: usize = 15;
    // Ch is split into two AND-operand bit-polys (see sha256.rs doc).
    pub const W_U_EF: usize = 16;
    pub const W_U_NEG_E_G: usize = 17;
    pub const W_MAJ: usize = 18;
    // Packed integer-carry witness column. Replaces the 5 prior int
    // carry columns. See sha256.rs cols doc for bit layout +
    // soundness argument.
    pub const W_MU_PACKED: usize = 19;
    // The Table 9 affine combinations B_1 / B_2 / B_3 are now declared
    // as packed virtual binary_poly columns in `signature()` via
    // `with_virtual_binary_poly_cols` — no committed columns.
    pub const NUM_BIN: usize = 20;
    pub const NUM_BIN_PUB: usize = 9;

    // ===== int section =====
    // SHA publics (0..6) — see sha256.rs cols module for chained-
    // compression layout details. S_INIT_PREFIX/S_FEEDFORWARD replace
    // the old S_INIT/S_FINAL pair. S_MSG_INIT gates C16 (message-init
    // pinning). SHA_S_ACTIVE_SCHED / SHA_S_ACTIVE_UPD are the active-
    // range selectors that pin each linear-constraint compensator to 0
    // on its honest range via the in-circuit `pa_c_* · s_active_* == 0`
    // assert_zero constraints (FF compensators reuse SHA_S_FEEDFORWARD
    // — see sha256.rs cols doc).
    pub const SHA_S_INIT_PREFIX: usize = 0;
    pub const SHA_S_FEEDFORWARD: usize = 1;
    pub const SHA_S_MSG_INIT: usize = 2;
    pub const SHA_PA_K: usize = 3;
    pub const SHA_S_ACTIVE_SCHED: usize = 4;
    pub const SHA_S_ACTIVE_UPD: usize = 5;
    // ECDSA publics (6..23), mirroring ecdsa.rs.
    pub const ECDSA_S_INIT: usize = 6;
    pub const ECDSA_S_ACTIVE: usize = 7;
    pub const ECDSA_S_FINAL: usize = 8;
    pub const ECDSA_S_ADD: usize = 9;
    pub const ECDSA_PA_B1: usize = 10;
    pub const ECDSA_PA_B2: usize = 11;
    pub const ECDSA_PA_B1B2: usize = 12;
    pub const ECDSA_PA_QX: usize = 13;
    pub const ECDSA_PA_QY: usize = 14;
    pub const ECDSA_PA_QGX: usize = 15;
    pub const ECDSA_PA_QGY: usize = 16;
    pub const ECDSA_PA_T_X: usize = 17;
    pub const ECDSA_PA_T_Y: usize = 18;
    pub const ECDSA_PA_R_INIT_X: usize = 19;
    pub const ECDSA_PA_R_INIT_Y: usize = 20;
    pub const ECDSA_PA_R_INIT_Z: usize = 21;
    /// Inverse of P_Z[FINAL_ROW] mod p; only the final-row cell matters
    /// (gated by ECDSA_S_FINAL). See [ecdsa.rs::cols::PA_Z_INV].
    pub const ECDSA_PA_Z_INV: usize = 22;
    /// Affine x-coordinate of P[FINAL_ROW]; the verifier is expected to
    /// check `R_x ≡ r (mod n)` off-protocol against the signature
    /// scalar r. See [ecdsa.rs::cols::PA_R_X].
    pub const ECDSA_PA_R_X: usize = 23;
    pub const NUM_INT_PUB: usize = 24;

    // The 5 prior SHA int carry columns (W_MU_W/A/E/JUNCTION_A/E) are
    // gone — replaced by W_MU_PACKED (binary_poly index 19), with
    // booleanity providing free range-checks. See sha256.rs cols doc.

    // ECDSA witnesses (24..37): state plus ten complete-RCB products,
    // mirroring ecdsa.rs. Doubled coordinates and addition products T3/T5
    // are inlined; T4 stays materialized to cap the maximum degree.
    pub const ECDSA_W_X: usize = 24;
    pub const ECDSA_W_Y: usize = 25;
    pub const ECDSA_W_Z: usize = 26;
    pub const ECDSA_W_D_T0: usize = 27;
    pub const ECDSA_W_D_T1: usize = 28;
    pub const ECDSA_W_D_T2: usize = 29;
    pub const ECDSA_W_D_T3: usize = 30;
    pub const ECDSA_W_D_T4: usize = 31;
    pub const ECDSA_W_D_T5: usize = 32;
    pub const ECDSA_W_A_T0: usize = 33;
    pub const ECDSA_W_A_T1: usize = 34;
    pub const ECDSA_W_A_T2: usize = 35;
    pub const ECDSA_W_A_T4: usize = 36;

    // SHA linear-constraint compensators (witness, 37..42). Each
    // `pa_c_*[k]` carries `−inner_*(2)` mod p so that
    // `(inner_* + pa_c_*) ∈ (X − 2)` holds on every row. Compensator-
    // zero on each constraint's active range is enforced in-circuit by
    // `pa_c_* · s_active_* == 0` (with FF using SHA_S_FEEDFORWARD).
    pub const SHA_PA_C_C7: usize = 37;
    pub const SHA_PA_C_C8: usize = 38;
    pub const SHA_PA_C_C9: usize = 39;
    pub const SHA_PA_C_FF_A: usize = 40;
    pub const SHA_PA_C_FF_E: usize = 41;

    pub const NUM_INT: usize = 42;

    // Flat indices (binary_poly || arbitrary_poly || int).
    pub const FLAT_W_A: usize = W_A;
    pub const FLAT_W_SIG0: usize = W_SIG0;
    pub const FLAT_W_E: usize = W_E;
    pub const FLAT_W_SIG1: usize = W_SIG1;
    pub const FLAT_W_W: usize = W_W;
    pub const FLAT_W_LSIG0: usize = W_LSIG0;
    pub const FLAT_W_LSIG1: usize = W_LSIG1;
    pub const FLAT_W_U_EF: usize = W_U_EF;
    pub const FLAT_W_U_NEG_E_G: usize = W_U_NEG_E_G;
    pub const FLAT_W_MAJ: usize = W_MAJ;
    pub const FLAT_SHA_PA_K: usize = NUM_BIN + SHA_PA_K;
    pub const FLAT_W_MU_PACKED: usize = W_MU_PACKED;
    pub const FLAT_ECDSA_W_X: usize = NUM_BIN + ECDSA_W_X;
    pub const FLAT_ECDSA_W_Y: usize = NUM_BIN + ECDSA_W_Y;
    pub const FLAT_ECDSA_W_Z: usize = NUM_BIN + ECDSA_W_Z;
}

// ---------------------------------------------------------------------------
// The merged UAIR.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ShaEcdsaUair<R>(PhantomData<R>);

impl<R> Uair for ShaEcdsaUair<R>
where
    R: EcdsaFpRing + From<u32>,
{
    type Ideal = Sha256Ideal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(cols::NUM_BIN, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(cols::NUM_BIN_PUB, 0, cols::NUM_INT_PUB);

        // Shifts: union of SHA's and ECDSA's (sorted by source_col by
        // UairSignature::new; insertion order breaks ties — list within
        // a source_col in shift_amount order to mirror sha256.rs).
        let shifts: Vec<ShiftSpec> = vec![
            // === SHA binary_poly shifts ===
            ShiftSpec::new(cols::FLAT_W_A, 1),
            ShiftSpec::new(cols::FLAT_W_A, 2),
            ShiftSpec::new(cols::FLAT_W_A, 4),
            ShiftSpec::new(cols::FLAT_W_SIG0, 3),
            ShiftSpec::new(cols::FLAT_W_E, 1),
            ShiftSpec::new(cols::FLAT_W_E, 2),
            ShiftSpec::new(cols::FLAT_W_E, 4),
            ShiftSpec::new(cols::FLAT_W_SIG1, 3),
            ShiftSpec::new(cols::FLAT_W_W, 3),
            ShiftSpec::new(cols::FLAT_W_W, 9),
            ShiftSpec::new(cols::FLAT_W_W, 16),
            ShiftSpec::new(cols::FLAT_W_LSIG0, 1),
            ShiftSpec::new(cols::FLAT_W_LSIG1, 14),
            ShiftSpec::new(cols::FLAT_W_U_EF, 2),
            ShiftSpec::new(cols::FLAT_W_U_EF, 3),
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 2),
            ShiftSpec::new(cols::FLAT_W_U_NEG_E_G, 3),
            ShiftSpec::new(cols::FLAT_W_MAJ, 2),
            ShiftSpec::new(cols::FLAT_W_MAJ, 3),
            // === SHA int shifts: only PA_K survives. The 3 mu_*
            //     shift specs are gone alongside the dropped int carry
            //     columns — carries now packed in W_MU_PACKED, accessed
            //     via BitOp::ShiftR virtuals declared below.
            ShiftSpec::new(cols::FLAT_SHA_PA_K, 3),
            // === ECDSA int shifts (X, Y, Z by 1 each: down.X[t] = R_{t+1}) ===
            ShiftSpec::new(cols::FLAT_ECDSA_W_X, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Y, 1),
            ShiftSpec::new(cols::FLAT_ECDSA_W_Z, 1),
        ];

        // discussion.
        let lookup_specs: Vec<LookupColumnSpec> = Vec::new();
        // Bit-op virtual columns over W for σ_0/σ_1. Mirrors the SHA
        // standalone UAIR — see `sha256::Sha256CompressionSliceUair::signature`
        // for the full mapping (`Rot(c)` ≡ `ROTR^{32-c}` ≡ multiplication
        // by `X^c mod (X^32 − 1)`). All six specs target FLAT_W_W.
        let bit_op_specs: Vec<BitOpSpec> = vec![
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(25)),    // σ_0: ROTR^7
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(14)),    // σ_0: ROTR^18
            BitOpSpec::new(cols::FLAT_W_W, BitOp::ShiftR(3)),  // σ_0: SHR^3
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(15)),    // σ_1: ROTR^17
            BitOpSpec::new(cols::FLAT_W_W, BitOp::Rot(13)),    // σ_1: ROTR^19
            BitOpSpec::new(cols::FLAT_W_W, BitOp::ShiftR(10)), // σ_1: SHR^10
            // Bit-op virtuals over W_MU_PACKED for extracting the 5
            // chained-comp carries. See sha256.rs cols doc.
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(2)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(5)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(8)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(9)),
            BitOpSpec::new(cols::FLAT_W_MU_PACKED, BitOp::ShiftR(10)),
        ];

        // Witness-relative col indices (post-public) for virtual specs.
        const W_A_WIT_IDX: usize = cols::W_A - cols::NUM_BIN_PUB; // 0
        const W_E_WIT_IDX: usize = cols::W_E - cols::NUM_BIN_PUB; // 2
        const W_U_EF_WIT_IDX: usize = cols::W_U_EF - cols::NUM_BIN_PUB; // 7
        const W_U_NEG_E_G_WIT_IDX: usize = cols::W_U_NEG_E_G - cols::NUM_BIN_PUB; // 8
        const W_MAJ_WIT_IDX: usize = cols::W_MAJ - cols::NUM_BIN_PUB; // 9
        // Order = the spec_idx that VirtualBinaryPolySource uses (must
        // match the ShiftSpec ordering above; UairSignature::new sorts
        // shifts by source_col, then shift_amount).
        const SBS_W_A_SH1: usize = 0;
        const SBS_W_A_SH2: usize = 1;
        const SBS_W_E_SH1: usize = 2;
        const SBS_W_E_SH2: usize = 3;
        const SBS_W_U_EF_SH2: usize = 4;
        const SBS_W_U_NEG_E_G_SH2: usize = 5;
        const SBS_W_MAJ_SH2: usize = 6;
        let shifted_bit_slice_specs = vec![
            ShiftedBitSliceSpec::new(W_A_WIT_IDX, 1),
            ShiftedBitSliceSpec::new(W_A_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_E_WIT_IDX, 1),
            ShiftedBitSliceSpec::new(W_E_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_U_EF_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_U_NEG_E_G_WIT_IDX, 2),
            ShiftedBitSliceSpec::new(W_MAJ_WIT_IDX, 2),
        ];
        // Virtual binary_poly cols — mirrors sha256.rs (Ch eq 62/63 and
        // Maj eq 64, all anchored at k = t-2). See sha256.rs's signature
        // for the residual definitions and the alt-complement form.
        let virtual_binary_poly_cols = vec![
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH2,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH1,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_U_EF_SH2,
                        },
                    ),
                ],
            },
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_E_SH2,
                        },
                    ),
                    (
                        -1,
                        VirtualBinaryPolySource::SelfWitnessCol {
                            witness_col_idx: W_E_WIT_IDX,
                        },
                    ),
                    (
                        2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_U_NEG_E_G_SH2,
                        },
                    ),
                    (
                        2,
                        VirtualBinaryPolySource::PublicCol {
                            public_col_idx: cols::PA_R_CH2_COMP,
                        },
                    ),
                ],
            },
            VirtualBinaryPolySpec {
                terms: vec![
                    (
                        1,
                        VirtualBinaryPolySource::SelfWitnessCol {
                            witness_col_idx: W_A_WIT_IDX,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_A_SH1,
                        },
                    ),
                    (
                        1,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_A_SH2,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::ShiftedWitnessCol {
                            shifted_spec_idx: SBS_W_MAJ_SH2,
                        },
                    ),
                    (
                        -2,
                        VirtualBinaryPolySource::PublicCol {
                            public_col_idx: cols::PA_R_MAJ_COMP,
                        },
                    ),
                ],
            },
        ];
        UairSignature::new(total, public, shifts, lookup_specs, bit_op_specs)
            .with_shifted_bit_slice_specs(shifted_bit_slice_specs)
            .with_virtual_binary_poly_cols(virtual_binary_poly_cols)
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        // ===================================================================
        // SHA-256 half — mirrors sha256.rs's constrain_general,
        // referencing merged column indices.
        // ===================================================================
        let bp = up.binary_poly;
        let int = up.int;

        let pa_a = &bp[cols::PA_A];
        let pa_e = &bp[cols::PA_E];
        let pa_ov_sig0 = &bp[cols::PA_OV_SIG0];
        let pa_ov_sig1 = &bp[cols::PA_OV_SIG1];
        let pa_ov_lsig0 = &bp[cols::PA_OV_LSIG0];
        let pa_ov_lsig1 = &bp[cols::PA_OV_LSIG1];
        let pa_m = &bp[cols::PA_M];
        let w_a = &bp[cols::W_A];
        let w_sig0 = &bp[cols::W_SIG0];
        let w_e = &bp[cols::W_E];
        let w_sig1 = &bp[cols::W_SIG1];
        let w_big_w = &bp[cols::W_W];
        let w_lsig0 = &bp[cols::W_LSIG0];
        let w_lsig1 = &bp[cols::W_LSIG1];

        let sha_s_init_prefix = &int[cols::SHA_S_INIT_PREFIX];
        // SHA_S_FEEDFORWARD doubles as the C12/C13 compensator-zero
        // selector: it is 1 on the junction window where the
        // feed-forward addition holds honestly (so PA_C_FF_{A,E} must
        // be 0), 0 elsewhere.
        let sha_s_feedforward = &int[cols::SHA_S_FEEDFORWARD];
        let sha_s_msg_init = &int[cols::SHA_S_MSG_INIT];
        let sha_s_active_sched = &int[cols::SHA_S_ACTIVE_SCHED];
        let sha_s_active_upd = &int[cols::SHA_S_ACTIVE_UPD];
        let sha_pa_k = &int[cols::SHA_PA_K];
        let pa_c_c7 = &int[cols::SHA_PA_C_C7];
        let pa_c_c8 = &int[cols::SHA_PA_C_C8];
        let pa_c_c9 = &int[cols::SHA_PA_C_C9];
        let pa_c_ff_a = &int[cols::SHA_PA_C_FF_A];
        let pa_c_ff_e = &int[cols::SHA_PA_C_FF_E];
        // The 5 prior int carry columns are gone — replaced by
        // W_MU_PACKED (binary_poly), accessed below via `up.bp[W_MU_PACKED]`
        // and the BitOp::ShiftR virtuals.
        let w_mu_packed = &bp[cols::W_MU_PACKED];

        // SHA `down` slots (in source-col-ascending order — see signature()).
        // bin slots (19 SHA shifts, then 0 ECDSA shifts on bin). The
        // sh1/sh2 entries on a/e/u_ef/u_¬e_g/Maj are kept in the shift
        // list to feed the booleanity batch's shifted-bit-slice
        // consistency check (declared via `with_shifted_bit_slice_specs`)
        // — they're not consumed by `constrain_general` directly.
        let _down_w_a_sh1 = &down.binary_poly[0];
        let _down_w_a_sh2 = &down.binary_poly[1];
        let down_w_a_sh4 = &down.binary_poly[2];
        let down_w_sig0_sh3 = &down.binary_poly[3];
        let _down_w_e_sh1 = &down.binary_poly[4];
        let _down_w_e_sh2 = &down.binary_poly[5];
        let down_w_e_sh4 = &down.binary_poly[6];
        let down_w_sig1_sh3 = &down.binary_poly[7];
        // Retained to preserve the pre-H8 shift geometry. Standard round
        // binding now consumes the row-local W cell directly.
        let _down_w_w_sh3 = &down.binary_poly[8];
        let down_w_w_sh9 = &down.binary_poly[9];
        let down_w_w_sh16 = &down.binary_poly[10];
        let down_w_lsig0_sh1 = &down.binary_poly[11];
        let down_w_lsig1_sh14 = &down.binary_poly[12];
        let _down_w_u_ef_sh2 = &down.binary_poly[13];
        let down_w_u_ef_sh3 = &down.binary_poly[14];
        let _down_w_u_neg_e_g_sh2 = &down.binary_poly[15];
        let down_w_u_neg_e_g_sh3 = &down.binary_poly[16];
        let _down_w_maj_sh2 = &down.binary_poly[17];
        let down_w_maj_sh3 = &down.binary_poly[18];
        // int slots: legacy SHA pa_K_sh3 (slot 0), then ECDSA X/Y/Z sh1
        // (1, 2, 3). The SHA shift remains only for pre-H8 proof geometry;
        // standard round binding consumes row-local PA_K. The 3 prior SHA
        // mu_* shifts are gone with the dropped carry columns.
        let _down_pa_k_sh3 = &down.int[0];
        let down_ecdsa_x_sh1 = &down.int[1];
        let down_ecdsa_y_sh1 = &down.int[2];
        let down_ecdsa_z_sh1 = &down.int[3];

        // Bit-op virtual columns. With FLAT_W_W < FLAT_W_MU_PACKED,
        // W's 6 bit-ops occupy slots 0-5, then W_MU_PACKED's 5
        // bit-ops occupy slots 6-10.
        let down_w_rot13 = &down.bit_op[0]; // σ_1: ROTR^19
        let down_w_rot14 = &down.bit_op[1]; // σ_0: ROTR^18
        let down_w_rot15 = &down.bit_op[2]; // σ_1: ROTR^17
        let down_w_rot25 = &down.bit_op[3]; // σ_0: ROTR^7
        let down_w_shr3 = &down.bit_op[4]; //  σ_0: SHR^3
        let down_w_shr10 = &down.bit_op[5]; // σ_1: SHR^10
        // Bit-extraction shifts on W_MU_PACKED.
        let down_w_mu_packed_shr2 = &down.bit_op[6];
        let down_w_mu_packed_shr5 = &down.bit_op[7];
        let down_w_mu_packed_shr8 = &down.bit_op[8];
        let down_w_mu_packed_shr9 = &down.bit_op[9];
        let down_w_mu_packed_shr10 = &down.bit_op[10];

        let ideal_rot_xw1 = ideal_from_ref(&Sha256Ideal::<R>::RotXw1);
        let ideal_rot_x2 = ideal_from_ref(&Sha256Ideal::<R>::RotX2(RotationIdeal::new(
            R::ONE + R::ONE,
        )));

        let rho_sig0 = rho_poly::<R>(&[10, 19, 30]);
        let rho_sig1 = rho_poly::<R>(&[7, 21, 26]);
        let two_scalar_sha = const_scalar::<R>(R::ONE + R::ONE);
        // Carry-extraction multipliers (mirror sha256.rs). Each
        // contribution `2^32 · mu_X` is built from
        // `2^32 · ShiftR(k_low) − 2^{32+w} · ShiftR(k_low+w)`.
        let const_2_to_32 = const_scalar::<R>(pow_two::<R>(32));
        let const_2_to_33 = const_scalar::<R>(pow_two::<R>(33));
        let const_2_to_34 = const_scalar::<R>(pow_two::<R>(34));
        let const_2_to_35 = const_scalar::<R>(pow_two::<R>(35));

        let mu_w_contrib = mbs(w_mu_packed, &const_2_to_32).expect("2^32 · w_mu_packed overflow")
            - &mbs(down_w_mu_packed_shr2, &const_2_to_34)
                .expect("2^34 · ShiftR(2)(w_mu_packed) overflow");
        let mu_a_contrib = mbs(down_w_mu_packed_shr2, &const_2_to_32)
            .expect("2^32 · ShiftR(2)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr5, &const_2_to_35)
                .expect("2^35 · ShiftR(5)(w_mu_packed) overflow");
        let mu_e_contrib = mbs(down_w_mu_packed_shr5, &const_2_to_32)
            .expect("2^32 · ShiftR(5)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr8, &const_2_to_35)
                .expect("2^35 · ShiftR(8)(w_mu_packed) overflow");
        let mu_ff_a_contrib = mbs(down_w_mu_packed_shr8, &const_2_to_32)
            .expect("2^32 · ShiftR(8)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr9, &const_2_to_33)
                .expect("2^33 · ShiftR(9)(w_mu_packed) overflow");
        let mu_ff_e_contrib = mbs(down_w_mu_packed_shr9, &const_2_to_32)
            .expect("2^32 · ShiftR(9)(w_mu_packed) overflow")
            - &mbs(down_w_mu_packed_shr10, &const_2_to_33)
                .expect("2^33 · ShiftR(10)(w_mu_packed) overflow");

        // C1: Sigma_0 rotation
        b.assert_in_ideal(
            mbs(w_a, &rho_sig0).expect("a · rho_sig0 overflow")
                - w_sig0
                - &mbs(pa_ov_sig0, &two_scalar_sha).expect("2 · ov_sig0 overflow"),
            &ideal_rot_xw1,
        );

        // C2: Sigma_1 rotation
        b.assert_in_ideal(
            mbs(w_e, &rho_sig1).expect("e · rho_sig1 overflow")
                - w_sig1
                - &mbs(pa_ov_sig1, &two_scalar_sha).expect("2 · ov_sig1 overflow"),
            &ideal_rot_xw1,
        );

        // C4 (was σ_0 (X^32 − 1) ideal-lift): row-local Q[X] equality
        // with bit-XOR overflow correction. See sha256.rs for the
        // derivation. `pa_ov_lsig0` retained — only the modular lift
        // and the C3/C5 right-shift decompositions go away.
        //   ROT^25(W) + ROT^14(W) + SHIFTR^3(W) − lsig0 − 2 · pa_ov_lsig0 == 0
        b.assert_zero(
            down_w_rot25.clone() + down_w_rot14 + down_w_shr3
                - w_lsig0
                - &mbs(pa_ov_lsig0, &two_scalar_sha).expect("2 · ov_lsig0 overflow"),
        );

        // C6 (was σ_1 (X^32 − 1) ideal-lift): σ_1 analogue of C4.
        //   ROT^15(W) + ROT^13(W) + SHIFTR^10(W) − lsig1 − 2 · pa_ov_lsig1 == 0
        b.assert_zero(
            down_w_rot15.clone() + down_w_rot13 + down_w_shr10
                - w_lsig1
                - &mbs(pa_ov_lsig1, &two_scalar_sha).expect("2 · ov_lsig1 overflow"),
        );

        // C7: Message-schedule modular sum. mu_W from up.w_mu_packed
        // bits 0-1 via mu_w_contrib (chained-comp re-anchoring stores
        // each carry at its constraint's anchor row).
        let sched_inner =
            down_w_w_sh16.clone() - w_big_w - down_w_lsig0_sh1 - down_w_w_sh9 - down_w_lsig1_sh14
            + &mu_w_contrib;
        b.assert_in_ideal(sched_inner + pa_c_c7, &ideal_rot_x2);

        // C8: Register-update for `a`. mu_a from bits 2-4 of W_MU_PACKED.
        let a_update_inner = down_w_a_sh4.clone()
            - w_e
            - down_w_sig1_sh3
            - down_w_u_ef_sh3
            - down_w_u_neg_e_g_sh3
            - sha_pa_k
            - w_big_w
            - down_w_sig0_sh3
            - down_w_maj_sh3
            + &mu_a_contrib;
        b.assert_in_ideal(a_update_inner + pa_c_c8, &ideal_rot_x2);

        // C9: Register-update for `e`. mu_e from bits 5-7 of W_MU_PACKED.
        let e_update_inner = down_w_e_sh4.clone()
            - w_a
            - w_e
            - down_w_sig1_sh3
            - down_w_u_ef_sh3
            - down_w_u_neg_e_g_sh3
            - sha_pa_k
            - w_big_w
            + &mu_e_contrib;
        b.assert_in_ideal(e_update_inner + pa_c_c9, &ideal_rot_x2);

        // C13–C15 (B_1/B_2/B_3 materializations) are gone — the
        // residuals are now packed virtual binary_poly columns,
        // declared in `signature()` via `with_virtual_binary_poly_cols`
        // and pinned by the booleanity sumcheck. See sha256.rs's
        // `signature()` for the residual definitions.

        // C10/C11: per-compression init-prefix pinning. See sha256.rs
        // for the chained-compression layout. Subsumes the old per-trace
        // init/final boundary constraints — every compression's init
        // prefix and the final H_N output block are pinned by the same
        // s_init_prefix selector.
        b.assert_zero(sha_s_init_prefix.clone() * &(w_a.clone() - pa_a));
        b.assert_zero(sha_s_init_prefix.clone() * &(w_e.clone() - pa_e));

        // C12/C13: SHA-256 feed-forward addition at each junction
        // window. Mirrors the standalone SHA UAIR — uses the public
        // compensator pattern (pa_c_ff_{a,e}) instead of a multiplicative
        // selector to keep the constraint at degree 1 in the trace MLEs
        // (so the merged UAIR can stay MLE-first eligible). References:
        //   up.w_a            = internal_final_i, j-th component
        //   up.pa_a           = H_i, j-th component (junction copy)
        //   down.w_a^↓4       = w_a[k+4] = H_{i+1}, j-th component (pinned by C10)
        //   up.sha_w_mu_junction_a = carry ∈ {0, 1}
        // mu_ff_a / mu_ff_e from bits 8 / 9 of W_MU_PACKED.
        let ff_a_inner = down_w_a_sh4.clone() - w_a - pa_a + &mu_ff_a_contrib;
        b.assert_in_ideal(ff_a_inner + pa_c_ff_a, &ideal_rot_x2);

        let ff_e_inner = down_w_e_sh4.clone() - w_e - pa_e + &mu_ff_e_contrib;
        b.assert_in_ideal(ff_e_inner + pa_c_ff_e, &ideal_rot_x2);

        // C16: message init (Table 9 row 77). Pin w_W to public message
        // words pa_m at the 16 message-block-seed rows of every
        // compression. Mirrors the standalone SHA UAIR.
        b.assert_zero(sha_s_msg_init.clone() * &(w_big_w.clone() - pa_m));

        // Compensator-zero pinning (C18-C22 in the SHA UAIR doc). Each
        // compensator must be 0 on its constraint's honest active
        // range — outside that range it freely absorbs `−inner(2)` so
        // that `(inner + comp) ∈ (X − 2)` everywhere. The compensators
        // are witness columns; the prover-claimed values are pinned
        // to 0 on the active rows by these in-circuit zero-ideal
        // constraints rather than by an out-of-band public-structure
        // check on a public column.
        //
        // Each constraint is polynomial-degree 2 (witness compensator
        // MLE × public selector MLE), but `assert_zero` constraints
        // are excluded from `count_effective_max_degree`, so MLE-first
        // eligibility is preserved — same pattern as C10/C11/C16.
        b.assert_zero(pa_c_c7.clone() * sha_s_active_sched);
        b.assert_zero(pa_c_c8.clone() * sha_s_active_upd);
        b.assert_zero(pa_c_c9.clone() * sha_s_active_upd);
        b.assert_zero(pa_c_ff_a.clone() * sha_s_feedforward);
        b.assert_zero(pa_c_ff_e.clone() * sha_s_feedforward);

        // C17 (renumbered from C22): high-bits-zero pin on W_MU_PACKED. Forces
        // positions 10..31 of w_mu_packed to be 0 at every row. Combined with
        // booleanity, confines mu_X to declared bit widths. See
        // sha256.rs cols doc for the soundness argument.
        b.assert_zero(down_w_mu_packed_shr10.clone());

        // Complete ECDSA half. Public and witness slices preserve the
        // standalone ecdsa.rs local layout; only their merged offsets differ.
        let ecdsa_public = &int[cols::ECDSA_S_INIT..cols::NUM_INT_PUB];
        let ecdsa_witness = &int[cols::ECDSA_W_X..cols::SHA_PA_C_C7];
        let ecdsa_down = [
            down_ecdsa_x_sh1.clone(),
            down_ecdsa_y_sh1.clone(),
            down_ecdsa_z_sh1.clone(),
        ];
        ecdsa::constrain_rcb_shamir(b, ecdsa_public, ecdsa_witness, &ecdsa_down, &from_ref, &mbs);
    }

    /// Verify both halves' public-column structural properties.
    ///
    /// The two tail-compensator columns (`PA_R_CH2_COMP`,
    /// `PA_R_MAJ_COMP`) must be zero on every inner row. The five
    /// linear-constraint compensators
    /// (`SHA_PA_C_C7`/`C8`/`C9`/`FF_A`/`FF_E`) are no longer in this
    /// list — they are now witness columns and their compensator-zero
    /// pins on each constraint's active range are enforced in-circuit
    /// by the `pa_c_* · sha_s_active_* == 0` assert_zero constraints
    /// in `constrain_general`. The ECDSA half has no compensator
    /// pattern that needs verifier-side inspection.
    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero + num_traits::One + PartialEq,
    {
        let n = 1usize << num_vars;
        for (column_family, expected, actual) in [
            (
                "binary_poly",
                cols::NUM_BIN_PUB,
                public_trace.binary_poly.len(),
            ),
            ("arbitrary_poly", 0, public_trace.arbitrary_poly.len()),
            ("int", cols::NUM_INT_PUB, public_trace.int.len()),
        ] {
            if actual != expected {
                return Err(PublicStructureError::WrongColumnCount {
                    column_family,
                    expected,
                    actual,
                });
            }
        }

        let pa_r_ch2_comp = &public_trace.binary_poly[cols::PA_R_CH2_COMP].evaluations;
        let pa_r_maj_comp = &public_trace.binary_poly[cols::PA_R_MAJ_COMP].evaluations;

        let inner_end = n.saturating_sub(2);
        for k in 0..inner_end {
            if !pa_r_ch2_comp[k].iter().all(|c| !c.into_inner()) {
                return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                    column: "PA_R_CH2_COMP",
                    row: k,
                });
            }
            if !pa_r_maj_comp[k].iter().all(|c| !c.into_inner()) {
                return Err(PublicStructureError::NonZeroOnRequiredZeroRow {
                    column: "PA_R_MAJ_COMP",
                    row: k,
                });
            }
        }

        ecdsa::verify_ecdsa_public_int_structure(
            &public_trace.int[cols::ECDSA_S_INIT..cols::NUM_INT_PUB],
            num_vars,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers (rho/monomial/const-scalar) — duplicated from sha256.rs since
// those are private to that module.
// ---------------------------------------------------------------------------

fn rho_poly<R: ConstSemiring>(positions: &[usize]) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    for &p in positions {
        debug_assert!(p < 32);
        coeffs[p] = R::ONE;
    }
    DensePolynomial::<R, 32>::new(coeffs)
}

/// Compute `2^k` as an `R` value via repeated doubling. Mirrors the
/// helper in sha256.rs (private there).
fn pow_two<R: ConstSemiring>(k: u32) -> R {
    let mut result = R::ONE;
    for _ in 0..k {
        let copy = result.clone();
        result += &copy;
    }
    result
}

fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// Trace generation — call both sub-UAIRs' generators, splice the int sections
// together at the merged column positions.
// ---------------------------------------------------------------------------

fn merge_traces<R>(
    sha_trace: UairTrace<'static, R, R, 32>,
    ecdsa_trace: UairTrace<'static, R, R, 32>,
) -> UairTrace<'static, R, R, 32>
where
    R: Clone + Default,
{
    assert_eq!(sha_trace.binary_poly.len(), sha256::cols::NUM_BIN);
    assert_eq!(sha_trace.int.len(), sha256::cols::NUM_INT);
    assert_eq!(ecdsa_trace.int.len(), ecdsa::cols::NUM_INT);

    let binary_poly: Vec<DenseMultilinearExtension<_>> = sha_trace.binary_poly.into_owned();

    let mut int: Vec<DenseMultilinearExtension<R>> = Vec::with_capacity(cols::NUM_INT);
    let sha_ints = sha_trace.int.into_owned();
    let ecdsa_ints = ecdsa_trace.int.into_owned();

    int.extend(sha_ints[0..6].iter().cloned());
    int.extend(ecdsa_ints[0..ecdsa::cols::NUM_INT_PUB].iter().cloned());
    int.extend(ecdsa_ints[ecdsa::cols::NUM_INT_PUB..].iter().cloned());
    int.extend(sha_ints[6..11].iter().cloned());

    debug_assert_eq!(int.len(), cols::NUM_INT);

    UairTrace {
        binary_poly: binary_poly.into(),
        int: int.into(),
        ..Default::default()
    }
}

/// Merge an already-built SHA trace with ECDSA scalars derived from its
/// declared output and one public compact signature.
pub fn build_trace_from_sha_and_signature<R>(
    num_vars: usize,
    sha_trace: UairTrace<'static, R, R, 32>,
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    signature_r_be: &[u8],
    signature_s_be: &[u8],
) -> Result<UairTrace<'static, R, R, 32>, ShaEcdsaTraceBuildError>
where
    R: EcdsaFpRing + From<u32> + From<Int<EC_FP_INT_LIMBS>> + Default,
{
    let sha_public = sha_trace.public(&<Sha256CompressionSliceUair<R> as Uair>::signature());
    let digest = extract_sha256_output(&sha_public).map_err(ShaEcdsaTraceBuildError::Scalar)?;
    let scalars = derive_ecdsa_verification_scalars(digest, signature_r_be, signature_s_be)
        .map_err(ShaEcdsaTraceBuildError::Scalar)?;
    let ecdsa_trace = ecdsa::build_trace_from_scalars(num_vars, q, scalars.u1, scalars.u2)
        .map_err(ShaEcdsaTraceBuildError::PublicKey)?;
    Ok(merge_traces(sha_trace, ecdsa_trace))
}

/// Build the complete H8 trace from one exact 400-byte message, signature,
/// and affine public key.
pub fn build_trace_from_message_and_signature<R>(
    num_vars: usize,
    message: &[u8],
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    signature_r_be: &[u8],
    signature_s_be: &[u8],
) -> Result<UairTrace<'static, R, R, 32>, ShaEcdsaTraceBuildError>
where
    R: EcdsaFpRing + From<u32> + From<Int<EC_FP_INT_LIMBS>> + Default,
{
    let sha_trace = sha256::build_trace_from_message(num_vars, message)
        .map_err(ShaEcdsaTraceBuildError::ShaMessage)?;
    build_trace_from_sha_and_signature(
        num_vars,
        sha_trace,
        q,
        signature_r_be,
        signature_s_be,
    )
}

/// Build a combined trace whose ECDSA bit columns are derived from the
/// generated SHA half and one public compact signature.
pub fn build_trace_from_signature<R, Rng>(
    num_vars: usize,
    rng: &mut Rng,
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    signature_r_be: &[u8],
    signature_s_be: &[u8],
) -> Result<UairTrace<'static, R, R, 32>, ShaEcdsaTraceBuildError>
where
    R: EcdsaFpRing + From<u32> + From<Int<EC_FP_INT_LIMBS>> + Default,
    Rng: RngCore + ?Sized,
{
    let sha_trace =
        <Sha256CompressionSliceUair<R> as GenerateRandomTrace<32>>::generate_random_trace(
            num_vars, rng,
        );
    build_trace_from_sha_and_signature(num_vars, sha_trace, q, signature_r_be, signature_s_be)
}

/// Build a combined trace with explicit ECDSA public key and Shamir scalars.
///
/// The SHA half remains an independent generated fixture until the later
/// digest-to-scalar binding slice is implemented.
pub fn build_trace_from_ecdsa_scalars<R, Rng>(
    num_vars: usize,
    rng: &mut Rng,
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    u1: CbUint<EC_FP_INT_LIMBS>,
    u2: CbUint<EC_FP_INT_LIMBS>,
) -> Result<UairTrace<'static, R, R, 32>, EcdsaPublicKeyBindingError>
where
    R: EcdsaFpRing + From<u32> + From<Int<EC_FP_INT_LIMBS>> + Default,
    Rng: RngCore + ?Sized,
{
    let sha_trace =
        <Sha256CompressionSliceUair<R> as GenerateRandomTrace<32>>::generate_random_trace(
            num_vars, rng,
        );
    let ecdsa_trace = ecdsa::build_trace_from_scalars(num_vars, q, u1, u2)?;
    Ok(merge_traces(sha_trace, ecdsa_trace))
}

impl<R> GenerateRandomTrace<32> for ShaEcdsaUair<R>
where
    R: EcdsaFpRing + From<u32> + From<Int<EC_FP_INT_LIMBS>> + Default,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let n_rows = 1usize << num_vars;
        assert!(
            n_rows > FINAL_ROW,
            "ShaEcdsa UAIR needs > {FINAL_ROW} rows; got {n_rows}",
        );

        let sha_trace =
            <Sha256CompressionSliceUair<R> as GenerateRandomTrace<32>>::generate_random_trace(
                num_vars, rng,
            );
        let ecdsa_trace =
            <super::EcdsaUair<R> as GenerateRandomTrace<32>>::generate_random_trace(num_vars, rng);

        // Int section: merge per the layout in `cols`.
        //
        // SHA standalone int layout (11 cols):
        //   0..6   public:  S_INIT_PREFIX, S_FEEDFORWARD, S_MSG_INIT,
        //                   PA_K, S_ACTIVE_SCHED, S_ACTIVE_UPD
        //   6..11  witness: PA_C_C7, PA_C_C8, PA_C_C9, PA_C_FF_A, PA_C_FF_E
        //
        // ECDSA standalone int layout (31 cols):
        //   0..18  public selectors, Q/G+Q, selected T, and boundaries
        //   18..31 witness: ordinary-projective state + complete-RCB products
        //
        // Merged layout (NUM_INT = 42, NUM_INT_PUB = 24):
        //   0..6   SHA publics  (sha[0..6])
        //   6..24  ECDSA publics (ecdsa[0..18])
        //   24..37 ECDSA witnesses (ecdsa[18..31])
        //   37..42 SHA witnesses (sha[6..11]) — the linear-constraint
        //                                       compensators
        merge_traces(sha_trace, ecdsa_trace)
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rng, rngs::StdRng};
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    const H1_DIGEST: [u8; 32] = [
        0x85, 0x97, 0x9d, 0xa5, 0x22, 0xf5, 0x62, 0x9d, 0x7e, 0x08, 0xdc, 0x19, 0xa1, 0x09, 0x01,
        0x54, 0xb4, 0x72, 0x0c, 0xea, 0x56, 0x21, 0x7f, 0x4a, 0x62, 0xdf, 0x07, 0x8e, 0x61, 0x01,
        0x42, 0x6c,
    ];
    const H1_R: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("12E2303536CCDD4F4FB63A8CBE473FAA259C3B28B00C3F62FF095E9E90A21217");
    const H1_S: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("36F59BC579CD0E1F456BA1C21047AF6412610617A2EBC7C74A0AFC5F4CE750E4");
    const H1_U1: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("30E48216899822D64BDB7F54DB72055CEF40F5D894433DC1AF0251E69AC1A2F0");
    const H1_U2: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("968A5F9440AA6BB07E2A112CF0FDCD15C681C90BA2102D9DD633CC6DAD0CD8F8");
    const H8_DIGEST: [u8; 32] = [
        0xe2, 0xd0, 0xfa, 0x71, 0x42, 0x2b, 0xd3, 0x3d, 0x79, 0x20, 0x95, 0xfe, 0xcf, 0x1d, 0x3c,
        0xe6, 0xd1, 0x39, 0x06, 0xe2, 0x62, 0x7e, 0xe5, 0x8c, 0x11, 0xe8, 0xc5, 0x6c, 0xa0, 0x03,
        0x91, 0x88,
    ];
    const H8_R: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302");
    const H8_S: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568");
    const H8_Q_X: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    const H8_Q_Y: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A");

    type TestInt = Int<EC_FP_INT_LIMBS>;

    fn uint_to_be_bytes(value: &CbUint<EC_FP_INT_LIMBS>) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, word) in value.as_words().iter().rev().enumerate() {
            let start = index * 8;
            bytes[start..start + 8].copy_from_slice(&word.to_be_bytes());
        }
        bytes
    }

    fn h8_message() -> Vec<u8> {
        let prefix = b"zinc-plus-lab:h8:honest-sha-ecdsa:v1\n";
        let mut message = prefix.to_vec();
        message.extend(
            (0_u16..)
                .map(|counter| counter as u8)
                .take(sha256::SEVEN_BLOCK_MESSAGE_BYTES - prefix.len()),
        );
        message
    }

    fn h8_trace() -> UairTrace<'static, TestInt, TestInt, 32> {
        build_trace_from_message_and_signature(
            9,
            &h8_message(),
            (H8_Q_X, H8_Q_Y),
            &uint_to_be_bytes(&H8_R),
            &uint_to_be_bytes(&H8_S),
        )
        .expect("frozen H8 oracle must build")
    }

    fn set_digest(trace: &mut UairTrace<'static, TestInt, TestInt, 32>, digest: [u8; 32]) {
        for (index, &(column, row)) in SHA_OUTPUT_WORD_CELLS.iter().enumerate() {
            let start = index * 4;
            let word = u32::from_be_bytes(
                digest[start..start + 4]
                    .try_into()
                    .expect("digest word has exactly four bytes"),
            );
            trace.binary_poly.to_mut()[column].evaluations[row] = word.into();
        }
    }

    fn set_scalar_selectors(
        trace: &mut UairTrace<'static, TestInt, TestInt, 32>,
        u1: &CbUint<EC_FP_INT_LIMBS>,
        u2: &CbUint<EC_FP_INT_LIMBS>,
    ) {
        for row in 0..ecdsa::NUM_SHAMIR_ROUNDS {
            let b1 = scalar_bit(u1, row);
            let b2 = scalar_bit(u2, row);
            trace.int.to_mut()[cols::ECDSA_PA_B1].evaluations[row] = TestInt::from(u32::from(b1));
            trace.int.to_mut()[cols::ECDSA_PA_B2].evaluations[row] = TestInt::from(u32::from(b2));
            trace.int.to_mut()[cols::ECDSA_PA_B1B2].evaluations[row] =
                TestInt::from(u32::from(b1 && b2));
            trace.int.to_mut()[cols::ECDSA_S_ADD].evaluations[row] =
                TestInt::from(u32::from(b1 || b2));
        }
    }

    fn binding_fixture(
        digest: [u8; 32],
        signature_r: &CbUint<EC_FP_INT_LIMBS>,
        signature_s: &CbUint<EC_FP_INT_LIMBS>,
    ) -> (
        UairTrace<'static, TestInt, TestInt, 32>,
        ShaEcdsaDerivedScalars,
    ) {
        let mut rng = StdRng::seed_from_u64(0x4837_5eed);
        let mut trace =
            <ShaEcdsaUair<TestInt> as GenerateRandomTrace<32>>::generate_random_trace(9, &mut rng);
        set_digest(&mut trace, digest);
        let scalars = derive_ecdsa_verification_scalars(
            digest,
            &uint_to_be_bytes(signature_r),
            &uint_to_be_bytes(signature_s),
        )
        .expect("fixture signature scalars must be canonical");
        set_scalar_selectors(&mut trace, &scalars.u1, &scalars.u2);
        (trace, scalars)
    }

    /// Sanity: 18 SHA + 20 complete-RCB ECDSA constraints.
    #[test]
    fn sha_ecdsa_constraint_shape() {
        type U = ShaEcdsaUair<Int<EC_FP_INT_LIMBS>>;
        let constraints = count_constraints::<U>();
        let max_degree = count_max_degree::<U>();
        let signature = <U as Uair>::signature();
        let total_binary = signature.total_cols().num_binary_poly_cols();
        let witness_binary = signature.witness_cols().num_binary_poly_cols();
        let public_int = signature.public_cols().num_int_cols();
        let witness_int = signature.witness_cols().num_int_cols();

        assert_eq!(constraints, 38);
        assert_eq!(max_degree, 5);
        assert_eq!(total_binary, 20);
        assert_eq!(witness_binary, 11);
        assert_eq!(public_int, 24);
        assert_eq!(witness_int, 18);
        println!(
            "H6 UAIR shape: constraints={constraints} max_degree={max_degree} \
             total_binary={total_binary} witness_binary={witness_binary} \
             public_int={public_int} witness_int={witness_int}",
        );
        println!(
            "H7 UAIR shape: constraints={constraints} max_degree={max_degree} \
             total_binary={total_binary} witness_binary={witness_binary} \
             public_int={public_int} witness_int={witness_int}",
        );
        println!(
            "H8 UAIR shape: constraints={constraints} max_degree={max_degree} \
             total_binary={total_binary} witness_binary={witness_binary} \
             public_int={public_int} witness_int={witness_int}",
        );
        let degrees = count_constraint_degrees::<U>();
        assert!(degrees.iter().all(|&degree| degree <= 5));
        assert!(
            degrees.iter().filter(|&&d| d == 2).count() >= 3,
            "expected ≥3 deg-2"
        );
    }

    /// The merged trace builder produces a trace with the right column
    /// shape (we don't re-run the full mod-p witness check here — the
    /// sub-UAIRs already test their halves individually).
    #[test]
    fn merged_trace_shape() {
        let num_vars = 9;
        let mut r = rng();
        let trace =
            <ShaEcdsaUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::generate_random_trace(
                num_vars, &mut r,
            );

        assert_eq!(trace.binary_poly.len(), cols::NUM_BIN);
        assert_eq!(trace.int.len(), cols::NUM_INT);
        for col in trace.binary_poly.iter() {
            assert_eq!(col.len(), 1 << num_vars);
        }
        for col in trace.int.iter() {
            assert_eq!(col.len(), 1 << num_vars);
        }
    }

    /// Re-export sanity: NUM_SHAMIR_ROUNDS, FINAL_ROW are accessible
    /// through this module (matching `crate::ecdsa`).
    #[test]
    fn re_exports() {
        let _ = ecdsa::NUM_SHAMIR_ROUNDS;
        let _ = FINAL_ROW;
        let _ = ecdsa::FINAL_ROW;
    }

    #[test]
    fn scalar_binding_matches_frozen_h1_oracle() {
        let (trace, expected) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        let r = uint_to_be_bytes(&H1_R);
        let s = uint_to_be_bytes(&H1_S);
        let actual = verify_sha_ecdsa_scalar_binding(&r, &s, &trace)
            .expect("H1 oracle values should satisfy scalar binding");

        assert_eq!(actual, expected);
        assert_eq!(actual.digest, H1_DIGEST);
        assert_eq!(
            actual.message_representative,
            CbUint::from_be_slice(&H1_DIGEST)
        );
        assert_eq!(actual.u1, H1_U1);
        assert_eq!(actual.u2, H1_U2);
    }

    #[test]
    fn h8_complete_binding_matches_independent_oracle() {
        let message = h8_message();
        let trace = h8_trace();
        let binding = verify_sha_ecdsa_h8_application_binding(
            &message,
            &uint_to_be_bytes(&H8_R),
            &uint_to_be_bytes(&H8_S),
            &uint_to_be_bytes(&H8_Q_X),
            &uint_to_be_bytes(&H8_Q_Y),
            &trace,
        )
        .expect("frozen H8 statement must bind");
        assert_eq!(binding.scalars.digest, H8_DIGEST);
        assert_eq!(binding.public_key.q, (H8_Q_X, H8_Q_Y));
        assert_eq!(binding.result_branch, EcdsaResultBranch::Direct);
    }

    #[test]
    fn public_structure_rejects_wrong_column_counts_with_typed_errors() {
        type U = ShaEcdsaUair<TestInt>;

        let trace = h8_trace();
        let public = trace.public(&U::signature());

        let mut missing_binary = public.clone();
        missing_binary
            .binary_poly
            .to_mut()
            .pop()
            .expect("H8 has public binary columns");
        assert!(matches!(
            U::verify_public_structure(&missing_binary, 9),
            Err(PublicStructureError::WrongColumnCount {
                column_family: "binary_poly",
                expected: cols::NUM_BIN_PUB,
                actual: 8,
            })
        ));

        let mut missing_int = public.clone();
        missing_int
            .int
            .to_mut()
            .pop()
            .expect("H8 has public integer columns");
        assert!(matches!(
            U::verify_public_structure(&missing_int, 9),
            Err(PublicStructureError::WrongColumnCount {
                column_family: "int",
                expected: cols::NUM_INT_PUB,
                actual: 23,
            })
        ));
    }

    #[test]
    fn h8_honest_sha_binding_owns_all_iv_message_and_junction_cells() {
        let message = h8_message();
        let mut trace = h8_trace();
        assert_eq!(
            verify_sha_ecdsa_honest_sha_binding(&message, &trace),
            Ok(H8_DIGEST),
        );

        for (column, name) in [(cols::PA_A, "PA_A"), (cols::PA_E, "PA_E")] {
            for row in 0..4 {
                let original = trace.binary_poly[column].evaluations[row].clone();
                let word = binary_word(&original);
                trace.binary_poly.to_mut()[column].evaluations[row] = (word ^ 1).into();
                assert_eq!(
                    verify_sha_ecdsa_honest_sha_binding(&message, &trace),
                    Err(ShaEcdsaHonestShaBindingError::IvMismatch { column: name, row }),
                );
                trace.binary_poly.to_mut()[column].evaluations[row] = original;
            }
        }

        for compression in 0..sha256::cols::NUM_COMPRESSIONS {
            let start = compression * sha256::cols::ROWS_PER_COMP;
            for word_index in 0..16 {
                let row = start + word_index;
                let original = trace.binary_poly[cols::PA_M].evaluations[row].clone();
                let word = binary_word(&original);
                trace.binary_poly.to_mut()[cols::PA_M].evaluations[row] = (word ^ 1).into();
                assert_eq!(
                    verify_sha_ecdsa_honest_sha_binding(&message, &trace),
                    Err(ShaEcdsaHonestShaBindingError::MessageWordMismatch { row }),
                );
                trace.binary_poly.to_mut()[cols::PA_M].evaluations[row] = original;
            }
        }

        for compression in 0..sha256::cols::NUM_COMPRESSIONS {
            let start = compression * sha256::cols::ROWS_PER_COMP;
            for (column, name) in [(cols::PA_A, "PA_A"), (cols::PA_E, "PA_E")] {
                for offset in 0..4 {
                    let input_row = start + offset;
                    let row = start + sha256::cols::ROUNDS_PER_COMP + offset;
                    let original = trace.binary_poly[column].evaluations[row].clone();
                    let word = binary_word(&original);
                    trace.binary_poly.to_mut()[column].evaluations[row] = (word ^ 1).into();
                    assert_eq!(
                        verify_sha_ecdsa_honest_sha_binding(&message, &trace),
                        Err(ShaEcdsaHonestShaBindingError::JunctionMismatch {
                            column: name,
                            row,
                            input_row,
                        }),
                    );
                    trace.binary_poly.to_mut()[column].evaluations[row] = original;
                }
            }
        }

        assert_eq!(
            verify_sha_ecdsa_honest_sha_binding(&message[..399], &trace),
            Err(ShaEcdsaHonestShaBindingError::MessageLength { actual_bytes: 399 }),
        );
    }

    #[test]
    fn h8_padding_regions_are_independently_bound() {
        let message = h8_message();
        let trace = h8_trace();
        let cases = [
            (0, "message byte"),
            (6 * sha256::cols::ROWS_PER_COMP + 4, "0x80 byte"),
            (6 * sha256::cols::ROWS_PER_COMP + 5, "zero padding"),
            (6 * sha256::cols::ROWS_PER_COMP + 15, "length field"),
        ];
        for (row, label) in cases {
            let mut mutated = trace.clone();
            let word = binary_word(&mutated.binary_poly[cols::PA_M].evaluations[row]);
            mutated.binary_poly.to_mut()[cols::PA_M].evaluations[row] = (word ^ 1).into();
            assert_eq!(
                verify_sha_ecdsa_honest_sha_binding(&message, &mutated),
                Err(ShaEcdsaHonestShaBindingError::MessageWordMismatch { row }),
                "{label}",
            );
        }
    }

    #[test]
    fn scalar_binding_rejects_disconnected_sha_public_structure() {
        let (trace, _) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        verify_sha_ecdsa_chain_structure(&trace).expect("generated SHA layout must be canonical");

        for (column, row, name) in [
            (cols::SHA_S_INIT_PREFIX, 0, "SHA_S_INIT_PREFIX"),
            (cols::SHA_S_FEEDFORWARD, 64, "SHA_S_FEEDFORWARD"),
            (cols::SHA_S_MSG_INIT, 0, "SHA_S_MSG_INIT"),
            (cols::SHA_S_ACTIVE_SCHED, 0, "SHA_S_ACTIVE_SCHED"),
            (cols::SHA_S_ACTIVE_UPD, 0, "SHA_S_ACTIVE_UPD"),
        ] {
            let mut mutated = trace.clone();
            mutated.int.to_mut()[column].evaluations[row] = TestInt::from(0_u32);
            assert_eq!(
                verify_sha_ecdsa_chain_structure(&mutated),
                Err(ShaEcdsaScalarBindingError::ShaStructureMismatch { column: name, row }),
            );
        }

        let mut wrong_k = trace.clone();
        wrong_k.int.to_mut()[cols::SHA_PA_K].evaluations[3] = TestInt::from(0_u32);
        assert_eq!(
            verify_sha_ecdsa_chain_structure(&wrong_k),
            Err(ShaEcdsaScalarBindingError::ShaStructureMismatch {
                column: "SHA_PA_K",
                row: 3,
            }),
        );

        let mut nonzero_message_slack = trace.clone();
        nonzero_message_slack.binary_poly.to_mut()[cols::PA_M].evaluations[16] = 1_u32.into();
        assert_eq!(
            verify_sha_ecdsa_chain_structure(&nonzero_message_slack),
            Err(ShaEcdsaScalarBindingError::ShaStructureMismatch {
                column: "PA_M",
                row: 16,
            }),
        );

        let mut missing_selector = trace.clone();
        missing_selector.int.to_mut()[cols::SHA_S_INIT_PREFIX]
            .evaluations
            .truncate(511);
        assert_eq!(
            verify_sha_ecdsa_chain_structure(&missing_selector),
            Err(ShaEcdsaScalarBindingError::MissingShaStructureCell {
                column: "SHA_S_INIT_PREFIX",
                row: 511,
            }),
        );
    }

    #[test]
    fn scalar_binding_accepts_high_digest_and_zero_u1() {
        let digest = uint_to_be_bytes(&ecdsa::SECP256K1_N_UINT);
        let one = CbUint::ONE;
        let (trace, expected) = binding_fixture(digest, &one, &one);
        assert_eq!(expected.message_representative, ecdsa::SECP256K1_N_UINT);
        assert_eq!(expected.u1, CbUint::ZERO);
        assert_eq!(expected.u2, CbUint::ONE);

        verify_sha_ecdsa_scalar_binding(&uint_to_be_bytes(&one), &uint_to_be_bytes(&one), &trace)
            .expect("SEC 1 permits e >= n and u1 = 0");
    }

    #[test]
    fn scalar_binding_rejects_signature_scalar_boundaries() {
        let (trace, _) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        let r = uint_to_be_bytes(&H1_R);
        let s = uint_to_be_bytes(&H1_S);
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(&r[..31], &s, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureREncoding { actual_bytes: 31 })
        ));
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(&r, &s[..31], &trace),
            Err(ShaEcdsaScalarBindingError::SignatureSEncoding { actual_bytes: 31 })
        ));
        let mut r_33 = [0_u8; 33];
        r_33[1..].copy_from_slice(&r);
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(&r_33, &s, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureREncoding { actual_bytes: 33 })
        ));
        let mut s_33 = [0_u8; 33];
        s_33[1..].copy_from_slice(&s);
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(&r, &s_33, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureSEncoding { actual_bytes: 33 })
        ));
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&[0_u8; 32], &s, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureRZero),
        );
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&r, &[0_u8; 32], &trace),
            Err(ShaEcdsaScalarBindingError::SignatureSZero),
        );
        let order = uint_to_be_bytes(&ecdsa::SECP256K1_N_UINT);
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&order, &s, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureROutOfRange),
        );
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&r, &order, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureSOutOfRange),
        );
        let order_plus_one = uint_to_be_bytes(
            &ecdsa::SECP256K1_N_UINT.wrapping_add(&CbUint::ONE),
        );
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&order_plus_one, &s, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureROutOfRange),
        );
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&r, &order_plus_one, &trace),
            Err(ShaEcdsaScalarBindingError::SignatureSOutOfRange),
        );
    }

    #[test]
    fn scalar_binding_does_not_impose_low_s_policy() {
        let high_s = ecdsa::SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE);
        let (trace, _) = binding_fixture(H1_DIGEST, &H1_R, &high_s);
        verify_sha_ecdsa_scalar_binding(
            &uint_to_be_bytes(&H1_R),
            &uint_to_be_bytes(&high_s),
            &trace,
        )
        .expect("plain SEC 1 verification accepts high-s signatures");
    }

    #[test]
    fn scalar_binding_rejects_every_single_bit_mutation() {
        let (mut trace, _) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        let r = uint_to_be_bytes(&H1_R);
        let s = uint_to_be_bytes(&H1_S);
        for (column, scalar) in [
            (cols::ECDSA_PA_B1, EcdsaVerificationScalar::U1),
            (cols::ECDSA_PA_B2, EcdsaVerificationScalar::U2),
        ] {
            for row in 0..ecdsa::NUM_SHAMIR_ROUNDS {
                let original = trace.int[column].evaluations[row];
                trace.int.to_mut()[column].evaluations[row] = if original == TestInt::from(0_u32) {
                    TestInt::from(1_u32)
                } else {
                    TestInt::from(0_u32)
                };
                assert!(matches!(
                    verify_sha_ecdsa_scalar_binding(&r, &s, &trace),
                    Err(ShaEcdsaScalarBindingError::ScalarBitMismatch {
                        scalar: actual_scalar,
                        column: actual_column,
                        row: actual_row,
                        ..
                    }) if actual_scalar == scalar && actual_column == column && actual_row == row
                ));
                trace.int.to_mut()[column].evaluations[row] = original;
            }
        }
    }

    #[test]
    fn scalar_binding_rejects_mod_n_alias_and_companion_mutations() {
        let zero_digest = [0_u8; 32];
        let one = CbUint::ONE;
        let (mut trace, _) = binding_fixture(zero_digest, &one, &one);
        set_scalar_selectors(&mut trace, &ecdsa::SECP256K1_N_UINT, &one);
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(
                &uint_to_be_bytes(&one),
                &uint_to_be_bytes(&one),
                &trace,
            ),
            Err(ShaEcdsaScalarBindingError::ScalarBitMismatch {
                scalar: EcdsaVerificationScalar::U1,
                ..
            })
        ));

        let (mut trace, _) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        trace.int.to_mut()[cols::ECDSA_PA_B1B2].evaluations[0] =
            if trace.int[cols::ECDSA_PA_B1B2].evaluations[0] == TestInt::from(0_u32) {
                TestInt::from(1_u32)
            } else {
                TestInt::from(0_u32)
            };
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(
                &uint_to_be_bytes(&H1_R),
                &uint_to_be_bytes(&H1_S),
                &trace,
            ),
            Err(ShaEcdsaScalarBindingError::CompanionSelectorMismatch {
                column: cols::ECDSA_PA_B1B2,
                row: 0,
                ..
            })
        ));

        let (mut trace, _) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        trace.int.to_mut()[cols::ECDSA_S_ADD].evaluations[0] =
            if trace.int[cols::ECDSA_S_ADD].evaluations[0] == TestInt::from(0_u32) {
                TestInt::from(1_u32)
            } else {
                TestInt::from(0_u32)
            };
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(
                &uint_to_be_bytes(&H1_R),
                &uint_to_be_bytes(&H1_S),
                &trace,
            ),
            Err(ShaEcdsaScalarBindingError::CompanionSelectorMismatch {
                column: cols::ECDSA_S_ADD,
                row: 0,
                ..
            })
        ));
    }

    #[test]
    fn scalar_binding_rejects_digest_wrong_s_non_boolean_and_missing_cells() {
        let (trace, _) = binding_fixture(H1_DIGEST, &H1_R, &H1_S);
        let r = uint_to_be_bytes(&H1_R);
        let s = uint_to_be_bytes(&H1_S);

        let mut wrong_digest = trace.clone();
        let (column, row) = SHA_OUTPUT_WORD_CELLS[7];
        let original = binary_word(&wrong_digest.binary_poly[column].evaluations[row]);
        wrong_digest.binary_poly.to_mut()[column].evaluations[row] = (original ^ 1).into();
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(&r, &s, &wrong_digest),
            Err(ShaEcdsaScalarBindingError::ScalarBitMismatch { .. })
        ));

        let wrong_s = H1_S.wrapping_add(&CbUint::ONE);
        assert!(matches!(
            verify_sha_ecdsa_scalar_binding(&r, &uint_to_be_bytes(&wrong_s), &trace),
            Err(ShaEcdsaScalarBindingError::ScalarBitMismatch { .. })
        ));

        let mut non_boolean = trace.clone();
        non_boolean.int.to_mut()[cols::ECDSA_PA_B1].evaluations[9] = TestInt::from(2_u32);
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&r, &s, &non_boolean),
            Err(ShaEcdsaScalarBindingError::NonBooleanScalarBit {
                column: cols::ECDSA_PA_B1,
                row: 9,
            }),
        );

        let mut missing_digest = trace.clone();
        missing_digest.binary_poly.to_mut()[cols::PA_A]
            .evaluations
            .truncate(SHA_OUTPUT_START_ROW + 3);
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&r, &s, &missing_digest),
            Err(ShaEcdsaScalarBindingError::MissingDigestWord {
                column: cols::PA_A,
                row: SHA_OUTPUT_START_ROW + 3,
            }),
        );

        let mut missing_bit = trace.clone();
        missing_bit.int.to_mut()[cols::ECDSA_PA_B2]
            .evaluations
            .truncate(255);
        assert_eq!(
            verify_sha_ecdsa_scalar_binding(&r, &s, &missing_bit),
            Err(ShaEcdsaScalarBindingError::MissingScalarBit {
                column: cols::ECDSA_PA_B2,
                row: 255,
            }),
        );
    }

    #[test]
    fn signature_builder_composes_h7_with_h6() {
        let mut rng = StdRng::seed_from_u64(0x4837_51a7);
        let sha_trace =
            <Sha256CompressionSliceUair<TestInt> as GenerateRandomTrace<32>>::generate_random_trace(
                9, &mut rng,
            );
        let digest = extract_sha256_output(
            &sha_trace.public(&<Sha256CompressionSliceUair<TestInt> as Uair>::signature()),
        )
        .expect("generated SHA trace has an output prefix");
        let e = CbUint::from_be_slice(&digest);
        let r = crate::ecdsa_doubling::SECP256K1_G_X_UINT;
        let e_reduced = if e >= ecdsa::SECP256K1_N_UINT {
            e.wrapping_sub(&ecdsa::SECP256K1_N_UINT)
        } else {
            e
        };
        let s = add_mod_n(&e_reduced, &r);
        assert_ne!(s, CbUint::ZERO, "the deterministic fixture needs nonzero s");
        let q = (
            crate::ecdsa_doubling::SECP256K1_G_X_UINT,
            crate::ecdsa_doubling::SECP256K1_G_Y_UINT,
        );
        let r_bytes = uint_to_be_bytes(&r);
        let s_bytes = uint_to_be_bytes(&s);
        let trace = build_trace_from_sha_and_signature(9, sha_trace, q, &r_bytes, &s_bytes)
            .expect("deterministic public signature should build a trace");

        let binding = verify_sha_ecdsa_application_binding(&r_bytes, &s_bytes, &trace)
            .expect("d = k = 1 fixture should satisfy H7 and H6");
        assert_eq!(binding.scalars.digest, digest);
        assert_eq!(binding.result_branch, EcdsaResultBranch::Direct);
    }
}
