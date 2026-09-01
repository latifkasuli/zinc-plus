//! ECDSA Shamir scalar-multiplication UAIR (F_p / EC ops — composed
//! UAIR).
//!
//! Per row, computes one Shamir step `R_{t+1} = 2·R_t + T_t` with the
//! complete Renes-Costello-Batina formulas in ordinary projective coordinates.
//! The selected addend `T_t` is one of `{O, G, Q, G+Q}` according to the
//! public `(b_1[t], b_2[t])` bit pair.
//!
//! - **Row chaining**: a single chained `(W_X, W_Y, W_Z)` triple where `up.X[t]
//!   = R_t` (the row's input) and `down.X[t] = R_{t+1}` (the next row's input).
//!   No separate input/output columns.
//! - **Complete boundaries**: row 0 is the projective identity `(0:1:0)`; the
//!   same formulas handle identity, equal, and inverse additions.
//! - **Bound addend**: two constraints bind public `(T_X,T_Y,S_ADD)` to the bit
//!   pair and public `Q`/`G+Q`; verifier-side structure checks pin selector
//!   shapes and canonical booleans.
//! - **Final boundary**: the final ordinary-projective state exposes `R_x =
//!   X/Z`; [`verify_ecdsa_result_binding`] enforces `R_x mod n == r` over exact
//!   verifier-side integers after the proof verifier binds that public cell.
//!
//! ## Constraint shape
//!
//! 20 constraints, maximum degree 5. The 13 witness columns are the chained
//! point, six doubling products, and three diagonal addition products. The
//! complete doubled point and the three one-use addition cross-products are
//! inlined instead of being committed as separate columns.
//! The 18 public columns include selector metadata, `Q`, `G+Q`, the selected
//! addend, and boundary values.
//!
//! ## Application binding
//!
//! The UAIR keeps `Q`, `G+Q`, and the scalar bits public. Application-level
//! helpers validate and bind those values exactly before proof verification;
//! they are not duplicated as non-native in-UAIR constraints.

use core::{fmt, marker::PhantomData};

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::{ConstSemiring, crypto_bigint_int::Int};
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, PublicStructureError, ShiftSpec, TotalColumnLayout,
    TraceRow, Uair, UairSignature, UairTrace, ideal::ImpossibleIdeal,
};

use crate::{
    GenerateRandomTrace,
    ecdsa_doubling::{
        EC_FP_INT_LIMBS, EcdsaFpRing, SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT,
        SECP256K1_P_HALF_UINT, SECP256K1_P_UINT,
    },
};

/// Number of Shamir doubling+add rounds. With `num_vars >= 9`,
/// trace rows = 512, so 256 active rounds + 1 final row + 255 padding
/// fits.
pub const NUM_SHAMIR_ROUNDS: usize = 256;

/// The trace row at which the affine-conversion / final-output
/// constraints apply (one past the last active doubling round).
pub const FINAL_ROW: usize = NUM_SHAMIR_ROUNDS;

const SECP256K1_N_HEX: &str = concat!(
    "FFFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFFE",
    "BAAEDCE6AF48A03B",
    "BFD25E8CD0364141",
);

/// secp256k1 subgroup order `n` (SEC 2 section 2.4.1).
pub const SECP256K1_N_UINT: CbUint<EC_FP_INT_LIMBS> = CbUint::from_be_hex(SECP256K1_N_HEX);

/// Which of the two possible base-field representatives matched `r`.
/// Since `p < 2n`, no other quotient is possible for a canonical `R_x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdsaResultBranch {
    /// `R_x = r`.
    Direct,
    /// `R_x = r + n`.
    PlusOrder,
}

/// Verifier-side ECDSA result-binding failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdsaResultBindingError {
    /// Compact `r` must be exactly one 32-byte big-endian integer.
    SignatureScalarEncoding { actual_bytes: usize },
    /// SEC 1 rejects `r = 0`.
    SignatureScalarZero,
    /// SEC 1 rejects `r >= n`.
    SignatureScalarOutOfRange,
    /// The public `PA_R_X` cell is not the unique centered encoding of an
    /// element in `[0, p)`.
    NonCanonicalFinalX,
    /// The expected result cell is absent from the verifier-owned public
    /// trace.
    MissingPublicResult { column: usize, row: usize },
    /// The canonical x-coordinate does not reduce to the signature scalar.
    ResultMismatch,
}

/// One affine coordinate in the public-key statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdsaCoordinate {
    X,
    Y,
}

/// Verifier-side public-key validation or trace-binding failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdsaPublicKeyBindingError {
    /// Each affine coordinate must use exactly 32 big-endian bytes.
    CoordinateEncoding {
        coordinate: EcdsaCoordinate,
        actual_bytes: usize,
    },
    /// SEC 1 requires each coordinate to be a canonical integer below p.
    CoordinateOutOfRange { coordinate: EcdsaCoordinate },
    /// The canonical affine pair does not satisfy y^2 = x^3 + 7 mod p.
    PointNotOnCurve,
    /// Q = -G, so the affine-only statement cannot encode G+Q.
    GeneratorSumAtInfinity,
    /// A public Q or G+Q trace cell is absent.
    MissingPublicPoint { column: usize, row: usize },
    /// A public Q or G+Q trace cell is not canonically centered modulo p.
    NonCanonicalPublicPoint { column: usize, row: usize },
    /// A public Q or G+Q trace cell differs from the verifier-derived value.
    PublicPointMismatch { column: usize, row: usize },
}

impl fmt::Display for EcdsaPublicKeyBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoordinateEncoding {
                coordinate,
                actual_bytes,
            } => write!(
                formatter,
                "public-key {coordinate:?} coordinate must be exactly 32 bytes, got {actual_bytes}",
            ),
            Self::CoordinateOutOfRange { coordinate } => write!(
                formatter,
                "public-key {coordinate:?} coordinate is not below secp256k1 p",
            ),
            Self::PointNotOnCurve => {
                formatter.write_str("public key is not an affine secp256k1 point")
            }
            Self::GeneratorSumAtInfinity => {
                formatter.write_str("public key equals -G, so G+Q is the point at infinity")
            }
            Self::MissingPublicPoint { column, row } => {
                write!(formatter, "missing public point cell at column {column}, row {row}")
            }
            Self::NonCanonicalPublicPoint { column, row } => write!(
                formatter,
                "public point cell at column {column}, row {row} is not canonically centered modulo p",
            ),
            Self::PublicPointMismatch { column, row } => write!(
                formatter,
                "public point mismatch at column {column}, row {row}",
            ),
        }
    }
}

impl std::error::Error for EcdsaPublicKeyBindingError {}

/// Canonical Q and verifier-derived G+Q values accepted by the key binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcdsaBoundPublicKey {
    pub q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    pub g_plus_q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
}

impl fmt::Display for EcdsaResultBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureScalarEncoding { actual_bytes } => write!(
                formatter,
                "signature scalar r must be exactly 32 bytes, got {actual_bytes}",
            ),
            Self::SignatureScalarZero => formatter.write_str("signature scalar r is zero"),
            Self::SignatureScalarOutOfRange => {
                formatter.write_str("signature scalar r is not below the secp256k1 order")
            }
            Self::NonCanonicalFinalX => {
                formatter.write_str("PA_R_X is not canonically centered modulo secp256k1 p")
            }
            Self::MissingPublicResult { column, row } => {
                write!(formatter, "missing public ECDSA result at column {column}, row {row}")
            }
            Self::ResultMismatch => formatter.write_str("R_x mod n does not equal r"),
        }
    }
}

impl std::error::Error for EcdsaResultBindingError {}

/// Decode the unique centered `Int<4>` representation of a secp256k1
/// base-field element into its canonical integer in `[0, p)`.
pub fn decode_canonical_final_x(
    value: &Int<EC_FP_INT_LIMBS>,
) -> Result<CbUint<EC_FP_INT_LIMBS>, EcdsaResultBindingError> {
    decode_canonical_field_element(value).ok_or(EcdsaResultBindingError::NonCanonicalFinalX)
}

fn decode_canonical_field_element(
    value: &Int<EC_FP_INT_LIMBS>,
) -> Option<CbUint<EC_FP_INT_LIMBS>> {
    let raw = *value.inner().as_uint();
    let is_negative = raw.as_words()[EC_FP_INT_LIMBS - 1] >> 63 != 0;

    if is_negative {
        let decoded = raw.wrapping_add(&SECP256K1_P_UINT);
        if decoded > SECP256K1_P_HALF_UINT && decoded < SECP256K1_P_UINT {
            Some(decoded)
        } else {
            None
        }
    } else if raw <= SECP256K1_P_HALF_UINT {
        Some(raw)
    } else {
        None
    }
}

/// Check the SEC 1 ECDSA result equation using exact 256-bit integer
/// arithmetic over verifier-owned public values.
pub fn verify_ecdsa_result_binding(
    signature_r_be: &[u8],
    final_x: &Int<EC_FP_INT_LIMBS>,
) -> Result<EcdsaResultBranch, EcdsaResultBindingError> {
    if signature_r_be.len() != 32 {
        return Err(EcdsaResultBindingError::SignatureScalarEncoding {
            actual_bytes: signature_r_be.len(),
        });
    }

    let signature_r = CbUint::<EC_FP_INT_LIMBS>::from_be_slice(signature_r_be);
    if signature_r == CbUint::ZERO {
        return Err(EcdsaResultBindingError::SignatureScalarZero);
    }
    if signature_r >= SECP256K1_N_UINT {
        return Err(EcdsaResultBindingError::SignatureScalarOutOfRange);
    }

    let final_x = decode_canonical_final_x(final_x)?;
    let (reduced, branch) = if final_x >= SECP256K1_N_UINT {
        (
            final_x.wrapping_sub(&SECP256K1_N_UINT),
            EcdsaResultBranch::PlusOrder,
        )
    } else {
        (final_x, EcdsaResultBranch::Direct)
    };

    if reduced == signature_r {
        Ok(branch)
    } else {
        Err(EcdsaResultBindingError::ResultMismatch)
    }
}

/// Read and verify a result cell from a verifier-owned public integer trace.
pub fn verify_ecdsa_result_binding_in_column(
    signature_r_be: &[u8],
    public_int: &[DenseMultilinearExtension<Int<EC_FP_INT_LIMBS>>],
    final_x_column: usize,
) -> Result<EcdsaResultBranch, EcdsaResultBindingError> {
    let final_x = public_int
        .get(final_x_column)
        .and_then(|column| column.evaluations.get(FINAL_ROW))
        .ok_or(EcdsaResultBindingError::MissingPublicResult {
            column: final_x_column,
            row: FINAL_ROW,
        })?;
    verify_ecdsa_result_binding(signature_r_be, final_x)
}

// ---------------------------------------------------------------------------
// Column layout.
// ---------------------------------------------------------------------------

pub mod cols {
    // === Public columns (verifier-supplied) ===

    /// `1` at row 0; `0` elsewhere.
    pub const S_INIT: usize = 0;
    /// `1` for `t ∈ 0..NUM_SHAMIR_ROUNDS`; `0` elsewhere.
    pub const S_ACTIVE: usize = 1;
    /// `1` at `FINAL_ROW`; `0` elsewhere.
    pub const S_FINAL: usize = 2;
    /// `1` if the row's bit pair is non-zero (addition takes effect);
    /// `0` otherwise. Equal to `b_1 + b_2 - b_1 · b_2`.
    /// Verifier-derivable from `(b_1, b_2)`.
    pub const S_ADD: usize = 3;
    /// First scalar bit of the row's $(b_1, b_2)$ Shamir pair, in
    /// $\{0,1\}$. Combined with `PA_B2`, selects the affine addend
    /// $T \in \{\OOO, G, Q, G+Q\}$ in-circuit via the formula
    ///   T = b_1·(1-b_2)·G + (1-b_1)·b_2·Q + b_1·b_2·(G+Q).
    pub const PA_B1: usize = 4;
    /// Second scalar bit; see `PA_B1`.
    pub const PA_B2: usize = 5;
    /// Verifier-checked product `PA_B1 * PA_B2`. Materializing this
    /// public selector keeps the selected-addend constraints cubic.
    pub const PA_B1B2: usize = 6;
    /// Affine X-coordinate of $Q$ (the public key). Constant across
    /// all rows of a single proof; consumed by the in-circuit addend
    /// formula.
    pub const PA_QX: usize = 7;
    /// Affine Y-coordinate of $Q$.
    pub const PA_QY: usize = 8;
    /// Affine X-coordinate of $G + Q$. Constant across all rows;
    /// consumed by the addend formula.
    pub const PA_QGX: usize = 9;
    /// Affine Y-coordinate of $G + Q$.
    pub const PA_QGY: usize = 10;
    /// Selected addend coordinates. Together with `S_ADD` these form
    /// the ordinary-projective point `(T_X:T_Y:S_ADD)`, including the
    /// identity `(0:1:0)` for bit pair `(0,0)`.
    pub const PA_T_X: usize = 11;
    pub const PA_T_Y: usize = 12;
    /// Initial ordinary-projective point coordinates (boundary input at row 0).
    pub const PA_R_INIT_X: usize = 13;
    pub const PA_R_INIT_Y: usize = 14;
    pub const PA_R_INIT_Z: usize = 15;
    /// Inverse of $P_Z[\mathrm{FINAL\_ROW}]$ in $\F_p$. Only the
    /// row-$\mathrm{FINAL\_ROW}$ cell is consumed (gated by
    /// $\col{S\_FINAL}$); all other rows can be zero.
    pub const PA_Z_INV: usize = 16;
    /// Affine $x$-coordinate of the loop's final point
    /// $R = (X[\mathrm{FINAL\_ROW}], Y[\mathrm{FINAL\_ROW}],
    /// Z[\mathrm{FINAL\_ROW}])$. Only the row-$\mathrm{FINAL\_ROW}$ cell is
    /// consumed; the verifier is expected to check $R_x \equiv r \pmod n$
    /// off-protocol against the signature scalar $r$.
    pub const PA_R_X: usize = 17;
    pub const NUM_INT_PUB: usize = 18;

    // === Witness columns ===

    // Chained ordinary-projective state. up.X[t] = R_t (input), down.X[t] =
    // R_{t+1} (output, written by the output-selection constraint at
    // row t).
    pub const W_X: usize = 18;
    pub const W_Y: usize = 19;
    pub const W_Z: usize = 20;

    // Six reusable products for complete RCB doubling.
    pub const W_D_T0: usize = 21;
    pub const W_D_T1: usize = 22;
    pub const W_D_T2: usize = 23;
    pub const W_D_T3: usize = 24;
    pub const W_D_T4: usize = 25;
    pub const W_D_T5: usize = 26;
    // Three reusable diagonal products and one cross-product for complete RCB
    // addition. `T3` and `T5` are inlined; retaining `T4` avoids multiplying
    // two degree-three expressions in the final x-coordinate.
    pub const W_A_T0: usize = 27;
    pub const W_A_T1: usize = 28;
    pub const W_A_T2: usize = 29;
    pub const W_A_T4: usize = 30;

    pub const NUM_INT: usize = 31;

    // Flat indices for shift specs (no bin/poly columns; flat = int).
    pub const FLAT_W_X: usize = W_X;
    pub const FLAT_W_Y: usize = W_Y;
    pub const FLAT_W_Z: usize = W_Z;
}

// ---------------------------------------------------------------------------
// The UAIR.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EcdsaUair<R>(PhantomData<R>);

impl<R> Uair for EcdsaUair<R>
where
    R: EcdsaFpRing,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, cols::NUM_INT_PUB);
        // Shift X, Y, Z by 1 so down.X[t] = X[t+1] = R_{t+1}.
        let shifts: Vec<ShiftSpec> = vec![
            ShiftSpec::new(cols::FLAT_W_X, 1),
            ShiftSpec::new(cols::FLAT_W_Y, 1),
            ShiftSpec::new(cols::FLAT_W_Z, 1),
        ];
        UairSignature::new(total, public, shifts, vec![], vec![])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        constrain_rcb_shamir(
            b,
            &up.int[..cols::NUM_INT_PUB],
            &up.int[cols::NUM_INT_PUB..],
            down.int,
            from_ref,
            mbs,
        );
    }

    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero + num_traits::One + PartialEq,
    {
        verify_ecdsa_public_int_structure(&public_trace.int, num_vars)
    }
}

/// Constrain one complete Shamir row using the RCB formulas in ordinary
/// projective coordinates. `public` uses the standalone ECDSA public layout;
/// `witness` starts at `W_X`; `down` contains the shifted state `(X,Y,Z)`.
pub(crate) fn constrain_rcb_shamir<R, B, FromR, MulByScalar>(
    b: &mut B,
    public: &[B::Expr],
    witness: &[B::Expr],
    down: &[B::Expr],
    from_ref: FromR,
    mbs: MulByScalar,
) where
    R: EcdsaFpRing,
    B: ConstraintBuilder,
    FromR: Fn(&DensePolynomial<R, 32>) -> B::Expr,
    MulByScalar: Fn(&B::Expr, &DensePolynomial<R, 32>) -> Option<B::Expr>,
{
    debug_assert_eq!(public.len(), cols::NUM_INT_PUB);
    debug_assert_eq!(witness.len(), cols::NUM_INT - cols::NUM_INT_PUB);
    debug_assert_eq!(down.len(), 3);

    let s_init = &public[cols::S_INIT];
    let s_active = &public[cols::S_ACTIVE];
    let s_final = &public[cols::S_FINAL];
    let s_add = &public[cols::S_ADD];
    let pa_b1 = &public[cols::PA_B1];
    let pa_b2 = &public[cols::PA_B2];
    let pa_b1b2 = &public[cols::PA_B1B2];
    let pa_qx = &public[cols::PA_QX];
    let pa_qy = &public[cols::PA_QY];
    let pa_qgx = &public[cols::PA_QGX];
    let pa_qgy = &public[cols::PA_QGY];
    let pa_t_x = &public[cols::PA_T_X];
    let pa_t_y = &public[cols::PA_T_Y];
    let pa_r_init_x = &public[cols::PA_R_INIT_X];
    let pa_r_init_y = &public[cols::PA_R_INIT_Y];
    let pa_r_init_z = &public[cols::PA_R_INIT_Z];
    let pa_z_inv = &public[cols::PA_Z_INV];
    let pa_r_x = &public[cols::PA_R_X];

    let at = |global: usize| -> &B::Expr { &witness[global - cols::NUM_INT_PUB] };
    let x = at(cols::W_X);
    let y = at(cols::W_Y);
    let z = at(cols::W_Z);
    let d_t0 = at(cols::W_D_T0);
    let d_t1 = at(cols::W_D_T1);
    let d_t2 = at(cols::W_D_T2);
    let d_t3 = at(cols::W_D_T3);
    let d_t4 = at(cols::W_D_T4);
    let d_t5 = at(cols::W_D_T5);
    let a_t0 = at(cols::W_A_T0);
    let a_t1 = at(cols::W_A_T1);
    let a_t2 = at(cols::W_A_T2);
    let a_t4 = at(cols::W_A_T4);
    let down_x = &down[0];
    let down_y = &down[1];
    let down_z = &down[2];

    let one_scalar = const_scalar::<R>(R::ONE);
    let two_scalar = const_scalar::<R>(R::from(2_u32));
    let three_scalar = const_scalar::<R>(R::from(3_u32));
    let four_scalar = const_scalar::<R>(R::from(4_u32));
    let b3_scalar = const_scalar::<R>(R::from(21_u32));
    let g_x_scalar = const_scalar::<R>(R::from(uint_to_int(SECP256K1_G_X_UINT)));
    let g_y_scalar = const_scalar::<R>(R::from(uint_to_int(SECP256K1_G_Y_UINT)));

    // Bind the public ordinary-projective addend `(T_X:T_Y:S_ADD)` to
    // the verifier-checked bit selectors and the public Q/G+Q points.
    let selected_coord = |q: &B::Expr, qg: &B::Expr, g: &DensePolynomial<R, 32>| -> B::Expr {
        let b1_g = mbs(pa_b1, g).expect("b1 * G overflow");
        let b2_q = pa_b2.clone() * q;
        let b11_qg = pa_b1b2.clone() * qg;
        let b11_g = mbs(pa_b1b2, g).expect("b1b2 * G overflow");
        let b11_q = pa_b1b2.clone() * q;
        b1_g + &b2_q + &b11_qg - &b11_g - &b11_q
    };
    let selected_x = selected_coord(pa_qx, pa_qgx, &g_x_scalar);
    let selected_y = selected_coord(pa_qy, pa_qgy, &g_y_scalar);
    b.assert_zero(s_active.clone() * &(pa_t_x.clone() - &selected_x));
    let identity_y = from_ref(&one_scalar) - s_add;
    b.assert_zero(s_active.clone() * &(pa_t_y.clone() - &selected_y - &identity_y));

    // Complete RCB doubling, specialized to secp256k1 (a=0, 3b=21).
    b.assert_zero(s_active.clone() * &(d_t0.clone() - &(x.clone() * x)));
    b.assert_zero(s_active.clone() * &(d_t1.clone() - &(y.clone() * y)));
    b.assert_zero(s_active.clone() * &(d_t2.clone() - &(z.clone() * z)));
    b.assert_zero(s_active.clone() * &(d_t3.clone() - &(x.clone() * y)));
    b.assert_zero(s_active.clone() * &(d_t4.clone() - &(x.clone() * z)));
    b.assert_zero(s_active.clone() * &(d_t5.clone() - &(y.clone() * z)));

    let b3_d_t2 = mbs(d_t2, &b3_scalar).expect("21 * d_t2 overflow");
    let d_x_base = d_t1.clone() - &b3_d_t2;
    let d_z_base = d_t1.clone() + &b3_d_t2;
    let two_d_t3 = mbs(d_t3, &two_scalar).expect("2 * d_t3 overflow");
    let two_d_t4 = mbs(d_t4, &two_scalar).expect("2 * d_t4 overflow");
    let d_t4_b3 = mbs(&two_d_t4, &b3_scalar).expect("42 * d_t4 overflow");
    let two_d_t5 = mbs(d_t5, &two_scalar).expect("2 * d_t5 overflow");
    let three_d_t0 = mbs(d_t0, &three_scalar).expect("3 * d_t0 overflow");

    let doubled_x = two_d_t3.clone() * &d_x_base - &(two_d_t5.clone() * &d_t4_b3);
    let doubled_y = d_x_base.clone() * &d_z_base + &(three_d_t0 * &d_t4_b3);
    let doubled_z_product = two_d_t5 * d_t1;
    let doubled_z = mbs(&doubled_z_product, &four_scalar).expect("4 * doubled z overflow");
    // Complete RCB addition of the doubled point and `(T_X:T_Y:S_ADD)`.
    // The doubled coordinates are exact expressions in the six doubling
    // products. Inlining them removes three committed columns while preserving
    // the same complete group law; the three diagonal product constraints
    // become degree 4.
    b.assert_zero(s_active.clone() * &(a_t0.clone() - &(doubled_x.clone() * pa_t_x)));
    b.assert_zero(s_active.clone() * &(a_t1.clone() - &(doubled_y.clone() * pa_t_y)));
    b.assert_zero(s_active.clone() * &(a_t2.clone() - &(doubled_z.clone() * s_add)));

    // `T3` and `T5` are consumed only here and are safe to inline together.
    // Keeping `T4` materialized prevents the final `T5 * T4` product from
    // multiplying two degree-three expressions, so the gated outputs stay at
    // degree 5.
    let add_t3 = (doubled_x.clone() + &doubled_y) * &(pa_t_x.clone() + pa_t_y)
        - a_t0
        - a_t1;
    let add_t4_expr = (doubled_x + &doubled_z) * &(pa_t_x.clone() + s_add) - a_t0 - a_t2;
    b.assert_zero(s_active.clone() * &(a_t4.clone() - &add_t4_expr));
    let add_t5 = (doubled_y + doubled_z) * &(pa_t_y.clone() + s_add) - a_t1 - a_t2;
    let b3_a_t2 = mbs(a_t2, &b3_scalar).expect("21 * a_t2 overflow");
    let add_x_base = a_t1.clone() - &b3_a_t2;
    let add_z_base = a_t1.clone() + &b3_a_t2;
    let three_a_t0 = mbs(a_t0, &three_scalar).expect("3 * a_t0 overflow");
    let b3_add_t4 = mbs(a_t4, &b3_scalar).expect("21 * add_t4 overflow");

    let added_x = add_t3.clone() * &add_x_base - &(add_t5.clone() * &b3_add_t4);
    let added_y = add_x_base.clone() * &add_z_base + &(three_a_t0.clone() * &b3_add_t4);
    let added_z = add_t5 * &add_z_base + &(add_t3 * &three_a_t0);
    b.assert_zero(s_active.clone() * &(down_x.clone() - &added_x));
    b.assert_zero(s_active.clone() * &(down_y.clone() - &added_y));
    b.assert_zero(s_active.clone() * &(down_z.clone() - &added_z));

    // Start from the verifier-pinned projective identity and expose the final
    // ordinary-projective affine x-coordinate.
    b.assert_zero(s_init.clone() * &(x.clone() - pa_r_init_x));
    b.assert_zero(s_init.clone() * &(y.clone() - pa_r_init_y));
    b.assert_zero(s_init.clone() * &(z.clone() - pa_r_init_z));
    let f1_lhs = s_final.clone() * &(z.clone() * pa_z_inv);
    b.assert_zero(f1_lhs - s_final);
    let f2_inner = x.clone() * pa_z_inv - pa_r_x;
    b.assert_zero(s_final.clone() * &f2_inner);
}

fn expect_public_value<IntT: PartialEq>(
    columns: &[DenseMultilinearExtension<IntT>],
    column: usize,
    row: usize,
    expected: &IntT,
    name: &'static str,
) -> Result<(), PublicStructureError> {
    if &columns[column][row] != expected {
        return Err(PublicStructureError::WrongValue { column: name, row });
    }
    Ok(())
}

/// Verify the row-wise public contract used by the complete ECDSA slice.
pub(crate) fn verify_ecdsa_public_int_structure<IntT>(
    public: &[DenseMultilinearExtension<IntT>],
    num_vars: usize,
) -> Result<(), PublicStructureError>
where
    IntT: Clone + num_traits::Zero + num_traits::One + PartialEq,
{
    let n = 1usize << num_vars;
    if n <= FINAL_ROW || public.len() != cols::NUM_INT_PUB {
        return Err(PublicStructureError::WrongValue {
            column: "ECDSA_PUBLIC_LAYOUT",
            row: n,
        });
    }
    let zero = IntT::zero();
    let one = IntT::one();
    let qx = public[cols::PA_QX][0].clone();
    let qy = public[cols::PA_QY][0].clone();
    let qgx = public[cols::PA_QGX][0].clone();
    let qgy = public[cols::PA_QGY][0].clone();

    for row in 0..n {
        let active = row < NUM_SHAMIR_ROUNDS;
        expect_public_value(
            public,
            cols::S_INIT,
            row,
            if row == 0 { &one } else { &zero },
            "S_INIT",
        )?;
        expect_public_value(
            public,
            cols::S_ACTIVE,
            row,
            if active { &one } else { &zero },
            "S_ACTIVE",
        )?;
        expect_public_value(
            public,
            cols::S_FINAL,
            row,
            if row == FINAL_ROW { &one } else { &zero },
            "S_FINAL",
        )?;

        if active {
            let b1 = &public[cols::PA_B1][row];
            let b2 = &public[cols::PA_B2][row];
            let b1_one = b1 == &one;
            let b2_one = b2 == &one;
            if !b1_one && b1 != &zero {
                return Err(PublicStructureError::WrongValue {
                    column: "PA_B1",
                    row,
                });
            }
            if !b2_one && b2 != &zero {
                return Err(PublicStructureError::WrongValue {
                    column: "PA_B2",
                    row,
                });
            }
            expect_public_value(
                public,
                cols::PA_B1B2,
                row,
                if b1_one && b2_one { &one } else { &zero },
                "PA_B1B2",
            )?;
            expect_public_value(
                public,
                cols::S_ADD,
                row,
                if b1_one || b2_one { &one } else { &zero },
                "S_ADD",
            )?;
            for (column, expected, name) in [
                (cols::PA_QX, &qx, "PA_QX"),
                (cols::PA_QY, &qy, "PA_QY"),
                (cols::PA_QGX, &qgx, "PA_QGX"),
                (cols::PA_QGY, &qgy, "PA_QGY"),
            ] {
                expect_public_value(public, column, row, expected, name)?;
            }
        } else {
            for (column, name) in [
                (cols::S_ADD, "S_ADD"),
                (cols::PA_B1, "PA_B1"),
                (cols::PA_B2, "PA_B2"),
                (cols::PA_B1B2, "PA_B1B2"),
                (cols::PA_QX, "PA_QX"),
                (cols::PA_QY, "PA_QY"),
                (cols::PA_QGX, "PA_QGX"),
                (cols::PA_QGY, "PA_QGY"),
                (cols::PA_T_X, "PA_T_X"),
                (cols::PA_T_Y, "PA_T_Y"),
            ] {
                expect_public_value(public, column, row, &zero, name)?;
            }
        }

        expect_public_value(public, cols::PA_R_INIT_X, row, &zero, "PA_R_INIT_X")?;
        expect_public_value(
            public,
            cols::PA_R_INIT_Y,
            row,
            if row == 0 { &one } else { &zero },
            "PA_R_INIT_Y",
        )?;
        expect_public_value(public, cols::PA_R_INIT_Z, row, &zero, "PA_R_INIT_Z")?;
        if row != FINAL_ROW {
            expect_public_value(public, cols::PA_Z_INV, row, &zero, "PA_Z_INV")?;
            expect_public_value(public, cols::PA_R_X, row, &zero, "PA_R_X")?;
        }
    }
    Ok(())
}

/// Build a constant-polynomial (degree 0) `c` as a `DensePolynomial<R, 32>`.
fn const_scalar<R: ConstSemiring>(c: R) -> DensePolynomial<R, 32> {
    let mut coeffs = [R::ZERO; 32];
    coeffs[0] = c;
    DensePolynomial::<R, 32>::new(coeffs)
}

// ---------------------------------------------------------------------------
// F_p arithmetic helpers.
// ---------------------------------------------------------------------------

fn inv_mod_p(a: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let p_odd = Odd::new(SECP256K1_P_UINT).expect("p is odd");
    a.invert_odd_mod(&p_odd)
        .expect("a has no inverse mod p (a == 0?)")
}

fn mul_mod_p(a: &CbUint<EC_FP_INT_LIMBS>, b: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = a.widening_mul(b).into();
    let p_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_P_UINT.resize();
    let p_wide_nz = NonZero::new(p_wide).expect("p is nonzero");
    let (_, rem) = wide.div_rem_vartime(&p_wide_nz);
    rem.resize()
}

fn add_mod_p(a: &CbUint<EC_FP_INT_LIMBS>, b: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    let a_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = a.resize();
    let b_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = b.resize();
    let sum = a_wide.wrapping_add(&b_wide);
    let p_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_P_UINT.resize();
    let p_wide_nz = NonZero::new(p_wide).expect("p is nonzero");
    let (_, rem) = sum.div_rem_vartime(&p_wide_nz);
    rem.resize()
}

/// Return whether `(x, y)` is a canonical affine secp256k1 point.
pub fn is_secp256k1_affine_point(
    x: &CbUint<EC_FP_INT_LIMBS>,
    y: &CbUint<EC_FP_INT_LIMBS>,
) -> bool {
    if *x >= SECP256K1_P_UINT || *y >= SECP256K1_P_UINT {
        return false;
    }
    let x_squared = mul_mod_p(x, x);
    let x_cubed = mul_mod_p(&x_squared, x);
    let rhs = add_mod_p(&x_cubed, &CbUint::from_u64(7));
    mul_mod_p(y, y) == rhs
}

fn small_mul_mod_p(a: &CbUint<EC_FP_INT_LIMBS>, k: u32) -> CbUint<EC_FP_INT_LIMBS> {
    let mut acc = CbUint::<EC_FP_INT_LIMBS>::ZERO;
    for _ in 0..k {
        acc = add_mod_p(&acc, a);
    }
    acc
}

fn sub_mod_p(a: &CbUint<EC_FP_INT_LIMBS>, b: &CbUint<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
    use crypto_bigint::CheckedSub;
    let p_nz = NonZero::new(SECP256K1_P_UINT).expect("p is nonzero");
    if a.checked_sub(b).is_some().into() {
        a.wrapping_sub(b).rem_vartime(&p_nz)
    } else {
        let difference = b.wrapping_sub(a);
        SECP256K1_P_UINT.wrapping_sub(&difference)
    }
}

/// Centered reduction — see twin in `ecdsa_doubling.rs`.
fn uint_to_int(u: CbUint<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    if u <= SECP256K1_P_HALF_UINT {
        Int::new(*u.as_int())
    } else {
        let wrapped = u.wrapping_sub(&SECP256K1_P_UINT);
        Int::new(*wrapped.as_int())
    }
}

// ---------------------------------------------------------------------------
// Reference per-step computation (for witness gen and tests).
// ---------------------------------------------------------------------------

/// One complete ordinary-projective Shamir step and its materialized products.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProjectivePoint {
    x: CbUint<EC_FP_INT_LIMBS>,
    y: CbUint<EC_FP_INT_LIMBS>,
    z: CbUint<EC_FP_INT_LIMBS>,
}

impl ProjectivePoint {
    fn identity() -> Self {
        Self {
            x: CbUint::ZERO,
            y: CbUint::ONE,
            z: CbUint::ZERO,
        }
    }

    fn affine(x: CbUint<EC_FP_INT_LIMBS>, y: CbUint<EC_FP_INT_LIMBS>) -> Self {
        Self {
            x,
            y,
            z: CbUint::ONE,
        }
    }
}

struct StepValues {
    doubled_products: [CbUint<EC_FP_INT_LIMBS>; 6],
    addition_products: [CbUint<EC_FP_INT_LIMBS>; 6],
    next: ProjectivePoint,
}

fn rcb_double(point: &ProjectivePoint) -> ([CbUint<EC_FP_INT_LIMBS>; 6], ProjectivePoint) {
    let t0 = mul_mod_p(&point.x, &point.x);
    let t1 = mul_mod_p(&point.y, &point.y);
    let t2 = mul_mod_p(&point.z, &point.z);
    let t3 = mul_mod_p(&point.x, &point.y);
    let t4 = mul_mod_p(&point.x, &point.z);
    let t5 = mul_mod_p(&point.y, &point.z);

    let b3_t2 = small_mul_mod_p(&t2, 21);
    let x_base = sub_mod_p(&t1, &b3_t2);
    let z_base = add_mod_p(&t1, &b3_t2);
    let two_t3 = small_mul_mod_p(&t3, 2);
    let t4_b3 = small_mul_mod_p(&small_mul_mod_p(&t4, 2), 21);
    let two_t5 = small_mul_mod_p(&t5, 2);
    let three_t0 = small_mul_mod_p(&t0, 3);

    let x = sub_mod_p(&mul_mod_p(&two_t3, &x_base), &mul_mod_p(&two_t5, &t4_b3));
    let y = add_mod_p(&mul_mod_p(&x_base, &z_base), &mul_mod_p(&three_t0, &t4_b3));
    let z = small_mul_mod_p(&mul_mod_p(&two_t5, &t1), 4);

    ([t0, t1, t2, t3, t4, t5], ProjectivePoint { x, y, z })
}

fn rcb_add(
    left: &ProjectivePoint,
    right: &ProjectivePoint,
) -> ([CbUint<EC_FP_INT_LIMBS>; 6], ProjectivePoint) {
    let t0 = mul_mod_p(&left.x, &right.x);
    let t1 = mul_mod_p(&left.y, &right.y);
    let t2 = mul_mod_p(&left.z, &right.z);
    let t3_raw = mul_mod_p(&add_mod_p(&left.x, &left.y), &add_mod_p(&right.x, &right.y));
    let t4_raw = mul_mod_p(&add_mod_p(&left.x, &left.z), &add_mod_p(&right.x, &right.z));
    let t5_raw = mul_mod_p(&add_mod_p(&left.y, &left.z), &add_mod_p(&right.y, &right.z));

    let t3 = sub_mod_p(&sub_mod_p(&t3_raw, &t0), &t1);
    let t4 = sub_mod_p(&sub_mod_p(&t4_raw, &t0), &t2);
    let t5 = sub_mod_p(&sub_mod_p(&t5_raw, &t1), &t2);
    let b3_t2 = small_mul_mod_p(&t2, 21);
    let x_base = sub_mod_p(&t1, &b3_t2);
    let z_base = add_mod_p(&t1, &b3_t2);
    let three_t0 = small_mul_mod_p(&t0, 3);
    let b3_t4 = small_mul_mod_p(&t4, 21);

    let x = sub_mod_p(&mul_mod_p(&t3, &x_base), &mul_mod_p(&t5, &b3_t4));
    let y = add_mod_p(&mul_mod_p(&x_base, &z_base), &mul_mod_p(&three_t0, &b3_t4));
    let z = add_mod_p(&mul_mod_p(&t5, &z_base), &mul_mod_p(&t3, &three_t0));

    // Only T4 is materialized by the UAIR. T3 and T5 are reconstructed from
    // their raw products in-circuit, while T4 must contain the cross term
    // after subtracting T0 and T2.
    (
        [t0, t1, t2, t3_raw, t4, t5_raw],
        ProjectivePoint { x, y, z },
    )
}

fn compute_step(state: &ProjectivePoint, addend: &ProjectivePoint) -> StepValues {
    let (doubled_products, doubled) = rcb_double(state);
    let (addition_products, next) = rcb_add(&doubled, addend);
    StepValues {
        doubled_products,
        addition_products,
        next,
    }
}

fn projective_to_affine(
    point: &ProjectivePoint,
) -> Option<(CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>)> {
    use crypto_bigint::Zero as _;
    if bool::from(point.z.is_zero()) {
        return None;
    }
    let z_inv = inv_mod_p(&point.z);
    Some((mul_mod_p(&point.x, &z_inv), mul_mod_p(&point.y, &z_inv)))
}

fn validate_public_key_uint(
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
) -> Result<EcdsaBoundPublicKey, EcdsaPublicKeyBindingError> {
    if q.0 >= SECP256K1_P_UINT {
        return Err(EcdsaPublicKeyBindingError::CoordinateOutOfRange {
            coordinate: EcdsaCoordinate::X,
        });
    }
    if q.1 >= SECP256K1_P_UINT {
        return Err(EcdsaPublicKeyBindingError::CoordinateOutOfRange {
            coordinate: EcdsaCoordinate::Y,
        });
    }
    if !is_secp256k1_affine_point(&q.0, &q.1) {
        return Err(EcdsaPublicKeyBindingError::PointNotOnCurve);
    }

    let generator = ProjectivePoint::affine(SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT);
    let q_projective = ProjectivePoint::affine(q.0, q.1);
    let (_, sum) = rcb_add(&generator, &q_projective);
    let g_plus_q = projective_to_affine(&sum)
        .ok_or(EcdsaPublicKeyBindingError::GeneratorSumAtInfinity)?;
    Ok(EcdsaBoundPublicKey { q, g_plus_q })
}

/// Parse, validate, and add G to an affine-only secp256k1 public key.
pub fn validate_secp256k1_public_key(
    q_x_be: &[u8],
    q_y_be: &[u8],
) -> Result<EcdsaBoundPublicKey, EcdsaPublicKeyBindingError> {
    if q_x_be.len() != 32 {
        return Err(EcdsaPublicKeyBindingError::CoordinateEncoding {
            coordinate: EcdsaCoordinate::X,
            actual_bytes: q_x_be.len(),
        });
    }
    if q_y_be.len() != 32 {
        return Err(EcdsaPublicKeyBindingError::CoordinateEncoding {
            coordinate: EcdsaCoordinate::Y,
            actual_bytes: q_y_be.len(),
        });
    }
    validate_public_key_uint((
        CbUint::from_be_slice(q_x_be),
        CbUint::from_be_slice(q_y_be),
    ))
}

/// Bind every active public Q/G+Q cell to one canonical affine statement key.
pub fn verify_ecdsa_public_key_binding_in_columns(
    q_x_be: &[u8],
    q_y_be: &[u8],
    public_int: &[DenseMultilinearExtension<Int<EC_FP_INT_LIMBS>>],
    columns: [usize; 4],
) -> Result<EcdsaBoundPublicKey, EcdsaPublicKeyBindingError> {
    let bound = validate_secp256k1_public_key(q_x_be, q_y_be)?;
    let expected = [bound.q.0, bound.q.1, bound.g_plus_q.0, bound.g_plus_q.1];

    for row in 0..NUM_SHAMIR_ROUNDS {
        for (column, expected) in columns.into_iter().zip(expected) {
            let cell = public_int
                .get(column)
                .and_then(|values| values.evaluations.get(row))
                .ok_or(EcdsaPublicKeyBindingError::MissingPublicPoint { column, row })?;
            let actual = decode_canonical_field_element(cell).ok_or(
                EcdsaPublicKeyBindingError::NonCanonicalPublicPoint { column, row },
            )?;
            if actual != expected {
                return Err(EcdsaPublicKeyBindingError::PublicPointMismatch { column, row });
            }
        }
    }
    Ok(bound)
}

fn select_addend(
    bits: (bool, bool),
    q: &(CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    g_plus_q: &(CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
) -> ProjectivePoint {
    match bits {
        (false, false) => ProjectivePoint::identity(),
        (true, false) => ProjectivePoint::affine(SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT),
        (false, true) => ProjectivePoint::affine(q.0, q.1),
        (true, true) => ProjectivePoint::affine(g_plus_q.0, g_plus_q.1),
    }
}

fn scalar_bits(value: &CbUint<EC_FP_INT_LIMBS>) -> [bool; NUM_SHAMIR_ROUNDS] {
    let mut result = [false; NUM_SHAMIR_ROUNDS];
    for (row, bit) in (0..NUM_SHAMIR_ROUNDS).rev().enumerate() {
        let word = bit / 64;
        let offset = bit % 64;
        result[row] = ((value.as_words()[word] >> offset) & 1) == 1;
    }
    result
}

/// Build a trace for explicit public-key and Shamir-scalar inputs.
///
/// This constructor is used by application-level proof tests that need a
/// deterministic result point rather than the random-trace fixture.
pub fn build_trace_from_scalars<R>(
    num_vars: usize,
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    u1: CbUint<EC_FP_INT_LIMBS>,
    u2: CbUint<EC_FP_INT_LIMBS>,
) -> Result<UairTrace<'static, R, R, 32>, EcdsaPublicKeyBindingError>
where
    R: EcdsaFpRing,
{
    let b1 = scalar_bits(&u1);
    let b2 = scalar_bits(&u2);
    let bits = core::array::from_fn(|row| (b1[row], b2[row]));
    build_complete_trace(num_vars, q, bits)
}

#[cfg(test)]
fn run_shamir(
    q: &(CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    bits: &[(bool, bool); NUM_SHAMIR_ROUNDS],
) -> Result<(ProjectivePoint, Vec<StepValues>), EcdsaPublicKeyBindingError> {
    let g_plus_q = validate_public_key_uint(*q)?.g_plus_q;
    let mut state = ProjectivePoint::identity();
    let mut steps = Vec::with_capacity(NUM_SHAMIR_ROUNDS);
    for &pair in bits {
        let addend = select_addend(pair, q, &g_plus_q);
        let step = compute_step(&state, &addend);
        state = step.next.clone();
        steps.push(step);
    }
    Ok((state, steps))
}

// ---------------------------------------------------------------------------
// Witness generator.
// ---------------------------------------------------------------------------

fn build_complete_trace<R>(
    num_vars: usize,
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    bits: [(bool, bool); NUM_SHAMIR_ROUNDS],
) -> Result<UairTrace<'static, R, R, 32>, EcdsaPublicKeyBindingError>
where
    R: EcdsaFpRing,
{
    let n_rows = 1usize << num_vars;
    assert!(
        n_rows > FINAL_ROW,
        "Shamir UAIR needs > {FINAL_ROW} rows; got {n_rows}",
    );

    let g_plus_q = validate_public_key_uint(q)?.g_plus_q;

    let mut states = Vec::with_capacity(FINAL_ROW + 1);
    let mut steps = Vec::with_capacity(NUM_SHAMIR_ROUNDS);
    states.push(ProjectivePoint::identity());
    for (row, &pair) in bits.iter().enumerate() {
        let addend = select_addend(pair, &q, &g_plus_q);
        let step = compute_step(&states[row], &addend);
        states.push(step.next.clone());
        steps.push(step);
    }

    let mut columns: Vec<Vec<R>> = (0..cols::NUM_INT).map(|_| vec![R::ZERO; n_rows]).collect();
    columns[cols::S_INIT][0] = R::ONE;
    columns[cols::S_FINAL][FINAL_ROW] = R::ONE;
    columns[cols::PA_R_INIT_Y][0] = R::ONE;

    for row in 0..NUM_SHAMIR_ROUNDS {
        let (b1, b2) = bits[row];
        let addend = select_addend((b1, b2), &q, &g_plus_q);
        columns[cols::S_ACTIVE][row] = R::ONE;
        columns[cols::S_ADD][row] = if b1 || b2 { R::ONE } else { R::ZERO };
        columns[cols::PA_B1][row] = if b1 { R::ONE } else { R::ZERO };
        columns[cols::PA_B2][row] = if b2 { R::ONE } else { R::ZERO };
        columns[cols::PA_B1B2][row] = if b1 && b2 { R::ONE } else { R::ZERO };
        columns[cols::PA_QX][row] = R::from(uint_to_int(q.0));
        columns[cols::PA_QY][row] = R::from(uint_to_int(q.1));
        columns[cols::PA_QGX][row] = R::from(uint_to_int(g_plus_q.0));
        columns[cols::PA_QGY][row] = R::from(uint_to_int(g_plus_q.1));
        columns[cols::PA_T_X][row] = R::from(uint_to_int(addend.x));
        columns[cols::PA_T_Y][row] = R::from(uint_to_int(addend.y));

        let state = &states[row];
        columns[cols::W_X][row] = R::from(uint_to_int(state.x));
        columns[cols::W_Y][row] = R::from(uint_to_int(state.y));
        columns[cols::W_Z][row] = R::from(uint_to_int(state.z));

        let step = &steps[row];
        for (column, value) in [
            (cols::W_D_T0, step.doubled_products[0]),
            (cols::W_D_T1, step.doubled_products[1]),
            (cols::W_D_T2, step.doubled_products[2]),
            (cols::W_D_T3, step.doubled_products[3]),
            (cols::W_D_T4, step.doubled_products[4]),
            (cols::W_D_T5, step.doubled_products[5]),
            (cols::W_A_T0, step.addition_products[0]),
            (cols::W_A_T1, step.addition_products[1]),
            (cols::W_A_T2, step.addition_products[2]),
            (cols::W_A_T4, step.addition_products[4]),
        ] {
            columns[column][row] = R::from(uint_to_int(value));
        }
    }

    let final_state = &states[FINAL_ROW];
    columns[cols::W_X][FINAL_ROW] = R::from(uint_to_int(final_state.x));
    columns[cols::W_Y][FINAL_ROW] = R::from(uint_to_int(final_state.y));
    columns[cols::W_Z][FINAL_ROW] = R::from(uint_to_int(final_state.z));
    let z_inv = inv_mod_p(&final_state.z);
    columns[cols::PA_Z_INV][FINAL_ROW] = R::from(uint_to_int(z_inv));
    columns[cols::PA_R_X][FINAL_ROW] = R::from(uint_to_int(mul_mod_p(&final_state.x, &z_inv)));

    let int = columns
        .into_iter()
        .map(|column| column.into_iter().collect())
        .collect::<Vec<DenseMultilinearExtension<R>>>();
    Ok(UairTrace {
        int: int.into(),
        ..Default::default()
    })
}

impl<R> GenerateRandomTrace<32> for EcdsaUair<R>
where
    R: EcdsaFpRing,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let mut bits = [(false, false); NUM_SHAMIR_ROUNDS];
        for pair in &mut bits {
            let random = rng.next_u32();
            *pair = (random & 1 != 0, random & 2 != 0);
        }
        if bits.iter().all(|&(b1, b2)| !b1 && !b2) {
            bits[NUM_SHAMIR_ROUNDS - 1] = (true, false);
        }
        build_complete_trace(num_vars, (SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT), bits)
            .expect("the generator public key is valid and G+G is affine")
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    fn int_to_uint(value: &Int<EC_FP_INT_LIMBS>) -> CbUint<EC_FP_INT_LIMBS> {
        let raw = *value.inner().as_uint();
        let is_negative = raw.as_words()[EC_FP_INT_LIMBS - 1] >> 63 != 0;
        if is_negative {
            raw.wrapping_add(&SECP256K1_P_UINT)
        } else {
            raw
        }
    }

    fn read_uint(
        trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
        column: usize,
        row: usize,
    ) -> CbUint<EC_FP_INT_LIMBS> {
        int_to_uint(&trace.int[column][row])
    }

    fn uint(hex: &str) -> CbUint<EC_FP_INT_LIMBS> {
        CbUint::from_be_hex(hex)
    }

    fn uint_to_be_bytes(value: &CbUint<EC_FP_INT_LIMBS>) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, word) in value.as_words().iter().rev().enumerate() {
            bytes[index * 8..(index + 1) * 8].copy_from_slice(&word.to_be_bytes());
        }
        bytes
    }

    fn bits_from_scalars(
        u1: CbUint<EC_FP_INT_LIMBS>,
        u2: CbUint<EC_FP_INT_LIMBS>,
    ) -> [(bool, bool); NUM_SHAMIR_ROUNDS] {
        let b1 = scalar_bits(&u1);
        let b2 = scalar_bits(&u2);
        core::array::from_fn(|row| (b1[row], b2[row]))
    }

    #[test]
    fn shamir_constraint_shape() {
        type U = EcdsaUair<Int<EC_FP_INT_LIMBS>>;
        assert_eq!(count_constraints::<U>(), 20);
        assert_eq!(count_max_degree::<U>(), 5);
        let degrees = count_constraint_degrees::<U>();
        assert!(degrees.iter().all(|&degree| degree <= 5));
        assert_eq!(degrees.iter().filter(|&&degree| degree == 2).count(), 3);
    }

    #[test]
    fn witness_replays_complete_rcb_rows() {
        let num_vars = 9;
        let mut random = rng();
        let trace =
            <EcdsaUair<Int<EC_FP_INT_LIMBS>> as GenerateRandomTrace<32>>::generate_random_trace(
                num_vars,
                &mut random,
            );
        assert_eq!(trace.int.len(), cols::NUM_INT);
        verify_ecdsa_public_int_structure(&trace.int[..cols::NUM_INT_PUB], num_vars)
            .expect("generated public structure must verify");

        for row in 0..NUM_SHAMIR_ROUNDS {
            let state = ProjectivePoint {
                x: read_uint(&trace, cols::W_X, row),
                y: read_uint(&trace, cols::W_Y, row),
                z: read_uint(&trace, cols::W_Z, row),
            };
            let addend = ProjectivePoint {
                x: read_uint(&trace, cols::PA_T_X, row),
                y: read_uint(&trace, cols::PA_T_Y, row),
                z: read_uint(&trace, cols::S_ADD, row),
            };
            let expected = compute_step(&state, &addend);
            let (_, doubled) = rcb_double(&state);
            let raw_t4 = mul_mod_p(
                &add_mod_p(&doubled.x, &doubled.z),
                &add_mod_p(&addend.x, &addend.z),
            );
            let expected_t4 = sub_mod_p(
                &sub_mod_p(&raw_t4, &expected.addition_products[0]),
                &expected.addition_products[2],
            );
            assert_eq!(
                read_uint(&trace, cols::W_A_T4, row),
                expected_t4,
                "materialized RCB T4 cross term at row {row}",
            );
            for (column, value) in [
                (cols::W_D_T0, expected.doubled_products[0]),
                (cols::W_D_T1, expected.doubled_products[1]),
                (cols::W_D_T2, expected.doubled_products[2]),
                (cols::W_D_T3, expected.doubled_products[3]),
                (cols::W_D_T4, expected.doubled_products[4]),
                (cols::W_D_T5, expected.doubled_products[5]),
                (cols::W_A_T0, expected.addition_products[0]),
                (cols::W_A_T1, expected.addition_products[1]),
                (cols::W_A_T2, expected.addition_products[2]),
                (cols::W_A_T4, expected.addition_products[4]),
            ] {
                assert_eq!(
                    read_uint(&trace, column, row),
                    value,
                    "column {column}, row {row}"
                );
            }
            assert_eq!(read_uint(&trace, cols::W_X, row + 1), expected.next.x);
            assert_eq!(read_uint(&trace, cols::W_Y, row + 1), expected.next.y);
            assert_eq!(read_uint(&trace, cols::W_Z, row + 1), expected.next.z);
        }
    }

    #[test]
    fn frozen_signature_matches_openssl_oracle() {
        let q = (
            uint("72d74be030e343fd313ab5af81b2f326e60f161778f0a555bd01baab27690558"),
            uint("1cd8822e229ec716381f7db49e7515d5ca48ec52ec0acab0c8da943ea00c26b6"),
        );
        let bits = bits_from_scalars(
            uint("30e48216899822d64bdb7f54db72055cef40f5d894433dc1af0251e69ac1a2f0"),
            uint("968a5f9440aa6bb07e2a112cf0fdcd15c681c90ba2102d9dd633cc6dad0cd8f8"),
        );
        let (final_point, _) =
            run_shamir(&q, &bits).expect("frozen public key must be valid");
        let affine = projective_to_affine(&final_point).expect("fixture result is finite");
        assert_eq!(
            affine.0,
            uint("12e2303536ccdd4f4fb63a8cbe473faa259c3b28b00c3f62ff095e9e90a21217")
        );
        assert_eq!(
            affine.1,
            uint("b0c26fdac74ffad75053a692649da3587f06bb03e90820645d421e0eda72ddbf")
        );

        let trace = build_complete_trace::<Int<EC_FP_INT_LIMBS>>(9, q, bits)
            .expect("fixture public key must be valid");
        verify_ecdsa_public_int_structure(&trace.int[..cols::NUM_INT_PUB], 9)
            .expect("fixture public structure must verify");
        assert_eq!(read_uint(&trace, cols::PA_R_X, FINAL_ROW), affine.0);
        let signature_r = uint_to_be_bytes(&affine.0);
        assert_eq!(
            verify_ecdsa_result_binding(&signature_r, &trace.int[cols::PA_R_X][FINAL_ROW]),
            Ok(EcdsaResultBranch::Direct),
        );
    }

    #[test]
    fn result_binding_covers_both_possible_representatives() {
        let r = CbUint::ONE;
        let r_bytes = uint_to_be_bytes(&r);
        assert_eq!(
            verify_ecdsa_result_binding(&r_bytes, &uint_to_int(r)),
            Ok(EcdsaResultBranch::Direct),
        );

        let r_plus_n = SECP256K1_N_UINT.wrapping_add(&r);
        assert!(r_plus_n < SECP256K1_P_UINT);
        assert_eq!(
            verify_ecdsa_result_binding(&r_bytes, &uint_to_int(r_plus_n)),
            Ok(EcdsaResultBranch::PlusOrder),
        );
    }

    #[test]
    fn result_binding_covers_order_and_field_boundaries() {
        let largest_scalar = SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE);
        assert_eq!(
            verify_ecdsa_result_binding(
                &uint_to_be_bytes(&largest_scalar),
                &uint_to_int(largest_scalar),
            ),
            Ok(EcdsaResultBranch::Direct),
        );

        let largest_x = SECP256K1_P_UINT.wrapping_sub(&CbUint::ONE);
        let largest_x_reduced = largest_x.wrapping_sub(&SECP256K1_N_UINT);
        assert_eq!(
            verify_ecdsa_result_binding(
                &uint_to_be_bytes(&largest_x_reduced),
                &uint_to_int(largest_x),
            ),
            Ok(EcdsaResultBranch::PlusOrder),
        );

        // x = n reduces to zero and therefore cannot equal any canonical
        // nonzero ECDSA r. This pins the branch boundary itself.
        assert_eq!(
            verify_ecdsa_result_binding(
                &uint_to_be_bytes(&CbUint::ONE),
                &uint_to_int(SECP256K1_N_UINT),
            ),
            Err(EcdsaResultBindingError::ResultMismatch),
        );
    }

    #[test]
    fn result_binding_rejects_scalar_range_and_encoding_errors() {
        let x = uint_to_int(CbUint::ONE);
        assert_eq!(
            verify_ecdsa_result_binding(&[0_u8; 31], &x),
            Err(EcdsaResultBindingError::SignatureScalarEncoding {
                actual_bytes: 31
            }),
        );
        assert_eq!(
            verify_ecdsa_result_binding(&[0_u8; 33], &x),
            Err(EcdsaResultBindingError::SignatureScalarEncoding {
                actual_bytes: 33
            }),
        );
        assert_eq!(
            verify_ecdsa_result_binding(&[0_u8; 32], &x),
            Err(EcdsaResultBindingError::SignatureScalarZero),
        );
        assert_eq!(
            verify_ecdsa_result_binding(&uint_to_be_bytes(&SECP256K1_N_UINT), &x),
            Err(EcdsaResultBindingError::SignatureScalarOutOfRange),
        );
    }

    #[test]
    fn public_key_validation_matches_frozen_libsecp_oracle() {
        let q = (
            uint("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"),
            uint("1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a"),
        );
        let bound = validate_secp256k1_public_key(&uint_to_be_bytes(&q.0), &uint_to_be_bytes(&q.1))
            .expect("2G must be a valid public key");
        assert_eq!(bound.q, q);
        assert_eq!(
            bound.g_plus_q,
            (
                uint("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"),
                uint("388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672"),
            ),
        );

        let doubled = validate_secp256k1_public_key(
            &uint_to_be_bytes(&SECP256K1_G_X_UINT),
            &uint_to_be_bytes(&SECP256K1_G_Y_UINT),
        )
        .expect("Q=G must use complete doubling");
        assert_eq!(doubled.g_plus_q, q);
    }

    #[test]
    fn public_key_validation_rejects_every_affine_boundary() {
        let generator_x = uint_to_be_bytes(&SECP256K1_G_X_UINT);
        let generator_y = uint_to_be_bytes(&SECP256K1_G_Y_UINT);
        assert_eq!(
            validate_secp256k1_public_key(&generator_x[..31], &generator_y),
            Err(EcdsaPublicKeyBindingError::CoordinateEncoding {
                coordinate: EcdsaCoordinate::X,
                actual_bytes: 31,
            }),
        );
        assert_eq!(
            validate_secp256k1_public_key(&generator_x, &generator_y[..31]),
            Err(EcdsaPublicKeyBindingError::CoordinateEncoding {
                coordinate: EcdsaCoordinate::Y,
                actual_bytes: 31,
            }),
        );
        assert_eq!(
            validate_secp256k1_public_key(&uint_to_be_bytes(&SECP256K1_P_UINT), &generator_y),
            Err(EcdsaPublicKeyBindingError::CoordinateOutOfRange {
                coordinate: EcdsaCoordinate::X,
            }),
        );
        assert_eq!(
            validate_secp256k1_public_key(&generator_x, &uint_to_be_bytes(&SECP256K1_P_UINT)),
            Err(EcdsaPublicKeyBindingError::CoordinateOutOfRange {
                coordinate: EcdsaCoordinate::Y,
            }),
        );
        let off_curve_y = SECP256K1_G_Y_UINT.wrapping_add(&CbUint::ONE);
        assert_eq!(
            validate_secp256k1_public_key(&generator_x, &uint_to_be_bytes(&off_curve_y)),
            Err(EcdsaPublicKeyBindingError::PointNotOnCurve),
        );
        assert_eq!(
            validate_secp256k1_public_key(&[0_u8; 32], &[0_u8; 32]),
            Err(EcdsaPublicKeyBindingError::PointNotOnCurve),
        );
        let minus_g_y = SECP256K1_P_UINT.wrapping_sub(&SECP256K1_G_Y_UINT);
        assert_eq!(
            validate_secp256k1_public_key(&generator_x, &uint_to_be_bytes(&minus_g_y)),
            Err(EcdsaPublicKeyBindingError::GeneratorSumAtInfinity),
        );
        assert!(matches!(
            build_trace_from_scalars::<Int<EC_FP_INT_LIMBS>>(
                9,
                (SECP256K1_G_X_UINT, minus_g_y),
                CbUint::ONE,
                CbUint::ONE,
            ),
            Err(EcdsaPublicKeyBindingError::GeneratorSumAtInfinity),
        ));
    }

    #[test]
    fn public_key_binding_owns_every_active_trace_cell() {
        let q = (
            uint("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"),
            uint("1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a"),
        );
        let trace = build_trace_from_scalars::<Int<EC_FP_INT_LIMBS>>(
            9,
            q,
            CbUint::ONE,
            CbUint::ONE,
        )
        .expect("2G trace must build");
        let public = &trace.int[..cols::NUM_INT_PUB];
        verify_ecdsa_public_key_binding_in_columns(
            &uint_to_be_bytes(&q.0),
            &uint_to_be_bytes(&q.1),
            public,
            [cols::PA_QX, cols::PA_QY, cols::PA_QGX, cols::PA_QGY],
        )
        .expect("honest Q/G+Q columns must bind");

        for column in [cols::PA_QX, cols::PA_QY, cols::PA_QGX, cols::PA_QGY] {
            for row in 0..NUM_SHAMIR_ROUNDS {
                let mut mutated = public.to_vec();
                mutated[column].evaluations[row] = Int::from(0_u32);
                assert_eq!(
                    verify_ecdsa_public_key_binding_in_columns(
                        &uint_to_be_bytes(&q.0),
                        &uint_to_be_bytes(&q.1),
                        &mutated,
                        [cols::PA_QX, cols::PA_QY, cols::PA_QGX, cols::PA_QGY],
                    ),
                    Err(EcdsaPublicKeyBindingError::PublicPointMismatch { column, row }),
                );
            }
        }

        let mut noncanonical = public.to_vec();
        let upper = SECP256K1_P_HALF_UINT.wrapping_add(&CbUint::ONE);
        noncanonical[cols::PA_QX].evaluations[0] = Int::new(*upper.as_int());
        assert_eq!(
            verify_ecdsa_public_key_binding_in_columns(
                &uint_to_be_bytes(&q.0),
                &uint_to_be_bytes(&q.1),
                &noncanonical,
                [cols::PA_QX, cols::PA_QY, cols::PA_QGX, cols::PA_QGY],
            ),
            Err(EcdsaPublicKeyBindingError::NonCanonicalPublicPoint {
                column: cols::PA_QX,
                row: 0,
            }),
        );
    }

    #[test]
    fn result_binding_rejects_noncanonical_centered_encodings() {
        let upper = SECP256K1_P_HALF_UINT.wrapping_add(&CbUint::ONE);
        assert_eq!(decode_canonical_final_x(&uint_to_int(upper)), Ok(upper));

        let noncanonical_positive = Int::new(*upper.as_int());
        assert_eq!(
            decode_canonical_final_x(&noncanonical_positive),
            Err(EcdsaResultBindingError::NonCanonicalFinalX),
        );

        let lower = SECP256K1_P_HALF_UINT;
        let noncanonical_negative =
            Int::new(*lower.wrapping_sub(&SECP256K1_P_UINT).as_int());
        assert_eq!(
            decode_canonical_final_x(&noncanonical_negative),
            Err(EcdsaResultBindingError::NonCanonicalFinalX),
        );
        assert_eq!(decode_canonical_final_x(&uint_to_int(lower)), Ok(lower));
    }

    #[test]
    fn exact_integer_check_rejects_fp_wraparound() {
        let r = SECP256K1_P_UINT
            .wrapping_sub(&SECP256K1_N_UINT)
            .wrapping_add(&CbUint::ONE);
        assert!(r > CbUint::ZERO && r < SECP256K1_N_UINT);
        let final_x = CbUint::ONE;

        // A bare F_p quotient equation with q=1 accepts this tuple because
        // final_x - r - n = -p. The exact integer reduction must reject it.
        assert_eq!(
            sub_mod_p(&sub_mod_p(&final_x, &r), &SECP256K1_N_UINT),
            CbUint::ZERO,
        );
        assert_eq!(
            verify_ecdsa_result_binding(&uint_to_be_bytes(&r), &uint_to_int(final_x)),
            Err(EcdsaResultBindingError::ResultMismatch),
        );
    }

    #[test]
    fn result_binding_rejects_missing_public_cell() {
        assert_eq!(
            verify_ecdsa_result_binding_in_column(&[1_u8; 32], &[], cols::PA_R_X),
            Err(EcdsaResultBindingError::MissingPublicResult {
                column: cols::PA_R_X,
                row: FINAL_ROW,
            }),
        );
    }

    #[test]
    fn complete_formulas_cover_full_chain_exceptions() {
        let half_g = (
            uint("00000000000000000000003b78ce563f89a0ed9414f5aa28ad0d96d6795f9c63"),
            uint("c0c686408d517dfd67c2367651380d00d126e4229631fd03f8ff35eef1a61e3c"),
        );
        let minus_half_g = (
            half_g.0,
            uint("3f3979bf72ae8202983dc989aec7f2ff2ed91bdd69ce02fc0700ca100e59ddf3"),
        );

        let equal_bits = bits_from_scalars(CbUint::from_u64(1), CbUint::from_u64(2));
        let (equal_result, _) =
            run_shamir(&half_g, &equal_bits).expect("half-generator public key must be valid");
        assert_eq!(
            projective_to_affine(&equal_result),
            Some((
                uint("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5"),
                uint("1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a"),
            ))
        );

        let restart_bits = bits_from_scalars(CbUint::from_u64(3), CbUint::from_u64(4));
        let (restart_result, _) = run_shamir(&minus_half_g, &restart_bits)
            .expect("minus-half-generator public key must be valid");
        assert_eq!(
            projective_to_affine(&restart_result),
            Some((SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT))
        );
    }

    #[test]
    fn public_structure_rejects_non_boolean_selector() {
        let bits = [(false, true); NUM_SHAMIR_ROUNDS];
        let trace = build_complete_trace::<Int<EC_FP_INT_LIMBS>>(
            9,
            (SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT),
            bits,
        )
        .expect("generator public key must be valid");
        let mut public = trace.int[..cols::NUM_INT_PUB].to_vec();
        public[cols::PA_B1].evaluations[17] = Int::from(2_u32);
        let error = verify_ecdsa_public_int_structure(&public, 9)
            .expect_err("non-boolean bit must be rejected");
        assert!(matches!(
            error,
            PublicStructureError::WrongValue {
                column: "PA_B1",
                row: 17
            }
        ));
    }
}
