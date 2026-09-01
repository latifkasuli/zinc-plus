//! H8-derived SHA-256 + ECDSA statement with a private signature witness.
//!
//! The public statement is `(message, Q)`. The compact signature `(r, s)`,
//! derived verification scalars `(u1, u2)`, Shamir selectors, and final affine
//! x-coordinate remain in witness columns. This is a private-input statement,
//! not a zero-knowledge claim: the underlying PCS is not asserted to hide its
//! committed witness.

use core::{fmt, marker::PhantomData};

use crypto_bigint::{ConstZero, Uint as CbUint};
use crypto_primitives::crypto_bigint_int::Int;
use num_traits::{One, Zero};
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    BitOp, BitOpSpec, ConstraintBuilder, PublicColumnLayout, PublicStructureError, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace,
    ideal::{DegreeOneIdeal, rotation::RotationIdeal},
};

use crate::{
    ecdsa::{self, EcdsaBoundPublicKey, EcdsaResultBindingError, EcdsaResultBranch},
    ecdsa_doubling::{EC_FP_INT_LIMBS, EcdsaFpRing, SECP256K1_P_HALF_UINT, SECP256K1_P_UINT},
    private_ecdsa_scalars::{
        self, PrivateEcdsaScalarsUair, build_private_ecdsa_scalar_trace,
        derive_private_ecdsa_scalars, verify_private_ecdsa_public_polynomials,
    },
    sha_ecdsa::{
        self, ShaEcdsaHonestShaBindingError, ShaEcdsaTraceBuildError, ShaEcdsaUair,
        build_trace_from_message_and_signature, derive_ecdsa_verification_scalars,
        extract_sha256_output, verify_sha_ecdsa_honest_sha_binding,
    },
    sha256::Sha256Ideal,
};

pub mod cols {
    use crate::{private_ecdsa_scalars, sha_ecdsa};

    pub const NUM_BINARY: usize = sha_ecdsa::cols::NUM_BIN + 1;
    pub const NUM_BINARY_PUBLIC: usize = sha_ecdsa::cols::NUM_BIN_PUB;
    pub const B_AUX: usize = sha_ecdsa::cols::NUM_BIN;

    pub const PA_QX: usize = 0;
    pub const PA_QY: usize = 1;
    pub const PA_QGX: usize = 2;
    pub const PA_QGY: usize = 3;
    pub const PRIVATE_PA_START: usize = 4;
    pub const PRIVATE_SELECTOR_START: usize =
        PRIVATE_PA_START + private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC;
    pub const PA_P_BIT: usize =
        PRIVATE_SELECTOR_START + private_ecdsa_scalars::cols::NUM_INT_PUBLIC;
    pub const PA_P_MINUS_N_BIT: usize = PA_P_BIT + 1;
    pub const NUM_ARBITRARY_PUBLIC: usize = PA_P_MINUS_N_BIT + 1;
    pub const PRIVATE_W_START: usize = NUM_ARBITRARY_PUBLIC;
    pub const NUM_ARBITRARY: usize = PRIVATE_W_START + private_ecdsa_scalars::cols::NUM_ARBITRARY
        - private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC;

    pub const NUM_INT_PUBLIC: usize = sha_ecdsa::cols::ECDSA_S_FINAL + 1;
    pub const W_R_BIT: usize = sha_ecdsa::cols::NUM_INT;
    pub const W_S_BIT: usize = W_R_BIT + 1;
    pub const W_R_LESS: usize = W_S_BIT + 1;
    pub const W_R_SEEN: usize = W_R_LESS + 1;
    pub const W_S_LESS: usize = W_R_SEEN + 1;
    pub const W_S_SEEN: usize = W_S_LESS + 1;
    pub const W_U1_LESS: usize = W_S_SEEN + 1;
    pub const W_U2_LESS: usize = W_U1_LESS + 1;
    pub const W_R_WORD: usize = W_U2_LESS + 1;
    pub const W_S_WORD: usize = W_R_WORD + 1;
    pub const W_U1_WORD: usize = W_S_WORD + 1;
    pub const W_U2_WORD: usize = W_U1_WORD + 1;
    pub const W_AUX: usize = W_U2_WORD + 1;
    pub const W_X_BIT: usize = W_AUX + 1;
    pub const W_X_LESS_P: usize = W_X_BIT + 1;
    pub const W_R_PMN_LESS: usize = W_X_LESS_P + 1;
    pub const W_R_PMN_GREATER: usize = W_R_PMN_LESS + 1;
    pub const W_X_FIELD: usize = W_R_PMN_GREATER + 1;
    pub const W_R_FIELD: usize = W_X_FIELD + 1;
    pub const W_RESULT_BRANCH: usize = W_R_FIELD + 1;
    pub const NUM_INT: usize = W_RESULT_BRANCH + 1;
}

const PRIVATE_ARBITRARY_WITNESS_COUNT: usize =
    private_ecdsa_scalars::cols::NUM_ARBITRARY - private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC;
const PRIVATE_COMPARATOR_AND_WORD_SHIFTS: usize = 10;
// H8 contributes the retained SHA PA_K shift followed by ECDSA X/Y/Z.
const BASE_INT_SHIFTS: usize = 4;
const PRIVATE_X_SHIFT_START: usize = BASE_INT_SHIFTS + PRIVATE_COMPARATOR_AND_WORD_SHIFTS;

#[derive(Clone, Debug)]
pub struct PrivateShaEcdsaUair<R>(PhantomData<R>);

fn flat_arbitrary(column: usize) -> usize {
    cols::NUM_BINARY + column
}

fn flat_int(column: usize) -> usize {
    cols::NUM_BINARY + cols::NUM_ARBITRARY + column
}

fn private_arbitrary_column(column: usize) -> usize {
    if column < private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC {
        cols::PRIVATE_PA_START + column
    } else {
        cols::PRIVATE_W_START + column - private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC
    }
}

fn private_int_column(column: usize) -> usize {
    match column {
        private_ecdsa_scalars::cols::W_R_BIT => cols::W_R_BIT,
        private_ecdsa_scalars::cols::W_S_BIT => cols::W_S_BIT,
        private_ecdsa_scalars::cols::W_U1_BIT => sha_ecdsa::cols::ECDSA_PA_B1,
        private_ecdsa_scalars::cols::W_U2_BIT => sha_ecdsa::cols::ECDSA_PA_B2,
        private_ecdsa_scalars::cols::W_R_LESS => cols::W_R_LESS,
        private_ecdsa_scalars::cols::W_R_SEEN => cols::W_R_SEEN,
        private_ecdsa_scalars::cols::W_S_LESS => cols::W_S_LESS,
        private_ecdsa_scalars::cols::W_S_SEEN => cols::W_S_SEEN,
        private_ecdsa_scalars::cols::W_U1_LESS => cols::W_U1_LESS,
        private_ecdsa_scalars::cols::W_U2_LESS => cols::W_U2_LESS,
        private_ecdsa_scalars::cols::W_R_WORD => cols::W_R_WORD,
        private_ecdsa_scalars::cols::W_S_WORD => cols::W_S_WORD,
        private_ecdsa_scalars::cols::W_U1_WORD => cols::W_U1_WORD,
        private_ecdsa_scalars::cols::W_U2_WORD => cols::W_U2_WORD,
        private_ecdsa_scalars::cols::W_AUX => cols::W_AUX,
        _ if column < private_ecdsa_scalars::cols::NUM_INT_PUBLIC => {
            panic!("private public selectors are represented as arbitrary-polynomial columns")
        }
        _ => panic!("unmapped private ECDSA scalar int column {column}"),
    }
}

impl<R> Uair for PrivateShaEcdsaUair<R>
where
    R: EcdsaFpRing + From<i32> + From<i64>,
{
    type Ideal = Sha256Ideal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let base = ShaEcdsaUair::<R>::signature();
        let old_int_start = sha_ecdsa::cols::NUM_BIN;
        let new_int_start = cols::NUM_BINARY + cols::NUM_ARBITRARY;
        let int_delta = new_int_start - old_int_start;
        let mut shifts = base
            .shifts()
            .iter()
            .map(|shift| {
                let source = if shift.source_col() < old_int_start {
                    shift.source_col()
                } else {
                    shift.source_col() + int_delta
                };
                ShiftSpec::new(source, shift.shift_amount())
            })
            .collect::<Vec<_>>();
        shifts.extend(
            (private_ecdsa_scalars::cols::W_R..=private_ecdsa_scalars::cols::W_C2_POLY)
                .map(|column| ShiftSpec::new(flat_arbitrary(private_arbitrary_column(column)), 1)),
        );
        shifts.extend(
            (private_ecdsa_scalars::cols::W_R_LESS..=private_ecdsa_scalars::cols::W_U2_WORD)
                .map(|column| ShiftSpec::new(flat_int(private_int_column(column)), 1)),
        );
        shifts.extend(
            [
                cols::W_X_LESS_P,
                cols::W_R_PMN_LESS,
                cols::W_R_PMN_GREATER,
                cols::W_X_FIELD,
                cols::W_R_FIELD,
            ]
            .map(|column| ShiftSpec::new(flat_int(column), 1)),
        );

        let mut bit_ops = base.bit_op_specs().to_vec();
        bit_ops.push(BitOpSpec::new(
            cols::B_AUX,
            BitOp::shift_r(private_ecdsa_scalars::LIMB_BITS as u32),
        ));
        let mut int_bits = base.int_witness_bit_cols().to_vec();
        int_bits.extend([
            sha_ecdsa::cols::ECDSA_PA_B1,
            sha_ecdsa::cols::ECDSA_PA_B2,
            cols::W_R_BIT,
            cols::W_S_BIT,
            cols::W_X_BIT,
            cols::W_RESULT_BRANCH,
        ]);

        UairSignature::new(
            TotalColumnLayout::new(cols::NUM_BINARY, cols::NUM_ARBITRARY, cols::NUM_INT),
            PublicColumnLayout::new(
                cols::NUM_BINARY_PUBLIC,
                cols::NUM_ARBITRARY_PUBLIC,
                cols::NUM_INT_PUBLIC,
            ),
            shifts,
            base.lookup_specs().to_vec(),
            bit_ops,
        )
        .with_booleanity_skip_indices(base.booleanity_skip_indices().to_vec())
        .with_int_witness_bit_cols(int_bits)
        .with_shifted_bit_slice_specs(base.shifted_bit_slice_specs().to_vec())
        .with_virtual_booleanity_cols(base.virtual_booleanity_cols().to_vec(), 32)
        .with_virtual_binary_poly_cols(base.virtual_binary_poly_cols().to_vec())
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
        <ShaEcdsaUair<R> as Uair>::constrain_general(
            b,
            up.clone(),
            down.clone(),
            &from_ref,
            &mbs,
            &ideal_from_ref,
        );

        let one = from_ref(&DensePolynomial::new([R::ONE]));
        let two = from_ref(&DensePolynomial::new([R::from(2_u32)]));
        let int = up.int;
        let arbitrary = up.arbitrary_poly;
        let s_init = &int[sha_ecdsa::cols::ECDSA_S_INIT];
        let s_active = &int[sha_ecdsa::cols::ECDSA_S_ACTIVE];
        let s_final = &int[sha_ecdsa::cols::ECDSA_S_FINAL];
        let s_add = &int[sha_ecdsa::cols::ECDSA_S_ADD];
        let b1 = &int[sha_ecdsa::cols::ECDSA_PA_B1];
        let b2 = &int[sha_ecdsa::cols::ECDSA_PA_B2];
        let b1b2 = &int[sha_ecdsa::cols::ECDSA_PA_B1B2];

        b.assert_zero(b1b2.clone() - &(b1.clone() * b2));
        b.assert_zero(s_add.clone() - b1 - b2 + b1b2);
        for (witness, public) in [
            (sha_ecdsa::cols::ECDSA_PA_QX, cols::PA_QX),
            (sha_ecdsa::cols::ECDSA_PA_QY, cols::PA_QY),
            (sha_ecdsa::cols::ECDSA_PA_QGX, cols::PA_QGX),
            (sha_ecdsa::cols::ECDSA_PA_QGY, cols::PA_QGY),
        ] {
            b.assert_zero(int[witness].clone() - &(s_active.clone() * &arbitrary[public]));
        }
        b.assert_zero(int[sha_ecdsa::cols::ECDSA_PA_R_INIT_X].clone());
        b.assert_zero(int[sha_ecdsa::cols::ECDSA_PA_R_INIT_Y].clone() - s_init);
        b.assert_zero(int[sha_ecdsa::cols::ECDSA_PA_R_INIT_Z].clone());
        let not_final = one.clone() - s_final;
        b.assert_zero(not_final.clone() * &int[sha_ecdsa::cols::ECDSA_PA_Z_INV]);
        b.assert_zero(not_final.clone() * &int[sha_ecdsa::cols::ECDSA_PA_R_X]);
        let not_active = one.clone() - s_active;
        b.assert_zero(not_active.clone() * &int[sha_ecdsa::cols::ECDSA_PA_T_X]);
        b.assert_zero(not_active * &int[sha_ecdsa::cols::ECDSA_PA_T_Y]);

        let candidate_binary = [up.binary_poly[cols::B_AUX].clone()];
        let candidate_arbitrary = (0..private_ecdsa_scalars::cols::NUM_ARBITRARY)
            .map(|column| arbitrary[private_arbitrary_column(column)].clone())
            .collect::<Vec<_>>();
        let mut candidate_int = (0..private_ecdsa_scalars::cols::NUM_INT_PUBLIC)
            .map(|column| arbitrary[cols::PRIVATE_SELECTOR_START + column].clone())
            .collect::<Vec<_>>();
        candidate_int.extend(
            (private_ecdsa_scalars::cols::W_R_BIT..private_ecdsa_scalars::cols::NUM_INT)
                .map(|column| int[private_int_column(column)].clone()),
        );
        let candidate_down_arbitrary =
            down.arbitrary_poly[..PRIVATE_ARBITRARY_WITNESS_COUNT].to_vec();
        let candidate_down_int = down.int
            [BASE_INT_SHIFTS..BASE_INT_SHIFTS + PRIVATE_COMPARATOR_AND_WORD_SHIFTS]
            .to_vec();
        let candidate_bit_op = [down.bit_op[down.bit_op.len() - 1].clone()];
        let candidate_up = TraceRow {
            binary_poly: &candidate_binary,
            arbitrary_poly: &candidate_arbitrary,
            int: &candidate_int,
            bit_op: &[],
        };
        let candidate_down = TraceRow {
            binary_poly: &[],
            arbitrary_poly: &candidate_down_arbitrary,
            int: &candidate_down_int,
            bit_op: &candidate_bit_op,
        };
        <PrivateEcdsaScalarsUair<R> as Uair>::constrain_general(
            b,
            candidate_up,
            candidate_down,
            &from_ref,
            &mbs,
            |_: &DegreeOneIdeal<R>| {
                ideal_from_ref(&Sha256Ideal::RotX2(RotationIdeal::new(R::from(2_u32))))
            },
        );

        let scalar_active =
            &arbitrary[cols::PRIVATE_SELECTOR_START + private_ecdsa_scalars::cols::S_SCALAR_ACTIVE];
        let scalar_init =
            &arbitrary[cols::PRIVATE_SELECTOR_START + private_ecdsa_scalars::cols::S_INIT];
        let scalar_final =
            &arbitrary[cols::PRIVATE_SELECTOR_START + private_ecdsa_scalars::cols::S_SCALAR_FINAL];
        b.assert_zero(s_init.clone() - scalar_init);
        b.assert_zero(s_active.clone() - scalar_active);
        b.assert_zero(s_final.clone() - scalar_final);

        let x_bit = &int[cols::W_X_BIT];
        let r_bit = &int[cols::W_R_BIT];
        let x_less = &int[cols::W_X_LESS_P];
        let r_less = &int[cols::W_R_PMN_LESS];
        let r_greater = &int[cols::W_R_PMN_GREATER];
        let x_field = &int[cols::W_X_FIELD];
        let r_field = &int[cols::W_R_FIELD];
        let branch = &int[cols::W_RESULT_BRANCH];
        let p_bit = &arbitrary[cols::PA_P_BIT];
        let pmn_bit = &arbitrary[cols::PA_P_MINUS_N_BIT];
        let scalar_inactive = one.clone() - scalar_active;
        let scalar_padding = scalar_inactive.clone() - scalar_final;

        let next_x_less = &down.int[PRIVATE_X_SHIFT_START];
        let next_r_less = &down.int[PRIVATE_X_SHIFT_START + 1];
        let next_r_greater = &down.int[PRIVATE_X_SHIFT_START + 2];
        let next_x_field = &down.int[PRIVATE_X_SHIFT_START + 3];
        let next_r_field = &down.int[PRIVATE_X_SHIFT_START + 4];

        let x_not_less = one.clone() - x_less;
        let x_less_update =
            x_less.clone() + &(p_bit.clone() * &x_not_less * &(one.clone() - x_bit));
        b.assert_zero(scalar_active.clone() * &(next_x_less.clone() - &x_less_update));
        b.assert_zero(scalar_active.clone() * &((one.clone() - p_bit) * &x_not_less * x_bit));
        b.assert_zero(scalar_init.clone() * x_less);
        b.assert_zero(scalar_final.clone() * &(x_less.clone() - &one));
        b.assert_zero(scalar_padding.clone() * x_less);

        let undecided = one.clone() - r_less - r_greater;
        let r_less_update =
            r_less.clone() + &(pmn_bit.clone() * &undecided * &(one.clone() - r_bit));
        let r_greater_update = r_greater.clone() + &((one.clone() - pmn_bit) * &undecided * r_bit);
        b.assert_zero(scalar_active.clone() * &(next_r_less.clone() - &r_less_update));
        b.assert_zero(scalar_active.clone() * &(next_r_greater.clone() - &r_greater_update));
        b.assert_zero(scalar_init.clone() * r_less);
        b.assert_zero(scalar_init.clone() * r_greater);
        b.assert_zero(scalar_padding.clone() * r_less);
        b.assert_zero(scalar_padding.clone() * r_greater);

        b.assert_zero(
            scalar_active.clone() * &(next_x_field.clone() - &(x_field.clone() * &two) - x_bit),
        );
        b.assert_zero(
            scalar_active.clone() * &(next_r_field.clone() - &(r_field.clone() * &two) - r_bit),
        );
        b.assert_zero(scalar_init.clone() * x_field);
        b.assert_zero(scalar_init.clone() * r_field);
        b.assert_zero(scalar_padding.clone() * x_field);
        b.assert_zero(scalar_padding.clone() * r_field);
        b.assert_zero(scalar_padding * branch);

        let n_centered = centered_int(ecdsa::SECP256K1_N_UINT);
        let n_scalar = DensePolynomial::new([R::from(n_centered)]);
        let branch_n = mbs(branch, &n_scalar).expect("branch * n must stay constant");
        b.assert_zero(scalar_final.clone() * &(branch.clone() * &(one.clone() - r_less)));
        b.assert_zero(scalar_final.clone() * &(x_field.clone() - r_field - &branch_n));
        b.assert_zero(
            scalar_final.clone() * &(x_field.clone() - &int[sha_ecdsa::cols::ECDSA_PA_R_X]),
        );
    }

    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + Zero + One + PartialEq,
    {
        for (family, expected, actual) in [
            (
                "binary_poly",
                cols::NUM_BINARY_PUBLIC,
                public_trace.binary_poly.len(),
            ),
            (
                "arbitrary_poly",
                cols::NUM_ARBITRARY_PUBLIC,
                public_trace.arbitrary_poly.len(),
            ),
            ("int", cols::NUM_INT_PUBLIC, public_trace.int.len()),
        ] {
            if expected != actual {
                return Err(PublicStructureError::WrongColumnCount {
                    column_family: family,
                    expected,
                    actual,
                });
            }
        }
        let rows = 1usize << num_vars;
        for (column_index, column) in public_trace.binary_poly.iter().enumerate() {
            if column.num_vars != num_vars || column.evaluations.len() != rows {
                return Err(PublicStructureError::WrongColumnShape {
                    column_family: "binary_poly",
                    column_index,
                    expected_num_vars: num_vars,
                    actual_num_vars: column.num_vars,
                    expected_rows: rows,
                    actual_rows: column.evaluations.len(),
                });
            }
        }
        for (column_index, column) in public_trace.arbitrary_poly.iter().enumerate() {
            if column.num_vars != num_vars || column.evaluations.len() != rows {
                return Err(PublicStructureError::WrongColumnShape {
                    column_family: "arbitrary_poly",
                    column_index,
                    expected_num_vars: num_vars,
                    actual_num_vars: column.num_vars,
                    expected_rows: rows,
                    actual_rows: column.evaluations.len(),
                });
            }
        }
        for (column_index, column) in public_trace.int.iter().enumerate() {
            if column.num_vars != num_vars || column.evaluations.len() != rows {
                return Err(PublicStructureError::WrongColumnShape {
                    column_family: "int",
                    column_index,
                    expected_num_vars: num_vars,
                    actual_num_vars: column.num_vars,
                    expected_rows: rows,
                    actual_rows: column.evaluations.len(),
                });
            }
        }
        if rows <= ecdsa::FINAL_ROW {
            return Err(PublicStructureError::WrongValue {
                column: "PRIVATE_SHA_ECDSA_ROWS",
                row: rows,
            });
        }
        let zero = IntT::zero();
        let one = IntT::one();
        for row in 0..rows {
            for (column, expected, name) in [
                (
                    sha_ecdsa::cols::ECDSA_S_INIT,
                    if row == 0 { &one } else { &zero },
                    "ECDSA_S_INIT",
                ),
                (
                    sha_ecdsa::cols::ECDSA_S_ACTIVE,
                    if row < ecdsa::NUM_SHAMIR_ROUNDS {
                        &one
                    } else {
                        &zero
                    },
                    "ECDSA_S_ACTIVE",
                ),
                (
                    sha_ecdsa::cols::ECDSA_S_FINAL,
                    if row == ecdsa::FINAL_ROW { &one } else { &zero },
                    "ECDSA_S_FINAL",
                ),
            ] {
                if &public_trace.int[column][row] != expected {
                    return Err(PublicStructureError::WrongValue { column: name, row });
                }
            }
        }
        Ok(())
    }
}

fn zero_poly() -> DensePolynomial<Int<EC_FP_INT_LIMBS>, 32> {
    DensePolynomial::new([Int::ZERO; 32])
}

fn const_poly(value: Int<EC_FP_INT_LIMBS>) -> DensePolynomial<Int<EC_FP_INT_LIMBS>, 32> {
    let mut coeffs = [Int::ZERO; 32];
    coeffs[0] = value;
    DensePolynomial::new(coeffs)
}

fn centered_int(value: CbUint<EC_FP_INT_LIMBS>) -> Int<EC_FP_INT_LIMBS> {
    if value <= SECP256K1_P_HALF_UINT {
        Int::new(*value.as_int())
    } else {
        Int::new(*value.wrapping_sub(&SECP256K1_P_UINT).as_int())
    }
}

fn bit_msb(value: &CbUint<EC_FP_INT_LIMBS>, row: usize) -> bool {
    let bit = 255 - row;
    ((value.as_words()[bit / 64] >> (bit % 64)) & 1) == 1
}

fn uint_to_be_bytes(value: &CbUint<EC_FP_INT_LIMBS>) -> [u8; 32] {
    let mut bytes = [0_u8; 32];
    for (index, word) in value.as_words().iter().rev().enumerate() {
        bytes[index * 8..(index + 1) * 8].copy_from_slice(&word.to_be_bytes());
    }
    bytes
}

fn append_int_column(
    columns: &mut Vec<DenseMultilinearExtension<Int<EC_FP_INT_LIMBS>>>,
    num_vars: usize,
    values: Vec<Int<EC_FP_INT_LIMBS>>,
) {
    columns.push(DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        values,
        Int::ZERO,
    ));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrivateShaEcdsaTraceBuildError {
    Base(ShaEcdsaTraceBuildError),
    Result(EcdsaResultBindingError),
}

impl fmt::Display for PrivateShaEcdsaTraceBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base(error) => write!(formatter, "base H8 trace failed: {error}"),
            Self::Result(error) => write!(formatter, "private result binding failed: {error}"),
        }
    }
}

impl std::error::Error for PrivateShaEcdsaTraceBuildError {}

pub fn build_private_signature_trace(
    num_vars: usize,
    message: &[u8],
    q: (CbUint<EC_FP_INT_LIMBS>, CbUint<EC_FP_INT_LIMBS>),
    signature_r_be: &[u8],
    signature_s_be: &[u8],
) -> Result<
    UairTrace<'static, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
    PrivateShaEcdsaTraceBuildError,
> {
    let base = build_trace_from_message_and_signature::<Int<EC_FP_INT_LIMBS>>(
        num_vars,
        message,
        q,
        signature_r_be,
        signature_s_be,
    )
    .map_err(PrivateShaEcdsaTraceBuildError::Base)?;
    let base_public = base.public(&ShaEcdsaUair::<Int<EC_FP_INT_LIMBS>>::signature());
    let digest = extract_sha256_output(&base_public)
        .map_err(ShaEcdsaTraceBuildError::Scalar)
        .map_err(PrivateShaEcdsaTraceBuildError::Base)?;
    let derived = derive_ecdsa_verification_scalars(digest, signature_r_be, signature_s_be)
        .map_err(ShaEcdsaTraceBuildError::Scalar)
        .map_err(PrivateShaEcdsaTraceBuildError::Base)?;
    let r = CbUint::from_be_slice(signature_r_be);
    let s = CbUint::from_be_slice(signature_s_be);
    let private = derive_private_ecdsa_scalars(derived.message_representative, r, s);
    debug_assert_eq!(private.u1, derived.u1);
    debug_assert_eq!(private.u2, derived.u2);
    let private_trace =
        build_private_ecdsa_scalar_trace::<Int<EC_FP_INT_LIMBS>>(num_vars, &private);
    let result_cell = &base.int[sha_ecdsa::cols::ECDSA_PA_R_X][ecdsa::FINAL_ROW];
    let result_branch = ecdsa::verify_ecdsa_result_binding(signature_r_be, result_cell)
        .map_err(PrivateShaEcdsaTraceBuildError::Result)?;
    let final_x = ecdsa::decode_canonical_final_x(result_cell)
        .map_err(PrivateShaEcdsaTraceBuildError::Result)?;
    let bound_key =
        ecdsa::validate_secp256k1_public_key(&uint_to_be_bytes(&q.0), &uint_to_be_bytes(&q.1))
            .map_err(ShaEcdsaTraceBuildError::PublicKey)
            .map_err(PrivateShaEcdsaTraceBuildError::Base)?;

    let rows = 1usize << num_vars;
    let mut binary = base.binary_poly.into_owned();
    binary.push(private_trace.binary_poly[private_ecdsa_scalars::cols::B_AUX].clone());

    let mut arbitrary = Vec::with_capacity(cols::NUM_ARBITRARY);
    for value in [
        bound_key.q.0,
        bound_key.q.1,
        bound_key.g_plus_q.0,
        bound_key.g_plus_q.1,
    ] {
        arbitrary.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![const_poly(centered_int(value)); rows],
            zero_poly(),
        ));
    }
    arbitrary.extend(
        private_trace.arbitrary_poly[..private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC]
            .iter()
            .cloned(),
    );
    for column in 0..private_ecdsa_scalars::cols::NUM_INT_PUBLIC {
        arbitrary.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            private_trace.int[column]
                .iter()
                .cloned()
                .map(const_poly)
                .collect(),
            zero_poly(),
        ));
    }
    let p_minus_n = SECP256K1_P_UINT.wrapping_sub(&ecdsa::SECP256K1_N_UINT);
    for bound in [SECP256K1_P_UINT, p_minus_n] {
        arbitrary.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            (0..rows)
                .map(|row| const_poly(Int::from(u32::from(row < 256 && bit_msb(&bound, row)))))
                .collect(),
            zero_poly(),
        ));
    }
    arbitrary.extend(
        private_trace.arbitrary_poly[private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC..]
            .iter()
            .cloned(),
    );

    let mut int = base.int.into_owned();
    for column in [
        private_ecdsa_scalars::cols::W_R_BIT,
        private_ecdsa_scalars::cols::W_S_BIT,
    ]
    .into_iter()
    .chain(private_ecdsa_scalars::cols::W_R_LESS..private_ecdsa_scalars::cols::NUM_INT)
    {
        int.push(private_trace.int[column].clone());
    }

    let mut x_bit_values = vec![Int::ZERO; rows];
    let mut x_less_values = vec![Int::ZERO; rows];
    let mut r_less_values = vec![Int::ZERO; rows];
    let mut r_greater_values = vec![Int::ZERO; rows];
    let mut x_field_values = vec![Int::ZERO; rows];
    let mut r_field_values = vec![Int::ZERO; rows];
    let mut branch_values = vec![Int::ZERO; rows];
    let mut x_less = false;
    let mut r_less = false;
    let mut r_greater = false;
    let mut x_prefix = CbUint::<EC_FP_INT_LIMBS>::ZERO;
    let mut r_prefix = CbUint::<EC_FP_INT_LIMBS>::ZERO;
    for row in 0..=256 {
        x_less_values[row] = Int::from(u32::from(x_less));
        r_less_values[row] = Int::from(u32::from(r_less));
        r_greater_values[row] = Int::from(u32::from(r_greater));
        x_field_values[row] = centered_int(x_prefix);
        r_field_values[row] = centered_int(r_prefix);
        if row == 256 {
            break;
        }
        let x_bit = bit_msb(&final_x, row);
        let r_bit = bit_msb(&r, row);
        x_bit_values[row] = Int::from(u32::from(x_bit));
        if !x_less {
            let p_bit = bit_msb(&SECP256K1_P_UINT, row);
            x_less = p_bit && !x_bit;
        }
        if !r_less && !r_greater {
            let bound_bit = bit_msb(&p_minus_n, row);
            r_less = bound_bit && !r_bit;
            r_greater = !bound_bit && r_bit;
        }
        x_prefix = x_prefix.wrapping_shl(1);
        r_prefix = r_prefix.wrapping_shl(1);
        if x_bit {
            x_prefix = x_prefix.wrapping_add(&CbUint::ONE);
        }
        if r_bit {
            r_prefix = r_prefix.wrapping_add(&CbUint::ONE);
        }
    }
    branch_values[ecdsa::FINAL_ROW] = Int::from(u32::from(matches!(
        result_branch,
        EcdsaResultBranch::PlusOrder
    )));
    for values in [
        x_bit_values,
        x_less_values,
        r_less_values,
        r_greater_values,
        x_field_values,
        r_field_values,
        branch_values,
    ] {
        append_int_column(&mut int, num_vars, values);
    }

    debug_assert_eq!(binary.len(), cols::NUM_BINARY);
    debug_assert_eq!(arbitrary.len(), cols::NUM_ARBITRARY);
    debug_assert_eq!(int.len(), cols::NUM_INT);
    Ok(UairTrace {
        binary_poly: binary.into(),
        arbitrary_poly: arbitrary.into(),
        int: int.into(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivateShaEcdsaApplicationBinding {
    pub digest: [u8; 32],
    pub public_key: EcdsaBoundPublicKey,
}

#[derive(Clone, Debug)]
pub enum PrivateShaEcdsaBindingError {
    HonestSha(ShaEcdsaHonestShaBindingError),
    PublicKey(ecdsa::EcdsaPublicKeyBindingError),
    PublicPolynomial(PublicStructureError),
}

fn exact_constant_cell(
    cell: &DensePolynomial<Int<EC_FP_INT_LIMBS>, 32>,
    expected: Int<EC_FP_INT_LIMBS>,
    column: &'static str,
    row: usize,
) -> Result<(), PublicStructureError> {
    if cell.coeffs[0] != expected || cell.coeffs[1..].iter().any(|value| *value != Int::ZERO) {
        return Err(PublicStructureError::WrongValue { column, row });
    }
    Ok(())
}

pub fn verify_private_signature_application_binding(
    message: &[u8],
    q_x_be: &[u8],
    q_y_be: &[u8],
    public_trace: &UairTrace<'_, Int<EC_FP_INT_LIMBS>, Int<EC_FP_INT_LIMBS>, 32>,
) -> Result<PrivateShaEcdsaApplicationBinding, PrivateShaEcdsaBindingError> {
    let num_vars = public_trace
        .binary_poly
        .first()
        .ok_or_else(|| {
            PrivateShaEcdsaBindingError::PublicPolynomial(PublicStructureError::WrongColumnCount {
                column_family: "binary_poly",
                expected: cols::NUM_BINARY_PUBLIC,
                actual: 0,
            })
        })?
        .num_vars;
    <PrivateShaEcdsaUair<Int<EC_FP_INT_LIMBS>> as Uair>::verify_public_structure(
        public_trace,
        num_vars,
    )
    .map_err(PrivateShaEcdsaBindingError::PublicPolynomial)?;
    let digest = verify_sha_ecdsa_honest_sha_binding(message, public_trace)
        .map_err(PrivateShaEcdsaBindingError::HonestSha)?;
    let public_key = ecdsa::validate_secp256k1_public_key(q_x_be, q_y_be)
        .map_err(PrivateShaEcdsaBindingError::PublicKey)?;
    let rows = public_trace.binary_poly[0].evaluations.len();
    for (column, expected, name) in [
        (cols::PA_QX, public_key.q.0, "PRIVATE_PA_QX"),
        (cols::PA_QY, public_key.q.1, "PRIVATE_PA_QY"),
        (cols::PA_QGX, public_key.g_plus_q.0, "PRIVATE_PA_QGX"),
        (cols::PA_QGY, public_key.g_plus_q.1, "PRIVATE_PA_QGY"),
    ] {
        for row in 0..rows {
            exact_constant_cell(
                &public_trace.arbitrary_poly[column][row],
                centered_int(expected),
                name,
                row,
            )
            .map_err(PrivateShaEcdsaBindingError::PublicPolynomial)?;
        }
    }

    let mut scalar_int = Vec::with_capacity(private_ecdsa_scalars::cols::NUM_INT_PUBLIC);
    for column in 0..private_ecdsa_scalars::cols::NUM_INT_PUBLIC {
        let source = &public_trace.arbitrary_poly[cols::PRIVATE_SELECTOR_START + column];
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            let cell = &source[row];
            if cell.coeffs[1..].iter().any(|value| *value != Int::ZERO) {
                return Err(PrivateShaEcdsaBindingError::PublicPolynomial(
                    PublicStructureError::WrongValue {
                        column: "PRIVATE_SELECTOR",
                        row,
                    },
                ));
            }
            values.push(cell.coeffs[0]);
        }
        scalar_int.push(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            values,
            Int::ZERO,
        ));
    }
    let scalar_public = UairTrace {
        binary_poly: Default::default(),
        arbitrary_poly: public_trace.arbitrary_poly[cols::PRIVATE_PA_START
            ..cols::PRIVATE_PA_START + private_ecdsa_scalars::cols::NUM_ARBITRARY_PUBLIC]
            .to_vec()
            .into(),
        int: scalar_int.into(),
    };
    let mut e = CbUint::<EC_FP_INT_LIMBS>::from_be_slice(&digest);
    if e >= ecdsa::SECP256K1_N_UINT {
        e = e.wrapping_sub(&ecdsa::SECP256K1_N_UINT);
    }
    verify_private_ecdsa_public_polynomials(&scalar_public, num_vars, &e)
        .map_err(PrivateShaEcdsaBindingError::PublicPolynomial)?;
    let p_minus_n = SECP256K1_P_UINT.wrapping_sub(&ecdsa::SECP256K1_N_UINT);
    for (column, bound, name) in [
        (cols::PA_P_BIT, SECP256K1_P_UINT, "PRIVATE_PA_P_BIT"),
        (
            cols::PA_P_MINUS_N_BIT,
            p_minus_n,
            "PRIVATE_PA_P_MINUS_N_BIT",
        ),
    ] {
        for row in 0..rows {
            exact_constant_cell(
                &public_trace.arbitrary_poly[column][row],
                Int::from(u32::from(row < 256 && bit_msb(&bound, row))),
                name,
                row,
            )
            .map_err(PrivateShaEcdsaBindingError::PublicPolynomial)?;
        }
    }
    Ok(PrivateShaEcdsaApplicationBinding { digest, public_key })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zinc_uair::{constraint_counter::count_constraints, degree_counter::count_max_degree};

    const QX: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    const QY: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A");
    const R: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302");
    const S: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568");

    fn message() -> Vec<u8> {
        let prefix = b"zinc-plus-lab:h8:honest-sha-ecdsa:v1\n";
        let mut message = prefix.to_vec();
        message.extend(
            (0_u16..)
                .map(|counter| counter as u8)
                .take(crate::sha256::SEVEN_BLOCK_MESSAGE_BYTES - prefix.len()),
        );
        message
    }

    #[test]
    fn shape_and_typed_statement_binding() {
        type U = PrivateShaEcdsaUair<Int<EC_FP_INT_LIMBS>>;
        let message = message();
        let r = uint_to_be_bytes(&R);
        let s = uint_to_be_bytes(&S);
        let trace = build_private_signature_trace(9, &message, (QX, QY), &r, &s)
            .expect("private H8 fixture must build");
        assert_eq!(trace.binary_poly.len(), cols::NUM_BINARY);
        assert_eq!(trace.arbitrary_poly.len(), cols::NUM_ARBITRARY);
        assert_eq!(trace.int.len(), cols::NUM_INT);
        let base_signature = ShaEcdsaUair::<Int<EC_FP_INT_LIMBS>>::signature();
        let base_int_shift_count = base_signature
            .shifts()
            .iter()
            .filter(|shift| shift.source_col() >= sha_ecdsa::cols::NUM_BIN)
            .count();
        assert_eq!(base_int_shift_count, BASE_INT_SHIFTS);
        let constraints = count_constraints::<U>();
        let max_degree = count_max_degree::<U>();
        println!(
            "PRIVATE_SIGNATURE_H8_SHAPE constraints={} max_degree={} total_binary={} public_binary={} total_arbitrary={} public_arbitrary={} total_int={} public_int={}",
            constraints,
            max_degree,
            cols::NUM_BINARY,
            cols::NUM_BINARY_PUBLIC,
            cols::NUM_ARBITRARY,
            cols::NUM_ARBITRARY_PUBLIC,
            cols::NUM_INT,
            cols::NUM_INT_PUBLIC,
        );
        assert_eq!(constraints, 118);
        assert_eq!(max_degree, 5);
        assert_eq!(cols::NUM_BINARY, 21);
        assert_eq!(cols::NUM_BINARY_PUBLIC, 9);
        assert_eq!(cols::NUM_ARBITRARY, 30);
        assert_eq!(cols::NUM_ARBITRARY_PUBLIC, 22);
        assert_eq!(cols::NUM_INT, 62);
        assert_eq!(cols::NUM_INT_PUBLIC, 9);
        let public = trace.public(&U::signature());
        verify_private_signature_application_binding(
            &message,
            &uint_to_be_bytes(&QX),
            &uint_to_be_bytes(&QY),
            &public,
        )
        .expect("typed private statement contract must accept the fixture");

        let mut changed_digest_link = public.clone();
        changed_digest_link.arbitrary_poly.to_mut()
            [cols::PRIVATE_PA_START + private_ecdsa_scalars::cols::PA_E]
            .evaluations[private_ecdsa_scalars::IDENTITY_FINAL_ROW]
            .coeffs[0] += Int::from(1_u32);
        assert!(
            verify_private_signature_application_binding(
                &message,
                &uint_to_be_bytes(&QX),
                &uint_to_be_bytes(&QY),
                &changed_digest_link,
            )
            .is_err(),
            "the typed boundary must bind the SHA digest to the scalar identity",
        );

        let mut changed_message = message.clone();
        changed_message[0] ^= 1;
        assert!(
            verify_private_signature_application_binding(
                &changed_message,
                &uint_to_be_bytes(&QX),
                &uint_to_be_bytes(&QY),
                &public,
            )
            .is_err(),
            "the typed boundary must bind the message",
        );

        let mut changed_qx = uint_to_be_bytes(&QX);
        changed_qx[31] ^= 1;
        assert!(
            verify_private_signature_application_binding(
                &message,
                &changed_qx,
                &uint_to_be_bytes(&QY),
                &public,
            )
            .is_err(),
            "the typed boundary must bind the public key",
        );

        let mut malformed_public = public.clone();
        malformed_public.binary_poly = Default::default();
        assert!(matches!(
            verify_private_signature_application_binding(
                &message,
                &uint_to_be_bytes(&QX),
                &uint_to_be_bytes(&QY),
                &malformed_public,
            ),
            Err(PrivateShaEcdsaBindingError::PublicPolynomial(
                PublicStructureError::WrongColumnCount {
                    column_family: "binary_poly",
                    expected: cols::NUM_BINARY_PUBLIC,
                    actual: 0,
                }
            ))
        ));

        let mut short_public = public.clone();
        short_public.int.to_mut()[0].evaluations.pop();
        assert!(matches!(
            verify_private_signature_application_binding(
                &message,
                &uint_to_be_bytes(&QX),
                &uint_to_be_bytes(&QY),
                &short_public,
            ),
            Err(PrivateShaEcdsaBindingError::PublicPolynomial(
                PublicStructureError::WrongColumnShape {
                    column_family: "int",
                    column_index: 0,
                    expected_num_vars: 9,
                    actual_num_vars: 9,
                    expected_rows: 512,
                    actual_rows: 511,
                }
            ))
        ));
    }
}
