//! Proof-enforced private ECDSA scalar derivation over the secp256k1 order.
//!
//! The construction uses 26-bit base-`2^26` limbs. Four private bit streams
//! (`r`, `s`, `u1`, and `u2`) are consumed by row-local Horner states, while
//! arbitrary-polynomial accumulators build their 10-limb representations.
//! Two quotient polynomials and two signed carry polynomials then enforce the
//! exact integer identities
//!
//! ```text
//! s * u1 = e + q1 * n
//! s * u2 = r + q2 * n
//! ```
//!
//! coefficient by coefficient. Binary-polynomial columns range the quotient
//! limbs and offset carries; prefix comparators enforce canonical scalar
//! representatives. No host-side range assertion is part of the argument.

use core::{array, marker::PhantomData};

use crypto_bigint::{NonZero, Odd, Uint as CbUint};
use crypto_primitives::ConstSemiring;
use num_traits::{One, Zero};
use zinc_poly::{
    mle::DenseMultilinearExtension,
    univariate::{binary::BinaryPoly, dense::DensePolynomial},
};
use zinc_uair::{
    BitOp, BitOpSpec, ConstraintBuilder, PublicColumnLayout, PublicStructureError, ShiftSpec,
    TotalColumnLayout, TraceRow, Uair, UairSignature, UairTrace, ideal::DegreeOneIdeal,
};

use crate::{
    ecdsa::SECP256K1_N_UINT, ecdsa_doubling::EC_FP_INT_LIMBS, private_scalar::SCALAR_BITS,
};

pub const LIMB_BITS: usize = 26;
pub const NUM_LIMBS: usize = (SCALAR_BITS + LIMB_BITS - 1) / LIMB_BITS;
pub const TOP_LIMB_BITS: usize = SCALAR_BITS - (NUM_LIMBS - 1) * LIMB_BITS;
pub const PRODUCT_COEFFS: usize = 2 * NUM_LIMBS - 1;
pub const NUM_CARRIES: usize = PRODUCT_COEFFS - 1;
pub const LIMB_BASE: i64 = 1_i64 << LIMB_BITS;
pub const MAX_CARRY_ABS_BOUND: i64 = NUM_LIMBS as i64 * (LIMB_BASE - 1) + 1;
pub const CARRY_OFFSET: i64 = 1_i64 << 31;
pub const SCALAR_FINAL_ROW: usize = SCALAR_BITS;
pub const AUX_Q1_START: usize = SCALAR_FINAL_ROW + 1;
pub const AUX_Q2_START: usize = AUX_Q1_START + NUM_LIMBS;
pub const AUX_C1_START: usize = AUX_Q2_START + NUM_LIMBS;
pub const AUX_C2_START: usize = AUX_C1_START + NUM_CARRIES;
pub const IDENTITY_FINAL_ROW: usize = AUX_C2_START + NUM_CARRIES;

pub mod cols {
    pub const B_AUX: usize = 0;
    pub const NUM_BINARY: usize = 1;

    pub const PA_LIMB_MONOMIAL: usize = 0;
    pub const PA_CARRY_MONOMIAL: usize = 1;
    pub const PA_N: usize = 2;
    pub const PA_E: usize = 3;
    pub const PA_CARRY_MASK: usize = 4;
    pub const NUM_ARBITRARY_PUBLIC: usize = 5;

    pub const W_R: usize = 5;
    pub const W_S: usize = 6;
    pub const W_U1: usize = 7;
    pub const W_U2: usize = 8;
    pub const W_Q1: usize = 9;
    pub const W_Q2: usize = 10;
    pub const W_C1_POLY: usize = 11;
    pub const W_C2_POLY: usize = 12;
    pub const NUM_ARBITRARY: usize = 13;

    pub const S_SCALAR_ACTIVE: usize = 0;
    pub const S_INIT: usize = 1;
    pub const S_SCALAR_FINAL: usize = 2;
    pub const S_ACC_ACTIVE: usize = 3;
    pub const S_IDENTITY_FINAL: usize = 4;
    pub const S_BLOCK_END: usize = 5;
    pub const N_BIT: usize = 6;
    pub const S_Q1: usize = 7;
    pub const S_Q2: usize = 8;
    pub const S_C1: usize = 9;
    pub const S_C2: usize = 10;
    pub const NUM_INT_PUBLIC: usize = 11;

    pub const W_R_BIT: usize = 11;
    pub const W_S_BIT: usize = 12;
    pub const W_U1_BIT: usize = 13;
    pub const W_U2_BIT: usize = 14;
    pub const W_R_LESS: usize = 15;
    pub const W_R_SEEN: usize = 16;
    pub const W_S_LESS: usize = 17;
    pub const W_S_SEEN: usize = 18;
    pub const W_U1_LESS: usize = 19;
    pub const W_U2_LESS: usize = 20;

    pub const W_R_WORD: usize = 21;
    pub const W_S_WORD: usize = 22;
    pub const W_U1_WORD: usize = 23;
    pub const W_U2_WORD: usize = 24;
    pub const W_AUX: usize = 25;
    pub const NUM_INT: usize = 26;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivateEcdsaScalars {
    pub e: CbUint<EC_FP_INT_LIMBS>,
    pub r: CbUint<EC_FP_INT_LIMBS>,
    pub s: CbUint<EC_FP_INT_LIMBS>,
    pub u1: CbUint<EC_FP_INT_LIMBS>,
    pub u2: CbUint<EC_FP_INT_LIMBS>,
    pub q1: CbUint<EC_FP_INT_LIMBS>,
    pub q2: CbUint<EC_FP_INT_LIMBS>,
    pub c1: [i64; NUM_CARRIES],
    pub c2: [i64; NUM_CARRIES],
}

#[derive(Clone, Debug)]
pub struct PrivateEcdsaScalarsUair<R>(PhantomData<R>);

fn flat_arbitrary(column: usize) -> usize {
    cols::NUM_BINARY + column
}

fn flat_int(column: usize) -> usize {
    cols::NUM_BINARY + cols::NUM_ARBITRARY + column
}

impl<R> Uair for PrivateEcdsaScalarsUair<R>
where
    R: ConstSemiring + From<i32> + From<i64> + 'static,
{
    type Ideal = DegreeOneIdeal<R>;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(cols::NUM_BINARY, cols::NUM_ARBITRARY, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, cols::NUM_ARBITRARY_PUBLIC, cols::NUM_INT_PUBLIC);
        let mut shifts = (cols::W_R..=cols::W_C2_POLY)
            .map(|column| ShiftSpec::new(flat_arbitrary(column), 1))
            .collect::<Vec<_>>();
        shifts.extend(
            (cols::W_R_LESS..=cols::W_U2_WORD).map(|column| ShiftSpec::new(flat_int(column), 1)),
        );
        let bit_ops = vec![BitOpSpec::new(
            cols::B_AUX,
            BitOp::shift_r(LIMB_BITS as u32),
        )];
        UairSignature::new(total, public, shifts, vec![], bit_ops).with_int_witness_bit_cols(vec![
            cols::W_R_BIT,
            cols::W_S_BIT,
            cols::W_U1_BIT,
            cols::W_U2_BIT,
        ])
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
        let one = from_ref(&DensePolynomial::new([R::from(1)]));
        let two = from_ref(&DensePolynomial::new([R::from(2)]));
        let two_ideal = ideal_from_ref(&DegreeOneIdeal::new(R::from(2)));

        let int = up.int;
        let arbitrary = up.arbitrary_poly;
        let binary = up.binary_poly;
        let scalar_active = &int[cols::S_SCALAR_ACTIVE];
        let init = &int[cols::S_INIT];
        let scalar_final = &int[cols::S_SCALAR_FINAL];
        let acc_active = &int[cols::S_ACC_ACTIVE];
        let identity_final = &int[cols::S_IDENTITY_FINAL];
        let block_end = &int[cols::S_BLOCK_END];
        let n_bit = &int[cols::N_BIT];
        let q1_row = &int[cols::S_Q1];
        let q2_row = &int[cols::S_Q2];
        let c1_row = &int[cols::S_C1];
        let c2_row = &int[cols::S_C2];
        let scalar_inactive = one.clone() - scalar_active;
        let scalar_padding = scalar_inactive.clone() - scalar_final;
        let acc_padding = one.clone() - acc_active - identity_final;
        let q_row = q1_row.clone() + q2_row;
        let carry_row = c1_row.clone() + c2_row;
        let aux_active = q_row.clone() + &carry_row;
        let aux_inactive = one.clone() - &aux_active;

        // Independent constant-in-X residuals occupy separate inner
        // coefficients. A zero polynomial forces every lane to vanish, so
        // this is conjunction rather than a scalar linear combination.
        let pack_constant_residuals = |residuals: Vec<B::Expr>| {
            residuals.into_iter().enumerate().fold(
                from_ref(&DensePolynomial::new([R::from(0)])),
                |packed, (coefficient, residual)| {
                    let lane = DensePolynomial {
                        coeffs: array::from_fn(|index| {
                            if index == coefficient {
                                R::from(1)
                            } else {
                                R::from(0)
                            }
                        }),
                    };
                    packed
                        + &mbs(&residual, &lane)
                            .expect("constant residual lane must fit inner degree")
                },
            )
        };

        let scalar_bits = [
            int[cols::W_R_BIT].clone(),
            int[cols::W_S_BIT].clone(),
            int[cols::W_U1_BIT].clone(),
            int[cols::W_U2_BIT].clone(),
        ];
        b.assert_zero(pack_constant_residuals(
            scalar_bits
                .iter()
                .map(|bit| scalar_inactive.clone() * bit)
                .collect(),
        ));

        // Six shifted comparator-state columns occupy down.int[0..6].
        let mut less_updates = Vec::with_capacity(4);
        let mut greater_forbidden = Vec::with_capacity(4);
        let mut less_initial = Vec::with_capacity(4);
        let mut less_final = Vec::with_capacity(4);
        let mut less_padding = Vec::with_capacity(4);
        let mut seen_updates = Vec::with_capacity(2);
        let mut seen_initial = Vec::with_capacity(2);
        let mut seen_final = Vec::with_capacity(2);
        let mut seen_padding = Vec::with_capacity(2);
        for (bit, less_col, seen_col, down_less, down_seen, require_nonzero) in [
            (
                &scalar_bits[0],
                cols::W_R_LESS,
                Some(cols::W_R_SEEN),
                0,
                Some(1),
                true,
            ),
            (
                &scalar_bits[1],
                cols::W_S_LESS,
                Some(cols::W_S_SEEN),
                2,
                Some(3),
                true,
            ),
            (&scalar_bits[2], cols::W_U1_LESS, None, 4, None, false),
            (&scalar_bits[3], cols::W_U2_LESS, None, 5, None, false),
        ] {
            let less = &int[less_col];
            let next_less = &down.int[down_less];
            let not_bit = one.clone() - bit;
            let not_less = one.clone() - less;
            let not_n_bit = one.clone() - n_bit;
            let less_update = less.clone() + &(n_bit.clone() * &not_less * &not_bit);
            less_updates.push(scalar_active.clone() * &(next_less.clone() - &less_update));
            greater_forbidden.push(scalar_active.clone() * &(not_n_bit * &not_less * bit));
            less_initial.push(init.clone() * less);
            less_final.push(scalar_final.clone() * &(less.clone() - &one));
            less_padding.push(scalar_padding.clone() * less);

            if let (Some(seen_col), Some(down_seen)) = (seen_col, down_seen) {
                let seen = &int[seen_col];
                let next_seen = &down.int[down_seen];
                let seen_update = seen.clone() + &((one.clone() - seen) * bit);
                seen_updates.push(scalar_active.clone() * &(next_seen.clone() - &seen_update));
                seen_initial.push(init.clone() * seen);
                seen_padding.push(scalar_padding.clone() * seen);
                if require_nonzero {
                    seen_final.push(scalar_final.clone() * &(seen.clone() - &one));
                }
            }
        }
        for residuals in [
            less_updates,
            greater_forbidden,
            less_initial,
            less_final,
            less_padding,
            seen_updates,
            seen_initial,
            seen_final,
            seen_padding,
        ] {
            b.assert_zero(pack_constant_residuals(residuals));
        }

        // Horner word states consume one scalar bit per row and reset after
        // every radix block. Their four shifted columns are down.int[6..10].
        let mut word_outputs = Vec::with_capacity(4);
        let mut word_updates = Vec::with_capacity(4);
        let mut word_initial = Vec::with_capacity(4);
        let mut word_final = Vec::with_capacity(4);
        let mut word_padding = Vec::with_capacity(4);
        for (bit, word_col, down_word) in [
            (&scalar_bits[0], cols::W_R_WORD, 6),
            (&scalar_bits[1], cols::W_S_WORD, 7),
            (&scalar_bits[2], cols::W_U1_WORD, 8),
            (&scalar_bits[3], cols::W_U2_WORD, 9),
        ] {
            let word = &int[word_col];
            let word_out = word.clone() * &two + bit;
            let reset_or_keep = (one.clone() - block_end) * &word_out;
            word_updates
                .push(scalar_active.clone() * &(down.int[down_word].clone() - &reset_or_keep));
            word_initial.push(init.clone() * word);
            word_final.push(scalar_final.clone() * word);
            word_padding.push(scalar_padding.clone() * word);
            word_outputs.push(word_out);
        }
        for residuals in [word_updates, word_initial, word_final, word_padding] {
            b.assert_zero(pack_constant_residuals(residuals));
        }

        // All quotient and carry limbs share one serialized auxiliary lane.
        // Quotients use 26 bits. Offset carries use the binary column's full
        // 32-bit range, so only quotient rows need a high-zero virtual check.
        let aux = &int[cols::W_AUX];
        b.assert_in_ideal(binary[cols::B_AUX].clone() - aux, &two_ideal);
        b.assert_zero(aux_inactive * aux);
        b.assert_zero(q_row * &down.bit_op[0]);

        // Eight arbitrary-polynomial accumulators are shifted in source order.
        // Each starts at zero, advances through the scalar and auxiliary
        // schedules, and is zero-owned again after the identity row.
        let limb_monomial = &arbitrary[cols::PA_LIMB_MONOMIAL];
        let carry_monomial = &arbitrary[cols::PA_CARRY_MONOMIAL];
        for (acc_col, value, down_acc) in [
            (cols::W_R, word_outputs[0].clone(), 0),
            (cols::W_S, word_outputs[1].clone(), 1),
            (cols::W_U1, word_outputs[2].clone(), 2),
            (cols::W_U2, word_outputs[3].clone(), 3),
            (cols::W_Q1, aux.clone() * q1_row, 4),
            (cols::W_Q2, aux.clone() * q2_row, 5),
        ] {
            let acc = &arbitrary[acc_col];
            let contribution = value * limb_monomial;
            b.assert_zero(
                acc_active.clone() * &(down.arbitrary_poly[down_acc].clone() - acc - &contribution),
            );
            b.assert_zero(init.clone() * acc);
            b.assert_zero(acc_padding.clone() * acc);
        }
        for (acc_col, selector, down_acc) in
            [(cols::W_C1_POLY, c1_row, 6), (cols::W_C2_POLY, c2_row, 7)]
        {
            let acc = &arbitrary[acc_col];
            let contribution = aux.clone() * selector * carry_monomial;
            b.assert_zero(
                acc_active.clone() * &(down.arbitrary_poly[down_acc].clone() - acc - &contribution),
            );
            b.assert_zero(init.clone() * acc);
            b.assert_zero(acc_padding.clone() * acc);
        }

        // Exact coefficient identities. Every coefficient is much smaller
        // than the fixed secp256k1 base field, so projection cannot turn a
        // nonzero bounded integer coefficient into zero.
        let carry_offset_poly = DensePolynomial::new([R::from(CARRY_OFFSET)]);
        let carry_offset_mask = mbs(&arbitrary[cols::PA_CARRY_MASK], &carry_offset_poly)
            .expect("carry offset mask must preserve the carry-polynomial degree");
        let c1 = arbitrary[cols::W_C1_POLY].clone() - &carry_offset_mask;
        let c2 = arbitrary[cols::W_C2_POLY].clone() - &carry_offset_mask;
        let base_minus_x = DensePolynomial::new([R::from(LIMB_BASE as i32), R::from(-1)]);
        let c1_term = mbs(&c1, &base_minus_x).expect("(2^26-X) * c1 must fit the product degree");
        let c2_term = mbs(&c2, &base_minus_x).expect("(2^26-X) * c2 must fit the product degree");
        let first = arbitrary[cols::W_S].clone() * &arbitrary[cols::W_U1]
            - &arbitrary[cols::PA_E]
            - &(arbitrary[cols::W_Q1].clone() * &arbitrary[cols::PA_N])
            + &c1_term;
        let second = arbitrary[cols::W_S].clone() * &arbitrary[cols::W_U2]
            - &arbitrary[cols::W_R]
            - &(arbitrary[cols::W_Q2].clone() * &arbitrary[cols::PA_N])
            + &c2_term;
        b.assert_zero(identity_final.clone() * &first);
        b.assert_zero(identity_final.clone() * &second);
    }

    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + Zero + One + PartialEq,
    {
        verify_public_layout(public_trace, num_vars)
    }
}

fn bit_msb(value: &CbUint<EC_FP_INT_LIMBS>, row: usize) -> bool {
    let bit = SCALAR_BITS - 1 - row;
    let word = bit / 64;
    let offset = bit % 64;
    ((value.as_words()[word] >> offset) & 1) == 1
}

fn scalar_block_degree(row: usize) -> Option<usize> {
    let end = row + 1;
    if end == TOP_LIMB_BITS {
        Some(NUM_LIMBS - 1)
    } else if end > TOP_LIMB_BITS && (end - TOP_LIMB_BITS) % LIMB_BITS == 0 {
        Some(NUM_LIMBS - 1 - (end - TOP_LIMB_BITS) / LIMB_BITS)
    } else {
        None
    }
}

fn radix_limbs(value: &CbUint<EC_FP_INT_LIMBS>) -> [u32; NUM_LIMBS] {
    array::from_fn(|index| {
        let bit = index * LIMB_BITS;
        (value.shr_vartime(bit as u32).as_words()[0] & ((1_u64 << LIMB_BITS) - 1)) as u32
    })
}

fn mul_mod_n(
    left: &CbUint<EC_FP_INT_LIMBS>,
    right: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let product: CbUint<{ EC_FP_INT_LIMBS * 2 }> = left.widening_mul(right).into();
    let order: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
    let order = NonZero::new(order).expect("secp256k1 order is nonzero");
    let (_, remainder) = product.div_rem_vartime(&order);
    remainder.resize()
}

fn exact_quotient(
    left: &CbUint<EC_FP_INT_LIMBS>,
    right: &CbUint<EC_FP_INT_LIMBS>,
    remainder: &CbUint<EC_FP_INT_LIMBS>,
) -> CbUint<EC_FP_INT_LIMBS> {
    let product: CbUint<{ EC_FP_INT_LIMBS * 2 }> = left.widening_mul(right).into();
    let remainder_wide: CbUint<{ EC_FP_INT_LIMBS * 2 }> = remainder.resize();
    assert!(
        product >= remainder_wide,
        "modular remainder exceeds product"
    );
    let numerator = product.wrapping_sub(&remainder_wide);
    let order: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
    let order = NonZero::new(order).expect("secp256k1 order is nonzero");
    let (quotient, exact_remainder) = numerator.div_rem_vartime(&order);
    assert_eq!(
        exact_remainder,
        CbUint::ZERO,
        "modular identity is not exact"
    );
    let quotient: CbUint<EC_FP_INT_LIMBS> = quotient.resize();
    assert!(quotient < SECP256K1_N_UINT, "quotient must be below n");
    quotient
}

fn identity_carries(
    left: &[u32; NUM_LIMBS],
    right: &[u32; NUM_LIMBS],
    remainder: &[u32; NUM_LIMBS],
    quotient: &[u32; NUM_LIMBS],
    modulus: &[u32; NUM_LIMBS],
) -> [i64; NUM_CARRIES] {
    let mut h = [0_i128; PRODUCT_COEFFS];
    for degree in 0..PRODUCT_COEFFS {
        for left_index in 0..NUM_LIMBS {
            if let Some(right_index) = degree.checked_sub(left_index) {
                if right_index < NUM_LIMBS {
                    h[degree] += i128::from(left[left_index]) * i128::from(right[right_index]);
                    h[degree] -=
                        i128::from(quotient[left_index]) * i128::from(modulus[right_index]);
                }
            }
        }
        if degree < NUM_LIMBS {
            h[degree] -= i128::from(remainder[degree]);
        }
    }

    let mut carries = [0_i64; NUM_CARRIES];
    let mut previous = 0_i128;
    for degree in 0..NUM_CARRIES {
        let numerator = previous - h[degree];
        assert_eq!(numerator % i128::from(LIMB_BASE), 0, "non-integral carry");
        let carry = numerator / i128::from(LIMB_BASE);
        assert!(
            carry.unsigned_abs() < CARRY_OFFSET as u128,
            "carry bound exceeded"
        );
        carries[degree] = carry as i64;
        previous = carry;
    }
    assert_eq!(h[PRODUCT_COEFFS - 1], previous, "top coefficient mismatch");
    carries
}

pub fn derive_private_ecdsa_scalars(
    mut e: CbUint<EC_FP_INT_LIMBS>,
    r: CbUint<EC_FP_INT_LIMBS>,
    s: CbUint<EC_FP_INT_LIMBS>,
) -> PrivateEcdsaScalars {
    assert!(
        r > CbUint::ZERO && r < SECP256K1_N_UINT,
        "r must be canonical"
    );
    assert!(
        s > CbUint::ZERO && s < SECP256K1_N_UINT,
        "s must be canonical"
    );
    if e >= SECP256K1_N_UINT {
        e = e.wrapping_sub(&SECP256K1_N_UINT);
    }
    let order = Odd::new(SECP256K1_N_UINT).expect("secp256k1 order is odd");
    let s_inverse = s
        .invert_odd_mod(&order)
        .expect("every canonical nonzero scalar is invertible");
    let u1 = mul_mod_n(&e, &s_inverse);
    let u2 = mul_mod_n(&r, &s_inverse);
    let q1 = exact_quotient(&s, &u1, &e);
    let q2 = exact_quotient(&s, &u2, &r);
    let n_limbs = radix_limbs(&SECP256K1_N_UINT);
    let c1 = identity_carries(
        &radix_limbs(&s),
        &radix_limbs(&u1),
        &radix_limbs(&e),
        &radix_limbs(&q1),
        &n_limbs,
    );
    let c2 = identity_carries(
        &radix_limbs(&s),
        &radix_limbs(&u2),
        &radix_limbs(&r),
        &radix_limbs(&q2),
        &n_limbs,
    );
    PrivateEcdsaScalars {
        e,
        r,
        s,
        u1,
        u2,
        q1,
        q2,
        c1,
        c2,
    }
}

fn zero_poly<R: Clone + Zero>() -> DensePolynomial<R, 32> {
    DensePolynomial {
        coeffs: array::from_fn(|_| R::zero()),
    }
}

fn monomial<R: Clone + Zero + One>(degree: usize) -> DensePolynomial<R, 32> {
    let mut coefficients = array::from_fn(|_| R::zero());
    coefficients[degree] = R::one();
    DensePolynomial {
        coeffs: coefficients,
    }
}

fn limb_poly<R>(value: &CbUint<EC_FP_INT_LIMBS>) -> DensePolynomial<R, 32>
where
    R: Clone + Zero + From<u32>,
{
    let limbs = radix_limbs(value);
    DensePolynomial {
        coeffs: array::from_fn(|index| {
            if index < NUM_LIMBS {
                R::from(limbs[index])
            } else {
                R::zero()
            }
        }),
    }
}

#[cfg(test)]
fn offset_carry_poly<R>(carries: &[i64; NUM_CARRIES]) -> DensePolynomial<R, 32>
where
    R: Clone + Zero + From<i64>,
{
    DensePolynomial {
        coeffs: array::from_fn(|index| {
            if index < NUM_CARRIES {
                R::from(carries[index] + CARRY_OFFSET)
            } else {
                R::zero()
            }
        }),
    }
}

fn fill_scalar_columns<R>(
    int: &mut [Vec<R>],
    value: &CbUint<EC_FP_INT_LIMBS>,
    less_col: usize,
    seen_col: Option<usize>,
    word_col: usize,
) where
    R: Clone + Zero + One + From<u32>,
{
    let mut less = false;
    let mut seen = false;
    let mut word = 0_u32;
    for row in 0..SCALAR_BITS {
        let bit = bit_msb(value, row);
        let n_bit = bit_msb(&SECP256K1_N_UINT, row);
        int[less_col][row] = if less { R::one() } else { R::zero() };
        if let Some(seen_col) = seen_col {
            int[seen_col][row] = if seen { R::one() } else { R::zero() };
        }
        int[word_col][row] = R::from(word);
        word = 2 * word + u32::from(bit);
        if scalar_block_degree(row).is_some() {
            word = 0;
        }
        less |= n_bit && !bit;
        seen |= bit;
    }
    int[less_col][SCALAR_FINAL_ROW] = if less { R::one() } else { R::zero() };
    if let Some(seen_col) = seen_col {
        int[seen_col][SCALAR_FINAL_ROW] = if seen { R::one() } else { R::zero() };
    }
    int[word_col][SCALAR_FINAL_ROW] = R::zero();
}

fn fill_limb_accumulator<R>(column: &mut [DensePolynomial<R, 32>], value: &CbUint<EC_FP_INT_LIMBS>)
where
    R: Clone + Zero + From<u32>,
{
    let limbs = radix_limbs(value);
    let mut current = zero_poly();
    for row in 0..=IDENTITY_FINAL_ROW {
        column[row] = current.clone();
        if let Some(degree) = (row < SCALAR_BITS)
            .then(|| scalar_block_degree(row))
            .flatten()
        {
            current.coeffs[degree] = R::from(limbs[degree]);
        }
    }
}

fn fill_scheduled_limb_accumulator<R>(
    column: &mut [DensePolynomial<R, 32>],
    value: &CbUint<EC_FP_INT_LIMBS>,
    start: usize,
) where
    R: Clone + Zero + From<u32>,
{
    let limbs = radix_limbs(value);
    let mut current = zero_poly();
    for row in 0..=IDENTITY_FINAL_ROW {
        column[row] = current.clone();
        if (start..start + NUM_LIMBS).contains(&row) {
            let degree = row - start;
            current.coeffs[degree] = R::from(limbs[degree]);
        }
    }
}

fn fill_scheduled_offset_carry_accumulator<R>(
    column: &mut [DensePolynomial<R, 32>],
    carries: &[i64; NUM_CARRIES],
    start: usize,
) where
    R: Clone + Zero + From<i64>,
{
    let mut current = zero_poly();
    for row in 0..=IDENTITY_FINAL_ROW {
        column[row] = current.clone();
        if (start..start + NUM_CARRIES).contains(&row) {
            let degree = row - start;
            current.coeffs[degree] = R::from(carries[degree] + CARRY_OFFSET);
        }
    }
}

pub fn build_private_ecdsa_scalar_trace<R>(
    num_vars: usize,
    witness: &PrivateEcdsaScalars,
) -> UairTrace<'static, R, R, 32>
where
    R: Clone + Zero + One + From<u32> + From<i64>,
{
    let rows = 1usize << num_vars;
    assert!(
        rows > IDENTITY_FINAL_ROW,
        "private ECDSA trace does not have enough rows for its auxiliary schedule"
    );
    let mut binary = vec![vec![BinaryPoly::<32>::from(0_u32); rows]; cols::NUM_BINARY];
    let mut arbitrary = vec![vec![zero_poly(); rows]; cols::NUM_ARBITRARY];
    let mut int = vec![vec![R::zero(); rows]; cols::NUM_INT];

    for row in 0..SCALAR_BITS {
        int[cols::S_SCALAR_ACTIVE][row] = R::one();
        int[cols::N_BIT][row] = if bit_msb(&SECP256K1_N_UINT, row) {
            R::one()
        } else {
            R::zero()
        };
        if let Some(degree) = scalar_block_degree(row) {
            int[cols::S_BLOCK_END][row] = R::one();
            arbitrary[cols::PA_LIMB_MONOMIAL][row] = monomial(degree);
        }
    }
    int[cols::S_INIT][0] = R::one();
    int[cols::S_SCALAR_FINAL][SCALAR_FINAL_ROW] = R::one();
    int[cols::S_IDENTITY_FINAL][IDENTITY_FINAL_ROW] = R::one();
    for row in 0..IDENTITY_FINAL_ROW {
        int[cols::S_ACC_ACTIVE][row] = R::one();
    }
    for (selector, start, len) in [
        (cols::S_Q1, AUX_Q1_START, NUM_LIMBS),
        (cols::S_Q2, AUX_Q2_START, NUM_LIMBS),
        (cols::S_C1, AUX_C1_START, NUM_CARRIES),
        (cols::S_C2, AUX_C2_START, NUM_CARRIES),
    ] {
        for row in start..start + len {
            int[selector][row] = R::one();
            let degree = row - start;
            if len == NUM_LIMBS {
                arbitrary[cols::PA_LIMB_MONOMIAL][row] = monomial(degree);
            } else {
                arbitrary[cols::PA_CARRY_MONOMIAL][row] = monomial(degree);
            }
        }
    }
    arbitrary[cols::PA_N][IDENTITY_FINAL_ROW] = limb_poly(&SECP256K1_N_UINT);
    arbitrary[cols::PA_E][IDENTITY_FINAL_ROW] = limb_poly(&witness.e);
    arbitrary[cols::PA_CARRY_MASK][IDENTITY_FINAL_ROW] = DensePolynomial {
        coeffs: array::from_fn(|index| {
            if index < NUM_CARRIES {
                R::one()
            } else {
                R::zero()
            }
        }),
    };

    fill_scalar_columns(
        &mut int,
        &witness.r,
        cols::W_R_LESS,
        Some(cols::W_R_SEEN),
        cols::W_R_WORD,
    );
    fill_scalar_columns(
        &mut int,
        &witness.s,
        cols::W_S_LESS,
        Some(cols::W_S_SEEN),
        cols::W_S_WORD,
    );
    fill_scalar_columns(
        &mut int,
        &witness.u1,
        cols::W_U1_LESS,
        None,
        cols::W_U1_WORD,
    );
    fill_scalar_columns(
        &mut int,
        &witness.u2,
        cols::W_U2_LESS,
        None,
        cols::W_U2_WORD,
    );

    for row in 0..SCALAR_BITS {
        for (column, value) in [
            (cols::W_R_BIT, &witness.r),
            (cols::W_S_BIT, &witness.s),
            (cols::W_U1_BIT, &witness.u1),
            (cols::W_U2_BIT, &witness.u2),
        ] {
            int[column][row] = if bit_msb(value, row) {
                R::one()
            } else {
                R::zero()
            };
        }
    }

    for (start, limbs) in [
        (AUX_Q1_START, radix_limbs(&witness.q1)),
        (AUX_Q2_START, radix_limbs(&witness.q2)),
    ] {
        for (degree, value) in limbs.into_iter().enumerate() {
            let row = start + degree;
            int[cols::W_AUX][row] = R::from(value);
            binary[cols::B_AUX][row] = BinaryPoly::from(value);
        }
    }
    for (start, carries) in [(AUX_C1_START, &witness.c1), (AUX_C2_START, &witness.c2)] {
        for (degree, carry) in carries.iter().enumerate() {
            let row = start + degree;
            let offset = *carry + CARRY_OFFSET;
            int[cols::W_AUX][row] = R::from(offset);
            binary[cols::B_AUX][row] = BinaryPoly::from(offset as u32);
        }
    }

    for (column, value) in [
        (cols::W_R, &witness.r),
        (cols::W_S, &witness.s),
        (cols::W_U1, &witness.u1),
        (cols::W_U2, &witness.u2),
    ] {
        fill_limb_accumulator(&mut arbitrary[column], value);
    }
    fill_scheduled_limb_accumulator(&mut arbitrary[cols::W_Q1], &witness.q1, AUX_Q1_START);
    fill_scheduled_limb_accumulator(&mut arbitrary[cols::W_Q2], &witness.q2, AUX_Q2_START);
    fill_scheduled_offset_carry_accumulator(
        &mut arbitrary[cols::W_C1_POLY],
        &witness.c1,
        AUX_C1_START,
    );
    fill_scheduled_offset_carry_accumulator(
        &mut arbitrary[cols::W_C2_POLY],
        &witness.c2,
        AUX_C2_START,
    );

    UairTrace {
        binary_poly: binary
            .into_iter()
            .map(|values| {
                DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    values,
                    BinaryPoly::from(0_u32),
                )
            })
            .collect::<Vec<_>>()
            .into(),
        arbitrary_poly: arbitrary
            .into_iter()
            .map(|values| {
                DenseMultilinearExtension::from_evaluations_vec(num_vars, values, zero_poly())
            })
            .collect::<Vec<_>>()
            .into(),
        int: int
            .into_iter()
            .map(|values| {
                DenseMultilinearExtension::from_evaluations_vec(num_vars, values, R::zero())
            })
            .collect::<Vec<_>>()
            .into(),
    }
}

fn verify_public_layout<RT, IntT, const D: usize>(
    public_trace: &UairTrace<'_, RT, IntT, D>,
    num_vars: usize,
) -> Result<(), PublicStructureError>
where
    RT: Clone,
    IntT: Clone + Zero + One + PartialEq,
{
    for (family, expected, actual) in [
        ("binary_poly", 0, public_trace.binary_poly.len()),
        (
            "arbitrary_poly",
            cols::NUM_ARBITRARY_PUBLIC,
            public_trace.arbitrary_poly.len(),
        ),
        ("int", cols::NUM_INT_PUBLIC, public_trace.int.len()),
    ] {
        if actual != expected {
            return Err(PublicStructureError::WrongColumnCount {
                column_family: family,
                expected,
                actual,
            });
        }
    }
    let rows = 1usize << num_vars;
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
    if rows <= IDENTITY_FINAL_ROW {
        return Err(PublicStructureError::WrongValue {
            column: "PRIVATE_ECDSA_PUBLIC_LAYOUT",
            row: rows,
        });
    }
    let zero = IntT::zero();
    let one = IntT::one();
    for row in 0..rows {
        let expected = [
            (cols::S_SCALAR_ACTIVE, row < SCALAR_BITS, "S_SCALAR_ACTIVE"),
            (cols::S_INIT, row == 0, "S_INIT"),
            (
                cols::S_SCALAR_FINAL,
                row == SCALAR_FINAL_ROW,
                "S_SCALAR_FINAL",
            ),
            (cols::S_ACC_ACTIVE, row < IDENTITY_FINAL_ROW, "S_ACC_ACTIVE"),
            (
                cols::S_IDENTITY_FINAL,
                row == IDENTITY_FINAL_ROW,
                "S_IDENTITY_FINAL",
            ),
            (
                cols::S_BLOCK_END,
                row < SCALAR_BITS && scalar_block_degree(row).is_some(),
                "S_BLOCK_END",
            ),
            (
                cols::N_BIT,
                row < SCALAR_BITS && bit_msb(&SECP256K1_N_UINT, row),
                "N_BIT",
            ),
            (
                cols::S_Q1,
                (AUX_Q1_START..AUX_Q1_START + NUM_LIMBS).contains(&row),
                "S_Q1",
            ),
            (
                cols::S_Q2,
                (AUX_Q2_START..AUX_Q2_START + NUM_LIMBS).contains(&row),
                "S_Q2",
            ),
            (
                cols::S_C1,
                (AUX_C1_START..AUX_C1_START + NUM_CARRIES).contains(&row),
                "S_C1",
            ),
            (
                cols::S_C2,
                (AUX_C2_START..AUX_C2_START + NUM_CARRIES).contains(&row),
                "S_C2",
            ),
        ];
        for (column, is_one, name) in expected {
            let expected_value = if is_one { &one } else { &zero };
            if &public_trace.int[column][row] != expected_value {
                return Err(PublicStructureError::WrongValue { column: name, row });
            }
        }
    }
    Ok(())
}

/// Typed verifier-side pinning for the public polynomial columns. The generic
/// UAIR hook can pin column counts and integer selectors, but its trait bounds
/// intentionally do not permit comparing arbitrary-polynomial coefficients.
pub fn verify_private_ecdsa_public_polynomials<R>(
    public_trace: &UairTrace<'_, R, R, 32>,
    num_vars: usize,
    e: &CbUint<EC_FP_INT_LIMBS>,
) -> Result<(), PublicStructureError>
where
    R: Clone + Zero + One + PartialEq + From<u32>,
{
    verify_public_layout(public_trace, num_vars)?;
    let rows = 1usize << num_vars;
    for row in 0..rows {
        let limb_monomial = if row < SCALAR_BITS && scalar_block_degree(row).is_some() {
            monomial(scalar_block_degree(row).expect("block-end degree must exist"))
        } else if (AUX_Q1_START..AUX_Q1_START + NUM_LIMBS).contains(&row) {
            monomial(row - AUX_Q1_START)
        } else if (AUX_Q2_START..AUX_Q2_START + NUM_LIMBS).contains(&row) {
            monomial(row - AUX_Q2_START)
        } else {
            zero_poly()
        };
        let carry_monomial = if (AUX_C1_START..AUX_C1_START + NUM_CARRIES).contains(&row) {
            monomial(row - AUX_C1_START)
        } else if (AUX_C2_START..AUX_C2_START + NUM_CARRIES).contains(&row) {
            monomial(row - AUX_C2_START)
        } else {
            zero_poly()
        };
        let n = if row == IDENTITY_FINAL_ROW {
            limb_poly(&SECP256K1_N_UINT)
        } else {
            zero_poly()
        };
        let e = if row == IDENTITY_FINAL_ROW {
            limb_poly(e)
        } else {
            zero_poly()
        };
        let carry_mask = if row == IDENTITY_FINAL_ROW {
            DensePolynomial {
                coeffs: array::from_fn(|index| {
                    if index < NUM_CARRIES {
                        R::one()
                    } else {
                        R::zero()
                    }
                }),
            }
        } else {
            zero_poly()
        };
        for (column, expected, name) in [
            (cols::PA_LIMB_MONOMIAL, limb_monomial, "PA_LIMB_MONOMIAL"),
            (cols::PA_CARRY_MONOMIAL, carry_monomial, "PA_CARRY_MONOMIAL"),
            (cols::PA_N, n, "PA_N"),
            (cols::PA_E, e, "PA_E"),
            (cols::PA_CARRY_MASK, carry_mask, "PA_CARRY_MASK"),
        ] {
            if public_trace.arbitrary_poly[column][row] != expected {
                return Err(PublicStructureError::WrongValue { column: name, row });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_uair::{
        BitOp, ConstraintBuilder, TraceRow,
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
        ideal::ImpossibleIdeal,
    };

    type ScalarInt = Int<EC_FP_INT_LIMBS>;

    const H8_E: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("E2D0FA68CFA8E68942FE9B038274B74A2F5E355D5C98CB9B6406E277B0F39188");
    const H8_R: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302");
    const H8_S: CbUint<EC_FP_INT_LIMBS> =
        CbUint::from_be_hex("7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568");

    #[test]
    fn exact_oracle_and_carry_bounds() {
        let witness = derive_private_ecdsa_scalars(H8_E, H8_R, H8_S);
        assert!(MAX_CARRY_ABS_BOUND < CARRY_OFFSET);
        assert_eq!(mul_mod_n(&witness.s, &witness.u1), witness.e);
        assert_eq!(mul_mod_n(&witness.s, &witness.u2), witness.r);
        assert!(
            witness
                .c1
                .iter()
                .all(|carry| carry.unsigned_abs() < CARRY_OFFSET as u64)
        );
        assert!(
            witness
                .c2
                .iter()
                .all(|carry| carry.unsigned_abs() < CARRY_OFFSET as u64)
        );
    }

    #[test]
    fn constraint_shape() {
        type U = PrivateEcdsaScalarsUair<ScalarInt>;
        assert_eq!(U::signature().shifts().len(), 18);
        assert_eq!(U::signature().bit_op_specs().len(), 1);
        assert_eq!(count_constraints::<U>(), 43);
        assert_eq!(count_max_degree::<U>(), 4);
        assert!(
            count_constraint_degrees::<U>()
                .iter()
                .all(|degree| *degree <= 4)
        );
    }

    #[test]
    fn trace_reconstructs_final_polynomials() {
        let witness = derive_private_ecdsa_scalars(H8_E, H8_R, H8_S);
        let trace = build_private_ecdsa_scalar_trace::<ScalarInt>(9, &witness);
        for (column, value) in [
            (cols::W_R, &witness.r),
            (cols::W_S, &witness.s),
            (cols::W_U1, &witness.u1),
            (cols::W_U2, &witness.u2),
            (cols::W_Q1, &witness.q1),
            (cols::W_Q2, &witness.q2),
        ] {
            assert_eq!(
                trace.arbitrary_poly[column][IDENTITY_FINAL_ROW],
                limb_poly(value)
            );
        }
        assert_eq!(
            trace.arbitrary_poly[cols::W_C1_POLY][IDENTITY_FINAL_ROW],
            offset_carry_poly(&witness.c1)
        );
        assert_eq!(
            trace.arbitrary_poly[cols::W_C2_POLY][IDENTITY_FINAL_ROW],
            offset_carry_poly(&witness.c2)
        );
    }

    #[test]
    fn typed_public_contract_rejects_polynomial_mutation() {
        let witness = derive_private_ecdsa_scalars(H8_E, H8_R, H8_S);
        let trace = build_private_ecdsa_scalar_trace::<ScalarInt>(9, &witness);
        let signature = PrivateEcdsaScalarsUair::<ScalarInt>::signature();
        let public = trace.public(&signature);
        verify_private_ecdsa_public_polynomials(&public, 9, &witness.e)
            .expect("honest public polynomial contract must verify");

        let mut owned = UairTrace {
            binary_poly: public.binary_poly.to_vec().into(),
            arbitrary_poly: public.arbitrary_poly.to_vec().into(),
            int: public.int.to_vec().into(),
        };
        owned.arbitrary_poly.to_mut()[cols::PA_N].evaluations[IDENTITY_FINAL_ROW].coeffs[0] +=
            ScalarInt::from(1_u32);
        assert!(verify_private_ecdsa_public_polynomials(&owned, 9, &witness.e).is_err());
    }

    struct ZeroChecker {
        row: usize,
        constraint: usize,
    }

    impl ConstraintBuilder for ZeroChecker {
        type Expr = i128;
        type Ideal = ImpossibleIdeal;

        fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal: &Self::Ideal) {
            self.constraint += 1;
        }

        fn assert_zero(&mut self, expr: Self::Expr) {
            assert_eq!(
                expr, 0,
                "assert_zero constraint {} failed at row {}",
                self.constraint, self.row
            );
            self.constraint += 1;
        }
    }

    fn eval_poly(poly: &DensePolynomial<i64, 32>, x: i128) -> i128 {
        poly.coeffs
            .iter()
            .rev()
            .fold(0_i128, |acc, coeff| acc * x + i128::from(*coeff))
    }

    fn eval_binary(poly: &BinaryPoly<32>, x: i128) -> i128 {
        poly.inner()
            .coeffs
            .iter()
            .rev()
            .fold(0_i128, |acc, bit| acc * x + i128::from(bit.into_inner()))
    }

    fn binary_u32(poly: &BinaryPoly<32>) -> u32 {
        poly.inner()
            .coeffs
            .iter()
            .enumerate()
            .fold(0_u32, |value, (bit, coefficient)| {
                value | (u32::from(coefficient.into_inner()) << bit)
            })
    }

    fn eval_flat(trace: &UairTrace<'_, i64, i64, 32>, flat: usize, row: usize, x: i128) -> i128 {
        if flat < cols::NUM_BINARY {
            eval_binary(&trace.binary_poly[flat][row], x)
        } else if flat < cols::NUM_BINARY + cols::NUM_ARBITRARY {
            eval_poly(&trace.arbitrary_poly[flat - cols::NUM_BINARY][row], x)
        } else {
            i128::from(trace.int[flat - cols::NUM_BINARY - cols::NUM_ARBITRARY][row])
        }
    }

    #[test]
    fn honest_trace_satisfies_every_zero_constraint_rowwise() {
        let witness = derive_private_ecdsa_scalars(H8_E, H8_R, H8_S);
        let trace = build_private_ecdsa_scalar_trace::<i64>(9, &witness);
        let signature = PrivateEcdsaScalarsUair::<i64>::signature();
        let rows = 1usize << 9;
        for x in [0_i128, 1, 2, 3, 5, -1] {
            for row in 0..rows {
                let up = (0..cols::NUM_BINARY + cols::NUM_ARBITRARY + cols::NUM_INT)
                    .map(|flat| eval_flat(&trace, flat, row, x))
                    .collect::<Vec<_>>();
                let mut down = signature
                    .shifts()
                    .iter()
                    .map(|shift| {
                        let shifted = row + shift.shift_amount();
                        if shifted < rows {
                            eval_flat(&trace, shift.source_col(), shifted, x)
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<_>>();
                for spec in signature.bit_op_specs() {
                    let source = binary_u32(&trace.binary_poly[spec.source_col()][row]);
                    let transformed = match spec.op() {
                        BitOp::Rot(amount) => source.rotate_left(amount),
                        BitOp::ShiftR(amount) => source >> amount,
                    };
                    down.push(eval_binary(&BinaryPoly::from(transformed), x));
                }
                let up = TraceRow::from_slice_with_layout(
                    &up,
                    signature.total_cols().as_column_layout(),
                );
                let down = TraceRow::from_slice_with_layout_and_bit_op(
                    &down,
                    signature.down_cols().as_column_layout(),
                    signature.bit_op_down_count(),
                );
                let mut checker = ZeroChecker { row, constraint: 0 };
                PrivateEcdsaScalarsUair::<i64>::constrain_general(
                    &mut checker,
                    up,
                    down,
                    |scalar| eval_poly(scalar, x),
                    |value, scalar| Some(*value * eval_poly(scalar, x)),
                    |_| ImpossibleIdeal,
                );
            }
        }
    }
}
