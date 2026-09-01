//! Proof-enforced canonical secp256k1 scalar ranges.
//!
//! Each private scalar is represented by 256 witness bits in the same
//! most-significant-bit-first row order consumed by the ECDSA Shamir trace.
//! A prefix comparator proves `scalar < n`; a second prefix state proves that
//! at least one bit is set.  The comparator states are derived recursively
//! from Boolean scalar bits, so they do not need separate Booleanity claims.

use core::marker::PhantomData;

use crypto_bigint::Uint as CbUint;
use crypto_primitives::ConstSemiring;
use rand::RngCore;
use zinc_poly::{mle::DenseMultilinearExtension, univariate::dense::DensePolynomial};
use zinc_uair::{
    ConstraintBuilder, PublicColumnLayout, PublicStructureError, ShiftSpec, TotalColumnLayout,
    TraceRow, Uair, UairSignature, UairTrace, ideal::ImpossibleIdeal,
};

use crate::{GenerateRandomTrace, ecdsa::SECP256K1_N_UINT, ecdsa_doubling::EC_FP_INT_LIMBS};

/// Number of bits in a secp256k1 scalar.
pub const SCALAR_BITS: usize = 256;

/// Row holding the comparator's state after all 256 bits were consumed.
pub const FINAL_ROW: usize = SCALAR_BITS;

/// Public and witness integer-column indices.
pub mod cols {
    pub const S_ACTIVE: usize = 0;
    pub const S_INIT: usize = 1;
    pub const S_FINAL: usize = 2;
    pub const N_BIT: usize = 3;
    pub const NUM_INT_PUB: usize = 4;

    pub const W_R_BIT: usize = 4;
    pub const W_R_LESS: usize = 5;
    pub const W_R_SEEN: usize = 6;
    pub const W_S_BIT: usize = 7;
    pub const W_S_LESS: usize = 8;
    pub const W_S_SEEN: usize = 9;
    pub const NUM_INT: usize = 10;
}

/// Public and witness integer-column indices for the three-state comparator.
pub mod packed_cols {
    pub const S_ACTIVE: usize = 0;
    pub const S_INIT: usize = 1;
    pub const S_FINAL: usize = 2;
    pub const N_BIT: usize = 3;
    pub const NUM_INT_PUB: usize = 4;

    pub const W_R_BIT: usize = 4;
    pub const W_R_STATE: usize = 5;
    pub const W_S_BIT: usize = 6;
    pub const W_S_STATE: usize = 7;
    pub const NUM_INT: usize = 8;
}

/// Standalone UAIR for `1 <= r,s < n`.
#[derive(Clone, Debug)]
pub struct PrivateScalarRangeUair<R>(PhantomData<R>);

/// Column-reduced comparator using the three states reachable after the
/// leading secp256k1-order bit: `0 = less/zero`, `1 = equal/nonzero`, and
/// `2 = less/nonzero`.  The row-zero transition is handled separately.
#[derive(Clone, Debug)]
pub struct PrivateScalarRangePackedUair<R>(PhantomData<R>);

impl<R> Uair for PrivateScalarRangeUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, cols::NUM_INT_PUB);
        let shifts = [
            cols::W_R_LESS,
            cols::W_R_SEEN,
            cols::W_S_LESS,
            cols::W_S_SEEN,
        ]
        .into_iter()
        .map(|column| ShiftSpec::new(column, 1))
        .collect();
        UairSignature::new(total, public, shifts, vec![], vec![])
            .with_int_witness_bit_cols(vec![cols::W_R_BIT, cols::W_S_BIT])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int = up.int;
        let active = &int[cols::S_ACTIVE];
        let init = &int[cols::S_INIT];
        let final_row = &int[cols::S_FINAL];
        let n_bit = &int[cols::N_BIT];
        let one = from_ref(&DensePolynomial::new([R::from(1_u32)]));
        let inactive = one.clone() - active;
        let padding = inactive.clone() - final_row;

        // Int shifts are returned in source-column order, which matches the
        // four state columns declared in `signature`.
        for (bit, less, seen, next_less, next_seen) in [
            (
                cols::W_R_BIT,
                cols::W_R_LESS,
                cols::W_R_SEEN,
                &down.int[0],
                &down.int[1],
            ),
            (
                cols::W_S_BIT,
                cols::W_S_LESS,
                cols::W_S_SEEN,
                &down.int[2],
                &down.int[3],
            ),
        ] {
            let bit = &int[bit];
            let less = &int[less];
            let seen = &int[seen];
            let not_bit = one.clone() - bit;
            let not_less = one.clone() - less;
            let not_seen = one.clone() - seen;
            let not_n_bit = one.clone() - n_bit;

            // Lexicographic prefix comparison.  A `0` below a modulus `1`
            // makes the prefix permanently less; a `1` above a modulus `0`
            // is forbidden while the prefixes are still equal.
            let less_update = less.clone() + &(n_bit.clone() * &not_less * &not_bit);
            b.assert_zero(active.clone() * &(next_less.clone() - &less_update));
            b.assert_zero(active.clone() * &(not_n_bit.clone() * &not_less * bit));

            // OR-reduce the private bits to exclude zero.
            let seen_update = seen.clone() + &(not_seen * bit);
            b.assert_zero(active.clone() * &(next_seen.clone() - &seen_update));

            // Boundary and ownership constraints.  Padding witness cells are
            // fixed to zero so no unowned private values survive off-domain.
            b.assert_zero(init.clone() * less);
            b.assert_zero(init.clone() * seen);
            b.assert_zero(final_row.clone() * &(less.clone() - &one));
            b.assert_zero(final_row.clone() * &(seen.clone() - &one));
            b.assert_zero(inactive.clone() * bit);
            b.assert_zero(padding.clone() * less);
            b.assert_zero(padding.clone() * seen);
        }
    }

    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero + num_traits::One + PartialEq,
    {
        verify_public_contract(public_trace, num_vars)
    }
}

impl<R> Uair for PrivateScalarRangePackedUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type Ideal = ImpossibleIdeal;
    type Scalar = DensePolynomial<R, 32>;

    fn signature() -> UairSignature {
        let total = TotalColumnLayout::new(0, 0, packed_cols::NUM_INT);
        let public = PublicColumnLayout::new(0, 0, packed_cols::NUM_INT_PUB);
        let shifts = [packed_cols::W_R_STATE, packed_cols::W_S_STATE]
            .into_iter()
            .map(|column| ShiftSpec::new(column, 1))
            .collect();
        UairSignature::new(total, public, shifts, vec![], vec![])
            .with_int_witness_bit_cols(vec![packed_cols::W_R_BIT, packed_cols::W_S_BIT])
    }

    fn constrain_general<B, FromR, MulByScalar, IFromR>(
        b: &mut B,
        up: TraceRow<B::Expr>,
        down: TraceRow<B::Expr>,
        from_ref: FromR,
        _mbs: MulByScalar,
        _ideal_from_ref: IFromR,
    ) where
        B: ConstraintBuilder,
        FromR: Fn(&Self::Scalar) -> B::Expr,
        MulByScalar: Fn(&B::Expr, &Self::Scalar) -> Option<B::Expr>,
        IFromR: Fn(&Self::Ideal) -> B::Ideal,
    {
        let int = up.int;
        let active = &int[packed_cols::S_ACTIVE];
        let init = &int[packed_cols::S_INIT];
        let final_row = &int[packed_cols::S_FINAL];
        let n_bit = &int[packed_cols::N_BIT];
        let one = from_ref(&DensePolynomial::new([R::from(1_u32)]));
        let two = one.clone() + &one;
        let body = active.clone() - init;
        let inactive = one.clone() - active;
        let padding = inactive.clone() - final_row;

        for (bit, state, next_state) in [
            (packed_cols::W_R_BIT, packed_cols::W_R_STATE, &down.int[0]),
            (packed_cols::W_S_BIT, packed_cols::W_S_STATE, &down.int[1]),
        ] {
            let bit = &int[bit];
            let state = &int[state];
            let not_bit = one.clone() - bit;
            let not_n_bit = one.clone() - n_bit;

            // Lagrange numerators for states 0, 1, 2. Each evaluates to
            // two at its own state and zero at the other two states.
            let j0 = (state.clone() - &one) * &(state.clone() - &two);
            let j1 = (state.clone() + state) * &(two.clone() - state);
            let j2 = state.clone() * &(state.clone() - &one);

            // Since n's leading bit is one, the first next-state is exactly
            // the private bit: zero enters less/zero; one enters
            // equal/nonzero.
            b.assert_zero(init.clone() * &(next_state.clone() - bit));
            b.assert_zero(init.clone() * state);

            // State transitions for rows 1..255:
            //   0 -> 2*bit
            //   1 -> 1 + n_bit*(1-bit)
            //   2 -> 2.
            // J_i = 2*I_i, so both sides are scaled by two.
            let rhs = j0 * &(bit.clone() + bit)
                + &(j1.clone() * &(one.clone() + &(n_bit.clone() * &not_bit)))
                + &(j2 * &two);
            b.assert_zero(body.clone() * &((next_state.clone() + next_state) - &rhs));

            // State 1 means the prefixes are still equal. A private one
            // against a modulus zero is the unique greater-than transition.
            b.assert_zero(body.clone() * &(j1 * &not_n_bit * bit));

            // State 2 is exactly the conjunction `less && nonzero`.
            b.assert_zero(final_row.clone() * &(state.clone() - &two));
            b.assert_zero(inactive.clone() * bit);
            b.assert_zero(padding.clone() * state);
        }
    }

    fn verify_public_structure<RT, IntT, const D: usize>(
        public_trace: &UairTrace<'_, RT, IntT, D>,
        num_vars: usize,
    ) -> Result<(), PublicStructureError>
    where
        RT: Clone,
        IntT: Clone + num_traits::Zero + num_traits::One + PartialEq,
    {
        verify_public_contract(public_trace, num_vars)
    }
}

fn bit_msb(value: &CbUint<EC_FP_INT_LIMBS>, row: usize) -> bool {
    let bit = SCALAR_BITS - 1 - row;
    let word = bit / 64;
    let offset = bit % 64;
    ((value.as_words()[word] >> offset) & 1) == 1
}

fn push_bit<R: num_traits::Zero + num_traits::One>(column: &mut [R], row: usize, bit: bool) {
    column[row] = if bit { R::one() } else { R::zero() };
}

/// Build the canonical-range trace for two private secp256k1 scalars.
pub fn build_private_scalar_range_trace<R>(
    num_vars: usize,
    r: CbUint<EC_FP_INT_LIMBS>,
    s: CbUint<EC_FP_INT_LIMBS>,
) -> UairTrace<'static, R, R, 32>
where
    R: Clone + num_traits::Zero + num_traits::One,
{
    let rows = 1usize << num_vars;
    assert!(
        rows > FINAL_ROW,
        "private-scalar trace needs at least 257 rows"
    );
    let mut int = vec![vec![R::zero(); rows]; cols::NUM_INT];

    for row in 0..SCALAR_BITS {
        int[cols::S_ACTIVE][row] = R::one();
        push_bit(&mut int[cols::N_BIT], row, bit_msb(&SECP256K1_N_UINT, row));
    }
    int[cols::S_INIT][0] = R::one();
    int[cols::S_FINAL][FINAL_ROW] = R::one();

    for (value, bit_col, less_col, seen_col) in [
        (r, cols::W_R_BIT, cols::W_R_LESS, cols::W_R_SEEN),
        (s, cols::W_S_BIT, cols::W_S_LESS, cols::W_S_SEEN),
    ] {
        let mut less = false;
        let mut seen = false;
        for row in 0..SCALAR_BITS {
            let bit = bit_msb(&value, row);
            let n_bit = bit_msb(&SECP256K1_N_UINT, row);
            push_bit(&mut int[bit_col], row, bit);
            push_bit(&mut int[less_col], row, less);
            push_bit(&mut int[seen_col], row, seen);
            less |= n_bit && !bit;
            seen |= bit;
        }
        push_bit(&mut int[less_col], FINAL_ROW, less);
        push_bit(&mut int[seen_col], FINAL_ROW, seen);
    }

    UairTrace {
        binary_poly: vec![].into(),
        arbitrary_poly: vec![].into(),
        int: int
            .into_iter()
            .map(|evaluations| {
                DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, R::zero())
            })
            .collect::<Vec<_>>()
            .into(),
    }
}

/// Build the trace for [`PrivateScalarRangePackedUair`].
pub fn build_private_scalar_range_packed_trace<R>(
    num_vars: usize,
    r: CbUint<EC_FP_INT_LIMBS>,
    s: CbUint<EC_FP_INT_LIMBS>,
) -> UairTrace<'static, R, R, 32>
where
    R: Clone + num_traits::Zero + num_traits::One + From<u32>,
{
    let rows = 1usize << num_vars;
    assert!(
        rows > FINAL_ROW,
        "private-scalar trace needs at least 257 rows"
    );
    let mut int = vec![vec![R::zero(); rows]; packed_cols::NUM_INT];

    for row in 0..SCALAR_BITS {
        int[packed_cols::S_ACTIVE][row] = R::one();
        push_bit(
            &mut int[packed_cols::N_BIT],
            row,
            bit_msb(&SECP256K1_N_UINT, row),
        );
    }
    int[packed_cols::S_INIT][0] = R::one();
    int[packed_cols::S_FINAL][FINAL_ROW] = R::one();

    for (value, bit_col, state_col) in [
        (r, packed_cols::W_R_BIT, packed_cols::W_R_STATE),
        (s, packed_cols::W_S_BIT, packed_cols::W_S_STATE),
    ] {
        let mut less = false;
        let mut seen = false;
        for row in 0..SCALAR_BITS {
            let bit = bit_msb(&value, row);
            let n_bit = bit_msb(&SECP256K1_N_UINT, row);
            push_bit(&mut int[bit_col], row, bit);
            if row > 0 {
                int[state_col][row] = R::from(if !seen {
                    0_u32
                } else if less {
                    2_u32
                } else {
                    1_u32
                });
            }
            less |= n_bit && !bit;
            seen |= bit;
        }
        int[state_col][FINAL_ROW] = R::from(if !seen {
            0_u32
        } else if less {
            2_u32
        } else {
            1_u32
        });
    }

    UairTrace {
        binary_poly: vec![].into(),
        arbitrary_poly: vec![].into(),
        int: int
            .into_iter()
            .map(|evaluations| {
                DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations, R::zero())
            })
            .collect::<Vec<_>>()
            .into(),
    }
}

fn verify_public_contract<RT, IntT, const D: usize>(
    public_trace: &UairTrace<'_, RT, IntT, D>,
    num_vars: usize,
) -> Result<(), PublicStructureError>
where
    RT: Clone,
    IntT: Clone + num_traits::Zero + num_traits::One + PartialEq,
{
    for (column_family, expected, actual) in [
        ("binary_poly", 0, public_trace.binary_poly.len()),
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

    let rows = 1usize << num_vars;
    if rows <= FINAL_ROW {
        return Err(PublicStructureError::WrongValue {
            column: "PRIVATE_SCALAR_PUBLIC_LAYOUT",
            row: rows,
        });
    }
    let zero = IntT::zero();
    let one = IntT::one();
    for row in 0..rows {
        for (column, expected, name) in [
            (
                cols::S_ACTIVE,
                if row < SCALAR_BITS { &one } else { &zero },
                "S_ACTIVE",
            ),
            (cols::S_INIT, if row == 0 { &one } else { &zero }, "S_INIT"),
            (
                cols::S_FINAL,
                if row == FINAL_ROW { &one } else { &zero },
                "S_FINAL",
            ),
            (
                cols::N_BIT,
                if row < SCALAR_BITS && bit_msb(&SECP256K1_N_UINT, row) {
                    &one
                } else {
                    &zero
                },
                "N_BIT",
            ),
        ] {
            if &public_trace.int[column][row] != expected {
                return Err(PublicStructureError::WrongValue { column: name, row });
            }
        }
    }
    Ok(())
}

impl<R> GenerateRandomTrace<32> for PrivateScalarRangeUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        // Small nonzero values are sufficient for generic random round trips;
        // dedicated boundary tests cover the full 256-bit interval.
        let r = CbUint::from_u64((rng.next_u64() % (u64::MAX - 1)) + 1);
        let s = CbUint::from_u64((rng.next_u64() % (u64::MAX - 1)) + 1);
        build_private_scalar_range_trace(num_vars, r, s)
    }
}

impl<R> GenerateRandomTrace<32> for PrivateScalarRangePackedUair<R>
where
    R: ConstSemiring + From<u32> + 'static,
{
    type PolyCoeff = R;
    type Int = R;

    fn generate_random_trace<Rng: RngCore + ?Sized>(
        num_vars: usize,
        rng: &mut Rng,
    ) -> UairTrace<'static, R, R, 32> {
        let r = CbUint::from_u64((rng.next_u64() % (u64::MAX - 1)) + 1);
        let s = CbUint::from_u64((rng.next_u64() % (u64::MAX - 1)) + 1);
        build_private_scalar_range_packed_trace(num_vars, r, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::crypto_bigint_int::Int;
    use zinc_uair::{
        constraint_counter::count_constraints,
        degree_counter::{count_constraint_degrees, count_max_degree},
    };

    type ScalarInt = Int<EC_FP_INT_LIMBS>;

    fn public_trace(
        r: CbUint<EC_FP_INT_LIMBS>,
        s: CbUint<EC_FP_INT_LIMBS>,
    ) -> UairTrace<'static, ScalarInt, ScalarInt, 32> {
        let trace = build_private_scalar_range_trace::<ScalarInt>(9, r, s);
        let signature = PrivateScalarRangeUair::<ScalarInt>::signature();
        let public = trace.public(&signature);
        UairTrace {
            binary_poly: public.binary_poly.to_vec().into(),
            arbitrary_poly: public.arbitrary_poly.to_vec().into(),
            int: public.int.to_vec().into(),
        }
    }

    #[test]
    fn range_constraint_shape() {
        type U = PrivateScalarRangeUair<ScalarInt>;
        assert_eq!(count_constraints::<U>(), 20);
        assert_eq!(count_max_degree::<U>(), 4);
        assert!(
            count_constraint_degrees::<U>()
                .iter()
                .all(|degree| *degree <= 4)
        );
        assert_eq!(
            U::signature().int_witness_bit_cols(),
            &[cols::W_R_BIT, cols::W_S_BIT]
        );
    }

    #[test]
    fn packed_range_constraint_shape() {
        type U = PrivateScalarRangePackedUair<ScalarInt>;
        assert_eq!(count_constraints::<U>(), 14);
        assert_eq!(count_max_degree::<U>(), 5);
        assert!(
            count_constraint_degrees::<U>()
                .iter()
                .all(|degree| *degree <= 5)
        );
        assert_eq!(
            U::signature().int_witness_bit_cols(),
            &[packed_cols::W_R_BIT, packed_cols::W_S_BIT]
        );
    }

    #[test]
    fn packed_boundary_trace_reaches_accepting_state() {
        let trace = build_private_scalar_range_packed_trace::<ScalarInt>(
            9,
            CbUint::ONE,
            SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE),
        );
        assert_eq!(
            trace.int[packed_cols::W_R_STATE][FINAL_ROW],
            ScalarInt::from(2_u32)
        );
        assert_eq!(
            trace.int[packed_cols::W_S_STATE][FINAL_ROW],
            ScalarInt::from(2_u32)
        );
    }

    #[test]
    fn boundary_trace_accepts_one_and_n_minus_one() {
        let public = public_trace(CbUint::ONE, SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE));
        PrivateScalarRangeUair::<ScalarInt>::verify_public_structure(&public, 9)
            .expect("canonical public structure must verify");
    }

    #[test]
    fn public_structure_rejects_mutated_modulus_bit() {
        let mut public = public_trace(CbUint::ONE, CbUint::ONE);
        let cell = &mut public.int.to_mut()[cols::N_BIT].evaluations[17];
        *cell = if *cell == ScalarInt::from(0_u32) {
            ScalarInt::from(1_u32)
        } else {
            ScalarInt::from(0_u32)
        };
        assert!(matches!(
            PrivateScalarRangeUair::<ScalarInt>::verify_public_structure(&public, 9),
            Err(PublicStructureError::WrongValue {
                column: "N_BIT",
                row: 17
            })
        ));
    }
}
