//! Zinc+ PIOP for UCS - end-to-end protocol.
//!
//! Implements the Zinc+ compiler pipeline (cf. paper, Section "Zinc+
//! Compiler"):
//!
//! ```text
//! Z[X]  --\phi_q-->  F_q[X]  --MLE eval-->  F_q[X]  --\psi_a-->  F_q
//!         Step 1               Step 2                  Step 3
//! ```
//!
//! After the three compiler steps, the protocol continues with:
//!
//! - Step 4: combined CPR + Lookup multi-degree sumcheck (CPR group at degree
//!   `max_deg+2`, one lookup group per table type; shared eval point `r*`)
//! - Step 5: multi-point evaluation sumcheck (combines up/down evals at r* into
//!   a single evaluation point r_0)
//! - Step 6: lift-and-project (unprojected MLE evaluations at r_0)
//! - Step 7: Zip+ PCS open/verify at r_0

pub mod application;
pub mod fixed_prime;
pub mod prover;
pub mod verifier;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crypto_primitives::{ConstIntRing, ConstIntSemiring, FromWithConfig, PrimeField, Semiring};
use std::{fmt::Debug, marker::PhantomData};
use thiserror::Error;
use zinc_piop::{
    combined_poly_resolver::{CombinedPolyResolverError, Proof as CombinedPolyResolverProof},
    ideal_check::{IdealCheckError, Proof as IdealCheckProof},
    lookup::{BatchedLookupProof, LookupError},
    multipoint_eval::{MultipointEvalError, Proof as MultipointEvalProof},
    projections::ProjectedTrace,
    sumcheck::multi_degree::MultiDegreeSumcheckProof,
};
use zinc_poly::{
    ConstCoeffBitWidth, EvaluationError as PolyEvaluationError,
    mle::DenseMultilinearExtension,
    univariate::{
        binary::BinaryPoly,
        dense::DensePolynomial,
        dynamic::over_field::{DynamicPolyVecF, DynamicPolynomialF},
    },
};
use zinc_primality::PrimalityTest;
use zinc_transcript::traits::{ConstTranscribable, GenTranscribable, Transcribable, Transcript};
use zinc_uair::{Uair, ideal::Ideal};
use zinc_utils::{cfg_extend, cfg_into_iter, cfg_iter, named::Named};
use zip_plus::{
    ZipError,
    code::LinearCode,
    pcs::structs::{ZipPlusCommitment, ZipTypes},
};

//
// Data structures
//

/// Full proof produced by the Zinc+ PIOP for UCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<F: PrimeField> {
    /// Zip+ commitments to the witness columns.
    pub commitments: (ZipPlusCommitment, ZipPlusCommitment, ZipPlusCommitment),
    /// Serialized PCS proof data (Zip+ proving transcripts).
    pub zip: Vec<u8>,
    /// Randomized ideal check proof.
    pub ideal_check: IdealCheckProof<F>,
    /// Combined polynomial resolver proof (up_evals + down_evals +
    /// bit_op_down_evals).
    pub resolver: CombinedPolyResolverProof<F>,
    /// Multi-degree sumcheck proof (CPR group + future lookup groups).
    pub combined_sumcheck: MultiDegreeSumcheckProof<F>,
    /// Multi-point evaluation sumcheck proof. Reduces all CPR claims at
    /// `r*` (up evals + row-shift down evals + bit-op virtual-column
    /// down evals) to a single evaluation point `r_0`. Bit-op sources
    /// are folded in as additional `up` slots; their consistency is
    /// discharged at `r_0` in Step 6 by applying the bit-op locally to
    /// the source's lifted eval.
    pub multipoint_eval: MultipointEvalProof<F>,
    /// Witness-only polynomial MLE evaluations at r_0 in F_q[X]
    /// (after \phi_q, before \psi_a), ordered as
    /// `[wit_bin..., wit_arb..., wit_int...]`.
    /// The verifier recomputes public lifted_evals from public data,
    /// interleaves them with these, and derives scalar open_evals via
    /// \psi_a for the sumcheck consistency check and Zip+ PCS verify.
    pub witness_lifted_evals: Vec<DynamicPolynomialF<F>>,
    /// Lookup argument proof. `None` when the UAIR has no lookup specs.
    pub lookup_proof: Option<BatchedLookupProof<F>>,
}

impl<F> GenTranscribable for Proof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    fn read_transcription_bytes_exact(bytes: &[u8]) -> Self {
        let (commit0, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit1, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);
        let (commit2, bytes) = ZipPlusCommitment::read_transcription_bytes_subset(bytes);

        let (zip_len, bytes) = u32::read_transcription_bytes_subset(bytes);
        let zip_len = usize::try_from(zip_len).expect("zip length must fit into usize");
        let (zip_bytes, bytes) = bytes.split_at(zip_len);
        let zip = zip_bytes.to_vec();

        let (ideal_check, bytes) = IdealCheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (resolver, bytes) =
            CombinedPolyResolverProof::<F>::read_transcription_bytes_subset(bytes);
        let (combined_sumcheck, bytes) =
            MultiDegreeSumcheckProof::<F>::read_transcription_bytes_subset(bytes);
        let (multipoint_eval, bytes) =
            MultipointEvalProof::<F>::read_transcription_bytes_subset(bytes);

        let (witness_vec, bytes) = DynamicPolyVecF::<F>::read_transcription_bytes_subset(bytes);
        let witness_lifted_evals = witness_vec.0;

        // TODO: deserialize lookup_proof once BatchedLookupProof gets
        // Transcribable impls (lookup is not yet implemented).
        assert!(bytes.is_empty(), "All bytes should be consumed");

        Self {
            commitments: (commit0, commit1, commit2),
            zip,
            ideal_check,
            resolver,
            combined_sumcheck,
            multipoint_eval,
            witness_lifted_evals,
            lookup_proof: None,
        }
    }

    fn write_transcription_bytes_exact(&self, mut buf: &mut [u8]) {
        // 3 commitments (ConstTranscribable - no length prefix)
        buf = self.commitments.0.write_transcription_bytes_subset(buf);
        buf = self.commitments.1.write_transcription_bytes_subset(buf);
        buf = self.commitments.2.write_transcription_bytes_subset(buf);

        // zip: u32 length + raw bytes
        let zip_len = u32::try_from(self.zip.len()).expect("zip length must fit into u32");
        zip_len.write_transcription_bytes_exact(&mut buf[..u32::NUM_BYTES]);
        buf = &mut buf[u32::NUM_BYTES..];
        buf[..self.zip.len()].copy_from_slice(&self.zip);
        buf = &mut buf[self.zip.len()..];

        // ideal_check: u32 length prefix + data
        buf = self.ideal_check.write_transcription_bytes_subset(buf);

        // resolver: u32 length prefix + data
        buf = self.resolver.write_transcription_bytes_subset(buf);

        // combined_sumcheck: u32 length prefix + data
        buf = self.combined_sumcheck.write_transcription_bytes_subset(buf);

        // multipoint_eval: u32 length prefix + data
        buf = self.multipoint_eval.write_transcription_bytes_subset(buf);

        // witness_lifted_evals: u32 length prefix + DynamicPolyVecF encoding
        // TODO: serialize lookup_proof once BatchedLookupProof gets
        // Transcribable impls (lookup is not yet implemented).
        DynamicPolyVecF::reinterpret(&self.witness_lifted_evals)
            .write_transcription_bytes_subset(buf);
    }
}

impl<F> Transcribable for Proof<F>
where
    F: PrimeField,
    F::Inner: ConstTranscribable,
    F::Modulus: ConstTranscribable,
{
    #[allow(clippy::arithmetic_side_effects)]
    fn get_num_bytes(&self) -> usize {
        let witness_vec = DynamicPolyVecF::reinterpret(&self.witness_lifted_evals);
        3 * ZipPlusCommitment::NUM_BYTES
            + u32::NUM_BYTES
            + self.zip.len()
            + IdealCheckProof::<F>::LENGTH_NUM_BYTES
            + self.ideal_check.get_num_bytes()
            + CombinedPolyResolverProof::<F>::LENGTH_NUM_BYTES
            + self.resolver.get_num_bytes()
            + MultiDegreeSumcheckProof::<F>::LENGTH_NUM_BYTES
            + self.combined_sumcheck.get_num_bytes()
            + MultipointEvalProof::<F>::LENGTH_NUM_BYTES
            + self.multipoint_eval.get_num_bytes()
            // TODO: add lookup_proof size once BatchedLookupProof gets
            // Transcribable impls (lookup is not yet implemented).
            + DynamicPolyVecF::<F>::LENGTH_NUM_BYTES
            + witness_vec.get_num_bytes()
    }
}

/// Trait bundling the various type parameters for the public inputs (NYI),
/// witness and Zinc+ PIOP.
pub trait ZincTypes<const DEGREE_PLUS_ONE: usize>: Clone + Debug {
    /// Main integer type for the protocol, used as a coefficient type for the
    /// arbitrary polynomial trace columns and for the integer trace columns.
    type Int: Semiring
        + ConstTranscribable
        + ConstCoeffBitWidth
        + Named
        + Default
        + Clone
        + Send
        + Sync
        + 'static;

    /// Projecting element to project Zip+ evaluations and UAIR scalars to the
    /// field.
    type Chal: ConstIntRing + ConstTranscribable + Named;

    /// Evaluation point type, used for all column types in Zip+ to evaluate
    /// multilinear polynomials.
    type Pt: ConstIntRing;

    /// Randomly sampled field modulus type, used throughout the protocol for
    /// finite field operations.
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    /// Primality test for the field modulus.
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the binary polynomial trace columns.
    /// `CombR` is independent per witness type — sized to fit the
    /// inner products that lane actually performs (binary is much
    /// narrower than arb/int).
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns.
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, DEGREE_PLUS_ONE>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns.
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Linear code used in Zip+ for the binary polynomial trace columns.
    type BinaryLc: LinearCode<Self::BinaryZt>;

    /// Linear code used in Zip+ for the arbitrary polynomial trace columns.
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;

    /// Linear code used in Zip+ for the integer trace columns.
    type IntLc: LinearCode<Self::IntZt>;
}

/// Type bundle for the **folded** Zinc+ PIOP (1× fold, 2× column splitting).
///
/// The PIOP runs at trace degree `D` (so the trace and UAIR are unchanged
/// from the unfolded path), but the binary commitment is over
/// `BinaryPoly<HALF_D>` — each `BinaryPoly<D>` witness column is split into two
/// `BinaryPoly<HALF_D>` halves before commit. This decouples the trace's
/// `BinaryPoly<D>` from the PCS's `BinaryPoly<HALF_D>`, which `ZincTypes<D>`
/// would otherwise force to be the same (`BinaryZt::Eval =
/// BinaryPoly<DEGREE_PLUS_ONE>`).
///
/// Arbitrary and integer commitments are unchanged.
pub trait FoldedZincTypes<const D: usize, const HALF_D: usize>: Clone + Debug {
    type Int: Semiring
        + ConstTranscribable
        + ConstCoeffBitWidth
        + Named
        + Default
        + Clone
        + Send
        + Sync
        + 'static;

    type Chal: ConstIntRing + ConstTranscribable + Named;

    type Pt: ConstIntRing;

    type Fmod: ConstIntSemiring + ConstTranscribable + Named;

    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the **split** binary trace columns.
    /// `Eval = BinaryPoly<HALF_D>` — one round of 2× folding.
    /// `CombR` is independent per witness type.
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<HALF_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns (unchanged from
    /// the unfolded path: degree-`D` polynomials).
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<Self::Int, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the integer trace columns (unchanged).
    type IntZt: ZipTypes<
            Eval = Self::Int,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;

    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;

    type IntLc: LinearCode<Self::IntZt>;
}

/// Like [`FoldedZincTypes`], but additionally folds int witness columns by
/// 2× via the `v = lo + 2^128 · hi` decomposition, with halves stored as
/// `Int<INT_HALF_LIMBS>`. The int Zip+ commits the split witness at length
/// `2n` and the protocol opens at the same extended point `(r_0 ‖ γ)` as
/// the binary fold; the verifier mirrors with `(1−γ) c1 + γ c2` and
/// `alpha_stride = 1` (since `IntZt::Cw` is scalar).
pub trait IntFoldedZincTypes<
    const D: usize,
    const HALF_D: usize,
    const INT_LIMBS: usize,
    const INT_HALF_LIMBS: usize,
>: Clone + Debug
{
    type Chal: ConstIntRing + ConstTranscribable + Named;
    type Pt: ConstIntRing;
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    /// Zip+ types for the split binary trace columns
    /// (`Eval = BinaryPoly<HALF_D>`).
    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<HALF_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the arbitrary polynomial trace columns
    /// (unchanged from the unfolded path).
    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<crypto_primitives::crypto_bigint_int::Int<INT_LIMBS>, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    /// Zip+ types for the split integer trace columns
    /// (`Eval = Int<INT_HALF_LIMBS>`).
    type IntZt: ZipTypes<
            Eval = crypto_primitives::crypto_bigint_int::Int<INT_HALF_LIMBS>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;
    type IntLc: LinearCode<Self::IntZt>;
}

/// 4× counterpart of [`IntFoldedZincTypes`]: also folds binary by 4× via
/// `BinaryPoly<D> → BinaryPoly<QUARTER_D>` AND folds int by 4× via
/// quarters (`v = q_0 + 2^64·q_1 + 2^128·q_2 + 2^192·q_3`), each stored
/// as `Int<INT_QUARTER_LIMBS>`. Both binary and int commit at length
/// `4n` and open at `(r_0 ‖ γ_1 ‖ γ_2)`. Verifier mirrors with the
/// 4-block algebra `(1−γ_1)(1−γ_2) c[0] + γ_1(1−γ_2) c[2] +
/// (1−γ_1)γ_2 c[1] + γ_1·γ_2 c[3]` and `alpha_stride = 1`.
pub trait IntFoldedZincTypes4x<
    const D: usize,
    const QUARTER_D: usize,
    const INT_LIMBS: usize,
    const INT_QUARTER_LIMBS: usize,
>: Clone + Debug
{
    type Chal: ConstIntRing + ConstTranscribable + Named;
    type Pt: ConstIntRing;
    type Fmod: ConstIntSemiring + ConstTranscribable + Named;
    type PrimeTest: PrimalityTest<Self::Fmod>;

    type BinaryZt: ZipTypes<
            Eval = BinaryPoly<QUARTER_D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type ArbitraryZt: ZipTypes<
            Eval = DensePolynomial<crypto_primitives::crypto_bigint_int::Int<INT_LIMBS>, D>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type IntZt: ZipTypes<
            Eval = crypto_primitives::crypto_bigint_int::Int<INT_QUARTER_LIMBS>,
            Chal = Self::Chal,
            Pt = Self::Pt,
            Fmod = Self::Fmod,
            PrimeTest = Self::PrimeTest,
        >;

    type BinaryLc: LinearCode<Self::BinaryZt>;
    type ArbitraryLc: LinearCode<Self::ArbitraryZt>;
    type IntLc: LinearCode<Self::IntZt>;
}

/// Main struct for the Zinc+ PIOP. The protocol is implemented as associated
/// functions on it.
///
/// (Note that type parameters are further constrained in the impl blocks for
/// the prover and verifier)
#[derive(Copy, Clone, Default, Debug)]
pub struct ZincPlusPiop<Zt, U, F, const DEGREE_PLUS_ONE: usize>(PhantomData<(Zt, U, F)>)
where
    Zt: ZincTypes<DEGREE_PLUS_ONE>,
    U: Uair,
    F: PrimeField;

/// Error type for error happening during the protocol execution (prover and
/// verifier).
#[derive(Debug, Error)]
pub enum ProtocolError<F: PrimeField, I: Ideal> {
    #[error("ideal check failed: {0}")]
    IdealCheck(#[from] IdealCheckError<F, I>),
    #[error("combined poly resolver failed: {0}")]
    Resolver(#[from] CombinedPolyResolverError<F>),
    #[error("scalar projection failed: {0}")]
    ScalarProjection(PolyEvaluationError),
    #[error("multi-point evaluation failed: {0}")]
    MultipointEval(#[from] MultipointEvalError<F>),
    #[error("lifted eval psi_a projection failed: {0}")]
    LiftedEvalProjection(PolyEvaluationError),
    #[error("lifted_evals bit-op consistency mismatch at bit_op spec {spec}")]
    LiftedEvalsBitOpMismatch { spec: usize },
    #[error("lookup argument failed: {0}")]
    Lookup(#[from] LookupError),
    #[error("booleanity check failed: {0}")]
    Booleanity(zinc_piop::lookup::booleanity::BooleanityError<F>),
    #[error("public-trace consistency check failed: {0}")]
    PublicConsistency(String),
    #[error("public-column structural check failed: {0}")]
    PublicStructure(zinc_uair::PublicStructureError),
    #[error("shifted bit-slice evaluation failed: {0}")]
    ShiftedBitSliceEval(zinc_poly::EvaluationError),
    #[error("PCS error: {0}")]
    Pcs(#[from] ZipError),
    #[error("PCS verification failed at column {0}: {1}")]
    PcsVerification(usize, ZipError),
}

//
// Helper functions
//

/// Absorb public column entries into the Fiat-Shamir transcript.
///
/// Each entry is serialized via `ConstTranscribable::write_transcription_bytes`
/// and absorbed. This must be called in the same order by both prover and
/// verifier, after commitments and before the random prime draw.
fn absorb_public_columns<T: ConstTranscribable>(
    transcript: &mut impl Transcript,
    cols: &[DenseMultilinearExtension<T>],
) {
    let mut buf = vec![0u8; T::NUM_BYTES];
    for col in cols {
        for entry in col.iter() {
            entry.write_transcription_bytes_exact(&mut buf);
            transcript.absorb_slice(&buf);
        }
    }
}

/// Compute per-column lifted MLE evaluations at `point`.
///
/// For each column j, returns `\sum_b eq(b, point) * v_j(b)` as a polynomial
/// in `F_q[X]` (coefficient-wise MLE evaluation). Dispatches on the trace
/// layout internally.
///
/// Binary columns exploit the 0/1 structure for conditional additions only.
/// The `eq(point, *)` table is built once and reused across all columns.
#[allow(clippy::arithmetic_side_effects)]
fn compute_lifted_evals<F: PrimeField, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &ProjectedTrace<F>,
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>> {
    compute_lifted_evals_capped::<F, D>(point, trace_bin_poly, projected_trace, field_cfg, None)
}

/// Like [`compute_lifted_evals`] but the non-binary section is capped at
/// `non_binary_cap` entries (counted from the start of the non-binary
/// region). Use `None` for full compute (matches `compute_lifted_evals`).
///
/// Use case: int-fold provers compute the int section separately via
/// [`compute_int_fold_lifted_evals`] / [`compute_int_fold_4x_lifted_evals`]
/// (returning 2/4-coeff bar_us), so computing the standard 1-coeff int
/// section here is wasted work. Pass `Some(num_total_arb_cols)` to stop
/// the non-binary iter right after arbitrary cols.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_lifted_evals_capped<F: PrimeField, const D: usize>(
    point: &[F],
    trace_bin_poly: &[DenseMultilinearExtension<BinaryPoly<D>>],
    projected_trace: &ProjectedTrace<F>,
    field_cfg: &F::Config,
    non_binary_cap: Option<usize>,
) -> Vec<DynamicPolynomialF<F>> {
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_lifted_evals: eq table build failed");

    let n_bin = trace_bin_poly.len();
    let zero = F::zero_with_cfg(field_cfg);

    // Binary columns: exploit 0/1 structure for conditional additions.
    // Pack each entry's up-to-64 boolean coefficients into a u64 so we
    // can (a) skip entries that are identically zero, and (b) walk only
    // the SET bits via `trailing_zeros` + Brian Kernighan's clear-lowest
    // instead of branching on every slot.
    debug_assert!(
        D <= 64,
        "compute_lifted_evals: bitmask packing assumes D <= 64"
    );
    let mut result: Vec<DynamicPolynomialF<F>> = cfg_iter!(trace_bin_poly)
        .map(|col| {
            let mut coeffs = vec![zero.clone(); D];
            for (b, entry) in col.iter().enumerate() {
                let mut bits: u64 = 0;
                for (l, coeff) in entry.iter().enumerate().take(D) {
                    if coeff.into_inner() {
                        bits |= 1u64 << l;
                    }
                }
                if bits == 0 {
                    continue;
                }
                let eq_b = &eq_table[b];
                let mut remaining = bits;
                while remaining != 0 {
                    let l = remaining.trailing_zeros() as usize;
                    coeffs[l] += eq_b;
                    remaining &= remaining - 1;
                }
            }
            DynamicPolynomialF::new_trimmed(coeffs)
        })
        .collect();

    // Non-binary columns: coefficient-wise eq-weighted sum.
    fn weighted_eq_sum<'a, F2: PrimeField + 'a>(
        col: impl Iterator<Item = &'a DynamicPolynomialF<F2>> + Clone,
        eq_table: &[F2],
        zero: &F2,
    ) -> DynamicPolynomialF<F2> {
        let num_coeffs = col.clone().map(|e| e.coeffs.len()).max().unwrap_or(0);
        let mut coeffs = vec![zero.clone(); num_coeffs];
        for (b, entry) in col.enumerate() {
            for (l, coeff) in entry.coeffs.iter().enumerate() {
                let mut term = eq_table[b].clone();
                term *= coeff;
                coeffs[l] += &term;
            }
        }
        DynamicPolynomialF::new_trimmed(coeffs)
    }

    match projected_trace {
        ProjectedTrace::RowMajor(t) => {
            let num_cols = t.first().map(|r| r.len()).unwrap_or(0);
            let non_binary_end = match non_binary_cap {
                Some(cap) => (n_bin + cap).min(num_cols),
                None => num_cols,
            };
            cfg_extend!(
                result,
                cfg_into_iter!(n_bin..non_binary_end).map(|col_idx| weighted_eq_sum(
                    t.iter().map(|row| &row[col_idx]),
                    &eq_table,
                    &zero,
                ))
            );
        }
        ProjectedTrace::ColumnMajor(t) => {
            let non_binary_end = match non_binary_cap {
                Some(cap) => (n_bin + cap).min(t.len()),
                None => t.len(),
            };
            cfg_extend!(
                result,
                cfg_iter!(t[n_bin..non_binary_end]).map(|col_mle| weighted_eq_sum(
                    col_mle.iter(),
                    &eq_table,
                    &zero,
                ))
            );
        }
    }

    result
}

/// 1× int-fold lifted-eval helper. Produces 2-coeff bar_us per int
/// column: `[lo_eval, hi_eval]` where each coeff is the MLE eval at
/// `point` of the corresponding 128-bit half.
///
/// `lo` is zero-extended into `Int<HALF_H>` (always non-negative);
/// `hi` is the signed arithmetic shift (sign-preserving). The original
/// column's lifted eval at `point` is recoverable as
/// `coeffs[0] + 2^128 · coeffs[1]` in F.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_int_fold_lifted_evals<F, const H: usize, const HALF_H: usize>(
    point: &[F],
    int_trace: &[DenseMultilinearExtension<crypto_primitives::crypto_bigint_int::Int<H>>],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField + for<'a> FromWithConfig<&'a crypto_primitives::crypto_bigint_int::Int<HALF_H>>,
{
    use crypto_primitives::crypto_bigint_int::Int;
    assert!(HALF_H >= 2);
    assert!(H >= HALF_H);
    const LO_LIMBS: usize = 2;
    let shift: u32 = (LO_LIMBS * 64) as u32;
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_int_fold_lifted_evals: eq table build failed");
    let zero = F::zero_with_cfg(field_cfg);

    cfg_iter!(int_trace)
        .map(|col| {
            let mut lo_eval = zero.clone();
            let mut hi_eval = zero.clone();
            for (b, entry) in col.iter().enumerate() {
                let v_words = entry.as_uint().to_words();
                let mut lo_words = [0u64; HALF_H];
                lo_words[0] = v_words[0];
                lo_words[1] = v_words[1];
                let lo: Int<HALF_H> = Int::from_words(lo_words);
                let hi: Int<HALF_H> = (*entry >> shift).resize();

                let mut term_lo = F::from_with_cfg(&lo, field_cfg);
                term_lo *= &eq_table[b];
                lo_eval += &term_lo;
                let mut term_hi = F::from_with_cfg(&hi, field_cfg);
                term_hi *= &eq_table[b];
                hi_eval += &term_hi;
            }
            DynamicPolynomialF::new_trimmed(vec![lo_eval, hi_eval])
        })
        .collect()
}

/// 4× int-fold lifted-eval helper. Produces 4-coeff bar_us per int
/// column: `[q0_eval, q1_eval, q2_eval, q3_eval]` where each coeff is
/// the MLE eval at `point` of the corresponding 64-bit quarter.
///
/// `q_0, q_1, q_2` are zero-extended single source limbs (always
/// non-negative); `q_3` is `(v >> 192).resize()` (signed). The original
/// column's lifted eval at `point` is recoverable as
/// `c[0] + 2^64·c[1] + 2^128·c[2] + 2^192·c[3]` in F.
///
/// Two fast-paths shave most of the per-cell cost on traces with
/// many small/zero int values (SHA carries):
/// 1. **Zero-quarter skip**: if `words[i] == 0` (and `q_3` is zero), skip the
///    `F::from_with_cfg + mul + add` for that quarter entirely. For typical SHA
///    carry columns where `|v| < 2^64`, this elides 3 of 4 monty-muls per row.
/// 2. **u64 fast-lift**: lift `q_0..q_2` via `F::from_with_cfg(u64)` rather
///    than building an `Int<Q>` and going through the signed `Int → F` path
///    (which calls `is_negative`, `abs`, and `resize`).
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_int_fold_4x_lifted_evals<F, const H: usize, const Q: usize>(
    point: &[F],
    int_trace: &[DenseMultilinearExtension<crypto_primitives::crypto_bigint_int::Int<H>>],
    field_cfg: &F::Config,
) -> Vec<DynamicPolynomialF<F>>
where
    F: PrimeField
        + FromWithConfig<u64>
        + for<'a> FromWithConfig<&'a crypto_primitives::crypto_bigint_int::Int<Q>>,
{
    use crypto_primitives::crypto_bigint_int::Int;
    assert!(Q >= 2);
    assert!(H >= 4);
    let eq_table = zinc_poly::utils::build_eq_x_r_vec(point, field_cfg)
        .expect("compute_int_fold_4x_lifted_evals: eq table build failed");
    let zero = F::zero_with_cfg(field_cfg);

    cfg_iter!(int_trace)
        .map(|col| {
            let mut q0_eval = zero.clone();
            let mut q1_eval = zero.clone();
            let mut q2_eval = zero.clone();
            let mut q3_eval = zero.clone();
            for (b, entry) in col.iter().enumerate() {
                let words = entry.as_uint().to_words();
                let eq_b = &eq_table[b];

                // q_0..q_2: unsigned single-limb lift via u64 fast-path,
                // skipping the multiply when the limb is zero.
                if words[0] != 0 {
                    let mut t = F::from_with_cfg(words[0], field_cfg);
                    t *= eq_b;
                    q0_eval += &t;
                }
                if words[1] != 0 {
                    let mut t = F::from_with_cfg(words[1], field_cfg);
                    t *= eq_b;
                    q1_eval += &t;
                }
                if words[2] != 0 {
                    let mut t = F::from_with_cfg(words[2], field_cfg);
                    t *= eq_b;
                    q2_eval += &t;
                }

                // q_3: signed arithmetic shift; check zero on both
                // limbs of the resulting Int<2> bit pattern. For
                // non-negative source values < 2^192, both limbs are
                // zero and we skip the multiply entirely.
                let q3_words = (*entry >> 192_u32).as_uint().to_words();
                let q3_lo = q3_words[0];
                let q3_hi = if Q >= 2 { q3_words[1] } else { 0 };
                if q3_lo != 0 || q3_hi != 0 {
                    let q3_v: Int<Q> = (*entry >> 192_u32).resize();
                    let mut t = F::from_with_cfg(&q3_v, field_cfg);
                    t *= eq_b;
                    q3_eval += &t;
                }
            }
            DynamicPolynomialF::new_trimmed(vec![q0_eval, q1_eval, q2_eval, q3_eval])
        })
        .collect()
}

/// Project a DensePolynomial scalar to DynamicPolynomialF by projecting each
/// coefficient via \phi_q.
pub fn project_scalar_fn<R, F, const D: usize>(
    scalar: &DensePolynomial<R, D>,
    field_cfg: &F::Config,
) -> DynamicPolynomialF<F>
where
    F: PrimeField + for<'a> FromWithConfig<&'a R>,
{
    scalar
        .iter()
        .map(|coeff| F::from_with_cfg(coeff, field_cfg))
        .collect()
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    use crypto_bigint::{NonZero, U64, Uint as CbUint};
    use crypto_primitives::{Field, crypto_bigint_int::Int, crypto_bigint_uint::Uint};
    use rand::{SeedableRng, rng, rngs::StdRng};
    use zinc_piop::{
        combined_poly_resolver::CombinedPolyResolverError, multipoint_eval::MultipointEvalError,
    };
    use zinc_poly::univariate::{
        binary::{BinaryPoly, BinaryPolyInnerProduct},
        dense::DensePolyInnerProduct,
    };
    use zinc_primality::MillerRabin;
    use zinc_test_uair::{
        BigLinearUair, BigLinearUairWithPublicInput, BinaryDecompositionUair, BitOpRotUair,
        EC_FP_INT_LIMBS, GenerateRandomTrace, PrivateEcdsaScalarsUair,
        PrivateScalarRangePackedUair, PrivateScalarRangeUair, Sha256CompressionSliceUair,
        Sha256Ideal, ShaEcdsaUair, TestUairMixedDegrees, TestUairMixedShifts,
        TestUairNoMultiplication, TestUairSimpleMultiplication,
        ecdsa::{
            EcdsaResultBindingError, EcdsaResultBranch, SECP256K1_N_UINT, decode_canonical_final_x,
            is_secp256k1_affine_point,
        },
        ecdsa_doubling::{SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT},
        private_ecdsa_scalars::{
            AUX_Q1_START as PRIVATE_ECDSA_Q1_START,
            IDENTITY_FINAL_ROW as PRIVATE_ECDSA_IDENTITY_ROW, LIMB_BITS as PRIVATE_ECDSA_LIMB_BITS,
            build_private_ecdsa_scalar_trace, cols as private_ecdsa_cols,
            derive_private_ecdsa_scalars, verify_private_ecdsa_public_polynomials,
        },
        private_scalar::{
            build_private_scalar_range_packed_trace, build_private_scalar_range_trace,
            cols as private_scalar_cols, packed_cols as private_scalar_packed_cols,
        },
        private_sha_ecdsa::{
            PrivateShaEcdsaUair, build_private_signature_trace, cols as private_sha_ecdsa_cols,
            verify_private_signature_application_binding,
        },
        sha_ecdsa::{
            ShaEcdsaApplicationBindingError, ShaEcdsaScalarBindingError,
            build_trace_from_ecdsa_scalars, build_trace_from_message_and_signature,
            build_trace_from_sha_and_signature, cols as sha_ecdsa_cols,
            derive_ecdsa_verification_scalars, extract_sha256_output,
            verify_sha_ecdsa_application_binding, verify_sha_ecdsa_h8_application_binding,
            verify_sha_ecdsa_result_binding,
        },
        sha256,
    };
    use zinc_uair::{
        UairTrace,
        ideal::{DegreeOneIdeal, rotation::RotationIdeal},
        ideal_collector::IdealOrZero,
    };
    use zinc_utils::{
        CHECKED,
        field::runtime_monty::Fp,
        from_ref::FromRef,
        inner_product::{MBSInnerProduct, ScalarProduct},
        projectable_to_field::ProjectableToField,
    };
    use zip_plus::{
        code::{
            iprs::{IprsCode, PnttConfigF65537},
            raa::{RaaCode, RaaConfig},
        },
        pcs::structs::{ZipPlus, ZipPlusParams},
        pcs_transcript::PcsProverTranscript,
    };

    const INT_LIMBS: usize = U64::LIMBS;
    // `fixed-prime` branch: 256-bit field modulus (4 × u64 limbs) so the
    // hardcoded secp256k1 base prime fits in `Fmod = Uint<FIELD_LIMBS>`.
    const FIELD_LIMBS: usize = U64::LIMBS * 4;
    const DEGREE_PLUS_ONE: usize = 32;
    const H6_PROOF_FIXTURE_SEED: u64 = 0x4836_5eed;

    struct ShaEcdsaApplicationStatement<'statement, 'trace> {
        signature_r: &'statement [u8],
        public_trace: &'statement UairTrace<'trace, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
    }

    struct ShaEcdsaH7ApplicationStatement<'statement, 'trace> {
        signature_r: &'statement [u8],
        signature_s: &'statement [u8],
        public_trace: &'statement UairTrace<'trace, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
    }

    struct ShaEcdsaH8ApplicationStatement<'statement, 'trace> {
        message: &'statement [u8],
        signature_r: &'statement [u8],
        signature_s: &'statement [u8],
        q_x: &'statement [u8],
        q_y: &'statement [u8],
        public_trace: &'statement UairTrace<'trace, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
    }

    fn uint_to_be_bytes(value: &CbUint<EC_FP_INT_LIMBS>) -> [u8; 32] {
        let mut bytes = [0_u8; 32];
        for (index, word) in value.as_words().iter().rev().enumerate() {
            bytes[index * 8..(index + 1) * 8].copy_from_slice(&word.to_be_bytes());
        }
        bytes
    }

    fn signature_r_for_public_trace(
        public_trace: &UairTrace<'_, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
    ) -> [u8; 32] {
        let final_x = decode_canonical_final_x(
            &public_trace.int[sha_ecdsa_cols::ECDSA_PA_R_X][zinc_test_uair::sha_ecdsa::FINAL_ROW],
        )
        .expect("generated PA_R_X must use the canonical centered encoding");
        let signature_r = if final_x >= SECP256K1_N_UINT {
            final_x.wrapping_sub(&SECP256K1_N_UINT)
        } else {
            final_x
        };
        assert_ne!(
            signature_r,
            CbUint::ZERO,
            "generated ECDSA r must be nonzero"
        );
        uint_to_be_bytes(&signature_r)
    }

    fn add_mod_order(
        left: &CbUint<EC_FP_INT_LIMBS>,
        right: &CbUint<EC_FP_INT_LIMBS>,
    ) -> CbUint<EC_FP_INT_LIMBS> {
        let left: CbUint<{ EC_FP_INT_LIMBS * 2 }> = left.resize();
        let right: CbUint<{ EC_FP_INT_LIMBS * 2 }> = right.resize();
        let sum = left.wrapping_add(&right);
        let order: CbUint<{ EC_FP_INT_LIMBS * 2 }> = SECP256K1_N_UINT.resize();
        let order = NonZero::new(order).expect("secp256k1 order is nonzero");
        let (_, remainder) = sum.div_rem_vartime(&order);
        remainder.resize()
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

    fn set_h8_scalar_selectors(
        public_trace: &mut UairTrace<'_, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
        u1: &CbUint<EC_FP_INT_LIMBS>,
        u2: &CbUint<EC_FP_INT_LIMBS>,
    ) {
        for row in 0..zinc_test_uair::ecdsa::NUM_SHAMIR_ROUNDS {
            let bit = zinc_test_uair::ecdsa::NUM_SHAMIR_ROUNDS - 1 - row;
            let word = bit / 64;
            let offset = bit % 64;
            let b1 = ((u1.as_words()[word] >> offset) & 1) == 1;
            let b2 = ((u2.as_words()[word] >> offset) & 1) == 1;
            public_trace.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_B1].evaluations[row] =
                ShaEcdsaInt::from(u32::from(b1));
            public_trace.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_B2].evaluations[row] =
                ShaEcdsaInt::from(u32::from(b2));
            public_trace.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_B1B2].evaluations[row] =
                ShaEcdsaInt::from(u32::from(b1 && b2));
            public_trace.int.to_mut()[sha_ecdsa_cols::ECDSA_S_ADD].evaluations[row] =
                ShaEcdsaInt::from(u32::from(b1 || b2));
        }
    }

    // Zip+ type parameters.

    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;

    /// Repetition factor for linear code, an inverse rate. Defaults to 4
    /// (rate 1/4); enabling the `iprs-rate-1-8` cargo feature switches
    /// every `IprsCode<..., REP, ...>` instance in this test module to
    /// inverse-rate 8 (rate 1/8), and `iprs-rate-1-16` switches to
    /// inverse-rate 16 (rate 1/16). `iprs-rate-1-16` takes precedence if
    /// both are enabled.
    const REP: usize = if cfg!(feature = "iprs-rate-1-16") {
        16
    } else if cfg!(feature = "iprs-rate-1-8") {
        8
    } else {
        4
    };

    /// Number of column openings the PCS performs. Tied to `REP`: rate 1/4
    /// uses 150 openings, rate 1/8 uses 100, rate 1/16 uses 75 (lower opening
    /// count is sound at the higher inverse rate because each column reveals
    /// more information about the codeword).
    const NUM_COL_OPENINGS_FOR_REP: usize = if cfg!(feature = "iprs-rate-1-16") {
        75
    } else if cfg!(feature = "iprs-rate-1-8") {
        100
    } else {
        150
    };

    // Value-sized field with the modulus installed once into `ProofSlot`
    // (drop-in for `MontyField<FIELD_LIMBS>`; see
    // utils/src/field/runtime_monty.rs). `F::make_cfg` (called via
    // `secp256k1_field_cfg`) installs the slot before any field arithmetic
    // runs.
    zinc_utils::define_modulus!(ProofSlot, FIELD_LIMBS);
    type F = Fp<ProofSlot, FIELD_LIMBS>;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypes {}
    impl ZipTypes for BinPolyZipTypes {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesIprs {}
    impl ZipTypes for ArbitraryPolyZipTypesIprs {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Arbitrary poly ZipTypes with wider codewords for RAA encoding.
    /// RAA accumulation grows the bit-width, so Cw needs more bits than Eval.
    #[derive(Debug, Clone)]
    pub struct ArbitraryPolyZipTypesRaa {}
    impl ZipTypes for ArbitraryPolyZipTypesRaa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<K>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal =
            DensePolyInnerProduct<i64, Self::Chal, Self::CombR, MBSInnerProduct, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    type ZtInt = i64;

    #[derive(Debug, Clone)]
    pub struct IntZipTypes {}
    impl ZipTypes for IntZipTypes {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = ZtInt;
        type Cw = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestZincTypesIprs;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesIprs {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[derive(Copy, Clone)]
    struct TestRaaConfig;
    impl RaaConfig for TestRaaConfig {
        const PERMUTE_IN_PLACE: bool = false;
        const CHECK_FOR_OVERFLOWS: bool = true;
    }

    #[derive(Clone, Debug)]
    struct TestZincTypesRaa;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestZincTypesRaa {
        type Int = i64;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypes;
        type ArbitraryZt = ArbitraryPolyZipTypesRaa;
        type IntZt = IntZipTypes;

        type BinaryLc = RaaCode<Self::BinaryZt, TestRaaConfig, REP>;
        type ArbitraryLc = RaaCode<Self::ArbitraryZt, TestRaaConfig, REP>;
        type IntLc = RaaCode<Self::IntZt, TestRaaConfig, REP>;
    }

    /// Use row size equal to poly size, resulting in flat single-row matrices
    fn make_iprs<Zt: ZipTypes>(num_vars: usize) -> IprsCode<Zt, PnttConfigF65537, REP, CHECKED> {
        let poly_size = 1 << num_vars;
        IprsCode::new_with_optimal_depth(poly_size).unwrap()
    }

    /// Set up Zip+ PCS parameters for a given number of MLE variables.
    #[allow(clippy::type_complexity)]
    fn setup_pp<Zt>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
    ) -> (
        ZipPlusParams<Zt::BinaryZt, Zt::BinaryLc>,
        ZipPlusParams<Zt::ArbitraryZt, Zt::ArbitraryLc>,
        ZipPlusParams<Zt::IntZt, Zt::IntLc>,
    )
    where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
    {
        let poly_size = 1 << num_vars;
        (
            ZipPlus::<Zt::BinaryZt, Zt::BinaryLc>::setup(poly_size, linear_codes.0),
            ZipPlus::<Zt::ArbitraryZt, Zt::ArbitraryLc>::setup(poly_size, linear_codes.1),
            ZipPlus::<Zt::IntZt, Zt::IntLc>::setup(poly_size, linear_codes.2),
        )
    }

    macro_rules! default_project_ideal {
        () => {
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg))
        };
    }

    #[allow(clippy::result_large_err)]
    fn do_test<Zt, U>(
        num_vars: usize,
        linear_codes: (Zt::BinaryLc, Zt::ArbitraryLc, Zt::IntLc),
        project_ideal: impl Fn(
            &IdealOrZero<U::Ideal>,
            &<F as PrimeField>::Config,
        ) -> IdealOrZero<DegreeOneIdeal<F>>
        + Copy,
        tamper: impl Fn(&mut Proof<F>),
        check_verification: impl Fn(Result<(), ProtocolError<F, IdealOrZero<DegreeOneIdeal<F>>>>),
    ) where
        Zt: ZincTypes<DEGREE_PLUS_ONE>,
        Zt::Int: num_traits::Zero + num_traits::One,
        <Zt::BinaryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Eval: ProjectableToField<F>,
        <Zt::ArbitraryZt as ZipTypes>::Cw: ProjectableToField<F>,
        <Zt::IntZt as ZipTypes>::Cw: ProjectableToField<F>,
        U: Uair<Scalar = DensePolynomial<Zt::Int, DEGREE_PLUS_ONE>>
            + GenerateRandomTrace<DEGREE_PLUS_ONE, PolyCoeff = Zt::Int, Int = Zt::Int>
            + 'static,
        F: for<'a> FromWithConfig<&'a Zt::Int>
            + for<'a> FromWithConfig<&'a <Zt::BinaryZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a <Zt::ArbitraryZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a <Zt::IntZt as ZipTypes>::CombR>
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>,
        <F as Field>::Inner: FromRef<Zt::Fmod>,
        <F as Field>::Modulus: FromRef<Zt::Fmod>,
    {
        let mut rng = rng();
        let pp = setup_pp::<Zt>(num_vars, linear_codes);

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = U::signature();
        let public_trace = trace.public(&sig);

        macro_rules! run_protocol {
            ($mle_first:ident) => {
                let mut proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<
                    { $mle_first },
                    CHECKED,
                >(&pp, &trace, num_vars, project_scalar_fn)
                .expect("Prover failed");

                // Checking that the proof can be properly serialized and deserialized
                let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
                transcript.write(&proof).expect("Failed to serialize proof");
                let mut transcript = transcript.into_verification_transcript();
                let proof_2 = transcript
                    .read()
                    .expect("Failed to deserialize proof after serialization");
                assert_eq!(proof, proof_2);

                tamper(&mut proof);

                let verification_result =
                    ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                        &pp,
                        proof,
                        &public_trace,
                        num_vars,
                        project_scalar_fn,
                        project_ideal,
                    );
                check_verification(verification_result);
            };
        }

        run_protocol!(false);

        // `MLE_FIRST = true` is now safe for any UAIR: it dispatches at
        // runtime to MLE-first (all-linear), Combined (all-non-linear), or
        // Hybrid (mixed). Always exercise it.
        run_protocol!(true);
    }

    /// End-to-end test: TestUairNoMultiplication.
    ///
    /// UAIR constraint: a + b - c \in (X - 2)
    /// (one constraint, no polynomial multiplication, ideal = <X - 2>).
    #[test]
    fn test_e2e_no_multiplication() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairNoMultiplication<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: TestUairSimpleMultiplication.
    ///
    /// UAIR constraints (3 total, no ideals):
    ///   up[0] * up[1] = down[0]
    ///   up[1] * up[2] = down[1]
    ///   up[0] * up[2] = down[2]
    ///
    /// Uses RAA code with small num_vars (2) because chained polynomial
    /// multiplication causes exponential growth in both degree and coefficient
    /// magnitude. With num_vars=2 (4 rows), max degree=6 and max coefficient
    /// ~= 127^8 ~= 2^56, which fits in i64.
    #[test]
    fn test_e2e_simple_multiplication() {
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairSimpleMultiplication<ZtInt>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: TestUairMixedDegrees.
    ///
    /// Two non-zero-ideal `assert_in_ideal` constraints — one linear
    /// (degree 1), one quadratic (degree 2). Exercises the hybrid
    /// ideal-check dispatch (`prove_hybrid`), which routes the linear
    /// constraint through the MLE-first lane and the quadratic constraint
    /// through the combined-poly lane, merging the per-constraint values
    /// into a single proof. Honest witness is the all-zero trace, which
    /// trivially satisfies both constraints.
    #[test]
    fn test_e2e_mixed_degrees() {
        // Use TestZincTypesRaa because the quadratic constraint with
        // arbitrary_poly column multiplication needs an RAA-style code.
        let num_vars = 2;
        do_test::<TestZincTypesRaa, TestUairMixedDegrees<ZtInt>>(
            num_vars,
            (
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
                RaaCode::new(num_vars),
            ),
            |ideal, field_cfg| ideal.map(|i| DegreeOneIdeal::from_with_cfg(i, field_cfg)),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: TestUairMixedShifts.
    ///
    /// Uses mixed shift amounts (col a: shift 1, col b: shift 2).
    /// Constraints: a[i+1] = a[i] + b[i], c[i] = b[i+2].
    #[test]
    fn test_e2e_mixed_shifts() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, TestUairMixedShifts<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BinaryDecompositionUair.
    ///
    /// Uses binary_poly (1 col) and int (1 col) trace types.
    /// UAIR constraint: binary_poly[0] - int[0] \in <X - 2>
    #[test]
    fn test_e2e_binary_decomposition() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BinaryDecompositionUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end proof for the maintained algebraic private-scalar range
    /// construction. The two private bit columns use the protocol's native
    /// Booleanity argument; shifted algebraic constraints bind the prefix
    /// states.
    #[test]
    fn test_e2e_private_scalar_range() {
        let num_vars = 9;
        do_test::<TestZincTypesIprs, PrivateScalarRangeUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.expect("verifier rejected canonical private scalars"),
        );
    }

    /// End-to-end proof for the column-reduced three-state comparator.
    #[test]
    fn test_e2e_private_scalar_range_packed() {
        let num_vars = 9;
        do_test::<TestZincTypesIprs, PrivateScalarRangePackedUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            |_| {},
            |res| res.expect("verifier rejected packed canonical private scalars"),
        );
    }

    /// Deterministic byte receipt for the standalone comparator candidate.
    /// Run with `iprs-rate-1-8` to match the frozen L3C rate/query profile.
    #[test]
    fn print_private_scalar_range_proof_bytes() {
        let num_vars = 9;
        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let trace = build_private_scalar_range_trace::<ZtInt>(
            num_vars,
            CbUint::ONE,
            SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE),
        );
        let public_trace = trace.public(&PrivateScalarRangeUair::<ZtInt>::signature());
        let proof = ZincPlusPiop::<
            TestZincTypesIprs,
            PrivateScalarRangeUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
        >::prove::<false, CHECKED>(&pp, &trace, num_vars, project_scalar_fn)
        .expect("private-scalar prover failed");
        let component_bytes = proof.get_num_bytes();
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript
            .write(&proof)
            .expect("proof serialization failed");
        let serialized = transcript.stream.get_ref();
        let zstd_bytes = zstd::encode_all(serialized.as_slice(), 3)
            .expect("proof compression failed")
            .len();
        println!(
            "PRIVATE_SCALAR_RANGE_PROOF serialized={} components={} zstd-3={}",
            serialized.len(),
            component_bytes,
            zstd_bytes,
        );

        ZincPlusPiop::<
            TestZincTypesIprs,
            PrivateScalarRangeUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
        >::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
        )
        .expect("private-scalar verifier rejected byte-receipt proof");
    }

    /// Deterministic byte receipt for the packed comparator candidate.
    #[test]
    fn print_private_scalar_range_packed_proof_bytes() {
        let num_vars = 9;
        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let trace = build_private_scalar_range_packed_trace::<ZtInt>(
            num_vars,
            CbUint::ONE,
            SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE),
        );
        let public_trace = trace.public(&PrivateScalarRangePackedUair::<ZtInt>::signature());
        let proof = ZincPlusPiop::<
            TestZincTypesIprs,
            PrivateScalarRangePackedUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
        >::prove::<false, CHECKED>(&pp, &trace, num_vars, project_scalar_fn)
        .expect("packed private-scalar prover failed");
        let component_bytes = proof.get_num_bytes();
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript
            .write(&proof)
            .expect("proof serialization failed");
        let serialized = transcript.stream.get_ref();
        let zstd_bytes = zstd::encode_all(serialized.as_slice(), 3)
            .expect("proof compression failed")
            .len();
        println!(
            "PRIVATE_SCALAR_RANGE_PACKED_PROOF serialized={} components={} zstd-3={}",
            serialized.len(),
            component_bytes,
            zstd_bytes,
        );

        ZincPlusPiop::<
            TestZincTypesIprs,
            PrivateScalarRangePackedUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
        >::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
        )
        .expect("packed private-scalar verifier rejected byte-receipt proof");
    }

    fn private_ecdsa_fixture() -> zinc_test_uair::PrivateEcdsaScalars {
        derive_private_ecdsa_scalars(
            CbUint::from_be_hex("E2D0FA68CFA8E68942FE9B038274B74A2F5E355D5C98CB9B6406E277B0F39188"),
            CbUint::from_be_hex("5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302"),
            CbUint::from_be_hex("7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568"),
        )
    }

    /// Full prove/verify and deterministic byte receipt for the exact
    /// base-2^26 private-scalar derivation candidate.
    #[test]
    fn test_e2e_private_ecdsa_scalars_and_print_bytes() {
        let num_vars = 9;
        let pp = setup_pp::<TestShaEcdsaZincTypes>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let witness = private_ecdsa_fixture();
        let trace = build_private_ecdsa_scalar_trace::<ShaEcdsaInt>(num_vars, &witness);
        let signature = PrivateEcdsaScalarsUair::<ShaEcdsaInt>::signature();
        let public_trace = trace.public(&signature);
        verify_private_ecdsa_public_polynomials(&public_trace, num_vars, &witness.e)
            .expect("typed public polynomial contract rejected honest fixture");

        let proof = ZincPlusPiop::<
            TestShaEcdsaZincTypes,
            PrivateEcdsaScalarsUair<ShaEcdsaInt>,
            F,
            DEGREE_PLUS_ONE,
        >::prove::<false, CHECKED>(&pp, &trace, num_vars, project_scalar_fn)
        .expect("private ECDSA scalar prover failed");
        let component_bytes = proof.get_num_bytes();
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript
            .write(&proof)
            .expect("proof serialization failed");
        let serialized = transcript.stream.get_ref();
        let zstd_bytes = zstd::encode_all(serialized.as_slice(), 3)
            .expect("proof compression failed")
            .len();
        println!(
            "PRIVATE_ECDSA_SCALARS_PROOF serialized={} components={} zstd-3={}",
            serialized.len(),
            component_bytes,
            zstd_bytes,
        );

        ZincPlusPiop::<
            TestShaEcdsaZincTypes,
            PrivateEcdsaScalarsUair<ShaEcdsaInt>,
            F,
            DEGREE_PLUS_ONE,
        >::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            default_project_ideal!(),
        )
        .expect("private ECDSA scalar verifier rejected honest proof");
    }

    #[test]
    fn private_ecdsa_scalar_proof_rejects_adversarial_mutations() {
        let num_vars = 9;
        let pp = setup_pp::<TestShaEcdsaZincTypes>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let witness = private_ecdsa_fixture();
        let signature = PrivateEcdsaScalarsUair::<ShaEcdsaInt>::signature();

        let reject = |trace: &UairTrace<'static, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
                      case: &str| {
            let public_trace = trace.public(&signature);
            verify_private_ecdsa_public_polynomials(&public_trace, num_vars, &witness.e)
                .unwrap_or_else(|error| panic!("{case}: public contract changed: {error}"));
            let proof =
                ZincPlusPiop::<
                    TestShaEcdsaZincTypes,
                    PrivateEcdsaScalarsUair<ShaEcdsaInt>,
                    F,
                    DEGREE_PLUS_ONE,
                >::prove::<false, CHECKED>(&pp, trace, num_vars, project_scalar_fn);
            if let Ok(proof) = proof {
                let result = ZincPlusPiop::<
                    TestShaEcdsaZincTypes,
                    PrivateEcdsaScalarsUair<ShaEcdsaInt>,
                    F,
                    DEGREE_PLUS_ONE,
                >::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    &public_trace,
                    num_vars,
                    project_scalar_fn,
                    default_project_ideal!(),
                );
                assert!(
                    result.is_err(),
                    "invalid private ECDSA trace was accepted: {case}"
                );
            }
        };

        let mut non_boolean = build_private_ecdsa_scalar_trace::<ShaEcdsaInt>(num_vars, &witness);
        non_boolean.int.to_mut()[private_ecdsa_cols::W_R_BIT].evaluations[128] =
            ShaEcdsaInt::from(2);
        reject(&non_boolean, "r bit=2");

        let mut bad_comparator =
            build_private_ecdsa_scalar_trace::<ShaEcdsaInt>(num_vars, &witness);
        bad_comparator.int.to_mut()[private_ecdsa_cols::W_S_LESS].evaluations[128] +=
            ShaEcdsaInt::from(1);
        reject(&bad_comparator, "s comparator-state mutation");

        let mut quotient_overflow =
            build_private_ecdsa_scalar_trace::<ShaEcdsaInt>(num_vars, &witness);
        let overflow = 1_u32 << PRIVATE_ECDSA_LIMB_BITS;
        quotient_overflow.int.to_mut()[private_ecdsa_cols::W_AUX].evaluations
            [PRIVATE_ECDSA_Q1_START] = ShaEcdsaInt::from(overflow);
        quotient_overflow.binary_poly.to_mut()[private_ecdsa_cols::B_AUX].evaluations
            [PRIVATE_ECDSA_Q1_START] = BinaryPoly::from(overflow);
        reject(&quotient_overflow, "2^26 quotient limb");

        let mut bad_identity = build_private_ecdsa_scalar_trace::<ShaEcdsaInt>(num_vars, &witness);
        bad_identity.arbitrary_poly.to_mut()[private_ecdsa_cols::W_Q1].evaluations
            [PRIVATE_ECDSA_IDENTITY_ROW]
            .coeffs[0] += ShaEcdsaInt::from(1);
        reject(&bad_identity, "q1 identity accumulator mutation");

        let mut opposite_lanes =
            build_private_ecdsa_scalar_trace::<ShaEcdsaInt>(num_vars, &witness);
        opposite_lanes.int.to_mut()[private_ecdsa_cols::W_R_LESS].evaluations[128] +=
            ShaEcdsaInt::from(1);
        opposite_lanes.int.to_mut()[private_ecdsa_cols::W_S_LESS].evaluations[128] -=
            ShaEcdsaInt::from(1);
        reject(&opposite_lanes, "opposite comparator-lane mutations");
    }

    fn private_scalar_proof_is_rejected(
        trace: &UairTrace<'static, ZtInt, ZtInt, DEGREE_PLUS_ONE>,
        num_vars: usize,
        case: &str,
    ) {
        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let public_trace = trace.public(&PrivateScalarRangeUair::<ZtInt>::signature());
        let proof = ZincPlusPiop::<
            TestZincTypesIprs,
            PrivateScalarRangeUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
        >::prove::<false, CHECKED>(&pp, trace, num_vars, project_scalar_fn);

        if let Ok(proof) = proof {
            let result = ZincPlusPiop::<
                TestZincTypesIprs,
                PrivateScalarRangeUair<ZtInt>,
                F,
                DEGREE_PLUS_ONE,
            >::verify::<_, CHECKED>(
                &pp,
                proof,
                &public_trace,
                num_vars,
                project_scalar_fn,
                |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            );
            assert!(
                result.is_err(),
                "invalid private-scalar trace was accepted: {case}"
            );
        }
    }

    fn private_scalar_packed_proof_is_rejected(
        trace: &UairTrace<'static, ZtInt, ZtInt, DEGREE_PLUS_ONE>,
        num_vars: usize,
        case: &str,
    ) {
        let pp = setup_pp::<TestZincTypesIprs>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );
        let public_trace = trace.public(&PrivateScalarRangePackedUair::<ZtInt>::signature());
        let proof = ZincPlusPiop::<
            TestZincTypesIprs,
            PrivateScalarRangePackedUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
        >::prove::<false, CHECKED>(&pp, trace, num_vars, project_scalar_fn);

        if let Ok(proof) = proof {
            let result = ZincPlusPiop::<
                TestZincTypesIprs,
                PrivateScalarRangePackedUair<ZtInt>,
                F,
                DEGREE_PLUS_ONE,
            >::verify::<_, CHECKED>(
                &pp,
                proof,
                &public_trace,
                num_vars,
                project_scalar_fn,
                |_ideal, _field_cfg| IdealOrZero::<DegreeOneIdeal<F>>::zero(),
            );
            assert!(
                result.is_err(),
                "invalid packed private-scalar trace was accepted: {case}"
            );
        }
    }

    #[test]
    fn private_scalar_range_rejects_boundaries_and_mutations() {
        let num_vars = 9;
        let one = CbUint::<EC_FP_INT_LIMBS>::ONE;
        let max = CbUint::<EC_FP_INT_LIMBS>::MAX;

        for (case, r, s) in [
            ("r=0", CbUint::ZERO, one),
            ("s=0", one, CbUint::ZERO),
            ("r=n", SECP256K1_N_UINT, one),
            ("s=n", one, SECP256K1_N_UINT),
            ("r=2^256-1", max, one),
        ] {
            let trace = build_private_scalar_range_trace::<ZtInt>(num_vars, r, s);
            private_scalar_proof_is_rejected(&trace, num_vars, case);
        }

        let mut non_boolean = build_private_scalar_range_trace::<ZtInt>(num_vars, one, one);
        non_boolean.int.to_mut()[private_scalar_cols::W_R_BIT].evaluations[255] = 2;
        private_scalar_proof_is_rejected(&non_boolean, num_vars, "r bit=2");

        let mut bad_less = build_private_scalar_range_trace::<ZtInt>(num_vars, one, one);
        bad_less.int.to_mut()[private_scalar_cols::W_R_LESS].evaluations[128] ^= 1;
        private_scalar_proof_is_rejected(&bad_less, num_vars, "r less-state mutation");

        let mut bad_seen = build_private_scalar_range_trace::<ZtInt>(num_vars, one, one);
        bad_seen.int.to_mut()[private_scalar_cols::W_S_SEEN].evaluations[255] ^= 1;
        private_scalar_proof_is_rejected(&bad_seen, num_vars, "s seen-state mutation");
    }

    #[test]
    fn private_scalar_range_packed_rejects_boundaries_and_mutations() {
        let num_vars = 9;
        let one = CbUint::<EC_FP_INT_LIMBS>::ONE;
        let max = CbUint::<EC_FP_INT_LIMBS>::MAX;

        for (case, r, s) in [
            ("r=0", CbUint::ZERO, one),
            ("s=0", one, CbUint::ZERO),
            ("r=n", SECP256K1_N_UINT, one),
            ("s=n", one, SECP256K1_N_UINT),
            ("r=2^256-1", max, one),
        ] {
            let trace = build_private_scalar_range_packed_trace::<ZtInt>(num_vars, r, s);
            private_scalar_packed_proof_is_rejected(&trace, num_vars, case);
        }

        let mut non_boolean = build_private_scalar_range_packed_trace::<ZtInt>(num_vars, one, one);
        non_boolean.int.to_mut()[private_scalar_packed_cols::W_R_BIT].evaluations[255] = 2;
        private_scalar_packed_proof_is_rejected(&non_boolean, num_vars, "r bit=2");

        let mut bad_state = build_private_scalar_range_packed_trace::<ZtInt>(num_vars, one, one);
        bad_state.int.to_mut()[private_scalar_packed_cols::W_R_STATE].evaluations[128] ^= 1;
        private_scalar_packed_proof_is_rejected(&bad_state, num_vars, "r state mutation");

        let mut bad_final = build_private_scalar_range_packed_trace::<ZtInt>(num_vars, one, one);
        bad_final.int.to_mut()[private_scalar_packed_cols::W_S_STATE].evaluations[256] = 1;
        private_scalar_packed_proof_is_rejected(&bad_final, num_vars, "s final state mutation");
    }

    /// End-to-end test: BigLinearUair.
    ///
    /// Uses 16 binary_poly cols and 1 int col.
    /// UAIR constraints:
    ///   sum(up.binary_poly[0..16]) - up.int[0] \in <X - 1>
    ///   down.binary_poly[0] - up.int[0] \in <X - 2>
    ///   up.binary_poly[i] - down.binary_poly[i] = 0, for i=1..15
    #[test]
    fn test_e2e_big_linear() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// End-to-end test: BigLinearUairWithPublicInput.
    ///
    /// Same as [`BigLinearUair`], but with the first few binary_poly columns as
    /// public inputs.
    #[test]
    fn test_e2e_big_linear_with_public_input() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    //
    // Negative tests for BigLinearUairWithPublicInput: verify that proof
    // tampering is detected.
    //

    #[test]
    fn test_big_linear_tamper_lifted_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.witness_lifted_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_up_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.resolver.up_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::ClaimValueDoesNotMatch { .. }
                    )
                ));
            },
        );
    }

    #[test]
    fn test_big_linear_tamper_down_evals() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.resolver.down_evals.swap(0, 1),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(
                        CombinedPolyResolverError::ClaimValueDoesNotMatch { .. }
                    )
                ));
            },
        );
    }

    // Tampering the commitment root causes the verifier to derive different
    // challenges from the Fiat–Shamir transcript. On the `fixed-prime`
    // branch the projecting prime `q` is hardcoded (not transcript-derived),
    // so prover and verifier still agree on `q` after the tamper; the first
    // observable divergence is at the combined-poly resolver, which catches
    // it as a sumcheck-sum mismatch.
    #[test]
    fn test_big_linear_tamper_commitment() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.commitments.0.root = Default::default(),
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::Resolver(CombinedPolyResolverError::WrongSumcheckSum { .. })
                ));
            },
        );
    }

    /// End-to-end test: BitOpRotUair (synthetic UAIR with one
    /// `BitOp::Rot(7)` virtual column).
    ///
    /// Two binary witness columns W (col 0) and V (col 1); witness sets
    /// V[i] = Rot(7)(W[i]). Constraint: V[i] − Rot(7)(W[i]) ∈ <X − 2>.
    /// Exercises the bit-op virtual column path end-to-end: CPR
    /// materialises an extra down MLE for the bit-op, the prover
    /// publishes a `bit_op_down_evals` entry, and the verifier checks
    /// it in Step 4.5 against ψ(rot_c(lifted_eval[col 0])).
    #[test]
    fn test_e2e_bit_op_rot() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |_| {},
            |res| res.unwrap(),
        );
    }

    /// Tampering the CPR-emitted `bit_op_down_evals` triggers the new
    /// `LiftedAtRStarBitOpMismatch` error at Step 4.5.
    ///
    /// We tamper the F_q[X]-lifted source eval at r* — easier to
    /// engineer than tampering `bit_op_down_evals` directly, because
    /// the latter participates in the CPR claim-value reconstruction
    /// (so the verifier rejects earlier with `ClaimValueDoesNotMatch`).
    /// Tampering the source's `lifted_evals_at_rstar` makes the source's
    /// up-eval ψ-projection still match `cpr_subclaim.up_evals[0]`
    /// only with extreme luck — but with a single coefficient swap the
    /// up-eval check fires first. To target the bit-op check, we
    /// instead permute coefficients of the source's lifted eval such
    /// that ψ_α projects to the same value (preserves dot product
    /// against `α^j`); a swap that holds the dot product fixed but
    /// changes `rot_c(·)` is not directly engineerable in a black-box
    /// test. As a robust proxy: tamper `bit_op_down_evals[0]` and
    /// observe the verifier rejects (whatever the precise error
    /// variant). For this test we verify it does reject — we check for
    /// the bit-op mismatch when the ResolveCheckValue happens to pass,
    /// otherwise any rejection is acceptable.
    #[test]
    fn test_bit_op_rot_tamper_bit_op_down_eval() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                // Mutate the only bit_op_down_eval — verifier must
                // reject (CPR claim-value reconstruction will catch it
                // first as a `ClaimValueDoesNotMatch`).
                if let Some(ev) = proof.resolver.bit_op_down_evals.first_mut() {
                    *ev = ev.clone() + ev.clone();
                }
            },
            |res| {
                assert!(res.is_err());
            },
        );
    }

    /// Tamper a coefficient of the source's witness lifted eval at
    /// r_0 (W_W's slot in `proof.witness_lifted_evals`). The swap
    /// changes `rot_c(·)` (so the bit-op slot's derived `open_eval`
    /// no longer matches mp_eval's expectation) and at the same time
    /// changes the source's own `open_eval`. The verifier rejects via
    /// mp_eval's `ClaimMismatch` (whichever slot trips the equation
    /// first — the per-slot identity is not separately surfaced).
    #[test]
    fn test_bit_op_rot_tamper_witness_lifted_source() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BitOpRotUair<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| {
                if let Some(p) = proof.witness_lifted_evals.first_mut() {
                    if p.coeffs.len() >= 2 {
                        p.coeffs.swap(0, 1);
                    }
                }
            },
            |res| {
                assert!(matches!(
                    res.unwrap_err(),
                    ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                        | ProtocolError::LiftedEvalsBitOpMismatch { .. }
                ));
            },
        );
    }

    //
    // SHA-ECDSA E2E + tampering tests for the new Step 4.5 layer.
    //
    // These pin the post-Commit-D behaviour:
    //   * lifted_evals_at_rstar up-half tamper rejected with
    //     `LiftedAtRStarUpMismatch`.
    //   * lifted_evals_at_rstar down-half tamper rejected with
    //     `LiftedAtRStarDownMismatch`.
    //   * SHA `W_W` source tamper rejected upstream of mp_eval (any
    //     `LiftedAtRStar*` variant).
    //   * No-tamper proof-shape pin: `lifted_evals_at_rstar.len()` and
    //     `bit_op_down_evals.len()` match the SHA-ECDSA signature.
    //

    type ShaEcdsaInt = Int<EC_FP_INT_LIMBS>;

    fn h7_application_fixture(
        seed: u64,
    ) -> (
        UairTrace<'static, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>,
        [u8; 32],
        [u8; 32],
        [u8; 32],
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let sha_trace = <Sha256CompressionSliceUair<ShaEcdsaInt> as GenerateRandomTrace<
            DEGREE_PLUS_ONE,
        >>::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng);
        let sha_public =
            sha_trace.public(&<Sha256CompressionSliceUair<ShaEcdsaInt> as Uair>::signature());
        let digest = extract_sha256_output(&sha_public).expect("generated SHA output must exist");
        let message_representative = CbUint::from_be_slice(&digest);
        let reduced_message = if message_representative >= SECP256K1_N_UINT {
            message_representative.wrapping_sub(&SECP256K1_N_UINT)
        } else {
            message_representative
        };
        let signature_r = SECP256K1_G_X_UINT;
        let signature_s = add_mod_order(&reduced_message, &signature_r);
        assert_ne!(
            signature_s,
            CbUint::ZERO,
            "fixture signature s must be nonzero"
        );
        let signature_r = uint_to_be_bytes(&signature_r);
        let signature_s = uint_to_be_bytes(&signature_s);
        let trace = build_trace_from_sha_and_signature(
            SHA_ECDSA_NUM_VARS,
            sha_trace,
            (SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT),
            &signature_r,
            &signature_s,
        )
        .expect("d = k = 1 fixture must build");
        (trace, signature_r, signature_s, digest)
    }

    /// Binary-poly Zip+ types tuned for the SHA-ECDSA cell type
    /// (`Int<EC_FP_INT_LIMBS>` / `Int<4>`, 256-bit). Mirrors
    /// `BinPolyZipTypes` but with a wider
    /// `CombR` to soak up SHA-ECDSA's per-row inner products.
    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesShaEcdsa {}
    impl ZipTypes for BinPolyZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Arbitrary-poly Zip+ types over `Int<EC_FP_INT_LIMBS>` cells.
    /// SHA-ECDSA itself has no arbitrary-poly columns; this is only
    /// here to satisfy the `ZincTypes` bundle.
    #[derive(Debug, Clone)]
    pub struct ArbPolyZipTypesShaEcdsa {}
    impl ZipTypes for ArbPolyZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = DensePolynomial<ShaEcdsaInt, DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<Int<6>, DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, DEGREE_PLUS_ONE>;
        type EvalDotChal = DensePolyInnerProduct<
            ShaEcdsaInt,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    /// Int Zip+ types over `Int<EC_FP_INT_LIMBS>` cells (the ECDSA
    /// ordinary-projective columns and the SHA integer carries).
    #[derive(Debug, Clone)]
    pub struct IntZipTypesShaEcdsa {}
    impl ZipTypes for IntZipTypesShaEcdsa {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = ShaEcdsaInt;
        type Cw = Int<6>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    // ── 4× int-fold variant of the ShaEcdsa types ──────────────────────
    //
    // For the `prove_folded_4x` round-trip test below.
    // Binary: `BinaryPoly<8>` quartered (matches existing
    // `BenchFoldedRealEcdsaZincTypes4x` from `protocol/benches/e2e.rs`).
    // Int: `Int<INT_QUARTER_LIMBS_TEST>` quartered.
    //
    // With ECDSA's centered representation `|v| < 2^255` (commit
    // `dde6f2e`), the 64-bit signed quarters `q_3 = (v >> 192).resize()`
    // satisfy `|q_3| < 2^63` so the upper quarter never overflows
    // `Int<2>`'s positive range; the lower three quarters are
    // bit-extracted single source limbs into `Int<2>` (top bit zeroed,
    // always non-negative). `Cw = Int<3>` gives one limb of headroom
    // for the encoder accumulator.

    const QUARTER_DEGREE_PLUS_ONE_TEST: usize = 8;
    const HALF_DEGREE_PLUS_ONE_TEST: usize = 16;
    const INT_QUARTER_LIMBS_TEST: usize = 2;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesShaEcdsaQuarter {}
    impl ZipTypes for BinPolyZipTypesShaEcdsaQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<QUARTER_DEGREE_PLUS_ONE_TEST>;
        type Cw = DensePolynomial<i64, QUARTER_DEGREE_PLUS_ONE_TEST>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<{ EC_FP_INT_LIMBS * 4 }>;
        type Comb = DensePolynomial<Self::CombR, QUARTER_DEGREE_PLUS_ONE_TEST>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, QUARTER_DEGREE_PLUS_ONE_TEST>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE_TEST,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Debug, Clone)]
    pub struct IntZipTypesShaEcdsaQuarter {}
    impl ZipTypes for IntZipTypesShaEcdsaQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = Int<INT_QUARTER_LIMBS_TEST>;
        type Cw = Int<{ INT_QUARTER_LIMBS_TEST + 1 }>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<6>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestShaEcdsaFolded4xZincTypes;

    impl
        IntFoldedZincTypes4x<
            DEGREE_PLUS_ONE,
            QUARTER_DEGREE_PLUS_ONE_TEST,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_TEST,
        > for TestShaEcdsaFolded4xZincTypes
    {
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesShaEcdsaQuarter;
        type ArbitraryZt = ArbPolyZipTypesShaEcdsa;
        type IntZt = IntZipTypesShaEcdsaQuarter;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    #[allow(clippy::type_complexity)]
    fn setup_folded_4x_pp_sha_ecdsa(
        num_vars: usize,
    ) -> (
        ZipPlusParams<
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::BinaryZt,
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::BinaryLc,
        >,
        ZipPlusParams<
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::ArbitraryZt,
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::ArbitraryLc,
        >,
        ZipPlusParams<
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::IntZt,
            <TestShaEcdsaFolded4xZincTypes as IntFoldedZincTypes4x<
                DEGREE_PLUS_ONE,
                QUARTER_DEGREE_PLUS_ONE_TEST,
                EC_FP_INT_LIMBS,
                INT_QUARTER_LIMBS_TEST,
            >>::IntLc,
        >,
    ) {
        let split4_size = 1 << (num_vars + 2);
        let normal_size = 1 << num_vars;
        (
            ZipPlus::setup(
                split4_size,
                IprsCode::new_with_optimal_depth(split4_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
            ZipPlus::setup(
                split4_size,
                IprsCode::new_with_optimal_depth(split4_size).unwrap(),
            ),
        )
    }

    /// `ZincTypes` bundle wiring SHA-ECDSA's `Int<4>` cells through
    /// IPRS-coded Zip+ commitments. Mirrors `RealEcdsaBenchZincTypes`
    /// from `protocol/benches/e2e.rs` (which already exercises this
    /// configuration in production benchmarks).
    #[derive(Clone, Debug)]
    struct TestShaEcdsaZincTypes;

    impl ZincTypes<DEGREE_PLUS_ONE> for TestShaEcdsaZincTypes {
        type Int = ShaEcdsaInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesShaEcdsa;
        type ArbitraryZt = ArbPolyZipTypesShaEcdsa;
        type IntZt = IntZipTypesShaEcdsa;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Project an `IdealOrZero<Sha256Ideal<ShaEcdsaInt>>` to
    /// `Sha256Ideal<F>` for the SHA-ECDSA verifier. Mirrors
    /// `sha256_real_project_ideal` in `protocol/benches/e2e.rs`.
    fn sha256_test_project_ideal(
        ideal: &IdealOrZero<Sha256Ideal<ShaEcdsaInt>>,
        field_cfg: &<F as PrimeField>::Config,
    ) -> Sha256Ideal<F> {
        match ideal {
            IdealOrZero::NonZero(Sha256Ideal::RotX2(r)) => {
                Sha256Ideal::RotX2(RotationIdeal::from_with_cfg(r, field_cfg))
            }
            IdealOrZero::NonZero(Sha256Ideal::RotXw1) => Sha256Ideal::RotXw1,
            IdealOrZero::Zero => {
                unreachable!("zero ideals are filtered before this closure runs")
            }
        }
    }

    /// Run a SHA-ECDSA round-trip end-to-end. Calls `tamper` on the
    /// generated proof before verification and feeds the resulting
    /// `Result` into `check_verification`. Patterned after `do_test`
    /// but specialised to the SHA-ECDSA UAIR / `Sha256Ideal` ideal
    /// type (which doesn't fit the `IdealOrZero<DegreeOneIdeal<F>>`
    /// signature `do_test` hard-codes).
    ///
    /// `MLE_FIRST` is kept `false` here to exercise the combined-projection
    /// path. The benchmark suite covers the separate MLE-first route.
    #[allow(clippy::result_large_err)]
    fn do_test_sha_ecdsa(
        num_vars: usize,
        tamper: impl Fn(&mut Proof<F>),
        tamper_public: impl Fn(&mut UairTrace<'_, ShaEcdsaInt, ShaEcdsaInt, DEGREE_PLUS_ONE>),
        check_verification: impl Fn(Result<(), ProtocolError<F, Sha256Ideal<F>>>),
    ) {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let mut rng = rng();
        let pp = setup_pp::<Zt>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
        );

        let trace = U::generate_random_trace(num_vars, &mut rng);

        let sig = <U as Uair>::signature();
        let mut public_trace = trace.public(&sig);

        let mut proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            num_vars,
            project_scalar_fn,
        )
        .expect("Prover failed");

        // Round-trip the proof through (de)serialisation as a sanity
        // check; mirrors `do_test`.
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript.write(&proof).expect("Failed to serialize proof");
        let mut transcript = transcript.into_verification_transcript();
        let proof_2 = transcript
            .read()
            .expect("Failed to deserialize proof after serialization");
        assert_eq!(proof, proof_2);

        tamper(&mut proof);
        tamper_public(&mut public_trace);

        let verification_result = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            sha256_test_project_ideal,
        );
        check_verification(verification_result);
    }

    /// `num_vars` for SHA-ECDSA tests. ECDSA's Shamir scalar
    /// multiplication needs `n_rows > 256`, so `num_vars >= 9`.
    const SHA_ECDSA_NUM_VARS: usize = 9;

    /// Tamper a coefficient of the SHA `W_W` slot in
    /// `proof.witness_lifted_evals` (the source of all six bit-op
    /// virtual columns). The verifier rejects via mp_eval's
    /// consistency check or `LiftedEvalsBitOpMismatch` (whichever
    /// trips first). Replaces the four pre-existing Step-4.5-specific
    /// tamper tests since Step 4.5 is gone.
    #[test]
    fn test_e2e_sha_ecdsa_tamper_witness_lifted_w() {
        do_test_sha_ecdsa(
            SHA_ECDSA_NUM_VARS,
            |proof| {
                // SHA-ECDSA's witness layout puts W_W at the same flat
                // index as in `cols::W_W` minus `NUM_BIN_PUB`. Easier
                // and equally diagnostic to tamper the first witness
                // slot instead — any of them feeds mp_eval.
                let p = &mut proof.witness_lifted_evals[0];
                assert!(
                    p.coeffs.len() >= 2,
                    "witness lifted eval polynomial has < 2 coefficients; cannot tamper",
                );
                p.coeffs.swap(0, 1);
            },
            |_| {},
            |res| {
                let err = res.unwrap_err();
                assert!(
                    matches!(
                        err,
                        ProtocolError::MultipointEval(MultipointEvalError::ClaimMismatch { .. })
                            | ProtocolError::LiftedEvalsBitOpMismatch { .. },
                    ),
                    "expected mp_eval ClaimMismatch or LiftedEvalsBitOpMismatch, got {err:?}",
                );
            },
        );
    }

    #[test]
    fn test_e2e_sha_ecdsa_rejects_non_boolean_public_selector() {
        do_test_sha_ecdsa(
            SHA_ECDSA_NUM_VARS,
            |_| {},
            |public_trace| {
                public_trace.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_B1].evaluations[17] =
                    Int::from(2_u32);
            },
            |res| {
                assert!(matches!(
                    res,
                    Err(ProtocolError::PublicStructure(
                        zinc_uair::PublicStructureError::WrongValue {
                            column: "PA_B1",
                            row: 17
                        }
                    ))
                ));
            },
        );
    }

    /// 4×-folded ShaEcdsa round-trip — binary AND int both quartered
    /// (BinaryPoly<8> / Int<2>) and committed under one Merkle tree
    /// via `MultiZip3`. Prints the serialized proof size.
    #[test]
    fn test_e2e_sha_ecdsa_folded_4x_round_trip() {
        use crate::{prover::prove_folded_4x, verifier::verify_folded_4x};

        type ZtF = TestShaEcdsaFolded4xZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let mut rng = StdRng::seed_from_u64(H6_PROOF_FIXTURE_SEED);
        let pp = setup_folded_4x_pp_sha_ecdsa(SHA_ECDSA_NUM_VARS);
        let trace = U::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng);
        let sig = <U as Uair>::signature();
        let public_trace = trace.public(&sig);

        let proof = prove_folded_4x::<
            ZtF,
            U,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE_TEST,
            QUARTER_DEGREE_PLUS_ONE_TEST,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_TEST,
            false,
            CHECKED,
        >(&pp, &trace, SHA_ECDSA_NUM_VARS, project_scalar_fn)
        .expect("4× int-fold prover failed");

        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript.write(&proof).expect("Failed to serialize proof");
        let serialized_len = transcript.stream.get_ref().len();
        println!(
            "4× folded ShaEcdsa proof size: {} bytes ({} KiB)",
            serialized_len,
            serialized_len.div_ceil(1024),
        );
        // The H9 soundness repair stores the RCB T4 cross term consumed by
        // the UAIR instead of the raw product. That changes the deterministic
        // transcript by 96 bytes without changing the declared proof shape.
        assert_eq!(
            serialized_len, 647_422,
            "fixed-seed folded-4x proof fixture changed",
        );
        let mut transcript = transcript.into_verification_transcript();
        let proof_2 = transcript.read().expect("Failed to deserialize proof");
        assert_eq!(proof, proof_2);

        verify_folded_4x::<
            ZtF,
            U,
            F,
            Sha256Ideal<F>,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE_TEST,
            QUARTER_DEGREE_PLUS_ONE_TEST,
            EC_FP_INT_LIMBS,
            INT_QUARTER_LIMBS_TEST,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("Verifier rejected an honest 4× folded ShaEcdsa proof");
    }

    /// No-tamper SHA-ECDSA round-trip + structural-shape pins. Prints
    /// the proof size so refactors that grow the proof are easy to
    /// catch.
    #[test]
    fn test_e2e_sha_ecdsa_proof_shape() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        // SHA standalone signature pins: NUM_BIN = 20 (9 public + 11
        // witness — 9 = PA_A, PA_E, 4× PA_OV_*, 2× PA_R_*_CORR, PA_M).
        // bit_op_specs = 6, virtual_binary_poly_cols = 3.
        let sha_sig = <Sha256CompressionSliceUair<ShaEcdsaInt> as Uair>::signature();
        assert_eq!(
            sha_sig.total_cols().num_binary_poly_cols(),
            20,
            "SHA-256 NUM_BIN drifted: expected 20 binary_poly columns",
        );
        assert_eq!(
            sha_sig.bit_op_specs().len(),
            11,
            "SHA-256 bit_op_specs.len() drifted: expected 11 (6 σ_0/σ_1 + 5 W_MU_PACKED ShiftRs)",
        );
        assert_eq!(
            sha_sig.virtual_binary_poly_cols().len(),
            3,
            "SHA-256 virtual_binary_poly_cols.len() drifted: expected 3 (B_1/B_2/B_3)",
        );

        // SHA-ECDSA composed signature (the actual UAIR exercised here).
        let sig = <U as Uair>::signature();
        let num_bit_op = sig.bit_op_specs().len();
        assert_eq!(
            sig.total_cols().num_binary_poly_cols(),
            20,
            "SHA-ECDSA NUM_BIN drifted: expected 20 binary_poly columns",
        );
        assert_eq!(
            sig.total_cols().num_int_cols(),
            42,
            "SHA-ECDSA NUM_INT drifted: expected 42 integer columns",
        );
        assert_eq!(
            sig.public_cols().num_int_cols(),
            24,
            "SHA-ECDSA NUM_INT_PUB drifted: expected 24 public integer columns",
        );
        assert_eq!(
            num_bit_op, 11,
            "SHA-ECDSA bit_op_specs.len() drifted: expected 11",
        );
        assert_eq!(
            sig.virtual_binary_poly_cols().len(),
            3,
            "SHA-ECDSA virtual_binary_poly_cols.len() drifted: expected 3",
        );

        // Round-trip a real proof and pin the post-rewrite proof size.
        let mut rng = StdRng::seed_from_u64(H6_PROOF_FIXTURE_SEED);
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let trace = U::generate_random_trace(SHA_ECDSA_NUM_VARS, &mut rng);
        let public_trace = trace.public(&sig);

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("Prover failed");

        assert_eq!(
            proof.resolver.bit_op_down_evals.len(),
            num_bit_op,
            "Proof.resolver.bit_op_down_evals.len() must equal bit_op_specs.len()",
        );

        let total_proof_bytes = proof.get_num_bytes();
        println!("total proof bytes: {total_proof_bytes}");
        // See the folded fixture above: the repaired T4 witness changes the
        // deterministic transcript, so the pre-H9 H8 receipt is superseded.
        assert_eq!(
            total_proof_bytes, 845_002,
            "fixed-seed flat proof fixture changed",
        );

        let signature_r = signature_r_for_public_trace(&public_trace);

        // The lower-level proof API has no signature-r argument. It proves the
        // UAIR relation but does not, by itself, claim ECDSA verification.
        ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof.clone(),
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("lower-level verifier rejected the honest UAIR proof");

        let mut wrong_r = [0_u8; 32];
        wrong_r[31] = if signature_r[31] == 1 && signature_r[..31].iter().all(|&b| b == 0) {
            2
        } else {
            1
        };
        let protocol_called = Cell::new(false);
        let wrong_r_result = application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaApplicationStatement {
                    signature_r: &wrong_r,
                    public_trace: &public_trace,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_result_binding(statement.signature_r, statement.public_trace)
                    .map(|_| ())
            },
            |proof, statement| {
                protocol_called.set(true);
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        );
        assert!(matches!(
            wrong_r_result,
            Err(application::ApplicationVerificationError::ResultBinding(
                EcdsaResultBindingError::ResultMismatch
            ))
        ));
        assert!(
            !protocol_called.get(),
            "binding failure should reject before PCS work"
        );

        let mut mutated_x_public = public_trace.clone();
        mutated_x_public.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_R_X].evaluations
            [zinc_test_uair::sha_ecdsa::FINAL_ROW] = Int::from(0_u32);
        let mutated_x_result = application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaApplicationStatement {
                    signature_r: &signature_r,
                    public_trace: &mutated_x_public,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_result_binding(statement.signature_r, statement.public_trace)
                    .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        );
        assert!(matches!(
            mutated_x_result,
            Err(application::ApplicationVerificationError::ResultBinding(
                EcdsaResultBindingError::ResultMismatch
            ))
        ));

        let mut mutated_z_inv_public = public_trace.clone();
        mutated_z_inv_public.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_Z_INV].evaluations
            [zinc_test_uair::sha_ecdsa::FINAL_ROW] = Int::from(0_u32);
        let mutated_z_inv_result = application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaApplicationStatement {
                    signature_r: &signature_r,
                    public_trace: &mutated_z_inv_public,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_result_binding(statement.signature_r, statement.public_trace)
                    .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        );
        assert!(matches!(
            mutated_z_inv_result,
            Err(application::ApplicationVerificationError::Protocol(_))
        ));

        application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaApplicationStatement {
                    signature_r: &signature_r,
                    public_trace: &public_trace,
                },
                proof,
            ),
            |statement| {
                verify_sha_ecdsa_result_binding(statement.signature_r, statement.public_trace)
                    .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        )
        .expect("typed application verifier rejected an honest SHA-ECDSA proof");
    }

    #[test]
    fn test_e2e_sha_ecdsa_plus_order_on_curve_round_trip() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        // The smallest nonzero plus-order representative with an affine point:
        // x = n + 2 and y^2 = x^3 + 7 mod p. With u1=0 and u2=1, the
        // Shamir result is exactly Q, so the application must take r+n.
        let q = (
            CbUint::from_be_hex("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364143"),
            CbUint::from_be_hex("36b1aa62eb77c1973025cbcbea9740eed8eacdab8772268b395064453269d1d3"),
        );
        assert!(is_secp256k1_affine_point(&q.0, &q.1));

        let mut rng = StdRng::seed_from_u64(H6_PROOF_FIXTURE_SEED ^ 1);
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let trace = build_trace_from_ecdsa_scalars::<ShaEcdsaInt, _>(
            SHA_ECDSA_NUM_VARS,
            &mut rng,
            q,
            CbUint::ZERO,
            CbUint::ONE,
        )
        .expect("plus-order public key must be valid");
        let public_trace = trace.public(&<U as Uair>::signature());
        let signature_r = uint_to_be_bytes(&CbUint::from_u64(2));
        assert_eq!(
            verify_sha_ecdsa_result_binding(&signature_r, &public_trace),
            Ok(EcdsaResultBranch::PlusOrder),
        );

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("plus-order prover failed");

        // Both x=r and x=r+n satisfy the typed SEC 1 result binding when
        // they are below p. The proof must still own which representative
        // the Shamir computation produced.
        let mut direct_representative = public_trace.clone();
        direct_representative.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_R_X].evaluations
            [zinc_test_uair::sha_ecdsa::FINAL_ROW] = Int::from(2_u32);
        assert_eq!(
            verify_sha_ecdsa_result_binding(&signature_r, &direct_representative),
            Ok(EcdsaResultBranch::Direct),
        );
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof.clone(),
                &direct_representative,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "the proof must reject the other binding-valid x representative",
        );

        application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaApplicationStatement {
                    signature_r: &signature_r,
                    public_trace: &public_trace,
                },
                proof,
            ),
            |statement| {
                verify_sha_ecdsa_result_binding(statement.signature_r, statement.public_trace)
                    .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        )
        .expect("typed application verifier rejected the on-curve plus-order proof");
    }

    #[test]
    fn test_e2e_sha_ecdsa_h7_scalar_binding_round_trip() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let (trace, signature_r, signature_s, digest) = h7_application_fixture(0x4837_7001);
        let public_trace = trace.public(&<U as Uair>::signature());
        let binding =
            verify_sha_ecdsa_application_binding(&signature_r, &signature_s, &public_trace)
                .expect("deterministic H7 fixture must satisfy both application postconditions");
        assert_eq!(binding.scalars.digest, digest);
        assert_eq!(binding.result_branch, EcdsaResultBranch::Direct);

        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("H7 fixture prover failed");
        println!("H7 flat proof bytes: {}", proof.get_num_bytes());

        application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaH7ApplicationStatement {
                    signature_r: &signature_r,
                    signature_s: &signature_s,
                    public_trace: &public_trace,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_application_binding(
                    statement.signature_r,
                    statement.signature_s,
                    statement.public_trace,
                )
                .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        )
        .expect("typed H7 application verifier rejected an honest proof");

        let signature_s_value = CbUint::from_be_slice(&signature_s);
        let wrong_s_value = if signature_s_value == SECP256K1_N_UINT.wrapping_sub(&CbUint::ONE) {
            signature_s_value.wrapping_sub(&CbUint::ONE)
        } else {
            signature_s_value.wrapping_add(&CbUint::ONE)
        };
        let wrong_s = uint_to_be_bytes(&wrong_s_value);
        let protocol_called = Cell::new(false);
        let wrong_s_result = application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaH7ApplicationStatement {
                    signature_r: &signature_r,
                    signature_s: &wrong_s,
                    public_trace: &public_trace,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_application_binding(
                    statement.signature_r,
                    statement.signature_s,
                    statement.public_trace,
                )
                .map(|_| ())
            },
            |_, _| {
                protocol_called.set(true);
                Ok::<_, ProtocolError<F, Sha256Ideal<F>>>(())
            },
        );
        assert!(matches!(
            wrong_s_result,
            Err(application::ApplicationVerificationError::ResultBinding(
                ShaEcdsaApplicationBindingError::Scalar(
                    ShaEcdsaScalarBindingError::ScalarBitMismatch { .. }
                )
            ))
        ));
        assert!(
            !protocol_called.get(),
            "wrong s must reject before PCS verification"
        );

        let mut disconnected_sha = public_trace.clone();
        disconnected_sha.int.to_mut()[sha_ecdsa_cols::SHA_S_INIT_PREFIX].evaluations[0] =
            Int::from(0_u32);
        let protocol_called = Cell::new(false);
        let disconnected_result = application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaH7ApplicationStatement {
                    signature_r: &signature_r,
                    signature_s: &signature_s,
                    public_trace: &disconnected_sha,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_application_binding(
                    statement.signature_r,
                    statement.signature_s,
                    statement.public_trace,
                )
                .map(|_| ())
            },
            |_, _| {
                protocol_called.set(true);
                Ok::<_, ProtocolError<F, Sha256Ideal<F>>>(())
            },
        );
        assert!(matches!(
            disconnected_result,
            Err(application::ApplicationVerificationError::ResultBinding(
                ShaEcdsaApplicationBindingError::Scalar(
                    ShaEcdsaScalarBindingError::ShaStructureMismatch {
                        column: "SHA_S_INIT_PREFIX",
                        row: 0,
                    }
                )
            ))
        ));
        assert!(
            !protocol_called.get(),
            "disconnected SHA selectors must reject before PCS verification"
        );

        // A second, internally consistent statement passes both public
        // postconditions. The first statement's proof must still reject it.
        let (other_trace, other_r, other_s, _) = h7_application_fixture(0x4837_7002);
        let other_public = other_trace.public(&<U as Uair>::signature());
        verify_sha_ecdsa_application_binding(&other_r, &other_s, &other_public)
            .expect("second application statement must be internally consistent");
        let mismatched_statement = application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaH7ApplicationStatement {
                    signature_r: &other_r,
                    signature_s: &other_s,
                    public_trace: &other_public,
                },
                proof,
            ),
            |statement| {
                verify_sha_ecdsa_application_binding(
                    statement.signature_r,
                    statement.signature_s,
                    statement.public_trace,
                )
                .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        );
        assert!(matches!(
            mismatched_statement,
            Err(application::ApplicationVerificationError::Protocol(_))
        ));
    }

    #[test]
    fn test_e2e_sha_ecdsa_h8_complete_statement_and_proof_ownership() {
        type Zt = TestShaEcdsaZincTypes;
        type U = ShaEcdsaUair<ShaEcdsaInt>;

        let message = h8_message();
        let signature_r = uint_to_be_bytes(&CbUint::from_be_hex(
            "5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302",
        ));
        let signature_s = uint_to_be_bytes(&CbUint::from_be_hex(
            "7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568",
        ));
        let q = (
            CbUint::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5"),
            CbUint::from_be_hex("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A"),
        );
        let q_x = uint_to_be_bytes(&q.0);
        let q_y = uint_to_be_bytes(&q.1);
        let trace = build_trace_from_message_and_signature::<ShaEcdsaInt>(
            SHA_ECDSA_NUM_VARS,
            &message,
            q,
            &signature_r,
            &signature_s,
        )
        .expect("frozen libsecp256k1 H8 statement must build");
        let sig = <U as Uair>::signature();
        let public_trace = trace.public(&sig);
        let binding = verify_sha_ecdsa_h8_application_binding(
            &message,
            &signature_r,
            &signature_s,
            &q_x,
            &q_y,
            &public_trace,
        )
        .expect("frozen H8 statement must satisfy every typed binding");
        assert_eq!(
            binding.scalars.digest,
            uint_to_be_bytes(&CbUint::from_be_hex(
                "E2D0FA71422BD33D792095FECF1D3CE6D13906E2627EE58C11E8C56CA0039188",
            )),
        );
        assert_eq!(binding.result_branch, EcdsaResultBranch::Direct);

        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("H8 prover failed");

        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript.write(&proof).expect("H8 proof must serialize");
        let raw_bytes = transcript.stream.get_ref().len();
        let zstd_bytes = zstd::encode_all(transcript.stream.get_ref().as_slice(), 3)
            .expect("H8 proof compression must succeed")
            .len();
        let component_bytes = proof.get_num_bytes();
        assert_eq!(
            raw_bytes,
            component_bytes + core::mem::size_of::<u32>(),
            "serialized Proof adds one u32 Zip-payload length prefix",
        );
        println!(
            "H8 flat proof bytes: serialized={raw_bytes} components={component_bytes} zstd-3={zstd_bytes}",
        );

        application::verify_application(
            application::ApplicationVerificationInput::new(
                ShaEcdsaH8ApplicationStatement {
                    message: &message,
                    signature_r: &signature_r,
                    signature_s: &signature_s,
                    q_x: &q_x,
                    q_y: &q_y,
                    public_trace: &public_trace,
                },
                proof.clone(),
            ),
            |statement| {
                verify_sha_ecdsa_h8_application_binding(
                    statement.message,
                    statement.signature_r,
                    statement.signature_s,
                    statement.q_x,
                    statement.q_y,
                    statement.public_trace,
                )
                .map(|_| ())
            },
            |proof, statement| {
                ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                    &pp,
                    proof,
                    statement.public_trace,
                    SHA_ECDSA_NUM_VARS,
                    project_scalar_fn,
                    sha256_test_project_ideal,
                )
            },
        )
        .expect("typed H8 verifier rejected the frozen statement");

        // The typed verifier intentionally does not re-hash natively. Change
        // the proof-owned digest and recompute every public scalar selector:
        // the cheap binding remains internally consistent, but the old proof
        // must reject the altered trace.
        let mut changed_digest = public_trace.clone();
        let output_row = sha256::cols::NUM_COMPRESSIONS * sha256::cols::ROWS_PER_COMP;
        changed_digest.binary_poly.to_mut()[sha_ecdsa_cols::PA_E].evaluations[output_row] =
            0_u32.into();
        let changed_digest_bytes =
            extract_sha256_output(&changed_digest).expect("mutated output must remain readable");
        assert_ne!(changed_digest_bytes, binding.scalars.digest);
        let changed_scalars =
            derive_ecdsa_verification_scalars(changed_digest_bytes, &signature_r, &signature_s)
                .expect("frozen signature remains canonically encoded");
        set_h8_scalar_selectors(
            &mut changed_digest,
            &changed_scalars.u1,
            &changed_scalars.u2,
        );
        verify_sha_ecdsa_h8_application_binding(
            &message,
            &signature_r,
            &signature_s,
            &q_x,
            &q_y,
            &changed_digest,
        )
        .expect("changed digest and selectors must pass the non-decorative typed boundary");
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof.clone(),
                &changed_digest,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "the original proof must own the SHA output",
        );

        // Change one non-initial compression input and its feed-forward
        // junction copy together. The cheap H8 binding sees a consistent
        // public copy pair, so only the proof-enforced previous feed-forward
        // and current compression constraints can reject the substitution.
        let mut changed_chain_copy = public_trace.clone();
        let input_row = sha256::cols::ROWS_PER_COMP;
        let junction_row = input_row + sha256::cols::ROUNDS_PER_COMP;
        let original_input =
            changed_chain_copy.binary_poly[sha_ecdsa_cols::PA_A].evaluations[input_row].clone();
        let original_junction =
            changed_chain_copy.binary_poly[sha_ecdsa_cols::PA_A].evaluations[junction_row].clone();
        assert_eq!(original_input, original_junction);
        let replacement = 0_u32.into();
        assert_ne!(original_input, replacement);
        changed_chain_copy.binary_poly.to_mut()[sha_ecdsa_cols::PA_A].evaluations[input_row] =
            replacement.clone();
        changed_chain_copy.binary_poly.to_mut()[sha_ecdsa_cols::PA_A].evaluations[junction_row] =
            replacement;
        verify_sha_ecdsa_h8_application_binding(
            &message,
            &signature_r,
            &signature_s,
            &q_x,
            &q_y,
            &changed_chain_copy,
        )
        .expect("a consistently changed init/junction copy must pass the typed boundary");
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof.clone(),
                &changed_chain_copy,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "the original proof must own a consistent init/junction substitution",
        );

        // Replace Q and G+Q together with another valid pair. This passes the
        // typed key check and leaves the old result cell untouched; only the
        // proof can reject the now-inconsistent Shamir computation.
        let generator_q = (SECP256K1_G_X_UINT, SECP256K1_G_Y_UINT);
        let generator_trace = build_trace_from_message_and_signature::<ShaEcdsaInt>(
            SHA_ECDSA_NUM_VARS,
            &message,
            generator_q,
            &signature_r,
            &signature_s,
        )
        .expect("generator public key must build");
        let generator_public = generator_trace.public(&sig);
        let mut changed_key = public_trace.clone();
        for column in [
            sha_ecdsa_cols::ECDSA_PA_QX,
            sha_ecdsa_cols::ECDSA_PA_QY,
            sha_ecdsa_cols::ECDSA_PA_QGX,
            sha_ecdsa_cols::ECDSA_PA_QGY,
        ] {
            changed_key.int.to_mut()[column] = generator_public.int[column].clone();
        }
        let generator_x = uint_to_be_bytes(&generator_q.0);
        let generator_y = uint_to_be_bytes(&generator_q.1);
        verify_sha_ecdsa_h8_application_binding(
            &message,
            &signature_r,
            &signature_s,
            &generator_x,
            &generator_y,
            &changed_key,
        )
        .expect("changed Q/G+Q pair must pass the typed boundary");
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof.clone(),
                &changed_key,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "the original proof must own Q and G+Q",
        );

        // Build and prove a second valid 400-byte statement using d=k=1.
        // Its own proof verifies, while the frozen statement's proof rejects
        // the alternate public trace.
        let mut other_message = message.clone();
        *other_message.last_mut().expect("H8 message is nonempty") ^= 1;
        let other_sha =
            sha256::build_trace_from_message::<ShaEcdsaInt>(SHA_ECDSA_NUM_VARS, &other_message)
                .expect("alternate 400-byte message must build");
        let other_sha_public =
            other_sha.public(&<Sha256CompressionSliceUair<ShaEcdsaInt> as Uair>::signature());
        let other_digest =
            extract_sha256_output(&other_sha_public).expect("alternate digest must exist");
        let other_r_value = SECP256K1_G_X_UINT;
        let other_s_value = add_mod_order(&CbUint::from_be_slice(&other_digest), &other_r_value);
        let other_r = uint_to_be_bytes(&other_r_value);
        let other_s = uint_to_be_bytes(&other_s_value);
        let other_trace = build_trace_from_message_and_signature::<ShaEcdsaInt>(
            SHA_ECDSA_NUM_VARS,
            &other_message,
            generator_q,
            &other_r,
            &other_s,
        )
        .expect("alternate d=k=1 statement must build");
        let other_public = other_trace.public(&sig);
        verify_sha_ecdsa_h8_application_binding(
            &other_message,
            &other_r,
            &other_s,
            &generator_x,
            &generator_y,
            &other_public,
        )
        .expect("alternate statement must pass every typed binding");
        let other_proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &other_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("alternate H8 prover failed");
        ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            other_proof,
            &other_public,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("alternate H8 proof must verify against its own statement");
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof,
                &other_public,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "a proof must not transfer between two valid H8 statements",
        );
    }

    #[test]
    fn test_e2e_private_signature_statement_and_print_bytes() {
        type Zt = TestShaEcdsaZincTypes;
        type U = PrivateShaEcdsaUair<ShaEcdsaInt>;

        let message = h8_message();
        let signature_r = uint_to_be_bytes(&CbUint::from_be_hex(
            "5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302",
        ));
        let signature_s = uint_to_be_bytes(&CbUint::from_be_hex(
            "7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568",
        ));
        let q = (
            CbUint::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5"),
            CbUint::from_be_hex("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A"),
        );
        let q_x = uint_to_be_bytes(&q.0);
        let q_y = uint_to_be_bytes(&q.1);
        let trace = build_private_signature_trace(
            SHA_ECDSA_NUM_VARS,
            &message,
            q,
            &signature_r,
            &signature_s,
        )
        .expect("private-signature H8 statement must build");
        let signature = U::signature();
        let public_trace = trace.public(&signature);
        verify_private_signature_application_binding(&message, &q_x, &q_y, &public_trace)
            .expect("typed private-signature statement binding rejected the fixture");

        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let prove_started = std::time::Instant::now();
        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("private-signature prover failed");
        let prove_ns = prove_started.elapsed().as_nanos();
        let component_bytes = proof.get_num_bytes();
        let mut transcript = PcsProverTranscript::new_from_commitments(std::iter::empty());
        transcript
            .write(&proof)
            .expect("private-signature proof must serialize");
        let serialized = transcript.stream.get_ref();
        let zstd_bytes = zstd::encode_all(serialized.as_slice(), 3)
            .expect("private-signature proof compression failed")
            .len();
        println!(
            "PRIVATE_SIGNATURE_H8_PROOF serialized={} components={} zstd-3={}",
            serialized.len(),
            component_bytes,
            zstd_bytes,
        );
        assert_eq!(
            serialized.len(),
            component_bytes + core::mem::size_of::<u32>()
        );

        let verify_started = std::time::Instant::now();
        ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
            &pp,
            proof,
            &public_trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
            sha256_test_project_ideal,
        )
        .expect("private-signature verifier rejected the fixture");
        let verify_ns = verify_started.elapsed().as_nanos();
        println!(
            "PRIVATE_SIGNATURE_H8_TIMING rep={} queries={} prove_ns={} verify_ns={}",
            REP, NUM_COL_OPENINGS_FOR_REP, prove_ns, verify_ns,
        );
    }

    #[test]
    fn private_signature_proof_rejects_result_branch_mutation() {
        type Zt = TestShaEcdsaZincTypes;
        type U = PrivateShaEcdsaUair<ShaEcdsaInt>;

        let message = h8_message();
        let signature_r = uint_to_be_bytes(&CbUint::from_be_hex(
            "5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302",
        ));
        let signature_s = uint_to_be_bytes(&CbUint::from_be_hex(
            "7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568",
        ));
        let q = (
            CbUint::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5"),
            CbUint::from_be_hex("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A"),
        );
        let q_x = uint_to_be_bytes(&q.0);
        let q_y = uint_to_be_bytes(&q.1);
        let mut trace = build_private_signature_trace(
            SHA_ECDSA_NUM_VARS,
            &message,
            q,
            &signature_r,
            &signature_s,
        )
        .expect("private-signature fixture must build");
        trace.int.to_mut()[private_sha_ecdsa_cols::W_RESULT_BRANCH].evaluations
            [zinc_test_uair::ecdsa::FINAL_ROW] += ShaEcdsaInt::from(1_u32);

        let public_trace = trace.public(&U::signature());
        verify_private_signature_application_binding(&message, &q_x, &q_y, &public_trace)
            .expect("a witness-only mutation must leave the typed public contract unchanged");
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("the adversarial trace must reach verifier-side constraint checking");
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof,
                &public_trace,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "the proof accepted a mutated private result branch",
        );
    }

    #[test]
    fn private_signature_proof_rejects_scalar_group_junction_mutation() {
        type Zt = TestShaEcdsaZincTypes;
        type U = PrivateShaEcdsaUair<ShaEcdsaInt>;

        let message = h8_message();
        let signature_r = uint_to_be_bytes(&CbUint::from_be_hex(
            "5CD26EE278677AEBEC2C8E7486023D9299EF1B5705D3BE62531E98247E5E8302",
        ));
        let signature_s = uint_to_be_bytes(&CbUint::from_be_hex(
            "7835018BB2CFF73325391068D670363C81AF019C87BC2CAC96672133C2B59568",
        ));
        let q = (
            CbUint::from_be_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5"),
            CbUint::from_be_hex("1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A"),
        );
        let q_x = uint_to_be_bytes(&q.0);
        let q_y = uint_to_be_bytes(&q.1);
        let mut trace = build_private_signature_trace(
            SHA_ECDSA_NUM_VARS,
            &message,
            q,
            &signature_r,
            &signature_s,
        )
        .expect("private-signature fixture must build");

        // The same witness column is consumed as the private u1 bit stream by
        // the scalar identity and as B1 by the Shamir group trace. Mutating it
        // exercises the cross-component junction rather than either typed
        // public boundary.
        trace.int.to_mut()[sha_ecdsa_cols::ECDSA_PA_B1].evaluations[17] += ShaEcdsaInt::from(1_u32);

        let public_trace = trace.public(&U::signature());
        verify_private_signature_application_binding(&message, &q_x, &q_y, &public_trace)
            .expect("a witness-only junction mutation must leave the public contract unchanged");
        let pp = setup_pp::<Zt>(
            SHA_ECDSA_NUM_VARS,
            (
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
                make_iprs(SHA_ECDSA_NUM_VARS),
            ),
        );
        let proof = ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::prove::<false, CHECKED>(
            &pp,
            &trace,
            SHA_ECDSA_NUM_VARS,
            project_scalar_fn,
        )
        .expect("the adversarial trace must reach verifier-side constraint checking");
        assert!(
            ZincPlusPiop::<Zt, U, F, DEGREE_PLUS_ONE>::verify::<_, CHECKED>(
                &pp,
                proof,
                &public_trace,
                SHA_ECDSA_NUM_VARS,
                project_scalar_fn,
                sha256_test_project_ideal,
            )
            .is_err(),
            "the proof accepted a mutated private scalar/group junction",
        );
    }

    #[test]
    fn test_big_linear_tamper_ideal_check() {
        let num_vars = 8;
        do_test::<TestZincTypesIprs, BigLinearUairWithPublicInput<ZtInt>>(
            num_vars,
            (
                make_iprs(num_vars),
                make_iprs(num_vars),
                make_iprs(num_vars),
            ),
            default_project_ideal!(),
            |proof| proof.ideal_check.combined_mle_values.swap(0, 1),
            |res| {
                assert!(matches!(res.unwrap_err(), ProtocolError::IdealCheck(..)));
            },
        );
    }

    //
    // Folded Zip+ (1× fold) — round-trip test
    //

    /// Half-degree binary Zip+ types for the split commitment side of the
    /// folded path. Mirrors [`BinPolyZipTypes`] but with `Eval =
    /// BinaryPoly<16>` and `Cw` over `DensePolynomial<i64, 16>`, so the PCS
    /// commits the post-split BinaryPoly<16> witnesses.
    const HALF_DEGREE_PLUS_ONE: usize = 16;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesHalf {}
    impl ZipTypes for BinPolyZipTypesHalf {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<HALF_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, HALF_DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, HALF_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, HALF_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            HALF_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    #[derive(Clone, Debug)]
    struct TestFoldedZincTypesIprs;

    impl FoldedZincTypes<DEGREE_PLUS_ONE, HALF_DEGREE_PLUS_ONE> for TestFoldedZincTypesIprs {
        type Int = ZtInt;
        type Chal = i128;
        type Pt = i128;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;

        type BinaryZt = BinPolyZipTypesHalf;
        type ArbitraryZt = ArbitraryPolyZipTypesIprs;
        type IntZt = IntZipTypes;

        type BinaryLc = IprsCode<Self::BinaryZt, PnttConfigF65537, REP, CHECKED>;
        type ArbitraryLc = IprsCode<Self::ArbitraryZt, PnttConfigF65537, REP, CHECKED>;
        type IntLc = IprsCode<Self::IntZt, PnttConfigF65537, REP, CHECKED>;
    }

    /// Set up Zip+ params for the folded path. The binary commitment is over
    /// the split column (length `2n` with `BinaryPoly<HALF_D>` entries), so
    /// its `num_vars` is `num_vars + 1`. Arbitrary and int are sized normally.
    #[allow(clippy::type_complexity)]
    fn setup_folded_pp(
        num_vars: usize,
    ) -> (
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::BinaryZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::BinaryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::ArbitraryZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::ArbitraryLc,
        >,
        ZipPlusParams<
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::IntZt,
            <TestFoldedZincTypesIprs as FoldedZincTypes<
                DEGREE_PLUS_ONE,
                HALF_DEGREE_PLUS_ONE,
            >>::IntLc,
        >,
    ) {
        let split_size = 1 << (num_vars + 1);
        let normal_size = 1 << num_vars;
        (
            ZipPlus::setup(
                split_size,
                IprsCode::new_with_optimal_depth(split_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
            ZipPlus::setup(
                normal_size,
                IprsCode::new_with_optimal_depth(normal_size).unwrap(),
            ),
        )
    }

    /// End-to-end test: BinaryDecompositionUair via the **folded** prover/
    /// verifier. Same UAIR, same trace generator, same field — only the
    /// binary commitment is over `BinaryPoly<16>` split columns, opened at
    /// the extended point `(r_0 ‖ γ)`.
    #[test]
    fn test_e2e_folded_binary_decomposition() {
        use crate::{prover::prove_folded, verifier::verify_folded};

        let num_vars = 8;
        let mut rng = rng();
        let pp = setup_folded_pp(num_vars);

        let trace = BinaryDecompositionUair::<ZtInt>::generate_random_trace(num_vars, &mut rng);
        let sig = <BinaryDecompositionUair<ZtInt> as Uair>::signature();
        let public_trace = trace.public(&sig);

        let proof = prove_folded::<
            TestFoldedZincTypesIprs,
            BinaryDecompositionUair<ZtInt>,
            F,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            false,
            CHECKED,
        >(&pp, &trace, num_vars, project_scalar_fn)
        .expect("Folded prover failed");

        verify_folded::<
            TestFoldedZincTypesIprs,
            BinaryDecompositionUair<ZtInt>,
            F,
            IdealOrZero<DegreeOneIdeal<F>>,
            DEGREE_PLUS_ONE,
            HALF_DEGREE_PLUS_ONE,
            CHECKED,
        >(
            &pp,
            proof,
            &public_trace,
            num_vars,
            project_scalar_fn,
            default_project_ideal!(),
        )
        .expect("Folded verifier rejected a valid proof");
    }

    //
    // Folded Zip+ (4× fold) — round-trip test
    //

    /// Quarter-degree binary Zip+ types for the doubly-split commitment side
    /// of the 4× folded path. Mirrors [`BinPolyZipTypesHalf`] but with
    /// `Eval = BinaryPoly<8>` and 8-coeff codewords.
    const QUARTER_DEGREE_PLUS_ONE: usize = 8;

    #[derive(Debug, Clone)]
    pub struct BinPolyZipTypesQuarter {}
    impl ZipTypes for BinPolyZipTypesQuarter {
        const NUM_COLUMN_OPENINGS: usize = NUM_COL_OPENINGS_FOR_REP;
        type Eval = BinaryPoly<QUARTER_DEGREE_PLUS_ONE>;
        type Cw = DensePolynomial<i64, QUARTER_DEGREE_PLUS_ONE>;
        type Fmod = Uint<FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = i128;
        type Pt = i128;
        type CombR = Int<M>;
        type Comb = DensePolynomial<Self::CombR, QUARTER_DEGREE_PLUS_ONE>;
        type EvalDotChal = BinaryPolyInnerProduct<Self::Chal, QUARTER_DEGREE_PLUS_ONE>;
        type CombDotChal = DensePolyInnerProduct<
            Self::CombR,
            Self::Chal,
            Self::CombR,
            MBSInnerProduct,
            QUARTER_DEGREE_PLUS_ONE,
        >;
        type ArrCombRDotChal = MBSInnerProduct;
    }
}
