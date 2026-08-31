//! Typed composition of a proof verifier with an application result binding.

use core::fmt::{self, Display};

/// Failure returned by [`verify_application`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplicationVerificationError<ProtocolError, ResultBindingError> {
    /// The verifier-owned public result does not satisfy the application's
    /// exact postcondition.
    ResultBinding(ResultBindingError),
    /// The underlying Zinc+ proof was rejected.
    Protocol(ProtocolError),
}

impl<P, B> Display for ApplicationVerificationError<P, B>
where
    P: Display,
    B: Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResultBinding(error) => write!(formatter, "result binding failed: {error}"),
            Self::Protocol(error) => write!(formatter, "protocol verification failed: {error}"),
        }
    }
}

impl<P, B> std::error::Error for ApplicationVerificationError<P, B>
where
    P: std::error::Error + 'static,
    B: std::error::Error + 'static,
{
}

/// One application statement and the proof claimed for that statement.
///
/// Keeping both values in one input prevents call sites from accidentally
/// binding a result against one public statement while verifying a proof
/// against another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicationVerificationInput<Statement, Proof> {
    statement: Statement,
    proof: Proof,
}

impl<Statement, Proof> ApplicationVerificationInput<Statement, Proof> {
    /// Construct one proof claim over one application statement.
    pub fn new(statement: Statement, proof: Proof) -> Self {
        Self { statement, proof }
    }
}

/// Verify an application result binding and its Zinc+ proof as one typed
/// operation over the same statement value.
///
/// The public result check runs first because it is cheap and independent of
/// proof validity. `Ok(())` is reachable only after both checks succeed.
pub fn verify_application<Statement, Proof, P, B>(
    input: ApplicationVerificationInput<Statement, Proof>,
    verify_result_binding: impl FnOnce(&Statement) -> Result<(), B>,
    verify_protocol: impl FnOnce(Proof, &Statement) -> Result<(), P>,
) -> Result<(), ApplicationVerificationError<P, B>> {
    let ApplicationVerificationInput { statement, proof } = input;
    verify_result_binding(&statement).map_err(ApplicationVerificationError::ResultBinding)?;
    verify_protocol(proof, &statement).map_err(ApplicationVerificationError::Protocol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestError(&'static str);

    impl Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for TestError {}

    #[test]
    fn success_requires_both_checks() {
        let binding_called = Cell::new(false);
        let protocol_called = Cell::new(false);

        verify_application(
            ApplicationVerificationInput::new(7_u8, "proof"),
            |statement| {
                binding_called.set(true);
                assert_eq!(*statement, 7);
                Ok::<_, TestError>(())
            },
            |proof, statement| {
                protocol_called.set(true);
                assert_eq!(proof, "proof");
                assert_eq!(*statement, 7);
                Ok::<_, TestError>(())
            },
        )
        .expect("both checks should pass");

        assert!(binding_called.get());
        assert!(protocol_called.get());
    }

    #[test]
    fn binding_failure_short_circuits_the_protocol() {
        let protocol_called = Cell::new(false);
        let result = verify_application(
            ApplicationVerificationInput::new(7_u8, "proof"),
            |_| Err::<(), _>(TestError("binding")),
            |_, _| {
                protocol_called.set(true);
                Ok::<_, TestError>(())
            },
        );

        assert_eq!(
            result,
            Err(ApplicationVerificationError::ResultBinding(TestError(
                "binding"
            )))
        );
        assert!(!protocol_called.get());
    }
}
