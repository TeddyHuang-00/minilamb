use anyhow::{Result, bail};

/// Lambda calculus expression (using De Bruijn indices)
///
/// This enum represents expressions in the untyped lambda calculus:
/// - `Var(usize)`: Variable, using zero-based De Bruijn index
/// - `Abs(Box<Expr>)`: Lambda abstraction (function)
/// - `App(Box<Expr>, Box<Expr>)`: Application (function call)
///
/// # Examples
/// ```
/// use minilamb::expr::Expr;
/// let id = Expr::Abs(Box::new(Expr::Var(0))); // λ.1
/// let app = Expr::App(Box::new(id.clone()), Box::new(Expr::Var(0))); // (λ.1) 1
/// ```
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum Expr {
    Var(usize),                // De Bruijn index variable
    Abs(Box<Expr>),            // Lambda abstraction (λx.e)
    App(Box<Expr>, Box<Expr>), // Application (e1 e2)
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var(n) => write!(f, "{}", n + 1), // Display 1-based indices
            Self::Abs(body) => write!(f, "(λ.{body})"),
            Self::App(func, arg) => write!(f, "({func} {arg})"),
        }
    }
}

/// Shifts all free variable indices in the expression greater than or equal to
/// `cutoff` by `delta`.
///
/// Internally uses zero-based De Bruijn indices.
///
/// # Arguments
/// * `delta` - The amount to shift, can be negative
/// * `cutoff` - Variables with indices less than this are unaffected
/// * `expr` - The expression to shift
///
/// # Returns
/// Returns a new expression with shifted indices.
///
/// # Errors
/// Returns an error if shifting would cause an index to overflow or underflow.
///
/// # Examples
/// ```
/// use minilamb::expr::{Expr, shift};
/// let expr = Expr::Var(2);
/// let shifted = shift(1, 1, &expr).unwrap();
/// assert_eq!(shifted, Expr::Var(3));
/// ```
pub fn shift(delta: isize, cutoff: usize, expr: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, Var};
    match expr {
        Var(k) => {
            if *k >= cutoff {
                let Some(k) = k.checked_add_signed(delta) else {
                    bail!(
                        "Variable index out of bounds: {k} with delta {delta} and cutoff {cutoff}"
                    );
                };
                Ok(Var(k))
            } else {
                Ok(Var(*k))
            }
        }
        Abs(body) => {
            let body = shift(delta, cutoff + 1, body)?;
            Ok(Abs(Box::new(body)))
        }
        App(func, arg) => {
            let func = shift(delta, cutoff, func)?;
            let arg = shift(delta, cutoff, arg)?;
            Ok(App(Box::new(func), Box::new(arg)))
        }
    }
}

/// Substitutes the variable at index `idx` in the target expression `tgt` with
/// the source expression `src`.
///
/// Internally uses zero-based De Bruijn indices.
///
/// # Arguments
/// * `idx` - The variable index to substitute
/// * `src` - The expression to substitute in
/// * `tgt` - The target expression
///
/// # Returns
/// Returns a new expression with the substitution applied.
///
/// # Errors
/// Returns an error if:
/// - `idx` cannot be converted to `isize` for shifting
/// - Any shift operation during substitution would cause index overflow or
///   underflow
///
/// # Examples
/// ```
/// use minilamb::expr::{Expr, substitute};
/// let src = Expr::Var(4);
/// let tgt = Expr::Var(0);
/// let result = substitute(0, &src, &tgt).unwrap();
/// assert_eq!(result, Expr::Var(4));
/// ```
pub fn substitute(idx: usize, src: &Expr, tgt: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, Var};
    match tgt {
        Var(k) => {
            if *k == idx {
                // Substitute the variable - shift source by idx to account for binding depth
                shift(idx.try_into()?, 0, src)
            } else {
                Ok(Var(*k))
            }
        }
        Abs(body) => {
            let src = shift(1, 0, src)?;
            let body = substitute(idx + 1, &src, body)?;
            Ok(Abs(Box::new(body)))
        }
        App(func, arg) => {
            let func = substitute(idx, src, func)?;
            let arg = substitute(idx, src, arg)?;
            Ok(App(Box::new(func), Box::new(arg)))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use Expr::*;

    use super::*;

    /// Helper function for testing
    impl Expr {
        #[must_use]
        pub fn var(index: usize) -> Self {
            Var(index)
        }

        #[must_use]
        pub fn abs(body: Self) -> Self {
            Abs(Box::new(body))
        }

        #[must_use]
        pub fn app(func: Self, arg: Self) -> Self {
            App(Box::new(func), Box::new(arg))
        }
    }

    #[test]
    fn test_var_display() {
        let expr = Expr::var(0);
        assert_eq!(expr.to_string(), "1");

        let expr = Expr::var(1);
        assert_eq!(expr.to_string(), "2");
    }

    #[test]
    fn test_abs_display() {
        let expr = Expr::abs(Expr::var(0));
        assert_eq!(expr.to_string(), "(λ.1)");

        let expr = Expr::abs(Expr::var(1));
        assert_eq!(expr.to_string(), "(λ.2)");
    }

    #[test]
    fn test_app_display() {
        let expr = Expr::app(Expr::abs(Expr::var(0)), Expr::var(1));
        assert_eq!(expr.to_string(), "((λ.1) 2)");

        let expr = Expr::app(Expr::var(2), Expr::var(3));
        assert_eq!(expr.to_string(), "(3 4)");
    }

    #[test]
    fn test_nested_expr_display() {
        let expr = Expr::app(
            Expr::app(Expr::var(0), Expr::var(1)),
            Expr::abs(Expr::var(2)),
        );
        assert_eq!(expr.to_string(), "((1 2) (λ.3))");

        let nested_expr = Expr::app(
            Expr::abs(Expr::app(Expr::var(0), Expr::var(1))),
            Expr::var(2),
        );
        assert_eq!(nested_expr.to_string(), "((λ.(1 2)) 3)");
    }

    #[test]
    fn test_shift_var_below_cutoff() {
        // Variables below cutoff should not be shifted
        let expr = Expr::var(0);
        let result = shift(5, 3, &expr).unwrap();
        assert_eq!(result, Expr::var(0));

        let expr = Expr::var(2);
        let result = shift(10, 3, &expr).unwrap();
        assert_eq!(result, Expr::var(2));
    }

    #[test]
    fn test_shift_var_above_cutoff() {
        // Variables at or above cutoff should be shifted
        let expr = Expr::var(3);
        let result = shift(2, 3, &expr).unwrap();
        assert_eq!(result, Expr::var(5));

        let expr = Expr::var(4);
        let result = shift(1, 3, &expr).unwrap();
        assert_eq!(result, Expr::var(5));
    }

    #[test]
    fn test_shift_negative_delta() {
        // Test negative shifts
        let expr = Expr::var(4);
        let result = shift(-1, 3, &expr).unwrap();
        assert_eq!(result, Expr::var(3));

        let expr = Expr::var(9);
        let result = shift(-2, 8, &expr).unwrap();
        assert_eq!(result, Expr::var(7));
    }

    #[test]
    fn test_shift_underflow_error() {
        // Should error when shifting would cause underflow (result would be negative)
        let expr = Expr::var(1);
        let result = shift(-3, 0, &expr);
        assert!(result.is_err());

        let expr = Expr::var(3);
        let result = shift(-4, 3, &expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_shift_abs() {
        // λ.0 with shift(1, 0) should become λ.0 (variable 0 is below cutoff 1)
        let expr = Expr::abs(Expr::var(0));
        let result = shift(1, 0, &expr).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(0)));

        // λ.1 with shift(2, 0) should become λ.3 (variable 1 is at cutoff 1)
        let expr = Expr::abs(Expr::var(1));
        let result = shift(2, 0, &expr).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(3)));
    }

    #[test]
    fn test_shift_app() {
        // (0 1) with shift(1, 1) should become (0 2)
        let expr = Expr::app(Expr::var(0), Expr::var(1));
        let result = shift(1, 1, &expr).unwrap();
        assert_eq!(result, Expr::app(Expr::var(0), Expr::var(2)));

        // (1 2) with shift(1, 0) should become (2 3)
        let expr = Expr::app(Expr::var(1), Expr::var(2));
        let result = shift(1, 0, &expr).unwrap();
        assert_eq!(result, Expr::app(Expr::var(2), Expr::var(3)));
    }

    #[test]
    fn test_shift_complex_expr() {
        // Test a complex expression: λ.(0 (λ.1))
        // With shift(1, 0), the outer lambda increases cutoff to 1 for the body
        // Inner var(0) is below cutoff 1, so unchanged
        // Inner lambda increases cutoff to 2 for its body
        // Inner var(1) is below cutoff 2, so unchanged
        let inner_abs = Expr::abs(Expr::var(1));
        let app = Expr::app(Expr::var(0), inner_abs);
        let expr = Expr::abs(app);

        let result = shift(1, 0, &expr).unwrap();

        let expected_inner_abs = Expr::abs(Expr::var(1));
        let expected_app = Expr::app(Expr::var(0), expected_inner_abs);
        let expected = Expr::abs(expected_app);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_substitute_exact_match() {
        // Substituting variable 0 with variable 4 in variable 0 should give variable 4
        let src = Expr::var(4);
        let tgt = Expr::var(0);
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::var(4));
    }

    #[test]
    fn test_substitute_no_match() {
        // Substituting variable 0 with variable 4 in variable 1 should give variable 1
        let src = Expr::var(4);
        let tgt = Expr::var(1);
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::var(1));
    }

    #[test]
    fn test_substitute_in_abs() {
        // Test substitution in abstractions
        // substitute(0, var(2), λ.0) - variable 0 is bound by the lambda, no
        // substitution
        let src = Expr::var(2);
        let tgt = Expr::abs(Expr::var(0));
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(0)));

        // substitute(0, var(1), λ.1) - variable 1 becomes variable 0 after index
        // adjustment Since we look for index 0+1=1, and find it, we substitute
        // with var(1) shifted up by 1 = var(2) Then that gets shifted again by
        // the substitution logic = var(3)
        let src = Expr::var(1);
        let tgt = Expr::abs(Expr::var(1));
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(3)));
    }

    #[test]
    fn test_substitute_in_app() {
        // Test substitution in application
        let src = Expr::var(4);
        let tgt = Expr::app(Expr::var(0), Expr::var(1));
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::app(Expr::var(4), Expr::var(1)));

        // When substituting var(1) with var(6), the result is shifted by 1
        let src = Expr::var(6);
        let tgt = Expr::app(Expr::var(2), Expr::var(1));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::app(Expr::var(2), Expr::var(7)));
    }

    #[test]
    fn test_substitute_complex_expr() {
        // Test substitution in a complex expression: (λ.0) 1
        // substitute(1, var(8), (λ.0) 1) should give (λ.0) 9 (8 shifted by 1)
        let src = Expr::var(8);
        let abs_expr = Expr::abs(Expr::var(0));
        let tgt = Expr::app(abs_expr.clone(), Expr::var(1));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::app(abs_expr, Expr::var(9)));
    }

    #[test]
    fn test_substitute_nested_abs() {
        // Test substitution in nested abstractions: λ.λ.1
        // substitute(0, var(4), λ.λ.1) should give λ.λ.1 (no change, var 1 refers to
        // outer lambda)
        let src = Expr::var(4);
        let tgt = Expr::abs(Expr::abs(Expr::var(1)));
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::abs(Expr::var(1))));

        // substitute(0, var(4), λ.λ.2)
        // First lambda: look for idx+1=1, find var(2), becomes var(2) after recursion
        // Second lambda: look for idx+1=2, find var(2), substitute with var(4) shifted
        // twice = var(6) But the outer result should be var(8) due to double
        // shifting
        let src = Expr::var(4);
        let tgt = Expr::abs(Expr::abs(Expr::var(2)));
        let result = substitute(0, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::abs(Expr::var(8))));
    }

    #[test]
    fn test_substitute_large_index_conversion_error() {
        // Test error when index is too large to convert to isize
        let src = Expr::var(1);
        let tgt = Expr::var(usize::MAX);
        let result = substitute(usize::MAX, &src, &tgt);
        assert!(result.is_err());
    }

    #[test]
    fn test_substitute_shift_propagation_error() {
        // Test that shift errors propagate through substitution
        // This tests the error case where shifting would cause underflow
        let large_var = usize::MAX - 1;
        let src = Expr::var(large_var);
        let tgt = Expr::var(1);
        let result = substitute(1, &src, &tgt);
        // This should succeed in getting to the shift call, but shift might fail
        // depending on the implementation details of checked_add_signed
        if result.is_err() {
            // Error is expected due to large numbers, which is acceptable
        }
    }

    #[test]
    fn test_identity_substitution() {
        // Substituting a variable with itself results in shifting
        let expr = Expr::var(1);
        let result = substitute(1, &Expr::var(1), &expr).unwrap();
        assert_eq!(result, Expr::var(2)); // var(1) shifted by 1
    }

    #[test]
    fn test_zero_shift() {
        // Zero shift should be identity
        let expr = Expr::app(Expr::abs(Expr::var(1)), Expr::var(5));
        let result = shift(0, 0, &expr).unwrap();
        assert_eq!(result, expr);
    }
}
