use anyhow::{Result, bail};

/// Lambda calculus expression (using De Bruijn indices)
///
/// This enum represents expressions in the untyped lambda calculus:
/// - `Var(usize)`: Variable, using 1-based De Bruijn index (must be > 0)
/// - `Abs(usize, Box<Expr>)`: Lambda abstraction (level, function body)
/// - `App(Box<Expr>, Box<Expr>)`: Application (function call)
///
/// # Examples
/// ```
/// use minilamb::expr::Expr;
/// let id = Expr::Abs(1, Box::new(Expr::Var(1))); // λ.1
/// ```
#[derive(Hash, Clone, PartialEq, Eq)]
pub enum Expr {
    Var(usize),                // De Bruijn index variable (1-based, must be > 0)
    Abs(usize, Box<Expr>),     // Lambda abstraction (level, body) - λλλ.body is Abs(3, body)
    App(Box<Expr>, Box<Expr>), // Application (e1 e2)
}

impl std::fmt::Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var(n) => {
                assert!(*n > 0, "Variable index must be positive");
                write!(f, "{n}")
            }
            Self::Abs(level, body) => {
                let lambdas = "λ".repeat(*level);
                write!(f, "({lambdas}.{body:?})")
            }
            Self::App(func, arg) => {
                write!(f, "({func:?} {arg:?})")
            }
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var(n) => {
                assert!(*n > 0, "Variable index must be positive");
                write!(f, "{n}")
            }
            Self::Abs(level, body) => {
                let lambdas = "λ".repeat(*level);
                write!(f, "{lambdas}.{body}")
            }
            Self::App(func, arg) => {
                // Handle precedence: abstraction extends to the right
                // Add parentheses around function if it's an abstraction
                // Add parentheses around argument if it's an application (to preserve
                // right-associativity)
                let func_str = match func.as_ref() {
                    Self::Abs(..) => format!("({func})"),
                    _ => func.to_string(),
                };
                let arg_str = match arg.as_ref() {
                    Self::App(..) => format!("({arg})"),
                    _ => arg.to_string(),
                };
                write!(f, "{func_str} {arg_str}")
            }
        }
    }
}

/// Shifts all free variable indices in the expression greater than or equal to
/// `cutoff` by `delta`.
///
/// Uses 1-based De Bruijn indices throughout.
///
/// # Arguments
/// * `delta` - The amount to shift, can be negative
/// * `cutoff` - Variables with indices less than this are unaffected (1-based)
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
/// let expr = Expr::Var(3);
/// let shifted = shift(1, 2, &expr).unwrap();
/// assert_eq!(shifted, Expr::Var(4));
/// ```
pub fn shift(delta: isize, cutoff: usize, expr: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, Var};
    match expr {
        Var(k) => {
            if *k == 0 {
                bail!("Invalid variable index: 0 (must be positive)");
            }
            if *k >= cutoff {
                let new_val = k.checked_add_signed(delta);
                match new_val {
                    Some(val) if val > 0 => Ok(Var(val)),
                    _ => bail!(
                        "Variable index out of bounds: {k} with delta {delta} and cutoff {cutoff}"
                    ),
                }
            } else {
                Ok(Var(*k))
            }
        }
        Abs(level, body) => {
            let body = shift(delta, cutoff + level, body)?;
            Ok(Abs(*level, Box::new(body)))
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
/// Uses 1-based De Bruijn indices throughout.
///
/// # Arguments
/// * `idx` - The variable index to substitute (1-based)
/// * `src` - The expression to substitute in
/// * `tgt` - The target expression
///
/// # Returns
/// Returns a new expression with the substitution applied.
///
/// # Errors
/// Returns an error if any shift operation during substitution would cause
/// index overflow or underflow.
///
/// # Examples
/// ```
/// use minilamb::expr::{Expr, substitute};
/// let src = Expr::Var(5);
/// let tgt = Expr::Var(1);
/// let result = substitute(1, &src, &tgt).unwrap();
/// assert_eq!(result, Expr::Var(6));
/// ```
pub fn substitute(idx: usize, src: &Expr, tgt: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, Var};
    match tgt {
        Var(k) => {
            if *k == 0 {
                bail!("Invalid variable index: 0 (must be positive)");
            }
            if *k == idx {
                // Substitute the variable - shift source by idx to account for binding depth
                shift(idx.try_into()?, 1, src)
            } else {
                Ok(Var(*k))
            }
        }
        Abs(level, body) => {
            let src = shift((*level).try_into()?, 1, src)?;
            let body = substitute(idx + level, &src, body)?;
            Ok(Abs(*level, Box::new(body)))
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
        /// Creates a variable with a De Bruijn index.
        ///
        /// # Panics
        /// Panics if `index` is 0, since De Bruijn indices are 1-based.
        #[must_use]
        pub fn var(index: usize) -> Self {
            assert!(index > 0, "Variable index must be positive");
            Var(index)
        }

        #[must_use]
        pub fn abs(body: Self) -> Self {
            Abs(1, Box::new(body))
        }

        /// Creates a multi-level abstraction with the given level and body
        ///
        /// # Panics
        /// Panics if level is 0 (must be positive)
        #[must_use]
        pub fn abs_multi(level: usize, body: Self) -> Self {
            assert!(level > 0, "Abstraction level must be positive");
            Abs(level, Box::new(body))
        }

        #[must_use]
        pub fn app(func: Self, arg: Self) -> Self {
            App(Box::new(func), Box::new(arg))
        }
    }

    #[test]
    fn test_var_display() {
        let expr = Expr::var(1);
        assert_eq!(expr.to_string(), "1");

        let expr = Expr::var(2);
        assert_eq!(expr.to_string(), "2");
    }

    #[test]
    fn test_abs_display() {
        let expr = Expr::abs(Expr::var(1));
        assert_eq!(expr.to_string(), "λ.1");

        let expr = Expr::abs(Expr::var(2));
        assert_eq!(expr.to_string(), "λ.2");
    }

    #[test]
    fn test_app_display() {
        let expr = Expr::app(Expr::abs(Expr::var(1)), Expr::var(2));
        assert_eq!(expr.to_string(), "(λ.1) 2");

        let expr = Expr::app(Expr::var(3), Expr::var(4));
        assert_eq!(expr.to_string(), "3 4");
    }

    #[test]
    fn test_nested_expr_display() {
        let expr = Expr::app(
            Expr::app(Expr::var(1), Expr::var(2)),
            Expr::abs(Expr::var(3)),
        );
        assert_eq!(expr.to_string(), "1 2 λ.3");

        let nested_expr = Expr::app(
            Expr::abs(Expr::app(Expr::var(1), Expr::var(2))),
            Expr::var(3),
        );
        assert_eq!(nested_expr.to_string(), "(λ.1 2) 3");
    }

    #[test]
    fn test_shift_var_below_cutoff() {
        // Variables below cutoff should not be shifted
        let expr = Expr::var(1);
        let result = shift(5, 3, &expr).unwrap();
        assert_eq!(result, Expr::var(1));

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
        // Should error when shifting would cause underflow (result would be zero or
        // negative)
        let expr = Expr::var(1);
        let result = shift(-3, 1, &expr);
        assert!(result.is_err());

        let expr = Expr::var(3);
        let result = shift(-4, 3, &expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_shift_abs() {
        // λ.1 with shift(1, 1) should become λ.1 (variable 1 is below cutoff 2)
        let expr = Expr::abs(Expr::var(1));
        let result = shift(1, 1, &expr).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(1)));

        // λ.2 with shift(2, 1) should become λ.4 (variable 2 is at cutoff 2)
        let expr = Expr::abs(Expr::var(2));
        let result = shift(2, 1, &expr).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(4)));
    }

    #[test]
    fn test_shift_app() {
        // (1 2) with shift(1, 2) should become (1 3)
        let expr = Expr::app(Expr::var(1), Expr::var(2));
        let result = shift(1, 2, &expr).unwrap();
        assert_eq!(result, Expr::app(Expr::var(1), Expr::var(3)));

        // (2 3) with shift(1, 1) should become (3 4)
        let expr = Expr::app(Expr::var(2), Expr::var(3));
        let result = shift(1, 1, &expr).unwrap();
        assert_eq!(result, Expr::app(Expr::var(3), Expr::var(4)));
    }

    #[test]
    fn test_shift_complex_expr() {
        // Test a complex expression: λ.(1 (λ.2))
        let inner_abs = Expr::abs(Expr::var(2));
        let app = Expr::app(Expr::var(1), inner_abs);
        let expr = Expr::abs(app);

        let result = shift(1, 1, &expr).unwrap();

        let expected_inner_abs = Expr::abs(Expr::var(2));
        let expected_app = Expr::app(Expr::var(1), expected_inner_abs);
        let expected = Expr::abs(expected_app);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_substitute_exact_match() {
        // Substituting variable 1 with variable 5 in variable 1 should give variable 5
        let src = Expr::var(5);
        let tgt = Expr::var(1);
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::var(6));
    }

    #[test]
    fn test_substitute_no_match() {
        // Substituting variable 1 with variable 5 in variable 2 should give variable 2
        let src = Expr::var(5);
        let tgt = Expr::var(2);
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::var(2));
    }

    #[test]
    fn test_substitute_in_abs() {
        // Test substitution in abstractions
        // substitute(1, var(3), λ.1) - variable 1 becomes variable 2 after index
        // adjustment
        let src = Expr::var(3);
        let tgt = Expr::abs(Expr::var(1));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(1)));

        // substitute(1, var(2), λ.2) - variable 2 becomes variable 1 after adjustment,
        // should substitute
        let src = Expr::var(2);
        let tgt = Expr::abs(Expr::var(2));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::var(5)));
    }

    #[test]
    fn test_substitute_in_app() {
        // Test substitution in application
        let src = Expr::var(5);
        let tgt = Expr::app(Expr::var(1), Expr::var(2));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::app(Expr::var(6), Expr::var(2)));

        let src = Expr::var(7);
        let tgt = Expr::app(Expr::var(3), Expr::var(2));
        let result = substitute(2, &src, &tgt).unwrap();
        assert_eq!(result, Expr::app(Expr::var(3), Expr::var(9)));
    }

    #[test]
    fn test_substitute_complex_expr() {
        // Test substitution in a complex expression: (λ.1) 2
        // substitute(2, var(9), (λ.1) 2) should give (λ.1) 10 (9 shifted by 1)
        let src = Expr::var(9);
        let abs_expr = Expr::abs(Expr::var(1));
        let tgt = Expr::app(abs_expr.clone(), Expr::var(2));
        let result = substitute(2, &src, &tgt).unwrap();
        assert_eq!(result, Expr::app(abs_expr, Expr::var(11)));
    }

    #[test]
    fn test_substitute_nested_abs() {
        // Test substitution in nested abstractions: λ.λ.2
        let src = Expr::var(5);
        let tgt = Expr::abs(Expr::abs(Expr::var(2)));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::abs(Expr::var(2))));

        // substitute(1, var(5), λ.λ.3)
        let src = Expr::var(5);
        let tgt = Expr::abs(Expr::abs(Expr::var(3)));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, Expr::abs(Expr::abs(Expr::var(10))));
    }

    #[test]
    fn test_identity_substitution() {
        // Substituting a variable with itself results in shifting
        let expr = Expr::var(2);
        let result = substitute(2, &Expr::var(2), &expr).unwrap();
        assert_eq!(result, Expr::var(4)); // var(2) shifted by 1
    }

    #[test]
    fn test_zero_shift() {
        // Zero shift should be identity
        let expr = Expr::app(Expr::abs(Expr::var(2)), Expr::var(6));
        let result = shift(0, 1, &expr).unwrap();
        assert_eq!(result, expr);
    }
}
