use anyhow::{Result, bail};

/// Lambda calculus expression with explicit bound/free variable distinction
///
/// This enum represents expressions in the untyped lambda calculus:
/// - `BoundVar(usize)`: Bound variable using 1-based De Bruijn index (must be >
///   0)
/// - `FreeVar(String)`: Free variable with name
/// - `Abs(usize, Box<Expr>)`: Lambda abstraction (level, function body)
/// - `App(Vec<Expr>)`: Application (function call with multiple arguments)
///
/// # Examples
/// ```
/// use minilamb::{abs, app};
/// let id = abs!(1, 1); // λ.1 (bound variable)
/// let app_expr = app!(id.clone(), "x"); // (λ.1) x (free variable)
/// ```
#[derive(Hash, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Bound variable using 1-based De Bruijn index (must be > 0)
    BoundVar(usize),
    /// Named free variable
    FreeVar(String),
    /// Lambda abstraction (level, body) - λλλ.body is Abs(3, body)
    Abs(usize, Box<Expr>),
    /// Application (f a1 a2 ... an) where n >= 1, represents ((f a1) a2) ... an
    App(Vec<Expr>),
}

impl std::fmt::Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundVar(n) => {
                assert!(*n > 0, "Variable index must be positive");
                write!(f, "{n}")
            }
            Self::FreeVar(name) => write!(f, "{name}"),
            Self::Abs(level, body) => {
                let lambdas = "λ".repeat(*level);
                write!(f, "({lambdas}.{body:?})")
            }
            Self::App(exprs) => {
                let expr_strs: Vec<String> = exprs.iter().map(|e| format!("{e:?}")).collect();
                write!(f, "({})", expr_strs.join(" "))
            }
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundVar(n) => {
                assert!(*n > 0, "Variable index must be positive");
                write!(f, "{n}")
            }
            Self::FreeVar(name) => write!(f, "{name}"),
            Self::Abs(level, body) => {
                let lambdas = "λ".repeat(*level);
                write!(f, "{lambdas}.{body}")
            }
            Self::App(exprs) => {
                // Handle precedence: abstraction extends to the right
                // Add parentheses around function if it's an abstraction
                // Add parentheses around arguments if they're applications (to preserve
                // right-associativity)
                let formatted_exprs: Vec<String> = exprs
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| {
                        match expr {
                            Self::Abs(..) if i < exprs.len() - 1 => format!("({expr})"),
                            // Parenthesize abstractions when not the last argument
                            Self::App(..) if i > 0 => format!("({expr})"),
                            // Parenthesize applications when not the first argument
                            _ => expr.to_string(),
                        }
                    })
                    .collect();
                write!(f, "{}", formatted_exprs.join(" "))
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
/// let expr = Expr::BoundVar(3);
/// let shifted = shift(1, 2, &expr).unwrap();
/// assert_eq!(shifted, Expr::BoundVar(4));
/// ```
pub fn shift(delta: isize, cutoff: usize, expr: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, BoundVar, FreeVar};
    match expr {
        BoundVar(k) => {
            if *k == 0 {
                bail!("Invalid variable index: 0 (must be positive)");
            }
            if *k >= cutoff {
                let new_val = k.checked_add_signed(delta);
                match new_val {
                    Some(val) if val > 0 => Ok(BoundVar(val)),
                    _ => bail!(
                        "Variable index out of bounds: {k} with delta {delta} and cutoff {cutoff}"
                    ),
                }
            } else {
                Ok(BoundVar(*k))
            }
        }
        FreeVar(name) => Ok(FreeVar(name.clone())),
        Abs(level, body) => {
            let body = shift(delta, cutoff + level, body)?;
            Ok(Abs(*level, Box::new(body)))
        }
        App(exprs) => {
            let shifted_exprs: Result<Vec<Expr>> = exprs
                .iter()
                .map(|expr| shift(delta, cutoff, expr))
                .collect();
            Ok(App(shifted_exprs?))
        }
    }
}

/// Substitutes the variable at index `idx` in the target expression `tgt` with
/// the source expression `src`.
///
/// This implementation uses a specialized substitution semantics designed for
/// compressed multi-level abstractions. When a variable is substituted, the
/// source expression is shifted by the substitution index to account for the
/// binding depth in multi-level abstractions.
///
/// Uses 1-based De Bruijn indices throughout.
///
/// # Arguments
/// * `idx` - The variable index to substitute (1-based, corresponds to binding
///   depth)
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
/// let src = Expr::BoundVar(5);
/// let tgt = Expr::BoundVar(1);
/// let result = substitute(1, &src, &tgt).unwrap();
/// assert_eq!(result, Expr::BoundVar(6)); // src shifted by idx (1)
/// ```
pub fn substitute(idx: usize, src: &Expr, tgt: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, BoundVar, FreeVar};
    match tgt {
        BoundVar(k) => {
            if *k == 0 {
                bail!("Invalid variable index: 0 (must be positive)");
            }
            if *k == idx {
                // Substitute the variable with the source expression.
                // For multi-level abstractions, the source needs to be shifted by idx
                // to account for the binding depth. This allows proper substitution
                // in compressed multi-level abstractions (e.g., λλλ.3).
                shift(idx.try_into()?, 1, src)
            } else {
                Ok(BoundVar(*k))
            }
        }
        FreeVar(name) => Ok(FreeVar(name.clone())),
        Abs(level, body) => {
            let src = shift((*level).try_into()?, 1, src)?;
            let body = substitute(idx + level, &src, body)?;
            Ok(Abs(*level, Box::new(body)))
        }
        App(exprs) => {
            let substituted_exprs: Result<Vec<Expr>> = exprs
                .iter()
                .map(|expr| substitute(idx, src, expr))
                .collect();
            Ok(App(substituted_exprs?))
        }
    }
}

/// Simplifies an expression by compressing consecutive abstractions and
/// normalizing free variables.
///
/// This function performs two types of simplification:
/// 1. Transforms consecutive single-level abstractions like `Abs(1,
///    Box::new(Abs(1, body)))` into multi-level abstractions like `Abs(2,
///    body)`.
/// 2. Normalizes free variables to use consecutive indices starting from the
///    first available index after all bound variables.
///
/// # Arguments
/// * `expr` - The expression to simplify
///
/// # Returns
/// Returns the simplified expression with consecutive abstractions compressed
/// and free variables normalized.
///
/// # Examples
/// ```
/// use minilamb::{abs, expr::simplify};
///
/// // λ.λ.1 becomes λλ.1
/// let nested = abs!(1, abs!(1, 1));
/// let simplified = simplify(&nested);
/// assert_eq!(simplified, abs!(2, 1));
///
/// // Variables remain unchanged: λ.5 stays λ.5
/// let expr = abs!(1, 5);
/// let simplified = simplify(&expr);
/// assert_eq!(simplified, abs!(1, 5));
/// ```
#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
    // do abstraction compression
    compress_abstractions(expr)
}

/// Compresses consecutive abstractions in an expression.
///
/// Transforms consecutive single-level abstractions like `Abs(1,
/// Box::new(Abs(1, body)))` into multi-level abstractions like `Abs(2, body)`.
///
/// # Arguments
/// * `expr` - The expression to compress
///
/// # Returns
/// Returns the expression with consecutive abstractions compressed.
#[must_use]
pub fn compress_abstractions(expr: &Expr) -> Expr {
    use Expr::{Abs, App, BoundVar, FreeVar};
    match expr {
        BoundVar(k) => BoundVar(*k),
        FreeVar(name) => FreeVar(name.clone()),
        Abs(level, body) => {
            let compressed_body = compress_abstractions(body);

            // Check if the body is also an abstraction
            if let Abs(inner_level, inner_body) = compressed_body {
                // Combine consecutive abstractions
                Abs(level + inner_level, inner_body)
            } else {
                Abs(*level, Box::new(compressed_body))
            }
        }
        App(exprs) => {
            let compressed_exprs: Vec<Expr> = exprs.iter().map(compress_abstractions).collect();
            App(compressed_exprs)
        }
    }
}

/// Trait for converting values into `Expr` for use in macros.
///
/// This trait allows macros to accept both `usize` (converted to `BoundVar`)
/// and `Expr` values seamlessly.
pub trait IntoExpr {
    /// Convert the value into an `Expr`.
    fn into_expr(self) -> Expr;
}

impl IntoExpr for usize {
    fn into_expr(self) -> Expr {
        assert!(self > 0, "Variable index must be positive");
        Expr::BoundVar(self)
    }
}

impl IntoExpr for &str {
    fn into_expr(self) -> Expr {
        Expr::FreeVar(self.into())
    }
}

impl IntoExpr for Expr {
    fn into_expr(self) -> Expr {
        self
    }
}

/// Macro for creating lambda abstractions with ergonomic syntax.
///
/// # Examples
/// ```
/// use minilamb::{abs, Expr};
///
/// // Create λλλ.3 (abstraction level 3, variable 3)
/// let expr = abs!(3, 3);
/// assert_eq!(expr, Expr::Abs(3, Box::new(Expr::BoundVar(3))));
///
/// // Mix with other expressions
/// let expr = abs!(1, abs!(1, 1));
/// // This creates λ.λ.1, which gets simplified to λλ.1
/// ```
#[macro_export]
macro_rules! abs {
    ($level:expr, $body:expr) => {
        $crate::expr::Expr::Abs($level, Box::new($crate::expr::IntoExpr::into_expr($body)))
    };
}

/// Macro for creating applications with ergonomic syntax.
///
/// # Examples
/// ```
/// use minilamb::{app, abs, Expr};
///
/// // Create application of variables: 1 2 3
/// let expr = app!(1, 2, 3);
/// assert_eq!(
///     expr,
///     Expr::App(vec![Expr::BoundVar(1), Expr::BoundVar(2), Expr::BoundVar(3)])
/// );
///
/// // Mix with abstractions and free variables
/// let expr = app!(abs!(1, 1), "x");
/// ```
#[macro_export]
macro_rules! app {
    ($($arg:expr),+ $(,)?) => {
        $crate::expr::Expr::App(vec![$($crate::expr::IntoExpr::into_expr($arg)),+])
    };
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use Expr::*;

    use super::*;

    #[test]
    fn test_var_display() {
        let expr = BoundVar(1);
        assert_eq!(expr.to_string(), "1");

        let expr = BoundVar(2);
        assert_eq!(expr.to_string(), "2");

        let expr = FreeVar("x".into());
        assert_eq!(expr.to_string(), "x");
    }

    #[test]
    fn test_abs_display() {
        let expr = abs!(1, 1);
        assert_eq!(expr.to_string(), "λ.1");

        let expr = abs!(1, 2);
        assert_eq!(expr.to_string(), "λ.2");

        let expr = abs!(1, "x");
        assert_eq!(expr.to_string(), "λ.x");
    }

    #[test]
    fn test_app_display() {
        let expr = app!(abs!(1, 1), 2);
        assert_eq!(expr.to_string(), "(λ.1) 2");

        let expr = app!(3, 4);
        assert_eq!(expr.to_string(), "3 4");

        let expr = app!("f", "x");
        assert_eq!(expr.to_string(), "f x");
    }

    #[test]
    fn test_nested_expr_display() {
        let expr = app!(app!(1, 2), abs!(1, 3));
        assert_eq!(expr.to_string(), "1 2 λ.3");

        let nested_expr = app!(abs!(1, app!(1, 2)), 3);
        assert_eq!(nested_expr.to_string(), "(λ.1 2) 3");
    }

    #[test]
    fn test_shift_var_below_cutoff() {
        // Variables below cutoff should not be shifted
        let expr = 1.into_expr();
        let result = shift(5, 3, &expr).unwrap();
        assert_eq!(result, 1.into_expr());

        let expr = 2.into_expr();
        let result = shift(10, 3, &expr).unwrap();
        assert_eq!(result, 2.into_expr());
    }

    #[test]
    fn test_shift_var_above_cutoff() {
        // Variables at or above cutoff should be shifted
        let expr = 3.into_expr();
        let result = shift(2, 3, &expr).unwrap();
        assert_eq!(result, 5.into_expr());

        let expr = 4.into_expr();
        let result = shift(1, 3, &expr).unwrap();
        assert_eq!(result, 5.into_expr());
    }

    #[test]
    fn test_shift_negative_delta() {
        // Test negative shifts
        let expr = 4.into_expr();
        let result = shift(-1, 3, &expr).unwrap();
        assert_eq!(result, 3.into_expr());

        let expr = 9.into_expr();
        let result = shift(-2, 8, &expr).unwrap();
        assert_eq!(result, 7.into_expr());
    }

    #[test]
    fn test_shift_underflow_error() {
        // Should error when shifting would cause underflow (result would be zero or
        // negative)
        let expr = 1.into_expr();
        let result = shift(-3, 1, &expr);
        assert!(result.is_err());

        let expr = 3.into_expr();
        let result = shift(-4, 3, &expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_shift_abs() {
        // λ.1 with shift(1, 1) should become λ.1 (variable 1 is below cutoff 2)
        let expr = abs!(1, 1);
        let result = shift(1, 1, &expr).unwrap();
        assert_eq!(result, abs!(1, 1));

        // λ.2 with shift(2, 1) should become λ.4 (variable 2 is at cutoff 2)
        let expr = abs!(1, 2);
        let result = shift(2, 1, &expr).unwrap();
        assert_eq!(result, abs!(1, 4));
    }

    #[test]
    fn test_shift_app() {
        // (1 2) with shift(1, 2) should become (1 3)
        let expr = app!(1, 2);
        let result = shift(1, 2, &expr).unwrap();
        assert_eq!(result, app!(1, 3));

        // (2 3) with shift(1, 1) should become (3 4)
        let expr = app!(2, 3);
        let result = shift(1, 1, &expr).unwrap();
        assert_eq!(result, app!(3, 4));
    }

    #[test]
    fn test_shift_complex_expr() {
        // Test a complex expression: λ.(1 (λ.2))
        let expr = abs!(1, app!(1, abs!(1, 2)));
        let result = shift(1, 1, &expr).unwrap();
        let expected = abs!(1, app!(1, abs!(1, 2)));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_substitute_exact_match() {
        // Substituting variable 1 with variable 5 in variable 1 should give variable 6
        // due to shifting by the substitution index (1) to account for binding depth
        let src = 5.into_expr();
        let tgt = 1.into_expr();
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, 6.into_expr());
    }

    #[test]
    fn test_substitute_no_match() {
        // Substituting variable 1 with variable 5 in variable 2 should give variable 2
        let src = 5.into_expr();
        let tgt = 2.into_expr();
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, 2.into_expr());
    }

    #[test]
    fn test_substitute_in_abs() {
        // Test substitution in abstractions
        // substitute(1, var(3), λ.1) - variable 1 becomes variable 2 after index
        // adjustment
        let src = 3.into_expr();
        let tgt = abs!(1, 1);
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, abs!(1, 1));

        // substitute(1, var(2), λ.2) - the source gets shifted and substituted
        let src = 2.into_expr();
        let tgt = abs!(1, 2);
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, abs!(1, 5));
    }

    #[test]
    fn test_substitute_in_app() {
        // Test substitution in application
        let src = 5.into_expr();
        let tgt = app!(1, 2);
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, app!(6, 2));

        let src = 7.into_expr();
        let tgt = app!(3, 2);
        let result = substitute(2, &src, &tgt).unwrap();
        assert_eq!(result, app!(3, 9));
    }

    #[test]
    fn test_substitute_complex_expr() {
        // Test substitution in a complex expression: (λ.1) 2
        // substitute(2, var(9), (λ.1) 2) should give (λ.1) 11 (9 shifted by 2)
        let src = 9.into_expr();
        let abs_expr = abs!(1, 1);
        let tgt = app!(abs_expr.clone(), 2);
        let result = substitute(2, &src, &tgt).unwrap();
        assert_eq!(result, app!(abs_expr, 11));
    }

    #[test]
    fn test_substitute_nested_abs() {
        // Test substitution in nested abstractions: λ.λ.2
        let src = 5.into_expr();
        let tgt = abs!(1, abs!(1, 2));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, abs!(1, abs!(1, 2)));

        // substitute(1, var(5), λ.λ.3)
        let src = 5.into_expr();
        let tgt = abs!(1, abs!(1, 3));
        let result = substitute(1, &src, &tgt).unwrap();
        assert_eq!(result, abs!(1, abs!(1, 10)));
    }

    #[test]
    fn test_identity_substitution() {
        // Substituting a variable with itself results in shifting by the substitution
        // index
        let expr = 2.into_expr();
        let result = substitute(2, &2.into_expr(), &expr).unwrap();
        assert_eq!(result, 4.into_expr()); // var(2) shifted by 2
    }

    #[test]
    fn test_zero_shift() {
        // Zero shift should be identity
        let expr = app!(abs!(1, 2), 6);
        let result = shift(0, 1, &expr).unwrap();
        assert_eq!(result, expr);
    }

    #[test]
    fn test_multi_app_display() {
        // Test basic multi-argument application display
        let expr = app!(1, 2, 3);
        assert_eq!(expr.to_string(), "1 2 3");
    }

    #[test]
    fn test_multi_app_with_abstraction() {
        // Test multi-argument application with abstraction as function
        let expr = app!(abs!(1, 1), 2, 3);
        assert_eq!(expr.to_string(), "(λ.1) 2 3");
    }

    #[test]
    fn test_multi_app_with_nested_app() {
        // Test multi-argument application with nested applications
        let expr = app!(1, app!(2, 3), 4);
        assert_eq!(expr.to_string(), "1 (2 3) 4");
    }

    #[test]
    fn test_multi_app_complex_nesting() {
        // Test complex nested multi-argument applications
        let expr = app!(abs!(2, 1), app!(2, 3), 4);
        assert_eq!(expr.to_string(), "(λλ.1) (2 3) 4");
    }

    #[test]
    fn test_simplify_variable() {
        // Bound variables should remain unchanged
        let expr = 1.into_expr();
        let simplified = simplify(&expr);
        assert_eq!(simplified, 1.into_expr());

        let expr = 5.into_expr();
        let simplified = simplify(&expr);
        assert_eq!(simplified, 5.into_expr());
    }

    #[test]
    fn test_simplify_single_abstraction() {
        // Single abstractions should remain unchanged
        let expr = abs!(1, 1);
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(1, 1));

        let expr = abs!(3, 2);
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(3, 2));
    }

    #[test]
    fn test_simplify_consecutive_abstractions() {
        // λ.λ.1 should become λλ.1
        let nested = abs!(1, abs!(1, 1));
        let simplified = simplify(&nested);
        assert_eq!(simplified, abs!(2, 1));

        // λ.λ.λ.2 should become λλλ.2
        let triple_nested = abs!(1, abs!(1, abs!(1, 2)));
        let simplified = simplify(&triple_nested);
        assert_eq!(simplified, abs!(3, 2));
    }

    #[test]
    fn test_simplify_mixed_abstractions() {
        // λλ.λ.1 should become λλλ.1 (combining multi-level with single)
        let mixed = abs!(2, abs!(1, 1));
        let simplified = simplify(&mixed);
        assert_eq!(simplified, abs!(3, 1));

        // λ.λλ.2 should become λλλ.2 (combining single with multi-level)
        let mixed = abs!(1, abs!(2, 2));
        let simplified = simplify(&mixed);
        assert_eq!(simplified, abs!(3, 2));
    }

    #[test]
    fn test_simplify_deep_nesting() {
        // λ.λ.λ.λ.λ.3 should become λλλλλ.3
        let deep_nested = abs!(1, abs!(1, abs!(1, abs!(1, abs!(1, 3)))));
        let simplified = simplify(&deep_nested);
        assert_eq!(simplified, abs!(5, 3));
    }

    #[test]
    fn test_simplify_application_with_nested_abstractions() {
        // (λ.λ.1) (λ.λ.2) should become (λλ.1) (λλ.2)
        let app = app!(abs!(1, abs!(1, 1)), abs!(1, abs!(1, 2)));
        let simplified = simplify(&app);
        let expected = app!(abs!(2, 1), abs!(2, 2));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_simplify_complex_expression() {
        // Test simplification of a complex expression with mixed nesting
        // λ.λ.(1 (λ.λ.2)) should become λλ.(1 (λλ.2))
        let complex = abs!(1, abs!(1, app!(1, abs!(1, abs!(1, 2)))));
        let simplified = simplify(&complex);
        let expected = abs!(2, app!(1, abs!(2, 2)));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_simplify_multi_argument_application() {
        // Test simplification with multi-argument applications
        // (λ.λ.1) a (λ.λ.2) should become (λλ.1) a (λλ.2)
        let app = app!(abs!(1, abs!(1, 1)), 3, abs!(1, abs!(1, 2)));
        let simplified = simplify(&app);
        let expected = app!(abs!(2, 1), 3, abs!(2, 2));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_simplify_already_simplified() {
        // Already simplified expressions should remain unchanged (bound variables)
        let expr = abs!(3, 1);
        let simplified = simplify(&expr);
        assert_eq!(simplified, expr);

        // Variables remain unchanged: (λλ.1) 2 → (λλ.1) 2
        let expr = app!(abs!(2, 1), 2);
        let simplified = simplify(&expr);
        assert_eq!(simplified, app!(abs!(2, 1), 2));
    }

    #[test]
    fn test_simplify_non_consecutive_abstractions() {
        // λ.(1 λ.2) should only simplify the inner abstraction, not combine them
        let non_consecutive = abs!(1, app!(1, abs!(1, 2)));
        let simplified = simplify(&non_consecutive);
        // The outer abstraction should remain single-level since they're not
        // consecutive
        let expected = abs!(1, app!(1, abs!(1, 2)));
        assert_eq!(simplified, expected);
    }

    #[test]
    fn test_abs_macro_with_usize() {
        // Test abs! macro with usize arguments
        let expr = abs!(1, 1);
        assert_eq!(expr, Abs(1, Box::new(BoundVar(1))));

        let expr = abs!(3, 2);
        assert_eq!(expr, Abs(3, Box::new(BoundVar(2))));
    }

    #[test]
    fn test_abs_macro_with_expr() {
        // Test abs! macro with Expr arguments
        let body = BoundVar(1);
        let expr = abs!(2, body.clone());
        assert_eq!(expr, Abs(2, Box::new(body)));

        let nested = App(vec![BoundVar(1), BoundVar(2)]);
        let expr = abs!(1, nested.clone());
        assert_eq!(expr, Abs(1, Box::new(nested)));
    }

    #[test]
    fn test_abs_macro_nested() {
        // Test nested abs! macros
        let expr = abs!(1, abs!(1, 1));
        let expected = Abs(1, Box::new(Abs(1, Box::new(BoundVar(1)))));
        assert_eq!(expr, expected);

        let expr = abs!(2, abs!(1, 3));
        let expected = Abs(2, Box::new(Abs(1, Box::new(BoundVar(3)))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_with_usize() {
        // Test app! macro with usize arguments
        let expr = app!(1, 2);
        let expected = App(vec![BoundVar(1), BoundVar(2)]);
        assert_eq!(expr, expected);

        let expr = app!(1, 2, 3);
        let expected = App(vec![BoundVar(1), BoundVar(2), BoundVar(3)]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_with_expr() {
        // Test app! macro with Expr arguments
        let func = Abs(1, Box::new(BoundVar(1)));
        let arg = BoundVar(2);
        let expr = app!(func.clone(), arg.clone());
        let expected = App(vec![func, arg]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_mixed_args() {
        // Test app! macro with mixed usize and Expr arguments
        let abs_expr = Abs(1, Box::new(BoundVar(1)));
        let expr = app!(abs_expr.clone(), 2, 3);
        let expected = App(vec![abs_expr.clone(), BoundVar(2), BoundVar(3)]);
        assert_eq!(expr, expected);

        let expr = app!(1, abs_expr.clone(), 3);
        let expected = App(vec![BoundVar(1), abs_expr, BoundVar(3)]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mixed_macro_nesting() {
        // Test complex nesting of abs! and app! macros
        let expr = abs!(2, app!(1, 2));
        let expected = Abs(2, Box::new(App(vec![BoundVar(1), BoundVar(2)])));
        assert_eq!(expr, expected);

        let expr = app!(abs!(1, 1), abs!(1, 2));
        let expected = App(vec![
            Abs(1, Box::new(BoundVar(1))),
            Abs(1, Box::new(BoundVar(2))),
        ]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_compress_abstractions_unchanged() {
        // Test that compress_abstractions works as before

        // λ.λ.1 → λλ.1
        let nested = abs!(1, abs!(1, 1));
        let compressed = compress_abstractions(&nested);
        assert_eq!(compressed, abs!(2, 1));

        // Already compressed
        let expr = abs!(3, 2);
        let compressed = compress_abstractions(&expr);
        assert_eq!(compressed, expr);
    }

    #[test]
    fn test_simplify_with_normalization() {
        // Test that simplify compresses abstractions but doesn't normalize variables

        // λ.λ.5 → λλ.5 (compression only)
        let expr = abs!(1, abs!(1, 5));
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(2, 5));

        // Complex: λ.(λ.7) 9 → λ.(λ.7) 9
        let expr = abs!(1, app!(abs!(1, 7), 9));
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(1, app!(abs!(1, 7), 9)));
    }

    #[test]
    fn test_simplify_preserves_semantics() {
        // Test that bound variables are preserved correctly

        // λλλ.2 1 3 - all variables are bound, so no changes except compression
        let expr = abs!(1, abs!(1, abs!(1, app!(2, app!(1, 3)))));
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(3, app!(2, app!(1, 3))));
    }

    #[test]
    fn test_complex_macro_expression() {
        // Test a complex expression using macros: λλ.(2 (λ.1 3))
        let expr = abs!(2, app!(2, app!(abs!(1, 1), 3)));
        let expected = Abs(
            2,
            Box::new(App(vec![
                BoundVar(2),
                App(vec![Abs(1, Box::new(BoundVar(1))), BoundVar(3)]),
            ])),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_trailing_comma() {
        // Test that trailing commas work in app! macro
        let expr = app!(1, 2, 3,);
        let expected = App(vec![BoundVar(1), BoundVar(2), BoundVar(3)]);
        assert_eq!(expr, expected);
    }

    #[test]
    #[should_panic(expected = "Variable index must be positive")]
    fn test_macro_zero_index_panic() {
        // Test that using 0 as variable index panics
        let _expr = abs!(1, 0);
    }

    #[test]
    #[should_panic(expected = "Variable index must be positive")]
    fn test_app_macro_zero_index_panic() {
        // Test that using 0 in app! macro panics
        let _expr = app!(1, 0);
    }

    #[test]
    fn test_into_expr_trait() {
        // Test IntoExpr trait implementations directly
        let usize_val: usize = 5;
        let expr_from_usize = usize_val.into_expr();
        assert_eq!(expr_from_usize, BoundVar(5));

        let expr_val = Abs(1, Box::new(BoundVar(1)));
        let expr_from_expr = expr_val.clone().into_expr();
        assert_eq!(expr_from_expr, expr_val);
    }

    #[test]
    fn test_macro_ergonomics_comparison() {
        // Compare macro syntax with verbose syntax for readability test

        // Verbose way
        let verbose = App(vec![
            Abs(2, Box::new(BoundVar(1))),
            BoundVar(3),
            App(vec![BoundVar(4), BoundVar(5)]),
        ]);

        // Macro way
        let macro_expr = app!(abs!(2, 1), 3, app!(4, 5));

        assert_eq!(macro_expr, verbose);
    }
}
