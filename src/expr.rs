use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

/// Lambda calculus expression (using De Bruijn indices)
///
/// This enum represents expressions in the untyped lambda calculus:
/// - `Var(usize)`: Variable, using 1-based De Bruijn index (must be > 0)
/// - `Abs(usize, Box<Expr>)`: Lambda abstraction (level, function body)
/// - `App(Vec<Box<Expr>>)`: Application (function call with multiple arguments)
///
/// # Examples
/// ```
/// use minilamb::{abs, app};
/// let id = abs!(1, 1); // λ.1
/// let app_expr = app!(id.clone(), 2); // (λ.1) 2
/// ```
#[derive(Hash, Clone, PartialEq, Eq)]
pub enum Expr {
    /// De Bruijn index variable (1-based, must be > 0)
    Var(usize),
    /// Lambda abstraction (level, body) - λλλ.body is Abs(3, body)
    Abs(usize, Box<Expr>),
    /// Application (f a1 a2 ... an) where n >= 1, represents ((f a1) a2) ... an
    App(Vec<Expr>),
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
            Self::Var(n) => {
                assert!(*n > 0, "Variable index must be positive");
                write!(f, "{n}")
            }
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
/// let src = Expr::Var(5);
/// let tgt = Expr::Var(1);
/// let result = substitute(1, &src, &tgt).unwrap();
/// assert_eq!(result, Expr::Var(6)); // src shifted by idx (1)
/// ```
pub fn substitute(idx: usize, src: &Expr, tgt: &Expr) -> Result<Expr> {
    use Expr::{Abs, App, Var};
    match tgt {
        Var(k) => {
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
                Ok(Var(*k))
            }
        }
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
/// // Free variable normalization: λ.5 becomes λ.2
/// let expr = abs!(1, 5);
/// let simplified = simplify(&expr);
/// assert_eq!(simplified, abs!(1, 2));
/// ```
#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
    // First, do abstraction compression
    let compressed = compress_abstractions(expr);

    // Then, normalize free variables
    normalize_free_variables(&compressed)
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
    use Expr::{Abs, App, Var};
    match expr {
        Var(k) => Var(*k),
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

/// Normalizes free variables in an expression to use consecutive indices.
///
/// Free variables are renumbered to start from the first available index after
/// all bound variables, using consecutive integers.
///
/// # Arguments
/// * `expr` - The expression to normalize
///
/// # Returns
/// Returns the expression with normalized free variable indices.
///
/// # Examples
/// ```
/// use minilamb::{abs, expr::normalize_free_variables, expr::IntoExpr};
///
/// // Standalone free variable: 5 becomes 1
/// let expr = 5_usize.into_expr();
/// let normalized = normalize_free_variables(&expr);
/// assert_eq!(normalized, 1_usize.into_expr());
///
/// // With abstraction: λ.5 becomes λ.2
/// let expr = abs!(1, 5);
/// let normalized = normalize_free_variables(&expr);
/// assert_eq!(normalized, abs!(1, 2));
/// ```
#[must_use]
pub fn normalize_free_variables(expr: &Expr) -> Expr {
    // Collect all free variables with their binding contexts
    let free_vars = collect_free_variables(expr, 0);

    if free_vars.is_empty() {
        // No free variables to normalize
        return expr.clone();
    }

    // Find the maximum binding depth in the expression
    let max_binding_depth = find_max_binding_depth(expr);

    // Create mapping from old indices to new consecutive indices
    let mut unique_free_vars: Vec<usize> = free_vars.into_iter().collect();
    unique_free_vars.sort_unstable();
    unique_free_vars.dedup();

    let mapping: HashMap<usize, usize> = unique_free_vars
        .into_iter()
        .enumerate()
        .map(|(i, old_index)| (old_index, max_binding_depth + i + 1))
        .collect();

    // Apply the mapping
    apply_free_variable_mapping(expr, &mapping, 0)
}

/// Collects all free variables in an expression.
///
/// A variable is considered free if its De Bruijn index is greater than the
/// current binding depth (i.e., it's not bound by any enclosing abstraction).
///
/// # Arguments
/// * `expr` - The expression to analyze
/// * `binding_depth` - The current binding depth (number of enclosing
///   abstractions)
///
/// # Returns
/// Returns a set of all free variable indices in the expression.
fn collect_free_variables(expr: &Expr, binding_depth: usize) -> HashSet<usize> {
    use Expr::{Abs, App, Var};
    match expr {
        Var(k) => {
            if *k > binding_depth {
                // This is a free variable
                std::iter::once(*k).collect()
            } else {
                HashSet::new()
            }
        }
        Abs(level, body) => collect_free_variables(body, binding_depth + level),
        App(exprs) => exprs
            .iter()
            .flat_map(|e| collect_free_variables(e, binding_depth))
            .collect(),
    }
}

/// Finds the maximum binding depth in an expression.
///
/// This determines how many variable indices are reserved for bound variables,
/// which helps determine where free variable indices should start.
///
/// # Arguments
/// * `expr` - The expression to analyze
///
/// # Returns
/// Returns the maximum binding depth in the expression.
fn find_max_binding_depth(expr: &Expr) -> usize {
    use Expr::{Abs, App, Var};
    match expr {
        Var(_) => 0,
        Abs(level, body) => *level + find_max_binding_depth(body),
        App(exprs) => exprs.iter().map(find_max_binding_depth).max().unwrap_or(0),
    }
}

/// Applies a mapping to replace free variables with their normalized indices.
///
/// # Arguments
/// * `expr` - The expression to transform
/// * `mapping` - The mapping from old indices to new indices
/// * `binding_depth` - The current binding depth
///
/// # Returns
/// Returns the expression with free variables replaced according to the
/// mapping.
fn apply_free_variable_mapping(
    expr: &Expr,
    mapping: &HashMap<usize, usize>,
    binding_depth: usize,
) -> Expr {
    use Expr::{Abs, App, Var};
    match expr {
        Var(k) => {
            if *k > binding_depth {
                // This is a free variable, apply mapping
                if let Some(&new_k) = mapping.get(k) {
                    Var(new_k)
                } else {
                    Var(*k) // Shouldn't happen if mapping is complete
                }
            } else {
                Var(*k) // Bound variable, keep as is
            }
        }
        Abs(level, body) => Abs(
            *level,
            Box::new(apply_free_variable_mapping(
                body,
                mapping,
                binding_depth + level,
            )),
        ),
        App(exprs) => App(exprs
            .iter()
            .map(|e| apply_free_variable_mapping(e, mapping, binding_depth))
            .collect()),
    }
}

/// Trait for converting values into `Expr` for use in macros.
///
/// This trait allows macros to accept both `usize` (converted to `Var`) and
/// `Expr` values seamlessly.
pub trait IntoExpr {
    /// Convert the value into an `Expr`.
    fn into_expr(self) -> Expr;
}

impl IntoExpr for usize {
    fn into_expr(self) -> Expr {
        assert!(self > 0, "Variable index must be positive");
        Expr::Var(self)
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
/// assert_eq!(expr, Expr::Abs(3, Box::new(Expr::Var(3))));
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
///     Expr::App(vec![Expr::Var(1), Expr::Var(2), Expr::Var(3)])
/// );
///
/// // Mix with abstractions
/// let expr = app!(abs!(1, 1), 2);
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
            App(vec![func, arg])
        }

        /// Creates a multi-argument application
        ///
        /// # Panics
        /// Panics if the vector has fewer than 2 elements
        #[must_use]
        pub fn app_multi(exprs: Vec<Self>) -> Self {
            assert!(
                exprs.len() >= 2,
                "Application must have at least 2 expressions (function + argument)"
            );
            App(exprs)
        }
    }

    #[test]
    fn test_var_display() {
        let expr = 1.into_expr();
        assert_eq!(expr.to_string(), "1");

        let expr = 2.into_expr();
        assert_eq!(expr.to_string(), "2");
    }

    #[test]
    fn test_abs_display() {
        let expr = abs!(1, 1);
        assert_eq!(expr.to_string(), "λ.1");

        let expr = abs!(1, 2);
        assert_eq!(expr.to_string(), "λ.2");
    }

    #[test]
    fn test_app_display() {
        let expr = app!(abs!(1, 1), 2);
        assert_eq!(expr.to_string(), "(λ.1) 2");

        let expr = app!(3, 4);
        assert_eq!(expr.to_string(), "3 4");
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
        // Bound variables should remain unchanged, free variables should be normalized
        let expr = 1.into_expr();
        let simplified = simplify(&expr);
        assert_eq!(simplified, 1.into_expr()); // Free variable normalized to 1

        let expr = 5.into_expr();
        let simplified = simplify(&expr);
        assert_eq!(simplified, 1.into_expr()); // Free variable normalized to 1
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

        // Free variables get normalized: (λλ.1) 2 → (λλ.1) 3 (2 is free, becomes 3)
        let expr = app!(abs!(2, 1), 2);
        let simplified = simplify(&expr);
        assert_eq!(simplified, app!(abs!(2, 1), 3));
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
        assert_eq!(expr, Abs(1, Box::new(Var(1))));

        let expr = abs!(3, 2);
        assert_eq!(expr, Abs(3, Box::new(Var(2))));
    }

    #[test]
    fn test_abs_macro_with_expr() {
        // Test abs! macro with Expr arguments
        let body = Var(1);
        let expr = abs!(2, body.clone());
        assert_eq!(expr, Abs(2, Box::new(body)));

        let nested = App(vec![Var(1), Var(2)]);
        let expr = abs!(1, nested.clone());
        assert_eq!(expr, Abs(1, Box::new(nested)));
    }

    #[test]
    fn test_abs_macro_nested() {
        // Test nested abs! macros
        let expr = abs!(1, abs!(1, 1));
        let expected = Abs(1, Box::new(Abs(1, Box::new(Var(1)))));
        assert_eq!(expr, expected);

        let expr = abs!(2, abs!(1, 3));
        let expected = Abs(2, Box::new(Abs(1, Box::new(Var(3)))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_with_usize() {
        // Test app! macro with usize arguments
        let expr = app!(1, 2);
        let expected = App(vec![Var(1), Var(2)]);
        assert_eq!(expr, expected);

        let expr = app!(1, 2, 3);
        let expected = App(vec![Var(1), Var(2), Var(3)]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_with_expr() {
        // Test app! macro with Expr arguments
        let func = Abs(1, Box::new(Var(1)));
        let arg = Var(2);
        let expr = app!(func.clone(), arg.clone());
        let expected = App(vec![func, arg]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_mixed_args() {
        // Test app! macro with mixed usize and Expr arguments
        let abs_expr = Abs(1, Box::new(Var(1)));
        let expr = app!(abs_expr.clone(), 2, 3);
        let expected = App(vec![abs_expr.clone(), Var(2), Var(3)]);
        assert_eq!(expr, expected);

        let expr = app!(1, abs_expr.clone(), 3);
        let expected = App(vec![Var(1), abs_expr, Var(3)]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mixed_macro_nesting() {
        // Test complex nesting of abs! and app! macros
        let expr = abs!(2, app!(1, 2));
        let expected = Abs(2, Box::new(App(vec![Var(1), Var(2)])));
        assert_eq!(expr, expected);

        let expr = app!(abs!(1, 1), abs!(1, 2));
        let expected = App(vec![Abs(1, Box::new(Var(1))), Abs(1, Box::new(Var(2)))]);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_collect_free_variables_simple() {
        // Test collecting free variables from simple expressions

        // Single free variable
        let expr = 5.into_expr();
        let free_vars = collect_free_variables(&expr, 0);
        assert_eq!(free_vars, std::iter::once(5).collect());

        // No free variables (bound)
        let expr = 1.into_expr();
        let free_vars = collect_free_variables(&expr, 2);
        assert!(free_vars.is_empty());

        // Mixed bound and free
        let free_vars = collect_free_variables(&3.into_expr(), 2);
        assert_eq!(free_vars, std::iter::once(3).collect());
    }

    #[test]
    fn test_collect_free_variables_abstraction() {
        // Test collecting free variables from abstractions

        // λ.3 - variable 3 is free (beyond binding depth 1)
        let expr = abs!(1, 3);
        let free_vars = collect_free_variables(&expr, 0);
        assert_eq!(free_vars, std::iter::once(3).collect());

        // λλ.3 - variable 3 is free (beyond binding depth 2)
        let expr = abs!(2, 3);
        let free_vars = collect_free_variables(&expr, 0);
        assert_eq!(free_vars, std::iter::once(3).collect());

        // λ.1 - no free variables (1 is bound)
        let expr = abs!(1, 1);
        let free_vars = collect_free_variables(&expr, 0);
        assert!(free_vars.is_empty());
    }

    #[test]
    fn test_collect_free_variables_application() {
        // Test collecting free variables from applications

        // 3 4 - both are free variables
        let expr = app!(3, 4);
        let free_vars = collect_free_variables(&expr, 2);
        assert_eq!(free_vars, [3, 4].into_iter().collect());

        // λ.2 3 - variable 3 is free, 2 is bound
        let expr = app!(abs!(1, 2), 3);
        let free_vars = collect_free_variables(&expr, 0);
        assert_eq!(free_vars, [2, 3].into_iter().collect());
    }

    #[test]
    fn test_find_max_binding_depth() {
        // Test finding maximum binding depth

        // Single variable - no binding
        let expr = 5.into_expr();
        assert_eq!(find_max_binding_depth(&expr), 0);

        // Single abstraction
        let expr = abs!(1, 1);
        assert_eq!(find_max_binding_depth(&expr), 1);

        // Multi-level abstraction
        let expr = abs!(3, 1);
        assert_eq!(find_max_binding_depth(&expr), 3);

        // Nested abstractions
        let expr = abs!(1, abs!(2, 1));
        assert_eq!(find_max_binding_depth(&expr), 3);

        // Application with different depths
        let expr = app!(abs!(2, 1), abs!(1, 1));
        assert_eq!(find_max_binding_depth(&expr), 2);
    }

    #[test]
    fn test_normalize_free_variables_simple() {
        // Test basic free variable normalization

        // Single free variable: 5 → 1
        let expr = 5.into_expr();
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, 1.into_expr());

        // Multiple free variables: 7 3 → 1 2
        let expr = app!(7, 3);
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, app!(2, 1)); // Note: sorted order
    }

    #[test]
    fn test_normalize_free_variables_with_abstractions() {
        // Test free variable normalization with abstractions

        // λ.5 → λ.2 (first free variable after binding depth 1)
        let expr = abs!(1, 5);
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, abs!(1, 2));

        // λλ.7 → λλ.3 (first free variable after binding depth 2)
        let expr = abs!(2, 7);
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, abs!(2, 3));

        // λ.3 5 → λ.2 3 (free variables become consecutive)
        let expr = abs!(1, app!(3, 5));
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, abs!(1, app!(2, 3)));
    }

    #[test]
    fn test_normalize_free_variables_complex() {
        // Test complex expressions with mixed bound and free variables

        // λ.λ.1 3 5 → λ.λ.1 3 4 (bound variable 1 unchanged, free variables normalized)
        let expr = abs!(1, abs!(1, app!(1, app!(3, 5))));
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, abs!(1, abs!(1, app!(1, app!(3, 4)))));

        // (λ.3) 7 → (λ.2) 3 (both free variables normalized)
        let expr = app!(abs!(1, 3), 7);
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, app!(abs!(1, 2), 3));
    }

    #[test]
    fn test_normalize_free_variables_no_change() {
        // Test expressions that should not change

        // No free variables
        let expr = abs!(2, 1);
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, expr);

        // Already normalized free variables
        let expr = abs!(1, 2);
        let normalized = normalize_free_variables(&expr);
        assert_eq!(normalized, expr);
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
        // Test that simplify now includes free variable normalization

        // λ.λ.5 → λλ.3 (compression + normalization)
        let expr = abs!(1, abs!(1, 5));
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(2, 3));

        // Complex: λ.(λ.7) 9 → λ.(λ.3) 4
        let expr = abs!(1, app!(abs!(1, 7), 9));
        let simplified = simplify(&expr);
        assert_eq!(simplified, abs!(1, app!(abs!(1, 3), 4)));
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
    fn test_apply_free_variable_mapping() {
        // Test the mapping application function
        let mapping = [(5, 2), (7, 3)].into_iter().collect();

        // Simple variable mapping
        let expr = 5.into_expr();
        let mapped = apply_free_variable_mapping(&expr, &mapping, 0);
        assert_eq!(mapped, 2.into_expr());

        // Bound variable unchanged
        let expr = 1.into_expr();
        let mapped = apply_free_variable_mapping(&expr, &mapping, 2);
        assert_eq!(mapped, 1.into_expr());

        // Mixed mapping in application
        let expr = app!(5, 7);
        let mapped = apply_free_variable_mapping(&expr, &mapping, 0);
        assert_eq!(mapped, app!(2, 3));
    }

    #[test]
    fn test_complex_macro_expression() {
        // Test a complex expression using macros: λλ.(2 (λ.1 3))
        let expr = abs!(2, app!(2, app!(abs!(1, 1), 3)));
        let expected = Abs(
            2,
            Box::new(App(vec![
                Var(2),
                App(vec![Abs(1, Box::new(Var(1))), Var(3)]),
            ])),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_app_macro_trailing_comma() {
        // Test that trailing commas work in app! macro
        let expr = app!(1, 2, 3,);
        let expected = App(vec![Var(1), Var(2), Var(3)]);
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
        assert_eq!(expr_from_usize, Var(5));

        let expr_val = Abs(1, Box::new(Var(1)));
        let expr_from_expr = expr_val.clone().into_expr();
        assert_eq!(expr_from_expr, expr_val);
    }

    #[test]
    fn test_macro_ergonomics_comparison() {
        // Compare macro syntax with verbose syntax for readability test

        // Verbose way
        let verbose = App(vec![
            Abs(2, Box::new(Var(1))),
            Var(3),
            App(vec![Var(4), Var(5)]),
        ]);

        // Macro way
        let macro_expr = app!(abs!(2, 1), 3, app!(4, 5));

        assert_eq!(macro_expr, verbose);
    }
}
