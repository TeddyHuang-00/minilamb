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
