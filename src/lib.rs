pub mod engine;
pub mod expr;
pub mod lexer;
pub mod parser;

use anyhow::Result;
// Re-export main types and functions for convenient use
pub use engine::{EvaluationError, evaluate, reduce_once, replace, shift, simplify, substitute};
pub use expr::{Expr, IntoExpr};
pub use lexer::{Lexer, Token};
pub use parser::{ParseError, ParseMode, Parser};

// Macros are automatically exported at the crate root by #[macro_export]

/// Parse a lambda calculus expression from a string.
///
/// This is a convenience function that automatically detects the input format
/// (De Bruijn indices vs named variables) and parses accordingly.
///
/// # Arguments
/// * `input` - The lambda calculus expression as a string
///
/// # Returns
/// * `Ok(Expr)` - The parsed expression
/// * `Err(ParseError)` - If parsing fails
///
/// # Errors
/// Returns `ParseError` if the input cannot be tokenized or parsed.
///
/// # Examples
/// ```
/// use minilamb::parse;
///
/// // De Bruijn format (1-based indices)
/// let expr = parse("λ.1").unwrap();
///
/// // Named variable format
/// let expr = parse("λx.x").unwrap();
///
/// // Alternative lambda symbols
/// let expr = parse("\\x.x").unwrap();
/// let expr = parse("/x.x").unwrap();
/// let expr = parse("|x.x").unwrap();
/// ```
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    Parser::parse(input)
}

/// Parse and evaluate a lambda calculus expression.
///
/// This is a convenience function that combines parsing and evaluation
/// with a default step limit.
///
/// # Arguments
/// * `input` - The lambda calculus expression as a string
/// * `max_steps` - Maximum number of reduction steps
///
/// # Returns
/// * `Ok(Expr)` - The evaluated expression in normal form
/// * `Err(_)` - If parsing or evaluation fails
///
/// # Errors
/// Returns an error if parsing fails or evaluation exceeds the step limit.
///
/// # Examples
/// ```
/// use minilamb::parse_and_evaluate;
///
/// // Identity function applied to itself
/// let result = parse_and_evaluate("(λx.x) (λy.y)", 100).unwrap();
/// ```
pub fn parse_and_evaluate(input: &str, max_steps: usize) -> Result<Expr> {
    let expr = parse(input)?;
    let result = evaluate(&expr, max_steps)?;
    Ok(result)
}
