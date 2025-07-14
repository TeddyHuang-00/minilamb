use thiserror::Error;

use crate::{
    expr::{Expr, simplify},
    lexer::{Lexer, Token},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseMode {
    /// Detect format automatically
    Auto,
    /// Force De Bruijn parsing
    DeBruijn,
    /// Force named variable parsing
    Named,
}

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    mode: ParseMode,
    context: Vec<String>,   // Variable binding context for De Bruijn conversion
    free_vars: Vec<String>, // Free variable names for consistent indexing
}

#[derive(Debug, Clone, Error)]
pub enum ParseError {
    #[error("Expected {expected} but found {found:?} at position {position}")]
    UnexpectedToken {
        expected: String,
        found: Token,
        position: usize,
    },
    #[error("Mixed variable formats (named and De Bruijn) at position {position}")]
    MixedVariableFormats { position: usize },
    #[error("Empty expression at position {position}")]
    EmptyExpression { position: usize },
}

impl Parser {
    #[must_use]
    pub const fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            mode: ParseMode::Auto,
            context: Vec::new(),
            free_vars: Vec::new(),
        }
    }

    /// Parse a lambda calculus expression from a string.
    ///
    /// # Errors
    /// Returns a `ParseError` if the input cannot be tokenized or parsed.
    pub fn parse(input: &str) -> Result<Expr, ParseError> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().map_err(|_e| ParseError::UnexpectedToken {
            expected: "valid token".to_string(),
            found: Token::Eof,
            position: 0,
        })?;

        let mut parser = Self::new(tokens);
        parser.detect_mode()?;
        let expr = parser.parse_expression()?;
        Ok(simplify(&expr))
    }

    /// Parse a lambda calculus expression with a specific parsing mode.
    ///
    /// # Errors
    /// Returns a `ParseError` if the input cannot be tokenized or parsed.
    pub fn parse_with_mode(input: &str, mode: ParseMode) -> Result<Expr, ParseError> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().map_err(|_e| ParseError::UnexpectedToken {
            expected: "valid token".to_string(),
            found: Token::Eof,
            position: 0,
        })?;

        let mut parser = Self::new(tokens);
        parser.mode = mode;
        let expr = parser.parse_expression()?;
        Ok(simplify(&expr))
    }

    fn detect_mode(&mut self) -> Result<(), ParseError> {
        // Auto-detect parsing mode based on token patterns
        let has_named_vars = self.tokens.iter().any(|t| matches!(t, Token::Ident(_)));
        let has_numbers = self.tokens.iter().any(|t| matches!(t, Token::Number(_)));

        self.mode = match (has_named_vars, has_numbers) {
            (true, false) => ParseMode::Named,
            (false, true | false) => ParseMode::DeBruijn, // Default
            (true, true) => return Err(ParseError::MixedVariableFormats { position: 0 }),
        };

        Ok(())
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.skip_whitespace();

        if self.is_at_end() {
            return Err(ParseError::EmptyExpression {
                position: self.current,
            });
        }

        // Check for abstraction or application
        if matches!(self.peek(), Some(Token::Lambda)) {
            self.parse_abstraction()
        } else {
            self.parse_application()
        }
    }

    fn parse_abstraction(&mut self) -> Result<Expr, ParseError> {
        // Parse: λx.e or λxy.e or λ.e (De Bruijn) or λλ.e (multiple abstractions)
        let mut lambda_count = 0;
        let mut bindings = Vec::new();

        // Count consecutive lambda symbols
        while matches!(self.peek(), Some(Token::Lambda)) {
            self.advance(); // consume lambda
            lambda_count += 1;
            self.skip_whitespace();

            // In De Bruijn mode, λλ.e means nested abstractions
            if matches!(self.mode, ParseMode::DeBruijn) && matches!(self.peek(), Some(Token::Dot)) {
                break;
            }

            // Collect bindings until dot
            while !matches!(self.peek(), Some(Token::Dot | Token::Lambda | Token::Eof)) {
                match self.peek() {
                    Some(Token::Ident(name)) => {
                        bindings.push(name.clone());
                        self.advance();
                        self.skip_whitespace();
                    }
                    Some(Token::Number(_)) if matches!(self.mode, ParseMode::DeBruijn) => {
                        // Numbers as binding names not allowed in abstractions
                        return Err(ParseError::UnexpectedToken {
                            expected: "variable name or '.'".to_string(),
                            found: self.peek().unwrap_or(&Token::Eof).clone(),
                            position: self.current,
                        });
                    }
                    Some(Token::Whitespace) => {
                        self.advance();
                    }
                    _ => break,
                }
            }
        }

        // Consume the dot
        if !matches!(self.peek(), Some(Token::Dot)) {
            return Err(ParseError::UnexpectedToken {
                expected: "'.'".to_string(),
                found: self.peek().unwrap_or(&Token::Eof).clone(),
                position: self.current,
            });
        }
        self.advance(); // consume dot
        self.skip_whitespace();

        // Handle different abstraction patterns
        let body = if matches!(self.mode, ParseMode::DeBruijn) && bindings.is_empty() {
            // λ.e or λλ.e pattern
            for _ in 0..lambda_count {
                self.context.push(format!("_{}", self.context.len()));
            }
            let body = self.parse_expression()?;
            // Remove context after parsing
            for _ in 0..lambda_count {
                self.context.pop();
            }
            body
        } else {
            // Named variable pattern: λx.e or λxy.e
            if bindings.is_empty() && lambda_count == 1 && matches!(self.mode, ParseMode::Named) {
                return Err(ParseError::UnexpectedToken {
                    expected: "variable binding".to_string(),
                    found: Token::Dot,
                    position: self.current,
                });
            }

            // Add bindings to context
            for binding in &bindings {
                self.context.push(binding.clone());
            }
            let body = self.parse_expression()?;
            // Remove bindings from context
            for _ in &bindings {
                self.context.pop();
            }
            body
        };

        // Build compressed abstraction
        let mut result = body;
        let abstraction_level = lambda_count.max(bindings.len());
        if abstraction_level > 0 {
            result = Expr::Abs(abstraction_level, Box::new(result));
        }

        Ok(result)
    }

    fn parse_application(&mut self) -> Result<Expr, ParseError> {
        // Parse left-associative application: f g h = ((f g) h)
        // Now compressed as App(vec![f, g, h])
        let mut exprs = vec![self.parse_atom()?];

        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RParen | Token::Eof)) {
            self.skip_whitespace();
            if matches!(self.peek(), Some(Token::RParen | Token::Eof)) {
                break;
            }
            let arg = self.parse_atom()?;
            exprs.push(arg);
        }

        // If only one expression, return it directly
        if exprs.len() == 1 {
            Ok(exprs.into_iter().next().unwrap_or_else(|| unreachable!()))
        } else {
            // Create compressed application
            Ok(Expr::App(exprs.into_iter().map(Box::new).collect()))
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        self.skip_whitespace();

        match self.peek() {
            Some(Token::LParen) => {
                self.advance(); // consume '('
                let expr = self.parse_expression()?;
                if !matches!(self.peek(), Some(Token::RParen)) {
                    return Err(ParseError::UnexpectedToken {
                        expected: "')'".to_string(),
                        found: self.peek().unwrap_or(&Token::Eof).clone(),
                        position: self.current,
                    });
                }
                self.advance(); // consume ')'
                Ok(expr)
            }
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.advance();
                Ok(self.resolve_variable(&name))
            }
            Some(Token::Number(index)) => {
                let index = *index;
                self.advance();
                if matches!(self.mode, ParseMode::Named) {
                    return Err(ParseError::MixedVariableFormats {
                        position: self.current,
                    });
                }
                // Use usize directly (already validated by lexer)
                Ok(Expr::Var(index))
            }
            Some(Token::Lambda) => {
                // Nested abstraction
                self.parse_abstraction()
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: token.clone(),
                position: self.current,
            }),
            None => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: Token::Eof,
                position: self.current,
            }),
        }
    }

    fn resolve_variable(&mut self, name: &str) -> Expr {
        // Convert variable name to De Bruijn index (1-based)
        for (i, var) in self.context.iter().rev().enumerate() {
            if var == name {
                // Convert 0-based context index to 1-based usize
                let index = i + 1;
                return Expr::Var(index);
            }
        }

        // Variable not found in context - treat as free variable
        // Check if we've seen this free variable before
        if let Some(free_idx) = self.free_vars.iter().position(|v| v == name) {
            // Use existing free variable index
            let index = self.context.len() + free_idx + 1;
            return Expr::Var(index);
        }

        // New free variable - add to the list
        self.free_vars.push(name.to_string());
        let index = self.context.len() + self.free_vars.len();
        Expr::Var(index)
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current)
    }

    fn advance(&mut self) {
        if self.current < self.tokens.len() {
            self.current += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Some(Token::Eof) | None)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(Token::Whitespace)) {
            self.advance();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::{abs, app};

    #[test]
    fn test_de_bruijn_formats() {
        // λ.1 (identity) - 1-based input stays 1-based internal
        let expr = Parser::parse("λ.1").unwrap();
        assert_eq!(expr, abs!(1, 1));

        // λλ.2 (const function) - 2 stays 2 internally
        let expr = Parser::parse("λλ.2").unwrap();
        assert_eq!(expr, abs!(2, 2));

        // λ.λ.2 1 (application) - consecutive abstractions get simplified to λλ
        let expr = Parser::parse("λ.λ.2 1").unwrap();
        let expected = abs!(2, app!(2, 1));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_named_variable_formats() {
        // λx.x (identity)
        let expr = Parser::parse("λx.x").unwrap();
        assert_eq!(expr, abs!(1, 1));

        // λx.λy.y x - consecutive abstractions get simplified
        let expr = Parser::parse("λx.λy.y x").unwrap();
        let expected = abs!(2, app!(1, 2));
        assert_eq!(expr, expected);

        // λx y.x (abbreviated abstraction)
        let expr = Parser::parse("λx y.x").unwrap();
        let expected = abs!(2, 2);
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_alternative_lambda_symbols() {
        // Test \, /, | as lambda symbols
        let patterns = ["\\x.x", "/x.x", "|x.x"];
        for pattern in &patterns {
            let expr = Parser::parse(pattern).unwrap();
            assert_eq!(expr, abs!(1, 1));
        }
    }

    #[test]
    fn test_application_associativity() {
        // Use free variables for testing application parsing
        let expr = Parser::parse("λx.λy.λz.x y z").unwrap();
        // This should parse as λλλ.((x y) z) due to simplification
        let expected = abs!(3, app!(3, 2, 1));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_abstraction_precedence() {
        // λx.f g should parse as λx.(f g), not (λx.f) g
        // Test with bound variables only to avoid unbound variable errors
        let expr = Parser::parse("λx.λy.x y").unwrap();
        let expected = abs!(2, app!(2, 1));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parentheses() {
        // Test explicit parentheses
        let expr = Parser::parse("(λx.x)").unwrap();
        assert_eq!(expr, abs!(1, 1));

        // Test grouping in applications
        let expr = Parser::parse("λf.λx.f (f x)").unwrap();
        let expected = abs!(2, app!(2, app!(2, 1)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mixed_formats_error() {
        // Should error when mixing named variables and De Bruijn indices
        let result = Parser::parse("λx.1");
        assert!(matches!(
            result,
            Err(ParseError::MixedVariableFormats { .. })
        ));
    }

    #[test]
    fn test_free_variables() {
        // Should treat unbound variables as free variables
        let result = Parser::parse("λx.y");
        assert!(result.is_ok());
        let expr = result.unwrap();
        // y should be treated as a free variable with index 2 (beyond the binding
        // context of x)
        assert_eq!(expr, abs!(1, 2));

        // Test multiple free variables
        let result = Parser::parse("λx.λy.z w");
        assert!(result.is_ok());
        let expr = result.unwrap();
        // z should be index 3, w should be index 4 (beyond binding context of x, y)
        assert_eq!(expr, abs!(2, app!(3, 4)));
    }

    #[test]
    fn test_empty_expression_error() {
        let result = Parser::parse("");
        assert!(matches!(result, Err(ParseError::EmptyExpression { .. })));
    }

    #[test]
    fn test_unexpected_token_error() {
        let result = Parser::parse("λx");
        assert!(matches!(result, Err(ParseError::UnexpectedToken { .. })));
    }

    #[test]
    fn test_invalid_debruijn_index_zero() {
        // De Bruijn indices must start from 1, not 0
        // This should now fail at the lexer level since PositiveNumber rejects 0
        let result = Parser::parse("λ.0");
        assert!(result.is_err());
        // Should be a tokenization error about positive numbers
    }

    #[test]
    fn test_parse_mode_detection() {
        // Auto-detect De Bruijn mode (1-based input)
        let expr = Parser::parse_with_mode("λ.1", ParseMode::Auto).unwrap();
        assert_eq!(expr, abs!(1, 1));

        // Auto-detect Named mode
        let expr = Parser::parse_with_mode("λx.x", ParseMode::Auto).unwrap();
        assert_eq!(expr, abs!(1, 1));

        // Force specific mode (1-based input)
        let expr = Parser::parse_with_mode("λ.1", ParseMode::DeBruijn).unwrap();
        assert_eq!(expr, abs!(1, 1));
    }

    #[test]
    fn test_whitespace_handling() {
        let expr = Parser::parse("   λ   x   .   x   ").unwrap();
        assert_eq!(expr, abs!(1, 1));

        let expr = Parser::parse("\n\tλx.\n\tx\n\t").unwrap();
        assert_eq!(expr, abs!(1, 1));
    }

    #[test]
    fn test_complex_expressions() {
        // Church boolean TRUE = λt.λf.t
        let expr = Parser::parse("λt.λf.t").unwrap();
        let expected = abs!(2, 2);
        assert_eq!(expr, expected);

        // Church boolean FALSE = λt.λf.f
        let expr = Parser::parse("λt.λf.f").unwrap();
        let expected = abs!(2, 1);
        assert_eq!(expr, expected);

        // Church numeral 2 = λf.λx.f (f x)
        let expr = Parser::parse("λf.λx.f (f x)").unwrap();
        let expected = abs!(2, app!(2, app!(2, 1)));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_roundtrip_simple_expressions() {
        let test_cases = vec!["λ.1", "λλ.2", "λλλ.3", "λx.x", "λx y.x", "λx y.y"];

        for input in test_cases {
            let expr = Parser::parse(input).unwrap();
            let output = expr.to_string();
            let reparsed = Parser::parse(&output).unwrap();
            assert_eq!(expr, reparsed, "Roundtrip failed for input: {input}");
        }
    }

    #[test]
    fn test_roundtrip_complex_expressions() {
        let test_cases = vec!["λf.λx.f x", "λt.λf.t", "λt.λf.f", "λx.λy.λz.x y z"];

        for input in test_cases {
            let expr = Parser::parse(input).unwrap();
            let output = expr.to_string();
            let reparsed = Parser::parse(&output).unwrap();
            assert_eq!(expr, reparsed, "Roundtrip failed for input: {input}");
        }

        // Test a case with explicit parentheses
        let input = "λf.λx.f (f x)";
        let expr = Parser::parse(input).unwrap();
        let output = expr.to_string();
        // The output preserves the parentheses in the application, but consecutive
        // abstractions are simplified
        assert_eq!(output, "λλ.2 (2 1)");
        let reparsed = Parser::parse(&output).unwrap();
        assert_eq!(expr, reparsed, "Roundtrip failed for input: {input}");
    }

    #[test]
    fn test_roundtrip_with_parentheses() {
        let test_cases = vec![
            ("(λx.x)", "λ.1"),
            ("λx.λy.x y", "λλ.2 1"), // Consecutive abstractions get simplified
            ("(λ.1)", "λ.1"),
        ];

        for (input, expected) in test_cases {
            let expr = Parser::parse(input).unwrap();
            let output = expr.to_string();
            assert_eq!(
                output, expected,
                "Display format incorrect for input: {input}"
            );

            let reparsed = Parser::parse(&output).unwrap();
            assert_eq!(expr, reparsed, "Roundtrip failed for input: {input}");
        }
    }

    #[test]
    fn test_multi_argument_application() {
        // Test with lambda context
        let expr = Parser::parse("λf.λa.λb.λc.f a b c").unwrap();
        let expected = abs!(4, app!(4, 3, 2, 1));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_multi_argument_debruijn() {
        // Test multi-argument application with De Bruijn indices
        let expr = Parser::parse("λ.λ.λ.3 2 1").unwrap();
        let expected = abs!(3, app!(3, 2, 1));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_nested_multi_argument() {
        // Test nested multi-argument applications
        let expr = Parser::parse("λf.λg.λx.λy.f (g x) y").unwrap();
        let expected = abs!(4, app!(4, app!(3, 2), 1));
        assert_eq!(expr, expected);
    }
}
