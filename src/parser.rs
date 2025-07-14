use crate::{
    expr::Expr,
    lexer::{Lexer, Token},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseMode {
    Auto,     // Detect format automatically
    DeBruijn, // Force De Bruijn parsing
    Named,    // Force named variable parsing
}

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    mode: ParseMode,
    context: Vec<String>, // Variable binding context for De Bruijn conversion
}

#[derive(Debug, Clone)]
pub enum ParseError {
    UnexpectedToken {
        expected: String,
        found: Token,
        position: usize,
    },
    UnboundVariable {
        name: String,
        position: usize,
    },
    InvalidDeBruijnIndex {
        index: usize,
        max_depth: usize,
        position: usize,
    },
    MixedVariableFormats {
        position: usize,
    },
    EmptyExpression {
        position: usize,
    },
}

impl Parser {
    #[must_use]
    pub const fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
            mode: ParseMode::Auto,
            context: Vec::new(),
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
        parser.parse_expression()
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
        parser.parse_expression()
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

        // Build nested abstractions
        let mut result = body;
        for _ in 0..lambda_count.max(bindings.len()) {
            result = Expr::Abs(Box::new(result));
        }

        Ok(result)
    }

    fn parse_application(&mut self) -> Result<Expr, ParseError> {
        // Parse left-associative application: f g h = ((f g) h)
        let mut expr = self.parse_atom()?;

        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RParen | Token::Eof)) {
            self.skip_whitespace();
            if matches!(self.peek(), Some(Token::RParen | Token::Eof)) {
                break;
            }
            let arg = self.parse_atom()?;
            expr = Expr::App(Box::new(expr), Box::new(arg));
        }

        Ok(expr)
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
                self.resolve_variable(&name)
            }
            Some(Token::Number(index)) => {
                let index = *index;
                self.advance();
                if matches!(self.mode, ParseMode::Named) {
                    return Err(ParseError::MixedVariableFormats {
                        position: self.current,
                    });
                }
                // Convert from 1-based user input to 0-based internal representation
                if index == 0 {
                    return Err(ParseError::InvalidDeBruijnIndex {
                        index,
                        max_depth: 0,
                        position: self.current,
                    });
                }
                Ok(Expr::Var(index - 1))
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

    fn resolve_variable(&self, name: &str) -> Result<Expr, ParseError> {
        // Convert variable name to De Bruijn index
        for (i, var) in self.context.iter().rev().enumerate() {
            if var == name {
                return Ok(Expr::Var(i));
            }
        }
        Err(ParseError::UnboundVariable {
            name: name.to_string(),
            position: self.current,
        })
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

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedToken {
                expected,
                found,
                position,
            } => {
                write!(
                    f,
                    "Expected {expected} but found {found:?} at position {position}"
                )
            }
            Self::UnboundVariable { name, position } => {
                write!(f, "Unbound variable '{name}' at position {position}")
            }
            Self::InvalidDeBruijnIndex {
                index,
                max_depth,
                position,
            } => {
                write!(
                    f,
                    "De Bruijn index {index} exceeds maximum depth {max_depth} at position {position}"
                )
            }
            Self::MixedVariableFormats { position } => {
                write!(
                    f,
                    "Mixed variable formats (named and De Bruijn) at position {position}"
                )
            }
            Self::EmptyExpression { position } => {
                write!(f, "Empty expression at position {position}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_de_bruijn_formats() {
        // λ.1 (identity) - 1-based input becomes 0-based internal
        let expr = Parser::parse("λ.1").unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));

        // λλ.2 (const function) - 2 becomes 1 internally
        let expr = Parser::parse("λλ.2").unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::Var(1))))));

        // λ.λ.2 1 (application) - 2->1, 1->0 internally
        let expr = Parser::parse("λ.λ.2 1").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::App(
            Box::new(Expr::Var(1)),
            Box::new(Expr::Var(0)),
        )))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_named_variable_formats() {
        // λx.x (identity)
        let expr = Parser::parse("λx.x").unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));

        // λx.λy.y x
        let expr = Parser::parse("λx.λy.y x").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::App(
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        )))));
        assert_eq!(expr, expected);

        // λx y.x (abbreviated abstraction)
        let expr = Parser::parse("λx y.x").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::Var(1)))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_alternative_lambda_symbols() {
        // Test \, /, | as lambda symbols
        let patterns = ["\\x.x", "/x.x", "|x.x"];
        for pattern in &patterns {
            let expr = Parser::parse(pattern).unwrap();
            assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));
        }
    }

    #[test]
    fn test_application_associativity() {
        // Use free variables for testing application parsing
        let expr = Parser::parse("λx.λy.λz.x y z").unwrap();
        // This should parse as λx.λy.λz.((x y) z)
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::Abs(Box::new(
            Expr::App(
                Box::new(Expr::App(Box::new(Expr::Var(2)), Box::new(Expr::Var(1)))),
                Box::new(Expr::Var(0)),
            ),
        ))))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_abstraction_precedence() {
        // λx.f g should parse as λx.(f g), not (λx.f) g
        // Test with bound variables only to avoid unbound variable errors
        let expr = Parser::parse("λx.λy.x y").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::App(
            Box::new(Expr::Var(1)),
            Box::new(Expr::Var(0)),
        )))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parentheses() {
        // Test explicit parentheses
        let expr = Parser::parse("(λx.x)").unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));

        // Test grouping in applications
        let expr = Parser::parse("λf.λx.f (f x)").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::App(
            Box::new(Expr::Var(1)),
            Box::new(Expr::App(Box::new(Expr::Var(1)), Box::new(Expr::Var(0)))),
        )))));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mixed_formats_error() {
        // Should error when mixing named variables and De Bruijn indices
        let result = Parser::parse("λx.0");
        assert!(matches!(
            result,
            Err(ParseError::MixedVariableFormats { .. })
        ));
    }

    #[test]
    fn test_unbound_variable_error() {
        // Should error on unbound variables
        let result = Parser::parse("λx.y");
        assert!(matches!(result, Err(ParseError::UnboundVariable { .. })));
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
        let result = Parser::parse("λ.0");
        assert!(matches!(
            result,
            Err(ParseError::InvalidDeBruijnIndex { .. })
        ));
    }

    #[test]
    fn test_parse_mode_detection() {
        // Auto-detect De Bruijn mode (1-based input)
        let expr = Parser::parse_with_mode("λ.1", ParseMode::Auto).unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));

        // Auto-detect Named mode
        let expr = Parser::parse_with_mode("λx.x", ParseMode::Auto).unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));

        // Force specific mode (1-based input)
        let expr = Parser::parse_with_mode("λ.1", ParseMode::DeBruijn).unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));
    }

    #[test]
    fn test_whitespace_handling() {
        let expr = Parser::parse("   λ   x   .   x   ").unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));

        let expr = Parser::parse("\n\tλx.\n\tx\n\t").unwrap();
        assert_eq!(expr, Expr::Abs(Box::new(Expr::Var(0))));
    }

    #[test]
    fn test_complex_expressions() {
        // Church boolean TRUE = λt.λf.t
        let expr = Parser::parse("λt.λf.t").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::Var(1)))));
        assert_eq!(expr, expected);

        // Church boolean FALSE = λt.λf.f
        let expr = Parser::parse("λt.λf.f").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::Var(0)))));
        assert_eq!(expr, expected);

        // Church numeral 2 = λf.λx.f (f x)
        let expr = Parser::parse("λf.λx.f (f x)").unwrap();
        let expected = Expr::Abs(Box::new(Expr::Abs(Box::new(Expr::App(
            Box::new(Expr::Var(1)),
            Box::new(Expr::App(Box::new(Expr::Var(1)), Box::new(Expr::Var(0)))),
        )))));
        assert_eq!(expr, expected);
    }
}
