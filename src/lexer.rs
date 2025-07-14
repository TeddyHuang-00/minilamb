#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// λ, \, /, |
    Lambda,
    /// .
    Dot,
    /// (
    LParen,
    /// )
    RParen,
    /// Variable names
    Ident(String),
    /// De Bruijn indices
    Number(usize),
    /// For explicit spacing in applications
    Whitespace,
    /// End of input
    Eof,
}

pub struct Lexer<'input> {
    chars: std::str::Chars<'input>,
    current_char: Option<char>,
}

impl<'input> Lexer<'input> {
    #[must_use]
    pub fn new(input: &'input str) -> Self {
        let mut chars = input.chars();
        let current_char = chars.next();
        Lexer {
            chars,
            current_char,
        }
    }

    /// Tokenizes the input string into a vector of tokens.
    ///
    /// # Errors
    /// Returns an error if an unexpected character is encountered.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        while let Some(ch) = self.current_char {
            match ch {
                // Lambda symbols
                'λ' | '\\' | '/' | '|' => {
                    tokens.push(Token::Lambda);
                    self.advance();
                }
                '.' => {
                    tokens.push(Token::Dot);
                    self.advance();
                }
                '(' => {
                    tokens.push(Token::LParen);
                    self.advance();
                }
                ')' => {
                    tokens.push(Token::RParen);
                    self.advance();
                }
                // Numbers (De Bruijn indices)
                '0'..='9' => {
                    let number = self.read_number()?;
                    tokens.push(Token::Number(number));
                }
                // Identifiers
                'a'..='z' | 'A'..='Z' | '_' => {
                    let ident = self.read_identifier();
                    tokens.push(Token::Ident(ident));
                }
                // Whitespace
                ' ' | '\t' | '\r' | '\n' => {
                    self.skip_whitespace();
                    // Only add whitespace token if it's significant for application parsing
                    if Self::is_significant_whitespace(&tokens) {
                        tokens.push(Token::Whitespace);
                    }
                }
                _ => return Err(format!("Unexpected character: '{ch}'")),
            }
        }

        tokens.push(Token::Eof);
        Ok(tokens)
    }

    fn advance(&mut self) {
        self.current_char = self.chars.next();
    }

    fn read_number(&mut self) -> Result<usize, String> {
        let mut number_str = String::new();
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if number_str.is_empty() {
            return Err("Empty number".to_string());
        }

        number_str
            .parse()
            .map_err(|_| format!("Invalid number: {number_str}"))
    }

    fn read_identifier(&mut self) -> String {
        let mut ident = String::new();
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        ident
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    const fn is_significant_whitespace(tokens: &[Token]) -> bool {
        // Whitespace is significant between atoms in applications
        if let Some(last_token) = tokens.last() {
            matches!(
                last_token,
                Token::Ident(_) | Token::Number(_) | Token::RParen
            )
        } else {
            false
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_lambda_symbols() {
        let test_cases = vec![
            ("λ", vec![Token::Lambda, Token::Eof]),
            ("\\", vec![Token::Lambda, Token::Eof]),
            ("/", vec![Token::Lambda, Token::Eof]),
            ("|", vec![Token::Lambda, Token::Eof]),
        ];

        for (input, expected) in test_cases {
            let mut lexer = Lexer::new(input);
            let tokens = lexer.tokenize().unwrap();
            assert_eq!(tokens, expected, "Failed for input: {input}");
        }
    }

    #[test]
    fn test_tokenize_punctuation() {
        let mut lexer = Lexer::new("().");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![Token::LParen, Token::RParen, Token::Dot, Token::Eof]
        );
    }

    #[test]
    fn test_tokenize_numbers() {
        let mut lexer = Lexer::new("0 1 42 123");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Number(0),
                Token::Whitespace,
                Token::Number(1),
                Token::Whitespace,
                Token::Number(42),
                Token::Whitespace,
                Token::Number(123),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_identifiers() {
        let mut lexer = Lexer::new("x y_var variable123");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".to_string()),
                Token::Whitespace,
                Token::Ident("y_var".to_string()),
                Token::Whitespace,
                Token::Ident("variable123".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_de_bruijn_expression() {
        let mut lexer = Lexer::new("λ.1");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![Token::Lambda, Token::Dot, Token::Number(1), Token::Eof]
        );
    }

    #[test]
    fn test_tokenize_named_expression() {
        let mut lexer = Lexer::new("\\x.x");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Lambda,
                Token::Ident("x".to_string()),
                Token::Dot,
                Token::Ident("x".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_application() {
        let mut lexer = Lexer::new("f g h");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("f".to_string()),
                Token::Whitespace,
                Token::Ident("g".to_string()),
                Token::Whitespace,
                Token::Ident("h".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_complex_expression() {
        let mut lexer = Lexer::new("(λx y.x) a");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Lambda,
                Token::Ident("x".to_string()),
                Token::Whitespace,
                Token::Ident("y".to_string()),
                Token::Dot,
                Token::Ident("x".to_string()),
                Token::RParen,
                Token::Whitespace,
                Token::Ident("a".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_whitespace_handling() {
        let mut lexer = Lexer::new("   λ   x   .   x   ");
        let tokens = lexer.tokenize().unwrap();
        // The lexer should handle whitespace but may insert whitespace tokens for
        // parsing
        let expected_tokens = vec![
            Token::Lambda,
            Token::Ident("x".to_string()),
            Token::Whitespace, // Between x and .
            Token::Dot,
            Token::Ident("x".to_string()),
            Token::Whitespace, // After final x
            Token::Eof,
        ];
        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    fn test_tokenize_error() {
        let mut lexer = Lexer::new("@");
        let result = lexer.tokenize();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unexpected character"));
    }

    #[test]
    fn test_empty_input() {
        let mut lexer = Lexer::new("");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens, vec![Token::Eof]);
    }
}
