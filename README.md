# minilamb

A minimal lambda calculus library in Rust.

[![Crates.io](https://img.shields.io/crates/v/minilamb)](https://crates.io/crates/minilamb)
[![Documentation](https://docs.rs/minilamb/badge.svg)](https://docs.rs/minilamb)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

## Features

- **Complete Lambda Calculus**: Full support for lambda abstractions, applications, and β-reduction
- **Mixed Variable System**: Support for both De Bruijn indices and named variables
- **Multi-Format Parser**: Supports `λ`, `\`, `/`, `|` lambda symbols with comprehensive error handling
- **Normal Order Evaluation**: Step-limited β-reduction with expression simplification
- **Zero Unsafe Code**: Memory-safe implementation with comprehensive error handling
- **Ergonomic Macros**: Convenient `abs!` and `app!` macros for expression construction

## Quick Start

Add minilamb to your `Cargo.toml`:

```toml
[dependencies]
minilamb = "0.1"
```

### Basic Usage

```rust
use minilamb::{parse, evaluate, abs, app};

// Parse and evaluate expressions
let expr = parse("(λx.x) (λy.y)")?;
let result = evaluate(&expr, 1000)?;
println!("{result}"); // λy.y

// Use ergonomic macros
let identity = abs!(1); // λx.x using De Bruijn index
let application = app!("f", "x", "y"); // f x y

// Parse different lambda formats
let expressions = ["λx.x", "\\x.x", "/x.x", "|x.x"];
for expr_str in expressions {
    let expr = parse(expr_str)?;
    println!("{expr_str} -> {expr}");
}
```

### Advanced Examples

```rust
use minilamb::{parse_and_evaluate, Expr};

// Church encodings
let church_true = parse("λt.λf.t")?;
let church_false = parse("λt.λf.f")?;
let church_two = parse("λf.λx.f (f x)")?;

// Evaluation with step limits
let result = parse_and_evaluate("(λx.λy.x) a b", 100)?;
println!("{result}"); // a
```

## Architecture

minilamb uses a 4-variant expression system optimized for both correctness and binary size:

```rust
pub enum Expr {
    BoundVar(usize),        // De Bruijn indices (1-based)
    FreeVar(String),        // Named variables
    Abs(usize, Box<Expr>),  // Lambda abstractions with explicit levels
    App(Vec<Expr>),         // Multi-argument applications
}
```

### Core Operations

- **Parsing**: Recursive descent parser with automatic format detection
- **Evaluation**: Normal order β-reduction with configurable step limits
- **Simplification**: Advanced expression normalization beyond basic evaluation
- **Variable Handling**: Sophisticated De Bruijn shift and substitution operations

## Examples

Run the included demo to see minilamb in action:

```bash
cargo run --example demo
```

This demonstrates:

- Alternative lambda symbol parsing (`λ`, `\`, `/`, `|`)
- De Bruijn vs named variable equivalence
- Church encodings (booleans, numerals)
- Complex expression evaluation
- Error handling and step limits

## Development

### Quality Checks

Always run these commands before committing:

```bash
# Format, lint, and test
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test

# Or use the justfile
just check
```

### Project Structure

```
minilamb/
├── src/
│   ├── lib.rs      # Public API and convenience functions
│   ├── expr.rs     # Expression types and IntoExpr trait
│   ├── engine.rs   # β-reduction evaluation and simplification
│   ├── parser.rs   # Recursive descent parser
│   └── lexer.rs    # Multi-format tokenizer
├── examples/
│   └── demo.rs     # Usage demonstration
└── tests/
    └── integration_test.rs # End-to-end tests
```

### Testing

The project maintains 135+ tests covering:

- Core operations (shift, substitute, simplify)
- Parser formats (De Bruijn, named variables, multi-argument)
- Evaluation correctness (single-step, full reduction)
- Standard library (combinators, Church encodings)
- Error handling and edge cases

## License

This project is licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-Apache](LICENSE-Apache))

at your option.

## Contributing

Contributions are welcome! Please ensure all quality checks pass:

```bash
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

The project follows strict security and quality guidelines:

- No unsafe code permitted
- Comprehensive error handling (no `.unwrap()` or `.expect()`)
- Extensive test coverage
- Clear documentation and examples
