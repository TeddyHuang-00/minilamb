WORKSPACE_FLAGS := "--workspace --all-targets --locked"

# Format code
format:
    cargo +nightly fmt --all
    cargo sort --workspace
    cargo sort-derives

# Check unused dependencies
deps:
    cargo +nightly udeps {{WORKSPACE_FLAGS}}

# Check for errors
check: format
    cargo clippy {{WORKSPACE_FLAGS}} --fix --allow-no-vcs
    @just format

# Unit tests
test: check
    cargo test

# Coverage report
coverage: check
    cargo tarpaulin {{WORKSPACE_FLAGS}} --out Html --output-dir coverage

