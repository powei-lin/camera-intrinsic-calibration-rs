# Contributing

Thank you for your interest in contributing to camera-intrinsic-calibration!

## Getting Started

1. Fork the repository and clone it locally.
2. Ensure you have Rust installed (MSRV: 1.70).
3. Install pre-commit hooks: `pip install pre-commit && pre-commit install`.
4. Run `cargo test` to ensure everything works.
5. Make your changes and run tests/lints: `cargo test --all-features`, `cargo clippy`, `cargo fmt --check`.

## Pull Requests

- Follow the existing code style (enforced by CI).
- Add tests for new features.
- Update CHANGELOG.md for user-facing changes.
- Ensure CI passes.

## Reporting Issues

Use GitHub issues for bugs or feature requests. Provide details and steps to reproduce.

## Code of Conduct

Be respectful and constructive in all interactions.