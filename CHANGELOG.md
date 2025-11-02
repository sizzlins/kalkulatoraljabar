# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CONTRIBUTING.md with branch, commit, and PR rules
- Pre-commit hooks with black, isort, ruff, flake8, mypy
- pyproject.toml with centralized tool configuration
- Enhanced CI workflow with lint, type-check, test, and build jobs
- requirements-dev.txt with pinned development dependencies
- Conventional Commits message format enforcement

### Changed
- Updated CI to use black line-length 88 (from 120)
- Standardized on ruff as primary linter (with flake8 fallback)
- Updated requirements.txt to remove dev dependencies

### Fixed
- Consistent code formatting across all modules
- Import ordering standardized

## [1.0.0] - 2025-01-XX

### Added
- Initial release of Kalkulator
- Core equation solving (linear, quadratic, polynomial)
- Pell equation detection and solving
- Inequality solving with chain support
- System of equations solving
- Calculus operations (differentiation, integration)
- Matrix determinant calculation
- ASCII and Matplotlib plotting support
- Sandboxed worker process execution
- Resource limits (CPU time, memory)
- Comprehensive input validation
- Security-focused expression parsing
- CLI with REPL mode
- Public Python API with typed returns
- Extensive test suite

### Security
- AST-based expression validation
- Whitelisted SymPy functions only
- Forbidden token detection
- Expression depth and node count limits
- Sandboxed worker processes with resource limits

## [0.1.0] - 2025-01-XX

### Added
- Initial development version
- Basic equation solving
- Core parser and solver modules

[Unreleased]: https://github.com/yourusername/kalkulator/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/kalkulator/releases/tag/v1.0.0
[0.1.0]: https://github.com/yourusername/kalkulator/releases/tag/v0.1.0

