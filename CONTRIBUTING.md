# Contributing to Kalkulator

Thank you for your interest in contributing to Kalkulator!

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Style

- Follow PEP 8 guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use type hints for all function signatures
- Document public functions with docstrings

### Example

```python
def solve_equation(eq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """Solve a single equation.
    
    Args:
        eq_str: Equation string (e.g., "x+1=0")
        find_var: Optional variable to solve for
        
    Returns:
        Dictionary with 'ok', 'type', and solution data
    """
    ...
```

## Testing

- Write unit tests for all new features
- Ensure tests pass before submitting PR
- Aim for >80% code coverage

### Running Tests

```bash
# All tests
python -m pytest tests/

# With coverage
python -m pytest --cov=kalkulator_pkg tests/

# Specific test
python -m pytest tests/test_parser.py::TestPreprocess
```

## Adding New Features

1. **Plan**: Document the feature in an issue first
2. **Implement**: Follow module structure and responsibilities
3. **Test**: Add comprehensive tests
4. **Document**: Update README and docstrings
5. **Submit**: Create a pull request with clear description

## Security Guidelines

- Never remove or weaken input validation
- All user input must pass through validation layers
- Test with malicious inputs (fuzzing encouraged)
- Document security assumptions

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run linting: `python -m flake8 kalkulator_pkg/`
5. Run tests: `python -m pytest tests/`
6. Commit with descriptive messages
7. Push and create a pull request

## Commit Messages

Use clear, descriptive commit messages:
- `fix: Correct parsing of nested matrices`
- `feat: Add support for complex number operations`
- `test: Add unit tests for inequality solver`

## Questions?

Open an issue with the `question` label for any questions about contributing.

