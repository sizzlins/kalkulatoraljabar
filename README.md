# Kalkulator - Algebraic Calculator

A modular, secure algebraic calculator with support for equations, inequalities, calculus, and matrix operations.

## Features

- **Equation Solving**: Single equations, systems, Pell equations
- **Inequality Solving**: Chained inequalities and constraints
- **Calculus**: Differentiation, integration, factoring, expansion
- **Matrix Operations**: Matrix creation and determinant calculation
- **Secure Sandboxing**: Worker processes with resource limits
- **High Performance**: Persistent worker pool with caching

## Architecture

The codebase is organized into modular components:

```
kalkulator_pkg/
├── config.py      # Configuration constants and whitelists
├── parser.py      # Input preprocessing and expression parsing
├── worker.py      # Sandboxed evaluation worker pool
├── solver.py      # Equation and inequality solving logic
├── cli.py         # Command-line interface and REPL
└── types.py       # Type definitions and result dataclasses
```

### Module Responsibilities

- **parser**: Input sanitization, preprocessing, expression tree validation
- **worker**: Sandboxed evaluation via persistent worker pool with resource limits
- **solver**: SymPy-based solving algorithms for equations/inequalities
- **cli**: User interface, argument parsing, output formatting

## Security

### Sandboxing

- Worker processes run with resource limits (CPU time, memory)
- Input validation through denylist and expression tree validation
- Whitelist-based function access (only allowed SymPy functions)
- Expression complexity limits (depth, node count, input length)

### Validation Layers

1. **String-level**: Denylist filtering for dangerous tokens
2. **Preprocessing**: Balanced parentheses, input length checks
3. **AST-level**: Expression tree traversal validating node types
4. **Resource Limits**: CPU and memory caps in worker processes

### Input Limits

- `MAX_INPUT_LENGTH`: 10,000 characters
- `MAX_EXPRESSION_DEPTH`: 100 levels
- `MAX_EXPRESSION_NODES`: 5,000 nodes

## Installation

```bash
# Install dependencies
pip install sympy

# Run the calculator
python kalkulator.py

# Or use PyInstaller to create executable
pyinstaller --onefile --console --collect-all sympy --collect-all mpmath kalkulator.py
```

## Usage

### Command Line

```bash
# Evaluate an expression
python kalkulator.py -e "2+2"

# Solve an equation
python kalkulator.py -e "x^2 - 1 = 0"

# JSON output
python kalkulator.py -j -e "sin(pi/2)"

# Configuration options
python kalkulator.py -t 30 -e "complex_expression"  # Set timeout
python kalkulator.py --no-numeric-fallback -e "equation"  # Disable numeric fallback
python kalkulator.py -p 10 -e "pi"  # Set precision
python kalkulator.py --cache-size 512  # Set cache size
python kalkulator.py --max-nsolve-guesses 20  # Limit numeric guesses
python kalkulator.py --worker-mode pool  # Worker execution mode
python kalkulator.py --method numeric -e "equation"  # Force numeric solving

# Logging
python kalkulator.py --log-level DEBUG --log-file kalkulator.log -e "expression"
```

### REPL Mode

```bash
python kalkulator.py
```

Commands:
- `help` - Show help text
- `clearcache` - Clear expression caches
- `quit` / `exit` - Exit REPL

### Examples

```python
# Basic arithmetic
>>> 2+2
4

# Variables
>>> x + 1 = 0
x = -1

# Calculus
>>> diff(x^3, x)
3*x^2

# Integration
>>> integrate(sin(x), x)
-cos(x)

# Matrices
>>> Matrix([[1,2],[3,4]])
Matrix([[1, 2], [3, 4]])

>>> det(Matrix([[1,2],[3,4]]))
-2

# Systems
>>> x+y=3, x-y=1
{x: 2, y: 1}
```

## Configuration

Key configuration options in `kalkulator_pkg/config.py`:

- `WORKER_CPU_SECONDS`: CPU time limit per worker (default: 30s)
- `WORKER_AS_MB`: Memory limit per worker (default: 400MB)
- `WORKER_TIMEOUT`: Wall-clock timeout (default: 60s)
- `WORKER_POOL_SIZE`: Number of concurrent workers (default: 1)
- `NUMERIC_FALLBACK_ENABLED`: Enable numeric root-finding fallback
- `OUTPUT_PRECISION`: Decimal precision for output (default: 6)

## Error Codes

The calculator returns structured error responses with codes:

- `VALIDATION_ERROR`: Input validation failure
- `PARSE_ERROR`: Expression parsing failure
- `EVAL_ERROR`: Evaluation failure
- `SOLVER_ERROR`: Solving failure
- `TIMEOUT`: Operation timed out
- `CANCELLED`: Request was cancelled
- `TOO_COMPLEX`: Expression exceeds complexity limits
- `FORBIDDEN_FUNCTION`: Unallowed function used

## Public API

For programmatic use, import from `kalkulator_pkg.api`:

```python
from kalkulator_pkg.api import (
    evaluate, solve_equation, solve_inequality, solve_system,
    diff, integrate_expr, det, plot
)

# Evaluate expressions
result = evaluate("2+2")
print(result.result)  # "4"

# Solve equations
sol = solve_equation("x^2 - 1 = 0")
print(sol.exact)  # ["-1", "1"]

# Solve inequalities
ineq = solve_inequality("x > 0")
print(ineq.solutions)

# Solve systems
sys_sol = solve_system("x+y=3, x-y=1")
print(sys_sol.system_solutions)

# Calculus operations
derivative = diff("x^3", variable="x")  # 3*x^2
integral = integrate_expr("sin(x)", variable="x")  # -cos(x)

# Matrix operations
determinant = det("Matrix([[1,2],[3,4]])")  # -2

# Plotting (requires matplotlib for graphical plots)
plot_result = plot("x^2", ascii=True)  # ASCII plot
# plot("x^2")  # Opens matplotlib window
```

All API functions return typed dataclasses (no side effects, no print statements).

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_parser.py

# Run with coverage
python -m pytest --cov=kalkulator_pkg tests/
```

## Development

### Code Style

- Follow PEP 8
- Use type hints where possible
- Document public functions
- Handle specific exceptions (avoid bare `except Exception`)

### Adding New Features

1. Add allowed functions to `ALLOWED_SYMPY_NAMES` in `config.py`
2. Update validation in `parser.py` if needed
3. Add tests in `tests/`
4. Update documentation

## Security Considerations

### Deployment Recommendations

1. **Resource Limits**: Ensure resource module is available on Unix systems
2. **Windows Limitations**: Resource limits don't apply on Windows; consider containerization
3. **Input Validation**: The multi-layer validation provides defense-in-depth
4. **Worker Isolation**: Workers run in separate processes with restricted resources

### Known Limitations

- SymPy's parser uses `eval` internally; validation layers mitigate risk
- Windows systems don't enforce resource limits (OS limitation)
- Complex expressions may still consume significant memory before limits apply

## License

Property of Muhammad Akhiel al Syahbana. Not to be freely distributed without author's permission.

## Version

Current version: 2025-10-31

