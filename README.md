# Kalkulator - Algebraic Calculator

A modular, secure algebraic calculator with support for equations, inequalities, calculus, and matrix operations.

## Features

- **Equation Solving**: Single equations, systems, Pell equations
- **Modulo Operations**: Modulo equations and systems of congruences (Chinese Remainder Theorem)
- **Inequality Solving**: Chained inequalities and constraints
- **Calculus**: Differentiation, integration, factoring, expansion
- **Matrix Operations**: Matrix creation and determinant calculation
- **Function Finding**: Advanced polynomial and rational function discovery from data points
- **Number Formats**: Automatic hexadecimal number detection and conversion
- **User-Friendly Errors**: Clear error messages with helpful hints and suggestions
- **Secure Sandboxing**: Worker processes with resource limits
- **High Performance**: Persistent worker pool with caching

## Architecture

The codebase is organized into modular components:

```
kalkulator_pkg/
├── config.py                  # Configuration constants and whitelists
├── parser.py                  # Input preprocessing and expression parsing
├── worker.py                  # Sandboxed evaluation worker pool
├── solver.py                  # Equation and inequality solving logic
├── cli.py                     # Command-line interface and REPL
├── function_manager.py        # Function definition and finding algorithms
├── function_finder_advanced.py # Advanced function finding capabilities
├── cache_manager.py           # Cache persistence and management
├── types.py                   # Type definitions and result dataclasses
└── api.py                     # Public Python API
```

### Module Responsibilities

- **parser**: Input sanitization, preprocessing, expression tree validation, hex number detection
- **worker**: Sandboxed evaluation via persistent worker pool with resource limits
- **solver**: SymPy-based solving algorithms for equations/inequalities, modulo operations, CRT
- **cli**: User interface, argument parsing, output formatting, REPL loop
- **function_manager**: Function definition, evaluation, and advanced function finding from data
- **function_finder_advanced**: Constant detection, high-precision parsing, sparse regression
- **cache_manager**: Persistent cache storage and retrieval
- **api**: Public Python API with typed returns

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

### Python API

```python
from kalkulator_pkg.api import evaluate, solve_equation, solve_inequality, solve_system

# Evaluate expressions
result = evaluate("2 + 2")
print(result.result)  # "4"

result = evaluate("sin(pi/2)")
print(result.result)  # "1"

# Solve equations
result = solve_equation("x + 1 = 0")
print(result.exact)  # ["-1"]

result = solve_equation("x^2 - 4 = 0")
print(result.exact)  # ["-2", "2"]

# Solve inequalities
result = solve_inequality("x > 0")
print(result.solutions)  # {"x": "x > 0"}

# Solve systems
result = solve_system("x+y=3, x-y=1")
print(result.system_solutions)  # [{"x": "2", "y": "1"}]

# Solve modulo equations
result = solve_equation("x % 2 = 0")
print(result.exact)  # Parametric solution

# Solve system of congruences (Chinese Remainder Theorem)
result = solve_system("x = 1 % 2, x = 3 % 6, x = 3 % 7")
print(result.system_solutions)  # System of congruences solution
```

**Error Handling:**

All API functions return typed dataclasses with `ok` field:
- `ok=True`: Operation succeeded
- `ok=False`: Operation failed, check `error` field

```python
from kalkulator_pkg.api import evaluate

result = evaluate("__import__('os')")
if not result.ok:
    print(f"Error: {result.error}")  # Error message
    # Some results also include error_code for programmatic handling
```

For complete API documentation with examples, see `kalkulator_pkg/api.py` docstrings.

### REPL Mode

```bash
python kalkulator.py
```

**REPL Commands:**
- `help` - Show comprehensive help text
- `quit`, `exit` - Exit the calculator
- `clearcache` - Clear all cached expressions
- `showcache [all]` - Show cached expressions (add 'all' for complete list)
- `savecache [file]` - Save cache to file (default: cache_backup.json)
- `loadcache [replace] [file]` - Load cache from file
- `timing [on|off]` - Enable/disable calculation time display
- `cachehits [on|off]` - Enable/disable cache hit tracking
- `showcachehits` - Show which expressions used cache
- `health` - Run health check to verify dependencies
- `plot <expr> [options]` - Plot functions (requires matplotlib)
  - Options: `variable=x`, `x_min=-10`, `x_max=10`, `points=100`, `--save filename`
  - Example: `plot sin(x), x_min=-pi, x_max=pi`

**Function Features:**
- Define: `f(x)=2*x`, `g(x,y)=x+y`
- Evaluate: `f(2)`, `g(1,2)`, `g(f(5))` (nested calls)
- Find from data: `f(1)=1, f(2)=2, find f(x)` (polynomial interpolation)
- Advanced function finding:
  - Discovers rational functions (e.g., `x*y/z^2` for Newton's gravitational law)
  - Extended basis with inverse terms (`1/z`, `1/z^2`, `x*y/z^2`, etc.)
  - Uses exact rational arithmetic for precise coefficients
  - Constant detection (π, e, sqrt(2), etc.) in coefficients

### Examples

```python
# Basic arithmetic
>>> 2+2
4

# Variables
>>> x + 1 = 0
x = -1

# Modulo operations
>>> x % 2 = 0
x = 2*t (for integer t)
Examples: x = 0, 2, 4, 6, ...

# System of congruences (Chinese Remainder Theorem)
>>> x = 1 % 2, x = 3 % 6, x = 3 % 7
Solution: x == 3 (mod 42)

# Hexadecimal numbers
>>> 0x123abc
1194684

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

# Function finding (rational functions)
>>> f(15, 299792458)=1348132768105226460, find f(x,y)
f(x, y) = x*y/y^2  # Discovers rational relationships

# Side effects
>>> print("Hello world")
Hello world
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
- `INCOMPLETE_EXPRESSION`: Expression ends with operator or backslash
- `SYNTAX_ERROR`: Invalid syntax with helpful hints

**User-Friendly Error Messages:**
- Clear, actionable error messages with suggestions
- Automatic detection of common mistakes (incomplete expressions, typos, etc.)
- Helpful hints for syntax errors and invalid input

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

1. **Resource Limits (Unix)**: On Unix systems, resource limits are automatically applied via the `resource` module
2. **Windows Limitations**: ⚠️ **Resource limits do not apply on Windows** (OS limitation). See `SECURITY.md` for mitigation strategies including:
   - Docker containerization with resource limits
   - Process isolation using restricted user accounts
   - External resource monitoring
3. **Input Validation**: The multi-layer validation provides defense-in-depth
4. **Worker Isolation**: Workers run in separate processes with restricted resources (Unix) or process isolation (Windows with mitigation)

### Known Limitations

- SymPy's parser uses `eval` internally; validation layers mitigate risk
- Windows systems don't enforce resource limits (OS limitation)
- Complex expressions may still consume significant memory before limits apply

## License

Property of Muhammad Akhiel al Syahbana. Please give credit to author

## Version

Current version: 1.0.0 (see `pyproject.toml` for source of truth)

