"""Input parsing and preprocessing module.

This module handles:
- Input sanitization and validation
- Expression preprocessing (symbol conversion, exponent handling, etc.)
- SymPy expression parsing with security validation
- Result formatting (superscripts, numbers, solutions)
- Balancing checks for parentheses/brackets
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import sympy as sp
from sympy import parse_expr

from .config import (
    ALLOWED_SYMPY_NAMES,
    AMBIG_FRACTION_REGEX,
    CACHE_SIZE_PARSE,
    DIGIT_LETTERS_REGEX,
    MAX_EXPRESSION_DEPTH,
    MAX_EXPRESSION_NODES,
    MAX_INPUT_LENGTH,
    OUTPUT_PRECISION,
    PERCENT_REGEX,
    SQRT_UNICODE_REGEX,
    TRANSFORMATIONS,
)
from .types import ValidationError


def superscriptify(input_str: str) -> str:
    """Convert numeric string to Unicode superscript characters.

    Args:
        input_str: Input string with digits and '-' (e.g., "123", "-5")

    Returns:
        String with superscript Unicode characters (e.g., "¹²³", "⁻⁵")
    """
    mapping = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻",
        "n": "ⁿ",
    }
    return "".join(mapping.get(char, char) for char in input_str)


def format_superscript(expr_str: str) -> str:
    """Replace Python power notation (**) with Unicode superscripts.

    Args:
        expr_str: Expression string (e.g., "x**2", "x**-3")

    Returns:
        String with superscripts (e.g., "x²", "x⁻³")
    """
    return re.sub(r"\*\*(\-?\d+)", lambda m: superscriptify(m.group(1)), expr_str)


def format_number(val: Any, precision: int = OUTPUT_PRECISION) -> str:
    """Format a numeric value with specified precision.

    Args:
        val: Numeric value to format
        precision: Number of significant digits (default: OUTPUT_PRECISION)

    Returns:
        Formatted string representation of the number
    """
    try:
        fmt = "{:." + str(int(precision)) + "g}"
        return fmt.format(float(val))
    except (ValueError, TypeError, OverflowError):
        # Fallback for non-numeric or invalid values
        return str(val)


def format_solution(sol: Any, exact: bool = True) -> str:
    """Format a solution value or tuple for display.

    Args:
        sol: Solution value(s) - can be a number, tuple, or list
        exact: If True, use superscript formatting; if False, use numeric formatting

    Returns:
        Formatted string representation
    """
    if isinstance(sol, (tuple, list)):
        return "(" + ", ".join(format_solution(v, exact) for v in sol) + ")"
    return format_superscript(str(sol)) if exact else format_number(sol)


def prettify_expr(expr_str: str) -> str:
    """Convert expression string to more readable format.

    Replaces 'sqrt(' with '√' and '*' with '×' for better readability.

    Args:
        expr_str: Expression string (e.g., "sqrt(4)*2")

    Returns:
        Prettified string (e.g., "√(4)×2")
    """
    result = re.sub(r"sqrt\(([^)]+)\)", r"√(\1)", expr_str)
    result = result.replace("*", "×")
    return result


def is_balanced(input_str: str) -> Tuple[bool, Optional[int]]:
    """Check if parentheses/brackets are balanced. Returns (is_balanced, error_position)."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: List[Tuple[str, int]] = []  # (char, position)
    for i, char in enumerate(input_str):
        if char in pairs:
            stack.append((char, i))
        elif char in pairs.values():
            if not stack:
                return False, i
            opening, pos = stack.pop()
            if pairs[opening] != char:
                return False, i
    if stack:
        return False, stack[0][1]  # Return position of first unmatched
    return True, None


# Basic denylist to avoid dangerous tokens before SymPy parsing
FORBIDDEN_TOKENS = (
    "__",
    "import",
    "lambda",
    "eval",
    "exec",
    "open",
    "os.",
    "sys.",
    "subprocess",
    "builtins",
    "getattr",
    "setattr",
    "delattr",
    "compile",
    "globals",
    "locals",
    "__class__",
    "__mro__",
    "__subclasses__",
    "memoryview",
    "bytes",
    "bytearray",
    "__import__",
)


def _validate_expression_tree(
    expr: Any, depth: int = 0, node_count: List[int] = None
) -> None:
    """Validate expression tree structure - reject dangerous nodes."""
    if node_count is None:
        node_count = [0]
    node_count[0] += 1
    if node_count[0] > MAX_EXPRESSION_NODES:
        raise ValidationError(
            f"Expression too complex (>{MAX_EXPRESSION_NODES} nodes)", "TOO_COMPLEX"
        )
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValidationError(
            f"Expression too deeply nested (>{MAX_EXPRESSION_DEPTH} levels)", "TOO_DEEP"
        )

    # Allow safe types - Numbers (includes Integer, Rational, Float, Complex)
    if isinstance(expr, (sp.Symbol, sp.Number)):
        return
    if isinstance(expr, sp.Function):
        # Only allow whitelisted functions
        # Try multiple methods to get function name
        func_name = None
        if hasattr(expr.func, "__name__"):
            func_name = expr.func.__name__
        elif hasattr(expr.func, "name"):
            func_name = expr.func.name
        else:
            # Fallback: extract from string representation
            func_str = str(expr.func)
            if "." in func_str:
                func_name = func_str.split(".")[-1].split("'")[0]
            else:
                func_name = func_str.split("'")[1] if "'" in func_str else func_str

        if func_name and func_name not in ALLOWED_SYMPY_NAMES:
            # Audit log blocked function
            try:
                from .logging_config import get_logger

                logger = get_logger("parser")
                logger.warning(
                    "Blocked forbidden function",
                    extra={"forbidden_function": func_name},
                )
            except ImportError:
                pass
            raise ValidationError(
                f"Function '{func_name}' not allowed", "FORBIDDEN_FUNCTION"
            )
        # Recurse into args
        for arg in expr.args:
            _validate_expression_tree(arg, depth + 1, node_count)
        return
    # Allow safe arithmetic operations
    if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
        for arg in expr.args:
            _validate_expression_tree(arg, depth + 1, node_count)
        return
    # Handle special SymPy singleton objects (they're still Numbers)
    # Check if it's a well-known singleton value
    try:
        if expr in (
            sp.S.One,
            sp.S.Zero,
            sp.S.NegativeOne,
            sp.S.Half,
            sp.S.NaN,
            sp.oo,
            -sp.oo,
        ):
            return
    except (ValueError, TypeError, AttributeError):
        # Expected for some singleton checks
        pass
    if isinstance(expr, sp.Matrix):
        for row in expr.tolist():
            for elem in row:
                _validate_expression_tree(elem, depth + 1, node_count)
        return
    # Check for dangerous types explicitly
    expr_type = type(expr).__name__
    # Reject Attribute access (could expose internals)
    if expr_type == "Attribute":
        raise ValidationError(
            f"Dangerous expression type '{expr_type}' not allowed", "FORBIDDEN_TYPE"
        )

    # Allow other SymPy Basic types (they're generally safe)
    if isinstance(expr, sp.Basic):
        # For expressions with args, validate children
        if hasattr(expr, "args") and expr.args:
            for arg in expr.args:
                _validate_expression_tree(arg, depth + 1, node_count)
        # For relational operators, validate both sides
        if hasattr(expr, "lhs") and hasattr(expr, "rhs"):
            _validate_expression_tree(expr.lhs, depth + 1, node_count)
            _validate_expression_tree(expr.rhs, depth + 1, node_count)
        return

    # Reject anything that's not a SymPy Basic type
    raise ValidationError(
        f"Expression type '{expr_type}' not allowed", "FORBIDDEN_TYPE"
    )


def preprocess(input_str: str, skip_exponent_conversion: bool = False) -> str:
    """Preprocess input string for parsing.

    Applies transformations:
    - Validates input length and forbidden tokens
    - Standardizes mathematical symbols (unicode variants to ASCII)
    - Converts exponents (^ to **, superscripts to **)
    - Handles percentages (50% -> (50/100))
    - Converts Unicode square root (√) to sqrt(
    - Inserts implicit multiplication (2x -> 2*x)
    - Validates balanced parentheses/brackets

    Args:
        input_str: Raw input string from user
        skip_exponent_conversion: If True, skips ^ and superscript conversion

    Returns:
        Preprocessed and sanitized string ready for SymPy parsing

    Raises:
        ValidationError: If input is too long, contains forbidden tokens,
                        or has unbalanced parentheses/brackets
    """
    # Input size check
    input_str = input_str.strip() if input_str else ""
    if not input_str:
        raise ValidationError("Input cannot be empty", "EMPTY_INPUT")
    if len(input_str) > MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input too long (>{MAX_INPUT_LENGTH} characters)", "TOO_LONG"
        )

    # Check for unterminated quotes (common syntax error)
    if input_str.count('"') % 2 != 0:
        raise ValidationError(
            "Unmatched double quotes detected. Check that all opening quotes have matching closing quotes.",
            "UNMATCHED_QUOTES",
        )
    if input_str.count("'") % 2 != 0:
        raise ValidationError(
            "Unmatched single quotes detected. Check that all opening quotes have matching closing quotes.",
            "UNMATCHED_QUOTES",
        )

    lowered = input_str.strip().lower()
    for tok in FORBIDDEN_TOKENS:
        if tok in lowered:
            # Audit log blocked input
            try:
                from .logging_config import get_logger

                logger = get_logger("parser")
                logger.warning(
                    "Blocked input containing forbidden token",
                    extra={
                        "forbidden_token": tok,
                        "input_length": len(input_str),
                        "input_preview": (
                            input_str[:100]
                            if len(input_str) <= 100
                            else input_str[:100] + "..."
                        ),
                    },
                )
            except ImportError:
                pass
            raise ValidationError(
                f"Input contains forbidden token: {tok}", "FORBIDDEN_TOKEN"
            )

    processed_str = input_str.strip()
    processed_str = processed_str.replace("−", "-").replace("–", "-")
    processed_str = processed_str.replace("Δ", "Delta")
    processed_str = processed_str.replace(":", "/")
    processed_str = processed_str.replace("×", "*")
    # Standardize inequality variations
    processed_str = processed_str.replace("=>", ">=")
    processed_str = processed_str.replace("=<", "<=")

    if not skip_exponent_conversion:
        processed_str = processed_str.replace("^", "**")
        from_superscript_map = {
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁻": "-",
            "ⁿ": "n",
        }
        pattern = re.compile(f"([{''.join(from_superscript_map.keys())}]+)")

        def from_superscript_converter(m):
            normal_str = "".join([from_superscript_map[char] for char in m.group(1)])
            return f"**{normal_str}"

        processed_str = pattern.sub(from_superscript_converter, processed_str)

    processed_str = PERCENT_REGEX.sub(r"(\1/100)", processed_str)
    processed_str = SQRT_UNICODE_REGEX.sub("sqrt(", processed_str)
    processed_str = AMBIG_FRACTION_REGEX.sub(r"((\1)/(\2))", processed_str)
    processed_str = DIGIT_LETTERS_REGEX.sub(r"\1*\2", processed_str)
    processed_str = re.sub(r"\s+", " ", processed_str).strip()

    # Apply sub-expression caching: replace cached sub-expressions with their values
    # This speeds up expressions like "(2+2)/2" by using cached "2+2" -> "4"
    # Example: If "2+2" is cached as "4", then "(2+2)/2" becomes "4/2" before parsing
    try:
        from .cache_manager import get_cached_subexpr

        # Strategy: Find parenthesized sub-expressions and check cache
        # Process from innermost to outermost to handle nested expressions
        paren_pattern = re.compile(r"\(([^()]+)\)")
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            matches = list(paren_pattern.finditer(processed_str))
            if not matches:
                break  # No more parentheses

            changed = False
            # Process matches from right to left to avoid index shifting issues
            for match in reversed(matches):
                subexpr = match.group(
                    1
                )  # Content inside parentheses (without the parentheses)
                # Try to get cached value for this sub-expression
                cached_value = get_cached_subexpr(subexpr)
                if cached_value is not None and cached_value:
                    # Safety check: only replace if cached value is numeric
                    # Avoid replacing if cached value contains variables, operators, or parentheses
                    unsafe_chars = [
                        "x",
                        "y",
                        "z",
                        "X",
                        "Y",
                        "Z",
                        "a",
                        "b",
                        "c",
                        "(",
                        ")",
                        "*",
                        "/",
                        "+",
                        "-",
                        "=",
                    ]
                    if not any(c in cached_value for c in unsafe_chars):
                        # Replace the parenthesized sub-expression with its cached value
                        before = processed_str[: match.start()]
                        after = processed_str[match.end() :]
                        processed_str = before + cached_value + after
                        changed = True
                        break  # Restart scanning after replacement

            if not changed:
                break  # No more replacements possible
            iteration += 1
    except (ImportError, AttributeError, ValueError, TypeError):
        # If cache manager not available or error occurs, continue without sub-expression caching
        pass

    balanced, error_pos = is_balanced(processed_str)
    if not balanced:
        hint = ""
        if error_pos is not None:
            # Show context around error
            start = max(0, error_pos - 10)
            end = min(len(processed_str), error_pos + 10)
            context = processed_str[start:end]
            pointer = " " * (error_pos - start) + "^"
            hint = f" at position {error_pos}: ...{context}...\n{pointer}"
        raise ValidationError(
            f"Mismatched or unbalanced parentheses/brackets{hint}. Check parentheses around position {error_pos or 'unknown'}.",
            "UNBALANCED_PARENS",
        )
    return processed_str


@lru_cache(maxsize=CACHE_SIZE_PARSE)
def parse_preprocessed(expr_str: str) -> Any:
    """Parse and validate a preprocessed expression string."""
    expr = parse_expr(
        expr_str,
        local_dict=ALLOWED_SYMPY_NAMES,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )
    # Validate expression tree structure
    _validate_expression_tree(expr)
    return expr


def format_inequality_solution(sol_str: str) -> str:
    """Format SymPy inequality solution string for readability.

    Converts complex inequality representations to more readable forms.
    Handles compound inequalities like "a < x < b".

    Args:
        sol_str: Raw inequality solution string from SymPy

    Returns:
        Formatted inequality string
    """
    pattern = re.compile(
        r"\((.*?)\s*([<>=!]+)\s*(.*?)\)\s*&\s*\((.*?)\s*([<>=!]+)\s*(.*?)\)"
    )
    match = pattern.match(sol_str)
    if not match:
        return sol_str
    groups = [group.strip() for group in match.groups()]
    expr1, op1, var1, expr2, op2, var2 = groups
    if var1 == expr2:
        if op1 in ("<", "<=") and op2 in ("<", "<="):
            return f"{expr1} {op1} {var1} {op2} {var2}"
    elif expr1 == var2:
        op_map = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
        if op1 in op_map and op2 in op_map:
            return f"{var1} {op_map[op1]} {expr1} {op_map[op2]} {var2}"
    return sol_str


def split_top_level_commas(input_str: str) -> List[str]:
    """Split string by commas that are not inside (), [], or {}."""
    parts: List[str] = []
    current = []
    depth_paren = depth_brack = depth_brace = 0
    for char in input_str:
        if char == "," and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace = max(0, depth_brace - 1)
        current.append(char)
    # append last segment
    last = "".join(current).strip()
    if last:
        parts.append(last)
    return parts
