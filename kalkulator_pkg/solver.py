"""Core equation and system solving module.

This module provides:
- Single equation solving (symbolic and numeric fallback)
- Inequality solving
- System of equations solving
- Special handling for Pell equations
- Expression evaluation
- Dedicated handlers for linear, quadratic, and polynomial equations

Note on simplify() usage:
- simplify() is NOT called by default to avoid performance issues
- simplify() is only used selectively:
  1. For identity/contradiction detection (on no-variable expressions)
  2. When explicitly needed for specific equation types
- Numeric fallback avoids simplify() entirely for better performance

All functions return dictionaries with 'ok' boolean and appropriate result fields.
Internal functions return dicts; public API (api.py) converts to typed dataclasses.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from .config import (
    COARSE_GRID_MIN_SIZE,
    MAX_NSOLVE_GUESSES,
    MAX_NSOLVE_STEPS,
    NUMERIC_FALLBACK_ENABLED,
    NUMERIC_TOLERANCE,
    ROOT_DEDUP_TOLERANCE,
    ROOT_SEARCH_TOLERANCE,
    VAR_NAME_RE,
)
from .parser import (
    parse_preprocessed,
    prettify_expr,
)
from .types import EvalResult, ParseError, ValidationError
from .worker import _worker_solve_cached, evaluate_safely

try:
    from .logging_config import get_logger

    logger = get_logger("solver")
except ImportError:
    # Fallback if logging not available
    class NullLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    logger = NullLogger()


def eval_user_expression(expr: str) -> EvalResult:
    data = evaluate_safely(expr)
    if not data.get("ok"):
        return EvalResult(ok=False, error=data.get("error") or "Unknown error")
    return EvalResult(
        ok=True,
        result=data.get("result"),
        approx=data.get("approx"),
        free_symbols=data.get("free_symbols"),
    )


def is_pell_equation_from_eq(eq: sp.Eq) -> bool:
    """
    Check if an equation is a Pell's equation.

    Args:
        eq: SymPy equation object

    Returns:
        True if the equation is a Pell's equation, False otherwise
    """
    syms = list(eq.free_symbols)
    if len(syms) != 2:
        return False
    try:
        if not eq.rhs.equals(1):
            return False
    except (AttributeError, TypeError, ValueError):
        # Invalid equation structure
        return False
    expanded_lhs = sp.expand(eq.lhs)
    x_sym, y_sym = syms[0], syms[1]

    # Check: coefficient of x^2 must be 1
    coeff_x2 = expanded_lhs.coeff(x_sym**2)
    # Use equals() for SymPy comparison since == might not work for all cases
    if not sp.sympify(coeff_x2).equals(1):
        return False

    # Check: coefficient of y^2 must be negative (non-zero)
    coeff_y2 = expanded_lhs.coeff(y_sym**2)
    if coeff_y2 == 0:
        return False

    # For Pell equation x^2 - D*y^2 = 1, we need D = -coeff_y2 to be a positive integer
    D = -coeff_y2
    try:
        D_val = int(sp.N(D))
        if D_val <= 0:
            return False
        # Check D is not a perfect square
        sqrt_D = sp.sqrt(D_val)
        if sqrt_D.is_rational:
            return False
    except (ValueError, TypeError):
        return False

    # Check no other terms exist (no constant, no xy terms, etc.)
    # Subtract the x^2 and y^2 terms and check remainder is 0
    remainder = expanded_lhs - x_sym**2 - coeff_y2 * y_sym**2
    if sp.simplify(remainder) != 0:
        return False

    return True


def fundamental_solution(D: int) -> Tuple[int, int]:
    """
    Find the fundamental solution to Pell's equation x^2 - D*y^2 = 1.

    Args:
        D: Non-square integer parameter

    Returns:
        Tuple (x, y) of the fundamental solution

    Raises:
        ValueError: If D is a perfect square or solution cannot be found
    """
    sqrt_D = sp.sqrt(D)
    if sqrt_D.is_rational:
        raise ValueError("D must be non-square for Pell's equation")

    # Use continued fraction to find the fundamental solution
    # For periodic CF [a0; [a1, a2, ..., an]], the fundamental solution
    # is found at the convergent before the end of the first period
    cf = sp.continued_fraction(sqrt_D)
    if not cf or len(cf) < 2:
        raise ValueError(f"Invalid continued fraction for D={D}")

    # Extract a0 and the periodic part
    a0 = cf[0] if isinstance(cf[0], (int, sp.Integer)) else int(cf[0])
    period = []
    if len(cf) > 1:
        period_item = cf[1]
        if isinstance(period_item, list):
            period = [
                int(x) if isinstance(x, (int, sp.Integer)) else int(sp.N(x))
                for x in period_item
            ]
        else:
            period = [
                (
                    int(period_item)
                    if isinstance(period_item, (int, sp.Integer))
                    else int(sp.N(period_item))
                )
            ]

    if not period:
        raise ValueError(f"Could not extract period for D={D}")

    L = len(period)

    # Compute convergents manually using recurrence
    # p[-2] = 0, p[-1] = 1
    # q[-2] = 1, q[-1] = 0
    # p[i] = a[i] * p[i-1] + p[i-2]
    # q[i] = a[i] * q[i-1] + q[i-2]
    p_minus2, p_minus1 = 0, 1
    q_minus2, q_minus1 = 1, 0

    # First convergent: a0/1
    p_prev = a0 * p_minus1 + p_minus2
    q_prev = a0 * q_minus1 + q_minus2

    # Check if a0/1 is a solution
    if p_prev * p_prev - D * q_prev * q_prev == 1:
        return int(p_prev), int(q_prev)

    # Iterate through the period (need to go through one full period)
    max_iter = 2 * L
    for i in range(max_iter):
        a_i = period[i % L] if period else a0
        p_curr = a_i * p_prev + p_minus1
        q_curr = a_i * q_prev + q_minus1

        # Check if this convergent is a solution
        if p_curr * p_curr - D * q_curr * q_curr == 1:
            return int(p_curr), int(q_curr)

        # Update for next iteration
        p_minus1, p_prev = p_prev, p_curr
        q_minus1, q_prev = q_prev, q_curr

    raise ValueError(f"Could not find fundamental solution for D={D}")


def solve_pell_equation_from_eq(eq: sp.Eq) -> str:
    """
    Solve a Pell's equation parametrically.

    Args:
        eq: SymPy equation object representing a Pell's equation

    Returns:
        String representation of the parametric solution
    """
    syms = list(eq.free_symbols)
    x_sym, y_sym = syms[0], syms[1]
    expanded_lhs = sp.expand(eq.lhs)
    coeff_y2 = expanded_lhs.coeff(y_sym**2)
    D = -coeff_y2
    x1, y1 = fundamental_solution(int(D))
    n = sp.symbols("n", integer=True)
    sol_x = ((x1 + y1 * sp.sqrt(D)) ** n + (x1 - y1 * sp.sqrt(D)) ** n) / 2
    sol_y = ((x1 + y1 * sp.sqrt(D)) ** n - (x1 - y1 * sp.sqrt(D)) ** n) / (
        2 * sp.sqrt(D)
    )
    return f"{x_sym} = {sol_x}\n{y_sym} = {sol_y}"


ZERO_TOL = 1e-12


def _solve_linear_equation(equation: sp.Eq, variable: sp.Symbol) -> List[sp.Basic]:
    """Solve a linear equation of the form a*x + b = 0.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of solutions
    """
    try:
        solutions = sp.solve(equation, variable)
        if isinstance(solutions, dict):
            return [solutions.get(variable)]
        elif isinstance(solutions, (list, tuple)):
            return list(solutions)
        else:
            return [solutions]
    except (NotImplementedError, ValueError, TypeError):
        return []


def _solve_quadratic_equation(equation: sp.Eq, variable: sp.Symbol) -> List[sp.Basic]:
    """Solve a quadratic equation using polynomial factorization.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of solutions
    """
    try:
        # Try polynomial approach for quadratics
        poly = sp.Poly(equation.lhs - equation.rhs, variable)
        if poly is not None and poly.degree() == 2:
            roots = poly.nroots()
            return [r for r in roots if abs(sp.im(r)) < NUMERIC_TOLERANCE]
    except (ValueError, TypeError):
        pass

    # Fallback to general solve
    try:
        solutions = sp.solve(equation, variable)
        if isinstance(solutions, dict):
            return [solutions.get(variable)]
        elif isinstance(solutions, (list, tuple)):
            return list(solutions)
        else:
            return [solutions]
    except (NotImplementedError, ValueError, TypeError):
        return []


def _solve_polynomial_equation(equation: sp.Eq, variable: sp.Symbol) -> List[sp.Basic]:
    """Solve a polynomial equation using Poly root finding.

    Args:
        equation: SymPy equation
        variable: Symbol to solve for

    Returns:
        List of numeric roots
    """
    try:
        poly = sp.Poly(equation.lhs - equation.rhs, variable)
        if poly is not None and poly.degree() > 0:
            # For low-degree polynomials, try exact roots first
            if poly.degree() <= 4:
                try:
                    exact_roots = poly.all_roots()
                    return [r for r in exact_roots if abs(sp.im(r)) < NUMERIC_TOLERANCE]
                except (NotImplementedError, ValueError):
                    pass
            # Fallback to numeric roots
            numeric_roots = poly.nroots()
            return [r for r in numeric_roots if abs(sp.im(r)) < NUMERIC_TOLERANCE]
    except (ValueError, TypeError):
        pass

    return []


def _numeric_roots_for_single_var(
    expr: sp.Basic,
    variable: sp.Symbol,
    interval: Tuple[float, float] = (-4 * sp.pi, 4 * sp.pi),
    max_guesses: Optional[int] = None,
) -> List[sp.Number]:
    """Find numeric roots of expression using multiple strategies.

    Args:
        expr: SymPy expression to find roots of (set to zero)
        variable: Symbol to solve for
        interval: Search interval (min, max)
        max_guesses: Maximum number of guess points (default: MAX_NSOLVE_GUESSES)

    Returns:
        List of numeric roots found
    """
    if max_guesses is None:
        max_guesses = MAX_NSOLVE_GUESSES
    roots: List[float] = []
    interval_min, interval_max = float(interval[0]), float(interval[1])

    # Strategy 1: Try solveset over the reals within interval
    try:
        from sympy import solveset

        solset = solveset(
            sp.Eq(expr, 0), variable, domain=sp.Interval(interval_min, interval_max)
        )
        finite_values = []
        for solution in solset:
            try:
                solution_value = float(sp.N(solution))
                finite_values.append(solution_value)
            except (ValueError, TypeError):
                continue
        if finite_values:
            unique_values = sorted(set(round(x_val, 12) for x_val in finite_values))
            return [sp.N(root_val) for root_val in unique_values]
    except (ValueError, TypeError, NotImplementedError):
        pass

    # Strategy 2: Try polynomial root finding
    try:
        poly = sp.Poly(expr, variable)
        if poly is not None and poly.total_degree() > 0:
            for root in poly.nroots():
                if abs(sp.im(root)) < NUMERIC_TOLERANCE:
                    roots.append(float(sp.re(root)))
            if roots:
                unique_roots = sorted(set(round(x_val, 12) for x_val in roots))
                return [sp.N(root_val) for root_val in unique_roots]
    except (ValueError, TypeError):
        pass
    except sp.polys.polyerrors.PolynomialError:
        # Expression is not a polynomial (e.g., contains trig functions, exponentials, etc.)
        # Skip this strategy and continue to Strategy 3
        pass

    # Strategy 3: Detect sign changes and use nsolve
    interval_min, interval_max = interval_min, interval_max
    coarse_grid_size = max(COARSE_GRID_MIN_SIZE, max_guesses // 3)
    sample_points = [
        interval_min + (interval_max - interval_min) * idx / coarse_grid_size
        for idx in range(coarse_grid_size + 1)
    ]
    candidate_points = []
    previous_value = None
    for sample_point in sample_points:
        try:
            current_value = float(sp.N(expr.subs({variable: sample_point})))
            if (
                previous_value is not None
                and current_value == current_value
                and previous_value == previous_value
                and previous_value * current_value <= 0
            ):
                candidate_points.append(sample_point)
            previous_value = current_value
        except (ValueError, TypeError):
            previous_value = None

    # De-duplicate and limit candidate points
    candidate_points = sorted(
        set(round(candidate, 8) for candidate in candidate_points)
    )[:COARSE_GRID_MIN_SIZE]
    for guess in candidate_points:
        try:
            root = sp.nsolve(
                expr,
                variable,
                guess,
                tol=ROOT_SEARCH_TOLERANCE,
                maxsteps=MAX_NSOLVE_STEPS,
            )
            if abs(sp.im(root)) > NUMERIC_TOLERANCE:
                continue
            root_real = float(sp.re(root))
            if not any(
                abs(existing - root_real) < ROOT_DEDUP_TOLERANCE for existing in roots
            ):
                roots.append(root_real)
        except (ValueError, TypeError, NotImplementedError):
            continue

    sorted_roots = sorted(roots)
    return [sp.N(root_val) for root_val in sorted_roots]


def solve_single_equation(
    eq_str: str, find_var: Optional[str] = None
) -> Dict[str, Any]:
    """
    Solve a single equation.

    Args:
        eq_str: Equation string (e.g., "x+1=0", "x^2-1=0")
        find_var: Optional variable to solve for (e.g., "x")

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: Result type ("equation", "pell", "identity_or_contradiction", "multi_isolate")
            - exact: List of exact solutions (strings)
            - approx: List of approximate solutions (strings or None)
            - error: Error message if ok is False
    """
    parts = eq_str.split("=", 1)
    if len(parts) != 2:
        return {
            "ok": False,
            "error": "Invalid equation format: Expected exactly one '='. Use format like 'x+1=0' or 'x^2=4'.",
            "error_code": "INVALID_FORMAT",
        }
    lhs_s, rhs_s = parts[0].strip(), parts[1].strip()

    # Handle empty RHS: treat as evaluation of LHS
    if not rhs_s:
        # If RHS is empty, evaluate the LHS expression
        lhs_eval = evaluate_safely(lhs_s)
        if lhs_eval.get("ok"):
            return {
                "ok": True,
                "type": "evaluation",
                "exact": [lhs_eval.get("result") or ""],
                "approx": [lhs_eval.get("approx")],
            }
        else:
            return {
                "ok": False,
                "error": f"Failed to evaluate '{lhs_s}': {lhs_eval.get('error')}",
                "error_code": lhs_eval.get("error_code", "EVAL_ERROR"),
            }

    rhs_s = rhs_s or "0"
    lhs = evaluate_safely(lhs_s)
    if not lhs.get("ok"):
        error_msg = f"Failed to parse left-hand side '{lhs_s}': {lhs.get('error')}"
        error_code = lhs.get("error_code", "PARSE_ERROR")

        # Provide helpful hints for common syntax errors
        if error_code == "UNBALANCED_PARENS":
            error_msg += ". Hint: Make sure your equation uses the format 'expression1 = expression2' (with spaces around =). For example, use 'x^x = log(25^x)/x' instead of '((x^x)=log(25^x))/(x)'."
        elif error_code == "UNMATCHED_QUOTES":
            error_msg += ". Check that all quotes are properly matched."
        elif error_code == "SYNTAX_ERROR":
            # Additional hints may already be in the error message
            pass
        elif "cannot assign" in str(lhs.get("error", "")).lower():
            error_msg += ". Hint: The left-hand side appears to contain an assignment '='. If you're trying to solve an equation, use '==' for comparison, not '='. For example: 'x == 5' not 'x = 5'."
        else:
            error_msg += ". Please check your input syntax."

        return {
            "ok": False,
            "error": error_msg,
            "error_code": error_code,
        }
    rhs = evaluate_safely(rhs_s)
    if not rhs.get("ok"):
        error_msg = f"Failed to parse right-hand side '{rhs_s}': {rhs.get('error')}"
        error_code = rhs.get("error_code", "PARSE_ERROR")

        # Provide helpful hints for common syntax errors
        if error_code == "UNBALANCED_PARENS":
            error_msg += ". Hint: Make sure your equation uses the format 'expression1 = expression2' (with spaces around =). For example, use 'x^x = log(25^x)/x' instead of '((x^x)=log(25^x))/(x)'."
        elif error_code == "UNMATCHED_QUOTES":
            error_msg += ". Check that all quotes are properly matched."
        elif error_code == "SYNTAX_ERROR":
            # Additional hints may already be in the error message
            pass
        elif "cannot assign" in str(rhs.get("error", "")).lower():
            error_msg += ". Hint: The right-hand side contains an equation '= 0'. If you're assigning a variable to an equation result, you cannot nest equations inside assignments. Try: 'a = expression' then solve 'expression = 0' separately."
        else:
            error_msg += ". Please check your input syntax."

        return {
            "ok": False,
            "error": error_msg,
            "error_code": error_code,
        }
    try:
        left_expr = parse_preprocessed(lhs["result"])
        right_expr = parse_preprocessed(rhs["result"])
    except (ParseError, ValidationError) as e:
        logger.warning("Parse error assembling SymPy expressions", exc_info=True)
        return {"ok": False, "error": f"Parse error: {e}", "error_code": "PARSE_ERROR"}
    except (ValueError, TypeError) as e:
        logger.warning("Type error assembling SymPy expressions", exc_info=True)
        return {"ok": False, "error": f"Type error: {e}", "error_code": "TYPE_ERROR"}
    equation = sp.Eq(left_expr, right_expr)
    if is_pell_equation_from_eq(equation):
        try:
            pell_str = solve_pell_equation_from_eq(equation)
            # Don't prettify Pell solutions to avoid Unicode issues on Windows
            return {"ok": True, "type": "pell", "solution": pell_str}
        except ValueError as e:
            logger.warning("Pell solver error: invalid equation", exc_info=True)
            return {
                "ok": False,
                "error": f"Pell solver error: {e}",
                "error_code": "PELL_SOLVER_ERROR",
            }
        except Exception as e:
            logger.error("Unexpected error in Pell solver", exc_info=True)
            return {
                "ok": False,
                "error": f"Pell solver error: {e}",
                "error_code": "PELL_SOLVER_ERROR",
            }
    symbols = list(equation.free_symbols)
    if not symbols:
        try:
            simp = sp.simplify(left_expr - right_expr)
            if simp == 0:
                return {
                    "ok": True,
                    "type": "identity_or_contradiction",
                    "result": "Identity",
                }
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            # These are expected for some expressions that can't be simplified
            simp = None
        except Exception as e:
            # Unexpected error - log it
            logger.debug(f"Unexpected error in simplify check: {e}", exc_info=True)
            simp = None
        try:
            diff = sp.N(left_expr - right_expr, 60)
            re_diff = sp.re(diff)
            im_diff = sp.im(diff)
            tol = sp.N(10) ** (-45)
            if abs(re_diff) < tol and abs(im_diff) < tol:
                return {
                    "ok": True,
                    "type": "identity_or_contradiction",
                    "result": "Identity (numeric)",
                }
            else:
                return {
                    "ok": True,
                    "type": "identity_or_contradiction",
                    "result": "Contradiction (numeric)",
                }
        except (TypeError, ValueError, AttributeError):
            # Expected for some expressions that can't be evaluated numerically
            return {
                "ok": True,
                "type": "identity_or_contradiction",
                "result": "Contradiction (unable to confirm identity symbolically or numerically)",
            }
        except Exception as e:
            # Unexpected error - log it but still return reasonable result
            logger.debug(
                f"Unexpected error in numeric identity check: {e}", exc_info=True
            )
            return {
                "ok": True,
                "type": "identity_or_contradiction",
                "result": "Contradiction (unable to confirm identity symbolically or numerically)",
            }

    # Use module-level _numeric_roots_for_single_var function (defined above)
    try:
        if find_var:
            sym = sp.symbols(find_var)
            if sym not in symbols:
                return {
                    "ok": False,
                    "error": f"Variable '{find_var}' not present.",
                    "error_code": "VARIABLE_NOT_FOUND",
                }
            sols = sp.solve(equation, sym)
            # Filter to only real solutions
            real_sols = []
            real_approx = []
            for solution in sols if isinstance(sols, (list, tuple)) else [sols]:
                try:
                    num_val = sp.N(solution)
                    # Check if solution is real (imaginary part is negligible)
                    if abs(sp.im(num_val)) < NUMERIC_TOLERANCE:
                        real_sols.append(solution)
                        real_approx.append(str(sp.re(num_val)))
                except (ValueError, TypeError, OverflowError, ArithmeticError):
                    # Can't evaluate numerically - check if it's obviously complex
                    sol_str = str(solution)
                    if "I" not in sol_str:
                        # Might be real but can't evaluate - include it
                        real_sols.append(solution)
                        real_approx.append(None)
                    # Otherwise skip complex solutions

            if not real_sols:
                # Check if equation is impossible (e.g., sin(x) = pi)
                error_hint = None
                if equation.has(sp.sin):
                    try:
                        if equation.lhs.has(sp.sin) and not equation.rhs.has(sp.sin):
                            rhs_val = float(sp.N(equation.rhs))
                            if abs(rhs_val) > 1:
                                error_hint = f"sin({find_var}) cannot equal {rhs_val} (|sin({find_var})| <= 1)"
                        elif equation.rhs.has(sp.sin) and not equation.lhs.has(sp.sin):
                            lhs_val = float(sp.N(equation.lhs))
                            if abs(lhs_val) > 1:
                                error_hint = f"sin({find_var}) cannot equal {lhs_val} (|sin({find_var})| <= 1)"
                    except (ValueError, TypeError, AttributeError):
                        pass
                if equation.has(sp.cos) and not error_hint:
                    try:
                        if equation.lhs.has(sp.cos) and not equation.rhs.has(sp.cos):
                            rhs_val = float(sp.N(equation.rhs))
                            if abs(rhs_val) > 1:
                                error_hint = f"cos({find_var}) cannot equal {rhs_val} (|cos({find_var})| <= 1)"
                        elif equation.rhs.has(sp.cos) and not equation.lhs.has(sp.cos):
                            lhs_val = float(sp.N(equation.lhs))
                            if abs(lhs_val) > 1:
                                error_hint = f"cos({find_var}) cannot equal {lhs_val} (|cos({find_var})| <= 1)"
                    except (ValueError, TypeError, AttributeError):
                        pass

                if error_hint:
                    return {
                        "ok": False,
                        "error": f"This equation has no real solutions: {error_hint}.",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }
                else:
                    return {
                        "ok": False,
                        "error": "This equation has no real solutions (only complex solutions exist).",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }

            exacts = [str(s) for s in real_sols]
            return {
                "ok": True,
                "type": "equation",
                "exact": exacts,
                "approx": real_approx,
            }
        if len(symbols) == 1:
            sym = symbols[0]
            # Check if equation contains trigonometric functions - use numeric fallback directly
            if NUMERIC_FALLBACK_ENABLED and equation.has(sp.sin, sp.cos, sp.tan):
                equation_expr = left_expr - right_expr
                numeric_roots = _numeric_roots_for_single_var(
                    equation_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi)
                )
                if numeric_roots:
                    exacts = [str(r) for r in numeric_roots]
                    approx = [str(sp.N(r)) for r in numeric_roots]
                    return {
                        "ok": True,
                        "type": "equation",
                        "exact": exacts,
                        "approx": approx,
                    }
                # No numeric roots found - equation may be unsolvable or have no real solutions
                # For equations like sin(x)=pi/2 (which has no real solutions since |sin(x)| <= 1)
                # Try sp.solve() for exact symbolic solution, but catch generator errors gracefully
                try:
                    sols = sp.solve(equation, sym)
                    if sols:
                        # Filter to only real solutions
                        real_sols = []
                        real_approx = []
                        for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                            try:
                                num_val = sp.N(s)
                                # Check if solution is real (imaginary part is negligible)
                                if abs(sp.im(num_val)) < NUMERIC_TOLERANCE:
                                    real_sols.append(s)
                                    real_approx.append(str(sp.re(num_val)))
                                else:
                                    # Solution is complex - check if it's from an impossible trig equation
                                    s_str = str(s)
                                    # Check for asin/acos with argument > 1
                                    if "asin" in s_str.lower():
                                        import re

                                        match = re.search(
                                            r"asin\(([^)]+)\)", s_str, re.IGNORECASE
                                        )
                                        if match:
                                            try:
                                                inner_val = float(sp.N(match.group(1)))
                                                if abs(inner_val) > 1:
                                                    # This is from an impossible equation (e.g., sin(x) = pi where pi > 1)
                                                    pass  # Will be handled below
                                            except (
                                                ValueError,
                                                TypeError,
                                                AttributeError,
                                            ):
                                                pass
                            except (
                                ValueError,
                                TypeError,
                                OverflowError,
                                ArithmeticError,
                            ):
                                # Can't evaluate - might be symbolic, skip for now
                                pass

                        # If we found real solutions, return them
                        if real_sols:
                            exacts = [str(s) for s in real_sols]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": real_approx,
                            }

                        # No real solutions found - check if equation is impossible
                        # Check if sin(x) = k or cos(x) = k where |k| > 1
                        error_hint = None
                        if equation.has(sp.sin):
                            try:
                                if equation.lhs.has(sp.sin) and not equation.rhs.has(
                                    sp.sin
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) > 1:
                                        error_hint = f"sin(x) cannot equal {rhs_val} (|sin(x)| <= 1)"
                                elif equation.rhs.has(sp.sin) and not equation.lhs.has(
                                    sp.sin
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) > 1:
                                        error_hint = f"sin(x) cannot equal {lhs_val} (|sin(x)| <= 1)"
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if equation.has(sp.cos) and not error_hint:
                            try:
                                if equation.lhs.has(sp.cos) and not equation.rhs.has(
                                    sp.cos
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) > 1:
                                        error_hint = f"cos(x) cannot equal {rhs_val} (|cos(x)| <= 1)"
                                elif equation.rhs.has(sp.cos) and not equation.lhs.has(
                                    sp.cos
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) > 1:
                                        error_hint = f"cos(x) cannot equal {lhs_val} (|cos(x)| <= 1)"
                            except (ValueError, TypeError, AttributeError):
                                pass

                        if error_hint:
                            return {
                                "ok": False,
                                "error": f"This trigonometric equation has no real solutions: {error_hint}.",
                                "error_code": "NO_REAL_SOLUTIONS",
                            }
                        else:
                            return {
                                "ok": False,
                                "error": "This trigonometric equation has no real solutions (only complex solutions exist).",
                                "error_code": "NO_REAL_SOLUTIONS",
                            }
                    # If sp.solve() returns empty list, check for impossible inverse trig equations
                    # Check for asin/acos/atan with impossible target values
                    error_hint = None
                    if equation.has(sp.asin):
                        try:
                            # asin(x) = k: range of asin is [-pi/2, pi/2]
                            if equation.lhs.has(sp.asin) and not equation.rhs.has(
                                sp.asin
                            ):
                                rhs_val = float(sp.N(equation.rhs))
                                if abs(rhs_val) > sp.pi / 2:
                                    error_hint = f"asin(x) cannot equal {rhs_val} (range of asin is [-pi/2, pi/2])"
                            elif equation.rhs.has(sp.asin) and not equation.lhs.has(
                                sp.asin
                            ):
                                lhs_val = float(sp.N(equation.lhs))
                                if abs(lhs_val) > sp.pi / 2:
                                    error_hint = f"asin(x) cannot equal {lhs_val} (range of asin is [-pi/2, pi/2])"
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if equation.has(sp.acos) and not error_hint:
                        try:
                            # acos(x) = k: range of acos is [0, pi]
                            if equation.lhs.has(sp.acos) and not equation.rhs.has(
                                sp.acos
                            ):
                                rhs_val = float(sp.N(equation.rhs))
                                if rhs_val < 0 or rhs_val > sp.pi:
                                    error_hint = f"acos(x) cannot equal {rhs_val} (range of acos is [0, pi])"
                            elif equation.rhs.has(sp.acos) and not equation.lhs.has(
                                sp.acos
                            ):
                                lhs_val = float(sp.N(equation.lhs))
                                if lhs_val < 0 or lhs_val > sp.pi:
                                    error_hint = f"acos(x) cannot equal {lhs_val} (range of acos is [0, pi])"
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if equation.has(sp.atan) and not error_hint:
                        try:
                            # atan(x) = k: range of atan is (-pi/2, pi/2), but we'll be lenient
                            # atan can actually approach but never equal pi/2 or -pi/2
                            if equation.lhs.has(sp.atan) and not equation.rhs.has(
                                sp.atan
                            ):
                                rhs_val = float(sp.N(equation.rhs))
                                if abs(rhs_val) >= sp.pi / 2:
                                    error_hint = f"atan(x) cannot equal {rhs_val} (range of atan is (-pi/2, pi/2))"
                            elif equation.rhs.has(sp.atan) and not equation.lhs.has(
                                sp.atan
                            ):
                                lhs_val = float(sp.N(equation.lhs))
                                if abs(lhs_val) >= sp.pi / 2:
                                    error_hint = f"atan(x) cannot equal {lhs_val} (range of atan is (-pi/2, pi/2))"
                        except (ValueError, TypeError, AttributeError):
                            pass

                    if error_hint:
                        return {
                            "ok": False,
                            "error": f"This equation has no real solutions: {error_hint}.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }

                    return {
                        "ok": False,
                        "error": "No real solutions found for this trigonometric equation.",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }
                except Exception as solve_err:
                    error_msg = str(solve_err).lower()
                    # Check for the specific generator error that occurs with trigonometric functions
                    # This can be ValueError, TypeError, or other SymPy-specific exceptions
                    if "generators" in error_msg or "contains an element" in error_msg:
                        return {
                            "ok": False,
                            "error": "This trigonometric equation cannot be solved symbolically. No real solutions found in the search interval.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }
                    # For other errors, return appropriate error message
                    return {
                        "ok": False,
                        "error": f"Solving error: {solve_err}",
                        "error_code": "SOLVER_ERROR",
                    }
            # Try specialized handlers first for polynomial equations
            try:
                # Detect equation type and route to appropriate handler
                poly = sp.Poly(equation.lhs - equation.rhs, sym)
                if poly is not None and poly.degree() > 0:
                    if poly.degree() == 1:
                        # Linear equation: use specialized handler
                        handler_sols = _solve_linear_equation(equation, sym)
                        if handler_sols:
                            sols = handler_sols
                        else:
                            # Fallback to general solve
                            sols = sp.solve(equation, sym)
                    elif poly.degree() == 2:
                        # Quadratic equation: use specialized handler
                        handler_sols = _solve_quadratic_equation(equation, sym)
                        if handler_sols:
                            sols = handler_sols
                        else:
                            # Fallback to general solve
                            sols = sp.solve(equation, sym)
                    elif poly.degree() > 2:
                        # Higher-degree polynomial: use specialized handler
                        handler_sols = _solve_polynomial_equation(equation, sym)
                        if handler_sols:
                            sols = handler_sols
                        else:
                            # Fallback to general solve
                            sols = sp.solve(equation, sym)
                    else:
                        # Not a polynomial or degree 0, use general solve
                        sols = sp.solve(equation, sym)
                else:
                    # Not a polynomial, use general solve
                    sols = sp.solve(equation, sym)
            except (ValueError, TypeError, AttributeError):
                # Poly construction failed (non-polynomial equation), try general solve
                poly = None
            except sp.polys.polyerrors.PolynomialError:
                # Explicitly caught PolynomialError - expression is not a polynomial
                # This happens when trying to construct Poly from expressions with
                # exponentials, logarithms, or other non-polynomial terms
                # Fall through to general solve
                poly = None
            # If Poly construction failed, try general solve
            if "poly" not in locals() or poly is None:
                # Check for trig functions first before attempting sp.solve()
                if NUMERIC_FALLBACK_ENABLED and equation.has(sp.sin, sp.cos, sp.tan):
                    equation_expr = left_expr - right_expr
                    numeric_roots = _numeric_roots_for_single_var(
                        equation_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi)
                    )
                    if numeric_roots:
                        exacts = [str(r) for r in numeric_roots]
                        approx = [str(sp.N(r)) for r in numeric_roots]
                        return {
                            "ok": True,
                            "type": "equation",
                            "exact": exacts,
                            "approx": approx,
                        }
                # Try general solve for non-polynomial, non-trig equations
                try:
                    sols = sp.solve(equation, sym)
                    # Check if solution list is empty and equation is impossible
                    if not sols:
                        # Check for impossible inverse trig equations
                        error_hint = None
                        if equation.has(sp.asin):
                            try:
                                if equation.lhs.has(sp.asin) and not equation.rhs.has(
                                    sp.asin
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) > sp.pi / 2:
                                        error_hint = f"asin(x) cannot equal {rhs_val} (range of asin is [-pi/2, pi/2])"
                                elif equation.rhs.has(sp.asin) and not equation.lhs.has(
                                    sp.asin
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) > sp.pi / 2:
                                        error_hint = f"asin(x) cannot equal {lhs_val} (range of asin is [-pi/2, pi/2])"
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if equation.has(sp.acos) and not error_hint:
                            try:
                                if equation.lhs.has(sp.acos) and not equation.rhs.has(
                                    sp.acos
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if rhs_val < 0 or rhs_val > sp.pi:
                                        error_hint = f"acos(x) cannot equal {rhs_val} (range of acos is [0, pi])"
                                elif equation.rhs.has(sp.acos) and not equation.lhs.has(
                                    sp.acos
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if lhs_val < 0 or lhs_val > sp.pi:
                                        error_hint = f"acos(x) cannot equal {lhs_val} (range of acos is [0, pi])"
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if equation.has(sp.atan) and not error_hint:
                            try:
                                if equation.lhs.has(sp.atan) and not equation.rhs.has(
                                    sp.atan
                                ):
                                    rhs_val = float(sp.N(equation.rhs))
                                    if abs(rhs_val) >= sp.pi / 2:
                                        error_hint = f"atan(x) cannot equal {rhs_val} (range of atan is (-pi/2, pi/2))"
                                elif equation.rhs.has(sp.atan) and not equation.lhs.has(
                                    sp.atan
                                ):
                                    lhs_val = float(sp.N(equation.lhs))
                                    if abs(lhs_val) >= sp.pi / 2:
                                        error_hint = f"atan(x) cannot equal {lhs_val} (range of atan is (-pi/2, pi/2))"
                            except (ValueError, TypeError, AttributeError):
                                pass

                        if error_hint:
                            return {
                                "ok": False,
                                "error": f"This equation has no real solutions: {error_hint}.",
                                "error_code": "NO_REAL_SOLUTIONS",
                            }
                        # If no error hint found but sols is empty, return appropriate error
                        return {
                            "ok": False,
                            "error": "No real solutions found for this equation.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }
                except NotImplementedError as e:
                    logger.info(
                        f"Symbolic solve not implemented, trying numeric fallback: {e}"
                    )
                    if NUMERIC_FALLBACK_ENABLED:
                        equation_expr = left_expr - right_expr
                        # Use wider interval for complex equations (e.g., with exponentials, mixed terms)
                        # Try positive domain first (many exponential equations require x > 0)
                        numeric_roots = _numeric_roots_for_single_var(
                            equation_expr, sym, interval=(0.01, 50)
                        )
                        if not numeric_roots:
                            # If no roots in positive domain, try full range
                            numeric_roots = _numeric_roots_for_single_var(
                                equation_expr, sym, interval=(-20, 20)
                            )
                        if numeric_roots:
                            exacts = [str(r) for r in numeric_roots]
                            approx = [str(sp.N(r)) for r in numeric_roots]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": approx,
                            }
                    return {
                        "ok": False,
                        "error": "This equation cannot be solved symbolically. Numeric root finding found no real solutions in the search interval. The equation may have no real solutions, or solutions may be outside the search range.",
                        "error_code": "NO_REAL_SOLUTIONS",
                    }
                except (ValueError, TypeError) as solve_error:
                    # sp.solve() failed - try numeric fallback
                    if NUMERIC_FALLBACK_ENABLED:
                        equation_expr = left_expr - right_expr
                        numeric_roots = _numeric_roots_for_single_var(
                            equation_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi)
                        )
                        if numeric_roots:
                            exacts = [str(r) for r in numeric_roots]
                            approx = [str(sp.N(r)) for r in numeric_roots]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": approx,
                            }
                    logger.warning(
                        f"Error in symbolic solve, trying numeric fallback: {solve_error}",
                        exc_info=True,
                    )
                    return {
                        "ok": False,
                        "error": f"Solving error: {solve_error}",
                        "error_code": "SOLVER_ERROR",
                    }
                except Exception as solve_error:
                    # Catch other exceptions from sp.solve() (e.g., generator errors)
                    error_msg = str(solve_error).lower()
                    # Check for the specific generator error that can occur with various function types
                    if "generators" in error_msg or "contains an element" in error_msg:
                        # Try numeric fallback for equations that can't be solved symbolically
                        if NUMERIC_FALLBACK_ENABLED:
                            equation_expr = left_expr - right_expr
                            numeric_roots = _numeric_roots_for_single_var(
                                equation_expr,
                                sym,
                                interval=(
                                    -20,
                                    20,
                                ),  # Wider interval for exponential equations
                            )
                            if numeric_roots:
                                exacts = [str(r) for r in numeric_roots]
                                approx = [str(sp.N(r)) for r in numeric_roots]
                                return {
                                    "ok": True,
                                    "type": "equation",
                                    "exact": exacts,
                                    "approx": approx,
                                }
                        return {
                            "ok": False,
                            "error": "This equation cannot be solved symbolically. No real solutions found in the search interval.",
                            "error_code": "NO_REAL_SOLUTIONS",
                        }
                    # For other errors, try numeric fallback if enabled
                    if NUMERIC_FALLBACK_ENABLED:
                        equation_expr = left_expr - right_expr
                        numeric_roots = _numeric_roots_for_single_var(
                            equation_expr, sym, interval=(-20, 20)
                        )
                        if numeric_roots:
                            exacts = [str(r) for r in numeric_roots]
                            approx = [str(sp.N(r)) for r in numeric_roots]
                            return {
                                "ok": True,
                                "type": "equation",
                                "exact": exacts,
                                "approx": approx,
                            }
                    logger.error("Unexpected error in symbolic solve", exc_info=True)
                    return {
                        "ok": False,
                        "error": f"Unexpected solving error: {solve_error}",
                        "error_code": "SOLVER_ERROR",
                    }
            # Check if sols is empty before processing
            if not sols:
                return {
                    "ok": False,
                    "error": "No real solutions found for this equation.",
                    "error_code": "NO_REAL_SOLUTIONS",
                }
            exacts = (
                [str(solution) for solution in sols]
                if isinstance(sols, (list, tuple))
                else [str(sols)]
            )
            approx = []
            for solution in sols if isinstance(sols, (list, tuple)) else [sols]:
                try:
                    approx.append(str(sp.N(solution)))
                except (ValueError, TypeError, OverflowError, ArithmeticError):
                    # Expected for some symbolic solutions
                    approx.append(None)
            return {"ok": True, "type": "equation", "exact": exacts, "approx": approx}
        multi_solutions: Dict[str, List[str]] = {}
        multi_approx: Dict[str, List[Optional[str]]] = {}
        for sym in symbols:
            try:
                sols_for_sym = sp.solve(equation, sym)
                if isinstance(sols_for_sym, dict):
                    sols_list = [str(v) for v in sols_for_sym.values()]
                    sols_exprs = list(sols_for_sym.values())
                elif isinstance(sols_for_sym, (list, tuple)):
                    sols_list = [str(s) for s in sols_for_sym]
                    sols_exprs = list(sols_for_sym)
                else:
                    sols_list = [str(sols_for_sym)]
                    sols_exprs = [sols_for_sym]
                multi_solutions[str(sym)] = sols_list
                approx_list: List[Optional[str]] = []
                for expr in sols_exprs:
                    try:
                        approx_list.append(str(sp.N(expr)))
                    except (ValueError, TypeError, OverflowError, ArithmeticError):
                        # Expected for some symbolic solutions
                        approx_list.append(None)
                multi_approx[str(sym)] = approx_list
            except NotImplementedError as e:
                logger.info(f"Solving for {sym} not implemented: {e}")
                multi_solutions[str(sym)] = [
                    "Solving not implemented for this variable"
                ]
                multi_approx[str(sym)] = [None]
            except (ValueError, TypeError) as e:
                logger.warning(f"Error solving for {sym}", exc_info=True)
                multi_solutions[str(sym)] = [f"Error: {e}"]
                multi_approx[str(sym)] = [None]
            except Exception as e:
                logger.error(f"Unexpected error solving for {sym}", exc_info=True)
                multi_solutions[str(sym)] = [f"Unexpected error: {e}"]
                multi_approx[str(sym)] = [None]
        return {
            "ok": True,
            "type": "multi_isolate",
            "solutions": multi_solutions,
            "approx": multi_approx,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"Solving error: {e}",
            "error_code": "SOLVER_ERROR",
        }


def _parse_relational_fallback(rel_str: str) -> sp.Basic:
    """Parse a relational expression fallback method.

    Args:
        rel_str: String containing relational operator

    Returns:
        SymPy expression (if single expression) or tuple of parsed parts
        Raises ValueError if parsing fails
    """
    parts = re.split(r"(<=|>=|<|>)", rel_str)
    if len(parts) == 1:
        res = evaluate_safely(rel_str)
        if not res.get("ok"):
            raise ValueError(res.get("error"))
        return parse_preprocessed(res["result"])
    expr_parts = parts[::2]
    ops = parts[1::2]
    parsed_parts = []
    for p in expr_parts:
        p_strip = p.strip()
        if not p_strip:
            continue
        res = evaluate_safely(p_strip)
        if not res.get("ok"):
            raise ValueError(
                f"Failed to parse component '{p_strip}': {res.get('error')}"
            )
        parsed_parts.append(parse_preprocessed(res["result"]))
    if len(parsed_parts) != len(ops) + 1:
        raise ValueError("Invalid inequality structure.")
    op_map = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}
    relations = []
    for i, op_str in enumerate(ops):
        op_func = op_map.get(op_str)
        if not op_func:
            raise ValueError(f"Unknown operator: {op_str}")
        relations.append(op_func(parsed_parts[i], parsed_parts[i + 1]))
    return sp.And(*relations) if len(relations) > 1 else relations[0]


def solve_inequality(ineq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """
    Solve an inequality.

    Args:
        ineq_str: Inequality string (e.g., "x > 0", "1 < x < 5")
        find_var: Optional variable to solve for

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: "inequality"
            - solutions: Dictionary mapping variable names to solution strings
            - error: Error message if ok is False
    """
    try:
        parsed = _parse_relational_fallback(ineq_str)
    except (ParseError, ValidationError) as e:
        logger.warning("Failed to parse inequality", exc_info=True)
        return {"ok": False, "error": f"Parse error: {e}", "error_code": "PARSE_ERROR"}
    except (ValueError, TypeError) as e:
        logger.warning("Type error parsing inequality", exc_info=True)
        return {
            "ok": False,
            "error": f"Invalid inequality: {e}",
            "error_code": "INVALID_INEQUALITY",
        }
    except Exception as e:
        logger.error("Unexpected error parsing inequality", exc_info=True)
        return {
            "ok": False,
            "error": f"Failed to parse inequality: {e}",
            "error_code": "PARSE_ERROR",
        }
    free_syms = list(parsed.free_symbols) if hasattr(parsed, "free_symbols") else []
    if find_var:
        target_sym = None
        for sym in free_syms:
            if str(sym) == find_var:
                target_sym = sym
                break
        if target_sym is None:
            return {
                "ok": False,
                "error": f"Variable '{find_var}' not found in the expression.",
            }
        vars_to_solve = [target_sym]
    else:
        if not free_syms:
            try:
                is_true = sp.simplify(parsed)
                return {
                    "ok": True,
                    "type": "inequality",
                    "solutions": {"result": str(is_true)},
                }
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(
                    f"Error finding variables in inequality: {e}", exc_info=True
                )
                return {
                    "ok": False,
                    "error": "No variable found in inequality",
                    "error_code": "NO_VARIABLE",
                }
        vars_to_solve = free_syms
    results = {}
    for v in vars_to_solve:
        try:
            ineqs_to_solve = (
                list(parsed.args) if isinstance(parsed, sp.And) else [parsed]
            )
            sol = sp.reduce_inequalities(ineqs_to_solve, v)
            results[str(v)] = str(sol)
        except NotImplementedError:
            logger.info(f"Inequality solving not implemented for {v}")
            results[str(v)] = "Solver not implemented for this type of inequality."
        except (ValueError, TypeError) as e:
            logger.warning(f"Error solving inequality for {v}", exc_info=True)
            results[str(v)] = f"Error solving for {v}: {e}"
        except Exception as e:
            logger.error(f"Unexpected error solving inequality for {v}", exc_info=True)
            results[str(v)] = f"Unexpected error solving for {v}: {e}"
    return {"ok": True, "type": "inequality", "solutions": results}


def solve_system(raw_no_find: str, find_token: Optional[str]) -> Dict[str, Any]:
    """
    Solve a system of equations.

    Args:
        raw_no_find: Comma-separated equation strings (e.g., "x+y=3, x-y=1")
        find_token: Optional variable to find in solutions

    Returns:
        Dictionary with keys:
            - ok: Boolean indicating success
            - type: Result type ("system", "system_var")
            - solutions: List of solution dictionaries (for system)
            - exact: List of exact solutions (for system_var)
            - approx: List of approximate solutions (for system_var)
            - error: Error message if ok is False
    """
    parts = [p.strip() for p in raw_no_find.split(",") if p.strip()]
    eqs_serialized = []
    assignments = {}
    for p in parts:
        if "=" not in p:
            continue
        lhs, rhs = p.split("=", 1)
        lhs_s = lhs.strip()
        rhs_s = rhs.strip()
        lhs_eval = evaluate_safely(lhs_s)
        if not lhs_eval.get("ok"):
            return {
                "ok": False,
                "error": f"LHS parse error: {lhs_eval.get('error')}",
                "error_code": lhs_eval.get("error_code", "PARSE_ERROR"),
            }
        rhs_eval = evaluate_safely(rhs_s)
        if not rhs_eval.get("ok"):
            return {
                "ok": False,
                "error": f"RHS parse error: {rhs_eval.get('error')}",
                "error_code": rhs_eval.get("error_code", "PARSE_ERROR"),
            }
        if VAR_NAME_RE.match(lhs_s):
            assignments[lhs_s] = {
                "result": rhs_eval.get("result"),
                "approx": rhs_eval.get("approx"),
            }
        eqs_serialized.append(
            {"lhs": lhs_eval.get("result"), "rhs": rhs_eval.get("result")}
        )
    if not eqs_serialized:
        return {
            "ok": False,
            "error": "No equations parsed.",
            "error_code": "NO_EQUATIONS",
        }
    if find_token and len(eqs_serialized) == 1:
        pair = eqs_serialized[0]
        lhs_s = pair.get("lhs")
        rhs_s = pair.get("rhs")
        try:
            lhs_expr = parse_preprocessed(lhs_s)
            rhs_expr = parse_preprocessed(rhs_s)
            equation = sp.Eq(lhs_expr, rhs_expr)
            sym = sp.symbols(find_token)
            if sym in equation.free_symbols:
                try:
                    sols = sp.solve(equation, sym)
                except (NotImplementedError, ValueError, TypeError) as e:
                    logger.warning(f"Error solving inequality: {e}", exc_info=True)
                    sols = []
                if sols:
                    exacts = (
                        [str(s) for s in sols]
                        if isinstance(sols, (list, tuple))
                        else [str(sols)]
                    )
                    approx = []
                    for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                        try:
                            approx.append(str(sp.N(s)))
                        except (ValueError, TypeError, OverflowError, ArithmeticError):
                            # Expected for some symbolic solutions
                            approx.append(None)
                    return {
                        "ok": True,
                        "type": "system_var",
                        "exact": exacts,
                        "approx": approx,
                    }
        except (ValueError, TypeError, AttributeError):
            # Expected for some expressions that can't be processed
            pass
    if find_token:
        defining = None
        for pair in eqs_serialized:
            if pair.get("rhs") == find_token:
                defining = ("lhs", pair.get("lhs"))
                break
            if pair.get("lhs") == find_token:
                defining = ("rhs", pair.get("rhs"))
                break
        if defining:
            side, expr_str = defining
            try:
                expr_sym = parse_preprocessed(expr_str)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Error extracting expression symbol: {e}", exc_info=True)
                expr_sym = None
            if expr_sym is not None:
                subs_map = {}
                for var, info in assignments.items():
                    if info.get("approx") is not None:
                        try:
                            subs_map[sp.symbols(var)] = sp.sympify(info.get("approx"))
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                f"Error in inequality solving: {e}", exc_info=True
                            )
                            try:
                                subs_map[sp.symbols(var)] = parse_preprocessed(
                                    info.get("result")
                                )
                            except (ParseError, ValidationError, ValueError, TypeError):
                                # Expected parsing errors
                                pass
                    else:
                        try:
                            subs_map[sp.symbols(var)] = parse_preprocessed(
                                info.get("result")
                            )
                        except (ParseError, ValidationError, ValueError, TypeError):
                            # Expected parsing errors
                            pass
                if subs_map:
                    try:
                        value = expr_sym.subs(subs_map)
                        try:
                            approx_obj = sp.N(value)
                            if (
                                abs(sp.re(approx_obj)) < ZERO_TOL
                                and abs(sp.im(approx_obj)) < ZERO_TOL
                            ):
                                return {
                                    "ok": True,
                                    "type": "system_var",
                                    "exact": ["0"],
                                    "approx": ["0"],
                                }
                            approx_val = str(approx_obj)
                        except (ValueError, TypeError, OverflowError, ArithmeticError):
                            # Expected for some symbolic expressions
                            approx_val = None
                        return {
                            "ok": True,
                            "type": "system_var",
                            "exact": [str(value)],
                            "approx": [approx_val],
                        }
                    except (ValueError, TypeError, AttributeError):
                        # Expected for some expressions
                        pass
    payload = {"equations": eqs_serialized, "find": find_token}
    try:
        stdout_text = _worker_solve_cached(json.dumps(payload))
    except (TimeoutError, ValueError, TypeError) as e:
        logger.warning(f"Error in worker-based solving: {e}", exc_info=True)
        return {
            "ok": False,
            "error": "Solving timed out (worker).",
            "error_code": "TIMEOUT",
        }
    try:
        data = json.loads(stdout_text)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Invalid JSON from worker-solve: {e}", exc_info=True)
        return {
            "ok": False,
            "error": f"Invalid worker-solve output: {e}.",
            "error_code": "INVALID_OUTPUT",
        }
    if not data.get("ok"):
        return data
    sols_list = data.get("solutions", [])
    if not find_token:
        return {"ok": True, "type": "system", "solutions": sols_list}
    found_vals = []
    for sol_dict in sols_list:
        if find_token in sol_dict:
            found_vals.append(sol_dict[find_token])
    if not found_vals:
        return {
            "ok": False,
            "error": f"No solution found for variable {find_token}.",
            "error_code": "NO_SOLUTION",
        }
    approx_vals = []
    for vstr in found_vals:
        try:
            approx_vals.append(str(sp.N(sp.sympify(vstr))))
        except (ValueError, TypeError, OverflowError, ArithmeticError):
            # Expected for some symbolic solutions
            approx_vals.append(None)
    return {
        "ok": True,
        "type": "system_var",
        "exact": found_vals,
        "approx": approx_vals,
    }
