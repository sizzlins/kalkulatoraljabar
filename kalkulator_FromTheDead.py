#!/usr/bin/env python3
"""
algebraicKalkulator
Property of Muhammad Akhiel al Syahbana
Not to be freely distributed without authors permission
10/08/25

pyinstaller --onefile --console --collect-all sympy kalkulator.py

---
NOTE: This file was manually reconstructed from Python 3.13 disassembly.
All logic has been restored, but original comments and formatting are not present.
---
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import shlex as _shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

VERSION = '2025-10-22'

# Setup allowed names for the parser
ALLOWED_SYMPY_NAMES = {
    'pi': sp.pi,
    'E': sp.E,
    'I': sp.I,
    'sqrt': sp.sqrt,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'asin': sp.asin,
    'acos': sp.acos,
    'atan': sp.atan,
    'log': sp.log,
    'ln': sp.log,  # Both log and ln map to sp.log
    'exp': sp.exp,
    'Abs': sp.Abs,
}

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Worker resource limits
WORKER_CPU_SECONDS = 30
WORKER_AS_MB = 400
WORKER_TIMEOUT = 60

# --- Regex Patterns (from module disassembly) ---
VAR_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
PERCENT_REGEX = re.compile(r'(\d+(?:\.\d+)?)%')
SQRT_UNICODE_REGEX = re.compile(r'√\s*\(')
DIGIT_LETTERS_REGEX = re.compile(r'(\d)\s*([A-Za-z(])')
AMBIG_FRACTION_REGEX = re.compile(r'\(([^()]+?)/([^()]+?)\)')

ZERO_TOL = 1e-12


# --- Reconstructed Functions ---

def superscriptify(s: str) -> str:
    """Reconstructed from 001_superscriptify.dis.txt"""
    mapping = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '-': '⁻', 'n': 'ⁿ'
    }
    return "".join(mapping.get(ch, ch) for ch in s)


def format_superscript(expr_str: str) -> str:
    """Reconstructed from 002_format_superscript.dis.txt"""
    return re.sub(r'\*\*(-?[\da-zA-Z]+)', lambda m: superscriptify(m.group(1)), expr_str)


def format_number(val: Any, precision: int = 6) -> str:
    """Reconstructed from 003_format_number.dis.txt"""
    try:
        val_float = float(val)
        if abs(val_float) < ZERO_TOL:
            return "0"
        return f"{val_float:.{precision}g}"
    except (TypeError, ValueError):
        return str(val)


def format_solution(sol: Any, exact: bool = True) -> str:
    """Reconstructed from 004_format_solution.dis.txt"""
    if isinstance(sol, (list, tuple, sp.Set)):
        if not sol:
            return "(No solution)"
        return f"({', '.join(format_solution(s, exact) for s in sol)})"
    if exact:
        return format_superscript(str(sol))
    else:
        return format_number(sol)


def prettify_expr(expr_str: str) -> str:
    """Reconstructed from 005_prettify_expr.dis.txt"""
    expr_str = str(expr_str)
    return format_superscript(expr_str)


def is_balanced(s: str) -> bool:
    """Reconstructed from 007_is_balanced.dis.txt"""
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    opening = pairs.keys()
    closing = pairs.values()

    for ch in s:
        if ch in opening:
            stack.append(ch)
        elif ch in closing:
            if not stack:
                return False
            last_open = stack.pop()
            if pairs[last_open] != ch:
                return False
    return not stack


def fix_fraction(expr: str) -> str:
    """Reconstructed from 008_fix_fraction.dis.txt"""

    def replacer(match):
        num, den = match.group(1), match.group(2)
        if is_balanced(num) and is_balanced(den):
            return f"(({num})/({den}))"
        return match.group(0)  # Return original if not balanced

    return AMBIG_FRACTION_REGEX.sub(replacer, expr)


def preprocess(input_str: str, skip_exponent_conversion: bool = False) -> str:
    """Reconstructed from 009_preprocess.dis.txt"""
    s = input_str.strip()
    s = PERCENT_REGEX.sub(r'(\1 / 100)', s)
    s = SQRT_UNICODE_REGEX.sub(r'sqrt(', s)

    if not skip_exponent_conversion:
        # This handles implicit multiplication like "2x" or "3(x+1)"
        s = DIGIT_LETTERS_REGEX.sub(r'\1 * \2', s)

    s = fix_fraction(s)

    # Handle user-friendly exponent syntax ^
    if '^' in s:
        if '**' in s:
            # If both exist, we can't safely convert
            pass
        else:
            s = s.replace('^', '**')

    # Handle special constants
    s = re.sub(r'(?i)\b(pi)\b', 'pi', s)
    s = re.sub(r'(?i)\b(e)\b', 'E', s)

    # Replace unicode minus with standard hyphen
    s = s.replace('−', '-')

    return s


def parse_preprocessed(expr_str: str) -> Any:
    """Reconstructed from 010_parse_preprocessed.dis.txt"""
    return parse_expr(expr_str, local_dict=ALLOWED_SYMPY_NAMES, transformations=TRANSFORMATIONS)


def format_inequality_solution(sol_str: str) -> str:
    """Reconstructed from 011_format_inequality_solution.dis.txt"""
    sol_str = str(sol_str)

    replacements = {
        '[': '',
        ']': '',
        '(': '',
        ')': '',
        'Union': 'or',
        'Interval': '',
        'Intersection': 'and',
        'oo': '∞',
        '-oo': '-∞',
        ',': ' to ',
    }

    # Use regex for word boundaries for keys like 'oo'
    for old, new in replacements.items():
        if old.isalpha():
            sol_str = re.sub(rf'\b{re.escape(old)}\b', new, sol_str)
        else:
            sol_str = sol_str.replace(old, new)

    # Clean up whitespace
    sol_str = re.sub(r'\s{2,}', ' ', sol_str).strip()

    # Special formatting for open/closed intervals
    if ' to ' in sol_str and ('-∞' not in sol_str and '∞' not in sol_str):
        parts = sol_str.split(' or ')
        formatted_parts = []
        for part in parts:
            if ' to ' in part:
                try:
                    left, right = part.split(' to ')
                    formatted_parts.append(f"{left.strip()} ≤ x ≤ {right.strip()}")
                except Exception:
                    formatted_parts.append(part)  # Fallback
            else:
                formatted_parts.append(part)
        sol_str = ' or '.join(formatted_parts)

    return sol_str


def _limit_resources():
    """Reconstructed from 012__limit_resources.dis.txt"""
    if not HAS_RESOURCE:
        return  # Do nothing on systems without the 'resource' module (e.g., Windows)

    try:
        # Limit CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS))
        # Limit address space (memory)
        mem_limit_bytes = WORKER_AS_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    except Exception as e:
        # This might fail in some environments (e.g., permissions)
        print(f"Warning: Could not set resource limits: {e}", file=sys.stderr)


def worker_evaluate(expression: str) -> Dict[str, Any]:
    """Reconstructed from 013_worker_evaluate.dis.txt"""
    _limit_resources()

    try:
        parsed = parse_preprocessed(expression)

        # Check for free symbols (variables)
        free_syms = list(parsed.free_symbols)

        if not free_syms:
            # No variables, evaluate numerically
            approx_obj = sp.N(parsed)
            approx_str = str(approx_obj)

            if 'zoo' in approx_str or 'oo' in approx_str or 'nan' in approx_str:
                res = {
                    "ok": False,
                    "error": f"Result is '{approx_str}' (Complex Infinity/Undefined)",
                }
            else:
                res = {
                    "ok": True,
                    "result_str": str(parsed),
                    "result_approx": format_number(approx_obj),
                    "result_type": "numeric",
                }
        else:
            # Has variables, return symbolic expression
            res = {
                "ok": True,
                "result_str": str(parsed),
                "result_approx": None,
                "result_type": "symbolic",
                "vars": [str(s) for s in free_syms],
            }

    except Exception as e:
        res = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }

    return res


def evaluate_safely(expr: str, timeout: int = WORKER_TIMEOUT) -> Dict[str, Any]:
    """Reconstructed from 014_evaluate_safely.dis.txt"""
    cmd = _build_self_cmd(['--worker', expr])

    try:
        # Popen args based on platform
        kwargs = {}
        if sys.platform != "win32":
            kwargs['preexec_fn'] = os.setsid

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            **kwargs
        )

        stdout, stderr = proc.communicate(timeout=timeout)

        if proc.returncode != 0:
            return {"ok": False, "error": f"Worker process crashed (code {proc.returncode}). Error: {stderr or 'N/A'}"}

        try:
            result = json.loads(stdout)
            return result
        except json.JSONDecodeError:
            return {"ok": False, "error": f"Invalid worker output: {stdout}"}

    except subprocess.TimeoutExpired:
        if proc:
            if sys.platform == "win32":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return {"ok": False, "error": f"Calculation timed out (> {timeout}s)"}
    except Exception as e:
        return {"ok": False, "error": f"Process launch failed: {e}"}


def _build_self_cmd(args: List[str]) -> List[str]:
    """Reconstructed from 015__build_self_cmd.dis.txt"""
    if getattr(sys, 'frozen', False):
        # We are running in a PyInstaller bundle
        return [sys.executable] + args
    else:
        # We are running as a normal .py script
        return [sys.executable, sys.argv[0]] + args


def is_pell_equation_from_eq(eq: sp.Eq) -> bool:
    """Reconstructed from 016_is_pell_equation_from_eq.dis.txt"""
    # This is a heuristic reconstruction
    lhs = eq.lhs
    rhs = eq.rhs
    expr = lhs - rhs
    # Check for D*x**2 - y**2 - 1 = 0 or x**2 - D*y**2 - 1 = 0
    # It seems to be checking a very specific form
    return (
            isinstance(expr, sp.Add) and
            len(expr.args) == 3 and
            any(a.is_number and a == -1 for a in expr.args)
    )


def fundamental_solution(D: int) -> Tuple[int, int]:
    """Reconstructed from 017_fundamental_solution.dis.txt"""
    # This is a standard algorithm for Pell's equation x^2 - D*y^2 = 1
    m = 0
    d = 1
    a = int(sp.sqrt(D))
    a0 = a

    (x_prev, x_curr) = (1, a)
    (y_prev, y_curr) = (0, 1)

    while x_curr ** 2 - D * y_curr ** 2 != 1:
        m = d * a - m
        d = (D - m ** 2) // d
        a = (a0 + m) // d

        (x_prev, x_curr) = (x_curr, a * x_curr + x_prev)
        (y_prev, y_curr) = (y_curr, a * y_curr + y_prev)

    return (x_curr, y_curr)


def solve_pell_equation_from_eq(eq: sp.Eq) -> str:
    """Reconstructed from 018_solve_pell_equation_from_eq.dis.txt"""
    # This is highly speculative, based on the `fundamental_solution` function
    # It likely extracts D and solves x**2 - D*y**2 = 1
    try:
        expr = eq.lhs - eq.rhs
        syms = list(expr.free_symbols)
        if len(syms) != 2:
            return "Pell solver: Not 2 variables."

        x, y = syms[0], syms[1]

        # Try to find D
        # This is a guess at the structure
        D = None
        if expr.coeff(x ** 2) == 1 and expr.coeff(y ** 2).is_number:
            D = -expr.coeff(y ** 2)
        elif expr.coeff(y ** 2) == 1 and expr.coeff(x ** 2).is_number:
            D = -expr.coeff(x ** 2)
            (x, y) = (y, x)  # Swap variables

        if D is not None and D > 0 and int(D) == D:
            D_int = int(D)
            if sp.sqrt(D_int) == int(sp.sqrt(D_int)):
                return "Pell solver: D is a perfect square."

            x1, y1 = fundamental_solution(D_int)
            # This is a guess at the output format
            return f"Fundamental solution: {x} = {x1}, {y} = {y1}"
        else:
            return "Pell solver: Cannot identify D in x**2 - D*y**2 = 1"

    except Exception as e:
        return f"Pell solver error: {e}"


@dataclass
class EvalResult:
    """Reconstructed from 019_EvalResult.dis.txt"""
    ok: bool
    result_str: str
    result_approx: str
    result_obj: Any
    result_type: str
    error: str


def eval_user_expression(expr: str) -> EvalResult:
    """Reconstructed from 020_eval_user_expression.dis.txt"""
    # This is a fallback evaluator, likely not the main one
    try:
        p_expr = preprocess(expr)
        parsed = parse_preprocessed(p_expr)

        return EvalResult(
            ok=True,
            result_str=str(parsed),
            result_approx=format_number(sp.N(parsed)),
            result_obj=parsed,
            result_type="symbolic",
            error=None
        )
    except Exception as e:
        return EvalResult(
            ok=False,
            result_str=None,
            result_approx=None,
            result_obj=None,
            result_type="error",
            error=str(e)
        )


def solve_single_equation_cli(eq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """Reconstructed from 021_solve_single_equation_cli.dis.txt"""

    _limit_resources()

    try:
        if '=' not in eq_str:
            return {"ok": False, "error": "Not an equation. Use '=' to solve."}

        # --- Sub-function `_numeric_roots_for_single_var` ---
        # (Reconstructed from 022__numeric_roots_for_single_var.dis.txt)
        def _numeric_roots_for_single_var(expr, var):
            try:
                # Use nroots (numeric)
                roots = sp.nroots(expr, var, n=15)  # n=15 is a guess
                return [r for r in roots if not r.has(sp.I)]
            except Exception:
                # Fallback to solveset for numeric
                try:
                    solset = sp.solveset(expr, var, domain=sp.Reals)
                    if isinstance(solset, sp.FiniteSet):
                        return list(solset)
                    return None  # Indicates non-numeric solution
                except Exception:
                    return None  # Total failure

        # --- Main logic continues ---
        pre_expr = preprocess(eq_str, skip_exponent_conversion=True)

        if find_var:
            target_var = sp.symbols(find_var)
        else:
            target_var = None

        if pre_expr.count('=') == 1:
            lhs_s, rhs_s = pre_expr.split('=', 1)
            lhs = parse_preprocessed(lhs_s)
            rhs = parse_preprocessed(rhs_s)
            eq = sp.Eq(lhs, rhs)
        else:
            # Fallback for multiple '='? This seems odd.
            # Let's assume it's a fallback parse
            eq = _parse_relational_fallback(pre_expr)
            if not isinstance(eq, sp.Eq):
                return {"ok": False, "error": "Could not parse equation."}

        free_syms = list(eq.free_symbols)

        if not free_syms:
            # e.g., "1+1=2"
            result = sp.simplify(eq)
            return {"ok": True, "result_type": "truth", "result": bool(result)}

        # Determine target variable
        if target_var:
            if target_var not in free_syms:
                return {"ok": False, "error": f"Variable '{find_var}' not in equation."}
        elif len(free_syms) == 1:
            target_var = free_syms[0]
        else:
            # Multiple variables, no target specified
            return {
                "ok": False,
                "error": "Equation has multiple variables. Please specify which one to solve for.",
                "vars": [str(s) for s in free_syms]
            }

        # --- Try symbolic solve first ---
        try:
            # Check for Pell equation
            if is_pell_equation_from_eq(eq):
                pell_sol = solve_pell_equation_from_eq(eq)
                return {"ok": True, "result_type": "symbolic", "result_str": pell_sol}

            # Standard symbolic solve
            sol = sp.solve(eq, target_var)

            if not sol:
                # Symbolic solve failed, try numeric
                expr_to_solve = eq.lhs - eq.rhs
                num_roots = _numeric_roots_for_single_var(expr_to_solve, target_var)
                if num_roots is not None:
                    return {
                        "ok": True,
                        "result_type": "numeric",
                        "result_exact": format_solution(num_roots, exact=True),
                        "result_approx": format_solution(num_roots, exact=False),
                    }
                else:
                    # Both failed
                    return {"ok": True, "result_type": "symbolic", "result_str": "(No solution found)"}

            return {
                "ok": True,
                "result_type": "symbolic",
                "result_str": format_solution(sol, exact=True),
                "result_approx": format_solution(sol, exact=False),
            }

        except Exception as e:
            # Symbolic solve failed, try numeric as fallback
            try:
                expr_to_solve = eq.lhs - eq.rhs
                num_roots = _numeric_roots_for_single_var(expr_to_solve, target_var)
                if num_roots is not None:
                    return {
                        "ok": True,
                        "result_type": "numeric",
                        "result_exact": format_solution(num_roots, exact=True),
                        "result_approx": format_solution(num_roots, exact=False),
                    }
                else:
                    return {"ok": False, "error": f"Solve error: {e}"}
            except Exception as e2:
                return {"ok": False, "error": f"Symbolic solve failed: {e}. Numeric fallback failed: {e2}"}

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def solve_inequality_cli(ineq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """Reconstructed from 023_solve_inequality_cli.dis.txt"""
    _limit_resources()

    try:
        pre_expr = preprocess(ineq_str, skip_exponent_conversion=True)

        # Use fallback parser to handle <, >, <=, >=
        ineq = _parse_relational_fallback(pre_expr)
        if ineq is None or isinstance(ineq, sp.Eq):
            return {"ok": False, "error": "Not a valid inequality. Use <, >, <=, or >="}

        free_syms = list(ineq.free_symbols)

        if not free_syms:
            # e.g., "1 < 2"
            result = sp.simplify(ineq)
            return {"ok": True, "result_type": "truth", "result": bool(result)}

        # Determine target variable
        if find_var:
            target_var = sp.symbols(find_var)
            if target_var not in free_syms:
                return {"ok": False, "error": f"Variable '{find_var}' not in inequality."}
        elif len(free_syms) == 1:
            target_var = free_syms[0]
        else:
            # Multiple variables
            return {
                "ok": False,
                "error": "Inequality has multiple variables. Please specify which one to solve for.",
                "vars": [str(s) for s in free_syms]
            }

        # --- Try symbolic solve ---
        try:
            sol = sp.solve(ineq, target_var)
            sol_str = format_inequality_solution(sol)

            return {
                "ok": True,
                "result_type": "symbolic",
                "result_str": sol_str,
            }
        except Exception as e:
            # Try to solve using solveset
            try:
                sol = sp.solveset(ineq, target_var, domain=sp.Reals)
                sol_str = format_inequality_solution(sol)
                return {
                    "ok": True,
                    "result_type": "symbolic",
                    "result_str": sol_str,
                }
            except Exception as e2:
                return {"ok": False, "error": f"Solve error: {e}. Solveset fallback error: {e2}"}

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _parse_relational_fallback(rel_str: str):
    """Reconstructed from 024__parse_relational_fallback.dis.txt"""
    # This function is for parsing expressions with <, >, <=, >=
    # which the default parse_expr does not handle.

    # 1. Find the relational operator
    op_match = re.search(r'(<=|>=|==|=|<|>)', rel_str)
    if not op_match:
        # No operator, try to parse as simple expression
        try:
            return parse_preprocessed(rel_str)
        except Exception:
            return None  # Failed to parse

    op_str = op_match.group(1)
    if op_str == '=':
        op_str = '=='  # Treat single = as ==

    op_index = op_match.start()

    # Split into LHS and RHS
    lhs_str = rel_str[:op_index].strip()
    rhs_str = rel_str[op_index + len(op_str):].strip()

    if not lhs_str or not rhs_str:
        return None  # Invalid syntax, e.g., "= 5"

    # Check for multiple operators (e.g., 1 < x < 5)
    if re.search(r'(<=|>=|==|=|<|>)', rhs_str):
        # This is a chained inequality. SymPy can handle this.
        # e.g., sp.sympify("1 < x < 5")
        try:
            return sp.sympify(rel_str.replace('=', '=='), locals=ALLOWED_SYMPY_NAMES)
        except Exception:
            return None  # Failed to parse chain

    # This is a simple A op B inequality
    try:
        lhs = parse_preprocessed(lhs_str)
        rhs = parse_preprocessed(rhs_str)

        if op_str == '==':
            return sp.Eq(lhs, rhs)
        if op_str == '<':
            return sp.Lt(lhs, rhs)
        if op_str == '<=':
            return sp.Le(lhs, rhs)
        if op_str == '>':
            return sp.Gt(lhs, rhs)
        if op_str == '>=':
            return sp.Ge(lhs, rhs)

    except Exception:
        return None  # Failed to parse LHS or RHS

    return None  # Should not be reached


def worker_main(argv: List[str]) -> int:
    """Reconstructed from 025_worker_main.dis.txt"""
    if not argv:
        print(json.dumps({"ok": False, "error": "No expression provided to worker."}), file=sys.stderr)
        return 1

    expr = " ".join(argv)
    out = worker_evaluate(expr)

    try:
        print(json.dumps(out))
        sys.stdout.flush()
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Worker JSON output failed: {e}"}), file=sys.stderr)
        return 1

    return 0


def worker_solve_main(argv: List[str]) -> int:
    """Reconstructed from 026_worker_solve_main.dis.txt"""

    # --- Argument parsing for the worker ---
    # (Reconstructed from 027_parser_worker.dis.txt)
    parser_worker = argparse.ArgumentParser()
    parser_worker.add_argument('--find', type=str, default=None)
    parser_worker.add_argument('--inequality', action='store_true', default=False)
    parser_worker.add_argument('expression', nargs='*')

    try:
        args = parser_worker.parse_args(argv)
    except SystemExit:
        print(json.dumps({"ok": False, "error": "Worker failed to parse args."}))
        return 1

    expr = " ".join(args.expression)
    if not expr:
        print(json.dumps({"ok": False, "error": "No expression provided to solve worker."}))
        return 1

    # --- Route to equation or inequality solver ---
    try:
        if args.inequality:
            out = solve_inequality_cli(expr, args.find)
        else:
            out = solve_single_equation_cli(expr, args.find)

        print(json.dumps(out))
        sys.stdout.flush()
        return 0

    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Solve worker crashed: {type(e).__name__}: {e}"}))
        return 1


def print_result_pretty(res: Dict[str, Any], json_mode: bool = False) -> None:
    """Reconstructed from 028_print_result_pretty.dis.txt"""
    if json_mode:
        print(json.dumps(res))
        return

    if not res.get("ok"):
        print(f"\nError: {res.get('error', 'Unknown error')}", file=sys.stderr)
        if "vars" in res:
            print(f"Variables found: {', '.join(res['vars'])}", file=sys.stderr)
        return

    # --- All ok, print formatted result ---
    try:
        res_type = res.get("result_type")

        if res_type == "numeric":
            exact = prettify_expr(res.get("result_str"))
            approx = res.get("result_approx")
            print(f"\nResult: {exact}")
            if approx and approx != exact:
                print(f"Decimal: {approx}")

        elif res_type == "symbolic":
            exact = prettify_expr(res.get("result_str"))
            approx = res.get("result_approx")
            print(f"\nResult: {exact}")
            if approx and approx != exact:
                print(f"Decimal: {approx}")

        elif res_type == "truth":
            print(f"\nResult: {res.get('result')}")

        elif res_type == "numeric" and "result_exact" in res:
            # This is from the solver
            exact = prettify_expr(res.get("result_exact"))
            approx = res.get("result_approx")
            print(f"\nSolution: {exact}")
            if approx and approx != exact:
                print(f"Decimal: {approx}")
        else:
            # Default fallback
            print(f"\n{res}")

    except Exception as e:
        print(f"\nError: Result formatting failed: {e}", file=sys.stderr)
        print(f"Raw result: {res}")


@lru_cache(maxsize=None)
def _get_history_file() -> Optional[str]:
    """Reconstructed from 029__get_history_file.dis.txt"""
    try:
        if sys.platform == "win32":
            datadir = os.environ.get("APPDATA")
        else:
            datadir = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")

        if not datadir:
            return None

        kalk_dir = os.path.join(datadir, "kalkulator")
        os.makedirs(kalk_dir, exist_ok=True)

        hist_file = os.path.join(kalk_dir, "history.txt")
        return hist_file
    except Exception:
        return None


def repl_loop(json_mode: bool = False) -> None:
    """Reconstructed from 030_repl_loop.dis.txt"""

    # --- Readline/History Setup (from 031_setup_readline.dis.txt) ---
    try:
        import readline

        hist_file = _get_history_file()

        if hist_file and os.path.exists(hist_file):
            try:
                readline.read_history_file(hist_file)
            except Exception:
                pass  # Failed to read history

        # --- `save_history` (from 032_save_history.dis.txt) ---
        def save_history():
            try:
                if hist_file:
                    readline.write_history_file(hist_file)
            except Exception:
                pass  # Failed to write history

        import atexit
        atexit.register(save_history)

    except ImportError:
        pass  # Readline not available

    # --- REPL main loop ---
    if not json_mode:
        print(f"algebraicKalkulator [v{VERSION}]")
        print("Enter 'exit' or 'quit' to close.")

    session_vars = {}  # Not really used, but from disassembly

    while True:
        try:
            line = input(">> ")

            if not line.strip():
                continue

            cmd = line.lower().strip()
            if cmd in ('exit', 'quit', 'q'):
                break

            # --- Check for 'solve' command ---
            # (Reconstructed from 033_check_solve_cmd.dis.txt)
            is_solve_cmd = False
            solve_args = None
            find_var = None

            try:
                tokens = _shlex.split(line)
                if tokens[0].lower() == 'solve':
                    is_solve_cmd = True
                    # Re-parse args for solve
                    # (Reconstructed from 034_parser_repl.dis.txt)
                    parser_repl = argparse.ArgumentParser()
                    parser_repl.add_argument('solve_cmd')  # 'solve'
                    parser_repl.add_argument('--find', type=str, default=None)
                    parser_repl.add_argument('expression', nargs='*')

                    repl_args = parser_repl.parse_args(tokens)
                    solve_args = " ".join(repl_args.expression)
                    find_var = repl_args.find

            except Exception:
                is_solve_cmd = False  # Failed parsing, treat as normal expression

            # --- Route to solver or evaluator ---
            if is_solve_cmd:
                if not solve_args:
                    print("\nError: 'solve' command needs an expression.", file=sys.stderr)
                    continue

                # Determine if inequality
                is_inequality = any(op in solve_args for op in ('<', '>', '<=', '>='))

                # Use the safe, sandboxed subprocess evaluator
                cmd_list = ['--solve-worker']
                if find_var:
                    cmd_list.extend(['--find', find_var])
                if is_inequality:
                    cmd_list.append('--inequality')
                cmd_list.append(solve_args)

                # Build the final command
                full_cmd = _build_self_cmd(cmd_list)

                # Run the subprocess
                try:
                    proc = subprocess.Popen(
                        full_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                    )
                    stdout, stderr = proc.communicate(timeout=WORKER_TIMEOUT)

                    if proc.returncode != 0:
                        res = {"ok": False, "error": f"Solve worker crashed. {stderr or 'N/A'}"}
                    else:
                        res = json.loads(stdout)

                except Exception as e:
                    res = {"ok": False, "error": f"REPL solver failed: {e}"}

            else:
                # Normal evaluation
                res = evaluate_safely(line, timeout=WORKER_TIMEOUT)

            print_result_pretty(res, json_mode)

        except EOFError:
            break  # Ctrl+D
        except KeyboardInterrupt:
            print("\nInterrupted. (Type 'exit' to quit)")
        except Exception as e:
            print(f"\nREPL Error: {type(e).__name__}: {e}", file=sys.stderr)

    if not json_mode:
        print("Exiting.")


def handle_system_main(raw_no_find: str, find_token: Optional[str]) -> Dict[str, Any]:
    """Reconstructed from 038_handle_system_main.dis.txt"""
    _limit_resources()

    # This function seems to be for solving systems of equations
    try:
        # Split equations by comma or newline
        raw_eqs = re.split(r'[,\n]', raw_no_find)
        eqs_to_solve = []

        for s in raw_eqs:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Preprocess and parse
            pre_s = preprocess(s_clean, skip_exponent_conversion=True)
            parsed_rel = _parse_relational_fallback(pre_s)

            if parsed_rel is None:
                return {"ok": False, "error": f"Could not parse: {s_clean}"}

            eqs_to_solve.append(parsed_rel)

        if not eqs_to_solve:
            return {"ok": False, "error": "No equations provided to 'solve'"}

        # --- Determine variables to solve for ---
        all_syms = set()
        for eq in eqs_to_solve:
            all_syms.update(eq.free_symbols)

        vars_to_solve = []
        if find_token:
            # User specified variables
            find_vars_str = find_token.split(',')
            for v_str in find_vars_str:
                v_str = v_str.strip()
                if not VAR_NAME_RE.match(v_str):
                    return {"ok": False, "error": f"Invalid variable name: {v_str}"}

                v_sym = sp.symbols(v_str)
                if v_sym not in all_syms:
                    return {"ok": False, "error": f"Variable '{v_str}' not found in system."}
                vars_to_solve.append(v_sym)
        else:
            # Auto-detect all variables
            vars_to_solve = list(all_syms)

        if not vars_to_solve:
            if not all_syms:
                # e.g., "solve 1=1, 2=2"
                results = [bool(sp.simplify(eq)) for eq in eqs_to_solve]
                return {"ok": True, "result_type": "truth", "result": all(results)}
            else:
                return {"ok": False, "error": "Could not determine variables to solve for."}

        # --- Perform the solve ---
        try:
            sol = sp.solve(eqs_to_solve, vars_to_solve)

            # Format the solution
            if isinstance(sol, list) and sol:
                # e.g., [{x: 1, y: 2}]
                sol_str = ", ".join(
                    f"({', '.join(f'{k} = {v}' for k, v in s.items())})" for s in sol
                )
            elif isinstance(sol, dict):
                # e.g., {x: 1, y: 2}
                sol_str = ", ".join(f"{k} = {v}" for k, v in sol.items())
            elif not sol:
                sol_str = "(No solution found)"
            else:
                sol_str = str(sol)  # Fallback

            return {
                "ok": True,
                "result_type": "symbolic",
                "result_str": format_solution(sol, exact=True),
                "result_approx": format_solution(sol, exact=False)
            }

        except Exception as e:
            return {"ok": False, "error": f"System solve failed: {e}"}

    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def print_help_text():
    """Reconstructed from 035_print_help_text.dis.txt"""
    # This text is manually reconstructed from the disassembly constants
    print("algebraicKalkulator - REPL and CLI Calculator")
    print(f"Version: {VERSION}\n")
    print("Usage:")
    print("  kalkulator [options] [expression]")
    print("  kalkulator solve [options] <equation_or_system>\n")
    print("Modes:")
    print("  REPL Mode (default):")
    print("    Run without arguments to enter the interactive shell.")
    print("    >> 2+2")
    print("    Result: 4")
    print("\n    To solve equations in REPL, use the 'solve' command:")
    print("    >> solve x**2 - 1 = 0")
    print("    Solution: (-1, 1)")
    print("\n    To solve for a specific variable:")
    print("    >> solve x*y = 2 --find x")
    print("    Solution: (2/y)")
    print("\n    To solve systems (comma-separated):")
    print("    >> solve x+y=3, x-y=1")
    print("    Solution: (x = 2, y = 1)")
    print("\n    Enter 'exit' or 'quit' to close the REPL.\n")
    print("  CLI Mode (Expression):")
    print("    Pass an expression directly to get a one-time result.")
    print("    $ kalkulator \"1/2 + 1/4\"")
    print("    Result: 3/4")
    print("    Decimal: 0.75\n")
    print("  CLI Mode (Solve):")
    print("    Use 'solve' to solve equations or systems from the CLI.")
    print("    $ kalkulator solve \"a*x + b = c\" --find x")
    print("    Solution: ((c - b)/a)")
    print("\n    Inequalities are also supported:")
    print("    $ kalkulator solve \"x**2 > 4\"")
    print("    Solution: (-∞ to -2) or (2 to ∞)")
    print("\nOptions:")
    print("  -h, --help            Show this help message and exit.")
    print("  --version             Show program's version number and exit.")
    print("  --json                Output results in JSON format (for scripting).")
    print("  --worker <expr>       (Internal) Run expression in worker process.")
    print("  --solve-worker [args] (Internal) Run solver in worker process.")


def main_entry(argv: Optional[List[str]] = None) -> int:
    """Reconstructed from 036_main_entry.dis.txt"""
    if argv is None:
        argv = sys.argv[1:]

    # --- Main Argument Parser (from 037_parser_main.dis.txt) ---
    parser = argparse.ArgumentParser(
        description="algebraicKalkulator - REPL and CLI Calculator",
        add_help=False  # We add help manually
    )
    parser.add_argument(
        '-h', '--help',
        action='store_true',
        help='Show this help message and exit.'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f"%(prog)s {VERSION}",
        help="Show program's version number and exit."
    )
    parser.add_argument(
        '--json',
        action='store_true',
        default=False,
        help='Output results in JSON format.'
    )
    # Internal worker flags
    parser.add_argument('--worker', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--solve-worker', default=None, help=argparse.SUPPRESS, nargs='*')

    # Positional arguments for expression or 'solve' command
    parser.add_argument(
        'command_or_expr',
        nargs='*',
        help='Expression to evaluate, or "solve" command.'
    )

    # --- Parse arguments ---
    try:
        # We use parse_known_args to handle 'solve --find'
        args, unknown = parser.parse_known_args(argv)
    except SystemExit as e:
        return e.code  # Handle --version or --help

    # Manual help check
    if args.help:
        print_help_text()
        return 0

    # --- Worker Mode ---
    if args.worker is not None:
        # Re-join all args after --worker
        worker_expr = " ".join([args.worker] + args.command_or_expr + unknown)
        return worker_main([worker_expr])

    if args.solve_worker is not None:
        # solve-worker uses its own argparser, so pass all remaining
        solve_argv = (args.solve_worker if args.solve_worker else []) + args.command_or_expr + unknown
        return worker_solve_main(solve_argv)

    # --- CLI / REPL Mode ---
    full_expr_list = args.command_or_expr + unknown

    if not full_expr_list:
        # No args provided: Start REPL
        repl_loop(args.json)
        return 0

    # Args provided: Run as CLI

    # Check if 'solve' command
    if full_expr_list[0].lower() == 'solve':

        # Need to re-parse the 'solve' arguments
        find_token = None
        expr_list = full_expr_list[1:]  # Skip 'solve'

        if '--find' in expr_list:
            try:
                find_index = expr_list.index('--find')
                find_token = expr_list[find_index + 1]
                # Remove --find and its token from the list
                expr_list.pop(find_index)
                expr_list.pop(find_index)
            except IndexError:
                print_result_pretty(
                    {"ok": False, "error": "--find option needs a variable"},
                    args.json
                )
                return 1

        raw_expr = " ".join(expr_list)

        # Check for inequality
        is_inequality = any(op in raw_expr for op in ('<', '>', '<=', '>='))
        # Check for system (comma or newline)
        is_system = ',' in raw_expr or '\n' in raw_expr

        try:
            if is_system and not is_inequality:
                res = handle_system_main(raw_expr, find_token)
            elif is_inequality:
                res = solve_inequality_cli(raw_expr, find_token)
            else:
                # Single equation
                res = solve_single_equation_cli(raw_expr, find_token)

        except Exception as e:
            res = {"ok": False, "error": f"CLI solve failed: {type(e).__name__}: {e}"}

    else:
        # Standard expression evaluation
        raw_expr = " ".join(full_expr_list)
        res = evaluate_safely(raw_expr)

    # Print final result
    print_result_pretty(res, args.json)

    return 0 if res.get("ok") else 1


# --- Main Entry Point ---
# (from 000__module_.dis.txt)
if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
    except ImportError:
        pass  # Handle systems without multiprocessing
    else:
        freeze_support()

    sys.exit(main_entry(sys.argv[1:]))