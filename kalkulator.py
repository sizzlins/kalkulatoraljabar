#!/usr/bin/env python3
"""
algebraicKalkulator

Property of Muhammad Akhiel al Syahbana
Not to be freely distributed without authors permission
10/08/25

pyinstaller --onefile --console --collect-all sympy kalkulator.py

"""

from __future__ import annotations


import argparse
import json
import subprocess
import sys
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import sympy as sp


HAS_RESOURCE = False
try:
    import resource  # type: ignore
    HAS_RESOURCE = True
except Exception:
    HAS_RESOURCE = False


from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy import parse_expr


# Try to import multiprocessing primitives; guarded at runtime
try:
    from multiprocessing import Process, Queue, Event
except Exception:
    Process = None  # type: ignore
    Queue = None  # type: ignore
    Event = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VERSION = "2025-10-31"
WORKER_CPU_SECONDS = 30  # CPU seconds
WORKER_AS_MB = 400  # address space (virtual memory) in MB
WORKER_TIMEOUT = 60  # wall-clock seconds for subprocess to finish
ENABLE_PERSISTENT_WORKER = True  # amortize startup cost with a single background worker

# whitelist for parse_expr (still used inside worker)


ALLOWED_SYMPY_NAMES = {
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "Abs": sp.Abs,
    # Calculus & algebra
    "diff": sp.diff,
    "integrate": sp.integrate,
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
    # Matrices (basic)
    "Matrix": sp.Matrix,
    "det": sp.det,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)



# ---------------------------------------------------------------------------
# Helpers: formatting, preprocess, parse (these are safe string manipulations)
# ---------------------------------------------------------------------------

def superscriptify(s: str) -> str:
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
    return "".join(mapping.get(ch, ch) for ch in s)


def format_superscript(expr_str: str) -> str:
    return re.sub(r"\*\*(\-?\d+)", lambda m: superscriptify(m.group(1)), expr_str)


def format_number(val: Any, precision: int = 6) -> str:
    try:
        return "{:.6g}".format(float(val))
    except Exception:
        return str(val)


def format_solution(sol: Any, exact: bool = True) -> str:
    if isinstance(sol, (tuple, list)):
        return "(" + ", ".join(format_solution(v, exact) for v in sol) + ")"
    return format_superscript(str(sol)) if exact else format_number(sol)


def prettify_expr(expr_str: str) -> str:
    result = re.sub(r"sqrt\(([^)]+)\)", r"√(\1)", expr_str)
    result = result.replace("*", "×")
    return result


VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_balanced(s: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: List[str] = []
    for ch in s:
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values():
            if not stack:
                return False
            opening = stack.pop()
            if pairs[opening] != ch:
                return False
    return not stack


# Preprocessing
PERCENT_REGEX = re.compile(r"(\d+(?:\.\d+)?)%")
SQRT_UNICODE_REGEX = re.compile(r"√\s*\(?")
DIGIT_LETTERS_REGEX = re.compile(r"(\d)\s*([A-Za-z(])")
AMBIG_FRACTION_REGEX = re.compile(r"\(([^()]+?)/([^()]+?)\)")


def fix_fraction(expr: str) -> str:
    if "/" in expr and "(" not in expr and ")" not in expr:
        parts = expr.split("/")
        if len(parts) == 2:
            return f"({parts[0]})/({parts[1]})"
    return expr


def preprocess(input_str: str, skip_exponent_conversion: bool = False) -> str:
    # Basic denylist to avoid dangerous tokens before SymPy parsing
    forbidden_tokens = (
        "__", "import", "lambda", "eval", "exec", "open", "os.", "sys.",
        "subprocess", "builtins", "getattr", "setattr", "delattr", "compile", "globals",
        "locals", "__class__", "__mro__", "__subclasses__", "memoryview", "bytes", "bytearray",
        "__import__"
    )
    lowered = input_str.strip().lower()
    for tok in forbidden_tokens:
        if tok in lowered:
            raise ValueError("Input contains forbidden token.")

    s = input_str.strip()
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("Δ", "Delta")
    s = s.replace(":", "/")
    s = s.replace("×", "*")

    # Standardize common inequality variations before parsing
    s = s.replace("=>", ">=")
    s = s.replace("=<", "<=")

    if not skip_exponent_conversion:
        s = s.replace("^", "**")

        # Define mapping from superscript characters back to regular characters
        from_superscript_map = {
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
            "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
            "⁻": "-", "ⁿ": "n",
        }
        # Build a regex pattern to match any sequence of the superscript characters
        pattern = re.compile(f"([{''.join(from_superscript_map.keys())}]+)")

        def from_superscript_converter(m):
            # For each matched sequence, convert it back to a regular string
            normal_str = "".join([from_superscript_map[char] for char in m.group(1)])
            # Prepend with '**' to form a valid SymPy exponent
            return f"**{normal_str}"

        # Apply the conversion to the entire string
        s = pattern.sub(from_superscript_converter, s)

    s = PERCENT_REGEX.sub(r"(\1/100)", s)
    s = SQRT_UNICODE_REGEX.sub("sqrt(", s)
    s = AMBIG_FRACTION_REGEX.sub(r"((\1)/(\2))", s)
    s = DIGIT_LETTERS_REGEX.sub(r"\1*\2", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not is_balanced(s):
        raise ValueError("Mismatched or unbalanced parentheses/brackets in the expression.")
    return s


@lru_cache(maxsize=1024)
def parse_preprocessed(expr_str: str) -> Any:
    # Use restricted local_dict and provided transformations
    return parse_expr(expr_str, local_dict=ALLOWED_SYMPY_NAMES,
                      transformations=TRANSFORMATIONS, evaluate=True)


def format_inequality_solution(sol_str: str) -> str:
    """Tries to reformat SymPy's '&' inequality notation into a chained inequality."""
    # Pattern to capture two-part inequalities like (a < x) & (x < b)
    # It captures the two expressions and the variable in the middle
    pattern = re.compile(
        r"\((.*?)\s*([<>=!]+)\s*(.*?)\)\s*&\s*\((.*?)\s*([<>=!]+)\s*(.*?)\)"
    )
    match = pattern.match(sol_str)

    if not match:
        return sol_str  # Return original if it doesn't match the pattern

    g = [s.strip() for s in match.groups()]
    expr1, op1, var1, expr2, op2, var2 = g

    # Check if the variable is the same and in the middle of the two expressions
    if var1 == expr2:  # e.g., (a < x) & (x < b)
        # Check for valid "outward-facing" operators
        if op1 in ("<", "<=") and op2 in ("<", "<="):
            return f"{expr1} {op1} {var1} {op2} {var2}"
    elif expr1 == var2:  # e.g., (x > a) & (b > x)
        # Flip the operators to make a standard chain
        op_map = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
        if op1 in op_map and op2 in op_map:
            return f"{var1} {op_map[op1]} {expr1} {op_map[op2]} {var2}"

    return sol_str  # Fallback to original string if logic doesn't fit

# ---------------------------------------------------------------------------
# Worker sandbox (internal) - runs when __main__ called with --worker
# ---------------------------------------------------------------------------

def _limit_resources():
    """Apply rlimits in worker process (Unix)."""
    if not HAS_RESOURCE:
        return
    import resource as _resource  # local import
    # CPU seconds (soft, hard)
    _resource.setrlimit(_resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS + 1))
    # Memory - convert MB to bytes
    _resource.setrlimit(_resource.RLIMIT_AS, (WORKER_AS_MB * 1024 * 1024, WORKER_AS_MB * 1024 * 1024 + 1))


def worker_evaluate(preprocessed_expr: str) -> Dict[str, Any]:
    """
    Evaluate a preprocessed expression string and return a JSON-serializable dict.
    This function is called inside the worker subprocess.
    """
    # Apply resource limits (if desired)
    if HAS_RESOURCE:
        try:
            _limit_resources()
        except Exception:
            # If setting limits fails, continue but don't crash here.
            pass

    try:
        expr = parse_preprocessed(preprocessed_expr)
    except Exception as e:
        return {"ok": False, "error": f"Parse error in worker: {e}"}

    try:
        # Evaluate numeric/symbolic
        res = sp.simplify(expr)
        result_str = str(res)
        free_syms = [str(s) for s in getattr(res, "free_symbols", set())]
        # try a numeric approximation when possible
        approx = None
        try:
            approx_val = sp.N(res)
            approx_str = str(approx_val)
            if approx_str not in ("zoo", "oo", "-oo", "nan"):
                approx = approx_str
        except Exception:
            approx = None

        return {"ok": True, "result": result_str, "approx": approx, "free_symbols": free_syms}
    except Exception as e:
        return {"ok": False, "error": f"Evaluation failed: {e}"}


# ---------------------------------------------------------------------------
# Persistent worker (multiprocessing) and subprocess-based APIs
# ---------------------------------------------------------------------------


class _WorkerManager:
    def __init__(self) -> None:
        self.proc = None
        self.req_q = None
        self.res_q = None
        self.stop_event = None

    def start(self) -> None:
        if not ENABLE_PERSISTENT_WORKER or Process is None:
            return
        if self.is_alive():
            return
        self.req_q = Queue()
        self.res_q = Queue()
        self.stop_event = Event()
        self.proc = Process(target=_worker_daemon_main, args=(self.req_q, self.res_q, self.stop_event), daemon=True)
        self.proc.start()

    def is_alive(self) -> bool:
        return bool(self.proc and self.proc.is_alive())

    def stop(self) -> None:
        try:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.proc is not None:
                self.proc.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self.proc = None
            self.req_q = None
            self.res_q = None
            self.stop_event = None

    def request(self, payload: Dict[str, Any], timeout: int) -> Optional[Dict[str, Any]]:
        if not ENABLE_PERSISTENT_WORKER or Process is None:
            return None
        if not self.is_alive():
            self.start()
        if not self.is_alive():
            return None
        try:
            self.req_q.put(payload)
            result = self.res_q.get(timeout=timeout)
            return result
        except Exception:
            try:
                self.stop()
                self.start()
                if self.is_alive():
                    self.req_q.put(payload)
                    result = self.res_q.get(timeout=timeout)
                    return result
            except Exception:
                self.stop()
            return None


def _worker_daemon_main(req_q, res_q, stop_event) -> None:
    if HAS_RESOURCE:
        try:
            _limit_resources()
        except Exception:
            pass
    while True:
        if stop_event.is_set():
            break
        try:
            msg = req_q.get(timeout=0.1)
        except Exception:
            continue
        try:
            kind = msg.get("type")
            if kind == "eval":
                pre = msg.get("preprocessed") or ""
                out = worker_evaluate(pre)
                res_q.put(out)
            elif kind == "solve":
                payload = msg.get("payload") or {}
                out = _worker_solve_dispatch(payload)
                res_q.put(out)
            else:
                res_q.put({"ok": False, "error": "Unknown request type"})
        except Exception as e:
            res_q.put({"ok": False, "error": f"Worker daemon error: {e}"})


def _worker_solve_dispatch(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        eqs_input = payload.get("equations", [])
        eq_objs = []
        for item in eqs_input:
            lhs_s = item.get("lhs")
            rhs_s = item.get("rhs")
            if lhs_s is None or rhs_s is None:
                continue
            lhs_expr = parse_preprocessed(lhs_s)
            rhs_expr = parse_preprocessed(rhs_s)
            eq_objs.append(sp.Eq(lhs_expr, rhs_expr))
        if not eq_objs:
            return {"ok": False, "error": "No valid equations provided to worker-solve."}
        if HAS_RESOURCE:
            try:
                _limit_resources()
            except Exception:
                pass
        solutions = sp.solve(eq_objs, dict=True)
        if not solutions:
            return {"ok": False, "error": "No solution found (sp.solve returned empty)."}
        sols = []
        for sol in solutions:
            sols.append({str(k): str(v) for k, v in sol.items()})
        return {"ok": True, "type": "system", "solutions": sols}
    except Exception as e:
        return {"ok": False, "error": f"Solver error: {e}"}


_WORKER_MANAGER = _WorkerManager()



@lru_cache(maxsize=2048)
def _worker_eval_cached(preprocessed_expr: str) -> str:
    """Evaluate via persistent worker if available; else subprocess. Returns JSON text."""
    resp = _WORKER_MANAGER.request({"type": "eval", "preprocessed": preprocessed_expr}, timeout=WORKER_TIMEOUT)
    if isinstance(resp, dict):
        return json.dumps(resp)
    cmd = _build_self_cmd(["--worker", "--expr", preprocessed_expr])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT)
    return proc.stdout or ""


def evaluate_safely(expr: str, timeout: int = WORKER_TIMEOUT) -> Dict[str, Any]:
    """Spawn a short-lived worker subprocess (same script/executable) to evaluate the preprocessed expr.
    Returns the parsed result as dict.
    """
    # Preprocess here (string operations are cheap)
    try:
        pre = preprocess(expr)
    except Exception as e:
        return {"ok": False, "error": f"Preprocess error: {e}"}

    try:
        stdout_text = _worker_eval_cached(pre)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Evaluation timed out."}
    try:
        data = json.loads(stdout_text)
        return data
    except Exception as e:
        return {"ok": False, "error": f"Invalid worker output: {e}."}


def _build_self_cmd(args: List[str]) -> List[str]:
    """
    Return a subprocess argv list that invokes this program (script or bundled exe)
    with the provided args.

    - When running normally: ["<python>", "<path/to/script.py>", ...args...]
    - When frozen (PyInstaller onefile): ["<path/to/exe>", ...args...]
    """
    if getattr(sys, "frozen", False):
        # Use the bundled executable path (sys.argv[0]) to launch a new copy of the exe.
        return [os.path.realpath(sys.argv[0])] + args
    else:
        # Dev mode: invoke the interpreter with the .py file path as usual.
        return [sys.executable, os.path.realpath(__file__)] + args


# ---------------------------------------------------------------------------
# Solver helpers (these use SymPy objects inside the main process for secondary tasks,
# but do NOT parse/evaluate user strings — parsing is reserved to the worker)
# NOTE: these functions expect inputs as SymPy expressions or strings that will be passed
# through evaluate_safely() when comi o0ng from user input.
# ---------------------------------------------------------------------------

def is_pell_equation_from_eq(eq: sp.Eq) -> bool:
    syms = list(eq.free_symbols)
    if len(syms) != 2:
        return False
    try:
        if not eq.rhs.equals(1):
            return False
    except Exception:
        return False
    expanded_lhs = sp.expand(eq.lhs)
    return expanded_lhs.coeff(syms[0] ** 2) == 1 and expanded_lhs.coeff(syms[1] ** 2) != 0


def fundamental_solution(D: int) -> Tuple[int, int]:
    sqrt_D = sp.sqrt(D)
    if sqrt_D.is_rational:
        raise ValueError("D must be non-square for Pell's equation")
    cf = sp.continued_fraction(sqrt_D)
    period = cf[1:]
    L = len(period)
    conv_index = L - 1 if L % 2 == 0 else 2 * L - 1
    terms = cf[:1] + period * (((conv_index + 1) - len(cf)) // L + 1)
    terms = terms[:conv_index + 1]
    num, den = sp.continued_fraction_reduce(terms)
    return int(num), int(den)


def solve_pell_equation_from_eq(eq: sp.Eq) -> str:
    syms = list(eq.free_symbols)
    x_sym, y_sym = syms[0], syms[1]
    expanded_lhs = sp.expand(eq.lhs)
    coeff_y2 = expanded_lhs.coeff(y_sym ** 2)
    D = -coeff_y2
    x1, y1 = fundamental_solution(int(D))
    n = sp.symbols("n", integer=True)
    sol_x = ((x1 + y1 * sp.sqrt(D)) ** n + (x1 - y1 * sp.sqrt(D)) ** n) / 2
    sol_y = ((x1 + y1 * sp.sqrt(D)) ** n - (x1 - y1 * sp.sqrt(D)) ** n) / (2 * sp.sqrt(D))
    return f"{x_sym} = {sol_x}\n{y_sym} = {sol_y}"


# ---------------------------------------------------------------------------
# CLI logic: interactive loop and high-level handlers
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    ok: bool
    result: Optional[str] = None
    approx: Optional[str] = None
    free_symbols: Optional[List[str]] = None
    error: Optional[str] = None


def eval_user_expression(expr: str) -> EvalResult:
    """Evaluate user expression via worker sandbox and return structured result."""
    data = evaluate_safely(expr)
    if not data.get("ok"):
        return EvalResult(ok=False, error=data.get("error") or "Unknown error")
    return EvalResult(ok=True, result=data.get("result"), approx=data.get("approx"),
                      free_symbols=data.get("free_symbols"))


# For the rest of solver functionality (like solving equations/systems),
# we continue to rely on evaluate_safely to parse each subexpression and then
# use SymPy to solve. This keeps user input parsing confined to the worker.

def solve_single_equation_cli(eq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """Solve a single equation string (user input). Returns dict for printing/JSON."""
    # split into LHS/RHS locally (safe, string operation)
    parts = eq_str.split("=", 1)
    if len(parts) != 2:
        return {"ok": False, "error": "Invalid equation format (need exactly one '=')."}
    lhs_s, rhs_s = parts[0].strip(), parts[1].strip() or "0"
    # Evaluate both sides safely
    lhs = evaluate_safely(lhs_s)
    if not lhs.get("ok"):
        return {"ok": False, "error": f"LHS parse error: {lhs.get('error')}"}
    rhs = evaluate_safely(rhs_s)
    if not rhs.get("ok"):
        return {"ok": False, "error": f"RHS parse error: {rhs.get('error')}"}
    # Convert to SymPy objects by parsing string representations _via_ parse_preprocessed.
    try:
        left_expr = parse_preprocessed(lhs["result"])
        right_expr = parse_preprocessed(rhs["result"])
    except Exception as e:
        return {"ok": False, "error": f"Internal parse error assembling SymPy expressions: {e}"}
    equation = sp.Eq(left_expr, right_expr)
    # Pell detection
    if is_pell_equation_from_eq(equation):
        try:
            pell_str = solve_pell_equation_from_eq(equation)
            return {"ok": True, "type": "pell", "solution": prettify_expr(pell_str)}
        except Exception as e:
            return {"ok": False, "error": f"Pell solver error: {e}"}
    # Solve
    symbols = list(equation.free_symbols)
    if not symbols:
        # First try a direct symbolic simplification test.
        try:
            simp = sp.simplify(left_expr - right_expr)
            # If SymPy reduces to exact 0, it's an identity.
            if simp == 0:
                return {"ok": True, "type": "identity_or_contradiction", "result": "Identity"}
        except Exception:
            simp = None

        # Fallback: high-precision numeric check. This handles cases like I**I == exp(-pi/2)
        try:
            diff = sp.N(left_expr - right_expr, 60)
            re_diff = sp.re(diff)
            im_diff = sp.im(diff)
            tol = sp.N(10) ** (-45)
            if abs(re_diff) < tol and abs(im_diff) < tol:
                return {"ok": True, "type": "identity_or_contradiction", "result": "Identity (numeric)"}
            else:
                return {"ok": True, "type": "identity_or_contradiction", "result": "Contradiction (numeric)"}
        except Exception:
            return {"ok": True, "type": "identity_or_contradiction",
                    "result": "Contradiction (unable to confirm identity symbolically or numerically)"}

    # Helper: numeric root-finding fallback for single-variable trig-ish equations
    def _numeric_roots_for_single_var(f_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi), guesses=36):
        roots = []
        # Prefer polynomial root finding when applicable
        try:
            poly = sp.Poly(f_expr, sym)
            if poly is not None and poly.total_degree() > 0:
                for r in poly.nroots():
                    if abs(sp.im(r)) < 1e-8:
                        roots.append(float(sp.re(r)))
                uniq = sorted(set(round(x, 12) for x in roots))
                return [sp.N(r) for r in uniq]
        except Exception:
            pass

        a = float(interval[0])
        b = float(interval[1])
        guesses_list = [a + (b - a) * i / guesses for i in range(guesses + 1)]
        for g in guesses_list:
            try:
                r = sp.nsolve(f_expr, sym, g, tol=1e-12, maxsteps=100)
                if abs(sp.im(r)) > 1e-8:
                    continue
                r_real = float(sp.re(r))
                if not any(abs(existing - r_real) < 1e-6 for existing in roots):
                    roots.append(r_real)
            except Exception:
                continue
        roots_sorted = sorted(roots)
        return [sp.N(r) for r in roots_sorted]

    try:
        # If user requested a specific variable, solve for it only.
        if find_var:
            sym = sp.symbols(find_var)
            if sym not in symbols:
                return {"ok": False, "error": f"Variable '{find_var}' not present."}
            sols = sp.solve(equation, sym)
            exacts = [str(s) for s in sols] if isinstance(sols, (list, tuple)) else [str(sols)]
            approx = []
            for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                try:
                    approx.append(str(sp.N(s)))
                except Exception:
                    approx.append(None)
            return {"ok": True, "type": "equation", "exact": exacts, "approx": approx}

        # No specific variable requested:
        if len(symbols) == 1:
            sym = symbols[0]
            try:
                sols = sp.solve(equation, sym)
            except Exception as e:
                # If symbolic solve fails and equation involves trig functions, try numeric root finding.
                if equation.has(sp.sin, sp.cos, sp.tan):
                    f = sp.simplify(left_expr - right_expr)
                    numeric_roots = _numeric_roots_for_single_var(f, sym, interval=(-4 * sp.pi, 4 * sp.pi), guesses=120)
                    if numeric_roots:
                        exacts = [str(r) for r in numeric_roots]
                        approx = [str(sp.N(r)) for r in numeric_roots]
                        return {"ok": True, "type": "equation", "exact": exacts, "approx": approx}
                return {"ok": False, "error": f"Solving error: {e}"}
            exacts = [str(s) for s in sols] if isinstance(sols, (list, tuple)) else [str(sols)]
            approx = []
            for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                try:
                    approx.append(str(sp.N(s)))
                except Exception:
                    approx.append(None)
            return {"ok": True, "type": "equation", "exact": exacts, "approx": approx}

        # Multiple symbols -> attempt to isolate each symbol separately (unchanged)
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
                    except Exception:
                        approx_list.append(None)
                multi_approx[str(sym)] = approx_list
            except Exception as e:
                multi_solutions[str(sym)] = [f"Error: {e}"]
                multi_approx[str(sym)] = [None]
        return {"ok": True, "type": "multi_isolate", "solutions": multi_solutions, "approx": multi_approx}
    except Exception as e:
        return {"ok": False, "error": f"Solving error: {e}"}



def solve_inequality_cli(ineq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
    """Solve inequality (string)."""
    try:
        # Use the specialized fallback parser first to handle chained inequalities
        parsed = _parse_relational_fallback(ineq_str)
    except Exception as e:
        return {"ok": False, "error": f"Failed to parse inequality: {e}"}

    free_syms = list(parsed.free_symbols) if hasattr(parsed, "free_symbols") else []

    if find_var:
        # ** THE FIX IS HERE **
        # Find the symbol object from the parsed expression that matches the name.
        # This ensures we use the exact same object the solver expects.
        target_sym = None
        for sym in free_syms:
            if str(sym) == find_var:
                target_sym = sym
                break
        if target_sym is None:
            return {"ok": False, "error": f"Variable '{find_var}' not found in the expression."}
        vars_to_solve = [target_sym]
    else:
        if not free_syms:
            # This can happen for trivial inequalities like 1 < 2, which is just 'True'
            try:
                is_true = sp.simplify(parsed)
                return {"ok": True, "type": "inequality", "solutions": {"result": str(is_true)}}
            except Exception:
                return {"ok": False, "error": "No variable found in inequality"}
        vars_to_solve = free_syms

    results = {}
    for v in vars_to_solve:
        try:
            # The solver expects a list of inequalities.
            # If our parser created an 'And' object, we unpack its arguments into a list.
            if isinstance(parsed, sp.And):
                ineqs_to_solve = list(parsed.args)
            else:
                ineqs_to_solve = [parsed]

            sol = sp.reduce_inequalities(ineqs_to_solve, v)
            results[str(v)] = str(sol)
        except NotImplementedError:
            results[str(v)] = "Solver not implemented for this type of inequality."
        except Exception as e:
            results[str(v)] = f"Error solving for {v}: {e}"

    return {"ok": True, "type": "inequality", "solutions": results}


def _parse_relational_fallback(rel_str: str):
    """
    Parses a chained relational string like "1 < 2*x < 5" by splitting it,
    safely evaluating each component, and reconstructing it as a SymPy And(...) object.
    """
    # Split the string by relational operators, keeping the operators
    parts = re.split(r"(<=|>=|<|>)", rel_str)
    if len(parts) == 1:
        # Not a chained inequality, evaluate it directly and safely
        res = evaluate_safely(rel_str)
        if not res.get("ok"):
            raise ValueError(res.get("error"))
        return parse_preprocessed(res["result"])

    # Extract expressions and operators
    expr_parts = parts[::2]  # e.g., ['1', '2*x', '5']
    ops = parts[1::2]  # e.g., ['<', '<']

    # Safely evaluate each expression part
    parsed_parts = []
    for p in expr_parts:
        p_strip = p.strip()
        if not p_strip: continue
        res = evaluate_safely(p_strip)
        if not res.get("ok"):
            raise ValueError(f"Failed to parse component '{p_strip}': {res.get('error')}")
        parsed_parts.append(parse_preprocessed(res["result"]))

    if len(parsed_parts) != len(ops) + 1:
        raise ValueError("Invalid inequality structure.")

    # Map string operators to SymPy classes
    op_map = {"<": sp.Lt, ">": sp.Gt, "<=": sp.Le, ">=": sp.Ge}

    # Build a list of individual relational objects
    relations = []
    for i, op_str in enumerate(ops):
        op_func = op_map.get(op_str)
        if not op_func:
            raise ValueError(f"Unknown operator: {op_str}")
        relations.append(op_func(parsed_parts[i], parsed_parts[i + 1]))

    # Combine them with And if there's more than one
    return sp.And(*relations) if len(relations) > 1 else relations[0]


# ---------------------------------------------------------------------------
# Worker command-line entrypoint
# ---------------------------------------------------------------------------

def worker_main(argv: List[str]) -> int:
    """Worker entrypoint executed in a subprocess. Receives --expr PREPROCESSED_EXPRESSION."""
    parser = argparse.ArgumentParser(prog="algebra_worker", add_help=False)
    parser.add_argument("--expr", required=True, help="Preprocessed expression string (already canonicalized)")
    args = parser.parse_args(argv)
    expr = args.expr
    out = worker_evaluate(expr)
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    return 0


def worker_solve_main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="algebra_worker_solve", add_help=False)
    parser.add_argument("--payload", required=True, help="JSON payload containing equations and optional find var")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.payload)
        eqs_input = payload.get("equations", [])
        eq_objs = []
        for item in eqs_input:
            lhs_s = item.get("lhs")
            rhs_s = item.get("rhs")
            if lhs_s is None or rhs_s is None:
                continue
            try:
                lhs_expr = parse_preprocessed(lhs_s)
                rhs_expr = parse_preprocessed(rhs_s)
                eq_objs.append(sp.Eq(lhs_expr, rhs_expr))
            except Exception as e:
                sys.stdout.write(json.dumps({"ok": False, "error": f"Parse error in worker-solve: {e}"}))
                sys.stdout.flush()
                return 0

        if not eq_objs:
            sys.stdout.write(json.dumps({"ok": False, "error": "No valid equations provided to worker-solve."}))
            sys.stdout.flush()
            return 0

        if HAS_RESOURCE:
            _limit_resources()

        try:
            solutions = sp.solve(eq_objs, dict=True)
        except Exception as e:
            sys.stdout.write(json.dumps({"ok": False, "error": f"Solver error: {e}"}))
            sys.stdout.flush()
            return 0

        if not solutions:
            sys.stdout.write(json.dumps({"ok": False, "error": "No solution found (sp.solve returned empty)."}))
            sys.stdout.flush()
            return 0

        sols = []
        for sol in solutions:
            sols.append({str(k): str(v) for k, v in sol.items()})
        out = {"ok": True, "type": "system", "solutions": sols}
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
        return 0

    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "error": f"Worker-solve failed: {e}"}))
        sys.stdout.flush()
        return 1


# ---------------------------------------------------------------------------
# Top-level CLI and REPL
# ---------------------------------------------------------------------------

def print_result_pretty(res: Dict[str, Any], json_mode: bool = False) -> None:
    if json_mode:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if not res.get("ok"):
        print("Error:", res.get("error"))
        return
    typ = res.get("type", "value")
    # --- equation / numeric / multi cases (unchanged) ---
    if typ == "equation":
        print("Exact:", ", ".join(format_solution(x) for x in res.get("exact", [])))
        if res.get("approx"):
            # filter out None approximations gracefully
            approx_display = ", ".join(format_number(x) for x in res.get("approx", []) if x is not None)
            if approx_display:
                print("Approx:", approx_display)
    elif typ == "multi_isolate":
        sols = res.get("solutions", {})
        approx = res.get("approx", {})
        # Print each variable on its own line, show exact solutions and decimals when available
        for var, sol_list in sols.items():
            if isinstance(sol_list, (list, tuple)):
                formatted = ", ".join(format_solution(s) for s in sol_list)
            else:
                formatted = format_solution(sol_list)
            print(f"{var} = {formatted}")
            approx_list = approx.get(var)
            if approx_list:
                approx_display = ", ".join(format_number(x) for x in approx_list if x is not None)
                if approx_display:
                    print(f"  Decimal: {approx_display}")
    elif typ == "inequality":
        for k, v in res.get("solutions", {}).items():
            # ** This is the updated line **
            formatted_v = format_inequality_solution(str(v))
            print(f"Solution for {k}: {formatted_v}")
    elif typ == "pell":
        print("Pell parametric solution:\n", res.get("solution"))
    elif typ == "identity_or_contradiction":
        print(res.get("result"))

    # --- new: plain evaluated value handling with expansion ---
    elif typ == "value":
        res_str = res.get("result")
        approx = res.get("approx")
        # print canonical form first
        if res_str is None:
            print(res)
            return
        try:
            print(f"{res_str}")
        except Exception:
            print(res_str)
        # try to parse and expand; if expanded differs, show it
        try:
            parsed = parse_preprocessed(res_str)
            expanded = sp.expand(parsed)
            if str(expanded) != str(parsed):
                print(f"Expanded: {format_solution(expanded)}")
        except Exception:
            # ignore parse/expand failures silently
            pass
        if approx:
            print("Decimal:", approx)
    else:
        # generic fallback
        print(res)


def repl_loop(json_mode: bool = False) -> None:
    # Optional history/editing support when available
    try:
        import readline  # type: ignore
    except Exception:
        readline = None  # type: ignore
    print("Kalkulator Aljabar — type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not raw:
            continue
        # Allow command-style --eval inside REPL (convenience)
        if raw.startswith("--eval"):
            parts = raw.split(None, 1)
            if len(parts) == 1:
                print("Usage: --eval <expression>")
                continue
            # take the expression after --eval, strip surrounding quotes if present
            expr = parts[1].strip()
            if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
                expr = expr[1:-1]
            try:
                # Delegate to the same handlers used by the CLI
                if any(op in expr for op in ("<", ">", "<=", ">=")):
                    res = solve_inequality_cli(expr, None)
                    print_result_pretty(res, json_mode=json_mode)
                elif "=" in expr:
                    pts = [p.strip() for p in expr.split(",") if p.strip()]
                    if len(pts) > 1:
                        # system of equations
                        res = handle_system_main(expr, None)
                        print_result_pretty(res, json_mode=json_mode)
                    else:
                        # single equation
                        res = solve_single_equation_cli(expr, None)
                        print_result_pretty(res, json_mode=json_mode)
                else:
                    # plain expression -> evaluate safely
                    eva = evaluate_safely(expr)
                    if not eva.get("ok"):
                        print("Error:", eva.get("error"))
                    else:
                        res_str = eva.get("result")
                        # print canonical form (keeps your previous superscript formatting)
                        print(f"{expr} = {format_superscript(res_str)}")
                        # attempt to parse and expand, print expanded version if it differs
                        try:
                            parsed = parse_preprocessed(res_str)
                            expanded = sp.expand(parsed)
                            if str(expanded) != str(parsed):
                                print(f"Expanded: {format_solution(expanded)}")
                        except Exception:
                            pass
                        if eva.get("approx"):
                            print("Decimal:", eva.get("approx"))
            except Exception as e:
                print("Error handling --eval in REPL:", e)
            continue
        # end --eval convenience handling

        cmd = raw.lower()
        if cmd in ("quit", "exit"):
            print("Goodbye.")
            break
        if cmd in ("help", "?", "--help"):
            print_help_text()
            continue
        # simple routing: inequalities if contains <, > else if contains comma many parts -> system/subst
        try:
            if any(op in raw for op in ("<", ">", "<=", ">=")):
                res = solve_inequality_cli(raw, None)
                print_result_pretty(res, json_mode=json_mode)
                continue
            # find token support
            find_tokens = re.findall(r"\bfind\s+(\w+)\b", raw, re.IGNORECASE)
            find = find_tokens[0] if find_tokens else None
            raw_no_find = re.sub(r"\bfind\s+\w+\b", "", raw, flags=re.IGNORECASE).strip()
            parts = [p.strip() for p in raw_no_find.split(",") if p.strip()]

            if not parts:
                print("No valid parts parsed.")
                continue
            all_assign = all("=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip()) for p in parts)
            all_eq = all("=" in p for p in parts)
            if all_assign and len(parts) > 0:
                # If user asked to 'find <var>' then treat the line as a system request
                # instead of the assignment-only shortcut.
                if find:
                    res = handle_system_main(raw_no_find, find)
                    print_result_pretty(res, json_mode=json_mode)
                    continue

                # chain assignments: reuse earlier logic but through worker
                subs = {}
                for p in parts:
                    left, right = p.split("=", 1)
                    var = left.strip()
                    rhs = right.strip() or "0"
                    res = evaluate_safely(rhs)
                    if not res.get("ok"):
                        print("Error evaluating assignment RHS:", res.get("error"))
                        continue
                    # parse into sympy object and substitute existing subs
                    try:
                        val = parse_preprocessed(res["result"]).subs(subs)
                    except Exception as e:
                        print("Error assembling assignment value:", e)
                        continue
                    subs[var] = val
                    print(f"{var} = {format_solution(val)}")
                continue
            if len(parts) > 1 and all_eq:
                # system of equations
                find = find if find else None
                res = handle_system_main(raw_no_find, find)
                print_result_pretty(res, json_mode=json_mode)
                continue
            elif len(parts) > 1:
                # substitution
                print(
                    "Substitution not implemented in REPL (use eval mode) — please use --eval or provide a single expression.")
                continue
            else:
                # single expression or equation
                part = parts[0]
                if "=" in part:
                    find = find if find else None
                    res = solve_single_equation_cli(part, find)
                    print_result_pretty(res, json_mode=json_mode)
                else:
                    eva = evaluate_safely(part)
                    if not eva.get("ok"):
                        print("Error:", eva.get("error"))
                    else:
                        res_str = eva.get("result")
                        print(f"{part} = {format_superscript(res_str)}")
                        try:
                            parsed = parse_preprocessed(res_str)
                            expanded = sp.expand(parsed)
                            if str(expanded) != str(parsed):
                                print(f"Expanded: {format_solution(expanded)}")
                        except Exception:
                            pass
                        if eva.get("approx"):
                            print("Decimal:", eva.get("approx"))
                continue
        except Exception:
            print("An error occurred. Please check your input and try again.")
            continue


# A small tolerance used for rounding near-zero numeric results:
ZERO_TOL = 1e-12



def handle_system_main(raw_no_find: str, find_token: Optional[str]) -> Dict[str, Any]:
    """
    Build serialized equations from raw input, try lightweight assignment-substitution shortcut
    for simple cases (assignments + a requested derived variable), otherwise call the
    sandboxed worker subprocess to solve the system.

    Added behavior: if there's exactly one equation and user asked `find <var>`, attempt
    to symbolically isolate that single equation for the requested variable here (in-process)
    before calling the worker. This handles cases like:
        k = (1/2)*m*v**2, find m
    which can be algebraically solved for m even though the system is underdetermined.
    """
    parts = [p.strip() for p in raw_no_find.split(",") if p.strip()]

    # Evaluate each side via evaluate_safely and collect info
    eqs_serialized = []
    assignments = {}  # var -> {"result": str, "approx": str or None}
    for p in parts:
        if "=" not in p:
            continue
        lhs, rhs = p.split("=", 1)
        lhs_s = lhs.strip()
        rhs_s = rhs.strip()
        lhs_eval = evaluate_safely(lhs_s)
        if not lhs_eval.get("ok"):
            return {"ok": False, "error": f"LHS parse error: {lhs_eval.get('error')}"}
        rhs_eval = evaluate_safely(rhs_s)
        if not rhs_eval.get("ok"):
            return {"ok": False, "error": f"RHS parse error: {rhs_eval.get('error')}"}

        # If this is a simple assignment like 'x = <expr>' where lhs is a simple var name, record it
        if VAR_NAME_RE.match(lhs_s):
            assignments[lhs_s] = {"result": rhs_eval.get("result"), "approx": rhs_eval.get("approx")}
        # Save serialized pair for worker fallback
        eqs_serialized.append({"lhs": lhs_eval.get("result"), "rhs": rhs_eval.get("result")})

    if not eqs_serialized:
        return {"ok": False, "error": "No equations parsed."}

    # NEW: If user asked to find a var and there's exactly one equation, try in-process isolation.
    if find_token and len(eqs_serialized) == 1:
        pair = eqs_serialized[0]
        lhs_s = pair.get("lhs")
        rhs_s = pair.get("rhs")
        try:
            lhs_expr = parse_preprocessed(lhs_s)
            rhs_expr = parse_preprocessed(rhs_s)
            equation = sp.Eq(lhs_expr, rhs_expr)
            sym = sp.symbols(find_token)
            # only try if the requested symbol actually appears in the equation
            if sym in equation.free_symbols:
                try:
                    sols = sp.solve(equation, sym)
                except Exception:
                    sols = []
                if sols:
                    exacts = [str(s) for s in sols] if isinstance(sols, (list, tuple)) else [str(sols)]
                    approx = []
                    for s in sols if isinstance(sols, (list, tuple)) else [sols]:
                        try:
                            approx.append(str(sp.N(s)))
                        except Exception:
                            approx.append(None)
                    return {"ok": True, "type": "system_var", "exact": exacts, "approx": approx}
        except Exception:
            # if anything goes wrong here, fallback to worker-solve below
            pass

    # Shortcut: if user requested a single variable 'find_token' and there exist
    # assignments that allow direct substitution (no general solving), try numeric substitution.
    if find_token:
        # Find an equation that defines find_token directly: look for equation where lhs or rhs equals find_token
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
            except Exception:
                expr_sym = None

            if expr_sym is not None:
                # Build substitution map from available assignments (use numeric approx when available)
                subs_map = {}
                for var, info in assignments.items():
                    if info.get("approx") is not None:
                        try:
                            subs_map[sp.symbols(var)] = sp.sympify(info.get("approx"))
                        except Exception:
                            try:
                                subs_map[sp.symbols(var)] = parse_preprocessed(info.get("result"))
                            except Exception:
                                pass
                    else:
                        try:
                            subs_map[sp.symbols(var)] = parse_preprocessed(info.get("result"))
                        except Exception:
                            pass

                if subs_map:
                    try:
                        value = expr_sym.subs(subs_map)
                        # Try numeric approximation and apply zero tolerance
                        try:
                            approx_obj = sp.N(value)
                            # If near zero (both real and imag), return 0
                            if abs(sp.re(approx_obj)) < ZERO_TOL and abs(sp.im(approx_obj)) < ZERO_TOL:
                                return {"ok": True, "type": "system_var", "exact": ["0"], "approx": ["0"]}
                            approx_val = str(approx_obj)
                        except Exception:
                            approx_val = None
                        return {"ok": True, "type": "system_var", "exact": [str(value)], "approx": [approx_val]}
                    except Exception:
                        # fallback to worker-solve
                        pass

    # Fallback: call worker-solve for general systems
    payload = {"equations": eqs_serialized, "find": find_token}
    try:
        stdout_text = _worker_solve_cached(json.dumps(payload))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Solving timed out (worker)."}

    try:
        data = json.loads(stdout_text)
    except Exception as e:
        return {"ok": False, "error": f"Invalid worker-solve output: {e}."}

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
        return {"ok": False, "error": f"No solution found for variable {find_token}."}

    approx_vals = []
    for vstr in found_vals:
        try:
            approx_vals.append(str(sp.N(sp.sympify(vstr))))
        except Exception:
            approx_vals.append(None)

    return {"ok": True, "type": "system_var", "exact": found_vals, "approx": approx_vals}


@lru_cache(maxsize=256)
def _worker_solve_cached(payload_json: str) -> str:
    """Run system solve via persistent worker if available; else subprocess. Return JSON text."""
    try:
        payload = json.loads(payload_json)
    except Exception:
        payload = {"equations": []}
    resp = _WORKER_MANAGER.request({"type": "solve", "payload": payload}, timeout=WORKER_TIMEOUT)
    if isinstance(resp, dict):
        return json.dumps(resp)
    cmd = _build_self_cmd(["--worker-solve", "--payload", payload_json])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT)
    return proc.stdout or ""


def print_help_text():
    help_text = help_text = f""" version {VERSION}

Usage (one-line input):
- Expression → evaluated (e.g. 2+3)
- Equation → solved (e.g. 2*x+3=7). Add ", find x" to request a specific variable.
- System → separate equations with commas (e.g. x+y=3, x-y=1)
- Inequality → use <, >, <=, >= (e.g. 1 < 2*x < 5)
- REPL chained assignments: a = 2, b = a+3 (evaluated right→left)

Commands:
- -e/--eval "<EXPR>"  evaluate once and exit
- -j/--json           machine-friendly JSON output
- -v/--version        show program version
- In REPL: help, quit, exit

Supported constants & functions:
- Constants: pi, E, I
- Functions: sqrt(), sin(), cos(), tan(), asin(), acos(), atan(), log()/ln(), exp(), Abs()

Input conveniences:
- '^' → '**'   ;  '²','³' → **2, **3
- '50%' → (50/100) ; '√' → sqrt(...)
- Implicit multiplication allowed (2x → 2*x)
- Balanced-parentheses check

Example inputs (organized by category)

Basic arithmetic
- 1+1
- 1-1
- 5*10
- 1/2

Fractions & percent
- 50%            (interpreted as 50/100)
- 50/100

Exponents & roots
- 2^2
- 2**2
- 2²
- √(2)
- sqrt(16)

Constants & simple functions
- pi
- E
- I
- sin(pi/6)
- sin(pi/2)
- cos(0)
- tan(pi/4)

Trigonometry (equations)
- 2*sin(x) + sqrt(3) = 0
- 3*tan(x) + sqrt(3) = 0
- 2*sin(x)**2 - 1 = 0
- 3*sin(x) + 2 = 1

Complex numbers & logs
- ln(I)
- log(2)
- log(pi)
- E^(I*pi)
- E^(I*pi) + 1 = x, find x

Assignments & REPL examples
- a = 2, b = a+3
- a = 1, b = 2, c = a + b, find c
- r = 5, pi*r^2 = n, find n

Single linear / algebraic equations
- 1 = 1
- 1 = 0
- x + y = 3, x - y = 1   (also a small system example)

Systems of linear equations
- x + y = 3, x - y = 1
- 2v + w + x + y + z = 5,
  v + 2w + x + y + z = 5,
  v + w + 2x + y + z = 6,
  v + w + x + 2y + z = 7,
  v + w + x + y + 2z = 8,
  v + w + x + y + z = a, find a
sqrt(x)+y=7, x+sqrt(y)=11
Polynomials / algebraic roots
- 6*x^2 - 17*x + 1 = 0, find x
- x^3 - 4*x^2 - 9*x + 36 = 0, find x
- x^3 - 9*x + 36 = 0, find x

Complex algebra / tricky expressions
- pi + E + I + sqrt(2) + sin(pi/2) + cos(0) + tan(pi/4) + asin(1) + acos(0) + atan(1) + log(10) + ln(E) + exp(1) + Abs(-5)
- sin(1/x)**-1 = (sin(x)/1)**-1, find x

Preprocessing demonstration (how input is normalized)
- 2^2        (becomes 2**2)
- 2²         (becomes 2**2)
- 50%        (becomes (50/100))
- √(2)       (becomes sqrt(2))
- 2x         (becomes 2*x via implicit multiplication)

Tips:
- Use --json for automatic parsing.
- If a computation times out, simplify the expression (smaller exponents, fewer nested functions).
- Non-finite results (division by zero, infinity) are reported as errors.

Calculus & matrices (new):
- diff(x^3, x)
- integrate(sin(x), x)
- factor(x^3 - 1)
- expand((x+1)^3)
- Matrix([[1,2],[3,4]])
- det(Matrix([[1,2],[3,4]]))

Property of Muhammad Akhiel al Syahbana — 31/October/2025
"""

    print(help_text)


# ---------------------------------------------------------------------------
# Main argument parsing & orchestration
# ---------------------------------------------------------------------------

def main_entry(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="algebra_solver_secure")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expr", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-solve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--payload", type=str, help=argparse.SUPPRESS)
    parser.add_argument("-e", "--eval", type=str, help="Evaluate one expression and exit (non-interactive)", dest="eval_expr")
    parser.add_argument("-j", "--json", action="store_true", help="Emit JSON for machine parsing")
    parser.add_argument("-v", "--version", action="store_true", help="Show program version")
    args = parser.parse_args(argv)

    # Worker mode (internal)
    if args.worker:
        out = worker_evaluate(args.expr or "")
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
        return 0

    # Worker-solve dispatched: pass only the payload to avoid argparse conflicts in the worker
    # Worker-solve dispatched: directly invoke local worker_solve_main
    if args.worker_solve:
        return worker_solve_main(['--payload', args.payload or ''])

    if args.version:
        print(VERSION)
        return 0

    if args.eval_expr:
        expr = args.eval_expr.strip()
        import re
        # detect "find <var>" (case-insensitive) and strip it from the expression so the same semantics as REPL apply
        find_tokens = re.findall(r"\bfind\s+(\w+)\b", expr, re.IGNORECASE)
        find = find_tokens[0] if find_tokens else None
        raw_no_find = re.sub(r"\bfind\s+\w+\b", "", expr, flags=re.IGNORECASE).strip()

        # heuristics: if contains inequality operators delegate to inequality solver
        if any(op in raw_no_find for op in ("<", ">", "<=", ">=")):
            res = solve_inequality_cli(raw_no_find, find)
        elif "=" in raw_no_find:
            # single equation or system depending on commas
            parts = [p.strip() for p in raw_no_find.split(",") if p.strip()]
            if len(parts) > 1:
                res = handle_system_main(raw_no_find, find)
            else:
                res = solve_single_equation_cli(parts[0], find)
        else:
            # plain expression: evaluate and return type "value" so print_result_pretty displays it (and expansion)
            eva = evaluate_safely(raw_no_find)
            if not eva.get("ok"):
                res = {"ok": False, "error": eva.get("error")}
            else:
                res = {"ok": True, "type": "value", "result": eva.get("result"), "approx": eva.get("approx")}
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print_result_pretty(res, json_mode=False)
        return 0

    # interactive
    repl_loop(json_mode=args.json)
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow frozen executables (PyInstaller) on Windows to spawn child processes.
    # multiprocessing.freeze_support() is a no-op on non-Windows platforms.
    try:
        from multiprocessing import freeze_support
    except Exception:
        # extremely defensive fallback (shouldn't happen)
        def freeze_support():
            return None

    # Call freeze_support early, before any child-process spawning.
    try:
        freeze_support()
    except Exception:
        # If something unexpected goes wrong, continue anyway.
        pass

    # Delegate to modular CLI entrypoint
    try:
        from kalkulator_pkg.cli import main_entry as pkg_main_entry
        sys.exit(pkg_main_entry(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
