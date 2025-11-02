"""Centralized configuration for Kalkulator.

This module defines:
- Resource limits (CPU time, memory, timeouts)
- Input validation limits (length, depth, node count)
- Cache sizes for performance optimization
- Allowed SymPy functions and transformations
- Regex patterns for parsing
- Solver configuration options

Configuration can be overridden via:
- CLI flags (see cli.py)
- Environment variables (prefixed with KALKULATOR_)
"""

import os
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
)

# Version is defined in pyproject.toml [project] section
# Import here for backward compatibility
try:
    import importlib.metadata

    VERSION = importlib.metadata.version("kalkulator")
except Exception:
    # Fallback if package not installed
    VERSION = "1.0.0"

# Resource limits (can be overridden via environment variables)
WORKER_CPU_SECONDS = int(os.getenv("KALKULATOR_WORKER_CPU_SECONDS", "30"))
WORKER_AS_MB = int(os.getenv("KALKULATOR_WORKER_AS_MB", "400"))
WORKER_TIMEOUT = int(os.getenv("KALKULATOR_WORKER_TIMEOUT", "60"))
ENABLE_PERSISTENT_WORKER = (
    os.getenv("KALKULATOR_ENABLE_PERSISTENT_WORKER", "true").lower() == "true"
)
WORKER_POOL_SIZE = int(
    os.getenv("KALKULATOR_WORKER_POOL_SIZE", "4")
)  # Number of parallel worker processes

# Solver configuration
NUMERIC_FALLBACK_ENABLED = (
    os.getenv("KALKULATOR_NUMERIC_FALLBACK_ENABLED", "true").lower() == "true"
)
OUTPUT_PRECISION = int(os.getenv("KALKULATOR_OUTPUT_PRECISION", "6"))
SOLVER_METHOD = os.getenv(
    "KALKULATOR_SOLVER_METHOD", "auto"
)  # "auto", "symbolic", "numeric"

# Input validation limits
MAX_INPUT_LENGTH = int(os.getenv("KALKULATOR_MAX_INPUT_LENGTH", "10000"))  # characters
MAX_EXPRESSION_DEPTH = int(
    os.getenv("KALKULATOR_MAX_EXPRESSION_DEPTH", "100")
)  # tree depth
MAX_EXPRESSION_NODES = int(
    os.getenv("KALKULATOR_MAX_EXPRESSION_NODES", "5000")
)  # total nodes

# Cache configuration
CACHE_SIZE_PARSE = int(os.getenv("KALKULATOR_CACHE_SIZE_PARSE", "1024"))
CACHE_SIZE_EVAL = int(os.getenv("KALKULATOR_CACHE_SIZE_EVAL", "2048"))
CACHE_SIZE_SOLVE = int(os.getenv("KALKULATOR_CACHE_SIZE_SOLVE", "256"))

# Numeric solver configuration
MAX_NSOLVE_GUESSES = int(
    os.getenv("KALKULATOR_MAX_NSOLVE_GUESSES", "50")
)  # Optimized for balance between speed and thoroughness

# Numeric tolerance constants (replacing magic numbers throughout codebase)
NUMERIC_TOLERANCE = float(
    os.getenv("KALKULATOR_NUMERIC_TOLERANCE", "1e-8")
)  # For imaginary part filtering
ROOT_SEARCH_TOLERANCE = float(
    os.getenv("KALKULATOR_ROOT_SEARCH_TOLERANCE", "1e-12")
)  # For root finding precision
MAX_NSOLVE_STEPS = int(
    os.getenv("KALKULATOR_MAX_NSOLVE_STEPS", "80")
)  # Maximum steps for nsolve
COARSE_GRID_MIN_SIZE = int(
    os.getenv("KALKULATOR_COARSE_GRID_MIN_SIZE", "12")
)  # Minimum grid size for root search
ROOT_DEDUP_TOLERANCE = float(
    os.getenv("KALKULATOR_ROOT_DEDUP_TOLERANCE", "1e-6")
)  # For deduplicating roots

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
    "abs": sp.Abs,  # lowercase alias for convenience
    # Calculus & algebra
    "diff": sp.diff,
    "integrate": sp.integrate,
    "factor": sp.factor,
    "expand": sp.expand,
    "simplify": sp.simplify,
    # Matrices (basic)
    "Matrix": sp.Matrix,
    "matrix": sp.Matrix,  # lowercase alias for convenience
    "det": sp.det,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PERCENT_REGEX = re.compile(r"(\d+(?:\.\d+)?)%")
SQRT_UNICODE_REGEX = re.compile(r"√\s*\(")
DIGIT_LETTERS_REGEX = re.compile(r"(\d)\s*([A-Za-z(])")
AMBIG_FRACTION_REGEX = re.compile(r"\(([^()]+?)/([^()]+?)\)")
