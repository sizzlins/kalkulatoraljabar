import sympy as sp
from sympy.parsing.sympy_parser import (
	standard_transformations,
	implicit_multiplication_application,
	convert_xor,
)
import re

VERSION = "2025-10-31"
WORKER_CPU_SECONDS = 30
WORKER_AS_MB = 400
WORKER_TIMEOUT = 60
ENABLE_PERSISTENT_WORKER = True
WORKER_POOL_SIZE = 1  # future: support >1 workers
NUMERIC_FALLBACK_ENABLED = True
OUTPUT_PRECISION = 6

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
