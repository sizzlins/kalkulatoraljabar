from __future__ import annotations

from functools import lru_cache
from typing import Any, List
import re
import sympy as sp
from sympy import parse_expr

from .config import (
	ALLOWED_SYMPY_NAMES,
	TRANSFORMATIONS,
	PERCENT_REGEX,
	SQRT_UNICODE_REGEX,
	DIGIT_LETTERS_REGEX,
	AMBIG_FRACTION_REGEX,
    OUTPUT_PRECISION,
)


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


def format_number(val: Any, precision: int = OUTPUT_PRECISION) -> str:
	try:
		fmt = "{:." + str(int(precision)) + "g}"
		return fmt.format(float(val))
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


# Basic denylist to avoid dangerous tokens before SymPy parsing
FORBIDDEN_TOKENS = (
	"__", "import", "lambda", "eval", "exec", "open", "os.", "sys.",
	"subprocess", "builtins", "getattr", "setattr", "delattr", "compile", "globals",
	"locals", "__class__", "__mro__", "__subclasses__", "memoryview", "bytes", "bytearray",
	"__import__",
)


def preprocess(input_str: str, skip_exponent_conversion: bool = False) -> str:
	lowered = input_str.strip().lower()
	for tok in FORBIDDEN_TOKENS:
		if tok in lowered:
			raise ValueError("Input contains forbidden token.")

	s = input_str.strip()
	s = s.replace("−", "-").replace("–", "-")
	s = s.replace("Δ", "Delta")
	s = s.replace(":", "/")
	s = s.replace("×", "*")
	# Standardize inequality variations
	s = s.replace("=>", ">=")
	s = s.replace("=<", "<=")

	if not skip_exponent_conversion:
		s = s.replace("^", "**")
		from_superscript_map = {
			"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
			"⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
			"⁻": "-", "ⁿ": "n",
		}
		pattern = re.compile(f"([{''.join(from_superscript_map.keys())}]+)")

		def from_superscript_converter(m):
			normal_str = "".join([from_superscript_map[char] for char in m.group(1)])
			return f"**{normal_str}"

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
	return parse_expr(expr_str, local_dict=ALLOWED_SYMPY_NAMES,
				      transformations=TRANSFORMATIONS, evaluate=True)


def format_inequality_solution(sol_str: str) -> str:
	pattern = re.compile(
		r"\((.*?)\s*([<>=!]+)\s*(.*?)\)\s*&\s*\((.*?)\s*([<>=!]+)\s*(.*?)\)"
	)
	match = pattern.match(sol_str)
	if not match:
		return sol_str
	g = [s.strip() for s in match.groups()]
	expr1, op1, var1, expr2, op2, var2 = g
	if var1 == expr2:
		if op1 in ("<", "<=") and op2 in ("<", "<="):
			return f"{expr1} {op1} {var1} {op2} {var2}"
	elif expr1 == var2:
		op_map = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
		if op1 in op_map and op2 in op_map:
			return f"{var1} {op_map[op1]} {expr1} {op_map[op2]} {var2}"
	return sol_str


def split_top_level_commas(s: str) -> List[str]:
	"""Split string by commas that are not inside (), [], or {}."""
	parts: List[str] = []
	current = []
	depth_paren = depth_brack = depth_brace = 0
	for ch in s:
		if ch == ',' and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
			part = ''.join(current).strip()
			if part:
				parts.append(part)
			current = []
			continue
		if ch == '(':
			depth_paren += 1
		elif ch == ')':
			depth_paren = max(0, depth_paren - 1)
		elif ch == '[':
			depth_brack += 1
		elif ch == ']':
			depth_brack = max(0, depth_brack - 1)
		elif ch == '{':
			depth_brace += 1
		elif ch == '}':
			depth_brace = max(0, depth_brace - 1)
		current.append(ch)
	# append last segment
	last = ''.join(current).strip()
	if last:
		parts.append(last)
	return parts
