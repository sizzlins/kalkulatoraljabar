from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import re
import json

import sympy as sp

from .parser import (
	parse_preprocessed,
	preprocess,
	format_solution,
	format_number,
	format_superscript,
	prettify_expr,
	format_inequality_solution,
)
from .config import VAR_NAME_RE, NUMERIC_FALLBACK_ENABLED
from .worker import evaluate_safely, _worker_solve_cached


@dataclass
class EvalResult:
	ok: bool
	result: Optional[str] = None
	approx: Optional[str] = None
	free_symbols: Optional[List[str]] = None
	error: Optional[str] = None


def eval_user_expression(expr: str) -> EvalResult:
	data = evaluate_safely(expr)
	if not data.get("ok"):
		return EvalResult(ok=False, error=data.get("error") or "Unknown error")
	return EvalResult(ok=True, result=data.get("result"), approx=data.get("approx"),
					  free_symbols=data.get("free_symbols"))


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


ZERO_TOL = 1e-12


def solve_single_equation_cli(eq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
	parts = eq_str.split("=", 1)
	if len(parts) != 2:
		return {"ok": False, "error": "Invalid equation format (need exactly one '=')."}
	lhs_s, rhs_s = parts[0].strip(), parts[1].strip() or "0"
	lhs = evaluate_safely(lhs_s)
	if not lhs.get("ok"):
		return {"ok": False, "error": f"LHS parse error: {lhs.get('error')}"}
	rhs = evaluate_safely(rhs_s)
	if not rhs.get("ok"):
		return {"ok": False, "error": f"RHS parse error: {rhs.get('error')}"}
	try:
		left_expr = parse_preprocessed(lhs["result"])
		right_expr = parse_preprocessed(rhs["result"])
	except Exception as e:
		return {"ok": False, "error": f"Internal parse error assembling SymPy expressions: {e}"}
	equation = sp.Eq(left_expr, right_expr)
	if is_pell_equation_from_eq(equation):
		try:
			pell_str = solve_pell_equation_from_eq(equation)
			return {"ok": True, "type": "pell", "solution": prettify_expr(pell_str)}
		except Exception as e:
			return {"ok": False, "error": f"Pell solver error: {e}"}
	symbols = list(equation.free_symbols)
	if not symbols:
		try:
			simp = sp.simplify(left_expr - right_expr)
			if simp == 0:
				return {"ok": True, "type": "identity_or_contradiction", "result": "Identity"}
		except Exception:
			simp = None
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

	def _numeric_roots_for_single_var(f_expr, sym, interval=(-4 * sp.pi, 4 * sp.pi), guesses=36):
		roots = []
		# First, try solveset over the reals within interval, which can be faster and more robust
		try:
			from sympy import solveset, S
			solset = solveset(sp.Eq(f_expr, 0), sym, domain=sp.Interval(float(interval[0]), float(interval[1])))
			finite = []
			for s in solset:
				try:
					s_val = float(sp.N(s))
					finite.append(s_val)
				except Exception:
					continue
			if finite:
				uniq = sorted(set(round(x, 12) for x in finite))
				return [sp.N(r) for r in uniq]
		except Exception:
			pass
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
		# Detect sign changes on a coarse grid to pick good initial guesses (reduce nsolve calls)
		coarse = max(12, guesses // 3)
		samples = [a + (b - a) * i / coarse for i in range(coarse + 1)]
		candidates = []
		prev_val = None
		for x in samples:
			try:
				val = float(sp.N(f_expr.subs({sym: x})))
				if prev_val is not None and val == val and prev_val == prev_val and prev_val * val <= 0:
					candidates.append(x)
				prev_val = val
			except Exception:
				prev_val = None
		# De-dup and limit candidates
		candidates = sorted(set(round(c, 8) for c in candidates))[:12]
		for g in candidates:
			try:
				r = sp.nsolve(f_expr, sym, g, tol=1e-12, maxsteps=80)
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
		if len(symbols) == 1:
			sym = symbols[0]
			try:
				sols = sp.solve(equation, sym)
			except Exception as e:
				if NUMERIC_FALLBACK_ENABLED and equation.has(sp.sin, sp.cos, sp.tan):
					f = sp.simplify(left_expr - right_expr)
					numeric_roots = _numeric_roots_for_single_var(f, sym, interval=(-4 * sp.pi, 4 * sp.pi), guesses=36)
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


def _parse_relational_fallback(rel_str: str):
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
			raise ValueError(f"Failed to parse component '{p_strip}': {res.get('error')}")
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


def solve_inequality_cli(ineq_str: str, find_var: Optional[str] = None) -> Dict[str, Any]:
	try:
		parsed = _parse_relational_fallback(ineq_str)
	except Exception as e:
		return {"ok": False, "error": f"Failed to parse inequality: {e}"}
	free_syms = list(parsed.free_symbols) if hasattr(parsed, "free_symbols") else []
	if find_var:
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
			try:
				is_true = sp.simplify(parsed)
				return {"ok": True, "type": "inequality", "solutions": {"result": str(is_true)}}
			except Exception:
				return {"ok": False, "error": "No variable found in inequality"}
		vars_to_solve = free_syms
	results = {}
	for v in vars_to_solve:
		try:
			ineqs_to_solve = list(parsed.args) if isinstance(parsed, sp.And) else [parsed]
			sol = sp.reduce_inequalities(ineqs_to_solve, v)
			results[str(v)] = str(sol)
		except NotImplementedError:
			results[str(v)] = "Solver not implemented for this type of inequality."
		except Exception as e:
			results[str(v)] = f"Error solving for {v}: {e}"
	return {"ok": True, "type": "inequality", "solutions": results}


def handle_system_main(raw_no_find: str, find_token: Optional[str]) -> Dict[str, Any]:
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
			return {"ok": False, "error": f"LHS parse error: {lhs_eval.get('error')}"}
		rhs_eval = evaluate_safely(rhs_s)
		if not rhs_eval.get("ok"):
			return {"ok": False, "error": f"RHS parse error: {rhs_eval.get('error')}"}
		if VAR_NAME_RE.match(lhs_s):
			assignments[lhs_s] = {"result": rhs_eval.get("result"), "approx": rhs_eval.get("approx")}
		eqs_serialized.append({"lhs": lhs_eval.get("result"), "rhs": rhs_eval.get("result")})
	if not eqs_serialized:
		return {"ok": False, "error": "No equations parsed."}
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
			except Exception:
				expr_sym = None
			if expr_sym is not None:
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
						try:
							approx_obj = sp.N(value)
							if abs(sp.re(approx_obj)) < ZERO_TOL and abs(sp.im(approx_obj)) < ZERO_TOL:
								return {"ok": True, "type": "system_var", "exact": ["0"], "approx": ["0"]}
							approx_val = str(approx_obj)
						except Exception:
							approx_val = None
						return {"ok": True, "type": "system_var", "exact": [str(value)], "approx": [approx_val]}
					except Exception:
						pass
	payload = {"equations": eqs_serialized, "find": find_token}
	try:
		stdout_text = _worker_solve_cached(json.dumps(payload))
	except Exception:
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
