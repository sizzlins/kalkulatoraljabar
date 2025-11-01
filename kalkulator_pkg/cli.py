from __future__ import annotations

import argparse
import json
import sys
import re
from typing import Any, Dict, Optional, List
import sympy as sp

from .config import VERSION, VAR_NAME_RE
import threading
from .parser import parse_preprocessed, format_solution, format_number, format_superscript, prettify_expr, format_inequality_solution, split_top_level_commas
from .solver import (
	solve_single_equation_cli,
	solve_inequality_cli,
	handle_system_main,
)
from .worker import evaluate_safely


def print_result_pretty(res: Dict[str, Any], json_mode: bool = False) -> None:
	if json_mode:
		print(json.dumps(res, indent=2, ensure_ascii=False))
		return
	if not res.get("ok"):
		print("Error:", res.get("error"))
		return
	typ = res.get("type", "value")
	if typ == "equation":
		print("Exact:", ", ".join(format_solution(x) for x in res.get("exact", [])))
		if res.get("approx"):
			approx_display = ", ".join(format_number(x) for x in res.get("approx", []) if x is not None)
			if approx_display:
				print("Approx:", approx_display)
	elif typ == "multi_isolate":
		sols = res.get("solutions", {})
		approx = res.get("approx", {})
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
			formatted_v = format_inequality_solution(str(v))
			print(f"Solution for {k}: {formatted_v}")
	elif typ == "pell":
		print("Pell parametric solution:\n", res.get("solution"))
	elif typ == "identity_or_contradiction":
		print(res.get("result"))
	elif typ == "value":
		res_str = res.get("result")
		approx = res.get("approx")
		if res_str is None:
			print(res)
			return
		try:
			print(f"{res_str}")
		except Exception:
			print(res_str)
		try:
			parsed = parse_preprocessed(res_str)
			expanded = sp.expand(parsed)
			if str(expanded) != str(parsed):
				print(f"Expanded: {format_solution(expanded)}")
		except Exception:
			pass
		if approx:
			print("Decimal:", approx)
	else:
		print(res)


def repl_loop(json_mode: bool = False) -> None:
	"""Interactive REPL loop with graceful interrupt handling."""
	try:
		import readline  # type: ignore
	except Exception:
		readline = None  # type: ignore
	print("Kalkulator Aljabar — type 'help' for commands, 'quit' to exit.")
	_current_req_id = None  # Track current request for cancellation
	
	def signal_handler(signum, frame):
		"""Handle interrupt signal gracefully."""
		nonlocal _current_req_id
		if _current_req_id:
			from .worker import cancel_current_request
			cancel_current_request(_current_req_id)
			print("\n[Cancelling request...]")
		else:
			print("\n[Press Ctrl+C again to exit]")
	
	# Register signal handler for graceful interrupt (Unix)
	try:
		import signal
		signal.signal(signal.SIGINT, signal_handler)
	except (ImportError, AttributeError):
		# Windows doesn't support signal.SIGINT the same way
		pass
	
	while True:
		try:
			raw = input(">>> ").strip()
			_current_req_id = None  # Clear on new input
		except (EOFError, KeyboardInterrupt):
			print("\nGoodbye.")
			break
		if not raw:
			continue
		if raw.lower() in ("clearcache", "clear cache"):
			from .worker import clear_caches
			clear_caches()
			print("Caches cleared.")
			continue
		if raw.startswith("--eval"):
			parts = raw.split(None, 1)
			if len(parts) == 1:
				print("Usage: --eval <expression>")
				continue
			expr = parts[1].strip()
			if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
				expr = expr[1:-1]
			try:
				if any(op in expr for op in ("<", ">", "<=", ">=")):
					res = solve_inequality_cli(expr, None)
					print_result_pretty(res, json_mode=json_mode)
				elif "=" in expr:
					pts = split_top_level_commas(expr)
					if len(pts) > 1:
						res = handle_system_main(expr, None)
						print_result_pretty(res, json_mode=json_mode)
					else:
						res = solve_single_equation_cli(expr, None)
						print_result_pretty(res, json_mode=json_mode)
				else:
					eva = evaluate_safely(expr)
					if not eva.get("ok"):
						print("Error:", eva.get("error"))
					else:
						res_str = eva.get("result")
						print(f"{expr} = {format_superscript(res_str)}")
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
		cmd = raw.lower()
		if cmd in ("quit", "exit"):
			print("Goodbye.")
			break
		if cmd in ("help", "?", "--help"):
			print_help_text()
			continue
		try:
			if any(op in raw for op in ("<", ">", "<=", ">=")):
				res = solve_inequality_cli(raw, None)
				print_result_pretty(res, json_mode=json_mode)
				continue
			find_tokens = re.findall(r"\bfind\s+(\w+)\b", raw, re.IGNORECASE)
			find = find_tokens[0] if find_tokens else None
			raw_no_find = re.sub(r"\bfind\s+\w+\b", "", raw, flags=re.IGNORECASE).strip()
			parts = split_top_level_commas(raw_no_find)
			if not parts:
				print("No valid parts parsed.")
				continue
			all_assign = all("=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip()) for p in parts)
			all_eq = all("=" in p for p in parts)
			if all_assign and len(parts) > 0:
				if find:
					res = handle_system_main(raw_no_find, find)
					print_result_pretty(res, json_mode=json_mode)
					continue
				subs = {}
				for p in parts:
					left, right = p.split("=", 1)
					var = left.strip()
					rhs = right.strip() or "0"
					res = evaluate_safely(rhs)
					if not res.get("ok"):
						print("Error evaluating assignment RHS:", res.get("error"))
						continue
					try:
						val = parse_preprocessed(res["result"]).subs(subs)
					except Exception as e:
						print("Error assembling assignment value:", e)
						continue
					subs[var] = val
					print(f"{var} = {format_solution(val)}")
				continue
			if len(parts) > 1 and all_eq:
				res = handle_system_main(raw_no_find, find)
				print_result_pretty(res, json_mode=json_mode)
				continue
			elif len(parts) > 1:
				print("Substitution not implemented in REPL (use eval mode) — please use --eval or provide a single expression.")
				continue
			else:
				part = parts[0]
				if "=" in part:
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
		except Exception:
			print("An error occurred. Please check your input and try again.")
			continue


def print_help_text():
	from .config import VERSION
	help_text = f""" version {VERSION}

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

Calculus & matrices (new):
- diff(x^3, x)
- integrate(sin(x), x)
- factor(x^3 - 1)
- expand((x+1)^3)
- Matrix([[1,2],[3,4]])
- det(Matrix([[1,2],[3,4]]))
"""
	print(help_text)


def main_entry(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(prog="algebra_solver_secure")
	parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
	parser.add_argument("--expr", type=str, help=argparse.SUPPRESS)
	parser.add_argument("--worker-solve", action="store_true", help=argparse.SUPPRESS)
	parser.add_argument("--payload", type=str, help=argparse.SUPPRESS)
	parser.add_argument("-e", "--eval", type=str, help="Evaluate one expression and exit (non-interactive)", dest="eval_expr")
	parser.add_argument("-j", "--json", action="store_true", help="Emit JSON for machine parsing")
	parser.add_argument("-t", "--timeout", type=int, help="Override worker timeout (seconds)")
	parser.add_argument("--no-numeric-fallback", action="store_true", help="Disable numeric root-finding fallback")
	parser.add_argument("-p", "--precision", type=int, help="Set output precision (significant digits)")
	parser.add_argument("-v", "--version", action="store_true", help="Show program version")
	parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set logging level")
	parser.add_argument("--log-file", type=str, help="Write logs to file")
	parser.add_argument("--cache-size", type=int, help="Set parse/eval cache size (default: 1024/2048)")
	parser.add_argument("--max-nsolve-guesses", type=int, help="Set maximum nsolve guesses for numeric root finding (default: 36)")
	parser.add_argument("--worker-mode", type=str, choices=["pool", "single", "subprocess"], help="Worker execution mode (default: pool)")
	parser.add_argument("--method", type=str, choices=["auto", "symbolic", "numeric"], help="Solver method (default: auto)")
	args = parser.parse_args(argv)
	
	# Setup logging
	try:
		from .logging_config import setup_logging
		setup_logging(level=args.log_level, log_file=args.log_file)
	except ImportError:
		pass  # Logging optional
	if args.timeout and args.timeout > 0:
		# Apply runtime timeout override
		import kalkulator_pkg.worker as _w
		_w.WORKER_TIMEOUT = int(args.timeout)
	if args.no_numeric_fallback:
		import kalkulator_pkg.config as _c
		_c.NUMERIC_FALLBACK_ENABLED = False
	if args.precision and args.precision > 0:
		import kalkulator_pkg.config as _c
		_c.OUTPUT_PRECISION = int(args.precision)
	if args.cache_size and args.cache_size > 0:
		import kalkulator_pkg.config as _c
		_c.CACHE_SIZE_PARSE = int(args.cache_size)
		_c.CACHE_SIZE_EVAL = int(args.cache_size * 2)
	if args.max_nsolve_guesses and args.max_nsolve_guesses > 0:
		import kalkulator_pkg.config as _c
		_c.MAX_NSOLVE_GUESSES = int(args.max_nsolve_guesses)
	if args.worker_mode:
		import kalkulator_pkg.config as _c
		if args.worker_mode == "subprocess":
			_c.ENABLE_PERSISTENT_WORKER = False
		elif args.worker_mode == "single":
			_c.WORKER_POOL_SIZE = 1
		# "pool" is default
	if args.method:
		import kalkulator_pkg.config as _c
		_c.SOLVER_METHOD = args.method
	if args.worker:
		from .worker import worker_evaluate
		out = worker_evaluate(args.expr or "")
		print(json.dumps(out))
		return 0
	if args.worker_solve:
		from .worker import _worker_solve_dispatch
		try:
			payload = json.loads(args.payload or "{}")
		except Exception:
			payload = {}
		print(json.dumps(_worker_solve_dispatch(payload)))
		return 0
	if args.version:
		print(VERSION)
		return 0
	if args.eval_expr:
		expr = args.eval_expr.strip()
		import re
		find_tokens = re.findall(r"\bfind\s+(\w+)\b", expr, re.IGNORECASE)
		find = find_tokens[0] if find_tokens else None
		raw_no_find = re.sub(r"\bfind\s+\w+\b", "", expr, flags=re.IGNORECASE).strip()
		if any(op in raw_no_find for op in ("<", ">", "<=", ">=")):
			res = solve_inequality_cli(raw_no_find, find)
		elif "=" in raw_no_find:
			parts = split_top_level_commas(raw_no_find)
			if len(parts) > 1:
				res = handle_system_main(raw_no_find, find)
			else:
				res = solve_single_equation_cli(parts[0], find)
		else:
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
	repl_loop(json_mode=args.json)
	return 0
