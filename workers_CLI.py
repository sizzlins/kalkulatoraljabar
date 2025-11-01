# workers_CLI.py
import argparse
import json
import sys
from typing import List

import sympy as sp

from helpers import parse_preprocessed
from workers import HAS_RESOURCE, _limit_resources, worker_evaluate


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
