from __future__ import annotations

import json
import os
import sys
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional, List

import sympy as sp

from .config import WORKER_CPU_SECONDS, WORKER_AS_MB, WORKER_TIMEOUT, ENABLE_PERSISTENT_WORKER, WORKER_POOL_SIZE
from .parser import parse_preprocessed

HAS_RESOURCE = False
try:
	import resource  # type: ignore
	HAS_RESOURCE = True
except Exception:
	HAS_RESOURCE = False

try:
	from multiprocessing import Process, Queue, Event
except Exception:
	Process = None  # type: ignore
	Queue = None  # type: ignore
	Event = None  # type: ignore


def _limit_resources():
	if not HAS_RESOURCE:
		return
	import resource as _resource
	_resource.setrlimit(_resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS + 1))
	_resource.setrlimit(_resource.RLIMIT_AS, (WORKER_AS_MB * 1024 * 1024, WORKER_AS_MB * 1024 * 1024 + 1))


def worker_evaluate(preprocessed_expr: str) -> Dict[str, Any]:
	if HAS_RESOURCE:
		try:
			_limit_resources()
		except Exception:
			pass
	try:
		expr = parse_preprocessed(preprocessed_expr)
	except Exception as e:
		return {"ok": False, "error": f"Parse error in worker: {e}"}
	try:
		res = sp.simplify(expr)
		result_str = str(res)
		free_syms = [str(s) for s in getattr(res, "free_symbols", set())]
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


def _build_self_cmd(args: List[str]) -> List[str]:
	if getattr(sys, "frozen", False):
		return [os.path.realpath(sys.argv[0])] + args
	else:
		return [sys.executable, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "kalkulator.py"))] + args


import uuid


class _WorkerManager:
	def __init__(self) -> None:
		self.procs = []
		self.req_qs = []
		self.res_q = None
		self.stop_event = None
		self._next_idx = 0
		self._resp_buffer = {}

	def start(self) -> None:
		if not ENABLE_PERSISTENT_WORKER or Process is None:
			return
		if self.is_alive():
			return
		self.res_q = Queue()
		self.stop_event = Event()
		self.req_qs = []
		self.procs = []
		n = max(1, int(WORKER_POOL_SIZE or 1))
		for _ in range(n):
			req = Queue()
			proc = Process(target=_worker_daemon_main, args=(req, self.res_q, self.stop_event), daemon=True)
			proc.start()
			self.req_qs.append(req)
			self.procs.append(proc)

	def is_alive(self) -> bool:
		return bool(self.procs and all(p.is_alive() for p in self.procs))

	def stop(self) -> None:
		try:
			if self.stop_event is not None:
				self.stop_event.set()
			for p in self.procs or []:
				p.join(timeout=1.0)
		except Exception:
			pass
		finally:
			self.procs = []
			self.req_qs = []
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
			# Correlate with an ID and route via round-robin to workers
			req_id = payload.get("id") or str(uuid.uuid4())
			payload = {**payload, "id": req_id}
			if not self.req_qs:
				return None
			idx = self._next_idx % len(self.req_qs)
			self._next_idx += 1
			self.req_qs[idx].put(payload)
			# First, see if we already buffered this id
			if req_id in self._resp_buffer:
				resp = self._resp_buffer.pop(req_id)
				return resp
			# Otherwise, read from res_q until matching id
			while True:
				msg = self.res_q.get(timeout=timeout)
				mid = msg.get("id")
				if mid == req_id:
					msg.pop("id", None)
					return msg
				# buffer for future requests
				self._resp_buffer[mid] = msg
		except Exception:
			try:
				self.stop()
				self.start()
				if self.is_alive():
					req_id = payload.get("id") or str(uuid.uuid4())
					payload = {**payload, "id": req_id}
					idx = self._next_idx % len(self.req_qs)
					self._next_idx += 1
					self.req_qs[idx].put(payload)
					while True:
						msg = self.res_q.get(timeout=timeout)
						mid = msg.get("id")
						if mid == req_id:
							msg.pop("id", None)
							return msg
						self._resp_buffer[mid] = msg
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
				out["id"] = msg.get("id")
				res_q.put(out)
			elif kind == "solve":
				payload = msg.get("payload") or {}
				out = _worker_solve_dispatch(payload)
				out["id"] = msg.get("id")
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
	resp = _WORKER_MANAGER.request({"type": "eval", "preprocessed": preprocessed_expr}, timeout=WORKER_TIMEOUT)
	if isinstance(resp, dict):
		return json.dumps(resp)
	cmd = _build_self_cmd(["--worker", "--expr", preprocessed_expr])
	proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT)
	return proc.stdout or ""


@lru_cache(maxsize=256)
def _worker_solve_cached(payload_json: str) -> str:
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


def evaluate_safely(expr: str, timeout: int = WORKER_TIMEOUT) -> Dict[str, Any]:
	from .parser import preprocess
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


def clear_caches() -> None:
	"""Clear worker-side LRU caches and parser cache."""
	try:
		_worker_eval_cached.cache_clear()
		_worker_solve_cached.cache_clear()
		from .parser import parse_preprocessed as _pp
		_pp.cache_clear()
	except Exception:
		pass
