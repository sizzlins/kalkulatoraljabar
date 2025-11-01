from __future__ import annotations

import json
import os
import sys
import subprocess
from functools import lru_cache
from typing import Any, Dict, Optional, List

import sympy as sp

from .config import (
	WORKER_CPU_SECONDS, WORKER_AS_MB, WORKER_TIMEOUT, ENABLE_PERSISTENT_WORKER, WORKER_POOL_SIZE,
	CACHE_SIZE_EVAL, CACHE_SIZE_SOLVE,
)
from .parser import parse_preprocessed
from .types import ValidationError, ParseError

try:
	from .logging_config import get_logger
	logger = get_logger("worker")
except ImportError:
	# Fallback if logging not available
	class NullLogger:
		def debug(self, *args, **kwargs): pass
		def info(self, *args, **kwargs): pass
		def warning(self, *args, **kwargs): pass
		def error(self, *args, **kwargs): pass
		def exception(self, *args, **kwargs): pass
	logger = NullLogger()

HAS_RESOURCE = False
try:
	import resource  # type: ignore
	HAS_RESOURCE = True
except Exception:
	HAS_RESOURCE = False

try:
	from multiprocessing import Process, Queue, Event, Manager
except Exception:
	Process = None  # type: ignore
	Queue = None  # type: ignore
	Event = None  # type: ignore
	Manager = None  # type: ignore


def _limit_resources():
	if not HAS_RESOURCE:
		return
	import resource as _resource
	_resource.setrlimit(_resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS + 1))
	_resource.setrlimit(_resource.RLIMIT_AS, (WORKER_AS_MB * 1024 * 1024, WORKER_AS_MB * 1024 * 1024 + 1))


def worker_evaluate(preprocessed_expr: str) -> Dict[str, Any]:
	"""Evaluate a preprocessed expression in a sandboxed worker."""
	logger.debug(f"Evaluating expression: {preprocessed_expr[:100]}...")
	if HAS_RESOURCE:
		try:
			_limit_resources()
			logger.debug("Resource limits applied")
		except (OSError, ValueError) as e:
			logger.warning(f"Failed to apply resource limits: {e}")
	try:
		expr = parse_preprocessed(preprocessed_expr)
	except ValidationError as e:
		logger.warning(f"Validation error: {e.code} - {e.message}")
		return {"ok": False, "error": str(e), "error_code": e.code}
	except (ValueError, SyntaxError) as e:
		logger.warning(f"Parse error: {e}")
		return {"ok": False, "error": f"Parse error: {e}", "error_code": "PARSE_ERROR"}
	except Exception as e:
		logger.exception("Unexpected parse error in worker")
		return {"ok": False, "error": "Parse error in worker", "error_code": "UNKNOWN_ERROR"}
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
		except (ValueError, TypeError, ArithmeticError):
			approx = None
		return {"ok": True, "result": result_str, "approx": approx, "free_symbols": free_syms}
	except (ValueError, TypeError, ArithmeticError) as e:
		return {"ok": False, "error": f"Evaluation failed: {e}", "error_code": "EVAL_ERROR"}
	except Exception as e:
		return {"ok": False, "error": "Evaluation failed", "error_code": "UNKNOWN_ERROR"}


def _build_self_cmd(args: List[str]) -> List[str]:
	if getattr(sys, "frozen", False):
		return [os.path.realpath(sys.argv[0])] + args
	else:
		return [sys.executable, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "kalkulator.py"))] + args


import uuid
import time


def _retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 0.1, max_delay: float = 2.0) -> Any:
	"""Retry a function with exponential backoff.
	
	Args:
		func: Callable that returns a result or raises an exception
		max_retries: Maximum number of retry attempts
		initial_delay: Initial delay in seconds
		max_delay: Maximum delay cap in seconds
		
	Returns:
		Result from func() if successful
		
	Raises:
		Last exception if all retries fail
	"""
	delay = initial_delay
	last_exception = None
	
	for attempt in range(max_retries + 1):
		try:
			return func()
		except Exception as e:
			last_exception = e
			error_code = getattr(e, 'code', None)
			# Check if error is transient (retryable)
			transient_codes = {"COMM_ERROR", "TIMEOUT", "UNKNOWN_ERROR"}
			if error_code not in transient_codes and attempt < max_retries:
				# Fatal error, don't retry
				raise
			if attempt < max_retries:
				logger.debug(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s")
				time.sleep(delay)
				delay = min(delay * 2, max_delay)
	
	# All retries exhausted
	raise last_exception


class _WorkerManager:
	def __init__(self) -> None:
		self.procs = []
		self.req_qs = []
		self.res_q = None
		self.stop_event = None
		self._next_idx = 0
		self._resp_buffer = {}
		self._manager = None
		self._cancel_flags: Optional[Dict[str, bool]] = None  # req_id -> cancel flag (shared dict)

	def start(self) -> None:
		if not ENABLE_PERSISTENT_WORKER or Process is None or Manager is None:
			return
		if self.is_alive():
			return
		# Create Manager for shared state (works on Windows)
		if self._manager is None:
			self._manager = Manager()
			self._cancel_flags = self._manager.dict()
		self.res_q = Queue()
		self.stop_event = Event()
		self.req_qs = []
		self.procs = []
		n = max(1, int(WORKER_POOL_SIZE or 1))
		for _ in range(n):
			req = Queue()
			proc = Process(target=_worker_daemon_main, args=(req, self.res_q, self.stop_event, self._cancel_flags), daemon=True)
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
			if self._cancel_flags is not None:
				self._cancel_flags.clear()

	def cancel_request(self, req_id: str) -> bool:
		"""Cancel a pending request by ID. Returns True if cancellation flag was found."""
		if self._cancel_flags is not None and req_id in self._cancel_flags:
			self._cancel_flags[req_id] = True
			return True
		return False

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
			# Initialize cancellation flag in shared dict
			if self._cancel_flags is not None:
				self._cancel_flags[req_id] = False
			
			if not self.req_qs:
				if self._cancel_flags and req_id in self._cancel_flags:
					del self._cancel_flags[req_id]
				return None
			idx = self._next_idx % len(self.req_qs)
			self._next_idx += 1
			self.req_qs[idx].put(payload)
			# First, see if we already buffered this id
			if req_id in self._resp_buffer:
				resp = self._resp_buffer.pop(req_id)
				if self._cancel_flags and req_id in self._cancel_flags:
					del self._cancel_flags[req_id]
				return resp
			# Otherwise, read from res_q until matching id
			while True:
				if self._cancel_flags and self._cancel_flags.get(req_id, False):
					if req_id in self._cancel_flags:
						del self._cancel_flags[req_id]
					return {"ok": False, "error": "Request cancelled", "error_code": "CANCELLED"}
				try:
					msg = self.res_q.get(timeout=min(0.5, timeout))
				except Exception:
					# Timeout - check cancellation
					if self._cancel_flags and self._cancel_flags.get(req_id, False):
						if req_id in self._cancel_flags:
							del self._cancel_flags[req_id]
						return {"ok": False, "error": "Request cancelled", "error_code": "CANCELLED"}
					continue
				mid = msg.get("id")
				if mid == req_id:
					msg.pop("id", None)
					if self._cancel_flags and req_id in self._cancel_flags:
						del self._cancel_flags[req_id]
					return msg
				# buffer for future requests
				self._resp_buffer[mid] = msg
		except Exception:
			req_id = payload.get("id")
			if req_id and self._cancel_flags and req_id in self._cancel_flags:
				del self._cancel_flags[req_id]
			try:
				self.stop()
				self.start()
				if self.is_alive():
					req_id = payload.get("id") or str(uuid.uuid4())
					payload = {**payload, "id": req_id}
					if self._cancel_flags is not None:
						self._cancel_flags[req_id] = False
					idx = self._next_idx % len(self.req_qs)
					self._next_idx += 1
					self.req_qs[idx].put(payload)
					while True:
						if self._cancel_flags and self._cancel_flags.get(req_id, False):
							if req_id in self._cancel_flags:
								del self._cancel_flags[req_id]
							return {"ok": False, "error": "Request cancelled", "error_code": "CANCELLED"}
						try:
							msg = self.res_q.get(timeout=min(0.5, timeout))
						except Exception:
							continue
						mid = msg.get("id")
						if mid == req_id:
							msg.pop("id", None)
							if self._cancel_flags and req_id in self._cancel_flags:
								del self._cancel_flags[req_id]
							return msg
						self._resp_buffer[mid] = msg
			except Exception:
				self.stop()
			return None


def _worker_daemon_main(req_q, res_q, stop_event, cancel_flags) -> None:
	"""Worker daemon main loop that processes requests from queue."""
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
			req_id = msg.get("id")
			
			# Check cancellation before processing
			if cancel_flags and cancel_flags.get(req_id, False):
				res_q.put({"ok": False, "error": "Request cancelled", "error_code": "CANCELLED", "id": req_id})
				continue
			
			if kind == "eval":
				pre = msg.get("preprocessed") or ""
				out = worker_evaluate(pre)
				out["id"] = req_id
				# Check cancellation after processing
				if cancel_flags and cancel_flags.get(req_id, False):
					res_q.put({"ok": False, "error": "Request cancelled", "error_code": "CANCELLED", "id": req_id})
				else:
					res_q.put(out)
			elif kind == "solve":
				payload = msg.get("payload") or {}
				out = _worker_solve_dispatch(payload)
				out["id"] = req_id
				# Check cancellation after processing
				if cancel_flags and cancel_flags.get(req_id, False):
					res_q.put({"ok": False, "error": "Request cancelled", "error_code": "CANCELLED", "id": req_id})
				else:
					res_q.put(out)
			else:
				res_q.put({"ok": False, "error": "Unknown request type", "id": req_id})
		except Exception as e:
			res_q.put({"ok": False, "error": f"Worker daemon error: {e}", "id": msg.get("id")})


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


@lru_cache(maxsize=CACHE_SIZE_EVAL)
def _worker_eval_cached(preprocessed_expr: str) -> str:
	resp = _WORKER_MANAGER.request({"type": "eval", "preprocessed": preprocessed_expr}, timeout=WORKER_TIMEOUT)
	if isinstance(resp, dict):
		return json.dumps(resp)
	cmd = _build_self_cmd(["--worker", "--expr", preprocessed_expr])
	proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT)
	return proc.stdout or ""


@lru_cache(maxsize=CACHE_SIZE_SOLVE)
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
	"""Safely evaluate an expression string via worker sandbox."""
	from .parser import preprocess
	try:
		pre = preprocess(expr)
	except ValidationError as e:
		return {"ok": False, "error": str(e), "error_code": e.code}
	except ValueError as e:
		return {"ok": False, "error": f"Preprocess error: {e}", "error_code": "PREPROCESS_ERROR"}
	except Exception as e:
		return {"ok": False, "error": "Preprocess error", "error_code": "UNKNOWN_ERROR"}
	try:
		stdout_text = _worker_eval_cached(pre)
	except subprocess.TimeoutExpired:
		return {"ok": False, "error": "Evaluation timed out.", "error_code": "TIMEOUT"}
	except Exception as e:
		return {"ok": False, "error": "Worker communication failed", "error_code": "COMM_ERROR"}
	try:
		data = json.loads(stdout_text)
		return data
	except (json.JSONDecodeError, ValueError) as e:
		return {"ok": False, "error": f"Invalid worker output: {e}.", "error_code": "INVALID_OUTPUT"}
	except Exception as e:
		return {"ok": False, "error": "Invalid worker output", "error_code": "UNKNOWN_ERROR"}


def clear_caches() -> None:
	"""Clear worker-side LRU caches and parser cache."""
	try:
		_worker_eval_cached.cache_clear()
		_worker_solve_cached.cache_clear()
		from .parser import parse_preprocessed as _pp
		_pp.cache_clear()
	except Exception:
		pass


def cancel_current_request(req_id: Optional[str] = None) -> bool:
	"""Cancel a pending worker request. If req_id is None, attempts to cancel the most recent."""
	if req_id:
		return _WORKER_MANAGER.cancel_request(req_id)
	return False  # For now, requires explicit ID - can be enhanced
