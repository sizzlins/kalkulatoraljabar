from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional

import sympy as sp

from .config import (
    CACHE_SIZE_EVAL,
    CACHE_SIZE_SOLVE,
    ENABLE_PERSISTENT_WORKER,
    WORKER_AS_MB,
    WORKER_CPU_SECONDS,
    WORKER_POOL_SIZE,
    WORKER_TIMEOUT,
)
from .parser import parse_preprocessed
from .types import ValidationError

try:
    from .logging_config import get_logger

    logger = get_logger("worker")
except ImportError:
    # Fallback if logging not available
    class NullLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    logger = NullLogger()

HAS_RESOURCE = False
try:
    import resource  # noqa: F401 - check if available
    HAS_RESOURCE = True
except (ImportError, OSError):
    HAS_RESOURCE = False

try:
    from multiprocessing import Event, Manager, Process, Queue
except Exception:
    Process = None  # type: ignore
    Queue = None  # type: ignore
    Event = None  # type: ignore
    Manager = None  # type: ignore


def _limit_resources() -> None:
    """Apply resource limits (Unix only).

    This function sets CPU time and memory limits on worker processes.
    On Windows, the `resource` module is not available, so these limits
    do not apply. For Windows deployments, consider:
    - Using containerization (Docker with resource limits)
    - Running worker processes under restricted user accounts
    - Monitoring resource usage externally

    See SECURITY.md for more details on Windows deployment strategies.
    """
    if not HAS_RESOURCE:
        return
    try:
        import resource as _resource
        _resource.setrlimit(
            _resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS + 1)
        )
        _resource.setrlimit(
            _resource.RLIMIT_AS,
            (WORKER_AS_MB * 1024 * 1024, WORKER_AS_MB * 1024 * 1024 + 1),
        )
    except (ImportError, OSError, ValueError):
        # Resource module not available (Windows) or limits failed
        pass


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
        error_msg = str(e)
        # Provide helpful hints for common errors
        if "cannot assign" in error_msg.lower() or "==" in error_msg.lower():
            error_msg += " Hint: If you're writing an equation, use '==' instead of '='. For assignments in REPL, use separate statements like 'a = expression' followed by 'equation = 0'."
        return {"ok": False, "error": f"Parse error: {error_msg}", "error_code": "PARSE_ERROR"}
    except Exception as e:
        # Check for specific error types that need better messages
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle TokenError (unterminated string literal, etc.)
        # TokenError comes from tokenize module when parsing Python-like syntax
        # Check if it's a tokenize.TokenError
        is_token_error = False
        try:
            import tokenize
            is_token_error = isinstance(e, tokenize.TokenError)
        except (ImportError, AttributeError):
            # Fallback: check by name
            is_token_error = error_type == "TokenError"
        
        if is_token_error:
            if "unterminated string literal" in error_msg.lower() or "unterminated" in error_msg.lower():
                return {
                    "ok": False,
                    "error": "Syntax error: Unmatched or unterminated string literal. Check that all quotes are properly closed and matched.",
                    "error_code": "SYNTAX_ERROR",
                }
            return {
                "ok": False,
                "error": f"Syntax error: {error_msg}",
                "error_code": "SYNTAX_ERROR",
            }
        
        # Log full traceback for unexpected errors but return clean message
        logger.exception("Unexpected parse error in worker")
        
        # Try to extract useful information from the error
        if "assign" in error_msg.lower() and "==" in error_msg.lower():
            return {
                "ok": False,
                "error": f"Parse error: {error_msg}. Hint: Use '==' for equations, '=' for assignments. Mixed assignments and equations must be separated (e.g., 'a = expression' on one line, then solve the equation separately).",
                "error_code": "PARSE_ERROR",
            }
        
        return {
            "ok": False,
            "error": f"Parse error: {error_msg}. Please check your syntax.",
            "error_code": "UNKNOWN_ERROR",
        }
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
        return {
            "ok": True,
            "result": result_str,
            "approx": approx,
            "free_symbols": free_syms,
        }
    except (ValueError, TypeError, ArithmeticError) as e:
        return {
            "ok": False,
            "error": f"Evaluation failed: {e}",
            "error_code": "EVAL_ERROR",
        }
    except Exception as e:
        # Catch-all for truly unexpected errors - log full traceback
        logger.exception("Unexpected evaluation error in worker")
        return {
            "ok": False,
            "error": f"Evaluation failed: {e}",
            "error_code": "UNKNOWN_ERROR",
        }


def _build_self_cmd(args: List[str]) -> List[str]:
    if getattr(sys, "frozen", False):
        return [os.path.realpath(sys.argv[0])] + args
    else:
        return [
            sys.executable,
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", "kalkulator.py")
            ),
        ] + args


import time
import uuid


def _retry_with_backoff(
    func, max_retries: int = 3, initial_delay: float = 0.1, max_delay: float = 2.0
) -> Any:
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
            error_code = getattr(e, "code", None)
            # Check if error is transient (retryable)
            transient_codes = {"COMM_ERROR", "TIMEOUT", "UNKNOWN_ERROR"}
            if error_code not in transient_codes and attempt < max_retries:
                # Fatal error, don't retry
                raise
            if attempt < max_retries:
                logger.debug(
                    f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s"
                )
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
        self._cancel_flags: Optional[Dict[str, bool]] = (
            None  # req_id -> cancel flag (shared dict)
        )

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
            proc = Process(
                target=_worker_daemon_main,
                args=(req, self.res_q, self.stop_event, self._cancel_flags),
                daemon=True,
            )
            proc.start()
            self.req_qs.append(req)
            self.procs.append(proc)

    def is_alive(self) -> bool:
        return bool(self.procs and all(p.is_alive() for p in self.procs))

    def stop(self) -> None:
        """Stop all worker processes gracefully."""
        try:
            if self.stop_event is not None:
                self.stop_event.set()
            for p in self.procs or []:
                try:
                    p.join(timeout=1.0)
                except (OSError, ValueError, AttributeError) as e:
                    # Process already dead or invalid - log but continue cleanup
                    try:
                        from .logging_config import safe_log
                        safe_log("worker", "warning", f"Error joining worker process: {e}")
                    except ImportError:
                        pass
        except (AttributeError, TypeError) as e:
            # Invalid state - log but continue cleanup
            try:
                from .logging_config import safe_log
                safe_log("worker", "warning", f"Error stopping workers: {e}")
            except ImportError:
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

    def request(
        self, payload: Dict[str, Any], timeout: int
    ) -> Optional[Dict[str, Any]]:
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
                    return {
                        "ok": False,
                        "error": "Request cancelled",
                        "error_code": "CANCELLED",
                    }
                try:
                    msg = self.res_q.get(timeout=min(0.5, timeout))
                except (KeyboardInterrupt, SystemExit):
                    # Stop workers and propagate interrupt to main process
                    try:
                        self.stop()
                    except Exception:
                        pass
                    raise
                except Exception as e:
                    # Queue timeout or other error - check cancellation
                    # Expected: queue.Empty on timeout, which is caught by generic Exception
                    # This is acceptable as queue.Empty is not available in all Python versions
                    if self._cancel_flags and self._cancel_flags.get(req_id, False):
                        if req_id in self._cancel_flags:
                            del self._cancel_flags[req_id]
                        return {
                            "ok": False,
                            "error": "Request cancelled",
                            "error_code": "CANCELLED",
                        }
                    # Continue waiting on timeout (expected behavior)
                    continue
                mid = msg.get("id")
                if mid == req_id:
                    msg.pop("id", None)
                    if self._cancel_flags and req_id in self._cancel_flags:
                        del self._cancel_flags[req_id]
                    return msg
                # buffer for future requests
                self._resp_buffer[mid] = msg
        except (KeyboardInterrupt, SystemExit):
            # Stop workers and propagate interrupt to main process
            try:
                self.stop()
            except Exception:
                pass
            raise
        except (AttributeError, TypeError, ValueError, OSError) as e:
            # Specific exceptions that can occur in worker communication
            req_id = payload.get("id")
            if req_id and self._cancel_flags and req_id in self._cancel_flags:
                del self._cancel_flags[req_id]
            try:
                from .logging_config import safe_log
                safe_log("worker", "warning", f"Worker communication error, restarting: {e}", exc_info=True)
            except ImportError:
                pass
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
                            return {
                                "ok": False,
                                "error": "Request cancelled",
                                "error_code": "CANCELLED",
                            }
                        try:
                            msg = self.res_q.get(timeout=min(0.5, timeout))
                        except (KeyboardInterrupt, SystemExit):
                            # Stop workers and propagate interrupt
                            try:
                                self.stop()
                            except Exception:
                                pass
                            raise
                        except Exception:
                            # Queue timeout - continue waiting (expected behavior)
                            continue
                        mid = msg.get("id")
                        if mid == req_id:
                            msg.pop("id", None)
                            if self._cancel_flags and req_id in self._cancel_flags:
                                del self._cancel_flags[req_id]
                            return msg
                        self._resp_buffer[mid] = msg
            except (KeyboardInterrupt, SystemExit):
                raise
            except (OSError, AttributeError, ValueError) as e:
                # Worker restart failed - stop and return error
                try:
                    from .logging_config import safe_log
                    safe_log("worker", "error", f"Failed to restart workers: {e}", exc_info=True)
                except ImportError:
                    pass
                self.stop()
            return None


def _worker_daemon_main(
    req_q: Any, res_q: Any, stop_event: Any, cancel_flags: Any
) -> None:
    """Worker daemon main loop that processes requests from queue."""
    # Apply resource limits if available (Unix only)
    # On Windows, HAS_RESOURCE is False, so this is skipped gracefully
    if HAS_RESOURCE:
        try:
            _limit_resources()
        except (ImportError, OSError, ValueError, AttributeError):
            # Resource limits not available (Windows) or failed to apply
            pass
    while True:
        if stop_event.is_set():
            break
        try:
            msg = req_q.get(timeout=0.1)
        except (KeyboardInterrupt, SystemExit):
            # Don't re-raise in worker processes - just check stop_event and exit gracefully
            # The main process will handle KeyboardInterrupt and set stop_event
            if stop_event.is_set():
                break
            # If not stopped yet, set stop_event ourselves and exit
            try:
                stop_event.set()
            except (AttributeError, TypeError):
                pass
            break
        except Exception:
            # Queue timeout or empty - continue waiting (expected behavior)
            continue
        try:
            kind = msg.get("type")
            req_id = msg.get("id")

            # Check cancellation before processing
            if cancel_flags and cancel_flags.get(req_id, False):
                res_q.put(
                    {
                        "ok": False,
                        "error": "Request cancelled",
                        "error_code": "CANCELLED",
                        "id": req_id,
                    }
                )
                continue

            if kind == "eval":
                pre = msg.get("preprocessed") or ""
                try:
                    out = worker_evaluate(pre)
                except Exception as eval_error:
                    # Handle any errors in worker_evaluate gracefully
                    out = {
                        "ok": False,
                        "error": str(eval_error),
                        "error_code": "EVAL_ERROR",
                    }
                out["id"] = req_id
                # Check cancellation after processing
                if cancel_flags and cancel_flags.get(req_id, False):
                    res_q.put(
                        {
                            "ok": False,
                            "error": "Request cancelled",
                            "error_code": "CANCELLED",
                            "id": req_id,
                        }
                    )
                else:
                    res_q.put(out)
            elif kind == "solve":
                payload = msg.get("payload") or {}
                out = _worker_solve_dispatch(payload)
                out["id"] = req_id
                # Check cancellation after processing
                if cancel_flags and cancel_flags.get(req_id, False):
                    res_q.put(
                        {
                            "ok": False,
                            "error": "Request cancelled",
                            "error_code": "CANCELLED",
                            "id": req_id,
                        }
                    )
                else:
                    res_q.put(out)
            else:
                res_q.put({"ok": False, "error": "Unknown request type", "id": req_id})
        except Exception as e:
            # Log the full error for debugging, but provide user-friendly message
            error_msg = str(e)
            # On Windows, resource module errors are expected - provide clearer message
            if "resource" in error_msg.lower() and "no module" in error_msg.lower():
                error_msg = "Resource limits unavailable on Windows (expected limitation)"
            res_q.put(
                {"ok": False, "error": f"Worker daemon error: {error_msg}", "id": msg.get("id")}
            )


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
            return {
                "ok": False,
                "error": "No valid equations provided to worker-solve.",
            }
        if HAS_RESOURCE:
            try:
                _limit_resources()
            except (ImportError, OSError, ValueError, AttributeError):
                # Resource limits failed - log but continue
                try:
                    from .logging_config import safe_log
                    safe_log("worker", "warning", "Failed to apply resource limits", exc_info=True)
                except ImportError:
                    pass
        solutions = sp.solve(eq_objs, dict=True)
        if not solutions:
            # Analyze why no solutions were found
            error_hints = []
            # Check for obviously impossible equations
            for eq in eq_objs:
                # Check for sin/cos/tan with impossible values
                if eq.has(sp.sin, sp.cos, sp.tan):
                    # Try to detect if any trig equation is impossible
                    if eq.has(sp.sin):
                        # Check if sin(x) = something where |something| > 1
                        try:
                            # Try to extract the value being compared
                            if eq.lhs.has(sp.sin) and not eq.rhs.has(sp.sin):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: sin(x) cannot equal {rhs_val} (|sin(x)| <= 1)")
                            elif eq.rhs.has(sp.sin) and not eq.lhs.has(sp.sin):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: sin(x) cannot equal {lhs_val} (|sin(x)| <= 1)")
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if eq.has(sp.cos):
                        try:
                            if eq.lhs.has(sp.cos) and not eq.rhs.has(sp.cos):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: cos(x) cannot equal {rhs_val} (|cos(x)| <= 1)")
                            elif eq.rhs.has(sp.cos) and not eq.lhs.has(sp.cos):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: cos(x) cannot equal {lhs_val} (|cos(x)| <= 1)")
                        except (ValueError, TypeError, AttributeError):
                            pass
            
            error_msg = "No solution found for this system of equations."
            if error_hints:
                error_msg += " Possible reasons:\n" + "\n".join(f"  - {hint}" for hint in error_hints)
            else:
                error_msg += " The system may be inconsistent, overdetermined, or have no real solutions. Check for contradictory equations."
            return {
                "ok": False,
                "error": error_msg,
                "error_code": "NO_SOLUTION",
            }
        # Filter solutions to only include real ones
        # Import tolerance constant
        from .config import NUMERIC_TOLERANCE
        
        real_sols = []
        complex_sols = []
        for sol in solutions:
            is_real = True
            for var, val in sol.items():
                try:
                    # Check if the value is real (imaginary part is negligible)
                    num_val = sp.N(val)
                    if abs(sp.im(num_val)) >= NUMERIC_TOLERANCE:
                        is_real = False
                        break
                except (ValueError, TypeError, AttributeError):
                    # If we can't evaluate, assume it might be complex
                    # Check if it's obviously complex (contains I or complex operations)
                    val_str = str(val)
                    if 'I' in val_str or 'asin(' in val_str.lower() or 'acos(' in val_str.lower():
                        # Check if asin/acos would produce complex (e.g., asin(pi) is complex)
                        if 'asin' in val_str.lower():
                            try:
                                # Try to extract what's inside asin
                                import re
                                match = re.search(r'asin\(([^)]+)\)', val_str, re.IGNORECASE)
                                if match:
                                    inner = match.group(1)
                                    inner_val = float(sp.N(inner))
                                    if abs(inner_val) > 1:
                                        is_real = False
                                        break
                            except (ValueError, TypeError, AttributeError):
                                pass
                        if 'acos' in val_str.lower():
                            try:
                                match = re.search(r'acos\(([^)]+)\)', val_str, re.IGNORECASE)
                                if match:
                                    inner = match.group(1)
                                    inner_val = float(sp.N(inner))
                                    if abs(inner_val) > 1:
                                        is_real = False
                                        break
                            except (ValueError, TypeError, AttributeError):
                                pass
            if is_real:
                real_sols.append(sol)
            else:
                complex_sols.append(sol)
        
        # If we have only complex solutions, provide helpful error
        if not real_sols and complex_sols:
            error_hints = []
            for eq in eq_objs:
                if eq.has(sp.sin, sp.cos, sp.tan):
                    if eq.has(sp.sin):
                        try:
                            if eq.lhs.has(sp.sin) and not eq.rhs.has(sp.sin):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: sin(x) cannot equal {rhs_val} (|sin(x)| <= 1)")
                            elif eq.rhs.has(sp.sin) and not eq.lhs.has(sp.sin):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: sin(x) cannot equal {lhs_val} (|sin(x)| <= 1)")
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if eq.has(sp.cos):
                        try:
                            if eq.lhs.has(sp.cos) and not eq.rhs.has(sp.cos):
                                rhs_val = float(sp.N(eq.rhs))
                                if abs(rhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: cos(x) cannot equal {rhs_val} (|cos(x)| <= 1)")
                            elif eq.rhs.has(sp.cos) and not eq.lhs.has(sp.cos):
                                lhs_val = float(sp.N(eq.lhs))
                                if abs(lhs_val) > 1:
                                    error_hints.append(f"Equation '{eq}' has no real solutions: cos(x) cannot equal {lhs_val} (|cos(x)| <= 1)")
                        except (ValueError, TypeError, AttributeError):
                            pass
            error_msg = "No real solutions found for this system of equations (only complex solutions exist)."
            if error_hints:
                error_msg += " Reasons:\n" + "\n".join(f"  - {hint}" for hint in error_hints)
            else:
                error_msg += " The system may have complex solutions but no real solutions."
            return {
                "ok": False,
                "error": error_msg,
                "error_code": "NO_REAL_SOLUTIONS",
            }
        
        # Return only real solutions
        sols = []
        for sol in real_sols:
            sols.append({str(k): str(v) for k, v in sol.items()})
        return {"ok": True, "type": "system", "solutions": sols}
    except Exception as e:
        return {"ok": False, "error": f"Solver error: {e}"}


_WORKER_MANAGER = _WorkerManager()


def _worker_eval_cached(preprocessed_expr: str) -> str:
    """Evaluate expression with persistent cache support."""
    # Check persistent cache first
    try:
        from .cache_manager import get_cached_eval, update_eval_cache, update_subexpr_cache
        cached_result = get_cached_eval(preprocessed_expr)
        if cached_result is not None:
            if logger:
                logger.debug(f"Cache hit for: {preprocessed_expr[:50]}")
            return cached_result
    except ImportError:
        pass
    
    # Not in persistent cache, evaluate normally
    resp = _WORKER_MANAGER.request(
        {"type": "eval", "preprocessed": preprocessed_expr}, timeout=WORKER_TIMEOUT
    )
    if isinstance(resp, dict):
        result_json = json.dumps(resp)
        # Save to persistent cache if evaluation was successful
        try:
            from .cache_manager import update_eval_cache, update_subexpr_cache
            if resp.get("ok"):
                update_eval_cache(preprocessed_expr, result_json)
                # Also cache as sub-expression if it's a simple numeric result
                result_value = resp.get("result", "")
                approx_value = resp.get("approx", "")
                # Only cache pure numeric expressions (no variables)
                if result_value and not any(c in result_value for c in ['x', 'y', 'z', 'X', 'Y', 'Z', 'a', 'b', 'c']):
                    # Cache the sub-expression mapping
                    cache_value = approx_value if approx_value else result_value
                    if cache_value:
                        update_subexpr_cache(preprocessed_expr, cache_value)
        except ImportError:
            pass
        return result_json
    cmd = _build_self_cmd(["--worker", "--expr", preprocessed_expr])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 bytes instead of raising error
        )
        result_text = proc.stdout or ""
        # Try to save to persistent cache
        try:
            from .cache_manager import update_eval_cache, update_subexpr_cache
            try:
                result_data = json.loads(result_text)
                if result_data.get("ok"):
                    update_eval_cache(preprocessed_expr, result_text)
                    result_value = result_data.get("result", "")
                    approx_value = result_data.get("approx", "")
                    # Only cache pure numeric expressions
                    if result_value and not any(c in result_value for c in ['x', 'y', 'z', 'X', 'Y', 'Z', 'a', 'b', 'c']):
                        cache_value = approx_value if approx_value else result_value
                        if cache_value:
                            update_subexpr_cache(preprocessed_expr, cache_value)
            except (json.JSONDecodeError, KeyError):
                pass
        except ImportError:
            pass
        return result_text
    except UnicodeDecodeError:
        # Fallback if UTF-8 decoding fails
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=WORKER_TIMEOUT,
        )
        return proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""


@lru_cache(maxsize=CACHE_SIZE_SOLVE)
def _worker_solve_cached(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Invalid JSON - log and use empty equations
        try:
            from .logging_config import safe_log
            safe_log("worker", "warning", f"Invalid JSON payload in solve cache: {e}")
        except ImportError:
            pass
        payload = {"equations": []}
    resp = _WORKER_MANAGER.request(
        {"type": "solve", "payload": payload}, timeout=WORKER_TIMEOUT
    )
    if isinstance(resp, dict):
        return json.dumps(resp)
    cmd = _build_self_cmd(["--worker-solve", "--payload", payload_json])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT,
            encoding="utf-8",
            errors="replace",  # Replace invalid UTF-8 bytes instead of raising error
        )
        return proc.stdout or ""
    except UnicodeDecodeError:
        # Fallback if UTF-8 decoding fails
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=WORKER_TIMEOUT,
        )
        return proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""


def evaluate_safely(expr: str, timeout: int = WORKER_TIMEOUT) -> Dict[str, Any]:
    """Safely evaluate an expression string via worker sandbox."""
    from .parser import preprocess

    try:
        pre = preprocess(expr)
    except ValidationError as e:
        return {"ok": False, "error": str(e), "error_code": e.code}
    except ValueError as e:
        return {
            "ok": False,
            "error": f"Preprocess error: {e}",
            "error_code": "PREPROCESS_ERROR",
        }
    except (ValueError, TypeError, AttributeError) as e:
        # Unexpected error in preprocessing - log it
        try:
            from .logging_config import safe_log
            safe_log("worker", "error", f"Unexpected preprocessing error: {e}", exc_info=True)
        except ImportError:
            pass
        return {"ok": False, "error": "Preprocess error", "error_code": "UNKNOWN_ERROR"}
    try:
        stdout_text = _worker_eval_cached(pre)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Evaluation timed out.", "error_code": "TIMEOUT"}
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        # Specific exceptions for worker communication failures
        try:
            from .logging_config import safe_log
            safe_log("worker", "error", f"Worker communication error: {e}", exc_info=True)
        except ImportError:
            pass
        return {
            "ok": False,
            "error": "Worker communication failed",
            "error_code": "COMM_ERROR",
        }
    try:
        data = json.loads(stdout_text)
        return data
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Specific exceptions for JSON parsing
        return {
            "ok": False,
            "error": f"Invalid worker output: {e}.",
            "error_code": "INVALID_OUTPUT",
        }
    except Exception as e:
        # Catch-all for truly unexpected errors - log full traceback
        logger.exception("Unexpected error parsing worker output")
        return {
            "ok": False,
            "error": f"Invalid worker output: {e}",
            "error_code": "UNKNOWN_ERROR",
        }


def clear_caches() -> None:
    """Clear worker-side LRU caches, parser cache, and persistent cache."""
    try:
        # Clear in-memory LRU caches
        try:
            _worker_eval_cached.cache_clear()
        except AttributeError:
            pass  # Function may not be decorated with lru_cache anymore
        _worker_solve_cached.cache_clear()
        from .parser import parse_preprocessed as _pp
        _pp.cache_clear()
        
        # Clear persistent cache
        try:
            from .cache_manager import clear_persistent_cache
            clear_persistent_cache()
        except ImportError:
            pass
    except (ValueError, TypeError, AttributeError):
        # Expected for some cache operations
        pass


def cancel_current_request(req_id: Optional[str] = None) -> bool:
    """Cancel a pending worker request. If req_id is None, attempts to cancel the most recent."""
    if req_id:
        return _WORKER_MANAGER.cancel_request(req_id)
    return False  # For now, requires explicit ID - can be enhanced
