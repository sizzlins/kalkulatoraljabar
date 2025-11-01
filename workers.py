
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
