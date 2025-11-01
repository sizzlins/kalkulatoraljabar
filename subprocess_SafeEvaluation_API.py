
def evaluate_safely(expr: str, timeout: int = WORKER_TIMEOUT) -> Dict[str, Any]:
    """Spawn a short-lived worker subprocess (same script/executable) to evaluate the preprocessed expr.
    Returns the parsed result as dict.
    """
    # Preprocess here (string operations are cheap)
    try:
        pre = preprocess(expr)
    except Exception as e:
        return {"ok": False, "error": f"Preprocess error: {e}"}

    # Build the correct command to spawn this program in worker mode
    cmd = _build_self_cmd(["--worker", "--expr", pre])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Evaluation timed out."}
    if proc.returncode != 0:
        # worker writes JSON to stdout on success/failure. If it crashed, show stderr.
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        # Try to parse stdout anyway
        try:
            data = json.loads(stdout)
            return data
        except Exception:
            return {"ok": False, "error": f"Worker failed. stderr: {stderr}"}
    # parse stdout
    try:
        data = json.loads(proc.stdout)
        return data
    except Exception as e:
        return {"ok": False, "error": f"Invalid worker output: {e}. stdout: {proc.stdout!r}"}


def _build_self_cmd(args: List[str]) -> List[str]:
    """
    Return a subprocess argv list that invokes this program (script or bundled exe)
    with the provided args.

    - When running normally: ["<python>", "<path/to/script.py>", ...args...]
    - When frozen (PyInstaller onefile): ["<path/to/exe>", ...args...]
    """
    if getattr(sys, "frozen", False):
        # Use the bundled executable path (sys.argv[0]) to launch a new copy of the exe.
        return [os.path.realpath(sys.argv[0])] + args
    else:
        # Dev mode: invoke the interpreter with the .py file path as usual.
        return [sys.executable, os.path.realpath(__file__)] + args