from __future__ import annotations

import argparse
import json
import re
from typing import Any

import sympy as sp

from .config import VAR_NAME_RE, VERSION
from .parser import (
    format_inequality_solution,
    format_number,
    format_solution,
    format_superscript,
    parse_preprocessed,
    split_top_level_commas,
)
from .solver import solve_inequality, solve_single_equation, solve_system
from .types import ParseError, ValidationError
from .worker import evaluate_safely


def _health_check() -> int:
    """Run health check to verify dependencies and basic operations.

    Returns:
        Exit code (0 for success, non-zero for failures)
    """
    checks_passed = 0
    checks_failed = 0

    print("Running Kalkulator health check...")
    print("-" * 50)

    # Check SymPy import
    try:
        import sympy as sp

        version = sp.__version__
        print(f"[OK] SymPy {version} imported successfully")
        checks_passed += 1
    except ImportError as e:
        print(f"[FAIL] SymPy import failed: {e}")
        checks_failed += 1

    # Check basic parsing
    try:
        from .parser import parse_preprocessed, preprocess

        test_expr = "2 + 2"
        preprocessed = preprocess(test_expr)
        parsed = parse_preprocessed(preprocessed)
        if parsed == 4:
            print("[OK] Basic parsing works")
            checks_passed += 1
        else:
            print(f"[FAIL] Basic parsing failed: expected 4, got {parsed}")
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Parsing check failed: {e}")
        checks_failed += 1

    # Check solving
    try:
        from .solver import solve_single_equation

        result = solve_single_equation("x + 1 = 0")
        if result.get("ok") and result.get("exact") == ["-1"]:
            print("[OK] Basic solving works")
            checks_passed += 1
        else:
            print(f"[FAIL] Solving check failed: {result}")
            checks_failed += 1
    except Exception as e:
        print(f"[FAIL] Solving check failed: {e}")
        checks_failed += 1

    # Check worker (if available)
    try:
        from .worker import evaluate_safely

        result = evaluate_safely("3 * 3")
        if result.get("ok") and result.get("result") == "9":
            print("[OK] Worker evaluation works")
            checks_passed += 1
        else:
            print(f"[FAIL] Worker check failed: {result}")
            checks_failed += 1
    except Exception as e:
        print(f"[WARN] Worker check skipped: {e}")

    # Check optional dependencies
    try:
        import numpy

        print(f"[OK] NumPy {numpy.__version__} available")
        checks_passed += 1
    except ImportError:
        print("[WARN] NumPy not available (plotting features limited)")

    try:
        import matplotlib

        print(f"[OK] Matplotlib {matplotlib.__version__} available")
        checks_passed += 1
    except ImportError:
        print("[WARN] Matplotlib not available (plotting features limited)")

    # Check Windows-specific limitations
    try:
        import sys

        if sys.platform == "win32":
            try:
                import resource  # noqa: F401

                print("[OK] Resource limits available (Unix-like behavior)")
                checks_passed += 1
            except ImportError:
                print(
                    "[WARN] Resource limits unavailable on Windows (expected limitation)"
                )
                print(
                    "[INFO] Consider containerization for production deployments on Windows"
                )
    except Exception:
        pass

    print("-" * 50)
    print(f"Results: {checks_passed} passed, {checks_failed} failed")

    if checks_failed > 0:
        print("\n[WARN] Some health checks failed. Core functionality may be impaired.")
        return 1

    print("\n[OK] All health checks passed!")
    try:
        import sys

        if sys.platform == "win32":
            try:
                import resource  # noqa: F401
            except ImportError:
                print("[INFO] See DEPLOYMENT.md for Windows deployment recommendations")
    except Exception:
        pass
    return 0


def print_result_pretty(res: dict[str, Any], output_format: str = "human") -> None:
    """Print result in specified format.

    Args:
        res: Result dictionary
        output_format: "json" for JSON output, "human" for human-readable
    """
    if output_format == "json":
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if not res.get("ok"):
        print("Error:", res.get("error"))
        return
    typ = res.get("type", "value")
    if typ == "equation":
        print("Exact:", ", ".join(format_solution(sol) for sol in res.get("exact", [])))
        if res.get("approx"):
            approx_display = ", ".join(
                format_number(approx_val)
                for approx_val in res.get("approx", [])
                if approx_val is not None
            )
            if approx_display:
                print("Approx:", approx_display)
    elif typ == "multi_isolate":
        sols = res.get("solutions", {})
        approx = res.get("approx", {})
        for var, sol_list in sols.items():
            if isinstance(sol_list, (list, tuple)):
                formatted = ", ".join(
                    format_solution(solution) for solution in sol_list
                )
            else:
                formatted = format_solution(sol_list)
            print(f"{var} = {formatted}")
            approx_list = approx.get(var)
            if approx_list:
                approx_display = ", ".join(
                    format_number(approx_val)
                    for approx_val in approx_list
                    if approx_val is not None
                )
                if approx_display:
                    print(f"  Decimal: {approx_display}")
    elif typ == "inequality":
        for k, v in res.get("solutions", {}).items():
            formatted_v = format_inequality_solution(str(v))
            print(f"Solution for {k}: {formatted_v}")
    elif typ == "pell":
        solution_str = res.get("solution", "")
        # Handle Unicode characters for Windows console compatibility
        try:
            print("Pell parametric solution:")
            print(solution_str)
        except UnicodeEncodeError:
            # Fallback: replace Unicode with ASCII
            safe_solution = solution_str.replace("\u221a", "sqrt").replace(
                "\u00b2", "^2"
            )
            print("Pell parametric solution:")
            print(safe_solution)
    elif typ == "identity_or_contradiction":
        print(res.get("result"))
    elif typ == "evaluation":
        # Handle evaluation results (e.g., "2+2=")
        exact_list = res.get("exact", [])
        approx_list = res.get("approx", [])
        if exact_list:
            print(exact_list[0] if exact_list[0] else "")
        if approx_list and approx_list[0]:
            print(f"Decimal: {approx_list[0]}")
    elif typ == "value":
        res_str = res.get("result")
        approx = res.get("approx")
        if res_str is None:
            print(res)
            return
        try:
            print(f"{res_str}")
        except (UnicodeEncodeError, OSError):
            # Handle encoding errors on Windows console
            try:
                # Try printing without formatting
                print(str(res_str))
            except (UnicodeEncodeError, OSError):
                # Last resort: print raw representation
                print(repr(res_str))
        try:
            parsed = parse_preprocessed(res_str)
            expanded = sp.expand(parsed)
            if str(expanded) != str(parsed):
                print(f"Expanded: {format_solution(expanded)}")
        except (ParseError, ValidationError, ValueError, TypeError, AttributeError):
            # Expected errors for some expressions - silently skip expansion
            pass
        if approx:
            print("Decimal:", approx)
    else:
        print(res)


def repl_loop(output_format: str = "human") -> None:
    """Interactive REPL loop with graceful interrupt handling."""
    try:
        import readline  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # readline not available on Windows - that's fine
        pass
    print("Kalkulator Aljabar — type 'help' for commands, 'quit' to exit.")
    _current_req_id = None  # Track current request for cancellation
    _timing_enabled = False  # Track whether timing is enabled

    def signal_handler(signum: Any, frame: Any) -> None:
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
            # Stop worker processes gracefully
            try:
                from .worker import _WORKER_MANAGER

                _WORKER_MANAGER.stop()
            except Exception:
                pass
            # Save persistent cache on shutdown
            try:
                from .cache_manager import save_cache_to_disk

                save_cache_to_disk()
            except ImportError:
                pass
            break
        if not raw:
            continue
        if raw.lower() in ("clearcache", "clear cache"):
            from .worker import clear_caches

            clear_caches()
            print("Caches cleared.")
            continue
        if raw.lower() in ("showcache", "show cache", "cache"):
            try:
                from .cache_manager import get_persistent_cache

                cache = get_persistent_cache()
                eval_cache = cache.get("eval_cache", {})
                subexpr_cache = cache.get("subexpr_cache", {})

                print("\n=== Cache Status ===")
                print(f"Evaluation cache entries: {len(eval_cache)}")
                print(f"Sub-expression cache entries: {len(subexpr_cache)}")
                print(f"Total entries: {len(eval_cache) + len(subexpr_cache)}")

                if subexpr_cache:
                    print("\n=== Sub-expression Cache (showing first 20) ===")
                    for _i, (expr, value) in enumerate(
                        list(subexpr_cache.items())[:20]
                    ):
                        print(f"  {expr} → {value}")
                    if len(subexpr_cache) > 20:
                        print(f"  ... and {len(subexpr_cache) - 20} more entries")

                if eval_cache:
                    print("\n=== Evaluation Cache (showing first 10) ===")
                    for _i, (expr, result_json) in enumerate(
                        list(eval_cache.items())[:10]
                    ):
                        try:
                            import json

                            result_data = json.loads(result_json)
                            result_str = result_data.get("result", "N/A")
                            if len(result_str) > 50:
                                result_str = result_str[:47] + "..."
                            print(f"  {expr[:50]:50} → {result_str}")
                        except (json.JSONDecodeError, KeyError):
                            print(f"  {expr[:50]:50} → [cached result]")
                    if len(eval_cache) > 10:
                        print(f"  ... and {len(eval_cache) - 10} more entries")
                else:
                    print("\nNo cached entries yet.")
            except ImportError:
                print("Cache manager not available.")
            continue
        if raw.lower().startswith("savecache"):
            try:
                import os

                from .cache_manager import export_cache_to_file, get_persistent_cache

                parts = raw.split(None, 1)
                if len(parts) > 1:
                    file_path = parts[1].strip()
                    # Remove quotes if present
                    if (file_path.startswith('"') and file_path.endswith('"')) or (
                        file_path.startswith("'") and file_path.endswith("'")
                    ):
                        file_path = file_path[1:-1]
                else:
                    # Default to cache_backup.json in current directory
                    file_path = "cache_backup.json"

                cache = get_persistent_cache()
                total_entries = len(cache.get("eval_cache", {})) + len(
                    cache.get("subexpr_cache", {})
                )

                if export_cache_to_file(file_path):
                    print(f"Cache exported successfully to: {file_path}")
                    print(f"  ({total_entries} total entries saved)")
                else:
                    print(f"Error: Failed to export cache to {file_path}")
            except Exception as e:
                print(f"Error saving cache: {e}")
            continue
        if raw.lower().startswith("loadcache"):
            try:
                import os

                from .cache_manager import (
                    get_persistent_cache,
                    import_cache_from_file,
                    replace_cache_from_file,
                )

                parts = raw.split()
                replace_mode = False
                file_path = None

                # Parse arguments: loadcache [replace] <file_path>
                if len(parts) > 1:
                    if parts[1].lower() == "replace":
                        replace_mode = True
                        if len(parts) > 2:
                            file_path = parts[2].strip()
                            if (
                                file_path.startswith('"') and file_path.endswith('"')
                            ) or (
                                file_path.startswith("'") and file_path.endswith("'")
                            ):
                                file_path = file_path[1:-1]
                    else:
                        file_path = parts[1].strip()
                        if (file_path.startswith('"') and file_path.endswith('"')) or (
                            file_path.startswith("'") and file_path.endswith("'")
                        ):
                            file_path = file_path[1:-1]

                if not file_path:
                    # Default to cache_backup.json in current directory
                    file_path = "cache_backup.json"

                if not os.path.exists(file_path):
                    print(f"Error: File not found: {file_path}")
                    continue

                if replace_mode:
                    if replace_cache_from_file(file_path):
                        cache = get_persistent_cache()
                        total_entries = len(cache.get("eval_cache", {})) + len(
                            cache.get("subexpr_cache", {})
                        )
                        print(f"Cache replaced from: {file_path}")
                        print(f"  ({total_entries} total entries loaded)")
                    else:
                        print(f"Error: Failed to replace cache from {file_path}")
                else:
                    if import_cache_from_file(file_path):
                        cache = get_persistent_cache()
                        total_entries = len(cache.get("eval_cache", {})) + len(
                            cache.get("subexpr_cache", {})
                        )
                        print(f"Cache merged from: {file_path}")
                        print(f"  ({total_entries} total entries in cache after merge)")
                    else:
                        print(f"Error: Failed to import cache from {file_path}")
            except Exception as e:
                print(f"Error loading cache: {e}")
            continue
        if raw.lower().startswith("timing"):
            parts = raw.lower().split()
            if len(parts) == 1 or (
                len(parts) == 2 and parts[1] in ("on", "enable", "1")
            ):
                _timing_enabled = True
                print("Timing enabled. Calculation time will be displayed.")
            elif len(parts) == 2 and parts[1] in ("off", "disable", "0"):
                _timing_enabled = False
                print("Timing disabled.")
            else:
                print("Usage: timing [on|off]")
            continue
        if raw.startswith("--eval"):
            parts = raw.split(None, 1)
            if len(parts) == 1:
                print("Usage: --eval <expression>")
                continue
            expr = parts[1].strip()
            if (expr.startswith('"') and expr.endswith('"')) or (
                expr.startswith("'") and expr.endswith("'")
            ):
                expr = expr[1:-1]
            try:
                import time

                start_time = time.perf_counter()
                # Check for unterminated quotes before processing
                if expr.count('"') % 2 != 0 or expr.count("'") % 2 != 0:
                    print(
                        "Error: Unmatched quotes detected. Check that all opening quotes have matching closing quotes."
                    )
                    if _timing_enabled:
                        print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                    continue

                # Check for assignment with equation mixed (e.g., "a=(expr=0)")
                if "," in expr:
                    parts = split_top_level_commas(expr)
                    for part in parts:
                        if "=" in part:
                            # Check if this looks like an assignment with nested equation
                            assign_parts = part.split("=", 1)
                            if len(assign_parts) == 2 and "=" in assign_parts[1]:
                                print(
                                    f"Error: Cannot use assignment '=' inside another assignment. Found: '{part}'"
                                )
                                print(
                                    "Hint: Separate assignments and equations. Example: Use 'a = expression' then solve 'equation = 0' separately."
                                )
                                if _timing_enabled:
                                    print(
                                        f"[Time: {time.perf_counter() - start_time:.4f}s]"
                                    )
                                continue

                if any(op in expr for op in ("<", ">", "<=", ">=")):
                    res = solve_inequality(expr, None)
                    elapsed = time.perf_counter() - start_time
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                elif "=" in expr:
                    pts = split_top_level_commas(expr)
                    if len(pts) > 1:
                        res = solve_system(expr, None)
                        elapsed = time.perf_counter() - start_time
                        print_result_pretty(res, output_format=output_format)
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                    else:
                        res = solve_single_equation(expr, None)
                        elapsed = time.perf_counter() - start_time
                        print_result_pretty(res, output_format=output_format)
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                else:
                    eva = evaluate_safely(expr)
                    elapsed = time.perf_counter() - start_time
                    if not eva.get("ok"):
                        error_msg = eva.get("error", "Unknown error")
                        error_code = eva.get("error_code", "UNKNOWN_ERROR")

                        # Provide helpful hints based on error code
                        if error_code == "SYNTAX_ERROR":
                            if (
                                "unterminated" in error_msg.lower()
                                or "unmatched" in error_msg.lower()
                            ):
                                print(f"Error: {error_msg}")
                                print(
                                    "Hint: Check that all quotes, parentheses, and brackets are properly matched."
                                )
                            else:
                                print(f"Error: {error_msg}")
                        elif error_code == "PARSE_ERROR":
                            print(f"Error: {error_msg}")
                            # Additional hints already included in error message from worker
                        else:
                            print(f"Error: {error_msg}")

                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                    else:
                        res_str = eva.get("result")
                        print(f"{expr} = {format_superscript(res_str)}")
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        try:
                            parsed = parse_preprocessed(res_str)
                            expanded = sp.expand(parsed)
                            if str(expanded) != str(parsed):
                                print(f"Expanded: {format_solution(expanded)}")
                        except (
                            ParseError,
                            ValidationError,
                            ValueError,
                            TypeError,
                            AttributeError,
                        ):
                            # Expected errors for some expressions - silently skip expansion
                            pass
                        if eva.get("approx"):
                            print("Decimal:", eva.get("approx"))
            except Exception as e:
                # Log full error but show clean message to user
                try:
                    from .logging_config import get_logger

                    logger = get_logger("cli")
                    logger.error(f"Error handling --eval in REPL: {e}", exc_info=True)
                except ImportError:
                    pass

                # Show clean error message without traceback
                error_str = str(e)
                if (
                    "TokenError" in str(type(e).__name__)
                    or "unterminated" in error_str.lower()
                ):
                    print(
                        "Error: Syntax error detected. Check that all quotes and parentheses are properly matched."
                    )
                elif "assign" in error_str.lower():
                    print(f"Error: {error_str}")
                    print(
                        "Hint: Use '==' for equations and '=' for assignments. Don't mix them in a single expression."
                    )
                else:
                    print(f"Error: {error_str}")
            continue
        cmd = raw.lower()
        if cmd in ("quit", "exit"):
            print("Goodbye.")
            break
        if cmd in ("help", "?", "--help"):
            print_help_text()
            continue
        try:
            import time

            start_time = time.perf_counter()
            if any(op in raw for op in ("<", ">", "<=", ">=")):
                res = solve_inequality(raw, None)
                elapsed = time.perf_counter() - start_time
                print_result_pretty(res, output_format=output_format)
                if _timing_enabled:
                    print(f"[Time: {elapsed:.4f}s]")
                continue
            find_tokens = re.findall(r"\bfind\s+(\w+)\b", raw, re.IGNORECASE)
            find = find_tokens[0] if find_tokens else None
            raw_no_find = re.sub(
                r"\bfind\s+\w+\b", "", raw, flags=re.IGNORECASE
            ).strip()
            parts = split_top_level_commas(raw_no_find)
            if not parts:
                print("No valid parts parsed.")
                continue
            all_assign = all(
                "=" in p and VAR_NAME_RE.match(p.split("=", 1)[0].strip())
                for p in parts
            )
            all_eq = all("=" in p for p in parts)
            if all_assign and len(parts) > 0:
                if find:
                    res = solve_system(raw_no_find, find)
                    elapsed = time.perf_counter() - start_time
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                    continue
                subs = {}
                for p in parts:
                    left, right = p.split("=", 1)
                    var = left.strip()
                    rhs = right.strip() or "0"
                    res = evaluate_safely(rhs)
                    if not res.get("ok"):
                        print("Error evaluating assignment RHS:", res.get("error"))
                        if _timing_enabled:
                            print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                        continue
                    try:
                        val = parse_preprocessed(res["result"]).subs(subs)
                    except Exception as e:
                        print("Error assembling assignment value:", e)
                        if _timing_enabled:
                            print(f"[Time: {time.perf_counter() - start_time:.4f}s]")
                        continue
                    subs[var] = val
                    print(f"{var} = {format_solution(val)}")
                if _timing_enabled:
                    elapsed = time.perf_counter() - start_time
                    print(f"[Time: {elapsed:.4f}s]")
                continue
            if len(parts) > 1 and all_eq:
                res = solve_system(raw_no_find, find)
                elapsed = time.perf_counter() - start_time
                print_result_pretty(res, output_format=output_format)
                if _timing_enabled:
                    print(f"[Time: {elapsed:.4f}s]")
                continue
            elif len(parts) > 1:
                print(
                    "Substitution not implemented in REPL (use eval mode) — please use --eval or provide a single expression."
                )
                continue
            else:
                part = parts[0]
                if "=" in part:
                    res = solve_single_equation(part, find)
                    elapsed = time.perf_counter() - start_time
                    print_result_pretty(res, output_format=output_format)
                    if _timing_enabled:
                        print(f"[Time: {elapsed:.4f}s]")
                else:
                    eva = evaluate_safely(part)
                    elapsed = time.perf_counter() - start_time
                    if not eva.get("ok"):
                        print("Error:", eva.get("error"))
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                    else:
                        res_str = eva.get("result")
                        print(f"{part} = {format_superscript(res_str)}")
                        if _timing_enabled:
                            print(f"[Time: {elapsed:.4f}s]")
                        try:
                            parsed = parse_preprocessed(res_str)
                            expanded = sp.expand(parsed)
                            if str(expanded) != str(parsed):
                                print(f"Expanded: {format_solution(expanded)}")
                        except (
                            ParseError,
                            ValidationError,
                            ValueError,
                            TypeError,
                            AttributeError,
                        ):
                            # Expected errors for some expressions - silently skip expansion
                            pass
                        if eva.get("approx"):
                            print("Decimal:", eva.get("approx"))
        except Exception as e:
            try:
                from .logging_config import get_logger

                logger = get_logger("cli")
                logger.error(f"Unexpected error in REPL: {e}", exc_info=True)
            except ImportError:
                pass
            print("An error occurred. Please check your input and try again.")
            continue


def print_help_text() -> None:
    """Print help text for REPL commands."""
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
- In REPL: help, quit, exit, clearcache, showcache, savecache [file], loadcache [replace] [file], timing [on|off]

Calculus & matrices (new):
- diff(x^3, x)
- integrate(sin(x), x)
- factor(x^3 - 1)
- expand((x+1)^3)
- Matrix([[1,2],[3,4]])
- det(Matrix([[1,2],[3,4]]))
"""
    print(help_text)


def main_entry(argv: list[str] | None = None) -> int:
    """
    Main entry point for Kalkulator CLI.

    Args:
        argv: Optional command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Load persistent cache on startup
    try:
        from .cache_manager import load_persistent_cache

        load_persistent_cache()  # Initialize cache
    except ImportError:
        pass  # Cache manager not available, continue without persistent cache

    parser = argparse.ArgumentParser(prog="kalkulator")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expr", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-solve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--payload", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "-e",
        "--eval",
        type=str,
        help="Evaluate one expression and exit (non-interactive)",
        dest="eval_expr",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Emit JSON for machine parsing (deprecated, use --format json)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "human"],
        default="human",
        help="Output format: json (machine-readable) or human (human-readable)",
    )
    parser.add_argument(
        "-t", "--timeout", type=int, help="Override worker timeout (seconds)"
    )
    parser.add_argument(
        "--no-numeric-fallback",
        action="store_true",
        help="Disable numeric root-finding fallback",
    )
    parser.add_argument(
        "-p", "--precision", type=int, help="Set output precision (significant digits)"
    )
    parser.add_argument(
        "-v", "--version", action="store_true", help="Show program version"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument("--log-file", type=str, help="Write logs to file")
    parser.add_argument(
        "--cache-size", type=int, help="Set parse/eval cache size (default: 1024/2048)"
    )
    parser.add_argument(
        "--max-nsolve-guesses",
        type=int,
        help="Set maximum nsolve guesses for numeric root finding (default: 36)",
    )
    parser.add_argument(
        "--worker-mode",
        type=str,
        choices=["pool", "single", "subprocess"],
        help="Worker execution mode (default: pool)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "symbolic", "numeric"],
        help="Solver method (default: auto)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and verify dependencies",
    )
    args = parser.parse_args(argv)

    # Determine output format
    output_format = args.format
    if args.json:  # Backward compatibility for deprecated -j flag
        output_format = "json"

    # Setup logging
    try:
        from .logging_config import setup_logging

        setup_logging(level=args.log_level, log_file=args.log_file)
    except ImportError:
        pass  # Logging optional
    # Apply CLI configuration overrides
    # Note: Direct module variable modification is used for simplicity.
    # Future improvement: Use dependency injection or a configuration object.
    import kalkulator_pkg.config as _config
    import kalkulator_pkg.worker as _worker_module

    if args.timeout and args.timeout > 0:
        _worker_module.WORKER_TIMEOUT = int(args.timeout)
    if args.no_numeric_fallback:
        _config.NUMERIC_FALLBACK_ENABLED = False
    if args.precision and args.precision > 0:
        _config.OUTPUT_PRECISION = int(args.precision)
    if args.cache_size and args.cache_size > 0:
        _config.CACHE_SIZE_PARSE = int(args.cache_size)
        _config.CACHE_SIZE_EVAL = int(args.cache_size * 2)
    if args.max_nsolve_guesses and args.max_nsolve_guesses > 0:
        _config.MAX_NSOLVE_GUESSES = int(args.max_nsolve_guesses)
    if args.worker_mode:
        if args.worker_mode == "subprocess":
            _config.ENABLE_PERSISTENT_WORKER = False
        elif args.worker_mode == "single":
            _config.WORKER_POOL_SIZE = 1
        # "pool" is default
    if args.method:
        _config.SOLVER_METHOD = args.method
    if args.worker:
        from .worker import worker_evaluate

        out = worker_evaluate(args.expr or "")
        print(json.dumps(out))
        return 0
    if args.worker_solve:
        from .worker import _worker_solve_dispatch

        try:
            payload = json.loads(args.payload or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            # Invalid JSON - use empty dict
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
            res = solve_inequality(raw_no_find, find)
        elif "=" in raw_no_find:
            parts = split_top_level_commas(raw_no_find)
            if len(parts) > 1:
                res = solve_system(raw_no_find, find)
            else:
                res = solve_single_equation(parts[0], find)
        else:
            eva = evaluate_safely(raw_no_find)
            if not eva.get("ok"):
                res = {"ok": False, "error": eva.get("error")}
            else:
                res = {
                    "ok": True,
                    "type": "value",
                    "result": eva.get("result"),
                    "approx": eva.get("approx"),
                }
        print_result_pretty(res, output_format=output_format)
        # Save cache after evaluation (periodic save)
        try:
            from .cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
        return 0

    # Add health check command
    if hasattr(args, "health_check") and args.health_check:
        return _health_check()

    try:
        repl_loop(output_format=output_format)
    finally:
        # Ensure worker processes are stopped on exit
        try:
            from .worker import _WORKER_MANAGER

            _WORKER_MANAGER.stop()
        except Exception:
            pass
        # Save persistent cache on exit
        try:
            from .cache_manager import save_cache_to_disk

            save_cache_to_disk()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    """Allow running the CLI module directly with python -m kalkulator_pkg.cli"""
    import sys

    sys.exit(main_entry())
