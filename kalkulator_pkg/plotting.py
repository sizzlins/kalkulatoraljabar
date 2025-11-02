"""Optional plotting functionality for single-variable functions."""

from __future__ import annotations

import sympy as sp

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .parser import parse_preprocessed
from .types import EvalResult
from .worker import evaluate_safely


def plot_function(
    expression: str,
    variable: str = "x",
    x_min: float = -10,
    x_max: float = 10,
    points: int = 100,
    ascii: bool = False,
) -> EvalResult:
    """Plot a single-variable function.

    Args:
        expression: Function expression (e.g., "x^2")
        variable: Variable name (default: "x")
        x_min: Minimum x value (default: -10)
        x_max: Maximum x value (default: 10)
        points: Number of points to plot (default: 100)
        ascii: If True, return ASCII plot; if False, open matplotlib window

    Returns:
        EvalResult with plot data or ASCII representation
    """
    if not HAS_MATPLOTLIB and not ascii:
        return EvalResult(
            ok=False, error="matplotlib not installed. Use ascii=True for ASCII plot."
        )

    try:
        eval_result = evaluate_safely(expression)
        if not eval_result.get("ok"):
            return EvalResult(ok=False, error=eval_result.get("error"))

        expr = parse_preprocessed(eval_result["result"])
        var_sym = sp.symbols(variable)

        # Create numeric evaluation function
        f = sp.lambdify(var_sym, expr, "numpy")

        if ascii:
            # ASCII plot
            import numpy as np

            x_vals = np.linspace(x_min, x_max, points)
            y_vals = f(x_vals)

            # Simple ASCII representation
            # Constants for ASCII plot dimensions
            rows = 20  # Height of ASCII plot in characters
            cols = 60  # Width of ASCII plot in characters
            plot_chars = [[" " for _ in range(cols)] for _ in range(rows)]

            # Find y range
            valid_y = [y for y in y_vals if y == y and -1e10 < y < 1e10]
            if not valid_y:
                return EvalResult(
                    ok=False, error="Cannot plot: function values out of range"
                )

            y_min, y_max = min(valid_y), max(valid_y)
            y_range = y_max - y_min if y_max != y_min else 1

            # Plot points
            for x, y in zip(x_vals, y_vals):
                if y != y or not (-1e10 < y < 1e10):
                    continue
                col = int((x - x_min) / (x_max - x_min) * (cols - 1))
                row = int((y_max - y) / y_range * (rows - 1))
                col = max(0, min(cols - 1, col))
                row = max(0, min(rows - 1, row))
                plot_chars[row][col] = "*"

            # Add axes
            x_axis_row = (
                int((0 - y_min) / y_range * (rows - 1)) if y_min <= 0 <= y_max else -1
            )
            y_axis_col = (
                int((0 - x_min) / (x_max - x_min) * (cols - 1))
                if x_min <= 0 <= x_max
                else -1
            )

            ascii_plot = []
            for r in range(rows):
                line = []
                for c in range(cols):
                    if r == x_axis_row and c == y_axis_col:
                        line.append("+")
                    elif r == x_axis_row:
                        line.append("-")
                    elif c == y_axis_col:
                        line.append("|")
                    else:
                        line.append(plot_chars[rows - 1 - r][c])
                ascii_plot.append("".join(line))

            plot_text = "\n".join(ascii_plot)
            return EvalResult(ok=True, result=f"ASCII plot:\n{plot_text}")

        else:
            # Matplotlib plot
            import numpy as np

            x_vals = np.linspace(x_min, x_max, points)
            y_vals = f(x_vals)

            plt.figure()
            plt.plot(x_vals, y_vals)
            plt.xlabel(variable)
            plt.ylabel("f(" + variable + ")")
            plt.title(f"Plot of {expression}")
            plt.grid(True)
            plt.show()

            return EvalResult(ok=True, result="Plot displayed")

    except (ValueError, TypeError, AttributeError) as e:
        return EvalResult(ok=False, error=f"Plotting error: {e}")
    except ImportError:
        return EvalResult(ok=False, error="NumPy required for plotting")
    except (ZeroDivisionError, OverflowError) as e:
        return EvalResult(ok=False, error=f"Numerical error in plotting: {e}")
    except Exception as e:
        try:
            from .logging_config import get_logger

            logger = get_logger("plotting")
            logger.error(f"Unexpected plotting error: {e}", exc_info=True)
        except ImportError:
            pass
        return EvalResult(ok=False, error="Plotting failed unexpectedly")
