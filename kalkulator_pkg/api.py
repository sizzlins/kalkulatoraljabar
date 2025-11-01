"""Public API for Kalkulator - returns structured objects without side effects."""

from __future__ import annotations

from typing import Optional
from .types import EvalResult, SolveResult, InequalityResult, ValidationError
from .worker import evaluate_safely
from .solver import solve_single_equation_cli, solve_inequality_cli, handle_system_main
from .parser import preprocess
from .calculus import differentiate, integrate, matrix_determinant
from .plotting import plot_function


def evaluate(expression: str) -> EvalResult:
	"""Evaluate a mathematical expression.
	
	Args:
		expression: Mathematical expression string (e.g., "2+2", "sin(pi/2)")
		
	Returns:
		EvalResult with result, approximation, and free symbols
	"""
	data = evaluate_safely(expression)
	if not data.get("ok"):
		return EvalResult(ok=False, error=data.get("error") or "Unknown error")
	return EvalResult(
		ok=True,
		result=data.get("result"),
		approx=data.get("approx"),
		free_symbols=data.get("free_symbols")
	)


def solve_equation(equation: str, find_var: Optional[str] = None) -> SolveResult:
	"""Solve a single equation.
	
	Args:
		equation: Equation string (e.g., "x+1=0", "x^2-1=0")
		find_var: Optional variable to solve for (e.g., "x")
		
	Returns:
		SolveResult with solutions
	"""
	data = solve_single_equation_cli(equation, find_var)
	if not data.get("ok"):
		return SolveResult(ok=False, type="equation", error=data.get("error"))
	
	result_type = data.get("type", "equation")
	if result_type == "pell":
		return SolveResult(ok=True, type="pell", solution=data.get("solution"))
	elif result_type == "identity_or_contradiction":
		result_str = data.get("result", "")
		return SolveResult(ok=True, type="identity_or_contradiction", error=result_str)
	elif result_type == "multi_isolate":
		return SolveResult(
			ok=True,
			type="multi_isolate",
			solutions=data.get("solutions"),
			exact=None,  # multi_isolate uses solutions dict
			approx=data.get("approx")
		)
	else:  # equation
		return SolveResult(
			ok=True,
			type="equation",
			exact=data.get("exact"),
			approx=data.get("approx")
		)


def solve_inequality(inequality: str, find_var: Optional[str] = None) -> InequalityResult:
	"""Solve an inequality.
	
	Args:
		inequality: Inequality string (e.g., "x > 0", "1 < x < 5")
		find_var: Optional variable to solve for
		
	Returns:
		InequalityResult with solutions
	"""
	data = solve_inequality_cli(inequality, find_var)
	if not data.get("ok"):
		return InequalityResult(ok=False, error=data.get("error"))
	return InequalityResult(
		ok=True,
		solutions=data.get("solutions")
	)


def solve_system(equations: str, find_var: Optional[str] = None) -> SolveResult:
	"""Solve a system of equations.
	
	Args:
		equations: Comma-separated equations (e.g., "x+y=3, x-y=1")
		find_var: Optional variable to find in solutions
		
	Returns:
		SolveResult with system solutions
	"""
	data = handle_system_main(equations, find_var)
	if not data.get("ok"):
		return SolveResult(ok=False, type="system", error=data.get("error"))
	
	result_type = data.get("type", "system")
	if result_type == "system":
		return SolveResult(
			ok=True,
			type="system",
			system_solutions=data.get("solutions")
		)
	else:  # system_var
		return SolveResult(
			ok=True,
			type="equation",  # single variable result
			exact=data.get("exact"),
			approx=data.get("approx")
		)


def validate_expression(expression: str) -> tuple[bool, Optional[str]]:
	"""Validate an expression without evaluating it.
	
	Args:
		expression: Expression string to validate
		
	Returns:
		Tuple of (is_valid, error_message)
	"""
	try:
		preprocess(expression)
		return True, None
	except (ValueError, ValidationError) as e:
		return False, str(e)
	except Exception as e:
		return False, f"Validation error: {e}"


def diff(expression: str, variable: Optional[str] = None) -> EvalResult:
	"""Differentiate an expression.
	
	Args:
		expression: Expression to differentiate (e.g., "x^3")
		variable: Variable to differentiate with respect to (optional)
		
	Returns:
		EvalResult with derivative
	"""
	return differentiate(expression, variable)


def integrate_expr(expression: str, variable: Optional[str] = None) -> EvalResult:
	"""Integrate an expression.
	
	Args:
		expression: Expression to integrate (e.g., "sin(x)")
		variable: Variable to integrate with respect to (optional)
		
	Returns:
		EvalResult with integral
	"""
	return integrate(expression, variable)


def det(matrix_str: str) -> EvalResult:
	"""Calculate matrix determinant.
	
	Args:
		matrix_str: Matrix expression (e.g., "Matrix([[1,2],[3,4]])")
		
	Returns:
		EvalResult with determinant
	"""
	return matrix_determinant(matrix_str)


def plot(expression: str, variable: str = "x", 
		 x_min: float = -10, x_max: float = 10, ascii: bool = False) -> EvalResult:
	"""Plot a single-variable function.
	
	Args:
		expression: Function expression
		variable: Variable name
		x_min: Minimum x value
		x_max: Maximum x value
		ascii: Return ASCII plot instead of opening window
		
	Returns:
		EvalResult with plot data
	"""
	return plot_function(expression, variable, x_min, x_max, ascii=ascii)

