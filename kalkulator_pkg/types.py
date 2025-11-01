"""Type definitions and result dataclasses for consistent API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class EvalResult:
	"""Result of evaluating a mathematical expression."""
	ok: bool
	result: Optional[str] = None
	approx: Optional[str] = None
	free_symbols: Optional[List[str]] = None
	error: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		d = {"ok": self.ok}
		if self.result is not None:
			d["result"] = self.result
		if self.approx is not None:
			d["approx"] = self.approx
		if self.free_symbols is not None:
			d["free_symbols"] = self.free_symbols
		if self.error is not None:
			d["error"] = self.error
		return d


@dataclass
class SolveResult:
	"""Result of solving an equation."""
	ok: bool
	type: str  # "equation", "pell", "identity_or_contradiction", "multi_isolate", "system"
	error: Optional[str] = None
	# For equation type
	exact: Optional[List[str]] = None
	approx: Optional[List[Optional[str]]] = None
	# For pell type
	solution: Optional[str] = None
	# For multi_isolate type
	solutions: Optional[Dict[str, List[str]]] = None
	# For system type
	system_solutions: Optional[List[Dict[str, str]]] = None

	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		d = {"ok": self.ok, "type": self.type}
		if self.error is not None:
			d["error"] = self.error
		if self.exact is not None:
			d["exact"] = self.exact
		if self.approx is not None:
			d["approx"] = self.approx
		if self.solution is not None:
			d["solution"] = self.solution
		if self.solutions is not None:
			d["solutions"] = self.solutions
		if self.system_solutions is not None:
			d["solutions"] = self.system_solutions
		return d


@dataclass
class InequalityResult:
	"""Result of solving an inequality."""
	ok: bool
	type: str = "inequality"
	error: Optional[str] = None
	solutions: Optional[Dict[str, str]] = None

	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		d = {"ok": self.ok, "type": self.type}
		if self.error is not None:
			d["error"] = self.error
		if self.solutions is not None:
			d["solutions"] = self.solutions
		return d


class ValidationError(Exception):
	"""Raised when input validation fails."""
	def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
		self.message = message
		self.code = code
		super().__init__(self.message)

	def __str__(self) -> str:
		return self.message


class ParseError(Exception):
	"""Raised when parsing fails."""
	def __init__(self, message: str, code: str = "PARSE_ERROR"):
		self.message = message
		self.code = code
		super().__init__(self.message)

	def __str__(self) -> str:
		return self.message


class SolverError(Exception):
	"""Raised when solving fails."""
	def __init__(self, message: str, code: str = "SOLVER_ERROR", transient: bool = False):
		self.message = message
		self.code = code
		self.transient = transient
		super().__init__(self.message)

	def __str__(self) -> str:
		return self.message

