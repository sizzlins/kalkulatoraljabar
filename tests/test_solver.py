"""Unit tests for solver module."""

import unittest
from kalkulator_pkg.solver import (
	solve_single_equation_cli,
	solve_inequality_cli,
	is_pell_equation_from_eq,
)
import sympy as sp


class TestEquationSolving(unittest.TestCase):
	"""Test equation solving functions."""

	def test_linear_equation(self):
		result = solve_single_equation_cli("x + 1 = 0")
		self.assertTrue(result.get("ok"))
		self.assertEqual(result.get("type"), "equation")

	def test_quadratic_equation(self):
		result = solve_single_equation_cli("x^2 - 1 = 0")
		self.assertTrue(result.get("ok"))
		self.assertIsNotNone(result.get("exact"))

	def test_identity(self):
		result = solve_single_equation_cli("1 = 1")
		self.assertTrue(result.get("ok"))
		self.assertEqual(result.get("type"), "identity_or_contradiction")

	def test_contradiction(self):
		result = solve_single_equation_cli("1 = 0")
		self.assertTrue(result.get("ok"))
		self.assertEqual(result.get("type"), "identity_or_contradiction")


class TestInequalitySolving(unittest.TestCase):
	"""Test inequality solving."""

	def test_simple_inequality(self):
		result = solve_inequality_cli("x > 0")
		self.assertTrue(result.get("ok"))
		self.assertEqual(result.get("type"), "inequality")

	def test_chained_inequality(self):
		result = solve_inequality_cli("0 < x < 5")
		self.assertTrue(result.get("ok"))


class TestPellEquation(unittest.TestCase):
	"""Test Pell equation detection."""

	def test_pell_detection(self):
		eq = sp.Eq(sp.symbols("x")**2 - 2*sp.symbols("y")**2, 1)
		self.assertTrue(is_pell_equation_from_eq(eq))


if __name__ == "__main__":
	unittest.main()

