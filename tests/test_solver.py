"""Unit tests for solver module."""

import unittest

import sympy as sp

from kalkulator_pkg.solver import (
    is_pell_equation_from_eq,
    solve_inequality,
    solve_single_equation,
)


class TestEquationSolving(unittest.TestCase):
    """Test equation solving functions."""

    def test_linear_equation(self):
        result = solve_single_equation("x + 1 = 0")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "equation")
        # Verify the actual solution
        exact_solutions = result.get("exact", [])
        self.assertIn("-1", exact_solutions)

    def test_quadratic_equation(self):
        result = solve_single_equation("x^2 - 1 = 0")
        self.assertTrue(result.get("ok"))
        self.assertIsNotNone(result.get("exact"))
        # Verify the actual solutions
        exact_solutions = result.get("exact", [])
        self.assertIn("1", exact_solutions)
        self.assertIn("-1", exact_solutions)

    def test_identity(self):
        result = solve_single_equation("1 = 1")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "identity_or_contradiction")

    def test_contradiction(self):
        result = solve_single_equation("1 = 0")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "identity_or_contradiction")


class TestInequalitySolving(unittest.TestCase):
    """Test inequality solving."""

    def test_simple_inequality(self):
        result = solve_inequality("x > 0")
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("type"), "inequality")

    def test_chained_inequality(self):
        result = solve_inequality("0 < x < 5")
        self.assertTrue(result.get("ok"))


class TestPellEquation(unittest.TestCase):
    """Test Pell equation detection."""

    def test_pell_detection(self):
        # Create symbols once to ensure they're the same instance
        x, y = sp.symbols("x y")
        eq = sp.Eq(x**2 - 2 * y**2, 1)
        self.assertTrue(is_pell_equation_from_eq(eq))


if __name__ == "__main__":
    unittest.main()
