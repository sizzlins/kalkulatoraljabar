"""Integration tests for end-to-end worker + main process workflows."""

import unittest
from kalkulator_pkg.worker import evaluate_safely
from kalkulator_pkg.solver import solve_single_equation_cli
from kalkulator_pkg.parser import preprocess


class TestWorkerIntegration(unittest.TestCase):
	"""Test worker evaluation end-to-end."""

	def test_simple_evaluation(self):
		result = evaluate_safely("2+2")
		self.assertTrue(result.get("ok"))
		self.assertIn("result", result)

	def test_variable_evaluation(self):
		result = evaluate_safely("x+1")
		self.assertTrue(result.get("ok"))
		# Should have x as free symbol
		self.assertIn("x", result.get("free_symbols", []))

	def test_function_evaluation(self):
		result = evaluate_safely("sin(0)")
		self.assertTrue(result.get("ok"))
		# Should evaluate to 0
		self.assertEqual(result.get("result"), "0")


class TestSolverIntegration(unittest.TestCase):
	"""Test equation solving end-to-end."""

	def test_linear_solve(self):
		result = solve_single_equation_cli("x + 1 = 0")
		self.assertTrue(result.get("ok"))
		self.assertEqual(result.get("type"), "equation")
		self.assertIsNotNone(result.get("exact"))


class TestPreprocessingIntegration(unittest.TestCase):
	"""Test preprocessing pipeline."""

	def test_complex_preprocessing(self):
		# Test multiple transformations together
		result = preprocess("2² + 50% * √(4)")
		self.assertIn("**", result)
		self.assertIn("/100", result)
		self.assertIn("sqrt", result)


if __name__ == "__main__":
	unittest.main()

