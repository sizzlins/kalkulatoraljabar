"""Advanced function finding capabilities.

This module implements:
1. Constant detection (PSLQ/nsimplify) for symbolic recognition
2. High-precision parsing (Decimal/mpmath)
3. Sparse regression (LASSO/OMP)
4. Model selection (AIC/BIC)
5. Tolerance and validation utilities
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any

import sympy as sp

try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

from .config import (
    ABSOLUTE_TOLERANCE,
    CONSTANT_DETECTION_TOLERANCE,
    LASSO_LAMBDA,
    OMP_MAX_ITERATIONS,
    RELATIVE_TOLERANCE,
    RESIDUAL_THRESHOLD,
    USE_AIC_BIC,
)

# Set high precision for Decimal
getcontext().prec = 50

# Library of known constants for detection
KNOWN_CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "sqrt(2)": sp.sqrt(2),
    "sqrt(3)": sp.sqrt(3),
    "sqrt(5)": sp.sqrt(5),
    "log(2)": sp.log(2),
    "log(10)": sp.log(10),
    "ln(2)": sp.log(2),
    "ln(10)": sp.log(10),
    "gamma": sp.EulerGamma,
    "EulerGamma": sp.EulerGamma,
    "catalan": sp.Catalan,
    "Catalan": sp.Catalan,
}


def detect_symbolic_constant(
    value: float | Fraction | Decimal, tolerance: float = CONSTANT_DETECTION_TOLERANCE
) -> sp.Basic | None:
    """Detect if a numeric value is close to a known symbolic constant.
    
    Uses SymPy's nsimplify and direct comparison with known constants.
    
    Args:
        value: Numeric value to check
        tolerance: Relative tolerance for matching
        
    Returns:
        Symbolic constant if detected, None otherwise
    """
    if isinstance(value, Fraction):
        float_val = float(value)
    elif isinstance(value, Decimal):
        float_val = float(value)
    else:
        float_val = float(value)
    
    # First, try SymPy's nsimplify
    try:
        # Try to simplify to a known constant
        simplified = sp.nsimplify(
            float_val,
            tolerance=tolerance,
            full=True,
        )
        
        # Check if it matches any known constant
        for const_name, const_symbol in KNOWN_CONSTANTS.items():
            const_val = float(sp.N(const_symbol))
            if abs(float_val - const_val) / (abs(const_val) + 1e-10) < tolerance:
                return const_symbol
        
        # Check if simplified is close to the original
        if isinstance(simplified, (sp.Number, sp.Rational, sp.Integer)):
            simplified_val = float(sp.N(simplified))
            if abs(float_val - simplified_val) / (abs(simplified_val) + 1e-10) < tolerance:
                # Check if it's a known constant expression
                if abs(simplified_val - math.pi) < tolerance * abs(math.pi):
                    return sp.pi
                elif abs(simplified_val - math.e) < tolerance * abs(math.e):
                    return sp.E
                elif abs(simplified_val - math.sqrt(2)) < tolerance * abs(math.sqrt(2)):
                    return sp.sqrt(2)
                elif abs(simplified_val - math.sqrt(3)) < tolerance * abs(math.sqrt(3)):
                    return sp.sqrt(3)
    except (ValueError, TypeError, AttributeError):
        pass
    
    # Direct comparison with known constants
    for const_name, const_symbol in KNOWN_CONSTANTS.items():
        try:
            const_val = float(sp.N(const_symbol))
            if abs(float_val - const_val) / (abs(const_val) + 1e-10) < tolerance:
                return const_symbol
        except (ValueError, TypeError):
            continue
    
    return None


def parse_with_precision(
    val: str | float | int | Fraction | Decimal,
) -> Fraction | Decimal:
    """Parse input with appropriate precision based on format.
    
    Strategy:
    - Integers → Fraction
    - Fractions (a/b) → Fraction
    - Decimals with few places → Fraction (if exact)
    - High-precision decimals → Decimal
    - Very high precision → mpmath (if available)
    
    Args:
        val: Input value in various formats
        
    Returns:
        Fraction or Decimal with appropriate precision
    """
    # Already a Fraction
    if isinstance(val, Fraction):
        return val
    
    # Already a Decimal
    if isinstance(val, Decimal):
        return val
    
    # Integer
    if isinstance(val, int):
        return Fraction(val)
    
    # String parsing
    if isinstance(val, str):
        val = val.strip()
        
        # Try fraction format
        if '/' in val:
            try:
                parts = val.split('/')
                if len(parts) == 2:
                    num = int(parts[0].strip())
                    den = int(parts[1].strip())
                    return Fraction(num, den)
            except (ValueError, TypeError):
                pass
        
        # Try decimal format
        try:
            # Count decimal places
            if '.' in val:
                decimal_places = len(val.split('.')[1])
                # If few decimal places, try Fraction first
                if decimal_places <= 6:
                    try:
                        return Fraction(val)
                    except (ValueError, TypeError):
                        pass
                # Otherwise use Decimal for high precision
                return Decimal(val)
            else:
                # Integer string
                return Fraction(int(val))
        except (ValueError, TypeError):
            pass
    
    # Float - try to convert to Fraction if it's a simple decimal
    if isinstance(val, float):
        # Check if it's close to a simple fraction
        try:
            frac = Fraction(val).limit_denominator(10000)
            if abs(float(frac) - val) < 1e-10:
                return frac
        except (ValueError, OverflowError):
            pass
        
        # Otherwise use Decimal for precision
        return Decimal(str(val))
    
    raise ValueError(f"Cannot parse {val} with precision")


def is_exact_fit(
    computed: float | Fraction | Decimal,
    expected: float | Fraction | Decimal,
    abs_tol: float = ABSOLUTE_TOLERANCE,
    rel_tol: float = RELATIVE_TOLERANCE,
) -> bool:
    """Check if a computed value matches expected value within tolerances.
    
    Uses both absolute and relative tolerance checks.
    
    Args:
        computed: Computed value
        expected: Expected value
        abs_tol: Absolute tolerance
        rel_tol: Relative tolerance
        
    Returns:
        True if match within tolerances, False otherwise
    """
    # Convert to float for comparison
    comp_float = float(computed)
    exp_float = float(expected)
    
    # Absolute difference
    abs_diff = abs(comp_float - exp_float)
    
    # Relative difference
    rel_diff = abs_diff / (abs(exp_float) + 1e-10)
    
    # Check both tolerances
    return abs_diff < abs_tol or rel_diff < rel_tol


def calculate_residuals(
    computed_values: list[float | Fraction],
    expected_values: list[float | Fraction],
) -> tuple[list[float], float, float]:
    """Calculate residuals and statistics.
    
    Args:
        computed_values: List of computed function values
        expected_values: List of expected values
        
    Returns:
        Tuple of (residuals, max_residual, mean_squared_error)
    """
    residuals = []
    for comp, exp in zip(computed_values, expected_values):
        residuals.append(float(comp) - float(exp))
    
    max_residual = max(abs(r) for r in residuals)
    mse = sum(r * r for r in residuals) / len(residuals) if residuals else 0.0
    
    return residuals, max_residual, mse


def calculate_aic(
    n_params: int, n_samples: int, mse: float
) -> float:
    """Calculate Akaike Information Criterion (AIC).
    
    AIC = 2k - 2*ln(L) where k = number of parameters, L = likelihood.
    For least squares: AIC = n*ln(MSE) + 2k
    
    Args:
        n_params: Number of parameters in the model
        n_samples: Number of data points
        mse: Mean squared error
        
    Returns:
        AIC value (lower is better)
    """
    if mse <= 0:
        return float('inf')
    return n_samples * math.log(mse) + 2 * n_params


def calculate_bic(
    n_params: int, n_samples: int, mse: float
) -> float:
    """Calculate Bayesian Information Criterion (BIC).
    
    BIC = k*ln(n) - 2*ln(L) where k = parameters, n = samples, L = likelihood.
    For least squares: BIC = n*ln(MSE) + k*ln(n)
    
    Args:
        n_params: Number of parameters in the model
        n_samples: Number of data points
        mse: Mean squared error
        
    Returns:
        BIC value (lower is better)
    """
    if mse <= 0:
        return float('inf')
    return n_samples * math.log(mse) + n_params * math.log(n_samples)


def orthogonal_matching_pursuit(
    A: list[list[float | Fraction]],
    b: list[float | Fraction],
    max_nonzero: int | None = None,
    max_iterations: int = OMP_MAX_ITERATIONS,
) -> tuple[list[float], list[int]]:
    """Orthogonal Matching Pursuit for sparse regression.
    
    Greedy algorithm that iteratively selects the column of A that best
    matches the residual.
    
    Args:
        A: Design matrix (list of rows, each row is a list)
        b: Target vector
        max_nonzero: Maximum number of non-zero coefficients (default: min(n, m))
        max_iterations: Maximum iterations
        
    Returns:
        Tuple of (coefficients, selected_indices)
    """
    import numpy as np
    
    # Convert to numpy arrays
    A_arr = np.array([[float(x) for x in row] for row in A])
    b_arr = np.array([float(x) for x in b])
    
    n_samples, n_features = A_arr.shape
    
    if max_nonzero is None:
        max_nonzero = min(n_samples, n_features)
    
    # Initialize
    residual = b_arr.copy()
    selected = []
    coefficients = np.zeros(n_features)
    
    for iteration in range(min(max_nonzero, max_iterations)):
        # Find column with maximum correlation with residual
        correlations = np.abs(A_arr.T @ residual)
        correlations[selected] = -np.inf  # Don't reselect
        
        if np.max(correlations) < 1e-10:
            break  # No significant correlation
        
        new_idx = np.argmax(correlations)
        selected.append(new_idx)
        
        # Solve least squares with selected columns
        A_selected = A_arr[:, selected]
        coeffs_selected = np.linalg.lstsq(A_selected, b_arr, rcond=None)[0]
        
        # Update residual
        residual = b_arr - A_selected @ coeffs_selected
        
        # Check convergence
        if np.linalg.norm(residual) < 1e-10:
            break
    
    # Set coefficients
    for i, idx in enumerate(selected):
        coefficients[idx] = coeffs_selected[i]
    
    return coefficients.tolist(), selected


def lasso_regression(
    A: list[list[float | Fraction]],
    b: list[float | Fraction],
    lambda_reg: float = LASSO_LAMBDA,
    max_iterations: int = 1000,
) -> list[float]:
    """L1-regularized (LASSO) regression using coordinate descent.
    
    Minimizes: ||Ax - b||² + λ||x||₁
    
    Args:
        A: Design matrix
        b: Target vector
        lambda_reg: Regularization parameter
        max_iterations: Maximum iterations
        
    Returns:
        Coefficient vector
    """
    try:
        from sklearn.linear_model import Lasso
        
        # Convert to numpy arrays
        import numpy as np
        A_arr = np.array([[float(x) for x in row] for row in A])
        b_arr = np.array([float(x) for x in b])
        
        # Use sklearn's LASSO
        lasso = Lasso(alpha=lambda_reg, max_iter=max_iterations, fit_intercept=False)
        lasso.fit(A_arr, b_arr)
        return lasso.coef_.tolist()
    except ImportError:
        # Fallback: simple coordinate descent implementation
        import numpy as np
        
        A_arr = np.array([[float(x) for x in row] for row in A])
        b_arr = np.array([float(x) for x in b])
        
        n_samples, n_features = A_arr.shape
        coefficients = np.zeros(n_features)
        
        for iteration in range(max_iterations):
            old_coeffs = coefficients.copy()
            
            for j in range(n_features):
                # Coordinate descent update
                r_j = b_arr - A_arr @ coefficients + A_arr[:, j] * coefficients[j]
                a_j = A_arr[:, j]
                
                # Soft thresholding
                numerator = a_j @ r_j
                denominator = a_j @ a_j
                
                if denominator > 1e-10:
                    z_j = numerator / denominator
                    coefficients[j] = np.sign(z_j) * max(0, abs(z_j) - lambda_reg / (2 * denominator))
            
            # Check convergence
            if np.linalg.norm(coefficients - old_coeffs) < 1e-6:
                break
        
        return coefficients.tolist()

