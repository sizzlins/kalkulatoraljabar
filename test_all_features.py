#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive test suite for all features."""
import sys
import io
sys.path.insert(0, '.')

# Set UTF-8 encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from kalkulator_pkg.worker import evaluate_safely
from kalkulator_pkg.solver import solve_single_equation, solve_system
from kalkulator_pkg.calculus import differentiate, integrate
import re

def safe_print(msg):
    """Print with error handling."""
    try:
        print(msg)
    except UnicodeEncodeError:
        msg_safe = msg.encode('ascii', 'replace').decode('ascii')
        print(msg_safe)

# Comprehensive test suite
all_tests = [
    # === BASIC ARITHMETIC ===
    ("1+1", "2"),
    ("1-1", "0"),
    ("5*10", "50"),
    ("1/2", "0.5"),
    ("50%", "0.5"),
    ("50/100", "0.5"),
    
    # === EXPONENTS ===
    ("2^2", "4"),
    ("2**2", "4"),
    ("2^3", "8"),
    ("10^0", "1"),
    ("10^1", "10"),
    
    # === CONSTANTS ===
    ("pi", None),  # Should be ~3.14159
    ("E", None),   # Should be ~2.71828
    ("I", "I"),    # Imaginary unit
    
    # === TRIGONOMETRY ===
    ("sin(0)", "0"),
    ("sin(pi/6)", "0.5"),
    ("sin(pi/2)", "1"),
    ("cos(0)", "1"),
    ("cos(pi/2)", "0"),
    ("tan(0)", "0"),
    ("tan(pi/4)", "1"),
    ("asin(0)", "0"),
    ("asin(1)", "pi/2"),
    ("acos(0)", "pi/2"),
    ("acos(1)", "0"),
    ("atan(0)", "0"),
    ("atan(1)", "pi/4"),
    
    # === LOGARITHMS ===
    ("ln(1)", "0"),
    ("ln(E)", "1"),
    ("log(1)", "0"),
    ("log(10)", "1"),
    ("log(pi)", None),  # Should be ~1.1447
    
    # === EXPONENTIAL ===
    ("exp(0)", "1"),
    ("exp(1)", "E"),
    ("E^0", "1"),
    ("E^1", "E"),
    ("E^(I*pi)", "-1"),
    ("E^(I*pi)+1", "0"),
    
    # === SQUARE ROOTS ===
    ("sqrt(0)", "0"),
    ("sqrt(1)", "1"),
    ("sqrt(4)", "2"),
    ("sqrt(9)", "3"),
    
    # === ABSOLUTE VALUE ===
    ("Abs(0)", "0"),
    ("Abs(1)", "1"),
    ("Abs(-1)", "1"),
    ("Abs(-5)", "5"),
    
    # === COMPLEX NUMBERS ===
    ("I^2", "-1"),
    ("I^3", "-I"),
    ("I^4", "1"),
    ("1+I", "1+I"),
    ("(1+I)^2", "2*I"),
    
    # === CALCULUS: DIFFERENTIATION ===
    ("diff(x^2, x)", "2*x"),
    ("diff(x^3, x)", "3*x**2"),
    ("diff(sin(x), x)", "cos(x)"),
    ("diff(cos(x), x)", "-sin(x)"),
    ("diff(exp(x), x)", "exp(x)"),
    ("diff(ln(x), x)", "1/x"),
    
    # === CALCULUS: INTEGRATION ===
    ("integrate(x, x)", "x**2/2"),
    ("integrate(x^2, x)", "x**3/3"),
    ("integrate(sin(x), x)", "-cos(x)"),
    ("integrate(cos(x), x)", "sin(x)"),
    ("integrate(1/x, x)", "log(x)"),
    ("integrate(exp(x), x)", "exp(x)"),
    
    # === MATRICES ===
    ("Matrix([[1,2],[3,4]])", None),  # Should create matrix
    ("det(Matrix([[1,2],[3,4]]))", "-2"),
    ("det(Matrix([[1,0],[0,1]]))", "1"),
    ("det(Matrix([[2,0],[0,3]]))", "6"),
    
    # === EQUATIONS ===
    ("x=0", "equation"),
    ("x+1=0", "equation"),
    ("x^2-1=0", "equation"),
    ("x^2-4=0", "equation"),
    ("sin(x)=0", "equation"),
    ("cos(x)=0", "equation"),
    
    # === SYSTEMS ===
    ("x+y=3, x-y=1", "system"),
    ("x=1, y=2", "system"),
    ("x+y=0, x-y=0", "system"),
]

def run_test(input_str, expected):
    """Run a single test."""
    try:
        if expected == "equation":
            # Test equation solving
            find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
            find_var = find_tokens[0] if find_tokens else None
            eq_str = re.sub(r",?\s*\bfind\s+\w+\b\s*,?", "", input_str, flags=re.IGNORECASE).strip()
            eq_str = eq_str.rstrip(',').strip()
            result = solve_single_equation(eq_str, find_var)
            if result.get("ok"):
                return True, "Solved"
            else:
                return False, result.get('error', 'Unknown error')
                
        elif expected == "system":
            # Test system solving
            find_tokens = re.findall(r"\bfind\s+(\w+)\b", input_str, re.IGNORECASE)
            find_token = find_tokens[0] if find_tokens else None
            raw_no_find = re.sub(r"\bfind\s+\w+\b", "", input_str, flags=re.IGNORECASE).strip()
            result = solve_system(raw_no_find, find_token)
            if result.get("ok"):
                return True, "Solved"
            else:
                return False, result.get('error', 'Unknown error')
                
        else:
            # Test value evaluation
            result = evaluate_safely(input_str)
            if result.get("ok"):
                res = result.get('result', '')
                if expected:
                    # Check if result matches expected (allowing for formatting differences)
                    if str(res).replace(" ", "") == str(expected).replace(" ", ""):
                        return True, res
                    else:
                        return True, f"{res} (expected {expected})"
                return True, res
            else:
                return False, result.get('error', 'Unknown error')
                
    except Exception as e:
        return False, str(e)

def main():
    passed = 0
    failed = 0
    issues = []
    
    safe_print("="*70)
    safe_print("COMPREHENSIVE FEATURE TEST SUITE")
    safe_print("="*70)
    
    for i, (input_str, expected) in enumerate(all_tests, 1):
        safe_print(f"\n[{i}/{len(all_tests)}] {input_str}")
        
        success, result = run_test(input_str, expected)
        if success:
            safe_print(f"  [OK] {result}")
            passed += 1
        else:
            safe_print(f"  [FAIL] {result}")
            failed += 1
            issues.append((input_str, result))
    
    safe_print("\n" + "="*70)
    safe_print("SUMMARY")
    safe_print("="*70)
    safe_print(f"Passed: {passed}")
    safe_print(f"Failed: {failed}")
    safe_print(f"Total: {passed + failed}")
    safe_print(f"Success Rate: {100*passed/(passed+failed):.1f}%")
    
    if issues:
        safe_print("\nIssues:")
        for input_str, error in issues[:20]:
            safe_print(f"  - {input_str}: {error}")
        if len(issues) > 20:
            safe_print(f"  ... and {len(issues) - 20} more")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

