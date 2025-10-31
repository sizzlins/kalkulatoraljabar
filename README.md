# kalkulatoraljabar

it is a kalkulator, neat thing it does is it can find x, and uses sympy and all its functions and constant
type or find equations like how youd typein cheats on source command or something like that

just run and type some equations

or 

put this in your terminal then run

pyinstaller --onefile --console --collect-all sympy kalkulator.py



def print_help_text():
    help_text = help_text = f""" version {VERSION}

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
- In REPL: help, quit, exit

Supported constants & functions:
- Constants: pi, E, I
- Functions: sqrt(), sin(), cos(), tan(), asin(), acos(), atan(), log()/ln(), exp(), Abs()

Input conveniences:
- '^' → '**'   ;  '²','³' → **2, **3
- '50%' → (50/100) ; '√' → sqrt(...)
- Implicit multiplication allowed (2x → 2*x)
- Balanced-parentheses check

Example inputs (organized by category)

Basic arithmetic
- 1+1
- 1-1
- 5*10
- 1/2

Fractions & percent
- 50%            (interpreted as 50/100)
- 50/100

Exponents & roots
- 2^2
- 2**2
- 2²
- √(2)
- sqrt(16)

Constants & simple functions
- pi
- E
- I
- sin(pi/6)
- sin(pi/2)
- cos(0)
- tan(pi/4)

Trigonometry (equations)
- 2*sin(x) + sqrt(3) = 0
- 3*tan(x) + sqrt(3) = 0
- 2*sin(x)**2 - 1 = 0
- 3*sin(x) + 2 = 1

Complex numbers & logs
- ln(I)
- log(2)
- log(pi)
- E^(I*pi)
- E^(I*pi) + 1 = x, find x

Assignments & REPL examples
- a = 2, b = a+3
- a = 1, b = 2, c = a + b, find c
- r = 5, pi*r^2 = n, find n

Single linear / algebraic equations
- 1 = 1
- 1 = 0
- x + y = 3, x - y = 1   (also a small system example)

Systems of linear equations
- x + y = 3, x - y = 1
- 2v + w + x + y + z = 5,
  v + 2w + x + y + z = 5,
  v + w + 2x + y + z = 6,
  v + w + x + 2y + z = 7,
  v + w + x + y + 2z = 8,
  v + w + x + y + z = a, find a
sqrt(x)+y=7, x+sqrt(y)=11
Polynomials / algebraic roots
- 6*x^2 - 17*x + 1 = 0, find x
- x^3 - 4*x^2 - 9*x + 36 = 0, find x
- x^3 - 9*x + 36 = 0, find x

Complex algebra / tricky expressions
- pi + E + I + sqrt(2) + sin(pi/2) + cos(0) + tan(pi/4) + asin(1) + acos(0) + atan(1) + log(10) + ln(E) + exp(1) + Abs(-5)
- sin(1/x)**-1 = (sin(x)/1)**-1, find x

Preprocessing demonstration (how input is normalized)
- 2^2        (becomes 2**2)
- 2²         (becomes 2**2)
- 50%        (becomes (50/100))
- √(2)       (becomes sqrt(2))
- 2x         (becomes 2*x via implicit multiplication)

Tips:
- Use --json for automatic parsing.
- If a computation times out, simplify the expression (smaller exponents, fewer nested functions).
- Non-finite results (division by zero, infinity) are reported as errors.

Calculus & matrices (new):
- diff(x^3, x)
- integrate(sin(x), x)
- factor(x^3 - 1)
- expand((x+1)^3)
- Matrix([[1,2],[3,4]])
- det(Matrix([[1,2],[3,4]]))

Property of Muhammad Akhiel al Syahbana — 31/October/2025
"""
