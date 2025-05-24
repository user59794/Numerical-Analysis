import streamlit as st
import numpy as np
import sympy as sp
import itertools
from sympy.abc import x
import matplotlib.pyplot as plt
from mpmath import mp

# Safe dictionary for sympify
safe_dict = {
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'log': sp.log,
    'ln': sp.ln,
    'exp': sp.exp,
    'sqrt': sp.sqrt,
    'abs': sp.Abs,
    'pi': sp.pi,
    'e': sp.E,
    'x': x
}

# Set high precision for mpmath
mp.dps = 50  # set decimal places to 50 for high precision

# Streamlit page configuration
st.set_page_config("Numerical Calculator", layout="centered")
st.title("🧲 Numerical Methods Calculator")

# Function to validate if input is a valid number
def is_valid_number(expr):
    try:
        # Try to convert the input to a valid expression
        sp.sympify(expr, locals=safe_dict)
        return True
    except:
        return False

# Tab Navigation Using Radio Button (Using `st.radio` to simulate Tabs)
tab_selection = st.radio("Select a section", ["Error Calculation", "Basic Operations", "Taylor Series Approximation", "Root Finding"])


# Tab 1: Error Calculation
if tab_selection == "Error Calculation":
    st.header("Absolute & Relative Error")
    approx_expr = st.text_input("Approximate value (e.g., sqrt(2))", value="")
    true_expr = st.text_input("True value (e.g., 2**0.5)", value="")

# Tab 2: Basic Operations
elif tab_selection == "Basic Operations":
    st.header("Basic Arithmetic Operations")

    # Select number of inputs
    num_inputs = st.slider("Select the number of inputs", 2, 5, 3)  # default 3 inputs
    inputs = []
    
    # Create input fields based on selected number of inputs
    for i in range(num_inputs):
        inputs.append(st.text_input(f"Input number {i+1}", value=""))

# Tab 3: Taylor Series Approximation
elif tab_selection == "Taylor Series Approximation":
    st.header("Taylor Series Approximation")

    expr = st.text_input("Function f(x):", value="")
    x0 = st.number_input("x₀ (base point)", value=0.0)
    x_val = st.number_input("Target x", value=1.0)
    n = st.slider("Number of terms", 1, 20, 5)  # default 5 terms

# Tab 4: Root Finding
if tab_selection == "Root Finding":
    st.header("Root Finding Methods")
    method = st.selectbox("Select method", ["Bisection", "False Position", "Newton-Raphson", "Secant", "Fixed Point"])

    # Common function input for all methods
    func_str = st.text_input("Function f(x):", value="x**3 - x - 2")

    try:
        # Parse the function
        f_expr = sp.sympify(func_str, locals=safe_dict)
        f = sp.lambdify(x, f_expr, "numpy")
        df_expr = sp.diff(f_expr, x)
        df = sp.lambdify(x, df_expr, "numpy")
    except:
        st.error("Error converting the function to a valid format.")
        f = None

    if f:
        # Bisection and False Position: Need two boundary points
        if method == "Bisection" or method == "False Position":
            a = st.number_input("a (left boundary)", value=0.0)
            b = st.number_input("b (right boundary)", value=1.0)
        else:
            # For Newton-Raphson, Secant, and Fixed Point, need initial guesses
            x0 = st.number_input("Initial Guess (x₀)", value=0.5)
            if method == "Secant":
                x1 = st.number_input("Second Initial Guess (x₁)", value=0.6)
            elif method == "Fixed Point":
                g_str = st.text_input("g(x) (e.g., sqrt(x + 2))", value="")
                try:
                    g_expr = sp.sympify(g_str, locals=safe_dict)
                    g = sp.lambdify(x, g_expr, "numpy")
                except:
                    st.error("Invalid g(x) expression.")
                    g = None

        tol = st.number_input("Tolerance", value=1e-5)
        max_iter = st.number_input("Max Iterations", value=50)

        # Root finding logic based on method selected
        if st.button("Submit Root Finding"):
            history = []
            try:
                if method == "Bisection":
                    for _ in range(int(max_iter)):
                        c = (a + b) / 2
                        history.append(c)
                        if abs(f(c)) < tol or abs(b - a) < tol:
                            break
                        if f(a) * f(c) < 0:
                            b = c
                        else:
                            a = c
                elif method == "False Position":
                    for _ in range(int(max_iter)):
                        c = b - f(b) * (b - a) / (f(b) - f(a))
                        history.append(c)
                        if abs(f(c)) < tol:
                            break
                        if f(a) * f(c) < 0:
                            b = c
                        else:
                            a = c
                elif method == "Newton-Raphson":
                    x = x0
                    for _ in range(int(max_iter)):
                        x_new = x - f(x) / df(x)
                        history.append(x_new)
                        if abs(x_new - x) < tol:
                            break
                        x = x_new
                elif method == "Secant":
                    for _ in range(int(max_iter)):
                        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
                        history.append(x2)
                        if abs(x2 - x1) < tol:
                            break
                        x0, x1 = x1, x2
                elif method == "Fixed Point" and g:
                    x = x0
                    for _ in range(int(max_iter)):
                        x_new = g(x)
                        history.append(x_new)
                        if abs(x_new - x) < tol:
                            break
                        x = x_new

                st.success(f"Approximate Root: {history[-1]}")
                st.line_chart(history)
                st.table({"Iteration": list(range(1, len(history) + 1)), "x": history})

            except Exception as e:
                st.error(f"Error in method: {e}")
