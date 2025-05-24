import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sympy.abc import x
from mpmath import mp
import itertools
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
    approx_expr = st.text_input("Approximate value (e.g., sqrt(2))",value="1.41", key="approx_expr")
    true_expr = st.text_input("True value (e.g., 2**0.5)", value="sqrt(2)", key="true_expr")

    if st.button("Submit Error Calculation"):
        try:
            approx_val = float(sp.N(sp.sympify(approx_expr, locals=safe_dict)))
            true_val = float(sp.N(sp.sympify(true_expr, locals=safe_dict)))
            abs_error = abs(true_val - approx_val)
            rel_error = abs_error / abs(true_val) if true_val != 0 else float("inf")

            st.write(f"🔸 Approximate: `{approx_val}`")
            st.write(f"🔸 True: `{true_val}`")
            st.write(f"🔸 Absolute Error: `{abs_error}`")
            st.write(f"🔸 Relative Error: `{rel_error}`")
        except Exception as e:
            st.error(f"Invalid input: {e}")

# Tab 2: Basic Operations
elif tab_selection == "Basic Operations":
    st.header("Basic Arithmetic Operations")

    # Select number of inputs
    num_inputs = st.slider("Select the number of inputs", 2, 5, 3)  # default 3 inputs
    inputs = []
    
    # Create input fields based on selected number of inputs
    for i in range(num_inputs):
        inputs.append(st.text_input(f"Input number {i+1}", value="2"))

    if st.button("Submit Arithmetic"):
        # Validate inputs
        valid_inputs = True
        numbers = []

        for expr in inputs:
            if not is_valid_number(expr):
                st.error(f"Error: One of the inputs is not valid. Please enter valid numbers.")
                valid_inputs = False
                break
            else:
                numbers.append(float(sp.N(sp.sympify(expr, locals=safe_dict))))

        if valid_inputs:
            # Perform operations based on the number of inputs
            sum_value = sum(numbers)
            st.write(f"Sum: `{sum_value}`")
            # Calculate errors
            abs_error_sum = abs(sum_value)
            rel_error_sum = abs_error_sum / abs(sum_value) if sum_value != 0 else float("inf")
            st.write(f"Absolute Error (Sum): `{abs_error_sum}`")
            st.write(f"Relative Error (Sum): `{rel_error_sum}`")

            # Subtraction operation
            diff_value = numbers[0]
            for num in numbers[1:]:
                diff_value -= num
            st.write(f"Difference: `{diff_value}`")
            # Calculate errors
            abs_error_diff = abs(diff_value)
            rel_error_diff = abs_error_diff / abs(diff_value) if diff_value != 0 else float("inf")
            st.write(f"Absolute Error (Difference): `{abs_error_diff}`")
            st.write(f"Relative Error (Difference): `{rel_error_diff}`")

            # Multiplication operation
            prod_value = np.prod(numbers)
            st.write(f"Product: `{prod_value}`")
            # Calculate errors
            abs_error_prod = abs(prod_value)
            rel_error_prod = abs_error_prod / abs(prod_value) if prod_value != 0 else float("inf")
            st.write(f"Absolute Error (Product): `{abs_error_prod}`")
            st.write(f"Relative Error (Product): `{rel_error_prod}`")

            # Division operation
            if 0 not in numbers[1:]:
                div_value = numbers[0]
                for num in numbers[1:]:
                    div_value /= num
                st.write(f"Division: `{div_value}`")
                # Calculate errors
                abs_error_div = abs(div_value)
                rel_error_div = abs_error_div / abs(div_value) if div_value != 0 else float("inf")
                st.write(f"Absolute Error (Division): `{abs_error_div}`")
                st.write(f"Relative Error (Division): `{rel_error_div}`")
            else:
                st.write("Division: Undefined (cannot divide by zero)")

            # Calculate maximum product from combinations
            max_product = float('-inf')  # Initial value for maximum product
            for r in range(2, len(numbers) + 1):  # From 2-item combinations upwards
                # Get combinations of different numbers of inputs
                for combination in itertools.combinations(numbers, r):
                    product = np.prod(combination)
                    if product > max_product:
                        max_product = product

            st.subheader("Maximum Product of Combinations")
            st.write(f"Maximum product from combinations: `{max_product}`")

# Tab 3: Taylor Series Approximation
elif tab_selection == "Taylor Series Approximation":
    st.header("Taylor Series Approximation")

    expr = st.text_input("Function f(x):", "sin(x) + log(x + 1)")
    x0 = st.number_input("x₀ (base point)", value=0.0)
    x_val = st.number_input("Target x", value=1.0)
    n = st.slider("Number of terms", 1, 20, 5)

    if st.button("Submit Taylor Approximation"):
        try:
            f_expr = sp.sympify(expr, locals=safe_dict)
            f = sp.lambdify(x, f_expr, modules=["numpy"])
            taylor = f_expr.series(x, x0, n=n).removeO()
            f_taylor = sp.lambdify(x, taylor, modules=["numpy"])

            approx_val = f_taylor(x_val)
            real_val = f(x_val)
            st.code(f"Taylor series: {taylor}")
            st.write(f"Approximate value: `{approx_val}`")
            st.write(f"True value: `{real_val}`")
            st.write(f"Absolute error: `{abs(real_val - approx_val)}`")

            # Plot results
            x_range = np.linspace(x0 - 2, x0 + 2, 400)
            y_true = f(x_range)
            y_approx = f_taylor(x_range)

            fig, ax = plt.subplots()
            ax.plot(x_range, y_true, label="f(x)", linewidth=2)
            ax.plot(x_range, y_approx, label=f"Taylor (n={n})", linestyle="--")
            ax.axvline(x_val, color="gray", linestyle=":")
            ax.set_title("Function and Taylor Series Approximation")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.legend()
            st.pyplot(fig)

            # Comparison Table
            sample_xs = np.linspace(x0 - 1, x0 + 1, 5)
            table_data = {
                "x": sample_xs,
                "f(x)": [f(val) for val in sample_xs],
                f"Taylor (n={n})": [f_taylor(val) for val in sample_xs],
                "Abs Error": [abs(f(val) - f_taylor(val)) for val in sample_xs],
            }
            st.subheader("Comparison Table")
            st.table(table_data)

        except Exception as e:
            st.error(f"Error in parsing function: {e}")

# Tab 4: Root Finding
if tab_selection == "Root Finding":
    st.header("Root Finding Methods")
    method = st.selectbox("Select method", ["Bisection", "False Position", "Newton-Raphson", "Secant", "Fixed Point"])

    # Common function input for all methods
    func_str = st.text_input("Function f(x):", "x**3 - x - 2")

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
            a = st.number_input("a (left boundary)", value=1.0)
            b = st.number_input("b (right boundary)", value=2.0)
        else:
            # For Newton-Raphson, Secant, and Fixed Point, need initial guesses
            x0 = st.number_input("Initial Guess (x₀)", value=1.5)
            if method == "Secant":
                x1 = st.number_input("Second Initial Guess (x₁)", value=1.6)
            elif method == "Fixed Point":
                g_str = st.text_input("g(x) (e.g., sqrt(x + 2))", "(x + 2)**(1/3)")
                try:
                    g_expr = sp.sympify(g_str, locals=safe_dict)
                    g = sp.lambdify(x, g_expr, "numpy")
                except:
                    st.error("Invalid g(x) expression.")
                    g = None

        tol = st.number_input("Tolerance", value=1e-5, format="%.1e")
        max_iter = st.number_input("Max Iterations", value=50, step=1)

        if st.button("Submit Root Finding"):
            history = []
            try:
                # Root finding logic based on method selected
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
                        x_new = x - f(x)/df(x)
                        history.append(x_new)
                        if abs(x_new - x) < tol:
                            break
                        x = x_new
                elif method == "Secant":
                    for _ in range(int(max_iter)):
                        x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
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
