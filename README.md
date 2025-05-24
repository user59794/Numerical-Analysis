
# Numerical Methods Calculator

This is a Streamlit-based web application designed for performing numerical method calculations. The calculator allows users to explore a variety of numerical methods, including error calculation, basic arithmetic operations, Taylor series approximation, and root finding methods.

## Features
- **Error Calculation**: Compute absolute and relative errors between approximate and true values.
- **Basic Operations**: Perform arithmetic operations (addition, subtraction, multiplication, division) with error analysis.
- **Taylor Series Approximation**: Approximate a given function using a Taylor series and visualize the comparison between the true and approximate values.
- **Root Finding Methods**: Solve nonlinear equations using different methods (Bisection, False Position, Newton-Raphson, Secant, Fixed Point).

## Installation

### Prerequisites
Ensure you have Python 3.6+ installed. You also need to install the following libraries:
- `streamlit`
- `numpy`
- `sympy`
- `pandas`
- `matplotlib`
- `mpmath`
- `reportlab`

You can install them using pip:

```bash
pip install streamlit numpy sympy pandas matplotlib mpmath reportlab
```

### Running the App
To run the application, navigate to the directory containing the script and use the following command:

```bash
streamlit run NumericalMethodsCalculator.py
```

This will open the application in your default web browser.

## Functionality

### 1. **Error Calculation**
- **Input**: Approximate value and true value (as a mathematical expression).
- **Output**: Absolute error and relative error between the values.

### 2. **Basic Operations**
- **Input**: A set of numbers for performing arithmetic operations (addition, subtraction, multiplication, and division).
- **Output**: Results of the operations along with absolute and relative errors.

### 3. **Taylor Series Approximation**
- **Input**: A function, a base point (`x₀`), target point (`x`), and the number of terms in the Taylor series.
- **Output**: Taylor series expansion, approximate value, true value, and error. A graph compares the function and the Taylor series approximation.

### 4. **Root Finding**
- **Input**: A function and initial guesses (and additional inputs depending on the method).
- **Methods**: 
  - **Bisection Method**: Iteratively narrows down the interval that contains the root.
  - **False Position Method**: Similar to the Bisection method but uses linear interpolation.
  - **Newton-Raphson Method**: Uses the derivative of the function to approximate the root.
  - **Secant Method**: An approximation to the Newton-Raphson method using two points.
  - **Fixed Point Iteration**: Uses a reformulation of the function for iterative root finding.
- **Output**: The root approximation, iteration history, and a line chart showing convergence.

## Example Usage

1. **Error Calculation**:
   - Enter the approximate value, such as `sqrt(2)` or `1.41`.
   - Enter the true value, like `2**0.5`.
   - Press "Submit" to get the absolute and relative error.

2. **Basic Operations**:
   - Enter a set of numbers (e.g., `2`, `3`, `4`).
   - Select arithmetic operations like addition or multiplication.
   - View the results with associated error calculations.

3. **Taylor Series Approximation**:
   - Input a function, such as `sin(x) + log(x + 1)`.
   - Set a base point (`x₀`) and the target value (`x`).
   - View the Taylor series approximation and the graph comparing it to the true function.

4. **Root Finding**:
   - Choose a root finding method, such as Bisection.
   - Enter the function and boundary values (for Bisection, enter `a` and `b`).
   - View the root approximation and iteration chart.

## Contributions

Feel free to fork the repository and submit pull requests if you wish to contribute improvements. We welcome any suggestions or improvements to the methods or UI/UX.

## License

This project is licensed under the **GNU General Public License, Version 3 (GPL-3.0)**. You are free to modify and distribute the code, but it must remain open-source under the same license.

### Full License Text

The full text of the GNU General Public License Version 3 can be found below:

---

### GNU GENERAL PUBLIC LICENSE  
Version 3, 29 June 2007  

**Copyright (C) 2007 Free Software Foundation, Inc.**  
Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

### Preamble

The GNU General Public License is a free, copyleft license for software and other kinds of works. The licenses for most software and other practical works are designed to take away your freedom to share and change the works. By contrast, the GNU General Public License is intended to guarantee your freedom to share and change all versions of a program—to make sure it remains free software for all its users.

When we speak of free software, we are referring to freedom, not price. To understand the concept, you should think of “free” as in “free speech,” not as in “free beer.”

**You may:**
- Copy, modify, and distribute the software, provided you adhere to the terms of this license.

**You may not:**
- Impose any further restrictions on the rights granted by the GPL.

For full details, you can read the full GPLv3 license [here](LICENSE).

---
