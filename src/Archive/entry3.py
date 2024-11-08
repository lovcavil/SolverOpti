import numpy as np
import matplotlib.pyplot as plt
# Initialize the counter as a global variable
function_calls = 0

def ode(t, y):
    """The differential equation dy/dt = cos(t)."""
    global function_calls
    function_calls += 1  # Increment the counter
    return np.cos(t)

def rk45_step(f, t, y, h, coeffs):
    """Performs a single step of the RK45 method with given coefficients."""
    a, b1, b2, c = coeffs
    
    k = []
    for i in range(6):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        k.append(ki)
    
    y4 = y + h * sum(b2[i] * k[i] for i in range(6))
    y5 = y + h * sum(b1[i] * k[i] for i in range(6))
    error = np.abs(y4 - y5)
    
    return y5, error

def solve_ode(f, t_span, y0, coeffs, h=0.00001, tol=1e-8):
    """Solves an ODE using an adaptive RK45 method with given coefficients."""
    global function_calls
    function_calls = 0  # Reset the counter at the start of each solve_ode call
    t0, tf = t_span
    t = t0
    y = y0
    times = [t]
    ys = [y]
    errors = []  # To store the error at each step
    
    while t < tf:
        h = min(h, tf - t)
        y_new, error = rk45_step(f, t, y, h, coeffs)
        
        if error < tol:
            print(f"h{h}")
            t += h
            y = y_new
            times.append(t)
            ys.append(y)
            errors.append(error)
        if error > 0:
            h *= (tol / error)**0.2 
        else :
            h=h * 2
    
    return np.array(times), np.array(ys), np.array(errors), function_calls

# RK45 coefficients Runge–Kutta–Fehlberg method has two methods of orders 5 and 4; it is sometimes dubbed RKF45 . Its extended Butcher Tableau is:
coeffs = (
    [0, 1/4, 3/8, 12/13, 1, 1/2],
    [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
    [25/216, 0, 1408/2565, 2197/4104, -1/5, 0],
    [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
)
# RK45 coefficientsCash and Karp have modified Fehlberg's original idea. The extended tableau for the Cash–Karp method is
coeffs1 = (
    [0, 1/5, 3/10, 3/5, 1, 7/8],
    [37/378, 0, 250/621, 125/594, 0, 512/1771],
    [	2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],
    [
        [],
        [1/5],
        [3/40,	9/40],
        [3/10,-9/10,6/5],
        [-11/54,	5/2,	-70/27,	35/27],
        [1631/55296,175/512,	575/13824,44275/110592,253/4096	]
    ]
)
# Solve the ODE for both tolerances
t_span = (0, 2)
y0 = 0
print("1")
# First run with tol=1e-7
times_1, ys_1, errors_1, total_function_calls_1 = solve_ode(ode, t_span, y0, coeffs, h=0.00001, tol=1e-7)
true_ys_1 = np.sin(times_1)
rmse_1 = np.sqrt(np.mean((ys_1 - true_ys_1)**2))
print("2")
# Second run with tol=1e-8
times_2, ys_2, errors_2, total_function_calls_2 = solve_ode(ode, t_span, y0, coeffs1, h=0.00001, tol=1e-7)
true_ys_2 = np.sin(times_2)
rmse_2 = np.sqrt(np.mean((ys_2 - true_ys_2)**2))

print(total_function_calls_1, rmse_1, total_function_calls_2, rmse_2)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot for the first run
plt.subplot(1, 2, 1)
plt.plot(times_1, ys_1, 'r-', label='RK45 Approximation')
plt.plot(times_1, true_ys_1, 'b--', label='True Solution $y=\sin(t)$')
plt.xlabel('Time (t)')
plt.ylabel('Solution (y)')
plt.title('First Run: tol=1e-7')
plt.legend()

# Plot for the second run
plt.subplot(1, 2, 2)
plt.plot(times_2, ys_2, 'r-', label='RK45 Approximation')
plt.plot(times_2, true_ys_2, 'b--', label='True Solution $y=\sin(t)$')
plt.xlabel('Time (t)')
plt.ylabel('Solution (y)')
plt.title('Second Run: tol=1e-7 coeffs1')
plt.legend()

plt.tight_layout()

# Plotting the error vs time for both runs
plt.figure(figsize=(10, 5))

# Error for the first run
error_1 = np.abs(ys_1 - true_ys_1)
plt.plot(times_1, error_1, 'b-', label='Error with tol=1e-7')

# Error for the second run
error_2 = np.abs(ys_2 - true_ys_2)
plt.plot(times_2, error_2, 'r--', label='Error with tol=1e-7 coeffs1')

plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.yscale('log')
plt.title('Error vs. Time for Different Tolerances')
plt.legend()
plt.grid(True)
plt.show()