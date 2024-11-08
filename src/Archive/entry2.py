import numpy as np
import matplotlib.pyplot as plt

def ode(t, y):
    """The differential equation dy/dt = cos(t)."""
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
            t += h
            y = y_new
            times.append(t)
            ys.append(y)
            errors.append(error)
        
        h *= (tol / error)**0.2 if error > 0 else h * 2
    
    return np.array(times), np.array(ys), np.array(errors)

# RK45 coefficients
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

# Solve the ODE
t_span = (0, 2)
y0 = 0
times, ys, errors = solve_ode(ode, t_span, y0, coeffs)

# True solution for comparison
true_ys = np.sin(times)

# Plotting the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(times, ys, 'r-', label='RK45 Approximation')
plt.plot(times, true_ys, 'b--', label='True Solution $y=\sin(t)$')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.title('RK45 Method vs. True Solution')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, np.abs(ys - true_ys), 'k-', label='Error')
plt.xlabel('t')
plt.ylabel('Error')
plt.yscale('log')
plt.legend()
plt.title('Error in RK45 Approximation')
plt.grid(True)

plt.tight_layout()
plt.show()
