import numpy as np
import matplotlib.pyplot as plt

def ode(t, y):
    """The differential equation to be solved."""
    return np.cos(t)

def rk45_step(f, t, y, h):
    """Performs a single step of the RK45 method."""
    # Butcher tableau for RK45
    a = [0, 1/4, 3/8, 12/13, 1, 1/2]
    b1 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    b2 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    c = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    
    k = []
    for i in range(6):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        k.append(ki)
    
    # Calculate the 4th and 5th order estimates
    y4 = y + h * sum(b2[i] * k[i] for i in range(6))
    y5 = y + h * sum(b1[i] * k[i] for i in range(6))
    
    # Estimate the error
    error = np.abs(y4 - y5)
    
    return y5, error

def solve_ode(f, t_span, y0, h=0.00001, tol=1e-8):
    """Solves an ODE using an adaptive RK45 method."""
    t0, tf = t_span
    t = t0
    y = y0
    times = [t]
    ys = [y]
    
    while t < tf:
        h = min(h, tf - t)  # Adjust step size if we're close to the end
        y_new, error = rk45_step(f, t, y, h)
        
        # If the error is acceptable, move to the next step
        if error < tol:
            t += h
            y = y_new
            times.append(t)
            ys.append(y)
        
        # Adjust the step size
        h *= (tol / error)**0.2
    
    return np.array(times), np.array(ys)

# Solve the ODE
t_span = (0, 2)
y0 = 0
times, ys = solve_ode(ode, t_span, y0)

# True solution for comparison
true_xs = np.linspace(t_span[0], t_span[1], 400)
true_ys = np.sin(true_xs)

# Plotting the results
plt.plot(true_xs, true_ys, 'b--', label='True Solution $y=\sin(x)$')

plt.legend()
plt.title('RK45 Method vs. True Solution')

# Plot the solution
plt.plot(times, ys)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()
