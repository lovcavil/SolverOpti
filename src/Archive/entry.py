import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation
def ode(t, y):
    return y - t**2 + 1

# Define the time span and initial condition
t_span = (0, 2)
y0 = [0.5]

# Solve the ODE
sol = solve_ivp(ode, t_span, y0, method='RK45', dense_output=True)

# Generate points to interpolate the solution
t = np.linspace(0, 2, 100)
y = sol.sol(t)

# Plot the solution
plt.plot(t, y.T)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Solution of the ODE using RK45 method')
plt.grid(True)
plt.show()
