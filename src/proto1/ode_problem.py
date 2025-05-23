import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import ode_solver_varh as ode_solver_varh
# Initialize the counter as a global variable

def ode(t, y):
    """The differential equation dy/dt = cos(t)."""

    return np.cos(t)

def ode_system(t, y):

    y1, y2 = y  # Unpack the array of dependent variables
    dy1dt = np.sin(t)+y1  # Compute the derivative of y1
    dy2dt = np.sin(t)+y2  # Compute the derivative of y2
    return np.array([dy1dt, dy2dt])

def ode_lorenz_system(t, y_, sigma=10, rho=28, beta=8/3):

    x, y, z = y_  # Unpack the array of dependent variables
    dxdt = sigma * (y - x)     # Compute the derivative of x
    dydt = x * (rho - z) - y   # Compute the derivative of y
    dzdt = x * y - beta * z    # Compute the derivative of z

    return np.array([dxdt, dydt, dzdt])

def ode_lorenz_system2(t, y_, sigma=20, rho=38, beta=8/3):

    x, y, z = y_  # Unpack the array of dependent variables
    dxdt = sigma * (y - x)     # Compute the derivative of x
    dydt = x * (rho - z) - y   # Compute the derivative of y
    dzdt = x * y - beta * z    # Compute the derivative of z

    return np.array([dxdt, dydt, dzdt])