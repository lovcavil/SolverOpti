import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import ode_solver_varh as ode_solver_varh
import time


def run_case_varh(ode, coeffs, tol,end_time=5,true_solution_filename="lorenz_solution.csv"):
    # Record start time
    start_time = time.time()
    t_span = (0, end_time)

    y0 = np.array([1, 1, 1])
    h = 0.00001
    
    # Solve the ODE and get the times, solutions (ys), and function call count
    times, ys, errors, total_function_calls = ode_solver_varh.solve_ode_varh(ode, t_span, y0, coeffs, h, tol)
    interpolators = {}
    df_true = pd.read_csv(true_solution_filename)
    for column in df_true.columns[1:]:  # Skip the first column assuming it's the time column
        interpolators[column] = interp1d(df_true['t'], df_true[column], kind='linear', fill_value="extrapolate")
    
    # Evaluate the interpolators at the solver times
    true_ys = np.array([interpolators[col](times) for col in interpolators]).T

    rmse = np.sqrt(np.mean((ys - true_ys)**2))
    
    # Make sure errors array starts with an initial value (e.g., 0) to match times[0]
    if len(errors) < len(times):
        # Prepend an initial error or adjust the array as needed
        errors = np.insert(errors, 0, 0)
    
    # Ensure errors array has a value for each time step, interpolating if necessary
    if len(errors) < len(times):
        # This should not happen now, but just in case, you can interpolate or extend the errors array
        interpolator = interp1d(np.linspace(times[0], times[-1], len(errors)), errors, kind='nearest', fill_value="extrapolate")
        interpolated_errors = interpolator(times)
    else:
        interpolated_errors = errors
        # Record end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Elapsed Time:", elapsed_time, "seconds")
    return {
            "times": times,
            "ys": ys,
            "true_ys": true_ys,
            "errors": interpolated_errors,
            "total_function_calls": total_function_calls,
            "rmse": rmse
        }
    