import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import p2.ode_solver_varh as ode_solver_varh
import time

    
def run_case_fixh(ode, coeffs, h,end_time=5,true_solution_filename="lorenz_solution.csv"):
    # Record start time
    start_time = time.time()
    t_span = (0, end_time)
    issue= False
    y0 = np.array([1, 1, 1])
    
    # Solve the ODE and get the times, solutions (ys), and function call count
    times, ys, total_function_calls = ode_solver_varh.solve_ode_fixh(ode, t_span, y0, coeffs, h)
    interpolators = {}
    df_true = pd.read_csv(true_solution_filename)
    for column in df_true.columns[1:]:  # Skip the first column assuming it's the time column
        interpolators[column] = interp1d(df_true['t'], df_true[column], kind='linear', fill_value="extrapolate")
    
    # Evaluate the interpolators at the solver times
    true_ys = np.array([interpolators[col](times) for col in interpolators]).T
    try:
        rmse = np.sqrt(np.mean((ys - true_ys)**2))
    except Exception as e:
        rmse = np.nan
        issue = True
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    #print("Elapsed Time:", elapsed_time, "seconds fc",total_function_calls)
    return {
            "times": times,
            "ys": ys,
            "true_ys": true_ys,
            "total_function_calls": total_function_calls,
            "rmse": rmse,
            "issue": issue
        }
    
