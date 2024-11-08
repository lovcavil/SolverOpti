import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import ode_solver
# Initialize the counter as a global variable
function_calls = 0

def ode(t, y):
    """The differential equation dy/dt = cos(t)."""
    global function_calls
    function_calls += 1  # Increment the counter
    return np.cos(t)

def ode_system(t, y):
    global function_calls
    function_calls += 1  # Increment the counter
    y1, y2 = y  # Unpack the array of dependent variables
    dy1dt = np.sin(t)+y1  # Compute the derivative of y1
    dy2dt = np.sin(t)+y2  # Compute the derivative of y2
    return np.array([dy1dt, dy2dt])

def ode_lorenz_system(t, y_, sigma=10, rho=28, beta=8/3):
    global function_calls
    function_calls += 1  # Increment the counter
    x, y, z = y_  # Unpack the array of dependent variables
    dxdt = sigma * (y - x)     # Compute the derivative of x
    dydt = x * (rho - z) - y   # Compute the derivative of y
    dzdt = x * y - beta * z    # Compute the derivative of z
    return np.array([dxdt, dydt, dzdt])


def run_case_varh(ode, coeffs, tol,end_time=5,true_solution_filename="lorenz_solution.csv"):
    # Record start time
    start_time = time.time()
    t_span = (0, end_time)

    y0 = np.array([1, 1,1])
    h = 0.00001
    
    # Solve the ODE and get the times, solutions (ys), and function call count
    times, ys, errors, total_function_calls = ode_solver.solve_ode(ode, t_span, y0, coeffs, h, tol)
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
    
def run_case_fixh(ode, coeffs, h,end_time=5,true_solution_filename="lorenz_solution.csv"):
    # Record start time
    start_time = time.time()
    t_span = (0, end_time)
    issue= False
    y0 = np.array([1, 1, 1])
    
    # Solve the ODE and get the times, solutions (ys), and function call count
    times, ys, total_function_calls = ode_solver.solve_ode_fixh(ode, t_span, y0, coeffs, h)
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

    print("Elapsed Time:", elapsed_time, "seconds")
    return {
            "times": times,
            "ys": ys,
            "true_ys": true_ys,
            "total_function_calls": total_function_calls,
            "rmse": rmse,
            "issue": issue
        }
    

import multiprocessing
import time
def long_running_function(queue,ode, coeffs, tol,end_time):
    print("Starting long-running task...")
    try:
        #result = run_case(ode, coeffs, tol,end_time)
        h=tol
        result = run_case_fixh(ode, coeffs, h,end_time)
        
    except KeyboardInterrupt:
        print("Task was interrupted")
        queue.put("Task interrupted")
        return
    queue.put(result)
#def call_with_timeout(func, timeout):
def run_case_with_timeout(ode, coeffs, tol=1e-15,end_time=5, timeout=5):


    # Create a Queue to receive the function's return value
    queue = multiprocessing.Queue()

    # Create and start a process that runs the function
    proc = multiprocessing.Process(target=long_running_function, args=(queue,ode, coeffs, tol,end_time))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()  # Forcefully terminate the process
        proc.join()
        return "Function execution exceeded the time limit of {} seconds.".format(timeout)
    else:
        # Retrieve the return value from the queue
        return queue.get()

# if __name__ == '__main__':
#     # Only run this block if the script is executed directly (not imported)
#     result = run_case_with_timeout(long_running_function, 5)
#     #print(result)