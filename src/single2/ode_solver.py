
from run import *
from ode_problem import *
import matplotlib.pyplot as plt
from flatten import *
def pre_eval_ode_functions(vars,ode_problem=ode_lorenz_system,):
    coeffs0=[0, 0.25, 0.375, 0.9230769230769231, 1, 0.5, 0.11851851851851852, 0, 0.5189863547758284, 0.5061314903420167, -0.18, 0.03636363636363636, 0.11574074074074074, 0, 0.5489278752436647, 0.5353313840155945, -0.2, 0, 0.25, 0.09375, 0.28125, 0.8793809740555303, -3.277196176604461, 3.3208921256258535, 2.0324074074074074, -8, 7.173489278752436, -0.20589668615984405, -0.2962962962962963, 2, -1.3816764132553607, 0.4529727095516569, -0.275]
    cases=[]
    for var in vars:
        for i in range(33):  # Assuming you want to assign the first three elements
            coeffs0[i] = var[i]
        a, b1, b2, c=reconstruct_coefficients(coeffs0)
        
        #print(a)
        coeffs0_1=(a, b1, b2, c)
        # Define your test cases here
        cases.append({"coeffs": coeffs0_1, "tol": 10**(var[1]), "timeout": 15,"end_time":5})

    # Execute the test cases
    results = run_tests(ode_problem, cases)
    return results

def pre_eval_ode_functions_fix(vars,ode_problem=ode_lorenz_system,):
    cases=[]
    for var in vars:
        a=[0,var[0]]
        b=[var[1],1-var[1]]
        c=[
            [],
         [var[0]],
        ]
        #print(a)
        coeffs_fix=(a, b, c)
        # Define your test cases here
        cases.append({"coeffs": coeffs_fix, "tol": 10**(var[2]), "timeout": 5,"end_time":5})

    # Execute the test cases
    results = run_tests(ode_problem, cases)
    return results

def run_tests(ode_problem, test_cases):
    """
    Executes multiple test cases with a timeout for each.
    
    Parameters:
    - ode: The ODE function to solve.
    - test_cases: A list of dictionaries, where each dictionary contains the coefficients, tolerance, and timeout for a test case.
    
    Returns:
    A dictionary of results for each test case.
    """
    results = {}
    
    for i, test_case in enumerate(test_cases, start=1):
        case_key = f"case{i}"
        coeffs, tol, timeout,end_time = test_case["coeffs"], test_case["tol"], test_case["timeout"],test_case["end_time"]
        
        # Execute the test case with a timeout
        #result = run_case_with_timeout(ode_problem, coeffs, tol,end_time, timeout=timeout)
        result = run_case_fixh(ode_problem, coeffs, tol,end_time)
        # Store the result with a unique key for each test case
        results[case_key] = result
    
    return results


def rk_step(f, t, y, h, coeffs):
    a, b1, b2, c = coeffs
    n=len(a)
    k = []
    for i in range(n):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        k.append(ki)
    
    y4 = y + h * sum(b2[i] * k[i] for i in range(len(b2)))
    y5 = y + h * sum(b1[i] * k[i] for i in range(len(b1)))
    error = np.linalg.norm(y4 - y5)  # Use norm for vector error
    
    return y5, error

def rk_fix_step(f, t, y, h, coeffs,function_calls):
    a, b, c = coeffs
    n=len(a)
    k = []
    for i in range(n):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        function_calls=function_calls+1
        k.append(ki)
    
    y4 = y + h * sum(b[i] * k[i] for i in range(len(b)))
    #y5 = y + h * sum(b1[i] * k[i] for i in range(len(b1)))
    #error = np.linalg.norm(y4 - y5)  # Use norm for vector error
    return y4,function_calls

from tqdm import tqdm
import numpy as np
#,h_fix= True
def solve_ode_varh(f, t_span, y0, coeffs, h=0.00001, tol=1e-8):
    """Solves an ODE using an adaptive RK45 method with given coefficients."""
    global function_calls
    function_calls = 0  # Reset the counter at the start of each solve_ode call
    t0, tf = t_span
    t = t0
    y = y0
    times = [t]
    ys = [y]
    errors = []  # To store the error at each step

    # Initialize tqdm progress bar
    with tqdm(total=tf-t0) as pbar:
        while t < tf:
            h = min(h, tf - t)
            y_new, error = rk_step(f, t, y, h, coeffs)

            if error < tol:
                prev_t = t  # Store the current time to calculate progress increment
                t += h
                y = y_new
                times.append(t)
                ys.append(y)
                errors.append(error)
                
                # Update progress bar by the increment in time
                pbar.update(t - prev_t)

            if error > 0:
                h *= (tol / error) ** 0.2
            else:
                h = h * 2

    return np.array(times), np.array(ys), np.array(errors), function_calls

def solve_ode_fixh(f, t_span, y0, coeffs, h=0.00001):
    """Solves an ODE using an adaptive RK45 method with given coefficients."""
    function_calls = 0  # Reset the counter at the start of each solve_ode call
    t0, tf = t_span
    t = t0
    y = y0
    times = [t]
    ys = [y]

    # Initialize tqdm progress bar
    with tqdm(total=tf-t0) as pbar:
        while t < tf:
            h = min(h, tf - t)
            (y_new,function_calls) = rk_fix_step(f, t, y, h, coeffs,function_calls)
            prev_t = t  # Store the current time to calculate progress increment
            t += h
            y = y_new
            times.append(t)
            ys.append(y)
                
            # Update progress bar by the increment in time
            pbar.update(t - prev_t)

    pass
    return np.array(times), np.array(ys), function_calls