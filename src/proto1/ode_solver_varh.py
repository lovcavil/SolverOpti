import numpy as np
import matplotlib.pyplot as plt
from run_case_varh import *
from ode_problem import *
from flatten import *
from tqdm import tqdm
from long_timeout_wrap import *
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def pre_eval_ode_functions(vars, ode_problem=ode_lorenz_system):
    cases = []
    for var in vars:
        a = [0, var[0]]
        c = [[], [var[0]]]
        b1 = [var[1], 1 - var[1]]
        b2 = [var[2], 1 - var[2]]
        coeffs = (a, b1, b2, c)
        cases.append({"coeffs": coeffs, "tol": 10**(-3), "timeout": 15, "end_time": 5})
    
    results = {}

    # Define a function for multithreaded execution of cases
    def run_case_multithreaded(case_index, test_case):
        coeffs, tol, timeout, end_time = test_case["coeffs"], test_case["tol"], test_case["timeout"], test_case["end_time"]
        result = run_case_with_timeout(run_case_varh, ode_problem, coeffs, tol, end_time,true_solution_filename="lorenz_solution.csv")
        return f"case{case_index}", result

    # Run each case in a separate thread
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-5) as executor:
        futures = {executor.submit(run_case_multithreaded, i, case): i for i, case in enumerate(cases, start=1)}
        for future in as_completed(futures):
            case_key, result = future.result()
            results[case_key] = result

    return results

def rk_step(f, t, y, h, coeffs, function_calls=0):
    a, b1, b2, c = coeffs
    n = len(a)
    k = []
    
    for i in range(n):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        function_calls += 1
        k.append(ki)
    
    y4 = y + h * sum(b2[i] * k[i] for i in range(len(b2)))
    y5 = y + h * sum(b1[i] * k[i] for i in range(len(b1)))
    error = np.linalg.norm(y4 - y5)
    
    return y5, error, function_calls

def solve_ode_varh(f, t_span, y0, coeffs, h=1e-5, tol=1e-8):
    function_calls = 0
    t0, tf = t_span
    t, y = t0, y0
    times, ys, errors = [t], [y], []

    while t < tf:
        h = min(h, tf - t)
        y_new, error, function_calls = rk_step(f, t, y, h, coeffs, function_calls)
        
        if error < tol:
            prev_t = t
            t += h
            y = y_new
            times.append(t)
            ys.append(y)
            errors.append(error)
        h *= (tol / error) ** 0.2 if error > 0 else 2
    # with tqdm(total=tf - t0) as pbar:
    #     while t < tf:
    #         h = min(h, tf - t)
    #         y_new, error, function_calls = rk_step(f, t, y, h, coeffs, function_calls)
            
    #         if error < tol:
    #             prev_t = t
    #             t += h
    #             y = y_new
    #             times.append(t)
    #             ys.append(y)
    #             errors.append(error)
    #             pbar.update(t - prev_t)

    #         h *= (tol / error) ** 0.2 if error > 0 else 2

    return np.array(times), np.array(ys), np.array(errors), function_calls
