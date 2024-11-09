import numpy as np
import matplotlib.pyplot as plt
from p2.run_case_fixh import *
from ode_problem import *
from flatten import *
from tqdm import tqdm

def pre_eval_ode_functions(vars, ode_problem=ode_lorenz_system):
    coeffs0 = [
        0, 0.25, 0.375, 0.9230769230769231, 1, 0.5, 0.11851851851851852, 0, 
        0.5189863547758284, 0.5061314903420167, -0.18, 0.03636363636363636, 
        0.11574074074074074, 0, 0.5489278752436647, 0.5353313840155945, 
        -0.2, 0, 0.25, 0.09375, 0.28125, 0.8793809740555303, -3.277196176604461, 
        3.3208921256258535, 2.0324074074074074, -8, 7.173489278752436, 
        -0.20589668615984405, -0.2962962962962963, 2, -1.3816764132553607, 
        0.4529727095516569, -0.275
    ]
    cases = []
    
    for var in vars:
        for i in range(33):
            coeffs0[i] = var[i]
        
        a, b1, b2, c = reconstruct_coefficients(coeffs0)
        coeffs = (a, b1, b2, c)
        cases.append({"coeffs": coeffs, "tol": 10**(var[1]), "timeout": 15, "end_time": 5})

    results = run_tests(ode_problem, cases)
    return results

def run_tests(ode_problem, test_cases):
    results = {}
    
    for i, test_case in enumerate(test_cases, start=1):
        case_key = f"case{i}"
        coeffs, tol, timeout, end_time = test_case["coeffs"], test_case["tol"], test_case["timeout"], test_case["end_time"]
        result = run_case_fixh(ode_problem, coeffs, tol, end_time)
        results[case_key] = result
    
    return results

def rk_step(f, t, y, h, coeffs):
    a, b1, b2, c = coeffs
    n = len(a)
    k = []
    
    for i in range(n):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        k.append(ki)
    
    y4 = y + h * sum(b2[i] * k[i] for i in range(len(b2)))
    y5 = y + h * sum(b1[i] * k[i] for i in range(len(b1)))
    error = np.linalg.norm(y4 - y5)  # Vector error norm
    
    return y5, error

def solve_ode_varh(f, t_span, y0, coeffs, h=1e-5, tol=1e-8):
    function_calls = 0
    t0, tf = t_span
    t, y = t0, y0
    times, ys, errors = [t], [y], []
    
    with tqdm(total=tf - t0) as pbar:
        while t < tf:
            h = min(h, tf - t)
            y_new, error = rk_step(f, t, y, h, coeffs)
            
            if error < tol:
                prev_t = t
                t += h
                y = y_new
                times.append(t)
                ys.append(y)
                errors.append(error)
                pbar.update(t - prev_t)

            h *= (tol / error) ** 0.2 if error > 0 else 2

    return np.array(times), np.array(ys), np.array(errors), function_calls
