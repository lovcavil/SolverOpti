import numpy as np
import matplotlib.pyplot as plt
from p2.run_case_fixh import *
from ode_problem import *
from flatten import *
from tqdm import tqdm


def pre_eval_ode_functions_fix(vars, ode_problem=ode_lorenz_system):
    cases = []
    
    for var in vars:
        a = [0, var[0]]
        b = [var[1], 1 - var[1]]
        c = [[], [var[0]]]
        coeffs = (a, b, c)
        cases.append({"coeffs": coeffs, "tol": 10**(var[2]), "timeout": 5, "end_time": 5})

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

def rk_fix_step(f, t, y, h, coeffs, function_calls):
    a, b, c = coeffs
    n = len(a)
    k = []
    
    for i in range(n):
        ti = t + a[i] * h
        yi = y + h * sum(c[i][j] * k[j] for j in range(i))
        ki = f(ti, yi)
        function_calls += 1
        k.append(ki)
    
    y_next = y + h * sum(b[i] * k[i] for i in range(len(b)))
    return y_next, function_calls

def solve_ode_fixh(f, t_span, y0, coeffs, h=1e-5):
    function_calls = 0
    t0, tf = t_span
    t, y = t0, y0
    times, ys = [t], [y]
    
    with tqdm(total=tf - t0) as pbar:
        while t < tf:
            h = min(h, tf - t)
            y_new, function_calls = rk_fix_step(f, t, y, h, coeffs, function_calls)
            prev_t = t
            t += h
            y = y_new
            times.append(t)
            ys.append(y)
            pbar.update(t - prev_t)

    return np.array(times), np.array(ys), function_calls
