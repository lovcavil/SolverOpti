

import numpy as np
import matplotlib.pyplot as plt
from long_timeout_wrap import *
from run_case_varh import *
from ode_problem import *
from flatten import *
from tqdm import tqdm
import pickle
import os
from post import post
# Read a specific environment variable

def test_eval_ode_functions_single(var,ode_problem=ode_lorenz_system):
    a = [0, var[0]]
    c = [[], [var[0]]]
    b1 = [var[1], 1 - var[1]]
    b2 = [var[2], 1 - var[2]]
    coeffs = (a, b1, b2, c)
    case={"coeffs": coeffs, "tol": 10**(-3), "timeout": 60, "end_time": 5}
    i=1

    coeffs, tol, timeout, end_time = case["coeffs"], case["tol"], case["timeout"], case["end_time"]
    # result = run_case_varh(ode_problem, coeffs, tol, end_time)
    result = run_case_with_timeout(run_case_varh,ode_problem, coeffs, tol, end_time,true_solution_filename="lorenz_solution.csv")
    
    return result

def test2(i,var):
    result=test_eval_ode_functions_single(var)
    #print(result)
    casename = f"case{i}"
    folder_path = os.getenv("RUNTEMP")
    folder_path = os.path.join(folder_path,"A17" ,"opti",casename)
    print(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    full_path = os.path.join(folder_path, "data.pkl")
    import pickle
    with open(full_path, 'wb') as file:
        pickle.dump(result, file)
    post("opti",casename)
import csv

if __name__ == '__main__':
    

    file_path=r"C:\temp\A17\opti\202411111510\optPop\Chrom.csv"
    data = []
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # Convert each value in the row to float
            data.append([float(value) for value in row])


    print(data)
    for (i,var) in enumerate(data):
        print(var)
        try:
            test2(i,var)
        except:
            continue
