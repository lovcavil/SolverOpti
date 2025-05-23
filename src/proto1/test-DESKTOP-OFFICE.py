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

# Save the dictionary to disk

def test_eval_ode_functions_precode(ode_problem=ode_lorenz_system):
    #Runge–Kutta–Fehlberg
    coeffs0 = [
        0, 0.25, 0.375, 0.9230769230769231, 1, 0.5, 0.11851851851851852, 0, 0.5189863547758284, 0.5061314903420167, -0.18, 0.03636363636363636, 
        0.11574074074074074, 0, 0.5489278752436647, 0.5353313840155945, -0.2, 0, 0.25, 0.09375, 0.28125, 0.8793809740555303, -3.277196176604461, 
        3.3208921256258535, 2.0324074074074074, -8, 7.173489278752436, -0.20589668615984405, -0.2962962962962963, 2, -1.3816764132553607, 0.4529727095516569, -0.275
    ]
    #Cash-Karp
    #coeffs0 =[0, 0.2, 0.3, 0.6, 1, 0.875, 0.09788359788359788, 0, 0.4025764895330113, 0.21043771043771045, 0, 0.2891022021456804, 0.10217737268518519, 0, 0.38390790343915343, 0.24459273726851852, 0.019321986607142856, 0.25, 0.2, 0.075, 0.225, 0.3, -0.9, 1.2, -0.2037037037037037, 2.5, -2.5925925925925926, 1.2962962962962963, 0.029495804398148147, 0.341796875, 0.041594328703703706, 0.40034541377314814, 0.061767578125]
    a, b1, b2, c = reconstruct_coefficients(coeffs0,6)
    coeffs = (a, b1, b2, c)
    case={"coeffs": coeffs, "tol": 10**(-8), "timeout": 15, "end_time": 5}
    i=1
    case_key = f"case{i}"
    coeffs, tol, timeout, end_time = case["coeffs"], case["tol"], case["timeout"], case["end_time"]
    # result = run_case_varh(ode_problem, coeffs, tol, end_time)
    result = run_case_with_timeout(run_case_varh,ode_problem, coeffs, tol, end_time)
    
    return result

def test_eval_ode_functions_single(var,ode_problem=ode_lorenz_system):
    stage=2
    coeffs0=var[:7]# stage 2 has 7 numbers
    a, b1, b2, c = reconstruct_coefficients(coeffs0,stage)
    coeffs = (a, b1, b2, c)
    case={"coeffs": coeffs, "tol": 10**(-3), "timeout": 15, "end_time": 5}
    i=1
    case_key = f"case{i}"
    coeffs, tol, timeout, end_time = case["coeffs"], case["tol"], case["timeout"], case["end_time"]
    # result = run_case_varh(ode_problem, coeffs, tol, end_time)
    result = run_case_with_timeout(run_case_varh,ode_problem, coeffs, tol, end_time)
    
    return result

def test1():
    result=test_eval_ode_functions_precode()
    #print(result)
    casename="Runge–Kutta–Fehlberg"
    folder_path = os.getenv("RUNTEMP")
    folder_path = os.path.join(folder_path,"A17" ,"singleresult",casename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    full_path = os.path.join(folder_path, "data.pkl")
    import pickle
    with open(full_path, 'wb') as file:
        pickle.dump(result, file)
    post("singleresult",casename)

def test2(casename,var):

    result=test_eval_ode_functions_single(var)
    #print(result)
    folder_path = os.getenv("RUNTEMP")
    folder_path = os.path.join(folder_path,"A17" ,"singleresult",casename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    full_path = os.path.join(folder_path, "data.pkl")
    import pickle
    with open(full_path, 'wb') as file:
        pickle.dump(result, file)
    post("singleresult",casename)

if __name__ == '__main__':
    # casename="Heun–Euler"
    # var=[0, 1, 0.5, 0.5, 1, 0, 1]
    # test2(casename,var)
    casename="m"
    var=[0, 0.5, 0.5, 0.5, 1, 0, 0.5]
    test2(casename,var)
    # casename="fake1"
    # var=[0, 1.909715817293929918, 7.378215165329875536e-01, 1-7.378215165329875536e-01, 1.326320112456623179e-01, 1-1.326320112456623179e-01, 1.909715817293929918]

    # test2(casename,var)
    # casename="fake2"
    # var=[0, 9.885160759978742773e-01, 0.5443494443274259353, 1-0.5443494443274259353, -4.817238145637867675e-01, 1+4.817238145637867675e-01, 9.885160759978742773e-01]
    # test2(casename,var)