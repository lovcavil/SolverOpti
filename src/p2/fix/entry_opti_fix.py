import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import math
from p2.ode_solver_varh import pre_eval_ode_functions_fix

class MyProblem(ea.Problem):  # Inherit from the Problem superclass
    def __init__(self):
        name = 'MyProblem'  
        M = 2  # Number of objectives
        maxormins = [1] * M  # 1 indicates minimization for each objective
        Dim = 3  # Number of decision variables
        varTypes = [0] * (Dim - 1) + [1]  # 0 for real, 1 for integer
        
        # Decision variable bounds
        lb = [0, 0, -3]
        ub = [1, 1, -3]
        lbin = [1] * Dim  # Inclusive lower bounds
        ubin = [1] * Dim  # Inclusive upper bounds
        
        # Initialize the superclass with problem parameters
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # Objective function
        results = pre_eval_ode_functions_fix(Vars)
        f1, f2, CV = [], [], []

        for case_key, case_data in results.items():
            if isinstance(case_data, str) or case_data.get("issue"):  # Handle errors
                print(f"{case_key} encountered an issue or error.")
                f1.append([999999999])
                f2.append([10])
                CV.append([1])
                continue
            
            # Extract data from valid cases
            total_function_calls = case_data["total_function_calls"]
            rmse = case_data["rmse"]

            f1.append([math.log(total_function_calls)])
            f2.append([rmse])
            CV.append([0])
            plt.close()  # Close plot if any were generated (prevents display issues)

        ObjV = np.hstack([f1, f2])  # Objective values matrix
        return ObjV, np.array(CV)   # Return objectives and constraint violations

def main():
    # Instantiate problem object
    ode_problem = MyProblem()
    
    # Configure the algorithm
    algorithm = ea.moea_NSGA2_templet(
        ode_problem,
        ea.Population(Encoding='RI', NIND=20),
        MAXGEN=10,  # Maximum number of generations
        logTras=0   # Log generation interval (0 for no logging)
    )
    
    # Solve the optimization problem
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='result1')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
