import numpy as np
import matplotlib.pyplot as plt
from run_case_varh import *
from ode_problem import *
from flatten import *
from tqdm import tqdm
import pickle
import os
def post(catlog="singleresult",case="defaultcase"):
    folder_path = os.getenv("RUNTEMP")
    #folder_path = os.path.join(folder_path,"A17" ,"singleresult","Cash-Karp")
    folder_path = os.path.join(folder_path,"A17" ,catlog,case)
    folder_path1 = os.path.join(folder_path,"data.pkl")
    with open(folder_path1, 'rb') as file:
        result = pickle.load(file)

    case_key="single"

    times, ys, true_ys, errors = result["times"], result["ys"], result["true_ys"], result["errors"]
    total_function_calls, rmse = result["total_function_calls"], result["rmse"]

    plt.subplot(2, 1, 1)
    plt.plot(times, ys, 'r-', label='RK45 Approximation')
    plt.plot(times, true_ys, 'b--', label='True Solution $y=\sin(t)$')
    plt.xlabel('Time (t)')
    plt.ylabel('Solution (y)')
    plt.title(f'{case_key}-Total f calls: {total_function_calls}')
    plt.legend()
                
    # Plotting the error for the current case

    plt.subplot(2, 1, 2)
    plt.plot(times, errors, 'g-', label='Error')
    plt.xlabel('Time (t)')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title(f'Error vs. Time for {case_key}-RMSE: {rmse}')
    plt.legend()

    f1=([total_function_calls])
    f2=([rmse])
    CV=([0])
    folder_path2 = os.path.join(folder_path,f'plot_output_{case_key}.png')
    plt.savefig(folder_path2, dpi=300)
    plt.close()    

    # Calculate the difference between RK45 approximation and the true solution
    difference = [y - true_y for y, true_y in zip(ys, true_ys)]

    # Plot the difference over time
    plt.plot(times, difference, 'g-', label='Difference (RK45 - True Solution)')
    plt.xlabel('Time (t)')
    plt.ylabel('Difference in Solution (y)')
    plt.title(f'{case_key} - Difference Plot | Total f calls: {total_function_calls}')
    plt.legend()
    folder_path2 = os.path.join(folder_path,f'plot_output_{case_key}1.png')
    plt.savefig(folder_path2, dpi=300)
    plt.close()    


if __name__ == '__main__':
    post("singleresult","Runge–Kutta–Fehlberg")