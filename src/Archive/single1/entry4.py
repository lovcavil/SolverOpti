
from single1.run import *
import matplotlib.pyplot as plt
from single1.flatten import *
def test():
    coeffs = (
        [0, 1/4, 3/8, 12/13, 1, 1/2],
        [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],
        [25/216, 0, 1408/2565, 2197/4104, -1/5, 0],
        [
            [],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ]
    )
    # RK45 coefficientsCash and Karp have modified Fehlberg's original idea. The extended tableau for the Cash–Karp method is
    coeffs1 = (
        [0, 1/5, 3/10, 3/5, 1, 7/8],
        [37/378, 0, 250/621, 125/594, 0, 512/1771],
        [	2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],
        [
            [],
            [1/5],
            [3/40,	9/40],
            [3/10,-9/10,6/5],
            [-11/54,	5/2,	-70/27,	35/27],
            [1631/55296,175/512,	575/13824,44275/110592,253/4096	]
        ]
    )
    
    coeffs0=[0, 0.25, 0.375, 0.9230769230769231, 1, 0.5, 0.11851851851851852, 0, 0.5189863547758284, 0.5061314903420167, -0.18, 0.03636363636363636, 0.11574074074074074, 0, 0.5489278752436647, 0.5353313840155945, -0.2, 0, 0.25, 0.09375, 0.28125, 0.8793809740555303, -3.277196176604461, 3.3208921256258535, 2.0324074074074074, -8, 7.173489278752436, -0.20589668615984405, -0.2962962962962963, 2, -1.3816764132553607, 0.4529727095516569, -0.275]
    a, b1, b2, c=reconstruct_coefficients(coeffs0)
    coeffs0_1=(a, b1, b2, c)
    # Define your test cases here
    test_cases = [
        {"coeffs": coeffs, "tol": 1e-15, "timeout": 15,"end_time":5},
        {"coeffs": coeffs1, "tol": 1e-15, "timeout": 15,"end_time":5},
        # Add more test cases as needed
    ]

    # Execute the test cases
    results = run_tests(ode_lorenz_system, test_cases)
    return results

def run_tests(ode, test_cases):
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
        result = run_case_with_timeout(ode, coeffs, tol,end_time, timeout=timeout)
        
        # Store the result with a unique key for each test case
        results[case_key] = result
    
    return results

def main():

    results = test()  # This function call is where you collect the test case results

    # Create a figure for the solution plots
    plt.figure(figsize=(12, len(results) * 3))

    # Loop through the results to plot each test case or print an error message
    for i, (case_key, case_data) in enumerate(results.items(), start=1):
        if isinstance(case_data, str):  # Check if the result is an error message (string)
            print(f"{case_key} encountered an error: {case_data}")
            continue  # Skip to the next iteration
        
        # Proceed with unpacking and plotting since case_data is not an error string
        times, ys, true_ys, errors = case_data["times"], case_data["ys"], case_data["true_ys"], case_data["errors"]
        
        # Plotting the results for the current case
        plt.subplot(len(results), 2, 2*i-1)
        plt.plot(times, ys, 'r-', label='RK45 Approximation')
        plt.plot(times, true_ys, 'b--', label='True Solution $y=\sin(t)$')
        plt.xlabel('Time (t)')
        plt.ylabel('Solution (y)')
        plt.title(f'{case_key}')
        plt.legend()
        
        # Plotting the error for the current case
        plt.subplot(len(results), 2, 2*i)
        plt.plot(times, errors, 'g-', label='Error')
        plt.xlabel('Time (t)')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.title(f'Error vs. Time for {case_key}')
        plt.legend()

        # Printing function call count and RMSE for cases that completed successfully
        total_function_calls, rmse = case_data["total_function_calls"], case_data["rmse"]
        print(f"{case_key}: Total function calls: {total_function_calls}, RMSE: {rmse}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()