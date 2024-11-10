import multiprocessing
import time

def long_running_function(queue, target_function, ode, coeffs, tol, end_time):
    try:
        h = tol
        result = target_function(ode, coeffs, h, end_time)
    except KeyboardInterrupt:
        print("Task was interrupted")
        queue.put("Task interrupted")
        return
    queue.put(result)

def run_case_with_timeout(target_function, ode, coeffs, tol, end_time=5, timeout=5):
    # Create a Queue to receive the function's return value
    queue = multiprocessing.Queue()
    
    # Start timing
    start_time = time.time()
    
    # Create and start a process that runs the function
    proc = multiprocessing.Process(target=long_running_function, args=(queue, target_function, ode, coeffs, tol, end_time))
    proc.start()

    # Wait for result from queue with a specified timeout
    try:
        result = queue.get(timeout=timeout)  # Attempt to get result within timeout period
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        return result
    except multiprocessing.queues.Empty:
        # If we exceed the timeout waiting for a result, we terminate the process
        proc.terminate()
        proc.join()
        print(f"Function execution exceeded the time limit of {timeout} seconds.")
        return f"Function execution exceeded the time limit of {timeout} seconds."
    finally:
        # Ensure process is cleaned up after result retrieval or timeout
        proc.join()

# Example usage (replace target_function, ode, coeffs, and tol with real values)
# result = run_case_with_timeout(target_function, ode, coeffs, tol, end_time=5, timeout=5)
