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
    queue = multiprocessing.Queue()
    start_time = time.time()
    proc = multiprocessing.Process(target=long_running_function, args=(queue, target_function, ode, coeffs, tol, end_time))
    proc.start()

    try:
        result = queue.get(timeout=timeout)
        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        return result
    except multiprocessing.queues.Empty:
        proc.terminate()
        proc.join()
        print(f"Function execution exceeded the time limit of {timeout} seconds.")
        return f"Function execution exceeded the time limit of {timeout} seconds."
    finally:
        proc.join()