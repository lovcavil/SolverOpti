
import multiprocessing

def long_running_function(queue,target_function,ode, coeffs, tol,end_time):
    #print("Starting long-running task...")
    try:
        #result = run_case(ode, coeffs, tol,end_time)
        h=tol
        result = target_function(ode, coeffs, h,end_time)
        
    except KeyboardInterrupt:
        print("Task was interrupted")
        queue.put("Task interrupted")
        return
    queue.put(result)
#def call_with_timeout(func, timeout):
def run_case_with_timeout(ode, coeffs, tol,end_time=5, timeout=5):

    # Create a Queue to receive the function's return value
    queue = multiprocessing.Queue()

    # Create and start a process that runs the function
    proc = multiprocessing.Process(target=long_running_function, args=(queue,ode, coeffs, tol,end_time))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()  # Forcefully terminate the process
        proc.join()
        return "Function execution exceeded the time limit of {} seconds.".format(timeout)
    else:
        # Retrieve the return value from the queue
        res=queue.get()
        print(res)
        return res