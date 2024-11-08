import multiprocessing
import time

def long_running_function():
    print("Starting long-running task...")
    try:
        for i in range(10):
            time.sleep(1)
            print(f"Task is running... ({i+1}/10)")
    except KeyboardInterrupt:
        print("Task was interrupted")
        return "Task interrupted"
    return "Task completed"

def call_with_timeout(func, timeout):
    # Create a process that runs the function
    proc = multiprocessing.Process(target=func)
    proc.start()  # Start the process
    proc.join(timeout=timeout)  # Wait for the process to finish or timeout

    if proc.is_alive():
        proc.terminate()  # Forcefully terminate the process
        proc.join()  # Make sure the process has cleaned up properly
        return "Function call timed out"
    else:
        return "Task completed"

if __name__ == '__main__':
    # Only run this block if the script is executed directly (not imported)
    result = call_with_timeout(long_running_function, 5)
    print(result)
