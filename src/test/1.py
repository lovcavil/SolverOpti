import concurrent.futures
import time

def long_running_function():
    print("Starting long-running task...")
    for i in range(10):
        time.sleep(1)
        print(f"Task is running... ({i+1}/10)")
      # Simulates a delay, e.g., a long computation or slow I/O operation
    return "Task completed"

def call_with_timeout(func, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "Function call timed out"

# Call the function with a 5-second timeout
result = call_with_timeout(long_running_function, 5)
print(result)
