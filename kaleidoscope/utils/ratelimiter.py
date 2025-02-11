import time
from functools import wraps


def ratelimiter(calls_per_second):
    interval = 1.0 / calls_per_second
    last_called = [0.0]  # Using a list to make it mutable in nested scope

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_called[0] >= interval:
                last_called[0] = current_time
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage:
    @ratelimiter(5)  # Limit to 2 calls per second
    def my_function():
        print("Function executed")

    # Test loop
    for _ in range(5):
        result = my_function()
        if result is None:
            print("Call ignored")
        time.sleep(0.1)
