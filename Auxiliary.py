from functools import wraps
import time
import logging
import asyncio


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# log a type of list
def log_list(logger_class, str_list):
    logger_class.info(str)
    for str in str_list:
        logger_class.info(str)


logger.log_list = log_list


# Decorator to delay the execution of a function, iteratively wait longer each time'''
def delay_execution(seconds=4, tries=5, default_return=[], exponential=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(1, tries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.error(f"Error: {e}. Retrying in {seconds*i} seconds")
                    time.sleep(seconds * i * exponential)
            return default_return

        return wrapper

    return decorator


# Asynchronous decorator to delay the execution of a function, with exponential backoff
def delay_execution_async(seconds=4, tries=5, default_return=[], exponential=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(1, tries + 1):
                try:
                    # Await the asynchronous function call
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    wait_time = seconds * (exponential ** (i - 1))
                    logger.error(f"Error: {e}. Retrying in {wait_time} seconds")
                    # Use asyncio.sleep for non-blocking delay
                    await asyncio.sleep(wait_time)
            return default_return

        return wrapper

    return decorator
