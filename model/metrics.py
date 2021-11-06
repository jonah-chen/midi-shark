import torch
from time import perf_counter_ns, sleep

SMOOTH = 1e-8  # smoothing divisions
DEBUG = 2


def stopwatch(func):
    """
    Decorator for benchmarking the execution time of a function.
    Requires a debug level greater or equal to 2.

    Example:
    ```
    @stopwatch
    def foo():
        sleep(1)

    >> foo executed in 1000ms0.
    ```
    """
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        result = func(*args, **kwargs)
        end = perf_counter_ns()
        if DEBUG > 4:
            print(
                f"{func.__name__}{','.join([str(args),str(kwargs)])} executed in {int(1e-6*(end-start))}ms{round(1e-3*(end-start))%1000}.")
        elif DEBUG > 2:
            print(
                f"{func.__name__}{str(args)} executed in {int(1e-6*(end-start))}ms{round(1e-3*(end-start))%1000}.")
        elif DEBUG > 1:
            print(
                f"{func.__name__} executed in {int(1e-6*(end-start))}ms{round(1e-3*(end-start))%1000}.")
        return result
    return wrapper


def memlog(func):
    """
    Decorator for benchmarking the pytorch GPU memory usage.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"After {func.__name__} executed: Mem={torch.cuda.memory_allocated()>>10}KB, Reserved={torch.cuda.memory_reserved()>>10}KB")
        return result
    return wrapper


def mean_iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))         # Will be zzero if both are 0

    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # This is equal to comparing with thresolds
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded.mean()
