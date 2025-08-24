import time
import torch

class StepTimer:
    """
    Utility class for measuring runtime and GPU memory usage of code segments.

    Attributes:
        device (torch.device): Execution device.
        cuda (bool): Whether CUDA is available and the device is a CUDA device.
        m (dict): Stores timing and memory statistics for each measured step.
    """
    def __init__(self, device):
        self.device = device
        self.cuda = torch.cuda.is_available() and (device.type == "cuda")
        self.m = {}

    def _tick(self):
        if self.cuda:
            torch.cuda.synchronize()

    def timeit(self, name, fn, *args, **kwargs):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self._tick()
        t1 = time.perf_counter()
        mem = torch.cuda.max_memory_allocated(self.device) if self.cuda else 0
        self.m[name] = {"time_ms": (t1 - t0) * 1000.0, "max_mem_bytes": int(mem)}
        return out

    def summary(self):
        return self.m

    