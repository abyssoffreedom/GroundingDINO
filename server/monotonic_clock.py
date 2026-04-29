import os
import time

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _query_performance_counter = _kernel32.QueryPerformanceCounter
    _query_performance_counter.argtypes = [ctypes.POINTER(wintypes.LARGE_INTEGER)]
    _query_performance_counter.restype = wintypes.BOOL
    _query_performance_frequency = _kernel32.QueryPerformanceFrequency
    _query_performance_frequency.argtypes = [ctypes.POINTER(wintypes.LARGE_INTEGER)]
    _query_performance_frequency.restype = wintypes.BOOL

    _qpc_frequency = wintypes.LARGE_INTEGER()
    if not _query_performance_frequency(ctypes.byref(_qpc_frequency)):
        raise OSError(ctypes.get_last_error(), "QueryPerformanceFrequency failed")


def server_now_ns() -> int:
    if os.name != "nt":
        return time.perf_counter_ns()

    counter = wintypes.LARGE_INTEGER()
    if not _query_performance_counter(ctypes.byref(counter)):
        raise OSError(ctypes.get_last_error(), "QueryPerformanceCounter failed")

    return (int(counter.value) * 1_000_000_000) // int(_qpc_frequency.value)
