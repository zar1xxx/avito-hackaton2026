"""Read this process's memory footprint without changing OS limits or settings."""
import os


def memory_snapshot():
    if os.name != 'nt':
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'peak_working_set_bytes': int(peak * (1 if sys.platform == 'darwin' else 1024))}
    import ctypes
    class Status(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                    *[(name,ctypes.c_ulonglong) for name in ('total','available','total_page','available_page',
                                                           'total_virtual','available_virtual','extended')]]
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong),
                    *[(name,ctypes.c_size_t) for name in ('peak','working','peak_paged','paged',
                                                        'peak_nonpaged','nonpaged','pagefile','peak_pagefile')]]
    status=Status(); status.length=ctypes.sizeof(status)
    result={}
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        result.update(total_physical_bytes=int(status.total),available_physical_bytes=int(status.available))
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    if query(process(),ctypes.byref(counters),counters.cb):
        result.update(working_set_bytes=int(counters.working),peak_working_set_bytes=int(counters.peak))
    return result
