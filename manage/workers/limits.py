"""
The limits of a worker process, on each platform.

- **Windows:** one Job Object per worker, made with ``ctypes`` (pywin32 is no dependency of the application): a memory
  limit per process (an allocation beyond it raises MemoryError inside the worker, which lives on), one active process
  (the worker cannot start another), kill on close (the worker ends with the server, a crash of the server included),
  and death on an unhandled exception (no error dialog waits).
- **Linux:** the worker limits itself before numpy loads (:func:`limit_self`): its data (``RLIMIT_DATA``, since Linux
  4.7 its private writable memory, mapped memory included), no core file (it would hold data), and the system's
  out-of-memory killer taking it first (``oom_score_adj`` 1000). Before each job its CPU time is limited too
  (:func:`cpu_deadline`): a worker whose parent can no longer end it stops by itself.
- **macOS:** the time limit only (and the CPU's): the system does not hold a data limit on mapped memory.

**No lower priority:** the time limit is the server's clock, so a worker the system starves of processor time would
miss it. With every core busy, a worker below normal priority did not start within 30 s three times out of three,
and ordinary criteria reached the time limit; at normal priority it started. A runaway expression is bounded by the
time limit, on at most ``CMP_FORMULA_WORKERS`` processes.

This module imports the standard library only: the worker's bootstrap calls it before anything else loads.
"""

from __future__ import annotations

import sys
from typing import Any

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")


def mechanism() -> str:
    """How a worker's memory is limited here, for the server's log."""
    if WINDOWS:
        return "a Job Object"
    if LINUX:
        return "a limit of its data (RLIMIT_DATA)"
    return "none: the time limit only"


def memory_limited() -> bool:
    return WINDOWS or LINUX


# --- the worker, on itself (POSIX) ------------------------------------------------------------------------------------


def limit_self(memory_mb: int) -> None:
    """In the worker, before numpy loads: its data limited to ``memory_mb`` (Linux), no core file, the out-of-memory
    killer's first choice. Nothing on Windows: its Job Object holds it."""
    if sys.platform == "win32":
        return
    import resource

    if sys.platform.startswith("linux"):
        _, hard = resource.getrlimit(resource.RLIMIT_DATA)
        wanted = int(memory_mb) * 2**20
        soft = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))
        try:
            with open("/proc/self/oom_score_adj", "w", encoding="ascii") as score:
                score.write("1000")
        except OSError:
            pass  # not allowed in some containers: the memory limit above still stops the worker
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def cpu_deadline(seconds: float) -> None:
    """In the worker, before each job (POSIX): its CPU time limited to what it used, the job's time limit and 5 s more,
    so that a worker its parent can no longer end stops by itself (SIGXCPU ends a process)."""
    if sys.platform == "win32":
        return
    import resource

    used = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(used.ru_utime + used.ru_stime + float(seconds) + 5) + 1
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY:
        soft = min(soft, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))


def machine_memory_mb() -> int | None:
    """The memory this server may use, in megabytes: its container's (cgroup ``memory.max``) or its machine's; None
    when it cannot be read."""
    if sys.platform.startswith("linux"):
        try:
            with open("/sys/fs/cgroup/memory.max", encoding="ascii") as limit:
                text = limit.read().strip()
            if text.isdigit():
                return int(text) // 2**20
        except OSError:
            pass  # no limit of a container to read: the machine's memory is read next
        try:
            with open("/proc/meminfo", encoding="ascii") as info:
                for line in info:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except OSError, ValueError, IndexError:
            return None
        return None
    if sys.platform == "win32":
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys // 2**20)
    return None


# --- Windows: a Job Object per worker ---------------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wintypes

    _EXTENDED_LIMIT_INFORMATION = 9
    _LIMIT_ACTIVE_PROCESS = 0x00000008
    _LIMIT_PROCESS_MEMORY = 0x00000100
    _LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _EXTENDED_LIMIT_INFORMATION_STRUCT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def _kernel32() -> Any:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.GetWindowsDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
        k32.GetWindowsDirectoryW.restype = wintypes.UINT
        return k32

    def job_for_worker(memory_mb: int) -> int:
        """A Job Object with a worker's limits; its handle (closing it ends the worker)."""
        k32 = _kernel32()
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _EXTENDED_LIMIT_INFORMATION_STRUCT()
        basic = info.BasicLimitInformation
        basic.LimitFlags = (
            _LIMIT_PROCESS_MEMORY
            | _LIMIT_ACTIVE_PROCESS
            | _LIMIT_KILL_ON_JOB_CLOSE
            | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        basic.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = int(memory_mb) * 2**20
        if not k32.SetInformationJobObject(
            job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            error = ctypes.get_last_error()
            k32.CloseHandle(job)
            raise ctypes.WinError(error)
        return int(job)

    def join(job: int, process: Any) -> None:
        """Put a started worker in its job."""
        k32 = _kernel32()
        if not k32.AssignProcessToJobObject(job, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close_job(job: int) -> None:
        _kernel32().CloseHandle(job)

    def job_limits(job: int) -> dict[str, int]:
        """What a job holds: its memory limit and peak (MB), its active processes, and its flags (for the tests)."""
        k32 = _kernel32()
        info = _EXTENDED_LIMIT_INFORMATION_STRUCT()
        k32.QueryInformationJobObject(
            job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info), None
        )
        basic = info.BasicLimitInformation
        return {
            "memory_mb": int(info.ProcessMemoryLimit // 2**20),
            "peak_mb": int(info.PeakProcessMemoryUsed // 2**20),
            "active": int(basic.ActiveProcessLimit),
            "kill_on_close": int(bool(basic.LimitFlags & _LIMIT_KILL_ON_JOB_CLOSE)),
        }

    def windows_directory() -> str:
        """The Windows folder, from the system rather than from the server's environment."""
        buffer = ctypes.create_unicode_buffer(260)
        size = _kernel32().GetWindowsDirectoryW(buffer, 260)
        return buffer.value if size else r"C:\Windows"
