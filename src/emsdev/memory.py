"""System memory: installed, and available to this process now."""
import os, sys


def windows_status():
    """Windows: (installed, available) bytes from GlobalMemoryStatusEx; None elsewhere or where it fails."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return float(st.ullTotalPhys), float(st.ullAvailPhys)
    except (AttributeError, OSError, ImportError):
        return None


def total():
    """Installed RAM in bytes (None where neither sysconf nor Windows can tell)."""
    win = windows_status()
    if win:
        return win[0]
    try:
        return float(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        return None


def available():
    """Bytes of memory available to this process: MemAvailable, capped by a cgroup limit (Linux); what Windows
    reports available; else psutil's, else 8 GB."""
    win = windows_status()
    if win:
        return win[1]
    try:
        avail = next((int(l.split()[1]) * 1024 for l in open('/proc/meminfo') if l.startswith('MemAvailable:')), None)
    except OSError:
        avail = None
    for lim, use in (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                     ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        try:
            L = open(lim).read().strip()
            if L != 'max' and int(L) < 1 << 60:
                free = int(L) - int(open(use).read())
                avail = free if avail is None else min(avail, free)
            break
        except (OSError, ValueError):
            continue
    if avail is None:
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except ImportError:
            avail = 8e9
    return float(avail)
