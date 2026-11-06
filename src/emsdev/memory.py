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


def _read(path):
    try:
        with open(path, encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        return None


def cgroup_free(root='/sys/fs/cgroup', proc='/proc/self/cgroup'):
    """Bytes this process's cgroup can still take: the least (limit - usage) over its own cgroup and every parent
    (a slice above can cap it too), cgroup v2 or v1's memory controller; None where no cgroup limits it. The
    process's own cgroup is the one /proc/self/cgroup names, not the root: a systemd scope's MemoryMax sits there
    (inside a container, whose root is its own cgroup, the two are the same)."""
    lines = (_read(proc) or '').splitlines()
    v2 = next((l.split(':', 2)[2] for l in lines if l.startswith('0::')), None)
    v1 = next((l.split(':', 2)[2] for l in lines if ':memory:' in l or l.split(':')[1:2] == ['memory']), None)
    cands = []
    if v2 is not None:
        cands.append((root, v2, 'memory.max', 'memory.current', 'inactive_file'))
    if v1 is not None:
        cands.append((root + '/memory', v1, 'memory.limit_in_bytes', 'memory.usage_in_bytes', 'total_inactive_file'))
    for base, path, lim, use, cache in cands:
        free, found = None, False
        parts = [x for x in path.split('/') if x]
        for k in range(len(parts), -1, -1):                     # own cgroup, then each parent up to the root
            d = '/'.join([base] + parts[:k])
            L, U = _read(f'{d}/{lim}'), _read(f'{d}/{use}')
            if L is None:
                continue
            found = True
            if L == 'max' or U is None:
                continue
            try:
                Li, Ui = int(L), int(U)
            except ValueError:
                continue
            if Li < 1 << 60:
                # usage counts the page cache: its inactive part is reclaimed before the limit bites
                stat = dict(l.split()[:2] for l in (_read(f'{d}/memory.stat') or '').splitlines() if len(l.split()) >= 2)
                try:
                    Ui -= int(stat.get(cache, 0))
                except ValueError:
                    pass
                free = Li - Ui if free is None else min(free, Li - Ui)
        if found:
            return None if free is None else max(0, free)
    return None


def available():
    """Bytes of memory available to this process: MemAvailable, capped by a cgroup limit (Linux); what Windows
    reports available; else psutil's, else 8 GB."""
    win = windows_status()
    if win:
        return win[1]
    try:
        avail = next((int(l.split()[1]) * 1024 for l in open('/proc/meminfo', encoding='utf-8') if l.startswith('MemAvailable:')), None)
    except OSError:
        avail = None
    free = cgroup_free()
    if free is not None:
        avail = free if avail is None else min(avail, free)
    if avail is None:
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except ImportError:
            avail = 8e9
    return float(avail)
