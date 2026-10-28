"""System memory: installed, and available to this process now."""
import os


def total():
    """Installed RAM in bytes (None where sysconf cannot tell)."""
    try:
        return float(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except (ValueError, OSError, AttributeError):
        return None


def available():
    """Bytes of memory available to this process: MemAvailable, capped by a cgroup limit."""
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
