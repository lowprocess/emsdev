"""The CPU: how many threads (or worker processes) a compiled core or a parallel map should use."""
import os, sys


def core_threads():
    """The default thread count: one per physical core of the fastest kind (the performance cores of a hybrid CPU;
    no hyper-threads). On an i7-13700K (8 P + 8 E cores, 24 threads) gfdtd's core ran 1.2-3x faster on 8 threads
    than on 24: the barriers wait for the slowest thread. OMP_NUM_THREADS overrides it where the caller lets it.
    Linux reads /sys, Windows asks the kernel (windows_fast_cores); elsewhere half the logical CPUs. mom25d asks
    here too (it had its own rule: 0.9 of the top clock and every CPU, not the ones the process may use)."""
    if sys.platform == 'win32':
        return windows_fast_cores() or max(1, (os.cpu_count() or 2) // 2)
    base = '/sys/devices/system/cpu'
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:                                          # not Linux
        return max(1, (os.cpu_count() or 2) // 2)
    info = {}
    for c in cpus:
        rd = lambda f: open(f"{base}/cpu{c}/{f}", encoding='utf-8').read().strip()
        try:
            fmax = int(rd('cpufreq/cpuinfo_max_freq'))
        except (OSError, ValueError):
            fmax = 0
        try:
            core = (rd('topology/physical_package_id'), rd('topology/core_id'))
        except OSError:
            core = ('0', str(c))
        info[c] = (fmax, core)
    pcores = _listed('/sys/devices/cpu_core/cpus')                 # an Intel hybrid's P-cores, as the kernel has them
    if pcores and pcores & set(info):
        fast = {core for c, (f, core) in info.items() if c in pcores}
    else:                                                           # the clocks: within FAST of the top (0.95 had
        top = max(f for f, _ in info.values())                      # split an i9-13900K's P-cores at 5.5 / 5.8 GHz
        fast = {core for f, core in info.values() if f >= FAST * top}     # and an X3D's two dies)
    return max(1, len(fast))


FAST = 0.85                     # a core of the fast class: its top clock at least this share of the fastest's (an
                                # i7-13700K's E-cores 4.2 / 5.4 = 0.78)


def _listed(path):
    """The CPUs of a /sys list file ('0-15,20'), or an empty set."""
    try:
        text = open(path, encoding='utf-8').read().strip()
    except OSError:
        return set()
    out = set()
    for part in filter(None, text.split(',')):
        a, _, b = part.partition('-')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def efficiency_classes(buf):
    """The efficiency class of each physical core in a GetLogicalProcessorInformationEx(RelationProcessorCore)
    buffer: a list of SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records, each Relationship (DWORD), Size (DWORD),
    then PROCESSOR_RELATIONSHIP: Flags (BYTE), EfficiencyClass (BYTE), ... A higher class is a faster core (the
    P-cores of a hybrid CPU); a CPU of one kind has them all 0."""
    out, off = [], 0
    while off + 10 <= len(buf):
        rel = int.from_bytes(buf[off:off + 4], 'little')
        size = int.from_bytes(buf[off + 4:off + 8], 'little')
        if size <= 0:
            break
        if rel == 0:                                                # RelationProcessorCore
            out.append(buf[off + 9])
        off += size
    return out


def windows_fast_cores():
    """Windows: the physical cores of the highest efficiency class (a hybrid CPU's P-cores; every core of the
    others), from GetLogicalProcessorInformationEx; None where it cannot tell."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        fn = k32.GetLogicalProcessorInformationEx
        fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL
        n = wintypes.DWORD(0)
        fn(0, None, ctypes.byref(n))                                # the size it needs
        if not n.value:
            return None
        buf = ctypes.create_string_buffer(n.value)
        if not fn(0, buf, ctypes.byref(n)):
            return None
        classes = efficiency_classes(buf.raw[:n.value])
    except (AttributeError, OSError, ImportError, ValueError):
        return None
    if not classes:
        return None
    return sum(c == max(classes) for c in classes)
