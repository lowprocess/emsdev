"""The CPU: how many threads (or worker processes) a compiled core or a parallel map should use."""
import os


def core_threads():
    """The default thread count: one per physical core of the fastest kind (the performance cores of a hybrid CPU;
    no hyper-threads). On an i7-13700K (8 P + 8 E cores, 24 threads) gfdtd's core ran 1.2-3x faster on 8 threads
    than on 24: the barriers wait for the slowest thread. OMP_NUM_THREADS overrides it where the caller lets it."""
    base = '/sys/devices/system/cpu'
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:                                          # not Linux
        return max(1, (os.cpu_count() or 2) // 2)
    info = {}
    for c in cpus:
        rd = lambda f: open(f"{base}/cpu{c}/{f}").read().strip()
        try:
            fmax = int(rd('cpufreq/cpuinfo_max_freq'))
        except (OSError, ValueError):
            fmax = 0
        try:
            core = (rd('topology/physical_package_id'), rd('topology/core_id'))
        except OSError:
            core = ('0', str(c))
        info[c] = (fmax, core)
    top = max(f for f, _ in info.values())
    fast = {core for f, core in info.values() if f >= 0.95 * top}
    return max(1, len(fast))
