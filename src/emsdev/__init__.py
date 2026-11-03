"""
emsdev - the machine as the kicad2ems solvers see it: GPUs, CPU cores, memory, and the compiled cores' build.

mom25d and gfdtd each run standalone and under kicad2ems; this package is what they share, so a device or a
backend is found, measured and picked one way, and kicad2ems reports what they will use without importing
either:

    from emsdev import gpu, cpu, memory
    gpu.devices()                   # [Device(index, name, library, api, total)]: the GPUs CuPy and Vulkan see
    vulkan.slang_device()           # the slangpy device the solvers' Vulkan backends run on
    gpu.cupy_device()               # the current CuPy device (raises gpu.Unavailable with the reason)
    gpu.preference('auto', 'MOM25D_GPU')    # a backend setting, the environment's override applied
    cpu.core_threads()              # the CPU cores' default thread count: one per fast physical core
    memory.available()              # bytes this process may use now (cgroup aware)

    python -m emsdev                 # all of it, one line each

No dependencies: CuPy and slangpy are imported only when asked about, and their absence is an answer;
Vulkan's GPUs are found through the system's Vulkan loader (ctypes).
"""
__version__ = '261103-r1'
