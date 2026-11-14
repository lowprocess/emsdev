# emsdev

The machine as the [kicad2ems](https://github.com/lowprocess/kicad2ems) solvers see it: the GPUs CuPy and Vulkan
can use, the CPU's thread count, the memory this process may use, and the build of a solver's compiled core.
[mom25d](https://github.com/lowprocess/mom25d) and [gfdtd](https://github.com/lowprocess/gfdtd) share it, so a
device rule is written once. It has no dependencies (`emsdev.afs` needs numpy).

```bash
pip install .
python -m emsdev            # the GPUs, CPU threads and memory
```

```python
from emsdev import gpu, cpu, memory
gpu.probe()                       # ([Device], why): every device CuPy and Vulkan see, and why one had none
gpu.cupy_device()                 # CuPy's current device, or raises gpu.Unavailable
gpu.vulkan_device(need=('float64',))   # the best Vulkan GPU (EMSDEV_VULKAN_DEVICE picks another), with those features
gpu.auto_order()                  # ('vulkan', 'cupy'), or ('cupy', 'vulkan') where CuPy runs on CUDA
gpu.preference('auto', 'MOM25D_GPU')   # 'off' | 'auto' | a backend: a setting with the environment's override
cpu.core_threads()                # one per fast physical core (no hyper-threads, no efficiency cores)
memory.available()                # bytes, capped by a cgroup limit
```

**Vulkan** (`emsdev.vulkan`): the GPUs come from the system's Vulkan loader through ctypes, so the probe needs
nothing installed; free memory is the driver's budget (`VK_EXT_memory_budget`). The solvers' kernels run through
[slangpy](https://pypi.org/project/slangpy/) on `vulkan.slang_device()`, one per thread. `vulkan.HostBuffer` is
host-cached memory the GPU reaches by address, as a numpy array: 24 GB/s over PCIe 4.0, where slangpy's own
read-back buffers read at 0.2-0.5 GB/s on RADV.

**The adaptive sweep** (`emsdev.afs`): `aaa(z, F)` fits one rational model to all S entries at once
(set-valued AAA), and `adaptive_sweep(solve, fmin, fmax, grid=f)` solves where two successive models disagree
most until they agree on the grid. mom25d's sweep and pcbqs's S-parameters use it; it needs numpy, which
they have (the rest of emsdev needs nothing).

**Compiled cores** (`emsdev.build`): `build_core(__file__, 'x_core.cpp', arch_env='X_ARCH')` builds a package's
`core/x_core.cpp` with pybind11 and OpenMP into `<package>/_core*.so` (`.pyd` on Windows; the `build` extra
installs pybind11). Linux and macOS use g++. Windows uses MSVC (Visual Studio 2022 or its Build Tools, *Desktop
development with C++*) with `/openmp:llvm`, else MinGW-w64's g++ on the PATH; `EMSDEV_COMPILER=msvc` or `gcc`
chooses. The compiler's runtime DLLs are copied next to the module. `X_ARCH` takes a g++ `-march=` (translated for
MSVC: `x86-64-v3` is `/arch:AVX2`) or an MSVC `/arch:`.
