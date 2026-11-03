# emsdev

The machine as the [kicad2ems](https://github.com/lowprocess/kicad2ems) solvers see it: the GPUs CuPy
and Vulkan can use (name, CUDA, ROCm/HIP or Vulkan, memory), the CPU's thread count, the memory this process may use, and
the build of a solver's compiled core. [mom25d](https://github.com/lowprocess/mom25d) and
[gfdtd](https://github.com/lowprocess/gfdtd) share it, so each new backend or device rule is written once; both
also run without kicad2ems, so it is a package of its own. It has no dependencies.

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

**Vulkan** (`emsdev.vulkan`): the GPUs come from the system's Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`)
through ctypes, so the probe needs nothing installed; free memory is the driver's budget (`VK_EXT_memory_budget`).
The solvers' kernels run through [slangpy](https://pypi.org/project/slangpy/): `vulkan.slang_device()` is the
device, one per thread. `vulkan.HostBuffer(device, nbytes)` is host memory the GPU reaches by address, mapped as a
numpy array: slangpy's own read-back buffers land in uncached memory on RADV, read at 0.2-0.5 GB/s, where a
HostBuffer (host-cached memory, filled by a kernel) moves 24 GB/s over PCIe 4.0.

`emsdev.build.build_core(__file__, 'x_core.cpp', arch_env='X_ARCH')` builds a package's `core/x_core.cpp` with
pybind11 and OpenMP into `<package>/_core*.so` (`_core*.pyd` on Windows; the `build` extra installs pybind11). Linux and
macOS use g++. Windows uses MSVC (Visual Studio 2022 or its Build Tools, with *Desktop development with C++*), found
through setuptools, with `/openmp:llvm`, or, without Visual Studio, MinGW-w64's g++ on the PATH (MSYS2's UCRT64
toolchain); `EMSDEV_COMPILER=msvc` or `gcc` chooses. The compiler's runtime DLLs (`libomp140.x86_64.dll`, or
MinGW's `libgomp-1.dll` and the rest) are copied next to the module, where Python finds them.
The `*_ARCH` variable takes a g++ `-march=` (translated for MSVC: `x86-64-v3` is `/arch:AVX2`) or an MSVC `/arch:`.
