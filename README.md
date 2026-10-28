# emsdev

The machine as the [kicad2ems](https://github.com/lowprocess/kicad2ems) solvers see it: the GPUs CuPy and
PyTorch can use (name, CUDA or ROCm/HIP, memory), the CPU's thread count, the memory this process may use, and
the build of a solver's compiled core. [mom25d](https://github.com/lowprocess/mom25d) and
[gfdtd](https://github.com/lowprocess/gfdtd) share it, so each new backend or device rule is written once; both
also run without kicad2ems, so it is a package of its own. It has no dependencies.

```bash
pip install .
python -m emsdev            # the GPUs, CPU threads and memory
```

```python
from emsdev import gpu, cpu, memory
gpu.probe()                       # ([Device], why): every device CuPy and PyTorch see, and why one had none
gpu.cupy_device()                 # CuPy's current device, or raises gpu.Unavailable
gpu.preference('auto', 'MOM25D_GPU')   # 'off' | 'auto' | a backend: a setting with the environment's override
cpu.core_threads()                # one per fast physical core (no hyper-threads, no efficiency cores)
memory.available()                # bytes, capped by a cgroup limit
```

`emsdev.build.build_core(__file__, 'x_core.cpp', arch_env='X_ARCH')` builds a package's `core/x_core.cpp` with
pybind11 and g++ (OpenMP) into `<package>/_core*.so` (the `build` extra installs pybind11).
