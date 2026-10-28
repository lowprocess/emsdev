"""
The GPUs: which ones the array libraries (CuPy, PyTorch) see, what each is (name, CUDA or ROCm/HIP, memory), and
how a solver's backend setting reads (preference). A solver keeps its own backends (mom25d's LU, gfdtd's update
kernels) and asks here which device they would run on.

    gpu.cupy_device()           Device of CuPy's current device, or raises Unavailable (no CuPy, no driver, none)
    gpu.torch_device()          the same through PyTorch (its ROCm builds answer as torch.cuda)
    gpu.devices()               every device of every library that has one: [Device]
    gpu.probe()                 (devices, why): why lists what failed, one reason per library
"""
import os
from dataclasses import dataclass, field

LIBRARIES = ('cupy', 'torch')               # the order 'auto' tries them in
OFF = {'0', 'false', 'no', 'off', 'none'}
ON = {'1', 'on', 'true', 'yes', 'auto', ''}


class Unavailable(RuntimeError):
    """A library has no usable GPU; the message says why."""


@dataclass
class Device:
    index: int
    name: str
    library: str                            # 'cupy' | 'torch'
    api: str                                # 'ROCm/HIP' | 'CUDA'
    total: float = None                     # bytes of device memory
    version: str = ''                       # the library's version
    _free: object = field(default=None, repr=False, compare=False)

    @property
    def hip(self):
        return self.api == 'ROCm/HIP'

    def free(self):
        """Bytes free on the device now (the library's cached blocks released first), None if it cannot tell."""
        return float(self._free()) if self._free else None

    def describe(self):
        mem = f", {self.total / 2 ** 30:.0f} GB" if self.total else ''
        return f"{self.name} ({self.api}, {self.library} {self.version}{mem})".replace(' )', ')')

    def as_dict(self):
        """The JSON form kicad2ems's sysinfo reports (GB)."""
        free = self.free()
        return dict(device=self.name, backend=self.library, api=self.api, index=self.index,
                    free=free / 1e9 if free is not None else None, total=self.total / 1e9 if self.total else None)


def _name(v, default='GPU'):
    v = v if v else default
    return v.decode() if isinstance(v, bytes) else str(v)


def cupy_device(index=None):
    """Device of CuPy's device `index` (None: its current device)."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise Unavailable("cupy: no GPU device")
        dev = cp.cuda.Device() if index is None else cp.cuda.Device(index)
    except Unavailable:
        raise
    except Exception as e:                                          # ImportError, CUDARuntimeError, ...
        raise Unavailable(f"cupy: {e}")
    try:
        name = _name(cp.cuda.runtime.getDeviceProperties(dev.id).get('name'))
    except Exception:
        name = 'GPU'

    def free():
        cp.get_default_memory_pool().free_all_blocks()
        return cp.cuda.Device(dev.id).mem_info[0] if index is not None else cp.cuda.Device().mem_info[0]
    try:
        total = float(dev.mem_info[1])
    except Exception:
        total = None
    return Device(int(dev.id), name, 'cupy', 'ROCm/HIP' if getattr(cp.cuda.runtime, 'is_hip', False) else 'CUDA',
                  total, str(getattr(cp, '__version__', '')), free)


def torch_device(index=0):
    """Device of PyTorch's device `index` (ROCm builds report through torch.cuda)."""
    try:
        import torch
        available = torch.cuda.is_available()
    except Exception as e:
        raise Unavailable(f"torch: {e}")
    if not available:
        raise Unavailable("torch: torch.cuda.is_available() is False")

    def free():
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0]
    try:
        total = float(torch.cuda.mem_get_info()[1])
    except Exception:
        total = None
    return Device(index, _name(torch.cuda.get_device_name(index)), 'torch',
                  'ROCm/HIP' if getattr(torch.version, 'hip', None) is not None else 'CUDA', total,
                  str(getattr(torch, '__version__', '')), free)


def _count(library):
    if library == 'cupy':
        import cupy
        return cupy.cuda.runtime.getDeviceCount()
    import torch
    return torch.cuda.device_count()


def probe(libraries=LIBRARIES):
    """(devices, why): every device of each library, and what kept a library from giving any."""
    out, why = [], []
    for lib in libraries:
        one = cupy_device if lib == 'cupy' else torch_device
        try:
            first = one(0)
        except Unavailable as e:
            why.append(str(e))
            continue
        out.append(first)
        try:
            n = _count(lib)
        except Exception:
            n = 1
        for i in range(1, n):
            try:
                out.append(one(i))
            except Unavailable:
                pass
    return out, why


def devices(libraries=LIBRARIES):
    return probe(libraries)[0]


def preference(pref='auto', env=None, aliases=None):
    """A backend setting as a solver reads it: 'off', 'auto' or a backend's name (lower case). The environment
    variable `env`, when set, overrides pref; aliases maps other names onto backends (gfdtd: gpu -> cupy)."""
    p = os.environ.get(env, '') if env else ''
    p = str(p or pref or 'auto').strip().lower()
    if p in OFF:
        return 'off'
    if p in ON:
        return 'auto'
    return (aliases or {}).get(p, p)
