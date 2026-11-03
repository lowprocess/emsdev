"""
The GPUs: which ones CuPy and Vulkan (emsdev.vulkan) see, what each is (name, CUDA,
ROCm/HIP or Vulkan, memory), and how a solver's backend setting reads (preference). A solver keeps its own backends (mom25d's LU, gfdtd's update
kernels) and asks here which device they would run on.

    gpu.cupy_device()           Device of CuPy's current device, or raises Unavailable (no CuPy, no driver, none)
    gpu.vulkan_device()         the best Vulkan GPU (emsdev.vulkan; slangpy runs the solvers' kernels on it)
    gpu.devices()               every device of every library that has one: [Device]
    gpu.probe()                 (devices, why): why lists what failed, one reason per library
    gpu.auto_order()            the order a solver's 'auto' tries them in: CuPy first on CUDA, else Vulkan first
"""
import os
from dataclasses import dataclass, field

LIBRARIES = ('cupy', 'vulkan')
OFF = {'0', 'false', 'no', 'off', 'none'}
ON = {'1', 'on', 'true', 'yes', 'auto', ''}


class Unavailable(RuntimeError):
    """A library has no usable GPU; the message says why."""


@dataclass
class Device:
    index: int
    name: str
    library: str                            # 'cupy' | 'vulkan'
    api: str                                # 'ROCm/HIP' | 'CUDA' | 'Vulkan'
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
        lib = self.version if self.library == 'vulkan' else f"{self.library} {self.version}"   # vulkan: 'slangpy x.y'
        return f"{self.name} ({self.api}, {lib}{mem})".replace(' )', ')').replace(', ,', ',')

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


def vulkan_device(index=None, need=()):
    """Device of Vulkan adapter `index` (None: the best, or EMSDEV_VULKAN_DEVICE's): emsdev.vulkan.device; `need`:
    the features a solver's kernels use (Unavailable where the GPU lacks one)."""
    from . import vulkan
    try:
        return vulkan.device(index, need)
    except vulkan.Unavailable as e:
        raise Unavailable(str(e))


def _count(library):
    if library == 'vulkan':
        from . import vulkan
        return len(vulkan.adapters())
    import cupy
    return cupy.cuda.runtime.getDeviceCount()


def probe(libraries=LIBRARIES):
    """(devices, why): every device of each library, and what kept a library from giving any."""
    out, why = [], []
    for lib in libraries:
        one = {'cupy': cupy_device, 'vulkan': vulkan_device}[lib]
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


def auto_order():
    """The libraries a solver's 'auto' tries, best first. Vulkan is the primary GPU API: one small wheel (slangpy)
    on AMD, Intel and NVIDIA, Windows and Linux, where CuPy needs a CUDA or ROCm stack matched to its wheel (and has
    no ROCm build for Windows). On an RX 7800 XT mom25d's Vulkan LU beat rocSOLVER and gfdtd's kernels matched
    ROCm's, so CuPy on ROCm comes after Vulkan. CuPy on CUDA (NVIDIA: cuBLAS / cuSOLVER) stays first until Vulkan
    has been measured against it there."""
    try:
        if not cupy_device().hip:
            return ('cupy', 'vulkan')
    except Unavailable:
        pass
    return ('vulkan', 'cupy')


def devices(libraries=LIBRARIES):
    return probe(libraries)[0]


def preference(pref='auto', env=None, aliases=None):
    """A backend setting as a solver reads it: 'off', 'auto' or a backend's name (lower case). The environment
    variable `env`, when set, overrides pref; aliases maps other names onto backends (gfdtd: vk -> vulkan)."""
    p = os.environ.get(env, '') if env else ''
    p = str(p or pref or 'auto').strip().lower()
    if p in OFF:
        return 'off'
    if p in ON:
        return 'auto'
    return (aliases or {}).get(p, p)
