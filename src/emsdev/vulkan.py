"""
The Vulkan GPUs, through the system's Vulkan loader (libvulkan.so.1, vulkan-1.dll: ctypes, no other dependency), and
the slangpy device a solver's Vulkan backend runs on. Vulkan is the GPU path where CuPy has none: AMD on Windows
(no ROCm CuPy there), older NVIDIA cards past CUDA's support.

    vulkan.adapters()           [Adapter]: the hardware GPUs (no CPU renderers such as llvmpipe), best first
    vulkan.device(index=None)   Device (emsdev.gpu's) of adapter `index` (None: the best; EMSDEV_VULKAN_DEVICE
                                picks it), or raises Unavailable
    vulkan.slang_device(index)  the slangpy Device on it, one per thread and adapter (slangpy: pip install slangpy)
    vulkan.device(need=('float64', ...))   the same, refused (Unavailable) where the GPU lacks a feature a solver's
                                kernels need: bda (buffer device addresses), int64, float64, atomic_add_f32

The best adapter: a discrete GPU before an integrated one, then the most device memory. Free memory is the device
heaps' budget less their use (VK_EXT_memory_budget), when the driver reports it.
"""
import ctypes, ctypes.util, os, sys, threading
from dataclasses import dataclass

VK_SUCCESS, VK_INCOMPLETE = 0, 5
TYPES = {1: 'integrated', 2: 'discrete', 3: 'virtual', 4: 'cpu'}
HEAP_DEVICE_LOCAL = 1
ST_APP, ST_INSTANCE = 0, 1
ST_MEM_PROPS2, ST_BUDGET = 1000059006, 1000237000
ST_PROPS2, ST_DRIVER_PROPS = 1000059001, 1000196000
API_1_1 = (1 << 22) | (1 << 12)


class Unavailable(RuntimeError):
    """No usable Vulkan GPU; the message says why."""


@dataclass
class Adapter:
    index: int                  # among the hardware adapters, best first (EMSDEV_VULKAN_DEVICE counts these)
    name: str
    kind: str                   # 'discrete' | 'integrated' | 'virtual'
    vendor_id: int
    device_id: int
    total: float                # bytes of device-local memory (the largest such heap)
    driver: str = ''
    api_version: str = ''
    budget: bool = False        # VK_EXT_memory_budget: free memory known
    features: dict = None       # bda, int64, float64, atomic_add_f32: what the solvers' kernels may need
    _phys: int = 0
    _order: int = 0             # the loader's order (vkEnumeratePhysicalDevices)

    def free(self):
        """Bytes of device memory free now, None if the driver cannot tell."""
        return _free(self) if self.budget else None


class _AppInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('pApplicationName', ctypes.c_char_p),
                ('applicationVersion', ctypes.c_uint32), ('pEngineName', ctypes.c_char_p),
                ('engineVersion', ctypes.c_uint32), ('apiVersion', ctypes.c_uint32)]


class _InstanceInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('flags', ctypes.c_uint32),
                ('pApplicationInfo', ctypes.POINTER(_AppInfo)), ('enabledLayerCount', ctypes.c_uint32),
                ('ppEnabledLayerNames', ctypes.c_void_p), ('enabledExtensionCount', ctypes.c_uint32),
                ('ppEnabledExtensionNames', ctypes.c_void_p)]


_state = dict(lib=None, inst=None, why=None)
_lock = threading.Lock()


def _loader():
    names = ['vulkan-1.dll'] if sys.platform == 'win32' else (
        ['libvulkan.1.dylib', 'libMoltenVK.dylib'] if sys.platform == 'darwin' else ['libvulkan.so.1', 'libvulkan.so'])
    found = ctypes.util.find_library('vulkan')
    for n in names + ([found] if found else []):
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    raise Unavailable("vulkan: no Vulkan loader (install the GPU driver)")


def _instance():
    """(loader, VkInstance), made once per process."""
    with _lock:
        if _state['inst'] is not None:
            return _state['lib'], _state['inst']
        if _state['why']:
            raise Unavailable(_state['why'])
        try:
            lib = _loader()
            app = _AppInfo(ST_APP, None, b'emsdev', 1, b'emsdev', 1, API_1_1)
            info = _InstanceInfo(ST_INSTANCE, None, 0, ctypes.pointer(app), 0, None, 0, None)
            inst = ctypes.c_void_p()
            lib.vkCreateInstance.restype = ctypes.c_int
            r = lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(inst))
            if r != VK_SUCCESS:
                raise Unavailable(f"vulkan: vkCreateInstance failed ({r}): no Vulkan 1.1 driver")
        except Unavailable as e:
            _state['why'] = str(e)
            raise
        _state['lib'], _state['inst'] = lib, inst
        return lib, inst


def _physical_devices(lib, inst):
    n = ctypes.c_uint32(0)
    lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
    arr = (ctypes.c_void_p * n.value)()
    if n.value:
        lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), arr)
    return list(arr)[:n.value]


def _extensions(lib, phys):
    n = ctypes.c_uint32(0)
    lib.vkEnumerateDeviceExtensionProperties(ctypes.c_void_p(phys), None, ctypes.byref(n), None)
    buf = ctypes.create_string_buffer(260 * max(n.value, 1))
    lib.vkEnumerateDeviceExtensionProperties(ctypes.c_void_p(phys), None, ctypes.byref(n), buf)
    return {buf.raw[i * 260:i * 260 + 256].split(b'\0', 1)[0].decode(errors='replace') for i in range(n.value)}


def _heaps(lib, phys, budget):
    """[(size, device_local, budget, usage)] of the device's memory heaps."""
    bud = ctypes.create_string_buffer(272)
    ctypes.c_int.from_buffer(bud, 0).value = ST_BUDGET
    props = ctypes.create_string_buffer(536 + 64)
    ctypes.c_int.from_buffer(props, 0).value = ST_MEM_PROPS2
    ctypes.c_void_p.from_buffer(props, 8).value = ctypes.addressof(bud) if budget else None
    lib.vkGetPhysicalDeviceMemoryProperties2(ctypes.c_void_p(phys), props)
    base = 16                                                           # VkPhysicalDeviceMemoryProperties
    nheap = ctypes.c_uint32.from_buffer(props, base + 4 + 32 * 8).value
    out = []
    for h in range(nheap):
        off = base + 264 + 16 * h
        size = ctypes.c_uint64.from_buffer(props, off).value
        flags = ctypes.c_uint32.from_buffer(props, off + 8).value
        b = ctypes.c_uint64.from_buffer(bud, 16 + 8 * h).value if budget else 0
        u = ctypes.c_uint64.from_buffer(bud, 144 + 8 * h).value if budget else 0
        out.append((size, bool(flags & HEAP_DEVICE_LOCAL), b, u))
    return out


def _driver(lib, phys):
    """The driver's name and version text (VkPhysicalDeviceDriverProperties, Vulkan 1.2), '' if not reported."""
    drv = ctypes.create_string_buffer(16 + 4 + 256 + 256 + 8 + 64)
    ctypes.c_int.from_buffer(drv, 0).value = ST_DRIVER_PROPS
    props = ctypes.create_string_buffer(16 + 2048)
    ctypes.c_int.from_buffer(props, 0).value = ST_PROPS2
    ctypes.c_void_p.from_buffer(props, 8).value = ctypes.addressof(drv)
    try:
        lib.vkGetPhysicalDeviceProperties2(ctypes.c_void_p(phys), props)
    except AttributeError:
        return ''
    name = drv.raw[20:276].split(b'\0', 1)[0].decode(errors='replace')
    info = drv.raw[276:532].split(b'\0', 1)[0].decode(errors='replace')
    return f"{name} {info}".strip()


def _features(lib, phys):
    """The features the solvers' kernels use: buffer device addresses, 64-bit integers and floats (shaderInt64,
    shaderFloat64), float32 atomic adds on buffers (VK_EXT_shader_atomic_float)."""
    atom = ctypes.create_string_buffer(16 + 4 * 16)
    ctypes.c_int.from_buffer(atom, 0).value = 1000260000            # ..._SHADER_ATOMIC_FLOAT_FEATURES_EXT
    bda = ctypes.create_string_buffer(16 + 4 * 4)
    ctypes.c_int.from_buffer(bda, 0).value = 1000257000             # ..._BUFFER_DEVICE_ADDRESS_FEATURES
    ctypes.c_void_p.from_buffer(bda, 8).value = ctypes.addressof(atom) if 'VK_EXT_shader_atomic_float' in _extensions(lib, phys) else None
    f2 = ctypes.create_string_buffer(16 + 4 * 64)
    ctypes.c_int.from_buffer(f2, 0).value = 1000059000              # ..._FEATURES_2
    ctypes.c_void_p.from_buffer(f2, 8).value = ctypes.addressof(bda)
    try:
        lib.vkGetPhysicalDeviceFeatures2(ctypes.c_void_p(phys), f2)
    except AttributeError:
        return {}
    b = lambda buf, off: bool(ctypes.c_uint32.from_buffer(buf, off).value)
    return dict(bda=b(bda, 16), float64=b(f2, 16 + 4 * 39), int64=b(f2, 16 + 4 * 40),
                atomic_add_f32=b(atom, 20))


def _free(a):
    lib, _ = _instance()
    heaps = [h for h in _heaps(lib, a._phys, True) if h[1]] or _heaps(lib, a._phys, True)
    big = max(heaps, key=lambda h: h[0])
    return float(max(big[2] - big[3], 0))


def adapters(include_cpu=False):
    """The Vulkan GPUs the loader finds, best first (CPU renderers left out unless include_cpu)."""
    lib, inst = _instance()
    out = []
    for order, phys in enumerate(_physical_devices(lib, inst)):
        props = ctypes.create_string_buffer(2048)
        lib.vkGetPhysicalDeviceProperties(ctypes.c_void_p(phys), props)
        api, _, vendor, dev, kind = (ctypes.c_uint32.from_buffer(props, 4 * k).value for k in range(5))
        kind = TYPES.get(kind, 'other')
        if kind == 'cpu' and not include_cpu:
            continue
        name = props.raw[20:276].split(b'\0', 1)[0].decode(errors='replace')
        budget = 'VK_EXT_memory_budget' in _extensions(lib, phys)
        heaps = _heaps(lib, phys, False)
        local = [h[0] for h in heaps if h[1]] or [0]
        out.append(Adapter(0, name, kind, vendor, dev, float(max(local)), _driver(lib, phys),
                           f"{api >> 22}.{(api >> 12) & 0x3ff}.{api & 0xfff}", budget, _features(lib, phys), phys,
                           order))
    out.sort(key=lambda a: (a.kind != 'discrete', a.kind != 'integrated', -a.total, a._order))
    for i, a in enumerate(out):
        a.index = i
    return out


def pick(index=None):
    """The adapter a solver runs on: `index` (EMSDEV_VULKAN_DEVICE when None), else the best."""
    try:
        ads = adapters()
    except Unavailable:
        raise
    except Exception as e:                                          # a broken driver: an answer, not a crash
        raise Unavailable(f"vulkan: {type(e).__name__}: {e}")
    if not ads:
        raise Unavailable("vulkan: no GPU (only CPU renderers)")
    if index is None:
        env = os.environ.get('EMSDEV_VULKAN_DEVICE', '').strip()
        index = int(env) if env.isdigit() else 0
    if not 0 <= index < len(ads):
        raise Unavailable(f"vulkan: no GPU {index} (there are {len(ads)})")
    return ads[index]


def device(index=None, need=()):
    """emsdev.gpu's Device of a Vulkan adapter (library 'vulkan', api 'Vulkan'). Needs slangpy, the solvers' Vulkan
    runtime: without it the GPU is there but nothing can run on it; and the features in `need`."""
    from .gpu import Device
    a = pick(index)
    lacks = [f for f in need if a.features is not None and not a.features.get(f, False)]
    if lacks:
        raise Unavailable(f"vulkan: {a.name} lacks {', '.join(lacks)} (its driver: {a.driver or '?'})")
    try:
        import slangpy                                              # noqa: F401
        version = getattr(slangpy, '__version__', '')
    except Exception as e:
        raise Unavailable(f"vulkan: {a.name} found, but slangpy cannot be imported ({e}): pip install slangpy")
    return Device(a.index, a.name, 'vulkan', 'Vulkan', a.total or None, f"slangpy {version}".strip(),
                  (lambda: a.free()) if a.budget else None)


_slang = threading.local()


def slang_device(index=None):
    """The slangpy Device on adapter `index` (pick's), made once per thread: the solvers' runs in threads each get
    their own (a Vulkan device and queue each, so their work overlaps on the GPU)."""
    a = pick(index)
    cache = getattr(_slang, 'devices', None)
    if cache is None:
        cache = _slang.devices = {}
    key = (a.vendor_id, a.device_id, a.name, a._order)
    if key not in cache:
        import slangpy as spy
        match = [x for x in spy.Device.enumerate_adapters(type=spy.DeviceType.vulkan) if x.name == a.name and
                 x.vendor_id == a.vendor_id and x.device_id == a.device_id]
        if not match:
            raise Unavailable(f"vulkan: slangpy does not list {a.name}")
        cache[key] = spy.Device(type=spy.DeviceType.vulkan, adapter_luid=match[0].luid, enable_hot_reload=False,
                                shader_cache_path=_cache_dir('shaders'), module_cache_path=_cache_dir('modules'))
    return cache[key]


def _cache_dir(what):
    base = os.environ.get('LOCALAPPDATA') if sys.platform == 'win32' else (
        os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'))
    path = os.path.join(base or os.path.expanduser('~'), 'emsdev', what)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path


# ---- host memory the GPU reaches by address ----------------------------------------------------------------------
# slangpy's read_back buffers land in uncached host memory on RADV: numpy reads them at 0.2-0.5 GB/s. A HostBuffer is
# allocated here instead, on the slangpy device's VkDevice, in host-cached memory where there is one (else host
# memory as the driver gives it), mapped once: the host reads and writes it as a numpy array, the kernels by its
# device address (copies to and from it are kernels: no command recording outside slangpy).

class _BufferInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('flags', ctypes.c_uint32),
                ('size', ctypes.c_uint64), ('usage', ctypes.c_uint32), ('sharingMode', ctypes.c_int),
                ('queueFamilyIndexCount', ctypes.c_uint32), ('pQueueFamilyIndices', ctypes.c_void_p)]


class _MemReq(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint64), ('alignment', ctypes.c_uint64), ('memoryTypeBits', ctypes.c_uint32)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('flags', ctypes.c_uint32),
                ('deviceMask', ctypes.c_uint32)]


class _AllocInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('allocationSize', ctypes.c_uint64),
                ('memoryTypeIndex', ctypes.c_uint32)]


class _AddrInfo(ctypes.Structure):
    _fields_ = [('sType', ctypes.c_int), ('pNext', ctypes.c_void_p), ('buffer', ctypes.c_uint64)]


HOST_VISIBLE, HOST_COHERENT, HOST_CACHED, DEVICE_LOCAL = 2, 4, 8, 1
USAGE_ALL = 0x1 | 0x2 | 0x20 | 0x20000           # transfer src / dst, storage, shader device address
_fn_cache = {}


def _device_fn(lib, vkdev, name, *argtypes):
    key = (vkdev, name)
    if key not in _fn_cache:
        gdpa = lib.vkGetDeviceProcAddr
        gdpa.restype, gdpa.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_char_p]
        ptr = gdpa(ctypes.c_void_p(vkdev), name.encode())
        if not ptr:
            raise Unavailable(f"vulkan: no {name}")
        _fn_cache[key] = ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(ptr)
    return _fn_cache[key]


def _memory_types(lib, phys):
    props = ctypes.create_string_buffer(520 + 64)
    lib.vkGetPhysicalDeviceMemoryProperties(ctypes.c_void_p(phys), props)
    n = ctypes.c_uint32.from_buffer(props, 0).value
    return [(ctypes.c_uint32.from_buffer(props, 4 + 8 * i).value, ctypes.c_uint32.from_buffer(props, 8 + 8 * i).value)
            for i in range(n)]


class HostBuffer:
    """`nbytes` of host memory on a slangpy device's VkDevice: .array (numpy uint8, mapped), .address (for kernels),
    .cached (host-cached memory: fast host reads). Freed with free() or when collected."""
    def __init__(self, slang_dev, nbytes):
        import numpy as np
        lib = _loader() if _state['lib'] is None else _state['lib']
        handles = {str(h.type).split('.')[-1]: h.value for h in slang_dev.native_handles}
        dev, phys = handles['VkDevice'], handles['VkPhysicalDevice']
        self._lib, self._dev, self.nbytes = lib, dev, int(nbytes)
        vp, u64 = ctypes.c_void_p, ctypes.c_uint64
        create = _device_fn(lib, dev, 'vkCreateBuffer', vp, vp, vp, vp)
        reqs = _device_fn(lib, dev, 'vkGetBufferMemoryRequirements', vp, u64, vp)
        alloc = _device_fn(lib, dev, 'vkAllocateMemory', vp, vp, vp, vp)
        bind = _device_fn(lib, dev, 'vkBindBufferMemory', vp, u64, u64, u64)
        mapf = _device_fn(lib, dev, 'vkMapMemory', vp, u64, u64, u64, ctypes.c_uint32, vp)
        addr = _device_fn(lib, dev, 'vkGetBufferDeviceAddress', vp, vp)
        addr.restype = u64
        self._buf, self._mem = u64(0), u64(0)
        info = _BufferInfo(12, None, 0, max(self.nbytes, 16), USAGE_ALL, 0, 0, None)
        r = create(vp(dev), ctypes.byref(info), None, ctypes.byref(self._buf))
        if r != VK_SUCCESS:
            raise MemoryError(f"vulkan: vkCreateBuffer ({r})")
        req = _MemReq()
        reqs(vp(dev), self._buf, ctypes.byref(req))
        types = _memory_types(lib, phys)
        want = [HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, HOST_VISIBLE | HOST_COHERENT]
        idx = None
        for w in want:
            for i, (flags, _) in enumerate(types):
                if req.memoryTypeBits >> i & 1 and flags & w == w and not flags & DEVICE_LOCAL:
                    idx = i
                    break
            if idx is not None:
                break
        if idx is None:
            self.free()
            raise MemoryError("vulkan: no host-visible memory for a host buffer")
        self.cached = bool(types[idx][0] & HOST_CACHED)
        fl = _AllocFlags(1000060000, None, 0x2, 0)               # VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
        ai = _AllocInfo(5, ctypes.cast(ctypes.pointer(fl), vp), req.size, idx)
        r = alloc(vp(dev), ctypes.byref(ai), None, ctypes.byref(self._mem))
        if r != VK_SUCCESS:
            self.free()
            raise MemoryError(f"vulkan: {self.nbytes / 1e9:.2f} GB of host memory for the GPU ({r})")
        bind(vp(dev), self._buf, self._mem, 0)
        ptr = vp()
        r = mapf(vp(dev), self._mem, 0, ctypes.c_uint64(~0 & 0xFFFFFFFFFFFFFFFF), 0, ctypes.byref(ptr))
        if r != VK_SUCCESS:
            self.free()
            raise MemoryError(f"vulkan: vkMapMemory ({r})")
        self.array = np.ctypeslib.as_array((ctypes.c_uint8 * self.nbytes).from_address(ptr.value))
        self.address = int(addr(vp(dev), ctypes.byref(_AddrInfo(1000244001, None, self._buf.value))))
        # freed before slangpy closes the device (at exit, say), however the objects are collected
        import weakref
        me = weakref.ref(self)

        def closing(*_):
            h = me()
            if h is not None:
                h.free()
        self._closing = closing
        try:
            slang_dev.register_device_close_callback(closing)
            self._slang = weakref.ref(slang_dev)
        except Exception:
            self._slang = None

    def view(self, dtype, count=None):
        import numpy as np
        a = self.array.view(dtype)
        return a if count is None else a[:count]

    def free(self):
        """Unmap and free (idempotent)."""
        vp, u64 = ctypes.c_void_p, ctypes.c_uint64
        dev = self._slang() if getattr(self, '_slang', None) is not None else None
        if dev is not None and getattr(self, '_closing', None) is not None:
            try:
                dev.unregister_device_close_callback(self._closing)
            except Exception:
                pass
            self._closing = None
        if getattr(self, '_mem', None) is not None and self._mem.value:
            self.array = None
            _device_fn(self._lib, self._dev, 'vkUnmapMemory', vp, u64)(vp(self._dev), self._mem)
        if getattr(self, '_buf', None) is not None and self._buf.value:
            _device_fn(self._lib, self._dev, 'vkDestroyBuffer', vp, u64, vp)(vp(self._dev), self._buf, None)
            self._buf = u64(0)
        if getattr(self, '_mem', None) is not None and self._mem.value:
            _device_fn(self._lib, self._dev, 'vkFreeMemory', vp, u64, vp)(vp(self._dev), self._mem, None)
            self._mem = u64(0)

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
