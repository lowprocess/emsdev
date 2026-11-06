"""emsdev without a GPU library and with a stand-in of CuPy (the calls the solvers rely on)."""
import os
import sys
import types

import pytest

from emsdev import cpu, gpu, memory, vulkan


def fake_cupy(n=2, hip=True, mem=(8e9, 16e9)):
    cp = types.ModuleType('cupy')
    cp.__version__ = '14.0'
    cp.get_default_memory_pool = lambda: types.SimpleNamespace(free_all_blocks=lambda: None)

    class Device:
        def __init__(self, i=0):
            self.id, self.mem_info = i, mem
    cp.cuda = types.SimpleNamespace(Device=Device, runtime=types.SimpleNamespace(
        getDeviceCount=lambda: n, is_hip=hip, getDeviceProperties=lambda i: {'name': f'Fake {i}'.encode()}))
    return cp


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', None)
    monkeypatch.setitem(sys.modules, 'slangpy', None)
    monkeypatch.setattr(vulkan, 'adapters', lambda include_cpu=False: [])
    return monkeypatch


def fake_adapters(*specs):
    """vulkan.adapters() as the loader would give them: (name, kind, GB) each, best first."""
    feats = dict(bda=True, int64=True, float64=True, atomic_add_f32=True)
    out = [vulkan.Adapter(0, n, k, 0x1002, 1 + i, gb * 1e9, driver='radv', api_version='1.4.0', budget=True,
                          features=dict(feats), _order=i) for i, (n, k, gb) in enumerate(specs)]
    out.sort(key=lambda a: (a.kind != 'discrete', -a.total))
    for i, a in enumerate(out):
        a.index = i
    return out


def test_memory_and_threads():
    assert memory.available() > 0 and (memory.total() is None or memory.total() >= memory.available() * 0.5)
    assert 1 <= cpu.core_threads() <= (os.cpu_count() or 1)


def test_no_library_is_an_answer(no_gpu):
    devs, why = gpu.probe()
    assert devs == [] and len(why) == 2 and why[0].startswith('cupy:')
    assert why[1] == 'vulkan: no GPU (only CPU renderers)'
    with pytest.raises(gpu.Unavailable):
        gpu.cupy_device()


def test_cupy_devices(no_gpu):
    no_gpu.setitem(sys.modules, 'cupy', fake_cupy())
    devs, why = gpu.probe()
    assert [(d.index, d.name, d.library, d.api) for d in devs] == [(0, 'Fake 0', 'cupy', 'ROCm/HIP'),
                                                                     (1, 'Fake 1', 'cupy', 'ROCm/HIP')]
    d = gpu.cupy_device()
    assert d.hip and d.free() == 8e9 and d.total == 16e9
    assert d.as_dict() == dict(device='Fake 0', backend='cupy', api='ROCm/HIP', index=0, free=8.0, total=16.0)
    assert 'Fake 0 (ROCm/HIP, cupy 14.0, 15 GB)' == d.describe()


def test_vulkan_devices(no_gpu):
    """The Vulkan GPUs: a discrete one first, EMSDEV_VULKAN_DEVICE picks another; without slangpy, the reason."""
    ads = fake_adapters(('Radeon 780M', 'integrated', 2), ('Radeon RX 9070', 'discrete', 16))
    no_gpu.setattr(vulkan, 'adapters', lambda include_cpu=False: ads)
    no_gpu.setattr(vulkan, '_free', lambda a: 12e9)
    with pytest.raises(gpu.Unavailable, match='slangpy cannot be imported'):
        gpu.vulkan_device()
    no_gpu.setitem(sys.modules, 'slangpy', types.SimpleNamespace(__version__='0.43.1'))
    d = gpu.vulkan_device()
    assert (d.name, d.library, d.api, d.total, d.free()) == ('Radeon RX 9070', 'vulkan', 'Vulkan', 16e9, 12e9)
    assert d.describe() == 'Radeon RX 9070 (Vulkan, slangpy 0.43.1, 15 GB)'
    no_gpu.setenv('EMSDEV_VULKAN_DEVICE', '1')
    assert gpu.vulkan_device().name == 'Radeon 780M'
    no_gpu.setenv('EMSDEV_VULKAN_DEVICE', '2')
    with pytest.raises(gpu.Unavailable, match='no GPU 2'):
        gpu.vulkan_device()
    no_gpu.delenv('EMSDEV_VULKAN_DEVICE')
    assert [(d.index, d.library) for d in gpu.probe()[0]] == [(0, 'vulkan'), (1, 'vulkan')]
    ads[0].features['float64'] = False                              # what a solver's kernels need, checked
    assert gpu.vulkan_device(need=('bda',)).name == 'Radeon RX 9070'
    with pytest.raises(gpu.Unavailable, match='lacks float64'):
        gpu.vulkan_device(need=('bda', 'float64'))


def test_vulkan_loader():
    """The real loader, where there is one: an answer either way (adapters, or Unavailable with the reason)."""
    try:
        ads = vulkan.adapters(include_cpu=True)
    except vulkan.Unavailable as e:
        assert str(e).startswith('vulkan:')
        return
    for a in ads:
        assert a.name and a.kind in ('discrete', 'integrated', 'virtual', 'cpu', 'other') and a.total >= 0
        assert set(a.features) >= {'bda', 'int64', 'float64', 'atomic_add_f32'} or a.features == {}
        assert a.free() is None or a.free() >= 0


def test_auto_order(no_gpu):
    """Vulkan first, except where CuPy runs on CUDA (NVIDIA)."""
    assert gpu.auto_order() == ('vulkan', 'cupy')
    no_gpu.setitem(sys.modules, 'cupy', fake_cupy(hip=True))
    assert gpu.auto_order() == ('vulkan', 'cupy')
    no_gpu.setitem(sys.modules, 'cupy', fake_cupy(hip=False))
    assert gpu.auto_order() == ('cupy', 'vulkan')


def test_cupy_without_devices(no_gpu):
    no_gpu.setitem(sys.modules, 'cupy', fake_cupy(n=0))
    with pytest.raises(gpu.Unavailable, match='no GPU device'):
        gpu.cupy_device()


def test_preference(monkeypatch):
    monkeypatch.delenv('X_GPU', raising=False)
    assert gpu.preference('auto', 'X_GPU') == 'auto' and gpu.preference(None) == 'auto'
    assert gpu.preference('Vulkan', 'X_GPU') == 'vulkan' and gpu.preference('0') == 'off'
    assert gpu.preference('gpu', aliases={'gpu': 'cupy'}) == 'cupy'
    monkeypatch.setenv('X_GPU', 'off')
    assert gpu.preference('cupy', 'X_GPU') == 'off'


def test_build_command(tmp_path):
    pytest.importorskip('pybind11')
    from emsdev.build import command
    cmd = command(str(tmp_path), 'x_core.cpp', 'out.so', '-march=x86-64-v3', [str(tmp_path)])
    assert cmd[:3] == ['g++', '-O3', '-march=x86-64-v3'] and '-fopenmp' in cmd and '-I' + str(tmp_path) in cmd
    assert cmd[-3:] == [os.path.join(str(tmp_path), 'x_core.cpp'), '-o', 'out.so']


def test_msvc_flags():
    """MSVC: OpenMP 3 through /openmp:llvm (collapse, simd), and the -march settings translated."""
    from emsdev.build import msvc_arch, msvc_flags
    assert '/openmp:llvm' in msvc_flags() and '/arch:AVX2' in msvc_flags('-march=native')
    assert msvc_arch('-march=x86-64-v3') == '/arch:AVX2' and msvc_arch('-march=x86-64-v2') == ''
    assert msvc_arch('/arch:AVX512') == '/arch:AVX512' and msvc_arch('-march=znver4') == ''
    assert not any(a.startswith('/arch') for a in msvc_flags('-march=x86-64-v2'))


def test_find_omp_dll(tmp_path):
    """libomp140.x86_64.dll from a Visual Studio install: the newest redistributable folder first."""
    from emsdev.build import find_omp_dll, OMP_DLL
    for ver in ('14.38.33130', '14.40.33807'):
        d = tmp_path / 'VC' / 'Redist' / 'MSVC' / ver / 'x64' / 'Microsoft.VC143.OpenMP.LLVM'
        d.mkdir(parents=True)
        (d / OMP_DLL).write_bytes(b'')
    assert '14.40.33807' in find_omp_dll([str(tmp_path)])
    import os, shutil, sys
    system = os.path.join(os.environ.get('SystemRoot', r'C:\\Windows'), 'System32', OMP_DLL)
    assert find_omp_dll([str(tmp_path / 'none')]) in (None, shutil.which(OMP_DLL),           # none in that install:
                                                      system if sys.platform == 'win32' else None)   # Windows' own


def test_find_omp_dll_any_layout(tmp_path):
    """Wherever a Visual Studio version keeps it under VC: the x64 release build, never x86, ARM64 or debug."""
    from emsdev.build import find_omp_dll, vs_root, OMP_DLL
    vc = tmp_path / 'VS18' / 'VC'
    for sub in ('Tools/MSVC/14.51.36231/bin/Hostx86/x64', 'Redist/MSVC/14.51.36231/debug_nonredist/x64/OpenMP',
                'Redist/MSVC/14.51.36231/x86/OpenMP.LLVM', 'Redist/MSVC/14.51.36231/arm64/OpenMP.LLVM'):
        (vc / sub).mkdir(parents=True)
        (vc / sub / OMP_DLL).write_bytes(b'')
    hit = find_omp_dll([str(tmp_path / 'VS18')])
    assert hit == str(vc / 'Tools/MSVC/14.51.36231/bin/Hostx86/x64' / OMP_DLL)
    cl = str(vc / 'Tools/MSVC/14.51.36231/bin/Hostx86/x64/cl.exe')
    assert vs_root(cl) == str(tmp_path / 'VS18') and vs_root('cl.exe') is None and vs_root(None) is None


def test_build_through_setuptools(tmp_path, monkeypatch):
    """The Windows route (setuptools' build_ext into the package folder, the module named for this Python): with MSVC
    on Windows, elsewhere with this platform's compiler and its flags in place of MSVC's. The package sits deep in a
    folder tree, as in a venv, which once took MSVC's object paths past Windows' 260 characters."""
    pytest.importorskip('pybind11')
    pytest.importorskip('setuptools')
    import shutil, subprocess, sys, sysconfig
    from emsdev import build
    windows = sys.platform == 'win32'
    if windows and not (build.vs_installs() or shutil.which('cl')):
        pytest.skip('no Visual Studio')
    if not windows and not (shutil.which('g++') or shutil.which('c++')):
        pytest.skip('no C++ compiler')
    deep = tmp_path / 'a-rather-long-folder-name-for-a-venv' / 'Lib' / 'site-packages' / 'a-package-with-a-long-name'
    core = deep / 'pkg' / 'core'
    core.mkdir(parents=True)
    (core / 'x_core.cpp').write_text('#include <pybind11/pybind11.h>\nPYBIND11_MODULE(_core, m) '
                                     '{ m.def("two", [] { return 2; }); }\n', encoding='utf-8')
    if not windows:
        monkeypatch.setattr(build, 'msvc_flags', lambda arch: ['-O1', '-std=c++17'])
        monkeypatch.setattr(build, 'find_omp_dll', lambda roots=None: None)
    monkeypatch.setattr(build, 'compiler', lambda: 'msvc')
    cwd = os.getcwd()
    out = build.build_core(str(core / 'build.py'), 'x_core.cpp')
    assert os.getcwd() == cwd                                         # the build's chdir undone
    assert out == str((deep / 'pkg' / ('_core' + sysconfig.get_config_var('EXT_SUFFIX'))).resolve())
    r = subprocess.run([sys.executable, '-c', 'import _core; print(_core.two())'], cwd=deep / 'pkg',
                       capture_output=True, text=True)
    assert r.stdout.strip() == '2', r.stderr
    assert sorted(p.name for p in core.iterdir()) == ['x_core.cpp']  # no build folder left beside the source


def test_efficiency_classes():
    """GetLogicalProcessorInformationEx's records: a core's efficiency class at byte 9, other relations skipped."""
    from emsdev.cpu import efficiency_classes
    rec = lambda rel, cls, size=48: (rel.to_bytes(4, 'little') + size.to_bytes(4, 'little') + bytes([0, cls])
                                     + bytes(size - 10))
    buf = rec(0, 1) * 8 + rec(2, 0, 32) + rec(0, 0) * 8
    assert efficiency_classes(buf) == [1] * 8 + [0] * 8
    assert efficiency_classes(b'') == []


def test_compiler_choice(monkeypatch):
    """EMSDEV_COMPILER chooses; on Windows MSVC when Visual Studio is there, else a MinGW g++ on the PATH."""
    from emsdev import build
    monkeypatch.setenv('EMSDEV_COMPILER', 'GCC')
    assert build.compiler() == 'gcc'
    monkeypatch.delenv('EMSDEV_COMPILER')
    monkeypatch.setattr(build.sys, 'platform', 'win32')
    monkeypatch.setattr(build, 'vs_installs', lambda: [])
    monkeypatch.setattr(build.shutil, 'which', lambda n: r'C:\msys64\ucrt64\bin\g++.exe' if n == 'g++' else None)
    assert build.compiler() == 'gcc'
    monkeypatch.setattr(build, 'vs_installs', lambda: [r'C:\VS'])
    assert build.compiler() == 'msvc'


def test_mingw_command(tmp_path):
    """MinGW-w64: no -fPIC, CPython's Windows defines, this Python's DLL linked, the C++ runtime static."""
    pytest.importorskip('pybind11')
    import sys
    from emsdev.build import command
    cmd = command(str(tmp_path), 'x_core.cpp', 'out.pyd', '-march=x86-64-v3', mingw=True)
    assert '-fPIC' not in cmd and '-DMS_WIN64' in cmd and '-fopenmp' in cmd and '-static-libstdc++' in cmd
    assert cmd[-3].endswith(f"python{sys.version_info.major}{sys.version_info.minor}.dll") and cmd[-2:] == ['-o', 'out.pyd']


def test_vs_installs_without_vswhere(tmp_path, monkeypatch):
    """No vswhere: Visual Studio found by its vcvars64.bat under Program Files, the newest first (18 is 2026)."""
    from emsdev.build import vs_installs
    for pf, ver, ed in (('pf86', '2022', 'BuildTools'), ('pf', '18', 'BuildTools'), ('pf', '18', 'Community')):
        b = tmp_path / pf / 'Microsoft Visual Studio' / ver / ed / 'VC' / 'Auxiliary' / 'Build'
        b.mkdir(parents=True)
        (b / 'vcvars64.bat').write_text('', encoding='utf-8')
    (tmp_path / 'pf' / 'Microsoft Visual Studio' / 'Installer').mkdir()      # an installer folder, no vswhere in it
    monkeypatch.setenv('ProgramFiles', str(tmp_path / 'pf'))
    monkeypatch.setenv('ProgramFiles(x86)', str(tmp_path / 'pf86'))
    roots = vs_installs()
    assert [os.path.relpath(r, tmp_path).replace(os.sep, '/') for r in roots][-1] == \
        'pf86/Microsoft Visual Studio/2022/BuildTools'
    assert {os.path.basename(r) for r in roots[:2]} == {'BuildTools', 'Community'} and len(roots) == 3


def test_cgroup_free_reads_the_process_own_cgroup_and_its_parents(tmp_path):
    """A systemd scope's MemoryMax sits in the process's own cgroup, not the root; a parent slice's tighter limit
    wins; the inactive page cache counts as free; no limit anywhere: None."""
    from emsdev.memory import cgroup_free
    proc = tmp_path / 'cgroup'
    proc.write_text('0::/user.slice/app.slice/run-1.scope\n')
    root = tmp_path / 'fs'
    scope = root / 'user.slice' / 'app.slice' / 'run-1.scope'
    scope.mkdir(parents=True)
    for d in (root, root / 'user.slice', root / 'user.slice' / 'app.slice'):
        (d / 'memory.max').write_text('max\n')
        (d / 'memory.current').write_text('100\n')
    (scope / 'memory.max').write_text(str(4 << 30))
    (scope / 'memory.current').write_text(str(1 << 30))
    (scope / 'memory.stat').write_text(f'anon 1\ninactive_file {256 << 20}\n')
    assert cgroup_free(str(root), str(proc)) == (4 << 30) - (1 << 30) + (256 << 20)
    (root / 'user.slice' / 'memory.max').write_text(str(2 << 30))          # a tighter slice above
    (root / 'user.slice' / 'memory.current').write_text(str(1 << 30))
    assert cgroup_free(str(root), str(proc)) == 1 << 30
    for d in (scope, root / 'user.slice'):
        (d / 'memory.max').write_text('max\n')
    assert cgroup_free(str(root), str(proc)) is None


def test_available_sees_a_systemd_scope_cap():
    """Inside a 2 GB scope (where systemd-run is there to make one), available() is at most the cap."""
    import shutil, subprocess, sys
    if not shutil.which('systemd-run') or sys.platform != 'linux':
        pytest.skip('no systemd-run')
    r = subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryMax=2G', sys.executable, '-c',
                        'from emsdev.memory import available; print(available())'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f'systemd-run --user failed: {r.stderr.strip()[:100]}')
    assert float(r.stdout) <= 2 * 2 ** 30
