"""emsdev without a GPU library and with stand-ins of CuPy and PyTorch (the calls the solvers rely on)."""
import os
import sys
import types

import pytest

from emsdev import cpu, gpu, memory


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
    monkeypatch.setitem(sys.modules, 'torch', None)
    return monkeypatch


def test_memory_and_threads():
    assert memory.available() > 0 and (memory.total() is None or memory.total() >= memory.available() * 0.5)
    assert 1 <= cpu.core_threads() <= (os.cpu_count() or 1)


def test_no_library_is_an_answer(no_gpu):
    devs, why = gpu.probe()
    assert devs == [] and len(why) == 2 and why[0].startswith('cupy:') and why[1].startswith('torch:')
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


def test_cupy_without_devices(no_gpu):
    no_gpu.setitem(sys.modules, 'cupy', fake_cupy(n=0))
    with pytest.raises(gpu.Unavailable, match='no GPU device'):
        gpu.cupy_device()


def test_preference(monkeypatch):
    monkeypatch.delenv('X_GPU', raising=False)
    assert gpu.preference('auto', 'X_GPU') == 'auto' and gpu.preference(None) == 'auto'
    assert gpu.preference('Torch', 'X_GPU') == 'torch' and gpu.preference('0') == 'off'
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
    assert find_omp_dll([str(tmp_path / 'none')]) in (None, __import__('shutil').which(OMP_DLL))


def test_build_through_setuptools(tmp_path, monkeypatch):
    """The Windows route (setuptools' build_ext into the package folder, the module named for this Python), run
    here with this platform's compiler and its flags in place of MSVC's."""
    pytest.importorskip('pybind11')
    pytest.importorskip('setuptools')
    import shutil, subprocess, sys, sysconfig
    if not (shutil.which('g++') or shutil.which('c++')):
        pytest.skip('no C++ compiler')
    from emsdev import build
    core = tmp_path / 'pkg' / 'core'
    core.mkdir(parents=True)
    (core / 'x_core.cpp').write_text('#include <pybind11/pybind11.h>\nPYBIND11_MODULE(_core, m) '
                                     '{ m.def("two", [] { return 2; }); }\n')
    monkeypatch.setattr(build, 'msvc_flags', lambda arch: ['-O1', '-std=c++17'])
    monkeypatch.setattr(build, 'find_omp_dll', lambda: None)
    monkeypatch.setattr(build, 'compiler', lambda: 'msvc')
    out = build.build_core(str(core / 'build.py'), 'x_core.cpp')
    assert out == str((tmp_path / 'pkg' / ('_core' + sysconfig.get_config_var('EXT_SUFFIX'))).resolve())
    r = subprocess.run([sys.executable, '-c', 'import _core; print(_core.two())'], cwd=tmp_path / 'pkg',
                       capture_output=True, text=True)
    assert r.stdout.strip() == '2', r.stderr
    assert not (core / 'build-msvc').exists()


def test_efficiency_classes():
    """GetLogicalProcessorInformationEx's records: a core's efficiency class at byte 9, other relations skipped."""
    from emsdev.cpu import efficiency_classes
    rec = lambda rel, cls, size=48: (rel.to_bytes(4, 'little') + size.to_bytes(4, 'little') + bytes([0, cls])
                                     + bytes(size - 10))
    buf = rec(0, 1) * 8 + rec(2, 0, 32) + rec(0, 0) * 8
    assert efficiency_classes(buf) == [1] * 8 + [0] * 8
    assert efficiency_classes(b'') == []
