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
