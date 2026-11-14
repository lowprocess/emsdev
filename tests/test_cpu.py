"""The fast cores (emsdev.cpu, which mom25d asks too) and an integrated GPU under auto (emsdev.vulkan)."""
import os
import sys

import pytest

from emsdev import cpu, gpu, vulkan


def test_a_sys_list_reads_as_its_cpus(tmp_path):
    (tmp_path / 'cpus').write_text('0-3,8,10-11\n')
    assert cpu._listed(str(tmp_path / 'cpus')) == {0, 1, 2, 3, 8, 10, 11}
    assert cpu._listed(str(tmp_path / 'none')) == set()


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux')
def test_the_fast_cores_are_those_this_process_may_use():
    """Fewer CPUs given to the process, fewer threads (mom25d's own count had taken every CPU of the machine)."""
    mine = sorted(os.sched_getaffinity(0))
    n = cpu.core_threads()
    assert 1 <= n <= len(mine)
    os.sched_setaffinity(0, mine[:1])
    try:
        assert cpu.core_threads() == 1
    finally:
        os.sched_setaffinity(0, mine)


def test_a_lone_integrated_gpu_is_not_taken_under_auto(monkeypatch):
    """Its device memory is the system's, which a solver's memory plan had counted twice: auto leaves it, naming
    it, and EMSDEV_VULKAN_DEVICE takes it."""
    from test_emsdev import fake_adapters
    ads = fake_adapters(('Radeon 780M', 'integrated', 2))
    monkeypatch.setattr(vulkan, 'adapters', lambda include_cpu=False: ads)
    monkeypatch.delenv('EMSDEV_VULKAN_DEVICE', raising=False)
    with pytest.raises(vulkan.Unavailable, match='only an integrated GPU'):
        vulkan.pick()
    monkeypatch.setenv('EMSDEV_VULKAN_DEVICE', '0')
    assert vulkan.pick().name == 'Radeon 780M'
