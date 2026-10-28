"""Build a solver's optional compiled core: one C++ file and pybind11, g++ with OpenMP, the module written next to
the package's own modules (mom25d._core, gfdtd._core). A package's core/build.py is

    from emsdev.build import build_core
    build_core(__file__, 'mom25d_core.cpp', arch_env='MOM25D_ARCH')

`arch_env` names the environment variable that replaces -march=native (a portable build: -march=x86-64-v3)."""
import os, subprocess, sysconfig


def command(here, source, out, arch='-march=native', includes=()):
    """The g++ command line that builds `source` (in `here`) into the extension module `out`."""
    import pybind11
    return ['g++', '-O3', arch, '-fopenmp', '-shared', '-fPIC', '-std=c++17', '-Wno-attributes',
            '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'], *('-I' + i for i in includes),
            os.path.join(here, source), '-o', out]


def build_core(build_py, source, arch_env=None, module='_core', includes_here=False):
    """Build core/<source> (next to build_py, a package's core/build.py) into <package>/<module><EXT_SUFFIX>.
    includes_here: the core's own directory is on the include path (its headers). Returns the module's path."""
    here = os.path.dirname(os.path.abspath(build_py))
    out = os.path.join(here, '..', module + sysconfig.get_config_var('EXT_SUFFIX'))
    arch = os.environ.get(arch_env, '-march=native') if arch_env else '-march=native'
    cmd = command(here, source, out, arch, [here] if includes_here else ())
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    print('built', os.path.abspath(out))
    return os.path.abspath(out)
