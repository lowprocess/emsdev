"""Build a solver's optional compiled core: one C++ file and pybind11 with OpenMP, the module written next to
the package's own modules (mom25d._core, gfdtd._core). A package's core/build.py is

    from emsdev.build import build_core
    build_core(__file__, 'mom25d_core.cpp', arch_env='MOM25D_ARCH')

Linux and macOS: g++ (-fopenmp). Windows: MSVC (Visual Studio 2022 or its Build Tools, "Desktop development with
C++"), found by setuptools as for any extension module, with /openmp:llvm: the cores use OpenMP 3 (collapse, simd),
which MSVC's default /openmp (2.0) rejects. That OpenMP's runtime, libomp140.x86_64.dll, is copied next to the
module, where Python finds a module's DLLs.

`arch_env` names the environment variable that replaces the default instruction set: -march=native for g++ (a
portable build: -march=x86-64-v3), /arch:AVX2 for MSVC. A -march=x86-64-vN given to MSVC is translated."""
import glob, os, shutil, subprocess, sys, sysconfig

MSVC_ARCH = {'native': '/arch:AVX2', 'x86-64': '', 'x86-64-v2': '', 'x86-64-v3': '/arch:AVX2',
             'x86-64-v4': '/arch:AVX512'}
OMP_DLL = 'libomp140.x86_64.dll'


def compiler():
    """'msvc' on Windows, else 'gcc' (g++)."""
    return 'msvc' if sys.platform == 'win32' else 'gcc'


def command(here, source, out, arch='-march=native', includes=()):
    """The g++ command line that builds `source` (in `here`) into the extension module `out`."""
    import pybind11
    return ['g++', '-O3', arch, '-fopenmp', '-shared', '-fPIC', '-std=c++17', '-Wno-attributes',
            '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'], *('-I' + i for i in includes),
            os.path.join(here, source), '-o', out]


def msvc_arch(arch):
    """An instruction-set setting as MSVC's /arch flag: an /arch: flag as it is, a -march=X translated ('' for
    MSVC's default, SSE2), '' for anything else."""
    arch = (arch or '').strip()
    if arch.startswith('/'):
        return arch
    if arch.startswith('-march='):
        return MSVC_ARCH.get(arch[len('-march='):], '')
    return ''


def msvc_flags(arch='-march=native'):
    """cl's compile flags for a core: optimised, C++17, exceptions (pybind11), OpenMP 3 through LLVM's runtime."""
    flags = ['/O2', '/std:c++17', '/EHsc', '/bigobj', '/openmp:llvm', '/DNOMINMAX', '/D_USE_MATH_DEFINES']
    a = msvc_arch(arch)
    return flags + ([a] if a else [])


def vs_installs():
    """Visual Studio installations with the C++ tools (vswhere), newest first; [] where there is no vswhere."""
    base = os.environ.get('ProgramFiles(x86)') or os.environ.get('ProgramFiles') or r'C:\Program Files (x86)'
    vswhere = os.path.join(base, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
    if not os.path.isfile(vswhere):
        return []
    try:
        out = subprocess.run([vswhere, '-products', '*', '-sort', '-requires',
                              'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def find_omp_dll(installs=None):
    """The path of MSVC's LLVM OpenMP runtime (libomp140.x86_64.dll): a Visual Studio install's redistributable
    or tools folder, else the PATH; None where there is none."""
    for root in vs_installs() if installs is None else installs:
        hits = sorted(glob.glob(os.path.join(root, 'VC', 'Redist', 'MSVC', '*', 'x64', '*OpenMP.LLVM', OMP_DLL))) + \
            sorted(glob.glob(os.path.join(root, 'VC', 'Tools', 'MSVC', '*', 'bin', 'Hostx64', 'x64', OMP_DLL)))
        if hits:
            return hits[-1]                                         # the newest toolset
    return shutil.which(OMP_DLL)


def build_msvc(here, source, out, arch='-march=native', includes=()):
    """Build `source` (in `here`) with MSVC into the extension module `out` (setuptools finds the compiler and
    sets its environment), then put the OpenMP runtime beside it."""
    import pybind11
    from setuptools import Distribution, Extension
    name = os.path.basename(out)[:-len(sysconfig.get_config_var('EXT_SUFFIX'))]
    ext = Extension(name, [os.path.join(here, source)], language='c++',
                    include_dirs=[pybind11.get_include(), *includes], extra_compile_args=msvc_flags(arch))
    dist = Distribution({'name': name, 'ext_modules': [ext]})
    cmd = dist.get_command_obj('build_ext')
    cmd.build_lib = os.path.dirname(os.path.abspath(out))           # the module lands here, named for this Python
    cmd.build_temp = os.path.join(here, 'build-msvc')
    cmd.inplace = False
    cmd.ensure_finalized()
    print('cl', ' '.join(msvc_flags(arch)), os.path.join(here, source))
    cmd.run()
    shutil.rmtree(cmd.build_temp, ignore_errors=True)
    dll = find_omp_dll()
    if dll:
        shutil.copy2(dll, os.path.join(os.path.dirname(os.path.abspath(out)), OMP_DLL))
    else:
        print(f"warning: {OMP_DLL} (the OpenMP runtime of /openmp:llvm) was not found in Visual Studio: the module "
              f"loads only where it is on the PATH (Visual Studio 2022's C++ tools have it)")


def build_core(build_py, source, arch_env=None, module='_core', includes_here=False):
    """Build core/<source> (next to build_py, a package's core/build.py) into <package>/<module><EXT_SUFFIX>.
    includes_here: the core's own directory is on the include path (its headers). Returns the module's path."""
    here = os.path.dirname(os.path.abspath(build_py))
    out = os.path.join(here, '..', module + sysconfig.get_config_var('EXT_SUFFIX'))
    arch = os.environ.get(arch_env, '-march=native') if arch_env else '-march=native'
    includes = [here] if includes_here else []
    if compiler() == 'msvc':
        build_msvc(here, source, out, arch, includes)
    else:
        cmd = command(here, source, out, arch, includes)
        print(' '.join(cmd))
        subprocess.check_call(cmd)
    print('built', os.path.abspath(out))
    return os.path.abspath(out)
