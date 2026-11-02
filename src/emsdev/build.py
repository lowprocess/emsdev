"""Build a solver's optional compiled core: one C++ file and pybind11 with OpenMP, the module written next to
the package's own modules (mom25d._core, gfdtd._core). A package's core/build.py is

    from emsdev.build import build_core
    build_core(__file__, 'mom25d_core.cpp', arch_env='MOM25D_ARCH')

Linux and macOS: g++ (-fopenmp). Windows, either of:
  MSVC (Visual Studio 2022 or its Build Tools, "Desktop development with C++"), found by setuptools as for any
       extension module, with /openmp:llvm: the cores use OpenMP 3 (collapse, simd), which MSVC's default /openmp
       (2.0) rejects. Its OpenMP runtime, libomp140.x86_64.dll, is copied next to the module.
  GCC  MinGW-w64's g++ on the PATH (MSYS2's UCRT64 toolchain, or a standalone one), linked against this Python's
       DLL; its runtime DLLs (libgomp, libstdc++, libgcc, winpthread) are copied next to the module.
  MSVC when Visual Studio is installed, else g++; EMSDEV_COMPILER=msvc or gcc chooses. Python finds a module's
  DLLs in the module's own folder, so the copies need no PATH.

`arch_env` names the environment variable that replaces the default instruction set: -march=native for g++ (a
portable build: -march=x86-64-v3), /arch:AVX2 for MSVC. A -march=x86-64-vN given to MSVC is translated."""
import glob, os, shutil, subprocess, sys, sysconfig

MSVC_ARCH = {'native': '/arch:AVX2', 'x86-64': '', 'x86-64-v2': '', 'x86-64-v3': '/arch:AVX2',
             'x86-64-v4': '/arch:AVX512'}
OMP_DLL = 'libomp140.x86_64.dll'
MINGW_DLLS = ('libgomp-1.dll', 'libwinpthread-1.dll', 'libstdc++-6.dll', 'libgcc_s_seh-1.dll')


def compiler():
    """'msvc' or 'gcc': EMSDEV_COMPILER where set; on Windows MSVC when Visual Studio's C++ tools are installed
    (or cl is on the PATH), else g++ (MinGW-w64) when it is on the PATH, else 'msvc' (whose error says what is
    missing); elsewhere g++."""
    want = os.environ.get('EMSDEV_COMPILER', '').strip().lower()
    if want in ('msvc', 'gcc'):
        return want
    if sys.platform != 'win32':
        return 'gcc'
    if vs_installs() or shutil.which('cl'):
        return 'msvc'
    return 'gcc' if shutil.which('g++') else 'msvc'


def command(here, source, out, arch='-march=native', includes=()):
    """The g++ command line that builds `source` (in `here`) into the extension module `out`."""
    import pybind11
    return ['g++', '-O3', arch, '-fopenmp', '-shared', '-fPIC', '-std=c++17', '-Wno-attributes',
            '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'], *('-I' + i for i in includes),
            os.path.join(here, source), '-o', out]


def mingw_command(here, source, out, arch='-march=native', includes=(), gxx='g++'):
    """The MinGW-w64 g++ command line for Windows: as on Linux (no -fPIC), with CPython's Windows defines and its
    DLL linked directly (MinGW's ld reads a DLL's exports; no import library needed)."""
    import pybind11
    prefix = sys.base_prefix
    dll = os.path.join(prefix, f"python{sys.version_info.major}{sys.version_info.minor}.dll")
    return [gxx, '-O3', arch, '-fopenmp', '-shared', '-std=c++17', '-Wno-attributes', '-DMS_WIN64',
            '-D_hypot=hypot', '-static-libgcc', '-static-libstdc++',
            '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'], *('-I' + i for i in includes),
            os.path.join(here, source), dll, '-o', out]


def mingw_dlls(gxx='g++'):
    """The runtime DLLs a MinGW-w64 build needs at import, from g++'s own bin folder: [paths] of those there."""
    exe = shutil.which(gxx)
    if not exe:
        return []
    bindir = os.path.dirname(os.path.realpath(exe))
    return [os.path.join(bindir, d) for d in MINGW_DLLS if os.path.isfile(os.path.join(bindir, d))]


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


def _toolset_key(path):
    """Sort key for a DLL path: its MSVC toolset version (the folder after MSVC), then the path."""
    parts = path.replace('/', '\\').split('\\')
    ver = next((parts[i + 1] for i, p in enumerate(parts[:-1]) if p.lower() == 'msvc'), '0')
    return tuple(int(x) if x.isdigit() else 0 for x in ver.split('.')), path


def vs_root(cl):
    """The Visual Studio install a cl.exe belongs to (the folder above its VC folder), or None."""
    parts = os.path.normpath(cl or '').split(os.sep)
    low = [p.lower() for p in parts]
    return os.sep.join(parts[:low.index('vc')]) if 'vc' in low[:-1] else None


def find_omp_dll(installs=None):
    """The path of MSVC's LLVM OpenMP runtime (libomp140.x86_64.dll): anywhere under a Visual Studio install's VC
    folder (its redistributable or tools folder; their layout differs between versions), the x64 release build,
    newest toolset first; else Windows' own (System32, where the VC++ redistributable puts it) or the PATH; None
    where there is none."""
    for root in vs_installs() if installs is None else installs:
        hits = []
        vc = os.path.join(root, 'VC')
        for folder, _, files in os.walk(vc):
            sub = [p.lower() for p in os.path.relpath(folder, vc).replace('/', '\\').split('\\')]
            if OMP_DLL in files and 'x86' not in sub and not any('debug' in p or 'arm' in p for p in sub):
                hits.append(os.path.join(folder, OMP_DLL))
        if hits:
            return max(hits, key=_toolset_key)
    sysdir = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', OMP_DLL)
    if sys.platform == 'win32' and os.path.isfile(sysdir):
        return sysdir
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
    cl = getattr(getattr(cmd, 'compiler', None), 'cc', None)       # the cl.exe setuptools used: its install first
    roots = [r for r in [vs_root(cl)] if r] + vs_installs()
    dll = find_omp_dll(list(dict.fromkeys(roots)))
    if dll:
        shutil.copy2(dll, os.path.join(os.path.dirname(os.path.abspath(out)), OMP_DLL))
    else:
        print(f"warning: {OMP_DLL} (the OpenMP runtime of /openmp:llvm) was not found under {', '.join(roots) or 'any '
              f'Visual Studio'}, System32 or the PATH: the module loads only where Windows finds it")


def build_core(build_py, source, arch_env=None, module='_core', includes_here=False):
    """Build core/<source> (next to build_py, a package's core/build.py) into <package>/<module><EXT_SUFFIX>.
    includes_here: the core's own directory is on the include path (its headers). Returns the module's path."""
    here = os.path.dirname(os.path.abspath(build_py))
    out = os.path.join(here, '..', module + sysconfig.get_config_var('EXT_SUFFIX'))
    arch = os.environ.get(arch_env, '-march=native') if arch_env else '-march=native'
    includes = [here] if includes_here else []
    if compiler() == 'msvc':
        build_msvc(here, source, out, arch, includes)
    elif sys.platform == 'win32':
        cmd = mingw_command(here, source, out, arch, includes)
        print(' '.join(cmd))
        subprocess.check_call(cmd)
        for dll in mingw_dlls():
            shutil.copy2(dll, os.path.join(os.path.dirname(os.path.abspath(out)), os.path.basename(dll)))
    else:
        cmd = command(here, source, out, arch, includes)
        print(' '.join(cmd))
        subprocess.check_call(cmd)
    print('built', os.path.abspath(out))
    return os.path.abspath(out)
