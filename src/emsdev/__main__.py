"""python -m emsdev: the GPUs, CPU threads and memory, one line each."""
from . import cpu, gpu, memory


def main():
    devs, why = gpu.probe()
    for d in devs:
        free = d.free()
        print(f"GPU {d.index}: {d.describe()}" + (f", {free / 1e9:.1f} GB free" if free is not None else ''))
    for w in why:
        print(f"no GPU: {w}")
    total = memory.total()
    print(f"CPU: {cpu.core_threads()} core threads; memory {memory.available() / 1e9:.1f} GB available"
          + (f" of {total / 1e9:.1f} GB" if total else ''))


if __name__ == '__main__':
    main()
