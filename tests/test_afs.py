"""The adaptive frequency sweep (emsdev.afs) that mom25d and pcbqs share: a rational model of all S entries, solved
where it is least sure."""
import numpy as np
import pytest

afs = pytest.importorskip('emsdev.afs')                             # (numpy: what it needs)


def resonator(f, f0=3e9, q=40):
    """A series RLC between two 50 ohm ports: S (2, 2) at f, its resonance at f0 (q: its reactance over 50 ohm)."""
    s = 2j * np.pi * f
    w0 = 2 * np.pi * f0
    L = q * 50 / w0
    C = 1 / (w0 * w0 * L)
    z = 5.0 + s * L + 1 / (s * C)                                 # 5 ohm of loss
    s21 = 100 / (100 + z)
    return np.array([[1 - s21, s21], [s21, 1 - s21]])


def test_a_rational_function_is_fitted_exactly():
    z = np.linspace(0.5, 6, 40)
    F = np.stack([1 / (z - (2 + 0.1j)), (z + 1) / (z * z + 1)], 1)
    r = afs.aaa(z, F)
    zz = np.linspace(0.6, 5.9, 301)
    want = np.stack([1 / (zz - (2 + 0.1j)), (zz + 1) / (zz * zz + 1)], 1)
    assert np.abs(r(zz) - want).max() < 1e-8 and len(r.support) < 12


def test_the_sweep_finds_a_resonance_between_its_first_samples():
    """A narrow resonance the five first samples step over: the sweep solves around it until two models agree,
    with fewer solves than the grid has points; a solve that gives NaN is said, with the caller's hint."""
    grid = np.linspace(1e9, 6e9, 801)
    solves = []
    g, S, fs, Ss, hist = afs.adaptive_sweep(lambda f: solves.append(f) or resonator(f), 1e9, 6e9, grid=grid,
                                            log=lambda *a: None)
    truth = np.array([resonator(f) for f in grid])
    assert hist[-1]['converged'] and len(fs) < 40 and np.abs(S - truth).max() < 5e-3
    assert np.abs(S[:, 1, 0]).max() == pytest.approx(np.abs(truth[:, 1, 0]).max(), rel=1e-3)
    with pytest.raises(FloatingPointError, match=r'non-finite S-parameters \(try this\)'):
        afs.adaptive_sweep(lambda f: resonator(f) * np.nan, 1e9, 6e9, log=lambda *a: None, hint='try this')
