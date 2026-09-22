#!/usr/bin/env python3
"""Minimal numerical sanity check for equal-energy graph action modes."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def path_laplacian(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    d = np.diag(a.sum(axis=1))
    return d - a


def main() -> None:
    n = 10
    eps = 1.0
    l = path_laplacian(n)
    vals, vecs = eigh(l)
    assert abs(vals[0]) < 1e-9
    common = vecs[:, 0]
    assert np.std(common) < 1e-8

    energies = []
    rayleigh = []
    for k in range(n):
        delta = vecs[:, k][:, None] * np.array([[1.0, 0.0]])
        delta *= eps / np.linalg.norm(delta)
        energies.append(float(np.linalg.norm(delta)))
        x = delta[:, 0]
        rayleigh.append(float(x @ l @ x / (x @ x)))

    assert max(abs(e - eps) for e in energies) < 1e-10
    assert np.all(np.diff(rayleigh) >= -1e-10)
    print("spectral smoke: PASS")
    print("eigenvalues:", np.round(vals, 6).tolist())
    print("energies:", np.round(energies, 6).tolist())


if __name__ == "__main__":
    main()
