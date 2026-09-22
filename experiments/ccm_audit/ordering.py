"""U2 ordering diagnostic primitives (pure numpy, no torch).

G = s*f_AB - max(s*f_A, s*f_B), s = 2*y_AB - 1.
A single threshold classifies A/B/AB all correctly iff G > 0
(strict separation; boundary equality reported separately).
Positive-affine and strictly-monotone scalar recalibrations preserve
sign(G) and hence H = P(G > 0).
"""
import numpy as np


def G_of(fA, fB, fAB, yAB):
    fA = np.asarray(fA, float)
    fB = np.asarray(fB, float)
    fAB = np.asarray(fAB, float)
    s = 2 * np.asarray(yAB, float) - 1.0
    return s * fAB - np.maximum(s * fA, s * fB)


def separable_mask(fA, fB, fAB, yAB):
    return G_of(fA, fB, fAB, yAB) > 0


def boundary_mask(fA, fB, fAB, yAB):
    return G_of(fA, fB, fAB, yAB) == 0


def H_of(fA, fB, fAB, yAB):
    g = G_of(fA, fB, fAB, yAB)
    return float(np.mean(g > 0)), {"n": int(g.size),
                                   "n_boundary": int(np.sum(g == 0))}


def fixed_threshold_J(fA, fB, fAB, yA, yB, yAB, t=0.0):
    pA = (np.asarray(fA, float) > t).astype(int)
    pB = (np.asarray(fB, float) > t).astype(int)
    pAB = (np.asarray(fAB, float) > t).astype(int)
    ok = ((pA == np.asarray(yA, int)) & (pB == np.asarray(yB, int))
          & (pAB == np.asarray(yAB, int)))
    return float(np.mean(ok))
