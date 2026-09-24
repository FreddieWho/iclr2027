"""R08 natural-E regressions: production E predicate, label machinery, lag displacement.

Every assertion calls production functions from football_repair.py / d06_roles.py.
The counterexample tests keep the two former errors detectable: "keep and flip
exist" is not E, and a per-pair passer position must not be flattened.
"""
import sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/f095_campaign'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d06_roles import quartet_search, labels_batch
from d06_oracle import channel_label
from football_repair import (natural_E_truth, natural_cell_labels, _labels_pairwise_p,
                             _player_grid, _lag_displacement, constraining_defender)


class NaturalERegression(unittest.TestCase):
    def test_E_predicate_and_legacy_keep_and_flip_mutant(self):
        self.assertTrue(natural_E_truth(1, 1, 1, 0))   # both singles keep, joint flips
        self.assertTrue(natural_E_truth(0, 0, 0, 1))   # blocked -> open direction
        self.assertFalse(natural_E_truth(1, 1, 1, 1))  # joint does not flip
        self.assertFalse(natural_E_truth(1, 0, 1, 1))  # one keep + one flip, no joint flip
        self.assertFalse(natural_E_truth(1, 1, 0, 1))
        # legacy mutant: existence of one keep and one flip over the tested edits
        legacy = lambda y0, A: any(a == y0 for a in A) and any(a != y0 for a in A)
        self.assertTrue(legacy(1, [1, 0]))             # mutant calls this E
        self.assertFalse(natural_E_truth(1, 1, 1, 1))  # production does not

    def test_production_labels_detect_search_witness(self):
        p = np.array([0., 0.]); r = np.array([10., 0.]); D = np.array([[5., .7]])
        res = quartet_search(p, r, D, .5, np.random.default_rng(1))
        self.assertTrue(res['E_found'])
        w = res['witness']
        RR = (r + w['receiver_delta'])[None]
        DD = D.copy(); DD[res['defender_index']] += w['defender_delta']; DD = DD[None]
        y0, yA, yB, yAB = natural_cell_labels(p, r, D, RR, DD, res['base_label'])
        got = [int(y0[0]), int(yA[0]), int(yB[0]), int(yAB[0])]
        self.assertEqual(got, w['labels_0_A_B_AB'])
        self.assertTrue(natural_E_truth(*got))

    def test_single_edit_flip_is_not_E(self):
        p = np.array([0., 0.]); r = np.array([10., 0.]); D = np.array([[5., .7]])
        base = int(channel_label(p, r, D, .5, 1.)['label'])
        RR = np.array([[10., 1.4]]); DD = D[None]    # receiver move alone closes the channel
        y0, yA, yB, yAB = natural_cell_labels(p, r, D, RR, DD, base)
        self.assertNotEqual(int(yA[0]), int(y0[0]))
        self.assertFalse(natural_E_truth(int(y0[0]), int(yA[0]), int(yB[0]), int(yAB[0])))

    def test_pairwise_passer_labels_match_production(self):
        p = np.array([0., 0.]); D = np.array([[5., .7], [8., 3.]])
        RR = np.array([[10., 0.], [9., 2.], [11., -3.]])
        DD = np.broadcast_to(D, (3, 2, 2)).copy()
        a = labels_batch(p, RR, DD)
        b = _labels_pairwise_p(np.broadcast_to(p, (3, 2)), RR, DD)
        np.testing.assert_array_equal(a, b)
        scalar = [channel_label(p, rr, dd, .5, 1.)['label'] for rr, dd in zip(RR, DD)]
        np.testing.assert_array_equal(b, scalar)

    def test_player_grid_and_lag_displacement(self):
        rows = [dict(game_section='h', timestamp_ms=t, person_id='a', x=t / 1000 * 5., y=0.)
                for t in (0, 40, 80)]
        rows.append(dict(game_section='h', timestamp_ms=0, person_id='b', x=1., y=0.))
        g = _player_grid(pd.DataFrame(rows))['h']
        times, col, X, Y = g
        self.assertEqual(len(times), 3)
        self.assertEqual(np.isnan(X[:, col['b']]).sum(), 2)   # absent stays NaN, never 0
        d = _lag_displacement(g, 40)
        np.testing.assert_allclose(np.sort(d[np.isfinite(d)]), [0.2, 0.2])

    def test_constraining_defender_matches_search_index(self):
        p = np.array([0., 0.]); r = np.array([10., 0.]); D = np.array([[5., .7], [3., 4.], [8., -2.]])
        res = quartet_search(p, r, D, .5, np.random.default_rng(1))
        self.assertEqual(constraining_defender(p, r, D), res['defender_index'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
