# P2 rigid formal report draft

Status: `RIGID_C2_RESULTS_READY_FOR_SCIENTIFIC_SYNTHESIS`

This run completed rigid matching, frozen-model response, and match-level statistics. The fracture operator remains `NOT_RUN_SEPARATE_OPERATOR` and is not part of these results.

## Primary low-energy common-support summary

```text
architecture                 control_family graph_id  epsilon  n_matches  mean_normalized_scg  normalized_ci_low  normalized_ci_high  positive_match_fraction
 DeepSets-AE            arbitrary_same_team delaunay     0.25         10             0.000007      -1.366398e-07            0.000013                      0.7
 DeepSets-AE            arbitrary_same_team delaunay     0.50         10             0.000002      -6.074260e-06            0.000010                      0.6
 DeepSets-AE            arbitrary_same_team     knn4     0.25         10             0.000003      -3.291310e-06            0.000008                      0.6
 DeepSets-AE            arbitrary_same_team     knn4     0.50         10            -0.000004      -1.356541e-05            0.000005                      0.4
 DeepSets-AE spectrum_exact_sign_randomized delaunay     0.25         10             0.000129       1.206244e-04            0.000138                      1.0
 DeepSets-AE spectrum_exact_sign_randomized delaunay     0.50         10             0.000247       2.270679e-04            0.000273                      1.0
 DeepSets-AE spectrum_exact_sign_randomized     knn4     0.25         10             0.000132       1.174496e-04            0.000145                      1.0
 DeepSets-AE spectrum_exact_sign_randomized     knn4     0.50         10             0.000258       2.123725e-04            0.000314                      1.0
 DeepSets-AE     topology_frequency_matched delaunay     0.25         10             0.000004      -5.492508e-06            0.000013                      0.7
 DeepSets-AE     topology_frequency_matched delaunay     0.50         10             0.000001      -1.754899e-05            0.000019                      0.7
 DeepSets-AE     topology_frequency_matched     knn4     0.25         10            -0.000002      -8.074830e-06            0.000005                      0.5
 DeepSets-AE     topology_frequency_matched     knn4     0.50         10            -0.000010      -2.113359e-05            0.000002                      0.3
GAT-small-AE            arbitrary_same_team delaunay     0.25         10             0.000085       7.067302e-05            0.000098                      1.0
GAT-small-AE            arbitrary_same_team delaunay     0.50         10             0.000133       9.536038e-05            0.000172                      1.0
GAT-small-AE            arbitrary_same_team     knn4     0.25         10             0.000078       6.361015e-05            0.000092                      1.0
GAT-small-AE            arbitrary_same_team     knn4     0.50         10             0.000120       8.553534e-05            0.000155                      1.0
GAT-small-AE spectrum_exact_sign_randomized delaunay     0.25         10             0.000064       4.635019e-05            0.000081                      1.0
GAT-small-AE spectrum_exact_sign_randomized delaunay     0.50         10             0.000114       8.513211e-05            0.000146                      1.0
GAT-small-AE spectrum_exact_sign_randomized     knn4     0.25         10             0.000041       2.172158e-05            0.000061                      0.9
GAT-small-AE spectrum_exact_sign_randomized     knn4     0.50         10             0.000084       4.628172e-05            0.000129                      0.9
GAT-small-AE     topology_frequency_matched delaunay     0.25         10             0.000049       2.689225e-05            0.000070                      0.8
GAT-small-AE     topology_frequency_matched delaunay     0.50         10             0.000083       4.203084e-05            0.000123                      0.8
GAT-small-AE     topology_frequency_matched     knn4     0.25         10             0.000019      -8.663339e-06            0.000047                      0.5
GAT-small-AE     topology_frequency_matched     knn4     0.50         10             0.000020      -3.503907e-05            0.000077                      0.6
   Phase-GAT            arbitrary_same_team delaunay     0.25         10             0.000076       1.895042e-05            0.000133                      0.8
   Phase-GAT            arbitrary_same_team delaunay     0.50         10             0.000149       9.264177e-05            0.000205                      1.0
   Phase-GAT            arbitrary_same_team     knn4     0.25         10             0.000080       4.812595e-05            0.000114                      1.0
   Phase-GAT            arbitrary_same_team     knn4     0.50         10             0.000142       1.019017e-04            0.000189                      1.0
   Phase-GAT spectrum_exact_sign_randomized delaunay     0.25         10             0.000135       7.540476e-05            0.000200                      1.0
   Phase-GAT spectrum_exact_sign_randomized delaunay     0.50         10             0.000237       1.567638e-04            0.000317                      1.0
   Phase-GAT spectrum_exact_sign_randomized     knn4     0.25         10             0.000115       4.721968e-05            0.000202                      0.9
   Phase-GAT spectrum_exact_sign_randomized     knn4     0.50         10             0.000201       8.326009e-05            0.000342                      0.8
   Phase-GAT     topology_frequency_matched delaunay     0.25         10             0.000050      -3.466525e-05            0.000138                      0.7
   Phase-GAT     topology_frequency_matched delaunay     0.50         10             0.000098      -1.827029e-05            0.000208                      0.7
   Phase-GAT     topology_frequency_matched     knn4     0.25         10             0.000048       3.228578e-06            0.000117                      0.7
   Phase-GAT     topology_frequency_matched     knn4     0.50         10             0.000093       2.621600e-05            0.000186                      0.8
```

Controls, energies, probe graphs, and architectures remain separate. Bootstrap resampled the 10 matches; model seeds were averaged within architecture and were not treated as independent matches. Heldout split summaries are descriptive because it contains two matches.

## Interpretation boundary

Choose one evidence description after inspecting the complete tables and figures: `organization_signal_survives`, `frequency_bias_only`, `mixed_or_graph_specific`, or `matching_inadequate`. No numerical threshold is applied automatically.

## Figures

- `figures/scg_primary_energy_0.25.png`
- `figures/scg_primary_energy_0.5.png`
- `figures/match_effect_forest.png`
- `figures/matching_coverage.png`
