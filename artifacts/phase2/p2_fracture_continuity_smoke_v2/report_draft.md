# P2 fracture continuity report

- stage: `P2_NOVELTY_DISCRIMINATOR`
- task: `fracture continuity`
- operator: `p2_fracture_endpoint_reallocation_v1`
- status: `WIRING_ONLY_NOT_SCIENTIFIC`
- this report is independent of P1 and P2 rigid response tables.

## Scope

- matching sets planned/complete: 144 / 124
- valid arms: 266 / 288
- response/statistics summary rows: 12
- anchor and control share the exact non-zero displacement-vector multiset; control support is same-team, same-size and disjoint.

## Interpretation boundary

The gap is a representation response contrast, not downstream task performance. Frequency/topology residuals, graph dependence, role dependence and match-level uncertainty must be read together. No fixed numerical threshold is used to force a scientific label.

## Summary table

| architecture   | graph_id   | role_group   |   epsilon |   n_matches |   raw_scg_mean |   raw_scg_median |   raw_scg_ci_low |   raw_scg_ci_high |   normalized_scg_mean |   normalized_scg_ci_low |   normalized_scg_ci_high |   positive_match_fraction |   frequency_residual_mean |   topology_residual_mean |
|:---------------|:-----------|:-------------|----------:|------------:|---------------:|-----------------:|-----------------:|------------------:|----------------------:|------------------------:|-------------------------:|--------------------------:|--------------------------:|-------------------------:|
| DeepSets-AE    | delaunay   | defender     |      0.25 |           1 |   -4.18226e-06 |     -4.18226e-06 |     -4.18226e-06 |      -4.18226e-06 |          -1.6729e-05  |            -1.6729e-05  |             -1.6729e-05  |                         0 |                         0 |                        0 |
| DeepSets-AE    | delaunay   | defender     |      0.5  |           1 |   -5.02666e-06 |     -5.02666e-06 |     -5.02666e-06 |      -5.02666e-06 |          -1.00533e-05 |            -1.00533e-05 |             -1.00533e-05 |                         0 |                         0 |                        0 |
| DeepSets-AE    | delaunay   | forward      |      0.25 |           1 |   -4.07298e-06 |     -4.07298e-06 |     -4.07298e-06 |      -4.07298e-06 |          -1.62919e-05 |            -1.62919e-05 |             -1.62919e-05 |                         0 |                         0 |                        0 |
| DeepSets-AE    | delaunay   | forward      |      0.5  |           1 |   -2.3966e-05  |     -2.3966e-05  |     -2.3966e-05  |      -2.3966e-05  |          -4.79321e-05 |            -4.79321e-05 |             -4.79321e-05 |                         0 |                         0 |                        0 |
| DeepSets-AE    | delaunay   | midfielder   |      0.25 |           1 |    2.22524e-05 |      2.22524e-05 |      2.22524e-05 |       2.22524e-05 |           8.90096e-05 |             8.90096e-05 |              8.90096e-05 |                         1 |                         0 |                        0 |
| DeepSets-AE    | delaunay   | midfielder   |      0.5  |           1 |    7.28418e-05 |      7.28418e-05 |      7.28418e-05 |       7.28418e-05 |           0.000145684 |             0.000145684 |              0.000145684 |                         1 |                         0 |                        0 |
| DeepSets-AE    | knn4       | defender     |      0.25 |           1 |    8.42909e-06 |      8.42909e-06 |      8.42909e-06 |       8.42909e-06 |           3.37164e-05 |             3.37164e-05 |              3.37164e-05 |                         1 |                         0 |                        0 |
| DeepSets-AE    | knn4       | defender     |      0.5  |           1 |    2.89331e-05 |      2.89331e-05 |      2.89331e-05 |       2.89331e-05 |           5.78662e-05 |             5.78662e-05 |              5.78662e-05 |                         1 |                         0 |                        0 |
| DeepSets-AE    | knn4       | forward      |      0.25 |           1 |   -5.36939e-06 |     -5.36939e-06 |     -5.36939e-06 |      -5.36939e-06 |          -2.14775e-05 |            -2.14775e-05 |             -2.14775e-05 |                         0 |                         0 |                        0 |
| DeepSets-AE    | knn4       | forward      |      0.5  |           1 |   -1.78764e-05 |     -1.78764e-05 |     -1.78764e-05 |      -1.78764e-05 |          -3.57529e-05 |            -3.57529e-05 |             -3.57529e-05 |                         0 |                         0 |                        0 |
| DeepSets-AE    | knn4       | midfielder   |      0.25 |           1 |    1.43896e-05 |      1.43896e-05 |      1.43896e-05 |       1.43896e-05 |           5.75582e-05 |             5.75582e-05 |              5.75582e-05 |                         1 |                         0 |                        0 |
| DeepSets-AE    | knn4       | midfielder   |      0.5  |           1 |    6.94096e-05 |      6.94096e-05 |      6.94096e-05 |       6.94096e-05 |           0.000138819 |             0.000138819 |              0.000138819 |                         1 |                         0 |                        0 |
