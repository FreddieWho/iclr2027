# Repair or Relocation? Compositional Generalization Beyond Endpoint Accuracy

**Supplementary Material — Anonymous submission to ICLR 2027**

Figure and table numbers without a supplementary prefix refer to the main paper.

# Supplementary Methods and Results

## Task, oracle, and eligible quartets

The coordinate task is a two-dimensional segment-crossing classification problem. A scene contains two line segments, represented by four planar endpoints and therefore eight scalar coordinates. The deterministic geometric oracle assigns a binary label according to whether the two segments cross under the task's intersection convention. A scene has an oracle label $y(x)$. An edit changes endpoint coordinates and has a known oracle outcome; it is not defined by the model's prediction or by a random perturbation direction.

For a base scene $x$, let $e_A$ and $e_B$ be two atomic edits. We write $x_A=x+e_A$, $x_B=x+e_B$, and $x_{AB}=x+e_A+e_B$. The evaluation population $E$ consists only of quartets for which each atomic edit preserves the base oracle label but the composition changes it:

$$
E=\{(x,e_A,e_B): y(x_A)=y(x_B)=y(x),\quad y(x_{AB})\ne y(x)\}.
$$

Thus each quartet has four oracle-labeled states, $x,A,B,AB$, with a prescribed transition pattern: both single edits are individually label-preserving and their joint application is label-changing. This contract is evaluated from the oracle before model performance is summarized. It prevents an arbitrary two-edit perturbation from being counted as a compositional test when either atom already changes the answer or their combination does not.

We distinguish three measures. Conditional composition miss (CCM) is the model's error on $AB$, conditional on both atomic predictions being correct and on quartet membership in $E$:
$$
\mathrm{CCM}=P(\hat y_{AB}\ne y_{AB}\mid \hat y_A=y_A,\hat y_B=y_B,E).
$$
It is a miss rate, not a conditional success rate. The three-state joint consistency rate is
$$
J_3=P(\hat y_A=y_A,\hat y_B=y_B,\hat y_{AB}=y_{AB}\mid E).
$$
When the base prediction is also required to be correct, the corresponding four-state rate is
$$
J_4=P(\hat y_x=y_x,\hat y_A=y_A,\hat y_B=y_B,\hat y_{AB}=y_{AB}\mid E).
$$
The reported raw-coordinate quartet table does not archive a base-state correctness column or a $J_4$ value; $J_4$ is therefore left unreported there. It is not inferred from marginal accuracies. CCM and $J_3$ also have different denominators: CCM conditions on model-correct atoms, whereas $J_3$ uses all oracle-eligible quartets.

## Coordinate models, splits, and repair-flow protocol

Training and development use fixed, separately identified scene pools. Coordinate models receive the eight raw endpoint coordinates and use a multilayer perceptron with hidden widths 64, 64, and 32 followed by a linear prediction head. The three coordinate training seeds are 11, 23, and 47; these are training replicates, not independent datasets. The clean condition is trained on the clean task examples. The flip condition adds oracle-mined boundary-crossing variations to expose the model to the update pattern of interest. The intervention is a diagnostic training condition, not an evaluation-time rule or a claim that every mined example is a member of the quartet test population.

Mechanism evaluation uses a frozen development bank of 237 eligible quartets from 106 parents. Every evaluated model is scored on the same edits. A separate confirmation bank contains 509 quartets from parents disjoint from development; the banks are not pooled. Settings for the relevant analyses are determined on development before confirmation. The repair-flow table below is a development-bank comparison and should not be described as a fresh-holdout estimate. The one-shot fresh holdout is a different evaluation with its own conditional denominator.

For each seed, define the hard subset $H$ from the clean model's predictions: A and B are both correct and AB is wrong. This is the baseline-110 state pattern. It is fixed before comparing the flip-trained model. Let endpoint repair be the fraction of $H$ for which the flip model predicts AB correctly. Let full repair be the fraction for which the flip model is correct on all three edited states A, B, and AB. Let migration be the fraction for which the flip model corrects AB but leaves at least one of A or B wrong. For binary correctness indicators on a quartet in $H$, the flip model's AB endpoint is either wrong, correct with both atoms correct, or correct with at least one atom wrong. These cases are mutually exclusive and exhaustive. Therefore, as an identity of rates on the same denominator,
$$
R_{\mathrm{endpoint}}=R_{\mathrm{full}}+M.
$$
This accounting identity does not assert why a prediction changed. In particular, it does not turn endpoint repair into full compositional repair.

| Seed | Hard baseline $H$ | Full repair | Migration | Endpoint repair | Migration / endpoint repair |
|---:|---:|---:|---:|---:|---:|
| 11 | 110 | 0.0636 | 0.3636 | 0.4273 | 0.8511 |
| 23 | 120 | 0.0250 | 0.4000 | 0.4250 | 0.9412 |
| 47 | 110 | 0.1000 | 0.3818 | 0.4818 | 0.7925 |

Values are rounded archived rates; the unrounded categories satisfy the identity exactly. $H$ differs by seed because it is defined by each clean model's correctness pattern. Accordingly, the table estimates conditional repair flow among that seed's clean-baseline hard cases, not the fraction of all test quartets repaired. Joint consistency $J_3$ changes only from approximately 0.05 under clean training to 0.06–0.08 under flip training in this raw-coordinate comparison. The visual analysis below is a separate task and does not supply a second estimate of this flow decomposition.

Uncertainty intervals for the raw-coordinate flow analysis use parent-cluster resampling: a bootstrap draw samples parent identities and retains the quartets belonging to each selected parent. The point estimates remain pooled over quartet rows. A parent contributing more eligible quartets can therefore contribute more rows to the point estimate; the bootstrap accounts for dependence within parents but does not change the estimand into an equally weighted mean of parent-specific rates. Seed-level denominators and intervals should be kept visible when describing this result.

## Geometry-by-supervision comparison

A four-cell comparison varies two factors on the development bank: input representation (raw coordinates or label-free relational geometry) and training exposure (clean or flip). Raw input is eight coordinates. The relational input is ten values: six pairwise distances among the four endpoints and four radii from the endpoints to their centroid. The crossing label itself is excluded. Both input conditions use the same 64–64–32 multilayer perceptron trunk and linear head. The four arms use the same quartet bank and three archived seeds. Flow columns use the same-seed raw-clean model to define the baseline-110 and baseline-111 subsets; they are conditional measures, not single-arm marginal metrics.

| Representation | Supervision | Mean $J_3$ | Full repair range | Relocation range |
|:--|:--|--:|--:|--:|
| Coordinates | Clean | 0.053 | — | — |
| Coordinates | Change-augmented | 0.068 | 0.025–0.100 | 0.364–0.400 |
| Geometry | Clean | 0.170 | 0.136–0.233 | 0.167–0.255 |
| Geometry | Change-augmented | 0.401 | 0.364–0.442 | 0.317–0.418 |

Mean joint scores are descriptive averages across three training seeds on one bank; ranges are the minimum and maximum seed estimates, not confidence intervals.

The table summarizes a bounded input-by-training comparison, not a universal synergy law. The relational-plus-flip arm has the highest $J_3$ among these four arms, but $J_3<0.5$: a majority of eligible quartets remain inconsistent. Its features are hand-designed and relation-aware by construction. The measurements establish an existence result for this input under this recipe, not that a general learned relation architecture is necessary. An analytic segment-intersection parser solves the official source-task E pairs in a separate strong-baseline check, which further bounds claims about the need for a learned model.

The per-quartet archive used for this fallback table does not contain $J_4$, a base-state column, parent-bootstrap intervals for every field, model parameter counts, or multiply-accumulate counts. These entries are missing, not estimated from another bank. Ranges for repair and migration span the three seeds; they should not be read as a pooled numerator over a common denominator.

## Rendered-image screen

A separate visual screen renders the same kind of four-state crossing geometry as native 64-by-64 RGB images. The data were generated into parent-disjoint train, development, and test splits: training has 128 parents and 323 singleton images; development has 32 parents and 81 singleton images; testing has 128 parents and 319 singleton images. The test quartets comprise 28 quartets, 112 state images, and 19 parents. There are no training or development quartets and no AB training labels. The test quartet parent set is separate from the image-training and model-selection parents.

The two compared arms use the same ImageNet-initialized ResNet18 backbone, RGB observations resized internally to 224 by 224 pixels, and a linear head on the global pooled image representation. Inference uses images only; coordinates, parent identities, quartet identities, and oracle labels are not model inputs. Images follow the same scaling and ImageNet normalization in training and evaluation.

Both conditions train for 20 epochs with batch size 32. The full-singleton arm uses 323 training images per epoch. The clean-only arm repeats its 128 unique clean training rows to match 323 presentations. Checkpoint selection uses binary cross-entropy on the same 81 singleton development images for both arms. Seeds are 803, 805, and 806. Matching presentation count does not match the number of unique training states; the intervention changes the supervised singleton distribution.

| Seed | Baseline failures | Full repairs | Relocations | Endpoint repairs |
|--:|--:|--:|--:|--:|
| 803 | 16 | 10 | 1 | 11 |
| 805 | 12 | 9 | 1 | 10 |
| 806 | 12 | 5 | 3 | 8 |

These counts use the same-seed clean model to define the baseline failure set. The full-singleton model's conditional composition miss rates are 0.273, 0.150, and 0.250, compared with 0.800, 0.632, and 0.667 for clean-only training. Figure 3 presents the paired joint-accuracy differences with 95% parent-cluster percentile intervals from 10,000 draws. No normal approximation over three training seeds is used.

## Per-seed uncertainty for coordinate repair

The following 95% parent-cluster intervals accompany Figure 2. They are copied from the stored paired analysis and retain the seed-specific baseline failure denominator.

| Seed | Full repair | Relocation | Endpoint repair |
|--:|:--|:--|:--|
| 11 | 0.0636 [0.0189, 0.1262] | 0.3636 [0.2365, 0.5096] | 0.4273 [0.2991, 0.5785] |
| 23 | 0.0250 [0.0000, 0.0574] | 0.4000 [0.2645, 0.5385] | 0.4250 [0.2913, 0.5684] |
| 47 | 0.1000 [0.0348, 0.1837] | 0.3818 [0.2362, 0.5313] | 0.4818 [0.3333, 0.6372] |

## Reporting and reproducibility

For each comparison, retain an oracle-defined quartet identifier, its parent identifier, the four labels, and both models' predictions. Construct the baseline failure set once, compute all transition categories on this fixed set, and report its quartet and parent counts. Report $J_3$, constituent coverage, and endpoint accuracy on the full eligible bank alongside the fixed-baseline repair fractions. If the baseline failure set is empty, its repair fractions are undefined rather than zero. If no constituents are jointly correct, CCM is undefined. If no endpoint is repaired, the relocation share among repairs is undefined.

The core numerical sources are saved quartet-level prediction tables and their parent-cluster summaries. The fresh-holdout composition result is retained as its original aggregate; it is not assigned an invented parent count or uncertainty interval. Experimental banks and training replicates are kept separate. The supplementary description specifies the metric definitions, models, evaluation populations, and the matched visual comparison. Full reproduction additionally requires the original scene arrays and checkpoints; those large assets are not embedded in this PDF. No claim of complete public artifact availability is made.
