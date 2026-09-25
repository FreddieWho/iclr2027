# Repair or Relocation? Measuring Compositional Generalization Beyond Endpoint Accuracy

**Anonymous submission to ICLR 2027**

## Abstract

Improving a model on a composed input does not establish that it has learned to compose: the same change can break predictions for the constituent inputs. We study this ambiguity through controlled geometric interventions with exact labels. Each evaluation quartet contains a scene, two individually label-preserving edits, and their label-changing composition. On a fresh holdout, a coordinate classifier misses 151 of 176 eligible compositions despite correctly classifying both constituent edits. More surprisingly, training on semantic changes repairs many composed endpoints while relocating the error: across three seeds, 79–94% of those endpoint repairs break at least one previously correct constituent prediction. We introduce a paired evaluation protocol that fixes the baseline failure set and partitions endpoint repair into full repair and error relocation. This accounting changes which interventions appear effective. On a 237-quartet bank, combining geometric inputs with change supervision raises joint correctness from 0.051–0.055 to 0.384–0.422; a separate 509-quartet bank shows the same ordering. An exposure-matched visual experiment demonstrates that the protocol also recognizes genuine gains from additional singleton supervision. The contribution is an evaluation principle: compositional repair requires preserving the constituent successes that made the original failure compositional.

**Keywords:** compositional generalization; model evaluation; model repair; semantic interventions; relational representations

# Introduction

Suppose a classifier correctly handles two simple changes to a scene but fails when they occur together. We train a new model, and its prediction on the composed scene becomes correct. Has compositional generalization improved? The answer depends on what happened to the two simple cases. If either is now wrong, the original error has moved within the same semantic neighborhood. An endpoint score records a success; a compositional evaluation should distinguish that success from preserving and extending the model's existing competence.

This distinction connects two familiar concerns that are usually measured separately. Compositional-generalization benchmarks ask whether learned components can be recombined (Lake and Baroni, 2018; Keysers et al., 2020), and conditional compositionality gaps isolate failures despite success on the component problems (Press et al., 2023). Model-update research asks whether a new predictor introduces errors on previously correct examples (Yan et al., 2021). We bring these concerns together **within the same controlled instance**. The constituents are not an unrelated regression set: their correctness is precisely what makes a failed composition informative.

We use a geometric task in which semantic changes can be specified and labeled exactly. Two edits each preserve the relation between a pair of line segments, yet their combination changes it. All four scenes can be presented independently to the same feed-forward classifier. The setting gives us an exact oracle, a shared parent scene, and a way to compare predictors on identical constituent and composed inputs. It removes uncertainty about whether the intended label changed, while retaining a nontrivial learning problem for the tested neural models. Its simplicity is useful: the question is how to measure learned repair, not whether a neural network is necessary to compute segment intersection.

The resulting picture differs from an endpoint-only account. A fresh evaluation yields an 85.8% composition miss rate conditional on correct constituent predictions. More importantly, in a paired comparison, change supervision fixes 42.5–48.2% of the baseline's failed composed endpoints, but only 2.5–10.0% of those baseline failures become jointly correct. Most apparent repairs introduce a constituent error. In contrast, geometric inputs combined with the same supervision produce substantially more full repairs. A visual experiment then demonstrates that this diagnostic can credit a successful intervention: with training exposures and selection data matched, additional singleton supervision improves joint accuracy in all three seeds.

Our contributions are:

1. **A paired protocol for compositional repair.** We construct oracle-defined quartets, fix the baseline failure population, and separate full repair from error relocation through an exact accounting identity.
2. **Evidence that endpoint repair can be dominated by relocation.** On the coordinate task, 79–94% of endpoint repairs fail to preserve both constituent successes. A factorial comparison distinguishes this outcome from the larger joint gains of geometric inputs with change supervision.
3. **A practical reporting criterion.** Joint accuracy, constituent coverage, conditional composition error, and paired repair flows provide complementary information. A matched visual comparison illustrates the criterion beyond direct coordinate inputs.

The paper contributes a diagnostic and an empirical finding, rather than a general-purpose composition architecture. Conditional evaluation itself has substantial prior work; our focus is what changes when a purported repair is evaluated on the *same* constituents that justified the original diagnosis.

# A controlled test of composition

## Scenes, edits, and oracle eligibility

Let $x$ be a parent scene with binary oracle label $y(x)$. Two edits $e_A$ and $e_B$ define $x_A=x+e_A$, $x_B=x+e_B$, and $x_{AB}=x+e_A+e_B$. We evaluate the population

$$
\mathcal E=\{(x,e_A,e_B):y(x_A)=y(x_B)=y(x),\quad y(x_{AB})\ne y(x)\}.
$$

The eligibility condition depends on the oracle, not the model. Individual edits leave the correct decision unchanged; their composition requires a different decision. A model that repeats the parent label throughout this neighborhood therefore appears invariant on both constituents but fails at the composed endpoint. This is a structured setting for the excessive-invariance problem studied by Jacobsen et al. (2019).

In the coordinate experiments, $x$ contains four planar points, grouped into two segments, and the target is their intersection relation. An exact geometric predicate supplies labels for all states. The neural predictor receives either eight coordinates or ten geometric quantities: six indexed pairwise distances and four distances to the centroid. The latter are computed without labels and exclude the target intersection indicator. Both input choices feed the same hidden-layer design. They are alternative representations of the same labeled problem, not additional annotations.

Every prediction is computed independently from its input. Composition refers to the relation among inputs and labels; it does not imply that a model has recurrent state, an explicit world model, or an internal edit operator. Similarly, the synthetic task measures controlled geometric composition, not compositional reasoning in unrestricted language or natural images.

![**An endpoint correction has two distinct outcomes.** The diagram follows correctness on A, B, and AB. A baseline pattern of 110 becomes a full repair only at 111. Correcting AB while breaking A or B relocates the error. The diagram is a schematic, not an empirical example; the bits indicate correctness, not predicted classes.](figures/fig1_protocol.png)

## Complementary scores on a fixed quartet bank

For predictor $f$, define $c_A(f)$, $c_B(f)$, and $c_{AB}(f)$ as binary correctness indicators. Three summaries answer different questions:

$$
\begin{aligned}
A(f)&=\Pr(c_Ac_B=1\mid\mathcal E),\\
J_3(f)&=\Pr(c_Ac_Bc_{AB}=1\mid\mathcal E),\\
\operatorname{CCM}(f)&=\Pr(c_{AB}=0\mid\mathcal E,c_Ac_B=1).
\end{aligned}
$$

Here $A$ is constituent coverage, $J_3$ is joint correctness, and CCM is the conditional composition **miss** rate. When $A>0$, $J_3=A(1-\operatorname{CCM})$. Thus a conditional miss rate describes behavior on the cases where the model has constituent competence, while joint correctness also accounts for how often that competence is present. We additionally distinguish $J_4$, which requires the parent prediction to be correct. Our reported $J_3$ values do not silently impose this extra condition.

The conditional population for CCM can change after training. A lower CCM can reflect a better composed prediction, a change in which constituents remain correct, or both. Conversely, a model can increase $J_3$ by becoming competent on previously excluded constituents. Neither situation is resolved by comparing conditional rates alone. We therefore pair model comparisons at the quartet level and retain the relevant denominators.

# Repair or relocation?

## Fixing the population before comparing models

Let $f_0$ be the baseline and $f_1$ a candidate repair. Fix

$$
\mathcal H=\{q\in\mathcal E:c_A(f_0)=c_B(f_0)=1,\ c_{AB}(f_0)=0\}.
$$

This set contains genuine composition failures of the baseline. It is selected once and is not redefined using the candidate model. On $\mathcal H$, report

$$
\begin{aligned}
R_{\mathrm{end}}&=\Pr(c_{AB}(f_1)=1\mid\mathcal H),\\
R_{\mathrm{full}}&=\Pr(c_A(f_1)c_B(f_1)c_{AB}(f_1)=1\mid\mathcal H),\\
M&=\Pr(c_{AB}(f_1)=1,\ c_A(f_1)c_B(f_1)=0\mid\mathcal H).
\end{aligned}
$$

The event that AB is corrected partitions according to whether both constituents stay correct:

$$
\boxed{R_{\mathrm{end}}=R_{\mathrm{full}}+M.}
$$

This is an accounting identity, not a causal model. It holds for any pair of predictors evaluated on the fixed set. Its value is operational: a large endpoint repair rate places no positive lower bound on full repair without information about constituent preservation. When $R_{\mathrm{end}}>0$, $M/R_{\mathrm{end}}$ is the fraction of endpoint repairs that relocate the error. The supplementary material gives the complete correctness partition.

We use “relocation” to describe observed predictions. It does not assert that an internal error representation physically moves, or that one gradient update causally trades one decision for another. Nonetheless, it distinguishes two materially different outcomes of a training intervention. A method that reaches 111 satisfies all three judgments; one that reaches 011 or 101 satisfies the endpoint by giving up a constituent success.

## Experimental design

The coordinate comparison crosses two input representations, raw coordinates and geometric features, with two supervision regimes, clean scenes and clean scenes augmented with oracle-changing single edits. The augmented regimes use labeled semantic changes; no claim of reduced oracle-query cost is made. The central comparison uses the same 237-quartet diagnostic bank for all four arms and three training seeds. This bank contains 106 parent scenes and was used for diagnostic development. The baseline-failure subsets contain 110, 120, and 110 quartets from 58, 65, and 56 parents, respectively.

We distinguish this diagnostic bank from two other evaluations. First, a separately generated, once-evaluated holdout supplies the conditional composition finding. Second, a parent-disjoint bank of 509 quartets provides an additional comparison of the geometric-input models. These banks have different roles and are never pooled. Seeds are repeated fits on the same data, not independent datasets.

Reported point estimates weight quartets equally. Where intervals are given, resampling is by parent, retaining the associated quartets; this preserves within-parent dependence without changing the point estimand to an equally weighted mean over parents. Per-seed results remain separate. The visual comparison uses the same principle, with 10,000 parent-cluster bootstrap draws. Detailed settings and denominator conventions are in the supplementary material.

# Endpoint success can conceal constituent failure

## Composition fails despite correct constituents

On the fresh holdout, the clean coordinate classifier misses **151 of 176 eligible compositions (85.8%)**, conditional on both constituent predictions being correct. This is a failure rate within a precisely defined population, not an error rate over arbitrary edits or scenes. The 176 cases must not be interpreted as 176 independent parents; the preserved aggregate does not support a parent-cluster interval for this headline. Its role is to establish that the composition failure extends beyond the diagnostic bank.

The more consequential comparison is paired repair. On the diagnostic bank, change supervision raises raw-coordinate AB accuracy from 0.477–0.519 to 0.574–0.679. Yet constituent coverage falls from 0.519–0.557 to 0.367–0.456. Joint correctness barely changes: it remains 0.059–0.076, compared with 0.051–0.055 for clean training. These scores already indicate that an improvement at AB is not a commensurate improvement across the quartet.

The fixed-baseline decomposition makes the discrepancy explicit (Figure 2). Endpoint repair is 0.427, 0.425, and 0.482, whereas full repair is only 0.064, 0.025, and 0.100. Error relocation accounts for **85.1%, 94.1%, and 79.2%** of corrected endpoints. The paired joint-accuracy differences are 0.013, 0.008, and 0.021; their parent-cluster intervals all include zero. The endpoint improvement is real, but most corrected endpoints do not retain the constituent successes that defined the original problem.

![**Most raw-coordinate endpoint repairs relocate an error.** Bars partition the baseline failure set into full repair and error relocation after change supervision. Total height is endpoint repair; annotations report the relocated fraction among endpoint repairs. The three seeds share the 237-quartet bank and have baseline-failure denominators 110, 120, and 110. These are descriptive point estimates; per-seed parent-cluster intervals are given in the supplementary material.](figures/fig2_relocation.png)

## Representation and supervision can yield full repair

The same protocol identifies a substantially more successful intervention. Table 1 shows the full input-by-supervision comparison. Geometric inputs alone increase $J_3$ to 0.131–0.211. Combining them with change supervision increases it to **0.384–0.422**. For each seed, the combined arm improves over either component alone, and the corresponding paired parent-cluster intervals exclude zero.

| Input | Supervision | Seed 11 | Seed 23 | Seed 47 |
|:--|:--|--:|--:|--:|
| Coordinates | Clean | 0.055 | 0.051 | 0.055 |
| Coordinates | Change-augmented | 0.068 | 0.059 | 0.076 |
| Geometry | Clean | 0.131 | 0.169 | 0.211 |
| Geometry | Change-augmented | **0.384** | **0.422** | **0.397** |

**Table 1. Joint correctness $J_3$ on the same 237 quartets from 106 parents.** Hidden-layer design and task are shared across input choices. Training seeds are reported individually. “Geometry” denotes label-free indexed distances and centroid radii; it is not an oracle feature.

Crucially, this gain survives the repair criterion. On the raw baseline's fixed failure set, geometric inputs with change supervision fully repair 0.373, 0.442, and 0.364 of cases. Relocation remains present at 0.373, 0.317, and 0.418, so the intervention is not regression-free. It nevertheless converts many more baseline failures into jointly correct predictions than raw-input change supervision does. The distinction is visible only when constituent and endpoint outcomes are accounted for together.

The factorial contrast in joint correctness,

$$
I=J_{\mathrm{geom,aug}}-J_{\mathrm{geom,clean}}-J_{\mathrm{raw,aug}}+J_{\mathrm{raw,clean}},
$$

is 0.241, 0.245, and 0.165, with per-seed parent-cluster intervals [0.134, 0.346], [0.121, 0.377], and [0.066, 0.264]. This is evidence of complementarity on the source task and tested recipe. It does not identify a unique internal mechanism or imply that the interaction must be positive on other tasks.

On the separate 509-quartet bank, geometric inputs with change supervision attain $J_3=0.474,0.448,0.436$, compared with $0.059,0.047,0.041$ for the clean coordinate baseline. Full repair on its seed-specific baseline-failure sets is 0.495, 0.471, and 0.449. This second bank supports the direction of the representation-assisted repair result; it is not merged with the development estimate, nor used to turn the source-specific interaction into a universal claim.

# The protocol also recognizes successful visual repair

A useful diagnostic must distinguish improvement, not merely expose failure. We therefore apply the same accounting to a rendered geometric task. An ImageNet-initialized ResNet18 (He et al., 2016; Deng et al., 2009) receives RGB scenes. The evaluation bank contains 28 quartets from 19 parents. Training uses singleton states only; composed AB states are absent from the training set.

We compare a clean-only model with a model trained on the full singleton set. To separate extra supervision from the number of training presentations, the clean-only arm repeats 128 unique clean images to match the full arm's **323 presentations per epoch**. Both run for 20 epochs, select a checkpoint by loss on the same 81 singleton development images, and use the same seed-specific backbone initialization. The comparison matches exposure and selection, while intentionally changing the diversity and labels of supervised singleton states. It does not isolate data diversity from label coverage.

Figure 3 shows consistent gains. Clean-only joint correctness is 0.143, 0.250, and 0.214; full-singleton supervision reaches 0.571, 0.607, and 0.536. Paired differences are 0.429, 0.357, and 0.321, with 95% parent-cluster intervals [0.222, 0.633], [0.172, 0.556], and [0.138, 0.517]. On the baseline 110 subsets, full repairs and relocations number **10 versus 1**, **9 versus 1**, and **5 versus 3**. The method is therefore credited with genuine constituent-preserving gains, while its remaining regressions stay visible.

![**Matched exposure reveals a genuine visual gain.** Left: joint correctness for clean-only and full-singleton supervision, with seeds shown individually. Right: paired differences and 95% parent-cluster percentile intervals, using 10,000 draws. Both arms receive 323 presentations per epoch for 20 epochs and select on the same 81 singleton development images. All estimates use one 28-quartet, 19-parent bank.](figures/fig3_visual.png)

This is a small controlled visual experiment, not a claim about natural-image compositionality. It establishes a narrower point relevant to the proposed protocol: requiring constituent preservation does not mechanically dismiss augmentation. The coordinate and visual interventions have different repair profiles, which a common accounting makes legible.

# Related work and scope

**Compositional generalization.** Controlled recombination has long served as a test of systematic generalization (Lake and Baroni, 2018; Keysers et al., 2020). Diagnostic visual reasoning datasets similarly use compositional structure to separate capabilities (Johnson et al., 2017). Press et al. (2023) explicitly study failures on a composed question despite correct answers to its subproblems. We build on that conditional logic. Our additional object is the *paired change* in constituent and composed correctness when a model is repaired, under oracle-defined edits to one shared scene.

**Learning from semantic changes.** Counterfactually augmented data can expose distinctions that standard training underuses (Kaushik et al., 2020). Excessive-invariance analysis emphasizes the complementary failure of ignoring task-relevant changes (Jacobsen et al., 2019). Our experiments concern these semantic changes rather than assuming that all small perturbations should preserve a prediction. The evidence does not establish that change supervision generally fails: its outcome depends on the representation and task.

**Preserving existing competence.** Positive-congruent training measures and discourages negative flips across model versions (Yan et al., 2021). Error relocation is related, but its preservation set is coupled to each successful composed repair. The question is whether the repaired composition retains its own prerequisites. We claim neither to originate regression measurement nor to provide a new preservation objective.

**Scope.** Exact geometric labels make eligibility and outcome accounting unusually clear. They also limit external validity. Hand-designed geometric inputs encode useful task structure; a deterministic intersection parser can solve the coordinate problem. The representation comparison is consequently about the behavior of learned predictors under a controlled input change. The 237-quartet bank supports diagnostic comparisons, the fresh holdout supports a separately scoped failure observation, and the visual bank supports a small paired extension. None establishes a population-wide prevalence of relocation in large models. The accounting protocol applies more broadly wherever constituents and their composition can be labeled reliably; that broader empirical scope remains to be established.

# Conclusion

A corrected composed prediction is incomplete evidence of compositional repair. In our coordinate experiments, most endpoint repairs break a constituent prediction that was previously correct. Fixing the baseline failure set and tracking the complete correctness pattern separates these relocations from full repairs, identifies substantially stronger gains from geometric inputs with change supervision, and recognizes a successful exposure-matched visual intervention. The practical recommendation is simple: evaluate a proposed compositional repair on the constituents that made the original failure compositional, and report which of those successes survive.


# AI use statement 

Generative AI assisted study design, implementation, result analysis, figure preparation, and manuscript writing. The authors take responsibility for the final content.

# Ethics statement 

The experiments reported in this paper use synthetic geometric scenes and rendered images. They do not involve human participants or sensitive personal data. The visual model uses publicly available ImageNet-pretrained weights. The controlled task is intended for evaluation of learned behavior; its results should not be treated as a safety certification for deployed models.

# References

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In \emphProceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.\ 248--255, 2009. \doi10.1109/CVPR.2009.5206848.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In \emphProceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.\ 770--778, 2016. \doi10.1109/CVPR.2016.90.

Jörn-Henrik Jacobsen, Jens Behrmann, Richard S. Zemel, and Matthias Bethge. Excessive invariance causes adversarial vulnerability. In \emphInternational Conference on Learning Representations (ICLR), 2019.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In \emphProceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Divyansh Kaushik, Eduard Hovy, and Zachary C. Lipton. Learning the difference that makes a difference with counterfactually-augmented data. In \emphInternational Conference on Learning Representations (ICLR), 2020.

Daniel Keysers, Nathanael Sch\"arli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, and Olivier Bousquet. Measuring compositional generalization: A comprehensive method on realistic data. In \emphInternational Conference on Learning Representations (ICLR), 2020.

Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In \emphProceedings of the 35th International Conference on Machine Learning (ICML 2018), volume 80 of \emphPMLR, pp.\ 2873--2882, 2018.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In \emphFindings of the Association for Computational Linguistics: EMNLP 2023, pp.\ 5687--5711, Singapore, 2023. Association for Computational Linguistics. \doi10.18653/v1/2023.findings-emnlp.378.

Sijie Yan, Yuanjun Xiong, Kaustav Kundu, Shuo Yang, Siqi Deng, Meng Wang, Wei Xia, and Stefano Soatto. Positive-congruent training: Towards regression-free model updates. In \emphProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.\ 14299--14308, 2021.

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
