# Introduction

Suppose a classifier correctly handles two simple changes to a scene but fails when they occur together. We train a new model, and its prediction on the composed scene becomes correct. Has compositional generalization improved? The answer depends on what happened to the two simple cases. If either is now wrong, the original error has moved within the same semantic neighborhood. An endpoint score records a success; a compositional evaluation should distinguish that success from preserving and extending the model's existing competence.

This distinction connects two familiar concerns that are usually measured separately. Compositional-generalization benchmarks ask whether learned components can be recombined [@lake2018generalization; @keysers2020measuring], and conditional compositionality gaps isolate failures despite success on the component problems [@press2023measuring]. Model-update research asks whether a new predictor introduces errors on previously correct examples [@yan2021positive]. We bring these concerns together **within the same controlled instance**. The constituents are not an unrelated regression set: their correctness is precisely what makes a failed composition informative.

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

The eligibility condition depends on the oracle, not the model. Individual edits leave the correct decision unchanged; their composition requires a different decision. A model that repeats the parent label throughout this neighborhood therefore appears invariant on both constituents but fails at the composed endpoint. This is a structured setting for the excessive-invariance problem studied by @jacobsen2019excessive.

In the coordinate experiments, $x$ contains four planar points, grouped into two segments, and the target is their intersection relation. An exact geometric predicate supplies labels for all states. The neural predictor receives either eight coordinates or ten geometric quantities: six indexed pairwise distances and four distances to the centroid. The latter are computed without labels and exclude the target intersection indicator. Both input choices feed the same hidden-layer design. They are alternative representations of the same labeled problem, not additional annotations.

Every prediction is computed independently from its input. Composition refers to the relation among inputs and labels; it does not imply that a model has recurrent state, an explicit world model, or an internal edit operator. Similarly, the synthetic task measures controlled geometric composition, not compositional reasoning in unrestricted language or natural images.

![**An endpoint correction has two distinct outcomes.** The diagram follows correctness on A, B, and AB. A baseline pattern of 110 becomes a full repair only at 111. Correcting AB while breaking A or B relocates the error. The diagram is a schematic, not an empirical example; the bits indicate correctness, not predicted classes.](figures/fig1_protocol.png){width=100%}

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

![**Most raw-coordinate endpoint repairs relocate an error.** Bars partition the baseline failure set into full repair and error relocation after change supervision. Total height is endpoint repair; annotations report the relocated fraction among endpoint repairs. The three seeds share the 237-quartet bank and have baseline-failure denominators 110, 120, and 110. These are descriptive point estimates; per-seed parent-cluster intervals are given in the supplementary material.](figures/fig2_relocation.png){width=100%}

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

A useful diagnostic must distinguish improvement, not merely expose failure. We therefore apply the same accounting to a rendered geometric task. An ImageNet-initialized ResNet18 [@he2016deep; @deng2009imagenet] receives RGB scenes. The evaluation bank contains 28 quartets from 19 parents. Training uses singleton states only; composed AB states are absent from the training set.

We compare a clean-only model with a model trained on the full singleton set. To separate extra supervision from the number of training presentations, the clean-only arm repeats 128 unique clean images to match the full arm's **323 presentations per epoch**. Both run for 20 epochs, select a checkpoint by loss on the same 81 singleton development images, and use the same seed-specific backbone initialization. The comparison matches exposure and selection, while intentionally changing the diversity and labels of supervised singleton states. It does not isolate data diversity from label coverage.

Figure 3 shows consistent gains. Clean-only joint correctness is 0.143, 0.250, and 0.214; full-singleton supervision reaches 0.571, 0.607, and 0.536. Paired differences are 0.429, 0.357, and 0.321, with 95% parent-cluster intervals [0.222, 0.633], [0.172, 0.556], and [0.138, 0.517]. On the baseline 110 subsets, full repairs and relocations number **10 versus 1**, **9 versus 1**, and **5 versus 3**. The method is therefore credited with genuine constituent-preserving gains, while its remaining regressions stay visible.

![**Matched exposure reveals a genuine visual gain.** Left: joint correctness for clean-only and full-singleton supervision, with seeds shown individually. Right: paired differences and 95% parent-cluster percentile intervals, using 10,000 draws. Both arms receive 323 presentations per epoch for 20 epochs and select on the same 81 singleton development images. All estimates use one 28-quartet, 19-parent bank.](figures/fig3_visual.png){width=100%}

This is a small controlled visual experiment, not a claim about natural-image compositionality. It establishes a narrower point relevant to the proposed protocol: requiring constituent preservation does not mechanically dismiss augmentation. The coordinate and visual interventions have different repair profiles, which a common accounting makes legible.

# Related work and scope

**Compositional generalization.** Controlled recombination has long served as a test of systematic generalization [@lake2018generalization; @keysers2020measuring]. Diagnostic visual reasoning datasets similarly use compositional structure to separate capabilities [@johnson2017clevr]. @press2023measuring explicitly study failures on a composed question despite correct answers to its subproblems. We build on that conditional logic. Our additional object is the *paired change* in constituent and composed correctness when a model is repaired, under oracle-defined edits to one shared scene.

**Learning from semantic changes.** Counterfactually augmented data can expose distinctions that standard training underuses [@kaushik2020learning]. Excessive-invariance analysis emphasizes the complementary failure of ignoring task-relevant changes [@jacobsen2019excessive]. Our experiments concern these semantic changes rather than assuming that all small perturbations should preserve a prediction. The evidence does not establish that change supervision generally fails: its outcome depends on the representation and task.

**Preserving existing competence.** Positive-congruent training measures and discourages negative flips across model versions [@yan2021positive]. Error relocation is related, but its preservation set is coupled to each successful composed repair. The question is whether the repaired composition retains its own prerequisites. We claim neither to originate regression measurement nor to provide a new preservation objective.

**Scope.** Exact geometric labels make eligibility and outcome accounting unusually clear. They also limit external validity. Hand-designed geometric inputs encode useful task structure; a deterministic intersection parser can solve the coordinate problem. The representation comparison is consequently about the behavior of learned predictors under a controlled input change. The 237-quartet bank supports diagnostic comparisons, the fresh holdout supports a separately scoped failure observation, and the visual bank supports a small paired extension. None establishes a population-wide prevalence of relocation in large models. The accounting protocol applies more broadly wherever constituents and their composition can be labeled reliably; that broader empirical scope remains to be established.

# Conclusion

A corrected composed prediction is incomplete evidence of compositional repair. In our coordinate experiments, most endpoint repairs break a constituent prediction that was previously correct. Fixing the baseline failure set and tracking the complete correctness pattern separates these relocations from full repairs, identifies substantially stronger gains from geometric inputs with change supervision, and recognizes a successful exposure-matched visual intervention. The practical recommendation is simple: evaluate a proposed compositional repair on the constituents that made the original failure compositional, and report which of those successes survive.
