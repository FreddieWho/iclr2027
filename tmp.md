# Title

Repair or Relocation? Compositional Generalization Beyond Endpoint Accuracy

# Abstract

A model can correct a composed prediction while introducing an error on a previously correct constituent. Endpoint accuracy counts this as a repair, obscuring whether compositional competence has improved. We introduce a paired evaluation protocol that fixes the baseline's composition failures and partitions endpoint corrections into full repairs and error relocation. Controlled geometric interventions provide exact labels for two individually label-preserving edits and their label-changing composition. In a paired coordinate comparison, 79–94% of endpoint repairs from change supervision introduce a constituent error across three training seeds. The same accounting identifies a stronger intervention: geometric inputs combined with change supervision raise joint correctness from 0.051–0.055 to 0.384–0.422, with the same ordering on a separate parent-disjoint bank. An exposure-matched visual experiment shows that additional singleton supervision can also produce constituent-preserving gains. Together, these results establish a practical criterion for compositional repair: evaluate both the endpoint a model corrects and the constituent successes it retains.

# Keywords

compositional generalization; model repair; error relocation; counterfactual evaluation; relational representations

# TL;DR

Correcting a composed prediction can break its constituents; paired evaluation separates error relocation from genuine compositional repair.
