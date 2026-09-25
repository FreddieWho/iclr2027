# A-aligned visual ablation protocol

Status: NOT RUN.

This file is a later-run contract. It is not a result, not a trained model, and not a new architecture. No GPU was rented and no training step was started for this protocol.

## Question it would be allowed to ask

On the already competent ResNet18 visual baseline, does a cross-segment interaction readout change full repair, rather than only current-state accuracy, relative to an additive ablation of that same readout?

If the direct ResNet18 is already as good, the publishable statement remains a reliability boundary. It does not become a claim that a new architecture won.

## Required comparison

All three unread arms below are NOT RUN. The finished direct-classification arm is the existing four-arm table, not a new run.

1. Same encoder, direct classification. This is the finished ResNet18 baseline in `artifacts/e1a933_review/vision_gpu_interpretation/`. A later ablation must use that same encoder family and the same renderer, not a renamed trunk.
2. Same encoder plus an interaction readout. The readout must see both segment representations at once, with a nonlinearity across segments. This is the A-side interaction correction, not a new named network.
3. Additive ablation. Same encoder, same segment representations, same training budget, but the cross-segment nonlinearity is removed. The only planned contrast between 2 and 3 is that interaction.

Do not add a free geometric bottleneck and an arbitrary head and call the pair a new arm. That was the old N03 failure mode.

## Observation contract

Segment roles must be the red and blue segments actually visible in the image. The current canonical renderer already draws those colors (`experiments/e1a933_review/vision_protocol.py`, `canonical()`; classical parser threshold in `artifacts/e1a933_review/vision_n03_analytic/result.json`). A later run may group patches or heatmaps by those visible colors.

Inference inputs stop at the image. They must not include hidden A/B indices, coordinates, AB truth, the answer to a single edit, or a combination ID. If a segment has no printed name, use a within-segment set or matching loss. Do not smuggle an unobservable index in through the loss at inference.

Oracle geometry is allowed only as explicitly labeled privileged supervision during training. It is not an inference input. A true-geometry swap is a diagnostic. It is not a knob for test labels.

If the run fits objects by color segmentation, PCA, or line fitting, it must also report the classical parser already measured on this bank. A strong classical parser on an easy image is a boundary on the claim, not a project failure.

## Measurement contract, if it is ever run

Use a new parent test before any improvement sentence. The already exposed 159-quartet canonical bank and the 178-quartet N02 bank may be used for engineering only. They must not be mixed, and a difference between them is not an improvement.

Report J3 with its denominator, atomic accuracy, AB accuracy, full repair, migration, and baseline-111 regression, for static and for the single-edit regimen. J4 is reported only if the base-inclusive field is actually stored. Otherwise write missing. Do not derive it after the fact and do not substitute marginal base accuracy.

Paper CCM is a miss rate: the joint error given both atomic predictions correct, on E. Do not store conditional joint accuracy under the name CCM. If that miss rate is not computed, write missing.

Keep total exposure and per-parent exposure separate. Matching the total image count and the optimizer-step count does not match per-parent exposure.

Account for training, selection, and inference cost separately. Freeze the BatchNorm behavior in the record. Do not use an atomic threshold to decide whether the joint result may be looked at.

## Stop rules

Stop if the only gain is auxiliary geometry error and the task counts do not move. Do not scan another loss list. Stop the architecture claim if direct ResNet18 already matches the interaction arm. Do not rename this protocol as a completed experiment.
