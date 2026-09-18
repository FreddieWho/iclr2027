# BRIDGE_R_EVIDENCE_TABLE (dev-only; holdout never opened)

| claim | number | numerator | denominator | split | status | artifact |
|---|---|---|---|---|---|---|
| DINOv2-B best dev atomic | 0.728 | — | 760 quartets | dev | measured | artifacts/bridge_r/curve/v1a_dinov2_R0_pooled_lr_n1000/result.json |
| DINOv2-B best both-correct frac | 0.609 | 463 | 760 | dev | measured | same |
| DINOv2-B B1 gate | FAIL (0.728<0.85) | — | — | dev | gate verdict | artifacts/bridge_r/DEV_SELECTION.json |
| SigLIP2-L best dev atomic | 0.638 | — | 760 quartets | dev | measured | artifacts/bridge_r/curve/siglip2_R1_mean_patch_mlp128_n1000/result.json |
| SigLIP2-L B1 gate | FAIL (0.638<0.85) | — | — | dev | gate verdict | artifacts/bridge_r/DEV_SELECTION.json |
| plateau size selection | n=1000 (all 12 configs) | — | — | dev | frozen rule | artifacts/bridge_r/DEV_SELECTION.json |
| DINOv2 pins | model f9e44c81 / BitImageProcessor | — | — | — | pinned | feats_dinov2_{train,dev}.manifest.json |
| SigLIP2 pins | model 1b426889 / SiglipImageProcessor | — | — | — | pinned | feats_siglip2_{train,dev}.manifest.json |
| DINOv3-L | BLOCKED_EXTERNAL_ACCESS | — | — | — | external block | reports/bridge_r/R0_AUDIT.md |
| holdout v2 | 895 quartets, SEALED | — | — | holdout | untouched | HOLDOUT_USED.json absent |
| matched singles | NOT BUILT (no B1 pass) | — | — | — | skipped-by-gate | — |
| comp-vs-single delta | UNTESTED (forbidden) | — | — | — | gate verdict | — |
