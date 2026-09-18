# R0 Repository Audit — Bridge-R (2026-09-18)

Repo: `https://github.com/FreddieWho/iclr2027`, local HEAD `50a0d45` at audit start
(tracked tree clean for Bridge-R paths; pre-existing untracked clutter untouched).

## Frozen claims (must not be disturbed)
- Headline: 151/176 = 85.8% conditional compositional miss, fresh `holdout_909`,
  one-shot sealed (`reports/last15h/E1_PREREG.md`, `E1_VERDICT.md`).
- `reports/FINAL_EVIDENCE_TABLE.md` is the only main-text number source.
- Paper limitations already admit architecture generality untested (Sec.07 ii).

## B01 pilot gaps confirmed (motivate Bridge-R, do not reuse as confirmation)
1. Hand-rolled normalize/resize, no official processor (`experiments/bridge/b01_pilot.py`).
2. DINOv2 fed at 224/BILINEAR; official `preprocessor_config.json`
   (BitImageProcessor) is resize shortest-edge 256 + center-crop 224, BICUBIC,
   ImageNet mean/std. Protocol deviation, not a valid DINOv2 baseline.
3. Train/eval nuisance RNG reused `60000 + qi` across splits.
4. Eval = 200 quartets from viewed `u01/quartets_eval.npz` (development data only).
5. No matched-single control; composition vs single difficulty never separated.
6. Probe ladder exceeded the new cap (LR + LR-CV/scale + 2 MLPs); Bridge-R caps at LR + MLP-128.

## Environment
- CPU-only (64 threads, 1TB RAM); torch 2.11+cu130, transformers 5.6.2, sklearn needs
  `LD_LIBRARY_PATH=/opt/anaconda3/lib`; **torchvision missing** (required by default
  processor backends; `requirements-models.txt` already lists it → install 0.26.0 CPU).
- Disk 97% full (1.5T avail): cache features locally, commit manifests + SHA + compact results only.
- HF cache: DINOv2-B `f9e44c81…` (346MB); CLIP-B two snapshots (B01 did not pin which —
  another reason B01 is not confirmatory).

## Model access decision
- `facebook/dinov3-vitl16-pretrain-lvd1689m`: `gated=manual`, no token, no cache →
  **status = BLOCKED_EXTERNAL_ACCESS**. Per protocol: no wait, no bypass.
- Tier-2 fallback `google/siglip2-large-patch16-384`: ungated Apache-2.0,
  sha `1b426889…`, official `SiglipImageProcessor` 384 config on file.
- Bridge-R matrix: **DINOv2-B (Tier 1) + SigLIP2-L/384 (Tier 2 fallback)**.
  SigLIP2 weights (~1.2GB) to be downloaded at pinned revision; CPU forward.

## Data generator
- `make_relational_scenes(n, seed, min_margin)` is deterministic and reusable with
  fresh seeds; existing pools use seeds 101/202/303/404/505/606/707/808/909.
- Bridge-R uses NEW generator seeds (default 26091801/02/03; conflict-check before lock)
  and writes to `artifacts/bridge_r/` (never touches sealed `holdout_909`).
