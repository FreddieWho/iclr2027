# Research: Competitive Landscape & Novelty Assessment for an ICLR 2027 Sports-Formation Representation-Learning Paper

**Target submission:** Representation learning on multi-agent soccer formations — (1) an "Action-Mode Spectrum" decomposing a learned representation's sensitivity by how displacements are distributed over entities (whole-team translation vs subgroup reallocation vs local disruption); (2) a dual-channel encoder separating absolute-deployment "context" semantics from globally-translation-invariant "intrinsic" semantics (2:1 update-allocation ratio on a shared encoder); (3) normalized local geometry (Jacobian Gram matrix, graph-Laplacian spectral features) predicting a frozen model's response to controlled support-reallocation interventions; validated on IDSSE (dev) and 8 matches of SoccerTrack-v2 (independent provider).

---

## Summary

The sports-trajectory/formation area is active but still a niche at ICLR: recent main-track acceptances exist (Sports-Traj @ ICLR 2025; JointDiff and "Solving Football" @ ICLR 2026), so a non-foundation-model domain paper is viable if framed as a *methodological* contribution. No direct novelty collision was found for the Action-Mode Spectrum or the geometry→intervention-response prediction pipeline; the closest prior art clusters around (i) equivariance-measurement work (Lie-derivative / Lenc-Vedaldi line), (ii) interchange-intervention/causal-abstraction evaluation (Geiger et al.), and (iii) TacticAI's symmetry-constrained geometric learning on soccer — each differs in what is decomposed, what is predicted, and what is intervened on. Reviewers will expect SoccerNet-ecosystem awareness, modern GNN/Transformer baselines (TacticAI, UniTraj-class models), and justification for IDSSE given its restricted access; the SoccerTrack-v2 cross-provider confirmation is a genuine strength worth foregrounding.

---

## (a) Most relevant recent papers at ICLR / NeurIPS / ICML / adjacent venues

1. **Sports-Traj: A Unified Trajectory Generation Model for Multi-Agent Movement in Sports** (Xu & Fu, **ICLR 2025**) — UniTraj, a masked-input generative model for trajectory prediction/imputation/recovery across sports; introduces the Basketball-U, Football-U, Soccer-U benchmarks. [OpenReview](https://openreview.net/forum?id=9aTZf71uiD)
2. **JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation** (Capellera et al., **ICLR 2026**) — Diffusion framework jointly denoising continuous player/ball trajectories and discrete possession events, evaluated on team-sports scenes. [OpenReview](https://openreview.net/forum?id=6jThckejtL) / [project page](https://guillem-cf.github.io/JointDiff)
3. **Solving Football by Exploiting Equilibrium Structure of 2p0s Differential Games with One-Sided Information** (Ghimire, Zhang, Xu, Ren, **ICLR 2026**) — Game-theoretic (not deep) solution of a 22-player football differential game; proof that "football" problems can clear the ICLR main-track bar via theory. [OpenReview](https://openreview.net/forum?id=vRwuBOxbsJ) / [arXiv:2502.00560](https://arxiv.org/abs/2502.00560)
4. **TacticAI: an AI assistant for football tactics** (Wang et al., DeepMind + Liverpool FC, **Nature Communications 2024**; arXiv 2023) — Geometric deep learning (D2-symmetric GNNs) on corner-kick tracking data with predictive and generative components; the reference point for representation learning + symmetry constraints on soccer formations. [Nature](https://www.nature.com/articles/s41467-024-45965-x) / [arXiv:2310.10553](https://arxiv.org/abs/2310.10553)
5. **Reconstructing Multi-Agent Soccer Trajectories Using Long-Term Multimodal Context** (Hughes et al., **AAAI 2025**) — Fuses tracking and event data with long-horizon multimodal context to reconstruct full player trajectories. [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33289)
6. **Imputing Multi-Agent Trajectories from Event and Snapshot Data in Soccer** (**CIKM 2025**) — Spatio-temporal attention model predicting all-player positions at event timestamps from partial observations. [ACM DL](https://dl.acm.org/doi/10.1145/3746252.3760868)
7. **AdaSports-Traj: Role- and Domain-Aware Adaptation for Multi-Agent Trajectory Modeling in Sports** (**ICDM 2025**) — Role- and domain-conditioned adaptation of Sports-Traj-style models; role-awareness is adjacent to the "intrinsic vs context" split. [arXiv:2509.16095](https://arxiv.org/abs/2509.16095)
8. **Multi-Agent System for Comprehensive Soccer Understanding** (**ICML 2025**, arXiv:2505.03735) — Agentic framework for holistic soccer video understanding; indicates the field's drift toward LLM/agent wrappers. [arXiv](https://arxiv.org/abs/2505.03735)
9. **U2Diff: Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling** (Capellera et al., **CVPR 2025**) — Uncertainty-aware diffusion for multi-agent sports trajectories; a natural baseline family. [CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/32755)
10. **TranSPORTmer** (ACCV 2024) and **FootBots** (ICIP 2024) — Transformer architectures for holistic trajectory understanding / motion prediction in soccer. [lab page](https://guillem-cf.github.io/JointDiff)
11. **SPORTU: A Comprehensive Sports Understanding Benchmark** (Xia et al., **ICLR 2025**) and **SportR** (ICLR 2026 submission cycle) — MLLM benchmarks for sports reasoning; define the "big-model" end of the competitive spectrum. [OpenReview SPORTU](https://openreview.net/forum?id=x1yOHtFfDh) / [SportR](https://openreview.net/forum?id=cPCGB402ff)
12. **TrajSV: A Trajectory-based Model for Sports Video Representations** (arXiv 2025) — Trajectory-conditioned sports video representations and retrieval; adjacent representation-learning framing. [arXiv:2508.11569](https://arxiv.org/html/2508.11569v1)

Classic lineage the paper must cite (pre-2024): Bialkowski et al. (formation templates / team style from tracking, ICDMW 2014, TKDE 2016) and Lucey-group role-based representations (permutation problem in tracking data); VaEP action valuation (Decroos et al., KDD 2020); Fernández & Bornn's SoccerMap; Google Research Football (Kurach et al., AAAI 2020) as the simulation environment. [Bialkowski TKDE](https://pmc.ncbi.nlm.nih.gov/articles/PMC12163489) (via review) / [GRF→simulated tracking demo, arXiv:2503.19809](https://arxiv.org/abs/2503.19809)

---

## (b) Overlapping claims — closest novelty collisions

### Adjacent areas (for positioning and citation)

- **Equivariance/invariance measurement of learned representations**: Lenc & Vedaldi, *Understanding Image Representations by Measuring Their Equivariance and Equivalence* (CVPR 2015); Gruver et al., *The Lie Derivative for Measuring Learned Equivariance* (ICLR 2023) — a principled scalar (local equivariance error, LEE) for how invariant/equivariant a frozen network is to group actions. [OpenReview](https://openreview.net/forum?id=JL7Va5Vy15J)
- **Partial / approximate symmetry in architectures**: Romero & Lohit, *Learning Partial Equivariances From Data* (NeurIPS 2022) — relaxes equivariance to learned subsets of a group; Wang, Walters & Yu, *Approximately Equivariant Networks* (ICML 2022); Huang et al., *Approximately Equivariant Graph Networks* (NeurIPS 2023); Petrache & Trivedi, *Approximation–Generalization Trade-offs under (Approximate) Group Equivariance* (NeurIPS 2023). [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ec51d1fe4bbb754577da5e18eb54e6d1-Abstract-Conference.html) / [NeurIPS 2023 AEGN](https://neurips.cc/virtual/2023/poster/72794)
- **Intervention-based evaluation of learned representations**: Geiger et al., *Causal Abstractions of Neural Networks* (NeurIPS 2021; theoretical foundation JMLR 2025) and *Inducing Causal Structure for Interpretable Neural Networks* (IIT, ICML 2022) — interchange interventions as the gold standard for what a representation encodes. [NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) / [JMLR 2025](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf). Note also Grant et al., *Addressing divergent representations from causal interventions* (ICLR 2026) — warns that input-level interventions shift internal representations OOD; a reviewer's standard objection to intervention studies. [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008487)
- **Spectral/Jacobian geometry of representations**: Wang et al., *Extended Data Jacobian Matrix* (ICML 2016) — spectrum of the data Jacobian as a complexity/behavior probe; Agarwala & Schoenholz, *Eigenvalues as predictors of where a network will fail* (FODS 2022) — local Jacobian eigenvalues of a frozen autoencoder predict failure locations; NTK literature (Jacot et al., NeurIPS 2018) establishes that the Jacobian Gram matrix *is* the NTK and its spectrum governs training/generalization dynamics. [ICML 2016](http://proceedings.mlr.press/v48/wanga16.pdf) / [FODS 2022](https://www.aimsciences.org/article/doi/10.3934/fods.2022004?viewType=HTML)
- **Disentangling pose/geometry from semantics**: Kwon et al., *Rotation and Translation Invariant Representation Learning with INRs* (ICML 2023); Achille & Soatto, *Emergence of Invariance and Disentanglement in Deep Representations* (JMLR 2018); symmetry-based disentanglement via group structure (ICLR 2026, Dang-Nhu et al.). [PMLR v202](https://proceedings.mlr.press/v202/kwon23a.html) / [JMLR](https://jmlr.org/papers/v19/17-646.html)

### Closest 3 potential novelty collisions

**Collision 1 — Gruver et al. (ICLR 2023), Lie-derivative equivariance measurement + the Lenc–Vedaldi line.**
*Overlap:* Probing a frozen model's response to controlled input transformations to characterize learned (in)variance — methodologically the same spirit as the Action-Mode Spectrum.
*Key difference:* They measure error along *named group actions* (global translation/rotation) and collapse it to a scalar equivariance error; your contribution decomposes sensitivity over a *spectrum of displacement-distribution modes* (whole-team / subgroup / local), which is not a group-theoretic decomposition — it is a combinatorial/spatial-scale decomposition of perturbations over entities, and it is tied to a prediction task (geometry → intervention response), not to symmetry certification. Position as "beyond group-action probing: mode-resolved sensitivity of permutation-structured sports representations."

**Collision 2 — Geiger et al. (NeurIPS 2021 / ICML 2022 / JMLR 2025), interchange interventions & causal abstraction.**
*Overlap:* Interventions on structured multi-component systems to validate claims about what a learned representation encodes.
*Key difference:* Their interventions are internal activation swaps used to align a network with a stipulated causal model; yours are *input-space support-reallocation interventions* used as ground truth to test whether local geometric descriptors (Jacobian Gram, Laplacian spectra) *predict* the frozen model's response. You make no causal-model-alignment claim; the intervention is the evaluation target, not an alignment tool. Still, cite them and preempt the OOD-representation critique (Grant et al., ICLR 2026).

**Collision 3 — TacticAI (Nature Communications 2024) + the partial-equivariance line (Romero & Lohit, NeurIPS 2022).**
*Overlap:* Symmetry treatment of soccer formations; context-vs-configuration decomposition is spiritually close to "partial equivariance": absolute pitch position breaks global translation symmetry, which is exactly what your dual-channel context/intrinsic split encodes. Partial G-CNNs *learn* where equivariance should hold via data; your 2:1 update-allocation dual channel *allocates capacity* between symmetry-breaking and symmetry-invariant semantics.
*Key difference:* TacticAI hard-codes D2 reflection symmetry for a corner-kick assistant; partial-equivariance work learns the symmetry level end-to-end; your contribution is an explicit architectural split with an *asymmetric training allocation* plus a sensitivity analysis showing what each channel captures — the analysis (Action-Mode Spectrum, geometry-predicted intervention response) is the novel object, not the symmetry handling itself. Frame the dual channel as "designed partial invariance with budgeted learning," not as an equivariance method.

*Assessment:* No paper found that (i) decomposes representation sensitivity by displacement-distribution mode over entities, or (ii) uses local Jacobian/Laplacian spectral geometry to *predict* a frozen model's response to structural interventions in multi-agent sports data. The combination appears genuinely unoccupied; the risk is reviewers reading (1)–(3) as incremental combinations of the above lines, so the related-work section must do the differentiation work explicitly.

---

## (c) Datasets and baselines reviewers expect

**Datasets (tracking/event data):**
- **SoccerNet ecosystem** — the de facto standard: SoccerNet-v2 action spotting (Deliège et al., IJCV 2021), **SoccerNet-Tracking** (Cioppa et al., CVPRW 2022, 200 MOT sequences), **SoccerNet Game State Reconstruction (GSR)** and ball-action-spotting challenges. Even if not used, reviewers expect the paper to situate itself against this ecosystem. [SoccerNet-Tracking](https://www.soccer-net.org/tasks/tracking)
- **Wyscout open event data** (Pappalardo et al., Scientific Data 2019; ~1,941 matches) and **StatsBomb open data** — the standard *event-data* benchmarks for action-value/VAEP-style work. [Wyscout mirror](https://github.com/koenvo/wyscout-soccer-match-event-dataset)
- **SkillCorner Open Data** — 10 matches of broadcast tracking + dynamic events (2024/25 A-League); increasingly the expected *open* tracking source. [SkillCorner opendata](https://skillcorner.github.io/opendata/)
- **SoccerTrack / SoccerTrack-v2** (Atom Scott et al.) — full-pitch 4K footage with GSR annotations, 10 matches; v2's independent-provider matches are exactly the right cross-validation choice and reviewers will recognize it. [SoccerTrack-v2](https://atomscott.github.io/SoccerTrack-v2)
- **TeamTrack** (CVPRW 2024) — multi-sport (soccer/basketball/handball) full-pitch MOT benchmark. [GitHub](https://github.com/AtomScott/TeamTrack)
- **Metrica Sports open tracking** (3 matches) and **Google Research Football** (Kurach et al., AAAI 2020; simulated trajectories, e.g., arXiv:2503.19809) — for controlled/simulated settings where interventions can be validated against ground truth; strongly consider adding a GRF-simulated arm since it removes the OOD-intervention critique entirely.
- **IDSSE** (J.League spatio-temporal data, Fujii group / DataStadium lineage) — used in OpenSTARLab-style Japanese sports-ML work; recognized in the sports-analytics community but *restricted-access*, which is a reproducibility liability at ICLR; the SoccerTrack-v2 confirmation mitigates this. [OpenSTARLab, Springer 2025](https://link.springer.com/article/10.1007/s40747-025-01965-y)

**Baselines reviewers will ask for:**
- Architecture baselines: TacticAI-style D2-symmetric GNN; permutation-invariant/role-sorted encoders (Bialkowski/Lucey lineage); Set Transformer / equivariant GNN; UniTraj (Sports-Traj) and JointDiff-family generative models as modern reference points.
- Analysis baselines: linear/MDL probes; canonical equivariance-error measures (Lie derivative, ICLR 2023) as a sanity check that the Action-Mode Spectrum recovers known global-translation behavior.
- Evaluation hygiene: cross-provider generalization (already present — highlight it), and ideally a simulated-intervention ground truth via Google Research Football.

---

## (d) Can a sports-formation paper without large models compete at ICLR main track?

**Yes — with conditions.** Evidence:
- ICLR 2026 accepted 5,355/13,763 decided submissions (**27.4%**), down from ~32% (2025); the venue is more competitive but still accepts non-LLM methodological work at scale. [ICLR 2026 retrospective](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process)
- Direct precedents of "football/sports" papers in the ICLR main track without foundation models: **Sports-Traj** (ICLR 2025), **JointDiff** (ICLR 2026), **Solving Football** (ICLR 2026 — a game-theory paper with a small-scale 22-player demo). None used large models; all sold a *method* or *structure* contribution with the sport as the testbed.
- The winning recipe in these precedents: the sport is the *evaluation domain*, not the contribution. Your paper fits this pattern — the Action-Mode Spectrum and geometry→intervention-response prediction are general claims about permutation-structured multi-entity representations, with soccer as a rich, well-instrumented testbed. Lead with the general method; do not let the paper read as "sports analytics."
- Risks to manage: (i) reviewer pool has little sports expertise → write a crisp 1-paragraph domain intro and define interventions formally; (ii) restricted IDSSE data → foreground SoccerTrack-v2 reproducibility, consider adding SkillCorner open data or GRF simulation as a public arm; (iii) "analysis-only" criticism → anchor at least one downstream utility result (e.g., intervention planning, formation retrieval, or downstream-task gain from the dual-channel split).
- Timeline note: ICLR 2027 abstract deadlines have historically fallen in mid-to-late September of the prior year (ICLR 2026: ~Sep 19–24, 2025); plan submission packaging accordingly.

---

## Sources

**Kept:**
- ICLR 2026 virtual paper list (https://iclr.cc/virtual/2026/papers.html) — confirmed sports papers in main track (Solving Football, JointDiff).
- Sports-Traj OpenReview (https://openreview.net/forum?id=9aTZf71uiD) — ICLR 2025 sports trajectory precedent.
- JointDiff OpenReview / project page (https://openreview.net/forum?id=6jThckejtL) — ICLR 2026 sports diffusion precedent.
- Solving Football OpenReview (https://openreview.net/forum?id=vRwuBOxbsJ) — ICLR 2026 football theory precedent.
- TacticAI, Nature Communications 2024 (https://www.nature.com/articles/s41467-024-45965-x) — closest soccer geometric-learning reference.
- Lie Derivative for Measuring Learned Equivariance (https://openreview.net/forum?id=JL7Va5Vy15J) — closest collision 1.
- Causal Abstractions of Neural Networks (https://proceedings.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) and JMLR 2025 foundation (https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf) — closest collision 2.
- Learning Partial Equivariances From Data (https://proceedings.neurips.cc/paper_files/paper/2022/hash/ec51d1fe4bbb754577da5e18eb54e6d1-Abstract-Conference.html) — closest collision 3 component.
- Extended Data Jacobian (ICML 2016, http://proceedings.mlr.press/v48/wanga16.pdf) and Agarwala & Schoenholz FODS 2022 (https://www.aimsciences.org/article/doi/10.3934/fods.2022004?viewType=HTML) — Jacobian-spectrum-as-predictor lineage.
- SoccerNet-Tracking (https://www.soccer-net.org/tasks/tracking), SoccerTrack-v2 (https://atomscott.github.io/SoccerTrack-v2), SkillCorner Open Data (https://skillcorner.github.io/opendata/), Wyscout mirror (https://github.com/koenvo/wyscout-soccer-match-event-dataset), TeamTrack (https://github.com/AtomScott/TeamTrack) — dataset expectations.
- ICLR 2026 review retrospective (https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process) — official acceptance statistics.
- AAAI 2025 reconstruction paper (https://ojs.aaai.org/index.php/AAAI/article/view/33289), CIKM 2025 imputation paper (https://dl.acm.org/doi/10.1145/3746252.3760868), AdaSports-Traj ICDM 2025 (https://arxiv.org/abs/2509.16095) — adjacent-venue landscape.
- ICLR 2026 divergent-representations paper (https://iclr.cc/virtual/2026/poster/10008487) — the standard objection to intervention studies.

**Dropped:**
- Paper Digest ICLR 2026 highlights and generic "sports ML" SEO/aggregator pages (aimind, catapult, kiqiq) — SEO-heavy, no primary evidence.
- Frontiers/Sage sports review articles — useful background but secondary; retained only for locating Bialkowski lineage.
- Medium blog posts (Fujii lab, cs224w) — non-peer-reviewed commentary.
- Physics/PR spectral-neural-network results (PhysRevResearch) — about weight-matrix spectra, not representation geometry; wrong subfield.
- Sloan/MIT Sloan Sports Conference abstracts — practitioner venue, low ML-novelty signal.

## Gaps

- **No exhaustive NeurIPS 2025 / ICML 2025 sports sweep**: conference virtual pages are JS-heavy and search returned few hits beyond TrajAgent (human mobility, not sports) and the ICML 2025 soccer agentic paper. A manual pass over NeurIPS 2025 accepted lists for "sports/football/trajectory" is recommended before submission.
- **"Action-Mode Spectrum" phrase check**: searched variants returned no direct collision, but the precise idea (mode-resolved perturbation decomposition over entity subsets) could exist under different terminology (e.g., wavelet/multiscale sensitivity analysis, structured-perturbation probing). A Semantic Scholar / Google Scholar citation-graph check around the Lie-derivative and EDJM papers would firm this up.
- **IDSSE provenance**: exact licensing/access terms were not verified; confirm citation and redistribution constraints before the camera-ready plan.
- **ICLR 2027 deadline dates**: inferred from the 2026 cycle; verify on iclr.cc when announced.
