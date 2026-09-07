# Research: ICLR 2027 — Official Logistics, Policies, and Review Facts

*Compiled 2026-09-07. All primary facts are from official ICLR 2027 pages (iclr.cc), the official ICLR blog, and OpenReview, cross-checked against independent trackers. ICLR 2027 details **are published** (CfP posted ~Jul 30, 2026), so almost nothing below is an estimate; explicitly flagged items are the exception.*

## Summary

ICLR 2027 (15th ICLR) will be held April 26–30, 2027 in San Francisco, CA, USA (Moscone Center), with abstract deadline **Sep 18, 2026 (11:59 PM AoE)** and full paper deadline **Sep 25, 2026 (11:59 PM AoE)**, reviews released Nov 5, 2026, rebuttal/discussion Nov 5–18, and final decisions Dec 16, 2026. This cycle introduces major new policies: author quotas (max 20 papers per author; max 1 paper for teams with no eligible reciprocal reviewer), a stricter mandatory AI-use disclosure policy, and tightened reviewer-accountability rules under which papers can be desk-rejected for their authors' delinquent reviewing.

## Findings

### (a) Key dates, venue

1. **Abstract deadline: Sep 18, 2026, 11:59 PM AoE (UTC-12). Full paper deadline: Sep 25, 2026, 11:59 PM AoE.** Deadlines are final; "we cannot make any accommodations for missing the abstract deadline or paper deadline." No authors can be added after the abstract deadline; author *order* can change until the full paper deadline. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [Dates](https://iclr.cc/Conferences/2027/Dates), [Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
2. **Reviews released Nov 5, 2026; author–reviewer discussion (rebuttal) Nov 5–18, 2026; reviewer–AC discussion and meta-reviewing Nov 19–Dec 16; final decisions Dec 16, 2026.** The AC guide gives finer granularity: assignment/AC adjustment Sep 26–30; review period Oct 1–21; late-review chasing/QC Oct 22–Nov 4; reviewer–AC discussion Nov 19–25; meta-reviewing Nov 26–Dec 2; AC/SAC discussions Dec 3–9; calibration/final decisions Dec 10–16. [Dates](https://iclr.cc/Conferences/2027/Dates), [AC Guidelines](https://iclr.cc/Conferences/2027/AreaChairGuidelines)
3. **Venue: San Francisco, CA, USA (Moscone Center per iclr.cc homepage); main conference Apr 26–28, 2027; workshops Apr 29–30, 2027.** OpenReview lists "ICLR 2027 San Francisco, CA, USA, Apr 26 2027." [iclr.cc](https://iclr.cc), [Dates](https://iclr.cc/Conferences/2027/Dates), [OpenReview venue group](https://openreview.net/group?id=ICLR.cc%2F2027%2FConference)
4. **⚠️ Deadline discrepancy note:** Some third-party posts (Reddit, Digg) and a stale sentence in the Reviewer Guidelines FAQ reference a **Sep 16** full-paper deadline; all current official pages (Author Guidelines, CfP, Dates page) and trackers state **Sep 25**. The deadline appears to have been moved from Sep 16 → Sep 25 after the initial CfP; trust the official Author Guidelines/Dates page and re-verify before submitting. [Dates](https://iclr.cc/Conferences/2027/Dates), [Reddit thread](https://www.reddit.com/r/MachineLearning/comments/1v9v4e7/iclr_2027_deadline_is_before_neurips_2026)

### (b) Submission format requirements

5. **Page limit: 9 pages of main text at submission; increased to 10 pages during discussion/rebuttal and for camera-ready.** Strictly enforced — papers with main text beyond the limit are desk-rejected. References do not count; unlimited bibliography pages allowed. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
6. **Template: official ICLR 2027 LaTeX style files required** (available via the ICLR website; includes boilerplate AI-use disclosure). Third-party template mirrors confirm the required AI statement is baked in. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [template mirror](https://coauthor.thesify.ai/templates/iclr-2027-latex-template)
7. **Anonymization: fully double-blind.** Any paper revealing author identity in main text **or supplementary material** will be desk-rejected. Related arXiv papers by the same authors do not break anonymity but must be cited in third person. OpenReview provides anonymized BibTeX for citing other ICLR 2027 submissions under review. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
8. **Dual submission policy:** submissions identical/substantially similar to previously published, accepted, or **concurrently submitted** work (to any conference or journal) are not allowed. arXiv preprints and workshop (non-archival) presentations are fine. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [CfP](https://iclr.cc/Conferences/2027/CallForPapers)
9. **Supplementary policy:** supplementary material is allowed and visible to reviewers and the public; reviewers are **not required** to read it (official reviewer guidance: "It is not necessary to read supplementary material"). It must be anonymized. [Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines), [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
10. **Withdrawal policy:** papers can be withdrawn any time before decisions; withdrawn-after-deadline papers remain publicly hosted on OpenReview in a "withdrawn papers" section and are de-anonymized. All submissions (including rejected/withdrawn) eventually become public with author names. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [CfP](https://iclr.cc/Conferences/2027/CallForPapers)
11. **Quotas (new for 2027):** max **20 submissions per author**; each author may be on **at most one** submission where no co-author is an eligible reciprocal reviewer (i.e., teams with no prior acceptance at a listed major venue may submit only one paper). [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [ICLR Blog](https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027)
12. **Mandatory AI-use disclosure section in the paper** (does not count toward the page limit), plus disclosure fields in the OpenReview submission form. [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)

### (c) Review criteria and scoring dimensions

13. **Review form dimensions:** reviews include a summary, strengths/weaknesses, questions for authors, and numeric scores. Recent ICLR review forms (2024–2026) use **Soundness, Presentation, Contribution (each 1–4), overall Rating (10-point scale, half-steps), and Confidence (1–5)**; a peer-reviewed analysis of ICLR 2024/2025 reviews confirms these five scored fields ("Soundness SC, Presentation SC, Contribution SC, Rating SC, Confidence SC"). The 2027 form additionally requires disclosure of AI use in writing the review. Exact 2027 scale labels should be confirmed on the live OpenReview form. [arXiv analysis](https://arxiv.org/pdf/2511.15462), [Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines), [AI Policy for Reviewers](https://iclr.cc/Conferences/2027/AIPolicyForReviewers)
14. **Reviewer obligations as officially stated:** timely, substantive reviews; adherence to Code of Ethics and Code of Conduct; flag potential Code-of-Ethics violations by Nov 18; LLM use in review writing permitted but must be disclosed, and reviewers must report their own original assessment. Review period Oct 1–21; bidding Sep 18–25. [Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines), [AI Policy for Reviewers](https://iclr.cc/Conferences/2027/AIPolicyForReviewers)
15. **Decision process:** ≥3 reviews per paper; AC meta-review with recommendation; reviewer–AC discussion Nov 19–25; SAC calibration before final decisions Dec 16. [AC Guidelines](https://iclr.cc/Conferences/2027/AreaChairGuidelines)

### (d) Desk-rejection criteria and common triggers

16. **Officially stated desk-reject triggers (2027):**
    - Main text beyond 9-page limit.
    - Any breach of anonymity in main text or supplementary.
    - Dual submission (identical/substantially similar work published/accepted/under parallel review).
    - Missing abstract by the abstract deadline (no full submission possible), or placeholder/duplicate abstracts ("will be [removed/rejected]" — abstracts must be genuine).
    - Quota violations (>20 papers/author; >1 paper for all-new-author teams).
    - **Reviewer-duty failures:** every submission must have ≥1 author registered to review ≥3 papers; authors on ≥3 submissions must review ≥6; authors who fail to produce complete, high-quality reviews by the rebuttal stage "may have their paper submissions desk rejected" (exemption for ACs/SACs/organizing chairs).
    - Code of Ethics violations, including LLM-produced falsehoods/plagiarism ("might lead to desk rejection").
    - US-sanctions restriction: as a US-incorporated entity, ICLR cannot accept submissions from certain sanctioned entities.
    [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)
17. **Empirical context:** at ICLR 2026, 779 of ~19,525 valid submissions were desk-rejected for procedural/content violations (and 497 specifically for AI-policy violations per secondary reporting); the 2027 blog notes a quarter of "no-reciprocal-reviewer" submissions at ICLR 2026 were desk-rejected and that group was accepted at ~half the normal rate — the motivation for the new quota. [ICLR Blog (2026 stats)](https://blog.iclr.cc/category/iclr-2026), [Submission policies blog](https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027)

### (e) 2027-specific changes vs 2026

18. **Author quotas (new):** 20-paper cap per author and the 1-paper cap for teams with no eligible reciprocal reviewer. Eligible venues for reciprocal-reviewer status: ICLR / NeurIPS / ICML / UAI / AISTATS / JMLR / TMLR (incl. Datasets & Benchmarks); ACL / EMNLP / EACL / NAACL / IJCNLP-AACL / CL / TACL (incl. Findings); COLM; CVPR / ICCV / ECCV / PAMI / 3DV (incl. Findings); AAAI / IJCAI / JAIR; ICRA / IROS / RSS / CoRL; KDD; COLT. **Eligibility is determined as of the abstract deadline** (clarified Aug 4, 2026; KDD and COLT were added after the initial CfP). [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [ICLR Blog](https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027), [@iclr_conf on X](https://x.com/iclr_conf?lang=en)
19. **Stricter AI policy (new/tightened):** mandatory AI-use section in the paper + submission-form disclosure, with a graded list of research subtasks (required: synthetic data generation, developing theoretical models/conceptual frameworks, formulating mathematical claims, proof assistance, methodology/experiment design, implementation, result interpretation; recommended: drafting/editing text). Authors bear full responsibility; LLM-produced falsehoods = Code of Ethics violation. This follows ICLR 2026's crackdown (497 papers desk-rejected for undisclosed AI use). [AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors), [AC Guidelines](https://iclr.cc/Conferences/2027/AreaChairGuidelines)
20. **Reviewer accountability (tightened):** explicit reviewer-paper requirements tied to submission count (1+ paper → review ≥3; 3+ papers → review ≥6) with desk-rejection risk for delinquent author-reviewers; new policies on low-quality/delinquent reviewers with AC escalation to SACs. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [AC Guidelines](https://iclr.cc/Conferences/2027/AreaChairGuidelines)
21. **Unchanged/continuing:** Reproducibility Statement remains **strongly encouraged, not mandatory** (paragraph at end of main text, before references; not counted separately — unlike NeurIPS, there is no mandatory reproducibility checklist). Ethics Statement is **recommended** (not required), placed before references; Code of Ethics acknowledgment is mandatory in the submission form. No separate Datasets & Benchmarks or Position-paper track for 2027 (single main track; subject areas include datasets/benchmarks as topics). [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [CfP](https://iclr.cc/Conferences/2027/CallForPapers)

### (f) OpenReview prerequisites authors often miss

22. **All authors (not just the submitter) need an OpenReview profile.** New profiles created **without an institutional email go through moderation that can take up to two weeks**; institutional-email profiles activate automatically. Create profiles well before the abstract deadline; keep profiles updated (they feed the reciprocal-reviewing eligibility check). [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [OpenReview docs](https://docs.openreview.net/getting-started/creating-an-openreview-profile)
23. **Abstract deadline traps:** no authors can be added after Sep 18; placeholder/duplicate abstracts are prohibited; reciprocal-reviewer eligibility freezes at the abstract deadline. [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [@iclr_conf](https://x.com/iclr_conf?lang=en)
24. **Reciprocal-review registration:** the designated author-reviewer must actually register and complete reviews, or the paper risks desk rejection. At ICLR 2026, papers were not at risk if the review invite expired due to frozen reviewer lists — but authors must still have registered in good faith. [Reviewer Guidelines FAQ](https://iclr.cc/Conferences/2027/ReviewerGuidelines)

## Sources

- Kept: ICLR 2027 Author Guidelines (https://iclr.cc/Conferences/2027/AuthorGuidelines) — primary source for deadlines, page limits, anonymity, dual submission, quotas, reciprocal reviewing, withdrawal, desk-rejection
- Kept: ICLR 2027 Call for Papers (https://iclr.cc/Conferences/2027/CallForPapers) — key dates, double-blind statement, subject areas
- Kept: ICLR 2027 Dates and Deadlines (https://iclr.cc/Conferences/2027/Dates) — full timeline including rebuttal and decision dates
- Kept: ICLR 2027 Reviewer Guidelines (https://iclr.cc/Conferences/2027/ReviewerGuidelines) — reviewer timeline, obligations, FAQ (note: contains stale "Sep 16" reference)
- Kept: ICLR 2027 Area Chair Guidelines (https://iclr.cc/Conferences/2027/AreaChairGuidelines) — granular process dates, new-policy enforcement duties
- Kept: ICLR 2027 AI Policy for Authors (https://iclr.cc/Conferences/2027/AIPolicyForAuthors) — mandatory AI disclosure mechanics
- Kept: ICLR 2027 AI Policy for Reviewers (https://iclr.cc/Conferences/2027/AIPolicyForReviewers) — LLM-in-review disclosure rules
- Kept: ICLR Blog, "Submission policies for ICLR 2027" (2026-09-02) (https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027) — official rationale for quotas; ICLR 2026 desk-reject stats
- Kept: ICLR Blog category ICLR 2026 (https://blog.iclr.cc/category/iclr-2026) — 2026 acceptance stats (19,525 valid submissions, 779 desk rejects, 27.4% acceptance)
- Kept: OpenReview ICLR 2027 venue group (https://openreview.net/group?id=ICLR.cc%2F2027%2FConference) — venue city/date, submission window
- Kept: OpenReview profile docs (https://docs.openreview.net/getting-started/creating-an-openreview-profile) — moderation process
- Kept: arXiv 2511.15462 — empirical confirmation of ICLR review-form scored fields (Soundness/Presentation/Contribution/Rating/Confidence)
- Kept: @iclr_conf on X — official clarifications (KDD/COLT added; eligibility as of abstract deadline; ACL & CVPR Findings qualify)
- Dropped: waset.org — scam clone conference, not ICLR
- Dropped: Reddit/Digg/LinkedIn/quasa/aiweekly/happycapy — secondary commentary, used only to flag the Sep 16→Sep 25 deadline change
- Dropped: underleaf/thesify/scispace template pages — SEO mirrors, only corroborated template/AI-statement requirement

## Gaps

1. **Exact 2027 review-form scale labels** (e.g., whether the 10-point rating anchors or the 1–4 wording changed) could not be verified verbatim from the live OpenReview form; values above are the confirmed 2024–2026 structure plus the documented 2027 AI-disclosure addition. Check the review invitation on OpenReview after Oct 1, 2026.
2. **Sep 16 vs Sep 25 discrepancy:** official pages now uniformly say Sep 25, but no official change-announcement post was found documenting the move; treat Sep 25 as authoritative but re-check iclr.cc closer to the deadline.
3. **Supplementary-material mechanics** (single combined PDF vs separate file, size limits, anonymized code-link rules) were not surfaced in searchable text for 2027; consult the Author Guidelines page directly.
4. Camera-ready deadlines for accepted papers (typically Feb–Mar 2027) are not yet published on the Dates page.
