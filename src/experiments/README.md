# Comparative Experiments

This directory contains the comparative analyses used in **Section 4** of the paper to characterize SynthForensics relative to nine existing publicly available synthetic-video benchmarks (and the FF++/DFD real-video baselines).

Each sub-folder is a self-contained pipeline that takes one or more video corpora as input and produces the per-video metric files plus the aggregate statistics used to draw the figures and tables in the paper.

## Experiments

| Experiment | Metric | Paper Section | Folder |
|---|---|---|---|
| Landmark Stability | FAN-based per-landmark heatmap peak $\bar{a}$ and landmark completeness $\mathrm{Comp}_\tau$ at multiple thresholds | Section 4.1 | [`Landmark_stability/`](Landmark_stability/) |
| Face Quality | TOPIQ-NR-Face score $Q$ (face-conditioned no-reference image quality) and 1D Wasserstein distance against FF++/DFD | Section 4.2 | [`Face_quality/`](Face_quality/) |
| Identity Audit | ArcFace identity similarity (I2V identity preservation, T2V similarity to the source subject) and identity count | Appendix B.7.3 | [`Identity_audit/`](Identity_audit/) |

The third comparative analysis of Section 4 (the paired-comparison **Human Study**, Section 4.3) is hosted separately under [`../survey/`](../survey/), which contains the survey export and the analysis pipeline.

The **Identity Audit** (Appendix B.7.3) characterizes the identities depicted in SynthForensics rather than comparing it with other benchmarks.

The [`Caption_coverage/`](Caption_coverage/) folder groups the structured source captions into content categories with keyword rules. This analysis is not reported in the paper.

## Reproducing the Results

Each experiment folder ships with its own README documenting:
- Required inputs (video corpora, expected directory layout)
- Environment setup (Python version, key dependencies)
- The exact commands used to produce the per-video metric files
- The aggregation / bootstrap / Wasserstein-distance scripts that turn the per-video files into the numbers reported in the paper
