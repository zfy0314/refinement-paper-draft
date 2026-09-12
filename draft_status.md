# Current manuscript status

The feedback-driven revision now contains the complete expanded manuscript:
Introduction; Related Work (two subsections); one merged Semantic Policy Structure
section; the six-part refinement method; Experiments; Discussion; and Conclusion.
Page-budget compression is deferred as requested. KIM remains background.
The prose now presents the method and experiments directly in a conference-paper
voice. Draft-history commentary, recovery narratives, and bookkeeping have been
removed from the manuscript, including captions and appendices. Scientific scope
and limitations are discussed in Section 6. The verification notes below and in
`result_verification.md` remain internal project records.

## What changed

- The abstract and introduction use the coding-agent debugging analogy and
  explicitly connect tabular analysis to semantically named policy latents.
- Headlines emphasize approximately 15% higher IL return, about one-fifth of the
  reported RL resets, and approximately 80% lower Door Opening distance in unseen
  task settings. These are improvements within the reported evaluation protocol.
- Related work adds ENPIRE, ASPIRE, AlphaEvolve, and the Darwin Gödel Machine,
  checked against their primary papers. “ENPIRE” is the paper's spelling.
- The supplied `teaser0909.pdf` appears as Figure 1 with top/bottom whitespace
  trimmed in LaTeX. Its caption retains the former overview's stage descriptions
  and references to Sections 4.1–4.6.
- Caption and method now follow: initial policy generation, demonstration fitting,
  analysis-code generation from structure, rollout/table collection, analysis
  execution and structural revision, then RL on the demonstration-aligned structure.
  Section 4.1 contains only initial generation. Diagnostic results and their
  resulting edits are together in 4.5; the trace illustration is introduced in 4.4.
- Figure 2 pairs a toy speed-control TikZ subgraph with corresponding PyTorch
  code. Names match exactly; colors distinguish observations, latents, actions,
  and trainable weights. The standalone method listing is removed; Eq. 1 retains
  the actual initial speed-control branch.
- Figure 3 combines before/after subgraphs with actual 1,000-step latent traces.
  It follows the wheel-slip-to-target-speed branch through three recorded edits.
- Figure 5 is rebuilt from the original arrays with a linear return axis and
  reduced to a half-width figure with surrounding text. The uncertainty band
  below zero is retained. Table 2 also uses a half-width wrapping layout with
  two-line column headings. Measurements and iterations are unchanged.
- The environment images have moved to Appendix A.2 (now Figure 6). Door Opening
  progression and feedback remain in the main text as Figures 4 and 5.
  Discussion and Conclusion are separate sections.
- Section 5 now has four subsections: setup/baselines, iteration and pipeline
  comparisons including RL, feedback ablation, and semantic analysis. The strategy
  table and four diagnostic categories from IJCAI26 appear in main-text 5.4.
- Six statistical comparisons are in 5.2 and Appendix C.3. Car Racing p-values
  approximate independent samples using rounded summaries; Door uses ten matched
  episode minima. All three iteration effects pass Holm correction; MLP IL and
  low RL gaps have adjusted p = 0.0579, and high RL has p = 0.1361.
  Computation and assumptions are saved in `scripts/compare_results.py` and
  `data/statistics/requested_comparisons.json`.

## Recovered evidence

Primary code/log root:
`/Users/zfy/Projects/meta/llm-log-analysis`.
The selected running example is `logs/IL_racecar`, with code snapshots, diagnostic
outputs, revision explanations, and CSV rollouts. The actual initial policy is
embedded in `prompt_iter000.json`; it is **not** the generic
`models/model_racecar_iter000.py` found elsewhere in the repository.

`data/evidence/` stores the selected policy snapshots, matched diagnostic outputs,
revision explanations, selected rollout columns, numeric feedback arrays, token
counts, and SHA-256 provenance. `scripts/prepare_evidence.py` extracts these data
and generates the trace/feedback figures without executing generated policies or
running experiments. `scripts/recover_token_usage.py` verifies the analysis row
against request metadata. `figures/diagrams/` holds editable TikZ sources.

The coefficient sequence is +0.1729, +3.8367, +6.6077, then −0.00390. The feature
changes from wheel mean to relative residual, positive residual, and finally a
compressed positive residual with a nonpositive coefficient. These are matched
program edits, not a synthesized repair story. Several other policy branches
also change, so the evidence does not isolate this branch's effect on return.

The log audit confirms a one-step offset between successor observation columns
and current-step latents. Shifted speed/lateral comparisons have zero error in
rounds 0–4; the figure uses same-forward-pass latent values. It also identifies a
historical heading/steering-angle description mismatch and unequal sensor
normalizations. Appendix D.1 documents these facts without altering the old runs.

The RL appendix now follows the recovered implementation: reward-to-go advantages
and products of action likelihoods over a window, with no GSPO length
normalization. Run-specific linkage remains to be confirmed.

## Remaining gaps

Archived-checkpoint inference now resolves the two immediate result mappings;
see `result_verification.md` and `data/verification/2026-09-09/`.
Car Racing's 881.99 ± 29.33 comes from January round 4 (`IL_racecar`, identical to
`IL_01172209`), with mean actions and seeds 1000–1009. The verified IL row now
counts four refinement resets and nine LLM calls including initial generation.
The six November final checkpoints do not reproduce this refined result under
the same evaluation.

Door Opening's four displayed results all reproduce from rounds 0–3 in `IL_door`
using sampled actions, 500 steps, seeds 0–9, and the **minimum** Euclidean
handle-to-target distance during each episode. The mean and population standard
deviation are computed across the ten episodes; final distance is different.
The observations confirm ten distinct initial configurations shared across rounds.

The remaining gaps concern the other comparison rows, RL initialization and
budget mapping, independent refinement-run counts, historical training settings,
and legacy coverage inventories. The reruns execute archived checkpoints only;
they do not train models or change the original diagnostic logs.

## Primary references added

- [ENPIRE](https://arxiv.org/abs/2606.19980)
- [ASPIRE](https://arxiv.org/abs/2607.00272)
- [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)

Prior verified references remain in `references.bib`, including published KIM,
InterPReT, MLES, and the source GSPO paper.

## Build and review

Run `make` to build `build/main.pdf`. The review copy is
`output/pdf/semantic_rollout_refinement_draft.pdf`.
The official NeurIPS style remains unmodified. Generated manuscript PDFs and
page renders are ignored by Git; source figures and evidence are tracked.
