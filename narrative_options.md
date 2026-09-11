# Chosen narrative: iterative refinement through semantic rollout analysis

**Title:** Iterative Policy Refinement through Semantic Rollout Analysis

**Thesis:** A structured policy's semantically meaningful latent space serves both
as an inductive bias for learning from demonstrations and as an interface for
investigating execution and revising the policy program.

The project extends KIM by reducing the expert's burden of describing a strategy.
The paper's broader question is how an initially plausible, LLM-generated
structure can be improved when its learned computations do not implement the
intended relationships. KIM supplies background, not a new contribution or a
second experimental study.

The explanation follows code debugging: generate an initial program, fit its
parameters from demonstrations, generate analysis code from its structure,
collect tabular rollouts, execute the analyses and revise the program, and fit it
again. RL then optimizes the demonstration-aligned structure. A check can describe a continuous behavioral tendency;
it need not be a complete pass/fail oracle. Semantics supplies anchors for these
checks and connects their results to editable computations.

Write as a conference paper presenting its own method and experiments. Lead with
ideas and findings; keep recovery histories, audit commentary, and bookkeeping
in internal project notes. Preserve scientific qualifications where they affect
the meaning of a result, and concentrate limitations in the Discussion.

## Current section plan

1. **Introduction:** Demonstrations do not explicitly specify policy structure;
   LLM knowledge can misalign with the task. Introduce debugging, the semantic
   tabular interface, and the overview figure.
2. **Related work:** Retain two subsections, covering structured policies and
   demonstration learning, then agentic program revision/execution feedback.
   Include ENPIRE and ASPIRE alongside recent coding-agent work.
3. **Semantic policy structure:** One merged section. Pair a small toy
   speed-control graph with matching PyTorch code, then introduce the actual
   initial speed branch through its equation. Explain concrete observations,
   latents, operations, trainable weights, and verification anchors.
4. **Iterative refinement:** Six subsections: initial policy generation;
   parameter fitting from demonstrations; analysis-code generation from the
   policy structure; rollout and tabular-trajectory collection; analysis execution
   and structural revision; and RL on the demonstration-aligned structure.
   Section 4.1 contains only initial generation and its running example. Fitting
   stays textual. Figure 3 connects the before subgraph and traces in 4.4 to the
   analysis results, after subgraph, and edits in 4.5.
5. **Experiments:** Four subsections: environment setup and baseline rationale;
   iteration gains and complete-pipeline comparisons (including RL efficiency);
   feedback-type ablation; and semantic analysis. Motivate Car Racing by error
   accumulation and Door Opening by stage transitions. Contrast MLP gradient
   learning with MLES program evolution without gradients. Include the six
   requested statistical comparisons, with their assumptions and Holm correction.
   Reuse the IJCAI26 strategy-coverage comparison and four diagnostic categories
   in 5.4. Environment images remain in Appendix A.2. Figure 5 stays linear and
   wrapped; Table 2 also wraps, and strategy coverage becomes main-text Table 3.
6. **Discussion:** Explain the mechanism, scope, diagnostic reliability, and
   future ablations of individual edits and components.
7. **Conclusion:** Close with the supported semantic-analysis/refinement result.

## Running example

Use the recorded Car Racing speed-control sequence from `logs/IL_racecar`.
The initial model labels mean wheel readings as slip. After training, its fitted
coefficient increases target speed, conflicting with its intended role. Generated
checks and explanations lead to a relative proxy, rectification, compression, and
a nonpositive parameterization over three revisions. The code/diagnostic chain
and actual before/after traces are recovered; no hypothetical edit is presented
as an observed one. Global return gains are not attributed to the isolated branch.

## Scope and evidence

The manuscript primarily develops the IJCAI26 iterative-refinement process.
Post-refinement RL stays in Sections 4.6/5.2, with implementation details in the
appendix. Archived-checkpoint inference reproduces the refined Car Racing IL
result and all four Door results. The verified IL costs are four refinement
resets and nine LLM calls; Door uses sampled actions and episode-minimum distance.
Other comparison rows and RL initialization/budgets remain to be mapped, as
detailed in `result_verification.md` and `revision_review.md`.
Car Racing significance estimates use rounded summary statistics under an
independence assumption; exact paired tests require the missing joint outcomes.
Door Opening uses matched episode minima from the same ten configurations.
Appendix C.3 reports raw and Holm-adjusted p-values for the six comparisons.

The eventual long-paper target is eight main-text pages. The current revision
prioritizes complete content and figures; page-budget compression is deferred at
the user's request. `content_mapping.md` maps each section to its sources.
