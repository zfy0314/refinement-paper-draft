# Current content map

The manuscript uses the IJCAI26 refinement study as its primary source and KIM
only for background. The table maps the current section structure to reused and
new material; page compression is deferred.
The manuscript uses a direct conference-paper voice. Provenance and unresolved
verification work belong to this internal map and the verification documents,
not to the paper's results narrative.

Source roots:

- **R:** `/Users/zfy/Projects/meta/IJCAI26_Structured_Policy_Refinement_Through_Policy_Rollout_Analysis/ijcai26.tex`
- **F:** the accompanying `feedback.md`.
- **K:** `/Users/zfy/Projects/IJCAI25/IJCAI25_Sample_Efficient_Behavior_Cloning/ijcai25.tex`
- **L:** `/Users/zfy/Projects/meta/llm-log-analysis`, especially `logs/IL_racecar`.

| Destination | Source material | Current adaptation / remaining evidence |
|---|---|---|
| Abstract | R method and results | Debugging analogy, semantic latent space, relative IL/RL improvements, and within-task Door Opening generalization. Door metric is episode-minimum distance. |
| 1 Introduction | R motivation, K expert-strategy input, F clarity feedback; supplied `teaser0909.pdf` | Approachable problem statement and tabular-analysis contribution. Supplied teaser replaces Figure 1; whitespace trimmed and stage/section descriptions moved into its caption. |
| 2.1 Structured policies and demonstrations | K; InterPReT; ALGAE; programmatic policy literature | Representation and trainable-parameter distinction, with current publication records. |
| 2.2 Program revision and execution feedback | R closest work; recent primary papers | Added ENPIRE, ASPIRE, AlphaEvolve, and Darwin Gödel Machine; distinguish edited object and feedback interface. |
| 3 Semantic policy structure | K representation; HRI26 Figure 2 schematic style; L initial code | Figure 2 pairs a new toy subgraph with executable PyTorch code, exact shared names, and colors for observations/latents/actions/trainable weights. Eq. 1 introduces the actual initial running example. |
| 4.1 Initial policy generation | R generation interface; L initial code | Initial-generation content only; refers to the graph/code illustration and actual initial equation. Separate listing removed. Full algorithm remains in Appendix B.2. |
| 4.2 Parameter fitting | R objective; L `exp_il.py` and fitted CSV values | Correct NLL direction; initialization versus training explained textually; coefficient change from −0.3 to +0.1729. |
| 4.3 Analysis-code generation | L generated tests and request history | Structure/schema define hypotheses and code before the trajectory values are analyzed. Arithmetic/sign check design and local derivative explain the running example. |
| 4.4 Rollout and tabular-trajectory collection | R logging interface; L `utils.py` and rounds 0/3 CSVs | Before/after Figure 3 uses actual latents. Observation timestamp offset discovered and documented. |
| 4.5 Analysis execution and structural revision | L matching results, model versions 1–3, and revision explanations | Run generated functions on the collected table; matched reconstruction/sign outputs motivate residual, rectification, compression, and nonpositive coefficient edits. Other simultaneous edits remain non-isolated. |
| 4.6 RL on the demonstration-aligned structure | R results; L `exp_rl.py`; original GSPO | Reward-based parameter learning after the final demonstration fit, with corrected reward-to-go/window-likelihood formulation in Appendix C.2. |
| 5.1 Environment setup and baselines | R protocol and task figures; L evaluation code; MLES primary paper | Motivate long-horizon error accumulation and stage transitions. MLP has gradients without prescribed semantics; MLES has program evolution without gradients. State evaluation and statistical protocols. |
| 5.2 Iteration and baseline comparisons | R 11-row table, RL budget comparison, and Door images; matched Door inference | Include refined vs initial IL, refined vs initial low RL, Door 3 vs 0, and refined vs MLP IL/low RL/high RL. Car tests approximate independent samples from summaries; Door test is paired. Include MLES query comparison without a test and RL efficiency formerly in 5.5. |
| 5.3 Feedback | R token table; L `count.py` arrays and metadata | Compact linear-axis Figure 5 and Table 2 wrap text. All measurements and bands are preserved. Analysis token aggregation verified across six rounds. |
| 5.4 Semantic analysis | R Section 5.5 and its semantic/diagnostic inventories | Main-text strategy table (15/10/12 categories); explain summary statistics, reconstruction errors, correlations/regressions, and temporal/action-limit checks. Legacy inventories still need their own run identifiers. |
| 6 Discussion | F concerns; L audit; R limitations | Mechanism, non-isolated edits, evaluation scope, and timing/units contract. |
| 7 Conclusion | New synthesis of supported results | Independent conclusion section, centered on semantic analysis and iteration. |
| Appendix A | R historical generation/environment prompts | Reproduce historical inputs and explain their role. Environment-code discrepancies documented in D.1. |
| Appendix B | R prompts; L matching diagnostics and driver | Test/results excerpts and six-stage loop pseudocode. Additional-test/stop/regeneration behavior retained; generation depends on policy and schema, independently of rollout values. Driver ordering is documented in internal notes. |
| Appendix C | L IL/RL/evaluation code, R results, matched Door outcomes | Optimization/evaluation details plus C.3 statistical assumptions, paired Door confidence interval, and raw/Holm p-values for six comparisons. |
| Appendix D | L policies/CSV/request history and archived-checkpoint inference | Policy-refinement examples, concise timing/sensor conventions, and the Door evaluation protocol with its distance definition. |
| Appendix E | R diagnostic inventory; L plotting script and token metadata | Diagnostic coverage, token aggregation, and refinement budgets. Strategy coverage moves to main-text 5.4. Unresolved mappings remain in internal notes. |

## Assets

- Existing IJCAI26 images remain unchanged in `figures/ijcai26/`.
- Figure 1: supplied `figures/teaser0909.pdf`, with top/bottom trim in LaTeX.
- Editable diagrams: `figures/diagrams/speed_control.tex`, paired with
  `speed_control.py`, plus `slip_before.tex` and `slip_after.tex`.
  The previous `initial_topology.tex` and `overview.tex` are retained but unused.
- Plots generated from data: `figures/generated/latent_round0.pdf`,
  `latent_round3.pdf`, and `refinement_rewards_linear.pdf` (also SVG).
- Evidence and source hashes: `data/evidence/`; extraction scripts: `scripts/`.
- Inference records and protocol: `data/verification/2026-09-09/` and
  `result_verification.md`; runner: `scripts/verify_reported_results.py`.
- Statistical computations: `scripts/compare_results.py`, with reproducible
  outputs and explicit assumptions in `data/statistics/requested_comparisons.json`.

## Evidence still needed

1. Map the initial table's 766.5 ± 100.4 and the RL rows to their exact models,
   checkpoints, and interaction budgets. The refined IL and Door rows are resolved.
2. Link other result rows to exact configurations, per-seed outcomes, and independent
   refinement repetitions; link legacy coverage tables to their model/scripts.
3. Verify the historical RL implementation and run settings behind retained
   results; distinguish these from the recovered code's current defaults.

Archived-checkpoint inference verifies the existing results without retraining.
Controlled edit/semantic ablations and reruns after correcting the logger remain
separate work.
