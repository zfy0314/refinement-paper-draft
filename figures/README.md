# Manuscript figures

Figure 1 uses the supplied `teaser0909.pdf`. Its LaTeX inclusion trims 69 bp from
the bottom and 127 bp from the top, leaving a small margin around the artwork.
The source PDF is unchanged. The previous overview's stage descriptions and
references to Sections 4.1–4.6 are now in the caption.
The caption follows the teaser's order: initial policy, demonstration fitting,
analysis-code generation, rollout collection, analysis execution/structural
revision, and RL. Figure 3's traces are introduced in Section 4.4 and its results
and edits in Section 4.5.

Editable TikZ diagrams are in `diagrams/`:

- `overview.tex`: previous debugging/refinement overview, retained but unused.
- `speed_control.tex`: toy speed-control subgraph, left panel of Figure 2.
  The right panel includes the corresponding `speed_control.py` directly.
  Graph and code use identical names: `speed`, `target_speed`, `speed_error`,
  `gain`, and `throttle`. Colors distinguish observations, latents, actions,
  and trainable weights; dashed rounded boxes denote operations, following the
  simple schematic style of the HRI26 Figure 2 reference.
- `initial_topology.tex`: previous full initial speed topology, retained unused.
- `slip_before.tex`, `slip_after.tex`: recorded subgraphs in Figure 3.

`generated/` contains plots made by `scripts/prepare_evidence.py` from the original
CSV rollouts and numerical reward arrays. Figure 3 uses all 1,000 rows of rounds
0 and 3, with shared limits. Figure 5 uses `refinement_rewards_linear.pdf`, with
a linear return axis extending below zero to preserve the full uncertainty
bands. It is sized for a half-width `wrapfigure`; the former symmetric-log
assets are retained but unused. Table 2 uses a matching half-width `wraptable`.

`ijcai26/` preserves eight byte-identical source images from
`/Users/zfy/Projects/meta/IJCAI26_Structured_Policy_Refinement_Through_Policy_Rollout_Analysis/figures/`.
The two environment images are Figure 6 in Appendix A.2. The four Door Opening
sequence images remain in the main text as Figure 4; the feedback plot is Figure 5.
The original IJCAI teaser and feedback raster are retained for provenance;
the manuscript uses the supplied teaser and reconstructed feedback plot instead.

Archived-checkpoint inference reproduces all four Door Opening numbers. The
caption now specifies episode-minimum distance, mean and population standard
deviation across ten sampled-policy rollouts. See `result_verification.md`.

Data, selected programs, diagnostic outputs, and source hashes are in
`data/evidence/`. Generated manuscript PDFs and review renders remain in ignored
`build/`, `output/`, and `tmp/` directories.
