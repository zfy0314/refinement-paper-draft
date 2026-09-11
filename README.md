# NEmo 2026 submission

Anonymous LaTeX starter using the official NeurIPS 2026 style.

## Current draft status

The full manuscript now follows the revised semantic-analysis/debugging narrative,
with KIM as background. Related work remains Section 2 with two subsections;
Section 3 is merged; the method retains its six subsections; Discussion and
Conclusion are separate sections. The current revision prioritizes complete
content and figures, with page compression deferred.

The running example uses actual Car Racing policy versions, diagnostics, and
rollout traces from `llm-log-analysis/logs/IL_racecar`. TikZ diagrams explain
the initial topology and before/after slip computation. Figure 1 uses the supplied
`teaser0909.pdf`, cropped in LaTeX, with stage descriptions and section references
in its caption. Environment images are in Appendix A.2; Door Opening progression
and result tables remain in the main text. The feedback
plot is regenerated from the original numerical arrays using a linear return
axis. Figure 5 and Table 2 use compact layouts with text wrapping around them.
Archived-checkpoint inference now reproduces the refined Car Racing
IL result and all four Door Opening results; see `result_verification.md`.

- `draft_status.md`: current changes and recovered evidence.
- `revision_review.md`: final reading, claim audit, and remaining evidence gaps.
- `result_verification.md`: matched checkpoints, evaluation protocols, corrected
  IL refinement costs, and reproducible inference commands.
- `narrative_options.md`: the chosen narrative and section plan.
- `content_mapping.md`: section-level source mapping.
- `data/evidence/`, `scripts/`: source snapshots, selected data, provenance, and
  reproducible extraction/plotting.

The review copy is `output/pdf/semantic_rollout_refinement_draft.pdf`.

## Build

Requires a TeX distribution with `pdflatex`, BibTeX, and `latexmk` (e.g. MacTeX
or TeX Live). Run from this directory:

```sh
make
# Equivalent: latexmk main.tex
```

The PDF is `build/main.pdf`. Run `make clean` to remove compiled output.
For Overleaf, upload the source tree and select `main.tex` with pdfLaTeX.
The unmodified `wrapfig.sty` (version 3.6, LPPL) is bundled from
[CTAN](https://ctan.org/pkg/wrapfig) for builds with minimal TeX installations.

## Edit

- `main.tex`: title, packages, review mode, section order, bibliography.
- `sections/`: abstract, introduction, related work, method, experiments,
  discussion/conclusion, and appendix.
- `references.bib`: BibTeX entries; keep citation keys alphabetized.
- `figures/`: tracked figure assets, including PDF figures.
- `neurips_2026.sty`: official, unmodified style; do not edit its layout.
- `vendor/neurips2026/`: original instructions and checklist for reference.

Use `\citet{key}` or `\citep{key}` for citations. Resolve the marked evidence gaps
before preparing a submission version.

## Submission requirements

Checked against the [NEmo call](https://nemo.semantic.review/#call) on September 9,
2026:

- Long papers: 8 pages; short papers: 4 pages. References and appendices are excluded.
- Use `\usepackage[dblblindworkshop]{neurips_2026}` and anonymize supplements.
- Deadline: September 10, 2026. The public page does not specify a time zone or time.
- Submit via [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/NEmo).
- Contributions are non-archival.

The workshop page limits override the main-conference limit in the bundled
instructions. The call does not explicitly require the NeurIPS checklist;
it is retained but disabled in `main.tex`. Check the submission form for any
additional requirements. After acceptance, add `final` to the workshop package
options, replace the anonymous author block, and fill acknowledgments.

## Git tracking

Track LaTeX/BibTeX sources, style files, documentation, build configuration, and
figure assets. `.gitignore` excludes `build/`, manuscript output, LaTeX
intermediates, editor temporary files, and local environment secrets. There is
no blanket `*.pdf` rule, so PDF figures remain trackable.

The repository uses the `main` branch. No remote is configured. To publish it
later, create a remote repository and use its actual URL:

```sh
git remote add origin <repository-url>
git push -u origin main
```

## Template provenance

Downloaded on September 9, 2026 from the
[official archive linked by NEmo](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip).
The two files under `vendor/neurips2026/` and the root style file are unmodified
archive contents. Build the starter at the root; the vendor example is retained
as documentation.

SHA-256:

```text
archive: 82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9
style:   c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a
```
