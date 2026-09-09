# NEmo 2026 submission

Anonymous LaTeX starter using the official NeurIPS 2026 style.

## Current draft status

The draft contains a detailed outline for **Iterative Policy Refinement through
Semantic Rollout Analysis**, not manuscript prose. The eight-page budget applies
to the eventual main text. The IJCAI26 refinement study is the primary source;
KIM is background only.
Related work is Section 2 and retains two subsections. A Car Racing running
example begins with a topology in Section 3 and continues through the method,
using textual parameter-fitting context and illustrated latent traces. The
debugging analogy explains the diagnostic loop. Post-refinement RL
remains in the main-text method and results, with technical details in the appendix.

- `narrative_options.md`: the chosen narrative and section budgets.
- `content_mapping.md`: source-to-subsection mapping, reusable assets, and gaps.
- `sections/`: subsection headings and brief content bullets, including appendices.

Bibliography output is temporarily commented out in `main.tex` because the
outline has no citations. Re-enable it when verified research citations are added.

## Build

Requires a TeX distribution with `pdflatex`, BibTeX, and `latexmk` (e.g. MacTeX
or TeX Live). Run from this directory:

```sh
make
# Equivalent: latexmk main.tex
```

The PDF is `build/main.pdf`. Run `make clean` to remove compiled output.
For Overleaf, upload the source tree and select `main.tex` with pdfLaTeX.

## Edit

- `main.tex`: title, packages, review mode, section order, bibliography.
- `sections/`: abstract, introduction, related work, method, experiments,
  discussion/conclusion, and appendix.
- `references.bib`: BibTeX entries; keep citation keys alphabetized.
- `figures/`: tracked figure assets, including PDF figures.
- `neurips_2026.sty`: official, unmodified style; do not edit its layout.
- `vendor/neurips2026/`: original instructions and checklist for reference.

Replace the instructional prose, paper title, and example citation. Update the
PDF title metadata in `main.tex` along with the visible title. Use `\citet{key}`
or `\citep{key}` for citations. Remove the appendix if unnecessary.

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
