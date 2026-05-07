---
description: Academic writing conventions for hmfast manuscripts in astrophysics and cosmology
---

# Paper writing

## Punctuation (hard rules)

- **Never use em dashes** (`—`, U+2014) as prose punctuation. Replace with a comma, semicolon, colon, or parentheses.
- **Never use en dashes** (`–`, U+2013) as prose punctuation. En dashes are acceptable only inside numeric ranges in equations or tables when the journal style sheet requires it (e.g. `z = 0.1\,\text{--}\,0.5`).
- Do not use ` -- ` in `.tex` source as a fake em dash.
- Use straight ASCII quotes inside `.tex`; LaTeX renders them correctly.

Before declaring a section finished, grep the source for `—`, `–`, and ` -- ` and replace them.

## Voice and tense

- **Past tense** for what was done (`We computed`, `the emulator was trained`).
- **Present tense** for results, equations, and figures (`Figure 1 shows`, `the model predicts`).
- Prefer **first-person plural** (`we`) over passive voice for actions taken by the authors.
- Keep claims **falsifiable and quantitative**. Replace qualitative phrases like `significantly improves` with the actual number and uncertainty.

## Equations and units

- Define every symbol on first use, including its dimensions.
- Inline math for short scalars (`$z=0.5$`); displayed math for anything longer than half a line.
- Use `\,` for thin spaces around units. Prefer `siunitx` (`\SI`, `\num`) when the journal supports it.
- Do not introduce a notation that conflicts with the standard cosmology literature without a one-line justification.

## Citations

- `\citep{}` for parenthetical, `\citet{}` for textual. Never let a bare `\citep{}` be the subject of a sentence.
- Cite the **earliest** relevant source plus the canonical modern reference where appropriate.
- Do not invent citations. If a reference is uncertain, mark it `\citep{?}` and flag it in the response.

## Figures (publication quality)

- Use **vector** formats (`.pdf` from matplotlib) for line plots. Use PNG at **>=300 dpi** only for raster heatmaps.
- Axis labels and tick labels must be **>= 10 pt at the printed size**. Verify after `\includegraphics` scaling.
- Use color-blind safe palettes (`viridis`, `cividis`, or Wong's eight-color set). Never rely on color alone; encode with linestyle or marker too.
- Every panel needs axis labels with units, a legend (or shared legend), and a caption that stands alone.

## Figure review loop (mandatory, VLM-driven)

For every figure added or changed:

1. Render the figure to **PDF** (for the manuscript) and **PNG** (for visual inspection).
2. **Open the PNG with the Read tool** and visually inspect it. Check legibility, overlapping labels, missing legends, clipping, panel alignment, color-blind safety, and whether the caption is self-contained.
3. If anything is wrong, fix the plotting script and **regenerate**, then re-inspect. **Loop until the figure is publication-ready.**
4. Record the final figure's git hash and seed in the caption-source comment so it can be reproduced.

## Reproducibility

- Every figure script writes the hmfast version, a config hash, and the data path used into the figure metadata or a sidecar `.json` file.
- Tables are produced by scripts that emit `.tex` fragments. Do not hand-edit generated tables.
- The manuscript repo and the run manifest must allow any figure to be regenerated from a single commit hash.
