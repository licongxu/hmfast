---
name: write-paper-section
description: >-
  Drafts or revises a section of an astrophysics or cosmology manuscript based
  on hmfast results. Use when the user asks for a draft, an edit pass, or a
  publishability review of a section, including its figures.
---

# Write paper section

Follow `.claude/rules/paper-writing.md` strictly. Highlights:

- **No em dashes or en dashes** as prose punctuation. Replace with comma, semicolon, colon, or parentheses. En dashes inside numeric ranges in equations are acceptable only when the journal style sheet requires it.
- Define every symbol on first use. Past tense for what was done, present tense for results.
- Quantitative claims only: every "improves", "matches", or "agrees" needs a number and an uncertainty.

## Drafting workflow

1. Read the latest version of the section in the manuscript repo.
2. Read the result files (`run.json`, plots, tables) the section will reference.
3. Draft prose that points to specific figures and tables by `\ref{}` label, not "the figure above".
4. Insert `\citep{?}` placeholders for any reference you are not sure about; do not invent citations.

## Figure review (mandatory, VLM-driven)

For every figure referenced or added:

1. Generate the figure as PDF (for the manuscript) **and** PNG (for visual inspection).
2. **Open the PNG with the Read tool** and inspect it. Check axis labels, units, legibility at print size, color-blind safety, panel alignment, and whether the caption is self-contained.
3. If anything is wrong, fix the plotting script and regenerate. **Loop until the figure is publication-quality.**
4. Record the final commit hash in the figure's source comment.

## Final pass

- `grep -nP '[\x{2014}\x{2013}]' <section>.tex` and ` -- ` and replace any matches.
- Compile the manuscript and confirm there are no missing references or undefined labels.
- Confirm every quantitative claim links back to a number in `run.json` or a table file.
