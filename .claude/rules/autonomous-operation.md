---
description: Rules for long-running autonomous Claude or OpenClaw sessions on hmfast (multi-day research-lab mode)
---

# Autonomous operation

These rules apply when Claude (or an OpenClaw-style agent) is launched as a long-running autonomous research lab. Treat every iteration as if no human will see the next message for hours.

## Session contract

- **Persist state on every step.** All progress must be recoverable from disk. Never rely on conversation context that will be compacted.
- **Idempotent steps.** Each action should be safe to retry. Tag generated artifacts with a content hash or run id so re-running produces the same result, not a new pile of files.
- **One claim, one artifact.** Every claim ("the model converged", "figure 3 looks good") must point to a saved artifact (log line, plot, JSON manifest) that a human can verify later.

## Logbook

Maintain a single project logbook (default: `runs/logbook.md`). Append one block per iteration with:

- timestamp (UTC), git hash, brief intent,
- commands launched and their job ids,
- artifacts produced (paths),
- a one-line conclusion, including whether to continue or stop.

Never delete past logbook entries. Mistakes stay in the record so the next session can avoid them.

## Stop conditions (mandatory)

The agent must **halt and wait for a human** if any of these happens:

1. Two consecutive runs fail with the same error.
2. A SLURM job is killed for OOM or timeout twice in a row on the same workload.
3. Estimated cost (compute time or external API) exceeds a budget the user has set in the run manifest.
4. A change would touch one of: `pyproject.toml`, `requirements.txt`, `.gitignore`, `CLAUDE.md`, anything under `.claude/`, or a file outside the repo. These require explicit human approval.
5. A test that previously passed now fails and the cause is not localized within one iteration.

## Non-destructive defaults

- Never run `git push --force`, `git reset --hard`, `rm -rf` on shared paths, or any database / web write without an explicit task entry that authorizes it.
- Prefer creating a new branch per experiment (`exp/<short-id>`) over pushing to `main`.
- Do not delete previous run outputs to free disk; report the disk pressure and stop.

## Hardware envelope

Every launched job must declare in its manifest:

- partition, `--gres`, `--time`, `--mem`,
- expected peak memory (with arithmetic),
- expected wall time (with calibration method).

If actual peak memory exceeds 1.5x the estimate, log a warning and reduce the next batch size by half.

## Heartbeat

Every iteration writes `runs/heartbeat.json` with a current timestamp and the active job id (if any). If the next iteration sees a stale heartbeat (`> 2 * --time` of the active job), it must investigate before launching anything new.

## What success looks like

A run is "done" only when:

- The logbook contains a final entry summarizing scientific outcome,
- All figures have been through the VLM review loop (`.claude/rules/paper-writing.md`),
- The manifest is complete,
- A human-readable `runs/<run-id>/REPORT.md` exists with: setup, results, figures (linked), open questions.

If any of these is missing, the run is not done. Do not claim it is.
