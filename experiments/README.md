# Aurora — autonomous research loop

Self-paced research loop that proposes, math-checks, implements, runs,
and logs single-shot transformer training optimizations against a
frozen baseline.

## What it does (per iteration)

1. Reads `idea_bank.jsonl` to avoid duplicates
2. Proposes ONE novel optimization idea + writes math gate
3. Creates an isolated git worktree
4. Implements the change (LoC budget: 500)
5. Runs `transformer_primitives` + `transformer_smoke` as fast gates
6. Trains 1 epoch on `enwik8_10m` with the frozen baseline config
7. Compares val BPB vs baseline
8. Appends result to `idea_bank.jsonl`
9. Cleans up the worktree (loss) or keeps it for review (win)
10. Schedules itself to fire again via `ScheduleWakeup`

Wall clock per iteration: ~25–30 min on GTX 1650.

## Setup

```bash
# 0. Make sure the 10M enwik8 corpus is present (one-time)
ls data/enwik8_10m.txt || bash scripts/get_corpus.sh enwik8-10m

# 1. Establish the frozen baseline (one-time per machine, ~22 min)
bash experiments/bootstrap_baseline.sh

# 2. In Claude Code, start the autonomous loop
/loop "Read experiments/RESEARCHER.md and execute exactly one full iteration of the workflow described there. Schedule the next iteration via ScheduleWakeup when done."
```

The loop runs until you interrupt it (`Ctrl-C` in Claude Code).

## File layout

| Path | Tracked? | Purpose |
| --- | --- | --- |
| `RESEARCHER.md` | yes | Per-iteration workflow prompt |
| `README.md` | yes | This file |
| `baseline_config.json` | yes | Frozen CLI args used for every run |
| `bootstrap_baseline.sh` | yes | One-time baseline producer |
| `idea_bank.jsonl` | yes | Append-only log of all attempts |
| `baseline.jsonl` | no | Machine-specific baseline metrics |
| `baseline.ckpt` | no | Baseline checkpoint binary |
| `results/<id>/` | partial | Per-experiment artifacts (no `.ckpt`) |
| `wt/<id>/` | no | Git worktree per experiment |

## Calibration caveats

- **Single seed**: screening uses one fixed seed for speed. A win at
  one seed is a *candidate*, not a confirmed result. Validate
  promising candidates with 3 seeds + longer training before merging.
- **1 epoch**: early-training dynamics can dominate. A change that
  helps at 1 epoch may not help at convergence (and vice versa).
- **BPB noise floor**: empirically ~0.005–0.01 between identical runs
  on this config. The default win threshold (`-0.02`) is set ~2× the
  noise floor; tune in `baseline_config.json:win_threshold_delta_bpb`
  if needed.

## Inspecting results

```bash
# wins so far
jq -r 'select(.status == "win") | "\(.id)\t\(.delta_bpb)\t\(.mechanism)"' \
  experiments/idea_bank.jsonl

# attempts by area
jq -r '.area' experiments/idea_bank.jsonl | sort | uniq -c | sort -rn

# rejected math (often interesting — failed for principled reasons)
jq -r 'select(.status == "rejected_math") | "\(.id): \(.reason)"' \
  experiments/idea_bank.jsonl

# kept worktrees (wins awaiting your review)
ls experiments/wt/
```

For any win:
```bash
cat experiments/results/<id>/proposal.md
cat experiments/results/<id>/diff.patch
```
Then validate by running with 3 different seeds and (optionally) more
epochs before deciding to merge to the main branch.

## Resetting

```bash
# nuke all attempts, keep baseline
rm experiments/idea_bank.jsonl && touch experiments/idea_bank.jsonl
rm -rf experiments/results/* experiments/wt/*

# re-establish baseline (if hyperparams or hardware changed)
bash experiments/bootstrap_baseline.sh --force
```

## Stopping the loop

Interrupt the `/loop` invocation in Claude Code. The loop's
`ScheduleWakeup` chain dies with the session.
