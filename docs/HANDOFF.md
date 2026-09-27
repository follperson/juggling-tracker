# Accuracy work handoff

Stopped at the user's request to conserve remaining usage.

## Current state

Branch: `codex/accuracy-foundations`. Changes are local and uncommitted; nothing
has been pushed. Existing edits under `legacy/` and pre-existing untracked
files belong to earlier work and were not changed by this batch. Do not stage
the entire working tree indiscriminately.

Implemented:

- Dataset assembly refuses populated destinations and duplicate source names/aliases.
- Inference uses explicit model → environment override → local v3 checkpoint.
- `benchmark` scores saved detections against reviewed event labels with fixed
  denominators, configuration/input hashes, and optional constant-rate replay.
- `detection-eval` measures one-to-one ball-center matches, misses, duplicate
  candidates, and other false positives independently of event counting.
- Evaluator outputs cannot overwrite inputs; invalid frame indices are rejected.
- CI workflow added, local test paths made portable, README/roadmap updated,
  and long tuning history moved out of configuration code.

Validation: full standard suite 321 passed / 3 detector tests deselected.
Lint and lockfile check passed. Re-verified on 2026-09-27 before commit, with
the same result. The optional detector tier now reports 2 passed / 1 skipped;
it previously failed because `test_export_coreml_real` ran without the optional
`coremltools` dependency installed. GitHub CI has not run.

## Resume here

1. Done on 2026-09-27. This batch is committed on `codex/accuracy-foundations`,
   excluding the unrelated `legacy/`, `scripts/` and `.superpowers/` changes.
   Nothing has been pushed.
2. Review difficult detection frames and assemble an untouched recording-level
   holdout before training or changing detector defaults.
3. Independently label run boundaries/catch totals/drop times before repairing
   run semantics and persistent live event identities.

Do not raise confidence or resolution globally based on these diagnostics:
confidence increases severely hurt recall on one difficult clip; higher
resolution adds false positives on the long clip. No detector thresholds,
weights, or counting algorithms were changed.

Details:

- [Plan and verification](superpowers/plans/2026-09-07-accuracy-foundations.md)
- [Measured detector findings](detection-findings-2026-09-07.md)
- [Benchmark commands and formats](benchmarking.md)

Detailed diagnostic JSONs are local under
`outputs/accuracy-review-2026-09-07/`; their measured summary is preserved in
the findings document above.
