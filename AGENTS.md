# Working in juggletrack

juggletrack counts juggling runs, catches, and drops from video. A detector
proposes ball positions. Everything after it is physics on normalized
coordinates: `y` grows downward and times are in seconds. `README.md` covers
setup, the commands, and the output files.

## Where to work

- Work in a git worktree off `main`, one concern per branch.
- Footage, saved detections, and weights are local and gitignored. In a new
  worktree, link them from the main checkout: `ln -s <main>/data data`, and the
  same for `outputs` and `models`. Read through those links and write nothing
  through them.
- Pass `--out` with a directory of your own to every command that writes. The
  default `outputs/<stem>` lands in the shared data.

## Checks

- CI runs `uv run pytest -q` and `uv run ruff check src tests scripts` on
  Python 3.12. `.python-version` pins the same interpreter locally.
- CI renders CLI output in color. Reproduce a CLI assertion that fails only in
  CI with `FORCE_COLOR=1`.

## Changing the event core

The event core is `src/juggletrack/arcs/`, `events/`, `detect/cluster.py`,
`analyze.py`, and `pipeline/realtime.py`. Unit tests stay green through changes
that move real counts, so prove parity on the saved sessions. The commands are
in `docs/benchmarking.md` under "Event-core parity".

- A change that should keep results the same ends in `PARITY OK`. For a change
  that should move results, report each session whose run, catch, or drop count
  changed, and why.
- Take both snapshots on one machine with one interpreter.
- Quote live numbers from `juggletrack live --detections`. A replay on
  `frame_idx / fps` can report different counts than the recorded clock.
- The fast paths in `arcs/extract.py` must agree with `fit_arc` exactly. The
  `*_matches_reference_*` tests in `tests/test_extract.py` hold that line.

## Decisions already made

Read the linked record before you reopen one of these.

- Accuracy figures are diagnostics, measured on clips that were also used for
  tuning. Read `docs/benchmarking.md` before you quote or compare one.
- The thresholds that suppress duplicate boxes and duplicate arcs were chosen
  by a recorded sweep. See `docs/parameter-tuning.md`.
- Hard negatives for training come only from frames that pass the
  contamination gates. See
  `docs/superpowers/plans/2026-07-19-negatives-reaudit-findings.md`.
- CoreML export is blocked by a measured torch and coremltools version
  mismatch. See `docs/superpowers/plans/2026-07-19-plan4-bench-findings.md`.

## Pull requests

- Title each pull request `type(scope): subject`.
- When a pull request stacks on another, retarget it with
  `gh pr edit <n> --base main` before you merge it. The repository keeps merged
  branches until someone deletes them.
