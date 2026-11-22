# Meschke dataset importer — execution notes

Branch: claude/juggletrack-flywheel-turn2.

## Corpus verification (before coding)
- data/raw/meschke/csv/ and data/raw/meschke/videos/: 159 files each, already fully mirrored
  (not just the probe's sample). Row count == frame count exactly on ss3_id_016 (6150/6150).
- Column counts across the corpus: even only (4,6,8,10,12,14,16) -> 2 to 8 balls per file. No
  odd-column files.
- Sentinel check: (0,0) and negative-coordinate rows exist but are rare per the probe notes.
- Naming-pairing quirk found (not handled by the importer, since csv_path/video_path are passed
  explicitly by the caller): 2 of 159 stems don't match by simple `.csv`<->`.MP4` replacement --
  `ss(6,6)(2,6x)_id_107.csv` pairs with `ss(66)(26x)_id_107.MP4` (commas dropped from the video
  filename during the original mirror/download), and `trick_5bShowerMultiplex_id_886.csv` pairs
  with `trick_5bShowerMultiplex_id_886.MP4.MP4` (doubled extension). Worth a lookup table if a
  future bulk-export script needs automatic pairing.

## Ball-size measurement (for `DEFAULT_BOX`)
Extracted frame 1000 of data/raw/meschke/videos/ss3_id_016.MP4 (480x848), cropped a 120x120px
window centered on that frame's labeled point (row 1000 of the CSV, ball 1: (325, 654) px) --
a blue ball held at a catch. Visually inspected the crop (filled roughly half the 120px window).
Confirmed with HSV color segmentation + largest-connected-component: bbox 30x34px, area-equivalent
diameter ~31px. That's within the probe notes' expected ~25-35px range at 480px width.
31/480 = 0.0646, rounded to `DEFAULT_BOX = 0.065` (normalized to width; applied as a fixed pixel
square using width's scale for both axes, so portrait frames like 480x848 don't get a
stretched/rectangular box).

## Build
`src/juggletrack/data/meschke_import.py`:
- `load_meschke_trajectories(csv_path, *, fps, width, height)` -- per-ball Detection
  trajectories, sentinel points dropped, frame_idx = CSV row index, no frame-count reconciliation
  (event-eval ground truth only; the caller supplies fps/width/height from its own video read).
- `export_meschke_source(csv_path, video_path, out_dir, *, frame_stride=5, max_frames=120,
  box=None, seed=0)` -- one VideoReader pass, seeded uniform subsample when stride-spaced
  candidates exceed max_frames, keeps only chosen frames with >=1 visible ball, writes canonical
  single-category COCO (category id 1 "ball") + manifest.json with the citation line. Row/frame
  count mismatch: truncates silently if <=2, raises ValueError if >2.

TDD: wrote `tests/test_meschke_import.py` first (6 tests), confirmed all failed with
`ModuleNotFoundError` before writing the module, then implemented and got all 6 green on the
first pass. Full suite: 143 passed, 2 deselected (detector-marked, matches existing config).
ruff clean on both new files (repo-wide `ruff check .` has 160 pre-existing errors, all in
`legacy/`, unrelated to this change).

## Concurrent-write hazard (resolved)
This worktree is shared with another actively-committing agent (HEAD moved e9f55a1 -> f464d11 ->
e1f374f -> 51a81aa while I worked, none of which were my commits). My first `git add` + bare
`git commit` raced with that process: my two files landed inside their commit
("fix: measure ball's pre-crop relative size against the long side") instead of getting their own
message. The other agent's session subsequently rewrote that commit (now 032177c), which dropped
my files back to untracked (content on disk was untouched throughout). Re-verified full suite
green at that HEAD, then re-committed with a path-scoped `git add <files> && git commit -m "..."
-- <files>` (per coordinator guidance) as commit 575f996, confirmed via `git show --stat HEAD` to
contain exactly `src/juggletrack/data/meschke_import.py` and `tests/test_meschke_import.py`
(386 insertions, no unrelated files).

## Post-commit exports (outputs/turn3/meschke-sample/)
Ran `export_meschke_source(frame_stride=5, max_frames=120, seed=0)` on 3 diverse siteswap prefixes:
ss3_id_016 (3-ball cascade), ss441_id_013 (441), ss(4,2t)0_id_174. All three: 480x848, 30fps,
n_images=120, n_boxes=360 (3 balls visible every chosen frame), box_px=31.2.

## Event-core probe: ss3_id_016
Loaded all 3 balls' trajectories (6150 detections each, 18450 merged), fed into
`analyze_detections` with default `AnalyzeConfig`. Result: 0 arcs, 0 runs, 0 catches, 0 drops.

Root cause: `filter_static_detections` and the greedy linker both work perfectly (3 fragments of
exactly 6150 points -- one per ball, tracked continuously for the whole 205s video, no ID
switches, since these are ground-truth points). `_split_ballistic` finds 319 locally-consistent
pieces, but none reach `ay` in the default `g_range`-implied band [0.25, 4.0] -- the largest
observed is ay=0.177 (dur=0.83s, rmse=0.0003, a clean fit). The y-coordinate across the whole
video only spans [0.478, 0.810], i.e. this footage's tight/close framing shows barely 1/3 of the
normalized frame height as the ball's actual flight range, so real ballistic curvature comes out
much smaller than the default calibration (tuned for `simulate_cascade`-style full-frame flights)
expects. Confirmed by re-running with `AnalyzeConfig(g_range=(0.05, 1.0))`: 58 arcs, 6 runs, 26
catches, 56 throws, 1 drop -- the event core does work on Meschke ground truth once g_range is
recalibrated per this footage's framing, but the default config produces nothing.
