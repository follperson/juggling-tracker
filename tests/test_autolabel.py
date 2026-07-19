import json

import numpy as np
import pytest

from juggletrack.arcs.extract import assign_detections, extract_arcs
from juggletrack.data.autolabel import (
    calibrate_label_boxes,
    export_hard_negatives,
    export_video_labels,
    select_autolabels,
)
from juggletrack.sim import CascadeParams, simulate_cascade
from juggletrack.types import Arc, Detection
from tests.helpers import write_test_video


@pytest.fixture()
def sim_with_junk():
    r = simulate_cascade(n_throws=8, fps=30.0, noise=0.002, seed=3)
    junk = [Detection(frame_idx=10, t=10 / 30.0, x=0.5, y=0.02),
            Detection(frame_idx=40, t=40 / 30.0, x=0.97, y=0.6)]
    return r, r.detections + junk


def test_select_autolabels_filters_unverified(sim_with_junk):
    r, dets = sim_with_junk
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    assert 0 < len(labels) < len(dets)
    assert all(lb.w == pytest.approx(0.04) and lb.h == pytest.approx(0.04) for lb in labels)
    # the junk frames must land in the review queue
    assert 10 in review and 40 in review


def test_export_writes_coco_and_frames(tmp_path, sim_with_junk):
    r, dets = sim_with_junk
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    n_frames = max(d.frame_idx for d in dets) + 1
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0, size=(320, 240))

    out = tmp_path / "labels_out"
    stats = export_video_labels(video, labels, out, review_frames=review)

    coco = json.loads((out / "annotations.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "ball"}]
    assert stats["n_images"] == len(coco["images"])
    assert stats["n_boxes"] == len(coco["annotations"]) == len(labels)
    labeled_frames = {d.frame_idx for d in labels}
    assert stats["n_images"] == len(labeled_frames)
    # every referenced image file exists with correct size metadata
    for im in coco["images"]:
        assert (out / "images" / im["file_name"]).exists()
        assert (im["width"], im["height"]) == (320, 240)
    # bbox sanity: pixel coords inside the image, area consistent
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert 0 <= x <= 320 and 0 <= y <= 240 and w > 0 and h > 0
        assert x + w <= 320 + 1e-6 and y + h <= 240 + 1e-6
        assert ann["area"] == pytest.approx(w * h)
        assert ann["iscrowd"] == 0 and ann["category_id"] == 1
    manifest = json.loads((out / "review_manifest.json").read_text())
    assert manifest["review_frames"] == sorted(set(review))


def test_export_bbox_matches_normalized_center(tmp_path):
    lb = Detection(frame_idx=0, t=0.0, x=0.5, y=0.5, w=0.1, h=0.2)
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=1, fps=30.0, size=(320, 240))
    out = tmp_path / "one"
    export_video_labels(video, [lb], out)
    coco = json.loads((out / "annotations.json").read_text())
    assert coco["annotations"][0]["bbox"] == pytest.approx([144.0, 96.0, 32.0, 48.0])


def _hand_crafted_arc(arc_id: int = 0, t_start: float = 0.0, n_points: int = 16) -> Arc:
    """Same flight shape as tests/test_drops.py's `_make_floor_candidate`: thrown
    from hand_line=0.65 with g=2.0 (ay=1.0). apex at dt=0.55, catch at dt=1.1.
    """
    return Arc(
        id=arc_id, t_start=t_start, t_end=t_start + 1.1,
        ay=1.0, by=-1.1, cy=0.65, bx=0.0, cx=0.5,
        n_points=n_points, rmse=0.0,
    )


def _labels_on_arc(arc: Arc, n: int = 16) -> list[Detection]:
    """n labels evenly spaced across the arc's flight, each lying exactly on it
    (zero residual, so `assign_detections` accepts all of them). Box size is
    0.05 for the slowest quarter (nearest the apex -- with bx=0 here, |vy| is
    exactly proportional to |dt - apex_dt|, so ranking by time-to-apex is
    identical to ranking by speed) and 0.02 for the rest: a real motion blob
    measures the ball cleanly at low speed and smears it at high speed.
    """
    dts = [arc.duration() * i / (n - 1) for i in range(n)]
    apex_dt = arc.apex_t() - arc.t_start
    order = sorted(range(n), key=lambda i: abs(dts[i] - apex_dt))
    slow_idx = set(order[: round(0.25 * n)])
    labels = []
    for i, dt in enumerate(dts):
        t = arc.t_start + dt
        w = h = 0.05 if i in slow_idx else 0.02
        labels.append(Detection(frame_idx=i, t=t, x=arc.x_at(t), y=arc.y_at(t), w=w, h=h))
    return labels


def test_calibrate_label_boxes_uses_slow_point_size():
    arc = _hand_crafted_arc()
    labels = _labels_on_arc(arc, n=16)
    calibrated = calibrate_label_boxes(labels, [arc])
    assert all(lb.w == pytest.approx(0.05) and lb.h == pytest.approx(0.05) for lb in calibrated)


def test_calibrate_label_boxes_quorum_guard():
    arc = _hand_crafted_arc()
    labels = _labels_on_arc(arc, n=16)[:3]
    # below the 8-assigned-label quorum: unchanged
    calibrated = calibrate_label_boxes(labels, [arc])
    assert calibrated == labels
    # no arcs at all: unchanged too
    assert calibrate_label_boxes(labels, []) == labels


def test_calibrate_label_boxes_preserves_centers():
    arc = _hand_crafted_arc()
    labels = _labels_on_arc(arc, n=16)
    calibrated = calibrate_label_boxes(labels, [arc])
    for before, after in zip(labels, calibrated):
        assert after.x == pytest.approx(before.x)
        assert after.y == pytest.approx(before.y)
        assert after.frame_idx == before.frame_idx
        assert after.t == pytest.approx(before.t)


def _persistent_junk_cascade(
    *, n_throws: int = 8, noise: float = 0.002, sim_seed: int = 3,
    junk_seed: int = 11, junk_density: int = 2, junk_jitter: float = 0.005,
):
    """Real cascade detections plus a persistent background false positive:
    a jittering point parked at (0.91, 0.71) -- nudged off (0.9, 0.7)'s exact
    static-filter grid line, same reasoning as test_extract.py's
    `_inject_static_cluster` -- for every frame of the whole video, at a spot
    the ball's own flight path never visits (hand_x in [0.41, 0.59], nowhere
    near x=0.91). Field motivation: an eye pupil the detector keeps firing on.

    Because this junk sits on every frame, it naturally produces both kinds
    of frame the hard-negative miner must tell apart: frames where no real
    ball is in flight (pure junk -- fair game for a zero-box negative) and
    frames where a real, arc-verified ball shares the frame with the junk
    (ambiguous -- must be skipped, since a zero-box negative there would
    un-teach a genuine ball).
    """
    r = simulate_cascade(n_throws=n_throws, fps=30.0, noise=noise, seed=sim_seed)
    real = list(r.detections)
    fps = 30.0
    n_frames = max(d.frame_idx for d in real) + 1
    rng = np.random.default_rng(junk_seed)
    junk = []
    for i in range(n_frames):
        t = i / fps
        for _ in range(junk_density):
            x = 0.91 + float(rng.uniform(-junk_jitter, junk_jitter))
            y = 0.71 + float(rng.uniform(-junk_jitter, junk_jitter))
            junk.append(Detection(frame_idx=i, t=t, x=x, y=y))
    dets = real + junk
    arcs = extract_arcs(dets)
    return r, dets, arcs, n_frames, fps


def _expected_candidates(dets, arcs, min_junk, pad=1.0):
    """Independently-computed reference for `export_hard_negatives`'s
    selection, mirroring its precedence: a frame with any arc-assigned
    detection is `ambiguous`; else if the frame's own time falls within
    `pad` of any arc's `[t_start, t_end]` it's `active` (temporally too
    close to real activity to trust, regardless of what its own unassigned
    detections look like -- held balls are unassigned by design); only the
    remainder is subject to the original all-unassigned + `min_junk` rule
    and becomes `pure_junk`.
    """
    assignment = assign_detections(dets, arcs)
    by_frame: dict[int, list[bool]] = {}
    frame_t: dict[int, float] = {}
    for d, arc_id in zip(dets, assignment):
        by_frame.setdefault(d.frame_idx, []).append(arc_id != -1)
        frame_t.setdefault(d.frame_idx, d.t)
    windows = [(a.t_start - pad, a.t_end + pad) for a in arcs]
    ambiguous: set[int] = set()
    active: set[int] = set()
    pure_junk: set[int] = set()
    for f, flags in by_frame.items():
        if any(flags):
            ambiguous.add(f)
            continue
        if any(lo <= frame_t[f] <= hi for lo, hi in windows):
            active.add(f)
            continue
        if len(flags) >= min_junk:
            pure_junk.add(f)
    return pure_junk, ambiguous, active


def test_export_hard_negatives_selects_pure_junk_frames_only(tmp_path):
    r, dets, arcs, n_frames, fps = _persistent_junk_cascade()
    assert arcs, "sanity: extraction must actually verify some real throws"
    # pad=0.0: this fixture's only junk-only content is the pre-first-throw
    # lead-in, which sits right up against the first arc -- exactly the kind
    # of near-activity content the temporal exclusion (default pad=1.0) now
    # correctly swallows. Isolate the base all-unassigned + min_junk rule
    # here; the activity-window behavior gets its own fixture/tests below.
    pure_junk, ambiguous, active = _expected_candidates(dets, arcs, min_junk=2, pad=0.0)
    assert pure_junk and ambiguous, "fixture must exercise both frame kinds"
    assert not active

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs, out, max_frames=len(pure_junk) + 5, min_junk=2, seed=0,
        pad=0.0,
    )

    assert stats["n_candidate_frames"] == len(pure_junk)
    assert stats["n_skipped_ambiguous"] == len(ambiguous)
    assert stats["n_skipped_active"] == 0
    assert stats["n_images"] == len(pure_junk)

    coco = json.loads((out / "annotations.json").read_text())
    assert coco["annotations"] == []
    assert coco["images"]
    assert coco["categories"] == [{"id": 1, "name": "ball"}]
    exported_frames = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert exported_frames == pure_junk
    assert exported_frames.isdisjoint(ambiguous)

    manifest = json.loads((out / "negatives_manifest.json").read_text())
    manifest_frames = {f["frame_idx"] for f in manifest["frames"]}
    assert manifest_frames == pure_junk
    for entry in manifest["frames"]:
        assert entry["n_junk"] >= 2
        assert len(entry["positions"]) == entry["n_junk"]


def test_export_hard_negatives_respects_max_frames_and_is_deterministic(tmp_path):
    r, dets, arcs, n_frames, fps = _persistent_junk_cascade()
    # pad=0.0: cap/determinism is orthogonal to the activity-window feature;
    # keep this fixture's original (pre-fix) candidate count so the cap math
    # below still holds. See the pad=1.0 default's own coverage in
    # test_export_hard_negatives_excludes_active_windows_despite_held_balls.
    pure_junk, _, _ = _expected_candidates(dets, arcs, min_junk=2, pad=0.0)
    cap = max(1, len(pure_junk) // 2)
    assert cap < len(pure_junk), "fixture must have more candidates than the cap"

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)

    out_a = tmp_path / "neg_a"
    stats_a = export_hard_negatives(video, dets, arcs, out_a, max_frames=cap, seed=7, pad=0.0)
    out_b = tmp_path / "neg_b"
    stats_b = export_hard_negatives(video, dets, arcs, out_b, max_frames=cap, seed=7, pad=0.0)

    assert stats_a["n_images"] == stats_b["n_images"] == cap
    assert stats_a["n_candidate_frames"] == stats_b["n_candidate_frames"] == len(pure_junk)
    assert "n_skipped_active" in stats_a and "n_skipped_active" in stats_b
    frames_a = sorted(f["frame_idx"] for f in json.loads(
        (out_a / "negatives_manifest.json").read_text())["frames"])
    frames_b = sorted(f["frame_idx"] for f in json.loads(
        (out_b / "negatives_manifest.json").read_text())["frames"])
    assert frames_a == frames_b


def test_export_hard_negatives_feeds_assemble_dataset(tmp_path):
    from juggletrack.data.dataset import assemble_dataset
    from tests.test_dataset import make_coco_source

    pos_src = make_coco_source(tmp_path, "positive", n_images=3)

    r, dets, arcs, n_frames, fps = _persistent_junk_cascade()
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    neg_src = tmp_path / "negatives_src"
    # pad=0.0: this fixture's only candidates are the pre-first-throw
    # lead-in (see rationale above); this test is about the assemble_dataset
    # integration, not the activity-window feature.
    neg_stats = export_hard_negatives(video, dets, arcs, neg_src, max_frames=5, seed=0, pad=0.0)
    assert neg_stats["n_images"] > 0

    out = tmp_path / "assembled"
    stats = assemble_dataset([pos_src, neg_src], out, val_fraction=0.5, seed=0)

    neg_split = "val" if neg_src.name in stats["val_sources"] else "train"
    neg_images = sorted((out / "images" / neg_split).glob(f"{neg_src.name}_*"))
    assert len(neg_images) == neg_stats["n_images"]
    for img in neg_images:
        label = out / "labels" / neg_split / f"{img.stem}.txt"
        assert label.exists()
        assert label.read_text().strip() == ""


def _persistent_junk_cascade_with_held(
    *, n_throws: int = 8, noise: float = 0.002, sim_seed: int = 3,
    junk_seed: int = 11, junk_density: int = 2, junk_jitter: float = 0.005,
    idle_gap_s: float = 2.0, idle_tail_frames: int = 30,
):
    """Same persistent-junk shape as `_persistent_junk_cascade`, but on a
    cascade with `include_held=True` and a slower cadence (2 balls,
    period_s=0.8, dwell_s=0.5, vs. the default 3-ball/0.45/0.25) so a hand's
    dwell window can land where NO ball is simultaneously in flight. That is
    the exact shape of the held-ball leak this fix targets: a held ball is
    arc-unassigned by design (no parabola while it's stationary in a hand),
    so under the OLD "every detection on this frame is unassigned" rule
    alone, a held-ball-only frame plus the persistent junk was
    indistinguishable from a genuine junk-only frame. The default 3-ball
    cadence's flights overlap too tightly for this to ever happen in
    practice (there's always a second ball aloft to make the frame
    ambiguous) -- exactly why the original `_persistent_junk_cascade`
    fixture can't exercise this defect and this one needs its own params.

    A synthetic idle tail is appended manually, well past
    `run_end + idle_gap_s` (anchored off the sim's own `run_end`, not off
    any arc, since the tail is constructed by hand and must not itself
    perturb arc extraction) -- junk only, no real or held content at all --
    representing genuine "the juggler put the balls down" dead time the fix
    must still mine.
    """
    fps = 30.0
    params = CascadeParams(n_balls=2, period_s=0.8, dwell_s=0.5)
    r = simulate_cascade(
        n_throws=n_throws, fps=fps, noise=noise, seed=sim_seed,
        include_held=True, params=params,
    )
    real = list(r.detections)
    n_frames = max(d.frame_idx for d in real) + 1
    rng = np.random.default_rng(junk_seed)

    def _junk_frame(i: int) -> list[Detection]:
        t = i / fps
        pts = []
        for _ in range(junk_density):
            x = 0.91 + float(rng.uniform(-junk_jitter, junk_jitter))
            y = 0.71 + float(rng.uniform(-junk_jitter, junk_jitter))
            pts.append(Detection(frame_idx=i, t=t, x=x, y=y))
        return pts

    junk = [d for i in range(n_frames) for d in _junk_frame(i)]

    tail_start = int(round((r.run_end + idle_gap_s) * fps))
    tail = [d for i in range(tail_start, tail_start + idle_tail_frames)
            for d in _junk_frame(i)]

    dets = real + junk + tail
    total_frames = tail_start + idle_tail_frames
    arcs = extract_arcs(dets)
    return r, dets, arcs, total_frames, fps, tail_start


def test_export_hard_negatives_excludes_active_windows_despite_held_balls(tmp_path):
    r, dets, arcs, n_frames, fps, tail_start = _persistent_junk_cascade_with_held()
    assert arcs, "sanity: extraction must actually verify some real throws"
    real_frames = {d.frame_idx for d in r.detections}

    # Sanity: this fixture really does reproduce the held-ball leak the OLD
    # all-unassigned + min_junk rule was vulnerable to -- with no temporal
    # awareness (pad=0.0), some "pure junk" candidate frames actually contain
    # a real held ball.
    naive_pure_junk, _, _ = _expected_candidates(dets, arcs, min_junk=2, pad=0.0)
    held_leak = naive_pure_junk & real_frames
    assert held_leak, "fixture must produce at least one held-ball leak frame"

    # The fix (default pad=1.0): every leaked frame gets caught by the
    # activity window and none survive into the final candidate set.
    pure_junk, ambiguous, active = _expected_candidates(dets, arcs, min_junk=2)
    assert held_leak <= active
    assert pure_junk.isdisjoint(real_frames)

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs, out, max_frames=len(pure_junk) + 5, min_junk=2, seed=0,
    )

    assert stats["n_candidate_frames"] == len(pure_junk)
    assert stats["n_skipped_ambiguous"] == len(ambiguous)
    assert stats["n_skipped_active"] == len(active)
    assert stats["n_images"] == len(pure_junk)

    coco = json.loads((out / "annotations.json").read_text())
    exported_frames = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert exported_frames.isdisjoint(real_frames), (
        "no exported negative frame may contain a real (held) ball"
    )

    manifest = json.loads((out / "negatives_manifest.json").read_text())
    manifest_frames = {f["frame_idx"] for f in manifest["frames"]}
    assert manifest_frames.isdisjoint(real_frames)


def test_export_hard_negatives_still_mines_genuinely_idle_tail(tmp_path):
    r, dets, arcs, n_frames, fps, tail_start = _persistent_junk_cascade_with_held()
    assert arcs, "sanity: extraction must actually verify some real throws"

    pure_junk, _, _ = _expected_candidates(dets, arcs, min_junk=2)
    tail_frames = {f for f in pure_junk if f >= tail_start}
    assert tail_frames, "the synthetic idle tail must survive as legitimate candidates"

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs, out, max_frames=len(pure_junk) + 5, min_junk=2, seed=0,
    )
    assert stats["n_images"] == len(pure_junk)

    coco = json.loads((out / "annotations.json").read_text())
    exported_frames = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert tail_frames <= exported_frames
