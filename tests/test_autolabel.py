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


def _idle_hold_cascade(
    *, junk_seed: int = 11, junk_density: int = 2, junk_jitter: float = 0.005,
    tail_frames: int = 30, hold_frames: int = 18,
):
    """The turn-4 field-leak shape (PXL_20260716_191622045 frame 87): a juggler
    HOLDING real balls during an idle stretch farther than `pad` from any arc.

    Layout on one timeline (fps 30): a 2-ball held cascade (same params as
    `_persistent_junk_cascade_with_held`, so hands genuinely dwell), persistent
    junk at (0.91, 0.71) on every content frame, a junk-only idle tail 2 s
    after `run_end`, then -- after a further 1 s gap -- an idle HOLD segment
    where each frame has the junk plus two ball detections sitting in the
    hands at (0.41, 0.65) / (0.59, 0.65).

    The hold segment is the leak: its frames are outside every activity
    window, all detections on them are arc-unassigned, and the held-ball
    clusters are even *persistent* (the same hand spots accumulate held
    detections throughout the run), so the temporal and persistence gates
    both pass them by design -- only the person-region veto can reject them.

    Returns (dets, arcs, total_frames, fps, tail_set, hold_set, person_box)
    where person_box is a normalized xyxy box covering the juggler (hands
    included, junk excluded).
    """
    fps = 30.0
    params = CascadeParams(n_balls=2, period_s=0.8, dwell_s=0.5)
    r = simulate_cascade(
        n_throws=8, fps=fps, noise=0.002, seed=3, include_held=True, params=params,
    )
    real = list(r.detections)
    n_frames = max(d.frame_idx for d in real) + 1
    rng = np.random.default_rng(junk_seed)

    def _junk_frame(i: int) -> list[Detection]:
        t = i / fps
        return [Detection(
            frame_idx=i, t=t,
            x=0.91 + float(rng.uniform(-junk_jitter, junk_jitter)),
            y=0.71 + float(rng.uniform(-junk_jitter, junk_jitter)),
        ) for _ in range(junk_density)]

    junk = [d for i in range(n_frames) for d in _junk_frame(i)]

    tail_start = int(round((r.run_end + 2.0) * fps))
    tail_set = set(range(tail_start, tail_start + tail_frames))
    tail = [d for i in sorted(tail_set) for d in _junk_frame(i)]

    hold_start = tail_start + tail_frames + int(fps)  # 1 s gap after the tail
    hold_set = set(range(hold_start, hold_start + hold_frames))
    hold = []
    for i in sorted(hold_set):
        t = i / fps
        hold.extend(_junk_frame(i))
        for hand_x in (params.hand_x(0), params.hand_x(1)):
            hold.append(Detection(
                frame_idx=i, t=t,
                x=hand_x + float(rng.uniform(-0.003, 0.003)),
                y=params.hand_y + float(rng.uniform(-0.003, 0.003)),
            ))

    dets = real + junk + tail + hold
    arcs = extract_arcs(dets)
    total_frames = hold_start + hold_frames
    person_box = (0.35, 0.20, 0.65, 0.95)
    return dets, arcs, total_frames, fps, tail_set, hold_set, person_box


def test_export_hard_negatives_person_veto_rejects_idle_held_balls(tmp_path):
    dets, arcs, n_frames, fps, tail_set, hold_set, person_box = _idle_hold_cascade()
    assert arcs, "sanity: extraction must actually verify some real throws"

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    # Person PRESENCE is the veto, not box-overlap with junk: the final-audit
    # field case (183035587 frame 221) had the carried balls produce NO
    # detection at all (motion blur), so detection-space reasoning is blind
    # to them -- the person box is the only visible evidence. And presence
    # DILATES +-person_dilate_frames: the same field case showed the person
    # detector itself missing the blurred entry frame (conf 0.246 vs even a
    # lowered threshold) while nailing its neighbors, and a person cannot
    # teleport. The hold frames all get boxes, plus ONE tail frame whose junk
    # is far outside the box: that frame AND its dilation neighborhood must
    # be rejected anyway.
    person_tail_frame = sorted(tail_set)[0]
    person_boxes = {f: [person_box] for f in sorted(hold_set) + [person_tail_frame]}
    stats = export_hard_negatives(
        video, dets, arcs, out, max_frames=100, min_junk=2, seed=0,
        person_boxes=person_boxes, person_dilate_frames=5,
    )

    vetoed_tail = {f for f in tail_set if abs(f - person_tail_frame) <= 5}
    assert len(vetoed_tail) == 6  # the box frame plus 5 dilation neighbors
    assert stats["n_skipped_person"] == len(hold_set) + len(vetoed_tail)
    assert stats["n_skipped_transient"] == 0
    assert stats["n_skipped_floor"] == 0
    assert stats["n_candidate_frames"] == len(tail_set) - len(vetoed_tail)
    assert stats["n_images"] == len(tail_set) - len(vetoed_tail)

    coco = json.loads((out / "annotations.json").read_text())
    exported = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert exported == tail_set - vetoed_tail
    assert exported.isdisjoint(hold_set)


def test_export_hard_negatives_rejects_transient_unstitched_flight(tmp_path):
    """Field leak #2 (frames 108/630): a real ball in flight the linker never
    stitched into an arc. Modeled as a 5-point ballistic fragment (below
    extract_arcs' min_points=6) crossing the first 5 idle-tail frames: fast
    per-frame motion means no persistent static cluster, so the frames it
    touches must be rejected as transient."""
    dets, arcs, n_frames, fps, tail_set, hold_set, person_box = _idle_hold_cascade(
        hold_frames=0,
    )
    assert arcs
    ghost_set = set(sorted(tail_set)[:5])
    ghost = [Detection(
        frame_idx=f, t=f / fps, x=0.20 + 0.03 * i, y=0.50 - 0.08 * i,
    ) for i, f in enumerate(sorted(ghost_set))]
    dets = dets + ghost
    arcs2 = extract_arcs(dets)
    assert len(arcs2) == len(arcs), "sanity: the 5-point ghost must not fit an arc"

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs2, out, max_frames=100, min_junk=2, seed=0,
    )

    assert stats["n_skipped_transient"] == len(ghost_set)
    assert stats["n_candidate_frames"] == len(tail_set) - len(ghost_set)

    coco = json.loads((out / "annotations.json").read_text())
    exported = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert exported == tail_set - ghost_set

    # trusted persistent clusters are recorded for future audits
    manifest = json.loads((out / "negatives_manifest.json").read_text())
    clusters = manifest["trusted_clusters"]
    assert any(
        abs(c["x"] - 0.91) < 0.02 and abs(c["y"] - 0.71) < 0.02 and c["n"] >= 8
        for c in clusters
    )
    for c in clusters:
        assert c["t_min"] <= c["t_max"]


def test_export_hard_negatives_rejects_slow_drift_through_junk_neighborhood(tmp_path):
    """Adversarial-review reproduction: a slow-moving arc-unassigned ball whose
    path passes within `persist_radius` of a trusted junk cluster must NOT
    inherit that cluster's trust. Greedy cluster membership alone would accept
    it (running centroid within 0.04); trust must be per-detection -- junk
    evidence recurring in a tight neighborhood of the detection's OWN position
    across a long span. A ball momentarily colocated with the junk core is the
    irreducible residual and is out of scope here: this drifter stays >= 0.02
    from the junk's own footprint at all times."""
    dets, arcs, n_frames, fps, tail_set, hold_set, person_box = _idle_hold_cascade(
        hold_frames=0,
    )
    assert arcs
    drift_set = set(sorted(tail_set)[:8])
    drift = [Detection(
        frame_idx=f, t=f / fps, x=0.870 + 0.002 * i, y=0.700,
    ) for i, f in enumerate(sorted(drift_set))]
    dets = dets + drift
    arcs2 = extract_arcs(dets)
    assert len(arcs2) == len(arcs), "sanity: the slow drifter must not fit an arc"

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs2, out, max_frames=100, min_junk=2, seed=0,
    )

    assert stats["n_skipped_transient"] == len(drift_set)
    assert stats["n_candidate_frames"] == len(tail_set) - len(drift_set)
    coco = json.loads((out / "annotations.json").read_text())
    exported = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert exported == tail_set - drift_set


def test_export_hard_negatives_floor_band_rejects_resting_ball(tmp_path):
    """A ball resting on the floor is static and persistent -- geometrically
    indistinguishable from background junk (turn-4 audit: clusters at y~0.96
    spanning 82-100% of the video). The floor band must reject it even though
    the persistence gate trusts it."""
    dets, arcs, n_frames, fps, tail_set, hold_set, person_box = _idle_hold_cascade(
        hold_frames=0,
    )
    assert arcs
    rng = np.random.default_rng(5)
    floor_set = set(sorted(tail_set)[:15])
    max_content = max(d.frame_idx for d in dets if d.frame_idx not in tail_set)
    floor_frames = sorted(set(range(max_content + 1)) | floor_set)
    floor = [Detection(
        frame_idx=f, t=f / fps,
        x=0.305 + float(rng.uniform(-0.003, 0.003)),
        y=0.952 + float(rng.uniform(-0.003, 0.003)),
    ) for f in floor_frames]
    dets = dets + floor
    arcs2 = extract_arcs(dets)

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs2, out, max_frames=100, min_junk=2, seed=0,
    )

    assert stats["n_skipped_floor"] == len(floor_set)
    assert stats["n_skipped_transient"] == 0
    assert stats["n_candidate_frames"] == len(tail_set) - len(floor_set)

    coco = json.loads((out / "annotations.json").read_text())
    exported = {int(im["file_name"][6:12]) for im in coco["images"]}
    assert exported == tail_set - floor_set


def test_export_hard_negatives_no_arcs_exports_nothing(tmp_path):
    """Zero extracted arcs = zero physics evidence the pipeline understood the
    video -- the maximum-contamination case (e.g. juggling footage where the
    linker failed everywhere). Nothing may be exported."""
    fps = 30.0
    rng = np.random.default_rng(11)
    dets = [Detection(
        frame_idx=i, t=i / fps,
        x=0.91 + float(rng.uniform(-0.005, 0.005)),
        y=0.71 + float(rng.uniform(-0.005, 0.005)),
    ) for i in range(60) for _ in range(2)]
    arcs = extract_arcs(dets)
    assert arcs == [], "sanity: static junk alone must not extract arcs"

    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=60, fps=fps)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(video, dets, arcs, out, max_frames=40, seed=0)

    assert stats["n_images"] == 0
    assert stats["n_candidate_frames"] == 0
    assert stats["n_skipped_no_arcs"] == 60
    coco = json.loads((out / "annotations.json").read_text())
    assert coco["images"] == [] and coco["annotations"] == []


def test_export_hard_negatives_person_model_wires_detected_boxes(tmp_path, monkeypatch):
    """`person_model=` computes boxes via detect_person_boxes on exactly the
    frames that survived the pure gates, then applies the same veto as
    injected `person_boxes`."""
    import juggletrack.data.autolabel as autolabel_mod

    dets, arcs, n_frames, fps, tail_set, hold_set, person_box = _idle_hold_cascade()
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=fps)

    seen: dict = {}

    def fake_detect(video_path, frame_indices, *, model, conf=0.25):
        seen["frames"] = sorted(frame_indices)
        seen["model"] = model
        return {f: [person_box] for f in frame_indices if f in hold_set}

    monkeypatch.setattr(autolabel_mod, "detect_person_boxes", fake_detect)
    out = tmp_path / "negatives"
    stats = export_hard_negatives(
        video, dets, arcs, out, max_frames=100, min_junk=2, seed=0,
        person_model="stub.pt",
    )

    assert seen["model"] == "stub.pt"
    # the query set is the candidates dilated by +-person_dilate_frames, so
    # boxes found on a blurred candidate's neighbors can veto it
    expected_query = sorted({
        f + off for f in (tail_set | hold_set) for off in range(-5, 6) if f + off >= 0
    })
    assert seen["frames"] == expected_query
    assert stats["n_skipped_person"] == len(hold_set)
    assert stats["n_images"] == len(tail_set)


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
