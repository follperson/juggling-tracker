import json
from pathlib import Path

import pytest

from juggletrack.data.meschke_import import (
    CITATION,
    DEFAULT_BOX,
    export_meschke_source,
    load_meschke_trajectories,
)
from tests.helpers import write_test_video


def _write_csv(path: Path, rows: list[tuple[int, ...]], header: str = "0,1,2,3") -> None:
    """Synthesize a Meschke-style CSV: junk/inconsistent header line (skipped
    unconditionally) + integer-pixel interleaved x,y columns per ball.
    """
    lines = [header] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def test_load_meschke_trajectories_normalizes_and_skips_sentinels(tmp_path):
    csv_path = tmp_path / "sample.csv"
    rows = []
    for i in range(20):
        b1 = (0, 0) if i == 5 else (100 + i, 200 + i)  # (0,0) sentinel at frame 5
        b2 = (-1, -1) if i == 10 else (300 + i, 50 + i)  # negative sentinel at frame 10
        rows.append((*b1, *b2))
    _write_csv(csv_path, rows)

    trajectories = load_meschke_trajectories(csv_path, fps=30.0, width=200, height=400)

    assert len(trajectories) == 2
    ball1, ball2 = trajectories
    assert len(ball1) == 19  # 20 frames minus the one sentinel
    assert len(ball2) == 19
    assert all(d.frame_idx != 5 for d in ball1)
    assert all(d.frame_idx != 10 for d in ball2)

    d0 = ball1[0]
    assert d0.frame_idx == 0
    assert d0.t == pytest.approx(0.0)
    assert d0.x == pytest.approx(100 / 200)
    assert d0.y == pytest.approx(200 / 400)
    assert d0.w == 0.0 and d0.h == 0.0
    assert d0.confidence == 1.0

    # frame_idx tracks the CSV row position, not the post-filter list index
    d_after_gap = next(d for d in ball1 if d.frame_idx == 6)
    assert d_after_gap.x == pytest.approx(106 / 200)

    d_ball2 = next(d for d in ball2 if d.frame_idx == 3)
    assert d_ball2.t == pytest.approx(3 / 30.0)
    assert d_ball2.x == pytest.approx(303 / 200)
    assert d_ball2.y == pytest.approx(53 / 400)


def test_export_meschke_source_writes_canonical_coco(tmp_path):
    csv_path = tmp_path / "sample.csv"
    n_frames = 20
    rows = [(100 + i, 200 + i, 300 + i, 50 + i) for i in range(n_frames)]
    _write_csv(csv_path, rows)
    video_path = tmp_path / "v.mp4"
    write_test_video(video_path, n_frames=n_frames, fps=30.0, size=(200, 400))

    out = tmp_path / "out"
    stats = export_meschke_source(
        csv_path, video_path, out, frame_stride=5, max_frames=10, box=0.1, seed=0
    )

    assert stats["fps"] == pytest.approx(30.0)
    assert stats["width"] == 200
    assert stats["height"] == 400
    assert stats["box_px"] == pytest.approx(0.1 * 200)
    # stride=5 over 20 frames -> candidates [0, 5, 10, 15]; all under max_frames=10
    assert stats["n_images"] == 4
    assert stats["n_boxes"] == 4 * 2  # 2 balls visible every chosen frame

    coco = json.loads((out / "annotations.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "ball"}]
    assert [im["id"] for im in coco["images"]] == list(range(1, len(coco["images"]) + 1))
    assert [a["id"] for a in coco["annotations"]] == list(range(1, len(coco["annotations"]) + 1))
    for im in coco["images"]:
        assert (out / "images" / im["file_name"]).exists()
        assert im["width"] == 200 and im["height"] == 400
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert w == pytest.approx(20.0) and h == pytest.approx(20.0)  # box=0.1 * width=200
        assert -1e-6 <= x <= 200 - w + 1e-6
        assert -1e-6 <= y <= 400 - h + 1e-6
        assert ann["area"] == pytest.approx(w * h)
        assert ann["category_id"] == 1 and ann["iscrowd"] == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["citation"] == CITATION


def test_export_meschke_source_respects_stride_max_frames_and_is_deterministic(tmp_path):
    csv_path = tmp_path / "sample.csv"
    n_frames = 60
    rows = [(100 + i, 200 + i, 300 + i, 50 + i) for i in range(n_frames)]
    _write_csv(csv_path, rows)
    video_path = tmp_path / "v.mp4"
    write_test_video(video_path, n_frames=n_frames, fps=30.0, size=(200, 400))

    # stride=2 over 60 frames -> 30 candidates, capped down to max_frames=5
    stats_a1 = export_meschke_source(
        csv_path, video_path, tmp_path / "a1", frame_stride=2, max_frames=5, seed=7
    )
    export_meschke_source(
        csv_path, video_path, tmp_path / "a2", frame_stride=2, max_frames=5, seed=7
    )
    assert stats_a1["n_images"] == 5
    coco_a1 = json.loads((tmp_path / "a1" / "annotations.json").read_text())
    coco_a2 = json.loads((tmp_path / "a2" / "annotations.json").read_text())
    assert [im["file_name"] for im in coco_a1["images"]] == [
        im["file_name"] for im in coco_a2["images"]
    ]

    export_meschke_source(
        csv_path, video_path, tmp_path / "b", frame_stride=2, max_frames=5, seed=99
    )
    coco_b = json.loads((tmp_path / "b" / "annotations.json").read_text())
    assert [im["file_name"] for im in coco_a1["images"]] != [
        im["file_name"] for im in coco_b["images"]
    ]


def test_export_meschke_source_default_box_is_module_constant(tmp_path):
    csv_path = tmp_path / "sample.csv"
    n_frames = 10
    rows = [(100 + i, 200 + i, 300 + i, 50 + i) for i in range(n_frames)]
    _write_csv(csv_path, rows)
    video_path = tmp_path / "v.mp4"
    write_test_video(video_path, n_frames=n_frames, fps=30.0, size=(200, 400))

    stats = export_meschke_source(csv_path, video_path, tmp_path / "out", frame_stride=5, max_frames=10)
    assert stats["box_px"] == pytest.approx(DEFAULT_BOX * 200)


def test_export_meschke_source_tolerates_small_row_frame_mismatch(tmp_path):
    n_frames = 20
    # 2 extra CSV rows beyond the video's frame count: within the +/-2 tolerance
    rows = [(100 + i, 200 + i, 300 + i, 50 + i) for i in range(n_frames + 2)]
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path, rows)
    video_path = tmp_path / "v.mp4"
    write_test_video(video_path, n_frames=n_frames, fps=30.0, size=(200, 400))

    stats = export_meschke_source(csv_path, video_path, tmp_path / "out", frame_stride=5, max_frames=10)
    assert stats["n_images"] == 4  # truncated to the video's 20 frames, same as the exact-match case


def test_export_meschke_source_raises_on_large_row_frame_mismatch(tmp_path):
    n_frames = 20
    # 5 extra CSV rows: outside the +/-2 tolerance
    rows = [(100 + i, 200 + i, 300 + i, 50 + i) for i in range(n_frames + 5)]
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path, rows)
    video_path = tmp_path / "v.mp4"
    write_test_video(video_path, n_frames=n_frames, fps=30.0, size=(200, 400))

    with pytest.raises(ValueError, match="row count"):
        export_meschke_source(csv_path, video_path, tmp_path / "out", frame_stride=5, max_frames=10)
