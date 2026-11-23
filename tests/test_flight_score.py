import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from juggletrack.types import Detection
from tests.helpers import cascade_throws, juggle_tracks, write_test_video

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "flight_score.py"
_SIZE = (320, 240)
_FPS = 30.0


def _flight_score(out, data, *args, cwd=None):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(out), "--root", str(data / "outputs"),
         "--csv-dir", str(data / "csv"), "--video-dir", str(data / "videos"), *map(str, args)],
        capture_output=True, text=True, cwd=cwd,
    )


def _save_clip(data, clip, *, labelled=True, clock_offset=0.0):
    """One clip in the Meschke layout: 24 clean cascade flights, labelled in whole pixels."""
    width, height = _SIZE
    tracks = juggle_tracks(cascade_throws(n_balls=3, per_ball=8))
    # One entry per frame, holding each ball's pixel position in that frame.
    frames = [[(round(d.x * width), round(d.y * height)) for d in balls] for balls in zip(*tracks)]

    detections = data / "outputs" / clip / "detections.jsonl"
    detections.parent.mkdir(parents=True)
    detections.write_text("".join(
        Detection(frame_idx=idx, t=idx / _FPS + clock_offset, x=px / width, y=py / height)
        .model_dump_json() + "\n"
        for idx, balls in enumerate(frames) for px, py in balls
    ))
    if labelled:
        (data / "csv").mkdir(exist_ok=True)
        (data / "videos").mkdir(exist_ok=True)
        rows = [",".join(f"{px},{py}" for px, py in balls) for balls in frames]
        (data / "csv" / f"{clip}.csv").write_text("\n".join(["0,1,2,3,4,5", *rows]) + "\n")
        write_test_video(data / "videos" / f"{clip}.MP4", n_frames=5, fps=_FPS, size=_SIZE)
    return detections


def _inputs(data):
    return {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}


def test_scores_a_clean_cascade_and_repeats_byte_for_byte(tmp_path):
    data = tmp_path / "data"
    _save_clip(data, "cascade")
    before = _inputs(data)
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    # Run outside any checkout: nothing may depend on the working directory.
    procs = [_flight_score(out, data, cwd=tmp_path) for out in (first, second)]

    assert [p.returncode for p in procs] == [0, 0], procs[0].stdout + procs[0].stderr
    assert first.read_bytes() == second.read_bytes()
    assert _inputs(data) == before
    report = json.loads(first.read_text())
    assert set(report["clips"]) == {"cascade"}
    clip = report["clips"]["cascade"]
    # The reference and the pipeline are separate code, and on clean footage
    # they agree on every one of the 24 throws.
    assert clip["trusted"] is True
    assert (clip["flights"], clip["catches"], clip["matched"]) == (24, 24, 24)
    assert (clip["duplicates"], clip["unmatched"], clip["missed"], clip["count_error"]) == (
        0, 0, 0, 0)
    assert (clip["explained"], clip["repeated"]) == (24, 0)
    # The pipeline's hand line is below the reference hand level here, so its
    # catches come a few frames after the reference landings.
    assert 0 < clip["offset_s"] < 0.1
    assert f"{clip['offset_s']:+.3f}" in procs[0].stdout
    assert (clip["fps"], clip["width"], clip["height"]) == (_FPS, *_SIZE)
    assert clip["frame_count"] == 5
    assert clip["video_sha256"] == hashlib.sha256(
        (data / "videos" / "cascade.MP4").read_bytes()).hexdigest()
    assert report["totals"]["all"] == report["totals"]["trusted"]
    assert report["totals"]["all"]["flights"] == 24
    for digest in (clip["detections_sha256"], clip["labels_sha256"],
                   report["meta"]["implementation_sha256"], report["meta"]["script_sha256"]):
        assert len(digest) == 64
    assert "cascade" in procs[0].stdout and "TRUSTED (1 clips)" in procs[0].stdout


@pytest.mark.parametrize("alias", ["same-path", "hardlink", "symlink", "named-detections"])
def test_refuses_to_overwrite_an_input(alias, tmp_path):
    data = tmp_path / "data"
    detections = _save_clip(data, "a")
    _save_clip(data, "b")
    before = _inputs(data)
    out = {
        "same-path": detections,
        "hardlink": tmp_path / "report.json",
        "symlink": tmp_path / "report.json",
        "named-detections": tmp_path / "elsewhere" / "detections.jsonl",
    }[alias]
    if alias == "hardlink":
        os.link(data / "csv" / "a.csv", out)
    elif alias == "symlink":
        out.symlink_to(data / "videos" / "a.MP4")

    # --only selects b, so a is outside the scored set but still protected.
    proc = _flight_score(out, data, "--only", "b")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "overwrite an input" in proc.stderr
    assert _inputs(data) == before
    assert not (tmp_path / "elsewhere").exists()


def test_exits_nonzero_when_no_clip_matches(tmp_path):
    data = tmp_path / "data"
    _save_clip(data, "labelled")
    _save_clip(data, "unlabelled", labelled=False)

    for only, message in (("nothing", "no clip"), ("unlabelled", "no Meschke labels")):
        out = tmp_path / f"{only}.json"
        proc = _flight_score(out, data, "--only", only)

        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert message in proc.stderr
        assert not out.exists()


def test_skips_a_clip_without_labels_and_scores_the_rest(tmp_path):
    data = tmp_path / "data"
    _save_clip(data, "labelled")
    _save_clip(data, "unlabelled", labelled=False)
    out = tmp_path / "report.json"

    proc = _flight_score(out, data)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "skip  unlabelled" in proc.stderr
    assert set(json.loads(out.read_text())["clips"]) == {"labelled"}


def test_refuses_detections_recorded_on_another_clock(tmp_path):
    data = tmp_path / "data"
    _save_clip(data, "late", clock_offset=0.5)
    out = tmp_path / "report.json"

    proc = _flight_score(out, data)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "clock" in proc.stderr
    assert not out.exists()


def test_live_scores_the_catches_a_realtime_replay_confirms(tmp_path):
    data = tmp_path / "data"
    _save_clip(data, "cascade")
    out = tmp_path / "live.json"

    proc = _flight_score(out, data, "--live")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert report["meta"]["mode"] == "live"
    clip = report["clips"]["cascade"]
    # Live mode confirms each of the 24 clean throws once, all in one run.
    assert (clip["flights"], clip["catches"], clip["matched"]) == (24, 24, 24)
    assert (clip["duplicates"], clip["unmatched"], clip["missed"]) == (0, 0, 0)
    assert (clip["runs"], clip["drops"]) == (1, 0)
    assert 0 < clip["offset_s"] < 0.1
    # A replay re-cuts its arcs every cycle, so no one arc set is scored.
    assert "explained" not in clip
    total = report["totals"]["all"]
    assert (total["catches"], total["runs"], total["drops"]) == (24, 1, 0)
    assert {"throws", "runs", "drops"} <= set(proc.stdout.splitlines()[0].split())


def test_controls_show_each_seeded_fault_in_its_own_column(tmp_path):
    data = tmp_path / "data"
    _save_clip(data, "cascade")
    out = tmp_path / "report.json"

    proc = _flight_score(out, data, "--controls")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    controls = report["clips"]["cascade"]["controls"]
    # 24 catches, so every tenth is three of them.
    deleted, doubled, shifted = (
        controls[k] for k in ("delete_10pct", "duplicate_10pct", "shift_2s"))
    assert (deleted["catches"], deleted["matched"], deleted["missed"]) == (21, 21, 3)
    assert (deleted["duplicates"], deleted["unmatched"]) == (0, 0)
    assert (doubled["catches"], doubled["matched"], doubled["duplicates"]) == (27, 24, 3)
    assert (doubled["unmatched"], doubled["missed"]) == (0, 0)
    # Two seconds is a third of a beat away from the next landing in the
    # same hand. The last catches move past the end of the labels.
    assert shifted["matched"] == 0 and shifted["missed"] == 24
    assert shifted["unmatched"] + shifted["ignored"] == 24 and shifted["unmatched"] >= 18
    assert report["totals"]["controls"]["shift_2s"]["matched"] == 0
