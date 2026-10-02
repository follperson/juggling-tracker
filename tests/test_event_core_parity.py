import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from juggletrack.pipeline.realtime import RealtimeAnalyzer
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "event_core_parity.py"


def _parity(*args):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *map(str, args)], capture_output=True, text=True,
    )


def _save_session(path, seed):
    path.parent.mkdir(parents=True)
    dets = simulate_cascade(n_throws=3, fps=30.0, seed=seed).detections
    path.write_text("".join(d.model_dump_json() + "\n" for d in dets))


@pytest.mark.parametrize("alias", ["same-path", "hardlink", "symlink", "named-detections"])
def test_snapshot_refuses_to_overwrite_saved_detections(alias, tmp_path):
    root = tmp_path / "outputs"
    a, b = root / "a" / "detections.jsonl", root / "b" / "detections.jsonl"
    _save_session(a, 1)
    _save_session(b, 2)
    before = a.read_bytes()
    out = {
        "same-path": a,
        "hardlink": tmp_path / "snap.json",
        "symlink": tmp_path / "snap.json",
        "named-detections": tmp_path / "elsewhere" / "detections.jsonl",
    }[alias]
    if alias == "hardlink":
        os.link(a, out)
    elif alias == "symlink":
        out.symlink_to(a)

    # --only selects b, so a is outside the analysed set but still protected.
    proc = _parity("snapshot", out, "--root", root, "--only", "b/")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "overwrite saved detections" in proc.stderr
    assert a.read_bytes() == before
    assert not (tmp_path / "elsewhere").exists()


def test_snapshot_writes_a_fresh_path_and_leaves_inputs_alone(tmp_path):
    root = tmp_path / "outputs"
    a = root / "a" / "detections.jsonl"
    _save_session(a, 1)
    before = a.read_bytes()
    out = tmp_path / "snap.json"

    proc = _parity("snapshot", out, "--root", root, "--realtime", "a/")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    snap = json.loads(out.read_text())
    assert set(snap["offline"]) == set(snap["realtime"]) == {"a/detections.jsonl"}
    assert a.read_bytes() == before


def _load_script():
    spec = importlib.util.spec_from_file_location("event_core_parity", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_realtime_replay_feeds_the_recorded_clock(tmp_path):
    """A variable-rate session must replay on the timestamps live mode
    stamped on each frame, not on a clock rebuilt from an average fps."""
    rng = np.random.default_rng(5)
    base = simulate_cascade(n_throws=8, fps=30.0, include_held=True, seed=5).detections
    first = min(d.frame_idx for d in base)
    n_frames = max(d.frame_idx for d in base) - first + 1
    clock = np.concatenate([[0.0], np.cumsum(rng.uniform(1 / 45, 1 / 20, n_frames - 1))])
    dets = [
        d.model_copy(update={"frame_idx": d.frame_idx - first,
                             "t": float(clock[d.frame_idx - first])})
        for d in base
    ]
    assert {d.frame_idx for d in dets} == set(range(n_frames))
    path = tmp_path / "outputs" / "vfr" / "detections.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(d.model_dump_json() + "\n" for d in dets))

    analyzer = RealtimeAnalyzer()
    digest = hashlib.sha256()
    for idx in range(n_frames):
        state = analyzer.feed([d for d in dets if d.frame_idx == idx], float(clock[idx]))
        digest.update(state.model_dump_json(exclude={"last_analysis_ms"}).encode())
    digest.update(analyzer.finalize().model_dump_json(exclude={"last_analysis_ms"}).encode())

    out = tmp_path / "snap.json"
    proc = _parity("snapshot", out, "--root", tmp_path / "outputs", "--realtime", "vfr/")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text())["realtime"]["vfr/detections.jsonl"]["digest"] == (
        digest.hexdigest())


def test_replay_clock_fills_empty_frames_from_detected_neighbours():
    parity = _load_script()
    by_frame = {
        f: [Detection(frame_idx=f, t=t, x=0.5, y=0.5)]
        for f, t in [(3, 1.0), (4, 1.1), (6, 1.3), (10, 1.5)]
    }

    clock = parity._replay_clock(by_frame)

    assert [clock[f] for f in (3, 4, 6, 10)] == [1.0, 1.1, 1.3, 1.5]
    assert clock[5] == pytest.approx(1.2)
    assert clock[7:10] == pytest.approx([1.35, 1.4, 1.45])
    # Leading frames step back at the median interval, 0.1 s.
    assert clock[:3] == pytest.approx([0.7, 0.8, 0.9])
    assert len(clock) == 11


def test_replay_clock_accepts_a_lone_detected_frame():
    parity = _load_script()

    assert parity._replay_clock({0: [Detection(frame_idx=0, t=0.0, x=0.5, y=0.5)]}) == [0.0]
    assert parity._replay_clock({2: [Detection(frame_idx=2, t=0.0, x=0.5, y=0.5)]}) == (
        [0.0, 0.0, 0.0])
