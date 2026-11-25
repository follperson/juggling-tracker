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


def _parity(*args, cwd=None):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *map(str, args)],
        capture_output=True, text=True, cwd=cwd,
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

    # Run outside any git checkout: git_head must still name the checkout
    # that holds the imported package.
    proc = _parity("snapshot", out, "--root", root, "--realtime", "a/", cwd=tmp_path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    snap = json.loads(out.read_text())
    assert set(snap["offline"]) == set(snap["realtime"]) == {"a/detections.jsonl"}
    assert a.read_bytes() == before
    package_head = subprocess.run(
        ["git", "-C", snap["meta"]["package"], "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert snap["meta"]["git_head"] == (package_head or "unknown")


@pytest.mark.parametrize("root_suffix, realtime", [
    ("", "nomatch/"),
    # Narrowing --root drops the directory the substring names.
    ("val", "val/"),
])
def test_snapshot_fails_when_a_realtime_selection_matches_nothing(root_suffix, realtime, tmp_path):
    root = tmp_path / "outputs"
    _save_session(root / "val" / "a" / "detections.jsonl", 1)
    out = tmp_path / "snap.json"

    proc = _parity("snapshot", out, "--root", root / root_suffix, "--realtime", "a/",
                   "--realtime", realtime)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert f"--realtime {realtime} matched no session" in proc.stderr
    assert "--realtime a/" not in proc.stderr
    assert not out.exists()


@pytest.mark.parametrize("args, message", [
    # Every session the value selects is over --max-detections.
    (["--max-detections", "1", "--realtime", "a/"], "--realtime a/ replayed no session"),
    # The value selects only an empty detections file.
    (["--realtime", "empty/"], "--realtime empty/ replayed no session"),
    (["--only", "outputs/a/", "--max-detections", "1"], "no session analysed"),
], ids=["skipped", "empty", "no-offline"])
def test_snapshot_fails_when_filters_leave_nothing_to_compare(args, message, tmp_path):
    root = tmp_path / "outputs"
    _save_session(root / "a" / "detections.jsonl", 1)
    (root / "empty").mkdir()
    (root / "empty" / "detections.jsonl").write_text("")
    out = tmp_path / "snap.json"

    proc = _parity("snapshot", out, "--root", root, *args)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert message in proc.stderr
    assert not out.exists()


_OFFLINE_ROW = {"input_sha256": "i", "digest": "d", "seconds": 1.0, "arcs": 1, "runs": 1,
                "catches": 1, "drops": 0}
_REALTIME_ROW = {"digest": "r", "feed_seconds": 1.0, "analysis_ms_p50": 1.0,
                 "analysis_ms_p95": 1.0, "final": [1, 14, 0]}


def _snapshot_file(path, implementation, offline=None, realtime=None):
    meta = {"implementation_sha256": implementation, "python": "3", "numpy": "2",
            "platform": "p"}
    offline = {"s": _OFFLINE_ROW} if offline is None else offline
    path.write_text(json.dumps({"meta": meta, "offline": offline, "realtime": realtime or {}}))
    return path


@pytest.mark.parametrize("same", [True, False])
def test_compare_says_when_both_snapshots_ran_the_same_implementation(same, tmp_path):
    base = _snapshot_file(tmp_path / "base.json", "a" * 64)
    new = _snapshot_file(tmp_path / "new.json", "a" * 64 if same else "b" * 64)

    proc = _parity("compare", base, new)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PARITY OK" in proc.stdout
    assert ("same implementation" in proc.stdout) == same


@pytest.mark.parametrize("offline, realtime, line", [
    ({"s": {**_OFFLINE_ROW, "digest": "e", "catches": 2}}, None,
     "offline: MISMATCH s {'catches': (1, 2)}"),
    ({"s": {**_OFFLINE_ROW, "input_sha256": "j"}}, None, "offline: INPUT CHANGED s"),
    ({"s": _OFFLINE_ROW, "t": _OFFLINE_ROW}, None,
     "offline: present in only one snapshot: t"),
    (None, {"s": {**_REALTIME_ROW, "digest": "q", "final": [1, 21, 0]}},
     "realtime: MISMATCH s {'final': ([1, 14, 0], [1, 21, 0])}"),
], ids=["digest", "input", "missing-session", "realtime-only"])
def test_compare_fails_on_any_difference(offline, realtime, line, tmp_path):
    base = _snapshot_file(tmp_path / "base.json", "a" * 64, realtime={"s": _REALTIME_ROW})
    new = _snapshot_file(tmp_path / "new.json", "b" * 64, offline=offline,
                         realtime=realtime or {"s": _REALTIME_ROW})

    proc = _parity("compare", base, new)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert line in proc.stdout.splitlines()
    assert "PARITY FAILED (1)" in proc.stdout


@pytest.mark.parametrize("base_offline, new_offline", [
    ({}, {}),
    ({"s": _OFFLINE_ROW}, {"t": _OFFLINE_ROW}),
], ids=["both-empty", "disjoint"])
def test_compare_fails_when_the_snapshots_share_no_session(base_offline, new_offline, tmp_path):
    base = _snapshot_file(tmp_path / "base.json", "a" * 64, offline=base_offline)
    new = _snapshot_file(tmp_path / "new.json", "b" * 64, offline=new_offline)

    proc = _parity("compare", base, new)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "PARITY OK" not in proc.stdout
    assert "nothing was compared" in proc.stderr


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
