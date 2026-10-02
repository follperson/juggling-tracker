import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from juggletrack.sim import simulate_cascade

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
