"""
Tests for py/training_status.py

Run with:
    python3 -m pytest test/test_training_status.py -v
"""

import json
import os
import stat
import sys
import time
import threading
import tempfile
import pathlib

import pytest

# Ensure py/ is on the path for direct import
_HERE = pathlib.Path(__file__).parent
_PY = _HERE.parent / "py"
sys.path.insert(0, str(_PY))

from training_status import read_status, write_status


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tmp_path(tmp_dir: str, name: str = "status.json") -> str:
    return os.path.join(tmp_dir, name)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestWriteStatus:
    def test_creates_file_with_correct_fields(self, tmp_path):
        path = str(tmp_path / "status.json")
        write_status(
            path,
            tier="shade",
            phase="training",
            epoch=42,
            epochs_total=200,
            train_loss=1.234,
            val_loss=1.189,
            best_val_loss=1.102,
            best_epoch=38,
            stopped_early=False,
            state="running",
            checkpoint="ckpt/shade_fp32.pt",
            eta_seconds=3600,
        )

        assert os.path.exists(path), "status.json was not created"
        with open(path) as f:
            data = json.load(f)

        assert data["tier"] == "shade"
        assert data["phase"] == "training"
        assert data["epoch"] == 42
        assert data["epochs_total"] == 200
        assert abs(data["train_loss"] - 1.234) < 1e-6
        assert abs(data["val_loss"] - 1.189) < 1e-6
        assert abs(data["best_val_loss"] - 1.102) < 1e-6
        assert data["best_epoch"] == 38
        assert data["stopped_early"] is False
        assert data["state"] == "running"
        assert data["checkpoint"] == "ckpt/shade_fp32.pt"
        assert data["eta_seconds"] == 3600
        assert "updated_at" in data
        assert "started_at" in data
        assert "pid" in data

    def test_atomic_write(self, tmp_path):
        """The file should never be partially written (no .tmp residue after write)."""
        path = str(tmp_path / "status.json")
        tmp_path_file = path + ".tmp"

        write_status(path, state="running", epoch=1)

        # The .tmp file must be gone after write
        assert not os.path.exists(tmp_path_file), ".tmp file left behind after write"
        # The real file must exist and be valid JSON
        with open(path) as f:
            data = json.load(f)
        assert data["state"] == "running"

    def test_atomic_write_no_partial_reads(self, tmp_path):
        """Concurrent reader should never see a half-written file."""
        path = str(tmp_path / "status.json")
        errors = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    with open(path) as f:
                        content = f.read()
                    if content.strip():
                        json.loads(content)  # must parse cleanly
                except FileNotFoundError:
                    pass
                except json.JSONDecodeError as e:
                    errors.append(str(e))

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        for i in range(50):
            write_status(path, epoch=i, state="running",
                         train_loss=float(i), val_loss=float(i) * 0.9)

        stop.set()
        t.join(timeout=2)

        assert not errors, f"Reader saw corrupt JSON during concurrent writes: {errors}"

    def test_merges_existing_fields(self, tmp_path):
        """write_status should preserve fields not in the update."""
        path = str(tmp_path / "status.json")

        # First write: establish tier + checkpoint
        write_status(path, tier="wisp", checkpoint="ckpt/wisp.pt", state="running")

        # Second write: update epoch only
        write_status(path, epoch=10)

        data = read_status(path)
        assert data is not None
        # Original fields must survive
        assert data["tier"] == "wisp"
        assert data["checkpoint"] == "ckpt/wisp.pt"
        assert data["state"] == "running"
        # New field must be present
        assert data["epoch"] == 10

    def test_merge_overwrites_changed_fields(self, tmp_path):
        """A second write should overwrite fields that changed."""
        path = str(tmp_path / "status.json")

        write_status(path, state="running", epoch=5)
        write_status(path, state="done", epoch=30)

        data = read_status(path)
        assert data["state"] == "done"
        assert data["epoch"] == 30

    def test_updated_at_advances(self, tmp_path):
        """updated_at should advance on each write (within reason)."""
        path = str(tmp_path / "status.json")

        write_status(path, state="running", epoch=1)
        d1 = read_status(path)
        t1 = d1["updated_at"]

        time.sleep(1.1)  # ensure clock advances at least 1 second
        write_status(path, epoch=2)
        d2 = read_status(path)
        t2 = d2["updated_at"]

        assert t2 >= t1, "updated_at did not advance"

    def test_creates_parent_dirs(self, tmp_path):
        """write_status should create missing parent directories."""
        path = str(tmp_path / "deep" / "nested" / "status.json")
        write_status(path, state="running")
        assert os.path.exists(path)

    def test_started_at_set_only_once(self, tmp_path):
        """started_at should be set on first write and not change on subsequent writes."""
        path = str(tmp_path / "status.json")

        write_status(path, state="running")
        d1 = read_status(path)
        started = d1["started_at"]

        time.sleep(1.1)
        write_status(path, epoch=5)
        d2 = read_status(path)

        assert d2["started_at"] == started, "started_at changed on second write"


class TestReadStatus:
    def test_returns_none_for_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        result = read_status(path)
        assert result is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("this is not json {{{")
        result = read_status(path)
        assert result is None

    def test_round_trip(self, tmp_path):
        """write then read should return the same data."""
        path = str(tmp_path / "status.json")
        write_status(
            path,
            tier="specter",
            phase="training",
            epoch=7,
            epochs_total=15,
            train_loss=0.912,
            val_loss=0.944,
            best_val_loss=0.900,
            best_epoch=5,
            stopped_early=False,
            state="running",
            checkpoint="ckpt/specter_fp32.pt",
            eta_seconds=7200,
        )

        data = read_status(path)
        assert data is not None
        assert data["tier"] == "specter"
        assert data["epoch"] == 7
        assert data["epochs_total"] == 15
        assert abs(data["train_loss"] - 0.912) < 1e-5
        assert abs(data["val_loss"] - 0.944) < 1e-5
        assert abs(data["best_val_loss"] - 0.900) < 1e-5
        assert data["best_epoch"] == 5
        assert data["stopped_early"] is False
        assert data["state"] == "running"
        assert data["checkpoint"] == "ckpt/specter_fp32.pt"
        assert data["eta_seconds"] == 7200

    def test_read_returns_dict(self, tmp_path):
        path = str(tmp_path / "status.json")
        write_status(path, state="done")
        result = read_status(path)
        assert isinstance(result, dict)


class TestWatchScript:
    """Smoke-tests for watch_training.sh — check it exists and is executable."""

    def test_script_exists(self):
        script = _HERE.parent / "scripts" / "watch_training.sh"
        assert script.exists(), f"watch_training.sh not found at {script}"

    def test_script_is_executable(self):
        script = _HERE.parent / "scripts" / "watch_training.sh"
        mode = os.stat(script).st_mode
        assert mode & stat.S_IXUSR, "watch_training.sh is not user-executable"

    def test_script_has_bash_shebang(self):
        script = _HERE.parent / "scripts" / "watch_training.sh"
        with open(script) as f:
            first_line = f.readline().strip()
        assert first_line in ("#!/bin/bash", "#!/usr/bin/env bash"), (
            f"unexpected shebang: {first_line!r}"
        )
