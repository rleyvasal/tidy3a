"""The §7 example on real nuScenes Mini data; skipped unless NUSCENES_DATAROOT is set.

    NUSCENES_DATAROOT=/path/to/nuscenes-mini python -m pytest tests/test_nuscenes.py
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("NUSCENES_DATAROOT")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "v1.0-mini").is_dir(),
    reason="set NUSCENES_DATAROOT to a nuScenes Mini directory to run",
)


def test_nuscenes_sweep_matches_numpy(capsys):
    spec = importlib.util.spec_from_file_location(
        "nuscenes_sweep", Path(__file__).parents[1] / "examples" / "nuscenes_sweep.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.main(Path(ROOT))
    output = capsys.readouterr().out
    assert "FAIL" not in output
    assert "ok   kept positions" in output and "ok   distances" in output
