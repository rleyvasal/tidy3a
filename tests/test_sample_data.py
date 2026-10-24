"""tidy3A on public sample data; skipped unless TIDY3A_SAMPLES is set.

    TIDY3A_SAMPLES=/path/to/samples python -m pytest tests/test_sample_data.py

The directory holds air_temperature.nc and eraint_uvz.nc (xarray tutorial
data) and L7_ETMs.tif (the R stars package); see examples/sample_data.py.
Reading them needs scipy and rioxarray.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("TIDY3A_SAMPLES")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "air_temperature.nc").exists(),
    reason="set TIDY3A_SAMPLES to the sample-data directory to run",
)


def test_sample_data_matches_xarray(capsys):
    pytest.importorskip("scipy")
    pytest.importorskip("rioxarray")
    spec = importlib.util.spec_from_file_location(
        "sample_data", Path(__file__).parents[1] / "examples" / "sample_data.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.main(Path(ROOT))
    output = capsys.readouterr().out
    assert "FAIL" not in output and "17 of 17 checks passed" in output
