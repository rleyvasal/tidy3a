"""table3's hook: ``table3.view(frame)`` and ``table3.show(frame)`` on a TidyArray.

``__table3__(n)`` gives the first *n* rows of ``as_table()`` (``long=True``
for more than one observation axis), the table's row count, and a caption,
converting only the positions those rows need.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a import as_table, col, filter, group_by, infer, read_array, tidy

pl = pytest.importorskip("polars")


@pytest.fixture
def cloud():
    points = np.arange(150, dtype="float32").reshape(50, 3)
    return read_array(points, dims={"point": infer, "feature": 3},
                      labels={"feature": ["x", "y", "z"]}, field_axis="feature")


@pytest.fixture
def grid():
    temp = xr.DataArray(np.arange(12.0).reshape(4, 3), dims=("time", "site"),
                        coords={"time": [10, 20, 30, 40], "site": ["a", "b", "c"]}, name="temp")
    return tidy(temp)


def test_first_rows_count_and_caption(cloud):
    rows, total, caption = cloud.__table3__(10)
    assert rows.equals((cloud >> as_table()).head(10))
    assert total == 50 and caption == "TidyArray (table)"


def test_after_a_filter_labels_stay_original_positions(cloud):
    kept = cloud >> filter(col("x") > 60)
    rows, total, _ = kept.__table3__(3)
    assert total == 29
    assert rows.equals((kept >> as_table()).head(3))
    assert rows["point"].to_list() == [22, 23, 24]  # positions count from 1


def test_more_than_one_axis_is_the_long_table(grid):
    rows, total, _ = grid.__table3__(5)
    assert total == 12
    assert rows.equals((grid >> as_table(long=True)).head(5))
    assert rows["time"].to_list() == [10, 10, 10, 20, 20]


def test_n_beyond_the_rows_and_zero(cloud, grid):
    rows, total, _ = cloud.__table3__(1000)
    assert rows.height == 50 and total == 50
    assert cloud.__table3__(0)[0].height == 0
    assert grid.__table3__(100)[0].height == 12


def test_groups_do_not_stop_it(grid):
    rows, total, _ = (grid >> group_by("site")).__table3__(4)
    assert rows.height == 4 and total == 12


def test_table3_shows_it(cloud, monkeypatch):
    table3 = pytest.importorskip("table3")
    import IPython
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj.data))
    assert table3.view(cloud, n=20) is cloud
    table3.show(cloud, n=20)
    explorer, paged = shown
    assert "table3-explore-" in explorer and ">TidyArray (table)<" in explorer
    assert "data-label='20 rows (first 20 of 50)'" in paged
