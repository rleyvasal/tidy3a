# tidy3A

Tidy-style verbs for named multidimensional arrays: point clouds, images,
sensor measurements, scientific and complex-valued arrays. Built on xarray
and NumPy, in the spirit of [tidy3](https://github.com/rleyvasal/tidy3).

**Status:** early development, not yet released. The design and its
decisions are in [docs/plan.md](docs/plan.md); the rules fit on one page in
[docs/decisions.md](docs/decisions.md).

To work on it:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The point-cloud example runs on real nuScenes Mini data and checks itself
against plain NumPy:

```bash
python examples/nuscenes_sweep.py /path/to/nuscenes-mini
```

Three public sample datasets (gridded air temperature, ERA-Interim wind, and
stars' Landsat scene) run the same way against plain xarray; see
`examples/sample_data.py` for the files and what each one checks.

Before pushing, run what CI runs, each job in a fresh environment with only
tidy3A's declared dependencies (needs [uv](https://docs.astral.sh/uv/)):

```bash
scripts/check.sh
```

```python
from tidy3a import *

cloud = read_array(
    path,
    value_type="float32",
    dims={"point": infer, "feature": 5},
    labels={"feature": ["x", "y", "z", "intensity", "ring"]},
    field_axis="feature",
)

nearby = (
    cloud
    >> mutate(p = dot(R, [x, y, z], over="feature") + offset)
    >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}"))
    >> filter(distance <= 50)
)
```

The example shows the planned interface. Implemented so far: `read_array`,
`infer`, `tidy()`, and the tidy-layer verbs (`select`, `filter`, `slice`,
`mutate`, `group_by`, `summarise`, joins, `pull`, `bind_cols`) in script
syntax, where names are written with `col("x")` or declared with
`x, y, z = fields("x", "y", "z")`, and the array layer (`dot`, numeric
functions, `conj`, `rename_axes`, `pack`), and `formula()` for plain
mathematical text and a strict LaTeX dialect. In notebooks,
`%load_ext tidy3a.jupyter` turns on bare names as in the example above; it
works alongside tidy3's notebook layer. Scripts declare names with
`fields()` or write `col("x")`.

## License

MIT
