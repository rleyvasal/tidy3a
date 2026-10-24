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
`infer`, and `tidy()`, with inspection output.

## License

MIT
