# tidy3A

Tidy-style verbs for named multidimensional arrays: point clouds, images,
sensor measurements, scientific and complex-valued arrays. Built on xarray
and NumPy, in the spirit of [tidy3](https://github.com/rleyvasal/tidy3).

**Status:** design stage. No code has been released yet. The design and
its decisions are in [docs/plan.md](docs/plan.md).

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

The example shows the planned interface; none of it is implemented yet.

## License

MIT
