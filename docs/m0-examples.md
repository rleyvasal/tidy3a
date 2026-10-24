# Milestone 0: acceptance examples in two styles

Status: prototype comparison for review. None of the tidy3A code below is implemented. Each example's plain-xarray equivalent runs in `prototypes/test_examples.py` on small synthetic data with hand-checked answers, and `prototypes/test_xarray_assumptions.py` checks the xarray and NumPy behavior the plan relies on. Both pass (39 tests, xarray 2026.9.0, NumPy 2.5.4).

The question this answers (plan §5.1, milestone 0): does `select` with a declared **field axis** read better than the **stars style**, where fields become separate variables (stars' `split`, tidy3A's `variables_from=`) and `select` picks only variables?

- **Style A, field axis:** `read_array(..., field_axis="feature")`. One array; field labels act as columns.
- **Style B, stars style:** `read_array(..., variables_from="feature")`. One variable per field, as after stars' `split(x, "feature")`. Vectors are rebuilt with a helper, shown here as `merge(x, y, z, along="feature")` after stars' `merge()`.
- **Style C, strict without splitting:** no field axis and no split. Field labels are reached only through `filter` and xarray. Shown once, for comparison.

## Example 1: point cloud

Synthetic data: four points that, after a 90° rotation about z and a 1 m offset, lie 49 m, 50 m, 51 m, and 3 m from the origin; `intensity` 10, 20, 30, 40; `ring` 1, 2, 1, 2.

**Style A**

```python
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

nearby >> group_by(ring) >> summarise(mean_intensity = mean(intensity))
```

**Style B**

```python
cloud = read_array(
    path,
    value_type="float32",
    dims={"point": infer, "feature": 5},
    labels={"feature": ["x", "y", "z", "intensity", "ring"]},
    variables_from="feature",
)

nearby = (
    cloud
    >> mutate(p = dot(R, merge(x, y, z, along="feature"), over="feature") + offset)
    >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}"))
    >> filter(distance <= 50)
)

nearby >> group_by(ring) >> summarise(mean_intensity = mean(intensity))
```

**Style C**

```python
xyz = cloud >> filter(feature.is_in(["x", "y", "z"]))
# Bare x, intensity, and ring do not exist; expressions need xarray:
#   cloud.to_xarray()["values"].sel(feature="ring")
```

Expected (checked): kept points 1, 2, 4 with original positions as labels, since 50 m is inside the boundary; intensities 10, 20, 40; mean intensity per ring: ring 1 → 10, ring 2 → 30.

## Example 2: sensor time series

Synthetic data: `values (time: 4, sensor: front rear, feature: x y z)`.

| Operation | Style A | Style B |
|---|---|---|
| Mean x per sensor | `group_by(sensor) >> summarise(mean_x = mean(x))` | same |
| All fields per sensor | `group_by(sensor) >> summarise(m = mean(values))` → `m (sensor, feature)` | one summary per field, e.g. `summarise(mx = mean(x), my = mean(y), mz = mean(z))` |
| Front sensor after time 2 | `filter(sensor == "front", time > 2)` | same |
| Times where any sensor has x > 20 | `filter(any(x > 20, over="sensor"))` | same |

Expected (checked): mean x is 4 (front) and 40 (rear); the front sensor's field means are x 4, y 3, z 0.5; the two-axis filter keeps `time: 2, sensor: 1`; the `any` filter keeps times 2, 3, 4.

## Example 3: geospatial raster

Synthetic data in the shape stars handles: a 3 × 2 pixel scene, bands `blue green red nir`, two dates, with `x`/`y` map coordinates. Near-infrared halves at the second date.

**Style A** (`field_axis="band"`)

```python
scene >> mutate(ndvi = (nir - red) / (nir + red))
scene >> filter(x > 500010, y > 4100010)                     # crop, as stars' filter
scene >> mutate(ndvi = (nir - red) / (nir + red)) >> group_by(x, y) >> summarise(mean_ndvi = mean(ndvi))
scene >> summarise(m = mean(values))                         # one mean per band; stars: st_apply(x, "band", mean)
scene >> mutate(ndvi = …) >> group_by(bins(ndvi, [-1, 0.25, 1])) >> summarise(pixels = n())
```

**Style B** (`variables_from="band"`)

```python
scene >> mutate(ndvi = (nir - red) / (nir + red))            # same
scene >> filter(x > 500010, y > 4100010)                     # same
# per-pixel mean NDVI and bins: same
scene >> summarise(blue = mean(blue), green = mean(green),   # one summary per band
                   red = mean(red), nir = mean(nir))
```

Expected (checked): NDVI at the first date is 0.5, 0, 0, 0.75, 0.5, 0 by pixel; the crop keeps `x: 2, y: 1`; the first pixel's mean NDVI over time is (0.5 + 0.2) / 2; the per-band mean of blue is 0.05; the NDVI bins hold 3 and 3 pixels.

## Comparison

| Aspect | Style A, field axis | Style B, stars style | Style C, strict |
|---|---|---|---|
| Scalar expressions (`distance <= 50`, NDVI, `mean(x)`) | Bare names | Bare names, identical text | Not available without xarray |
| Vector operations (rotation) | `[x, y, z]` directly | Needs a re-packing helper (`merge`) | xarray |
| One summary for every field | `mean(values)` keeps the field axis | One expression per field, or an `across()`-style helper | xarray |
| Fields with different types (`ring` as integer) | Not possible: one array, one type | Natural | Not possible |
| Concepts a user must learn | Field axis, observation axis | Separate variables only | Separate variables only |
| Rules in the tidy layer | More: field-axis select, `var()`/`label()` collisions, write-back, type promotion | Fewer | Fewest |

Findings:

1. **Style C is clearly worst.** Without either a field axis or splitting, the most common point-cloud and raster operations leave the pipeline. It is not a real option.
2. **Styles A and B read the same for scalar work.** Filters, NDVI, grouping, and bins are character-for-character identical. The choice of style changes nothing for most expressions.
3. **Style A wins where fields form a vector or a spectrum:** rotation (`[x, y, z]` versus `merge(...)`), and one summary for every band or component (`mean(values)` versus one line per field).
4. **Style B wins where fields are heterogeneous:** different types and units, such as nuScenes' `ring`, and it needs fewer rules.
5. Both styles are already in the plan, as `field_axis=` and `variables_from=`. The comparison supports keeping both, with documented guidance: use `field_axis=` when fields are components of one quantity (`x y z`, spectral bands, RGB), and `variables_from=` when they are different quantities.

Recommendation for review: keep the field axis. Its advantage is real but narrow (vectors and spectra), so the documentation should present `variables_from=` as equally first-class, and milestone 2 should check the tidy layer's one-page complexity budget (`docs/decisions.md`) before adding more field-axis rules.

Open item found while writing Style B: the re-packing helper (`merge(x, y, z, along=)`) is not in the plan. Style B needs it, or an equivalent, for vector operations on split variables. Its name clashes conceptually with joins, so it needs its own decision.
