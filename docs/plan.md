# tidy3A: repository and implementation plan

Status: design handoff for review, not an implemented API.

This document collects the decisions made in the discussion and identifies the remaining decisions explicitly. All tidy3A imports, readers, wrappers, and expressions below are proposed interfaces. NumPy, xarray, and nuScenes calls are existing APIs. No new repository or remote has been created by this plan.

## 1. Purpose and package boundary

Build a separate Python package for tidy-style manipulation of named multidimensional arrays. Use the display name **tidy3A**, with **tidy3a** as the proposed repository, distribution, and Python import name. Check name availability before creating or publishing them.

The package should serve point clouds, images, sensor measurements, scientific arrays, and complex-valued arrays. Autonomous-vehicle data is an acceptance example, not the organizing principle of the package.

Keep tidy3's Polars implementation and release pipeline separate. tidy3A should use xarray for named-array semantics and NumPy for basic numerical storage and computation. Optional array and file backends can follow later. Do not make tidy3A a new execution backend inside tidy3 as the first implementation.

Success means a user can import an array, understand its organization, select and filter observations, derive values, and summarize measurements using familiar verbs. Matching all of APL, building a quantum simulator, or reproducing every tidy3 verb is outside the first release.

### 1.1 Two layers

Tidy verbs fit array data that is mostly observations with fields laid out over a few extra axes: point clouds (`point × field`), sensor logs (`time × sensor × field`), image pixels (`row × col × channel`). Array mathematics treats axes as first-class: contractions over chosen axes, batch axes, and index binding. Hiding axis names there would need ever more inference rules and end up harder to understand than plain xarray. tidy3A therefore has two layers with different conventions:

| Layer | Contents | Axis convention |
|---|---|---|
| **Tidy layer** | `read_array`, `tidy`, `select`, `filter`, `mutate`, `group_by`, `summarise` | Axis roles are declared once (`field_axis=`); verbs never take `dim=` |
| **Array layer** | `dot`, reductions, `formula`, elementary numeric functions | Axes and indices are named openly: `over=` on helpers, explicit indices such as `\sum_i` in formulas |

Array-layer helpers are used inside tidy-layer verbs, for example `mutate(p = dot(R, [x, y, z], over="feature"))`, so the two layers compose without sharing rules. Complexity budget: the rules for the tidy layer (name resolution, select versus filter, grouping) must fit on one page of documentation. If they grow beyond that, simplify the layer rather than adding rules. See §13 for the prior art behind this split.

## 2. Decisions carried forward

| Topic | Direction |
|---|---|
| Distribution | Separate package, dependencies, CI, and releases |
| User vocabulary | Reuse `tidy`, `select`, `filter`, `mutate`, `group_by`, `ungroup`, and `summarise` |
| New verbs | Keep additions minimal; do not initially add `reshape`, `label_coords`, or spatial transformation verbs |
| Import | Put initial reshaping, axis naming, and labeling in `read_array` |
| Import sizes and labels | `dims` holds sizes and `infer` only; `labels` is a separate mapping of label lists, one axis per line |
| Import shape | One array by default; `variables_from="<axis>"` opts in to one variable per label |
| Import return | `read_array` returns the tidy3A wrapper; `tidy()` wraps existing xarray objects |
| Import memory | Load into memory by default; `lazy=True` opts in to memory-mapping |
| Inferred size | Use a proposed `infer` marker instead of requiring `-1` |
| Type terminology | Keep `value_type` as a working name for one stored numeric type; do not block implementation on spelling |
| Two layers | Tidy layer (verbs, axis roles declared once) and array layer (math helpers that name axes openly); see §1.1 |
| Axis arguments | Do not require `dim=` on routine `select`, `filter`, `mutate`, and `summarise` calls |
| Array-layer axes | Math helpers name summed axes with `over=`; formulas name indices explicitly; no inference of batch axes |
| Select syntax | `data >> select(x, y, z)` is primary; scripts use strings or `fields(...)` |
| Select result | Keeps the imported shape and takes less of it; never reorganizes data |
| Field axis | `read_array(..., field_axis="feature")` declares the one axis whose labels act as column names; other axes are observation axes |
| Select vs filter | `select` picks variables and field-axis labels (columns); observation-axis labels are values and go through `filter` |
| Bare field names | In expressions, a field-axis label is the field with the field axis dropped, e.g. `x(point)` |
| Vector form | `[x, y, z]` of field-axis references is a vector view keeping the narrowed field axis; lists of values or strings stay literals |
| Mutate on field labels | `mutate(x = …)` writes back into the shared array; the result must have the field's shape |
| Multi-axis filter | Error with hints: reduce with `any`/`all(…, over=)`, or mask explicitly with `mutate(… where …)` |
| Filter across axes | One-axis predicates on different axes each filter their own axis (AND, comma or `&`); OR across axes is an error |
| Empty filter result | Returned with a zero-length axis, like dplyr; flagged in inspection output |
| Filter positions | Filtering an unlabeled axis keeps original positions as labels; never renumbers |
| Slice | `slice(point = …)` selects by current position on the named axis; 1-based, negatives exclude, matching tidy3; `0` is an error |
| Alignment default | Exact: labels on a shared axis must match or the expression is an error; joins are explicit (§5.5) |
| Intentional joins | `left_join`, `inner_join`, `full_join` align on shared axes via `xr.merge`; expressions stay exact |
| Join name clashes | Suffix like tidy3, `suffix=("_x", "_y")` by default; renamed before `xr.merge` |
| Labeled vs unlabeled | One side labeled and the other not is an error, even at equal length; both unlabeled pair by position |
| Outside arrays and labels | `bind_cols(scores, along="point")` attaches by position; `pull(point)` returns an axis's labels; no accessor method |
| Label tolerance | None by default; rounding mismatches are an error that suggests `tolerance=` on join verbs |
| Label order | Same labels in a different order are matched by label, reordered to the first operand, and reported |
| Axis renaming | `rename_axes(expr, output="input")` inside expressions; distinct from tidy3's variable `rename` |
| Complex conjugation | `dot` never conjugates; inner products use explicit `conj()`, as in xarray and NumPy |
| Grouping by axis | `group_by` axes are kept; `summarise` reduces all other observation axes; the field axis is never reduced (stars' `st_apply` MARGIN) |
| Grouping by value | `group_by(ring)` groups the one axis `ring` varies along; result gets a `ring` axis of sorted unique values (xarray `groupby`) |
| Missing grouping axis | A summary using a variable without the grouping axis is an error; summarise it separately |
| Grouping by bins | `group_by(bins(distance, [0, 10, 50]))` or `bins(time, "2 days")`; left-closed, right-open; out-of-range values are an error |
| Missing values | Summaries propagate missing values like dplyr/tidy3; `na_rm=True` skips them; `min_count` as in tidy3 |
| Grouped mutate | Grouped summaries in `mutate` broadcast back onto each observation, like dplyr |
| Groups after summarise | Follow tidy3's `groups=`: `"drop_last"` by default, `"drop"`, `"keep"`; `ungroup()` removes grouping |
| Formula conventions | Follow plot3's fixed LaTeX meanings; letters side by side multiply; multi-letter names use `\mathrm{...}`; a run matching a data name is an error |
| Formula subscripts | `p_i` indexes axis `i`, `p_x` selects label `x`, `p_1` a 1-based position; underscore names use `\mathrm{x\_1}` |
| Formula index binding | Letters not matching an axis name are bound with `bind={"j": "feature"}`; by name, never by position |
| Formula bounds | `\sum_i` sums the whole axis (main form); numeric bounds are an optional checked 1-based slice; `n` must be bound |
| Imaginary unit | Plain `i` is always an index or name; the imaginary unit is `\mathrm{i}` (or `\mathrm{j}`) |
| Formula show-back | Every formula displays its bindings and compiled array code in notebooks and `repr()`; never printed during execution |
| Notebook coexistence | Each package's transformer rewrites only calls to functions it owns, checked in the live namespace; requires a tidy3 change (§5.6) |
| Notebook code | Copy and adapt tidy3's `masking.py` into tidy3A with attribution; no dependency on tidy3 |
| Variable vs label names | `var("x")` for the variable, `label("x")` for the field-axis label when a name is both |
| Selection helpers | tidy3's tidyselect helpers (`~`, `starts_with`, …) on variables and field labels |
| Masking function | `if_else(cond, x, y)` as in dplyr/tidy3; three-argument `where` is an error pointing to `if_else` |
| Slice and axes | `slice(time = 1)` keeps `time: 1`, like dplyr; `drop=True` removes the axis explicitly |
| First release | Raw and `.npy` via `read_array`, `tidy()` for xarray objects, `lazy=True` memory-mapping, eager NumPy; Dask and `read_binary` later |
| Mutate order | Sequential like dplyr/tidy3; vector updates use one assignment, `[x, y, z] = …` (notebook) or `assign([x, y, z], …)` (script) |
| Write-back value type | A different type promotes the whole array to the common type, never casts down, and is reported in inspection output |
| Data vs notebook names | Data-only names are data, namespace-only names are Python values, names in both are an error (`col()`/`env()`); deliberately differs from tidy3, where data wins |
| Select across variables | Field-axis label selection applies to every variable with that axis; others are unchanged |
| Mathematical helpers | Provide `dot` for generalized multiplication and summation, and `formula` for mathematical notation |
| Formula input | Support ordinary mathematical text such as `"x^2"` and a documented subset of LaTeX |
| Notebook use | Support bare variable/field names, consistent with tidy3's notebook experience |
| Generality | No built-in nuScenes schema or automatic interpretation of XYZ, intensity, or quantum amplitudes |

These are design directions. Detailed selection, grouping, and index-binding semantics need the small prototype in milestone 0 before becoming a public contract.

## 3. Explain the data model first

Use these terms consistently in the README, errors, and API documentation:

| Term | Meaning | Point-cloud example |
|---|---|---|
| Value | One stored measurement | A coordinate or intensity number |
| Variable | A named array of values | `values`, later `p` and `distance` |
| Dimension / axis | A direction used to index an array | `point`, `feature` |
| Size | Number of positions along an axis | N points, five features |
| Label / coordinate | Identifies a position along an axis | Feature labels `x`, `y`, `z`, `intensity`, `ring` |
| Field axis | The one axis, declared at import, whose labels act as column names | `feature` |
| Observation axis | Any other axis; its labels are values, selected with `filter` | `point`, `time`, `sensor` |
| Metadata | Additional meaning supplied by the source/user | Metres, sensor frame, timestamps |

tidy3A organizes labeled fields in one of two structures. Other fields use different names for the same two structures; the documentation should list them so readers from any background recognize what they have:

| tidy3A term | Also called | Example |
|---|---|---|
| **One array** (default) | *interleaved* (audio, imaging), *array of structs (AoS)* (HPC, graphics), *unstructured* array (NumPy), DataArray with a labeled dimension (xarray) | `values (point, feature)` with feature labels `x y z intensity ring` |
| **Separate variables** (`variables_from=`) | *planar* (audio, imaging), *struct of arrays (SoA)* (HPC, graphics), Dataset of variables (xarray `to_dataset(dim=...)`), attributes made with stars' `split(x, "band")` (reversed by `merge()`); related to tidyr's `pivot_wider(names_from=...)` | `x`, `y`, `z`, `intensity`, `ring`, each `(point)` |

Avoid "structured" and "stacked/unstacked" in the API. A NumPy structured array is still one array of interleaved records, and xarray's `unstack` already means unfolding a multi-index dimension.

Three spatial coordinates do not require three independent array axes. A raw cloud of N irregular points with five fields is naturally an N-by-5 array. A voxel grid indexed independently by x, y, and z is a different representation.

Maintain an xarray Dataset internally. Wrap an incoming DataArray as one variable, preserving its existing name or using `values` when it is unnamed. A Dataset can contain variables with different dimensions and value types. A conventional numeric DataArray has one value type.

Proposed small wrapper: `TidyArray`. Its responsibilities are pipe execution, grouping context, expression evaluation, and inspection. Keep its xarray data accessible through `to_xarray()`; define `collect()` to return a computed xarray Dataset. For an eager array, collection should not make an unnecessary copy. This gives one predictable return type even for a single-variable import.

## 4. Import contract

### 4.1 Homogeneous raw arrays

Decided interface:

```python
data = read_array(
    path,                              # file path or in-memory NumPy array
    value_type="float32",              # required for raw bytes; checked against .npy
    dims={"time": infer, "sensor": 2, "feature": 3},
    labels={
        "sensor": ["front", "rear"],
        "feature": ["x", "y", "z"],
    },
    field_axis="feature",              # labels on this axis act as column names
    # optional:
    # variables_from="feature"  each label becomes its own variable (default: one array)
    # name="values"         variable name when the data stays one array
    # byte_order="little"   or "big"; raw files only
    # order="C"             or "F"; raw files only
    # offset=0              bytes to skip, e.g. a file header
    # lazy=False            True memory-maps instead of loading into memory
)
```

The reader returns the tidy3A wrapper, ready for pipes; `.to_xarray()` leaves it. `tidy()` remains the way to wrap existing xarray objects, and calling `tidy()` on an already wrapped object returns it unchanged.

The ordered `dims` mapping defines shape and axis names and holds only sizes and `infer`. `labels` is a separate mapping of label lists, written one axis per line for readability. A size is kept even when labels imply it: a mismatch between the size and the label count is an error, which catches mislabelled raw bytes. Labels are always lists; there is no space-separated string shorthand, so labels containing spaces and numeric labels need no special rule. Unlabeled axes are usable by integer position; do not invent timestamps or physical coordinates. The reader must explain what it inferred in its representation or an inspection report:

```text
<TidyArray source: raw file, float32 little-endian, row-major>
values (time: 1200, sensor: 2, feature: 3) float32
  feature: x  y  z          field axis (labels act as columns)
  sensor:  front  rear      observation axis (labels are values)
  time:    unlabeled, 0 … 1199
inferred: time = 28,800 bytes / 4 bytes / (2 × 3) = 1200
```

`field_axis=` declares which axis holds field names, the labels that would be column names in a tidy table, such as `x y z intensity ring`. Every other axis is an observation axis, whose labels are values, such as `front rear` or timestamps. Each array has at most one field axis. The reader never guesses it from the labels; without `field_axis=`, the array has no field axis and `select` picks only variables (§5.1).

Separate variables are opt-in. By default the data stays one array named `values`, with labels on its axes. `variables_from="feature"` ("take the variables from the `feature` axis") turns each label on that axis into its own variable, keeping every other axis: `(time, sensor, feature)` becomes `x(time, sensor)`, `y(time, sensor)`, and `z(time, sensor)`. A single array suits components of one vector, such as `x y z`; separate variables suit fields with different meanings or units, such as nuScenes' `intensity` and `ring`. See §3 for the terms other fields use for these two structures. Rules:

- The `variables_from` axis must have labels; they become the variable names.
- `name=` applies only to a single array; passing both `name=` and `variables_from=` is an error.
- `field_axis=` must name an axis that still exists in the result. Naming the same axis as `variables_from=` is an error, because that axis becomes variables and no longer exists; the fields are already selectable as columns. A different axis is valid: with `dims=(time, sensor, feature)`, `variables_from="feature"` plus `field_axis="sensor"` gives `x(time, sensor)`, `y(time, sensor)`, and `z(time, sensor)`, and `select(front)` narrows `sensor` on all three.

```text
ReadError: field_axis="feature" names an axis that variables_from="feature" turns into variables.
  Use one: variables_from="feature" already makes x, y, z, … selectable as columns.
```
- Labels that are not valid Python identifiers still become variables but require `col("...")` in expressions.

The option names the result, not the file layout. If on-disk planar files need support later, describe them with a separate file-layout option using `interleaved`/`planar`, so the two are never confused.

Memory-mapping is opt-in with `lazy=True`. The reader never switches behavior based on file size, so results do not depend on the file remaining on disk unless the user asked for a memory-mapped array.

For raw files, the type describes the on-disk encoding, not a desired post-import cast. Account for byte order and storage order, with documented defaults and explicit overrides. The example uses interleaved, row-major records. Do not guess those properties from the values.

Inference rules:

- Infer at most one unknown size from the total element count and the known sizes.
- Require the values to fit the requested shape exactly. Never truncate, repeat, or pad them silently.
- Check that labels match axis lengths and that dimension names are unique.
- Define empty-file behavior: zero observations with a positive known record width is valid; shapes that do not uniquely determine the inferred size are errors.
- Report conflicts between explicitly supplied information and stored metadata.

The wrapper and all core operations must work at any array rank, not only on N-by-5 examples.

### 4.2 Formats with stored metadata

Initially support `.npy` alongside raw numeric files and wrapping existing xarray objects. Read stored shape and type from `.npy`; avoid loading object arrays through pickle by default. Detect `.npy` by its header, not its suffix. For `.npy`, users only name axes and add labels, for example `read_array("sensors.npy", dims={"time": infer, "sensor": 3})`; a `value_type` that conflicts with the stored type is an error, not a cast. Preserve dimensions and coordinates when an existing xarray object is wrapped.

NetCDF and Zarr are wrapped through `tidy(xr.open_dataset(...))` in the first release rather than read by `read_array`. Dedicated readers can be added later as optional integrations. Preserve their stored types, dimensions, coordinates, attributes, and lazy/chunked storage where supported. Infer only information the format actually supplies. Import-time metadata discovery and statistical guessing from raw bytes are different things.

Do not require users to repeat information already present in the source. Avoid a registry of dataset-specific profiles in the core package.

### 4.3 Mixed binary records

Include this in the roadmap, but do not overload a homogeneous numeric reader with an unclear record interpretation.

Working proposal: a generic `read_binary` reader accepting an ordered `value_types` mapping. A different name or one-reader design is a review decision, not an additional committed API.

```python
records = read_binary(
    path,
    value_types={
        "x": "float32",
        "category": fixed_text(20, encoding="utf-8"),
    },
)
```

This is a generic description of file layout, not a named domain schema. Fixed-width strings require byte widths, encoding, and documented padding behavior. Numeric fields require encoding and byte order. Padding/alignment between fields must be handled explicitly when present.

Return separate typed variables indexed by an observation dimension. Variable-length records require framing information such as lengths or delimiters and can be deferred. A `.bin` suffix does not identify the format.

In actual nuScenes LiDAR files, all five point fields are stored as float32, including the ring field. Convert ring values to integers only after decoding, if desired.

## 5. Familiar verbs and reference resolution

### 5.1 Selection without exposing axis arguments everywhere

The desired notebook experience is:

```python
xyz = data >> select(x, y, z)
```

The equivalent script syntax is:

```python
xyz = data >> select("x", "y", "z")
```

The bare-name form is the primary documented syntax. Notebooks support it through an AST rewrite of names inside verb calls into deferred references, resolved against the data when the pipe runs, following tidy3's `jupyter.py`/`masking.py` precedent. Plain scripts use strings, or declare the names once with a small proposed helper, `x, y, z = fields("x", "y", "z")`, and then use them bare.

Decided: as in the tidyverse, `select` picks columns and `filter` picks rows. In tidy3A, columns are variables and labels on the field axis (§4.1). Labels on observation axes are values, so they go through `filter`:

| Axis | Labels | In a tidy table these would be | Verb |
|---|---|---|---|
| `feature` (field axis) | `x y z intensity ring` | column names | `select(x, y, z)` |
| `sensor` | `front rear` | values in a `sensor` column | `filter(sensor == "front")` |
| `time` | timestamps | values in a `time` column | `filter(time > t0)` |

```text
SelectError: "front" is a value on the sensor axis, not a column.
  To keep those observations, use: filter(sensor == "front")
```

Decided: `select` keeps the imported shape and just takes less of it; it never reorganizes the data.

| How the data was imported | `select(x, y, z)` returns |
|---|---|
| One array (default) | One array, `values (point, feature: 3)`, with feature labels `x y z` |
| `variables_from="feature"` | The three variables `x`, `y`, `z`, each `(point)` |

Keeping components together in the default case lets vector operations such as `dot(R, values, over="feature")` follow directly. Users who want separate variables choose `variables_from=` at import.

Resolution rule, to validate in milestone 0:

1. If the names identify Dataset variables, select those variables.
2. Otherwise, if the names identify labels on a field axis, select those positions on that axis for every variable that has it.
3. A name found only on an observation axis is an error that suggests the equivalent `filter`.
4. Keep all other axes and their coordinate identities intact.
5. If a name is both a variable and a field-axis label, report both meanings; do not pick one silently.

Decided: `var("x")` names the variable and `label("x")` names the field-axis label when a name is both. `col()` and `env()` cannot do this, because they separate data from notebook values, and both meanings here are data.

```python
data >> select(var("x"))                    # the variable x
data >> select(label("x"))                  # the x label on the field axis
data >> mutate(d = var("x") - label("x"))
```

`label("x")` applies to every field axis that has `x`, as label selection does across variables. Both helpers work in `select` and in expressions; in a formula, a collision is resolved with `bind={"x": var("x")}`. The error lists both spellings:

```text
SelectError: "x" is both a variable and a label on feature.
  variable:  var("x")
  label:     label("x")
```

Precedents: dplyr's `.data$x` / `.env$x` pronouns name which kind of thing a name means; xarray separates `ds["x"]` from `da.sel(feature="x")`.

Decided: `select` accepts tidy3's selection helpers (`tidyselect.py`), applied to variables and field-axis labels: `~` and `-` to exclude, `everything()`, `starts_with()`, `ends_with()`, `contains()`, `matches()`, `num_range()`, `all_of()`, `any_of()`, `where(predicate)`, and `last_col()`. Labels on observation axes are never selected; they belong to `filter`.

```python
cloud >> select(~intensity)              # values with feature narrowed to x y z ring
cloud >> select(~[intensity, ring])      # x y z only
data  >> select(starts_with("temp"))     # variables or field labels starting with "temp"
```

`where` is therefore the tidyselect selector, as in tidy3 (`select(where(is_numeric))`). Masking uses `if_else(cond, x, y)`, as in dplyr and tidy3 (`expr.py`, `if_else`); `case_when` follows tidy3 too. NumPy and xarray users will type `np.where`/`xr.where`-style masking out of habit, so a three-argument `where` is an error that points to `if_else`:

```text
TypeError: where() selects columns by predicate, as in tidy3 (select(where(is_numeric))).
  For NumPy/xarray-style masking use if_else(cond, x, y).
```

Precedents: tidy3's selection helpers and `if_else`; dplyr; stars' `select`, which passes through to dplyr's tidyselect; `xr.where` and `np.where` on the array side.

Relation to stars. The field axis extends stars, the closest precedent, which takes the strict route: its `select` picks only attributes (variables), and a field-like dimension such as spectral bands is an ordinary dimension, subset with `slice(band, 6)` or `filter(band > 3)`; users who want bands as columns call `split(x, "band")`. In tidy3A terms, stars style would be `cloud >> filter(feature.is_in(["x", "y", "z"]))`, where the field axis allows `cloud >> select(x, y, z)`. tidy3A keeps the field axis because point-cloud and record fields are column-like in a way spectral bands mostly are not, and the strict style makes the most common point-cloud operation awkward. Several decisions depend on it: bare field names in expressions (§5.2), the `[x, y, z]` vector form, `mutate` write-back (§5.4), and the `variable` hint. Because this is the largest departure from a proven design, milestone 0 must test both styles, field-axis select and stars-style strict select, on the point-cloud and time-series examples before the field axis becomes a public contract.

Relation to xarray. xarray avoids this ambiguity by always naming the axis: `ds["x"]` selects a variable and `da.sel(feature="x")` selects a label, with every axis treated alike. In its tabular view, `Dataset.to_dataframe()` makes only variables into columns and every dimension into the row index, and the NetCDF/CF practice it follows stores distinct quantities, such as wind components `u` and `v`, as separate variables. xarray still marks an axis that holds variable names as special: `Dataset.to_dataarray()` (formerly `to_array()`) and `to_stacked_array()` name it `variable` by default. The field-axis rule exists because tidy3A drops the explicit axis name from `select(x)`, so the axis's role has to be declared instead. Declaring `field_axis="feature"` means `select` treats the data as if `to_dataset(dim="feature")` had been applied, without moving any data; `variables_from="feature"` performs that conversion at import. `tidy()` accepts the same explicit `field_axis=` as `read_array`, for example `tidy(da, field_axis="variable")`.

Decided: an axis named `variable` is not treated as the field axis automatically. That would decide an axis's role from its name, which is a guess: `variable` is only xarray's default, users rename it, and some data uses `variable` as an ordinary axis whose labels are values. Instead, the inspection output offers a hint:

```text
values (variable: 3, time: 1200)
  variable: temp  pressure  humidity
  hint: "variable" looks like an axis of variable names (xarray's to_dataarray default).
        To select its labels as columns: tidy(da, field_axis="variable")
```

If milestone 0 shows the `to_dataarray()` round trip is common, automatic recognition can be promoted to a rule later; removing an automatic behavior afterwards would break pipelines that depend on it.

Support native xarray selection as an explicit escape hatch for ambiguous advanced cases. Whether to add a qualified selector later is open. Do not add a `dim=` parameter to every verb merely to accommodate future ambiguities.

Decided: when several variables share a field axis, label selection applies to every variable that has that axis, matching `xarray.Dataset.sel`. Variables without the axis are kept unchanged.

```text
position  (time, feature)   feature: x y z
velocity  (time, feature)   feature: x y z
```

`data >> select(x)` returns `position(time, feature: 1)` and `velocity(time, feature: 1)`. To narrow one variable only, select it first: `data >> select(position) >> select(x)`. Single-source raw arrays must not be ambiguous.

### 5.2 Bare field labels inside expressions

A field-axis label is not automatically a Dataset variable. Make this explicit in inspection output and documentation.

Decided: inside an expression, a bare field-axis label such as `x` is the field with the field axis dropped. On `values (point, feature)` with `field_axis="feature"`, `x` behaves like a variable `x(point)`; the remaining point, time, or other axes stay attached. This is exactly what `x` would mean after `variables_from="feature"`, so expressions read the same whichever import form was used, and results carry no leftover length-1 `feature` axis. In scripts, `col("x")` uses the same resolution rule.

```python
data >> filter(x > 10)          # x is x(point); filters points
data >> mutate(r = x * 2)       # r is a new variable r(point)
```

Decided: a list of field-axis references, `[x, y, z]`, is a vector view inside an expression. It keeps the field axis, narrowed to those labels, and stays inside the pipeline, so it remains aligned after earlier `filter` steps. This extends tidy3's existing notebook rewrite, which treats `select([x, y])` like R's `c(x, y)` (`masking.py`, `visit_List`).

```python
data >> mutate(p = dot(R, [x, y, z], over="feature"))   # [x, y, z] is (point, feature: 3)
```

Rules:

- In notebooks, the rewrite turns each bare name in the list into a data reference. In scripts, `x, y, z = fields("x", "y", "z")` makes `[x, y, z]` a plain list of references with the same meaning.
- A list whose items are all data references becomes a vector view. A list of plain values, such as `[1.0, 0.0, 0.0]`, stays a literal.
- A list of strings, such as `["x", "y", "z"]`, stays a literal; it is never read as data.
- A list mixing references and values, such as `[x, 0.0, z]`, is an error.
- All references must be labels on the same field axis; otherwise the error names the axes involved.

`c()` is not added: tidy3 has no `c()`, and `c` is a common notebook name that would collide under the name-resolution rule below. The documentation notes that `[x, y, z]` corresponds to R's `c(x, y, z)`.

Decided: when the pipe runs, each bare name inside a verb resolves by where it exists.

| Name exists in | Resolves to |
|---|---|
| The data only (variable or field-axis label) | The data |
| The notebook/Python namespace only, such as `R` or `offset` | The Python value |
| Both | Error; the user must write `col("x")` for data or `env("x")` for the Python value |

```text
NameError: "x" is both a field in the data and a notebook variable (x = 5).
  data:      col("x")
  notebook:  env("x")
```

A silent winner would let a variable defined many cells earlier change what a pipeline computes. The cost is that common notebook names such as `x`, `i`, or `n` sometimes need `col()` or `env()`. Resolve actual variables and field-axis labels consistently under this rule.

Difference from tidy3, kept deliberately: tidy3's notebook layer lets data names win, following dplyr's data mask; its `__tidy3_name__` reference "picks the column when the frame has one, … else the value" (`masking.py`). tidy3A raises an error instead, because array notebooks routinely define short names such as `x`, `i`, `n`, `R`, and `t`, which makes collisions far more likely than in table workflows. The same code can therefore behave differently in the two packages when a name collides; tidy3A's error message should mention this for users coming from tidy3.

Prototype this with variables and labels having the same spelling, duplicated labels, multiple labeled axes, and imports in different orders. Agree on the behavior before implementing a broad notebook rewrite.

### 5.3 Filter

`filter` also handles labels on observation axes, which are values rather than columns: `filter(sensor == "front")` keeps the `front` position on `sensor` for every variable that has that axis. An observation-axis name in a predicate refers to that axis's labels.

For a predicate varying along a single observation axis, filter that axis and slice every dependent variable consistently. Example: a `distance(point)` predicate filters points and their associated intensity, ring, and transformed position values.

Decided: a predicate that varies along more than one axis is an error with hints; it is never masked or flattened automatically. On `data (time, sensor, feature)`, `x` is `x(time, sensor)`, so `filter(x > 10)` selects an irregular set of cells that no array can hold:

```text
FilterError: x > 10 varies along time and sensor; filter keeps whole positions on one axis.
  Keep times where any sensor matches:  filter(any(x > 10, over="sensor"))
  Keep times where all sensors match:   filter(all(x > 10, over="sensor"))
  Keep the shape, blank failing cells:  mutate(x = if_else(x > 10, x, nan))
```

`any`/`all` with `over=` (the array-layer convention, §6.1) reduce the predicate to one axis, making it an ordinary filter. Shape-preserving masking is an explicit `mutate` with `if_else` (§5.1); no `mask` verb is added. Automatic masking was rejected because nothing is dropped and integer arrays have no missing value; flattening to a table was rejected because it reorganizes the data, contrary to the rule chosen for `select`. This goes beyond R's stars, whose documented `filter` selects on dimension coordinates (`filter(x > 289000, x < 291000, band > 3)`), not on attribute values.

Decided: one-axis predicates on different axes each filter their own axis, and together they keep a rectangle. As in dplyr, comma-separated predicates combine with AND; `&` between predicates on different axes is treated the same way. OR across axes would keep an L-shaped set of cells, so it is the multi-axis error above:

| Written | Result | Allowed? |
|---|---|---|
| `filter(sensor == "front", time > t0)` | front sensor × later times | Yes |
| `filter((sensor == "front") & (time > t0))` | same rectangle | Yes |
| `filter((sensor == "front") \| (time > t0))` | front at all times plus every sensor after `t0` | No: multi-axis error |

The rule follows from one principle: the result of `filter` must be a rectangle. stars behaves the same way for conditions on different dimensions: its documented example `filter(x > 289000, x < 291000, band > 3)` keeps a rectangle.

Decided: a filter that matches nothing returns an empty result, as dplyr returns a zero-row tibble. `cloud >> filter(distance <= 0.1)` with no matches gives `values (point: 0, feature: 5)`, with every variable on that axis cut to zero positions and all other axes and labels intact. This matches the `read_array` rule that an empty file gives `point: 0`, and keeps batch pipelines over many files running when some have no matches. The inspection output flags it, for example `point: 0 (all removed by filter(distance <= 0.1))`. Reductions over an empty axis follow the missing-value rules decided with grouping (§5.4).

Decided: filtering an unlabeled axis keeps the original positions as labels; positions are never renumbered. After `cloud >> filter(distance <= 50)`, kept points are labelled with their original positions, so point 15 remains point 15. Implementation: before slicing an axis without a coordinate, attach its positions (`0 … N-1`) as the coordinate. Reasons:

- Other data is often matched by position, such as nuScenes per-point segmentation labels stored in a separate file.
- Exact alignment compares labels, so combining a filtered result with unfiltered data either lines up correctly or raises an error, instead of silently pairing the wrong points when lengths happen to match.
- §5.3's promise that filtering keeps variables aligned to the retained point identifiers depends on it.

```text
values (point: 12034, feature: 5) float32
  point: original positions 3, 7, 8, 15, … (12,034 of 34,688 kept)
```

Positions are then no longer consecutive. `filter(point < 100)` compares labels (original positions), so selecting by current position needs `slice`.

Decided: add `slice` for current positions, following dplyr's `slice` and stars' `slice(along=)`. The keyword names the axis, which fits the no-`dim=` rule because the axis name is the argument:

```python
nearby >> slice(point = range(1, 101))   # first 100 kept points
nearby >> slice(time = 1)                # first time step
nearby >> slice(time = -1)               # every time step except the first
```

Like `filter`, `slice` applies to every variable that has the axis and keeps the original labels; `slice` selects by position and `filter` by labels and values.

Decided: positions are 1-based and negative positions exclude, matching tidy3's `slice` (`verbs.py`: "1-based row positions per group; negative positions exclude"), dplyr, stars, and the formula's mathematical positions (`\sum_{i=1}^{n}`). tidy3 translates to 0-based Polars at the boundary; tidy3A does the same on top of 0-based xarray (`isel`). Position `0` is an error that explains the convention:

```text
SliceError: positions start at 1, as in tidy3 and dplyr; position 0 does not exist.
  For the first item use slice(point = 1), or slice_head(point = 1).
```

Decided: `slice` keeps the axis even when one position remains, as dplyr and tidy3 never drop anything. `slice(time = 1)` on `data (time, sensor, feature)` gives `(time: 1, sensor, feature)`; `slice(time = 1, drop=True)` removes the axis, using stars' argument name. Precedents: stars and xarray both drop a single position by default (xarray's `isel(time=0)` drops, `isel(time=[0])` keeps); that was rejected because later steps such as `group_by(time)` or a join on `time` would break depending on `1` versus `[1]`. Verbs never reshape unless asked; bare `x` in expressions drops the field axis because that is what makes it a column (§5.2).

Proposed companions, following tidy3: `slice_head(point = 100)` and `slice_tail(point = 100)` for the first and last N positions without ranges. Note that `from tidy3a import *` shadows Python's built-in `slice`, as tidy3 already does.

### 5.4 Mutate and summaries

Decided: assigning to a field-axis label writes back into the shared array. On `values (point, feature)` with `field_axis="feature"`, `mutate(x = x * 2)` replaces the `x` part of `values`; `values` keeps its shape, and `y`, `z`, `intensity`, and `ring` are unchanged. This matches dplyr, where `mutate(x = x * 2)` replaces the column, and is consistent with `select` and expressions already treating field labels as columns. The new values must have the field's shape (the array without its field axis, e.g. `(point)`); a result with extra or missing axes is an error. Decided: when written-back values have a different value type, the whole shared array is promoted to the common type (NumPy's `result_type` of the two), so no information is lost. It is never cast down silently. Because one array holds every field on the field axis, promotion changes the type of all of them, for example from `float32` to `float64`, doubling memory, or from `float64` to `complex128`. A value that already fits, such as `int16` into `float32`, leaves the type unchanged. Promotion is reported in the inspection output:

```text
values (point: 34688, feature: 5) float64
  feature: x  y  z  intensity  ring      field axis (labels act as columns)
  point:   unlabeled, 0 … 34687
  changed: values promoted float32 → float64 by mutate(x = …); all fields on feature now float64
```

Promoting a memory-mapped (`lazy=True`) array makes an in-memory copy, which the report also states. The single array cannot hold fields of different types: `mutate(ring = as_type(ring, "int16"))` still stores `ring` as the array's type. Documentation should point to `variables_from=` for fields that need their own type. A new name, such as `mutate(x2 = x * 2)`, creates a separate variable `x2(point)`.

`mutate` adds/replaces variables while preserving the remaining data. Use named-dimension broadcasting and coordinate alignment.

Decided: assignments run in order, as in dplyr and tidy3 (`verbs.py`: a name means "a column that exists or that an earlier assignment" created). `mutate(a = x * 2, b = a + 1)` works in one call. Updating vector components one at a time therefore uses already replaced values and is a documented pitfall:

```python
data >> mutate(x = R[0,0]*x + R[0,1]*y,
               y = R[1,0]*x + R[1,1]*y)   # wrong: y uses the new x
```

Decided: vector updates assign all components in one assignment, so every component is computed from the original vector:

```python
data >> mutate([x, y, z] = dot(R, [x, y, z], over="feature"))   # notebook
data >> mutate(assign([x, y, z], dot(R, [x, y, z], over="feature")))   # script
```

`[x, y, z] = …` is not valid Python keyword syntax, so the notebook form needs a source preparser that rewrites it before parsing into the script form. tidy3 has the precedent: its preparser turns backtick names into `__tidy3_assign__("hp new", value)` positional assignments (`masking.py`, `make_named_assign`). Rules for the assigned value:

- It must have the target fields' shape plus exactly one extra axis whose labels equal the target labels, in order (`x y z`). `dot(R, …)` with `R` on axes `(i, feature)` produces axis `i` labelled `x y z`, which maps onto `feature`.
- Matching is by labels, never by length alone; a length-3 axis with different or missing labels is an error.
- Write-back follows the field-label rules above, including type promotion.

Decided: `group_by` on an axis keeps that axis, and `summarise` reduces every other observation axis. The field axis is never reduced, because its labels are columns, and a dplyr summary never averages across columns. On `data (time, sensor, feature)` with `field_axis="feature"`:

```python
data >> group_by(sensor) >> summarise(mean_x = mean(x))   # mean_x (sensor): mean over time
data >> group_by(sensor) >> summarise(m = mean(values))   # m (sensor, feature): each field separately
data >> summarise(m = mean(values))                       # m (feature): no groups, all observation axes reduced
```

This is stars' `st_apply(x, MARGIN, FUN)`, whose `MARGIN` dimensions are kept and all others reduced; stars itself has no `group_by` or `summarise`. An explicit `over=` inside `summarise` must agree with the rule: `summarise(m = mean(x, over="sensor"))` after `group_by(sensor)` is an error.

Decided: `group_by` on a value field groups positions by value. `ring(point)` varies along `point`, so `group_by(ring)` groups points:

```python
cloud >> group_by(ring) >> summarise(mean_intensity = mean(intensity))
# mean_intensity (ring: 32), labelled with the sorted unique ring values
```

Rules:

- The field must vary along exactly one observation axis; `summarise` replaces that axis with a new axis named after the field, labelled with its sorted unique values.
- Other observation axes follow the axis rule above: reduced unless also named in `group_by`. The field axis is kept.
- A value field varying along several axes, such as `x(time, sensor)`, is an error, for the same reason as a multi-axis `filter`: the groups would not form a rectangle.

This follows xarray's `groupby` with a one-dimensional array, which produces the same new axis of sorted unique values; use flox for performance. dplyr gives one row per group sorted by key. stars has no grouping by attribute values; its `aggregate` groups only by time intervals or geometries. In nuScenes, `ring` is stored as `float32`, so labels are `0.0, 1.0, …`; the single array cannot give `ring` an integer type, so documentation shows `variables_from="feature"` when integer labels are needed.

Decided: a summary that uses a variable lacking the grouping axis is an error. With `intensity(time, sensor)`, `gain(sensor)`, and `temperature(time)`:

```python
data >> group_by(sensor) >> summarise(m = mean(intensity * gain))   # fine: both have sensor
data >> group_by(sensor) >> summarise(t = mean(temperature))        # error
```

```text
SummariseError: temperature has no sensor axis, so it cannot be summarised by group_by(sensor).
  Its summary would be the same for every sensor.
  Summarise it separately without grouping: data >> summarise(t = mean(temperature))
```

The same applies to value grouping: after `group_by(ring)`, a variable without the `point` axis is an error. The check covers every data variable in a summary expression, including combinations such as `mean(intensity) / mean(temperature)`; Python constants and `env()` values are not data variables and are unaffected. Silently keeping the variable's natural axes or broadcasting it across groups were both rejected in favor of an explicit separate summary.

Decided: grouping into ranges uses a `bins()` helper inside `group_by`, following dplyr's `group_by(cut(x, breaks))` habit:

```python
cloud >> group_by(bins(distance, [0, 10, 20, 50])) >> summarise(m = mean(intensity))
# m (distance_bin: 3)   labels: [0, 10)  [10, 20)  [20, 50)

data >> group_by(bins(time, "2 days")) >> summarise(peak = max(x))
```

Rules:

- Bins are left-closed and right-open, as in stars' `aggregate`. Edges are a list of values, or a duration string for time axes and time-valued fields.
- The new axis is named `<field>_bin` and labelled with the intervals.
- Values outside every bin are an error, not silently dropped as in xarray's default; the error reports how many fell outside and suggests widening the edges.
- Otherwise the value-grouping rules apply: the binned field must vary along one observation axis, and variables without that axis are an error.

Implementation: xarray's `groupby_bins` for value edges and `resample` for time windows on a time axis. Precedents: stars' `aggregate(x, by = "2 days")` and `by =` break vectors; xarray's `groupby_bins` and `resample(time="2D")`.

Decided: summaries propagate missing values, as in dplyr and tidy3; `na_rm=True` skips them. tidy3 already uses this spelling and default: its `sum(x, *, na_rm=False, min_count=None)` "makes the result missing when any value is missing" (`expr.py`). This departs from xarray, whose float reductions skip missing values by default (`skipna=True`), so tidy3A passes `skipna=not na_rm` explicitly. Missing values arise from failed readings, `if_else` masking, and empty groups or axes.

- Empty groups or axes: `mean`, `min`, and `max` give missing; `sum` and counts give `0`, as in R. tidy3's `min_count` makes a sum missing when fewer than that many values are present; tidy3A adopts the same argument.
- `NaN` counts as missing, as in tidy3 ("NaN is NA", `helpers.py`).
- Integer arrays have no missing value, so a summary that comes out missing is promoted to `float`, following the write-back promotion rule (§5.4).
- stars leaves this to the aggregation function (`na.rm = TRUE` passed through `aggregate`). Grouped aggregate expressions inside `mutate` broadcast back onto the original observations.

```python
result = (
    tidy(readings)
    >> group_by("sensor")
    >> summarise(mean_intensity=mean(col("intensity")))
)
```

This is an axis-preserving implementation of familiar intent, not a request to materialize a long table. Grouping by axes, values, and bins, variables without the grouping axis, and missing values are decided above.

Decided: grouped summaries inside `mutate` broadcast back onto each observation, as in dplyr. The result keeps the original shape:

```python
cloud >> group_by(ring) >> mutate(centered = intensity - mean(intensity))
# centered (point): each point minus its own ring's mean
```

With axis grouping, `group_by(sensor)` means `mean(x)` is taken over every other observation axis and broadcast back along them. The `summarise` rules carry over: a variable without the grouping axis is an error, and missing values propagate unless `na_rm=True`. Implementation: xarray's `x - x.groupby(ring).mean()`, which broadcasts back through the group labels, or `groupby(...).map(...)`. stars has no grouped `mutate`.

After `summarise`, grouping follows tidy3's `summarise(groups=...)`: the default drops the last grouping level (`"drop_last"`), with `"drop"` and `"keep"` available (tidy3 `verbs.py`, `summarise`); tidy3's `"rowwise"` has no array meaning and is not carried over. `ungroup()` removes grouping explicitly. tidy3's per-call `by=` grouping could be added later in the same form.

Reuse `rename`, `arrange`, and other verbs only after their array semantics are needed and specified. In particular, sorting a coordinate axis must reorder every variable attached to it.

### 5.5 Alignment

Alignment decides what happens when arrays in one expression have different labels on the same axis.

Decided: the default is exact matching. Labels on a shared axis must match, or the expression is an error. The evaluator runs inside `xr.set_options(arithmetic_join="exact")`. xarray's own arithmetic default, an inner join, silently drops labels missing from either side; outer fills gaps with missing values and turns integer arrays into floats; left silently ignores mismatches. Exact matching protects the decision to keep original positions after `filter` (§5.3): combining filtered and unfiltered data can never silently pair or drop points.

```text
AlignError: point labels differ between intensity (12,034) and full_intensity (34,688).
  22,654 labels are only in full_intensity; 0 only in intensity.
  To keep shared points:   inner_join(full_intensity)
  To keep this data's:     left_join(full_intensity)
```

Decided: intentional alignment reuses dplyr's join verbs. A join brings another dataset into the pipeline, aligned on shared axes by label, and its variables are then used by name:

```python
nearby >> left_join(full_labels) >> mutate(diff = intensity - full_intensity)
```

| Verb | Keeps on each shared axis |
|---|---|
| `left_join(other)` | this data's labels; the other's extra labels are dropped, gaps become missing |
| `inner_join(other)` | only labels present in both |
| `full_join(other)` | all labels; gaps become missing |

Implementation: `xr.merge([data, other], join="left" | "inner" | "outer")`. Rule: expressions are always exact; only join verbs combine different labels. A join changes which labels exist, like `filter` or `slice`, so it is a visible pipeline step rather than part of an expression. The `AlignError` hints name these verbs, e.g. `left_join(full_intensity)`.

Decided: variables with the same name on both sides get tidy3's suffixes, `suffix=("_x", "_y")` by default, overridable with any pair of strings (tidy3 `verbs.py`, `_join_suffix`). This comes up constantly because every `read_array` result names its single array `values`: `nearby >> left_join(full)` gives `values_x` and `values_y`, and `suffix=("", "_full")` gives `values` and `values_full`. Both copies are kept under visibly different names, so nothing is resolved silently. stars agrees in spirit: its `c()` example output suffixes a duplicate name (`L7_ETMs.tif.1`). Implementation: `xr.merge` raises on same-named variables with different values rather than renaming, so tidy3A renames clashing variables before merging. Field axes survive the rename; field-axis selection then applies to both copies (§5.1).

Later additions, not decided: tidy3's `semi_join` and `anti_join` map naturally to arrays as keeping positions whose labels are, or are not, present in the other data.

Decided: when two operands have the same labels on an axis in a different order, they are matched by label and the second operand is reordered to the first operand's order. Reordering never drops or mispairs anything, so exact matching has nothing to protect against here. The inspection output reports it:

```text
changed: sensor reordered to match (rear front → front rear) in front_rear - rear_front
```

Implementation: xarray's `"exact"` join treats a different order as unequal, so before evaluation tidy3A checks each shared axis; if the label sets are equal but ordered differently, it reorders the second operand to match the first, then applies the exact check. Different label sets remain an `AlignError`.

Comparison with stars (checked against its reference pages): stars arithmetic (`Ops.stars`) uses positional R array recycling and asks users to reorder dimensions with `aperm` themselves; it does not align by label. Its `c()` requires the same dimensionality and compares dimension values with a numeric `tolerance` (default `sqrt(.Machine$double.eps)`); `try_hard = TRUE` combines only arrays matching the first. stars has no join verbs; tidy3A's joins come from tidy3 and dplyr.

Decided: there is no default tolerance. Decimal labels that differ only by rounding, for example grids built with `np.arange` and `np.linspace`, are an `AlignError` like any other mismatch, following xarray, whose alignment has no tolerance, rather than stars. The error recognizes the rounding case and names the fix:

```text
AlignError: lon labels differ between a and b, but only by up to 2.8e-14 (rounding).
  To match them within a tolerance:  left_join(b, tolerance=1e-9)
```

Join verbs accept an opt-in `tolerance=`, implemented with xarray's `reindex(method="nearest", tolerance=...)`. Expressions stay strictly exact. A default tolerance was rejected because, on very fine grids or timestamps stored as decimals, it would silently merge distinct positions.

Decided: when one operand's axis has labels and the other's does not, the expression is an error, even if the lengths match. When neither side has labels, operands pair by position, since that is the only possible meaning (for example, two arrays fresh from `read_array`). xarray matches an unindexed axis by length alone and stars always pairs by position; tidy3A departs from both, following §6.2's rule never to align axes solely because their lengths match. This makes the decision to keep original positions after `filter` (§5.3) effective when outside results, such as model predictions, are brought in:

```text
AlignError: point is labelled on intensity (original positions) but unlabelled on scores.
  Pairing by length could match the wrong points.
  If scores follows nearby's order, attach it by position:  nearby >> bind_cols(scores, along="point")
```

Decided: outside arrays are attached by position with `bind_cols`, and an axis's labels are read with `pull`; there is no accessor method, which would break tidy3A's verbs-and-pipes style.

```python
nearby >> bind_cols(scores, along="point")   # scores becomes a variable on point, taking nearby's labels
nearby >> pull(point)                        # labels on the point axis, as a NumPy array
nearby >> pull(intensity)                    # a variable's values, as in tidy3
```

- `bind_cols` pairs by position because the user asked for it by choosing the verb, unlike alignment, which never pairs by length. Lengths must match exactly, or it is an error. `along=` names the axis; tidy3's `bind_cols` (`verbs.py`) combines frames column-wise by position, as in dplyr.
- `pull(point)` uses the rule that an observation axis's name refers to its labels (`filter(point < 100)` compares labels). For an unlabeled axis it returns positions `1 … N`. tidy3's `pull(var)` returns a column's values; stars' `pull` returns an attribute as a plain array; xarray's equivalent is `.coords["point"].values`.

### 5.6 Notebook layer

tidy3's notebook rewrite classifies calls by verb name alone: `filter`, `select`, `mutate`, `group_by`, and the rest are fixed sets of strings in `masking.py`, and `Tidy3MaskTransformer.visit_Call` rewrites any call with one of those names. tidy3A reuses the same names, so in a notebook that imports both packages, tidy3's transformer would rewrite tidy3A calls with tidy3's references (`col`, `__tidy3_name__`), and two transformers could rewrite the same cell in an order set by imports.

Decided: each package's transformer rewrites only calls whose function it owns, checked in the live notebook namespace when the cell is transformed. After `from tidy3a import *`, `filter` is tidy3A's, so tidy3's transformer leaves it alone, and the reverse. Details:

- Ownership uses a provenance check like tidy3's existing `_is_tidy3_owned` (`jupyter.py`).
- Prefixed calls such as `ta.filter(...)` and `t3.filter(...)` are unambiguous and handled the same way.
- The cell that performs the import follows tidy3's plot3 precedent (`masking.py`, `_plot3_masking_for_importing_cell`): masking is applied to that cell explicitly, and the package's transformer is turned on for later cells.
- A shared rewrite package was rejected for v1 because it needs its own maintainer; disabling one package's rewrite when the other is imported was rejected because it breaks that package's pipelines.

Decided: tidy3A copies and adapts the parts of tidy3's `masking.py` it needs into its own `jupyter.py`, with a note crediting tidy3, rather than depending on tidy3. Reasons: tidy3A must not require tidy3 (§9); the code differs anyway (tidy3A's verb list, the error on name collisions instead of tidy3's data-wins rule, the `[x, y, z] = …` preparser, and its own ownership check); and the roughly 1,200-line module is self-contained enough to own. tidy3A's maintainers own the copy, which satisfies §6.1's requirement of an identified maintenance owner. Once both copies have settled, compare them; if they remain close, extracting a small shared package is still possible.

Required tidy3 change (prerequisite for milestone 5): `Tidy3MaskTransformer` must check that a call's function is tidy3-owned in the live namespace before rewriting, instead of matching the verb name alone. This is a separate change in the tidy3 repository with its own tests, including a notebook that imports both packages.

## 6. Mathematical operations

### 6.1 One expression system

Use an expression representation that handles arithmetic, comparisons, reductions, indexing, external arrays/scalars, broadcasting, and generalized dot products. Both ordinary syntax and `formula()` must compile into that representation and use the same evaluator.

This expression system is the array layer of §1.1. Its helpers follow one axis convention, `over=`, naming the axes that are summed or reduced:

```python
dot(R, [x, y, z], over="feature")   # sum R × [x, y, z] over feature
sum(p**2, over="i")             # reduce p over i
mean(intensity, over="time")    # reduce intensity over time
```

`over=` accepts one axis name or a list. Every axis not named in `over=` stays in the result, so batch axes such as `time` are kept without special rules. `over=` may be omitted only when the choice is unambiguous: for `dot`, when the inputs share exactly one axis; for a reduction, when the input has exactly one axis. Otherwise omitting it is an error that lists the candidate axes. Inside `summarise` after `group_by`, the tidy layer's grouping rule (§5.4) decides which axes are reduced; an explicit `over=` there must not contradict the grouping.

Do not carry over Polars-specific compilation as the core engine. Inspect and reuse small independent parts of tidy3/plot3 where practical, preserving their license obligations. Do not import plot rendering machinery or the full tidy3 dependency tree solely for parsing or notebook shorthand. Avoid copying an entire parser without an identified maintenance owner.

### 6.2 Dot

`dot(A, x, over="feature")` multiplies matching components and sums them over the named axes. This follows xarray's generalized dot product model (`xr.dot(..., dim=...)`). `over=` is the explicit-axis option, following the §6.1 convention. It belongs to the array layer and does not imply a `dim` argument on the verbs.

Define the rules precisely:

- Contraction sums over the `over=` axes; every other axis remains in the result. Without `over=`, the inputs must share exactly one axis, which is then summed.
- Inputs and outputs of a matrix use different axis names when they represent different roles.
- Validate shared coordinate labels. The recommended initial policy is exact alignment, permitting explicit/reported reordering but never silent dropping of unmatched coordinates.
- Do not align axes solely because their lengths happen to match.
- Decided: `dot` never conjugates either input; inner products write `conj()` explicitly:

  ```python
  mutate(out = dot(gate, amplitude, over="input"))       # operator application: no conjugate
  mutate(overlap = dot(conj(a), b, over="state"))        # inner product
  ```

  This matches xarray's `xr.dot`, NumPy's `np.dot`, and Tullio/einsum, and keeps conjugation visible where it happens; real-valued data never needs `conj`. Julia's `LinearAlgebra.dot` and MATLAB's `dot` conjugate their first argument, and NumPy's `vdot` does too; the documentation states the difference in one line. A separate `inner()` and a `conj=` flag were rejected. The complex-array acceptance test checks both cases.
- Axes not named in `over=` remain attached and broadcast according to named-axis rules. If both inputs share a batch axis such as `time`, `dot(A, x, over="feature")` keeps `time`; without `over=` the call is an error, because two axes are shared. Equivalent formula indices also retain `time`. The package never infers which shared axes are batch axes.

Decided: `rename_axes(expr, old="new")` renames axes inside expressions, so results can be chained or compared on the axis they started from:

```python
mutate(amp1 = dot(gate, amplitude, over="input"))                          # amp1 is (output)
mutate(amp2 = dot(gate, rename_axes(amp1, output="input"), over="input"))  # chain a second gate
```

Rules: labels move with the renamed axis, and the data's shape and values are unchanged; renaming onto a name the expression already has is an error. It is an array-layer helper, usable on any expression, and matches xarray's `.rename({...})` and Haliax's `rename`; stars uses `st_set_dimensions(names=)`. It is named `rename_axes` to stay distinct from tidy3's `rename`, which renames variables, and has nothing to do with pandas' `rename_axis`. An option on `dot` to name its output axis was rejected because only `dot` would have it. The rotation example does not need it, since the `[x, y, z] = …` assignment maps `i` onto `feature` by label (§5.4), and formula index notation such as `\sum_j G_{ij} a_j` names output indices directly.

Test both eager and supported lazy behavior. Avoid importing a second computational framework merely for matrix multiplication.

### 6.3 Formula

Both forms are desired:

```python
formula("x^2")
formula(r"\sum_i x_i^2")
```

Inside formulas, `^` means exponentiation. Ordinary Python expressions continue using `**`. Strings passed directly to `mutate` remain text literals; `formula` marks a string as mathematics.

Initial grammar: arithmetic, powers, parentheses, square roots, fractions, documented numeric functions, and a single bounded sum with indexed array access. Extend to multiple sums and matrix-index expressions after the basic binding tests pass.

LaTeX does not define meaning, so formula follows the fixed conventions plot3 already uses (`plot3/latexin.py`): `\sin x^2` is `sin(x**2)`, `\sin^2 x` is `(sin x)**2`, `\sin^{-1}` is `arcsin`, and `e^{x}` is `exp(x)`. Keeping the conventions identical means the same LaTeX means the same thing in both packages.

Decided: letters written side by side multiply, as in plot3 and ordinary mathematical typesetting: in LaTeX, `xy` is `x·y`. Multi-letter names are written `\mathrm{...}`, which plot3 already recognizes:

```latex
\frac{\mathrm{intensity}}{\mathrm{distance}^2}
```

One rule is added for tidy3A's named data: if a run of letters matches a data or notebook name, such as `xy` when the data has a field `xy`, it is an error that asks for `\mathrm{xy}` or `x \cdot y`, reusing the name-collision rule of §5.2. Plain text keeps plot3's behavior: `xy` there is an error suggesting `x*y`. The interpretation shown back to the user displays the multiplication explicitly.

Decided: a subscript on a data variable refers to that variable's axes, matched by name, never by position. This differs from plot3, where `x_1` is the name `x_1`; array index notation requires it.

| Written | Means | Example |
|---|---|---|
| `p_i` | `p` indexed along its axis `i` | `\sum_i p_i^2` sums over `i` |
| `p_x` | the label `x` on one of `p`'s axes | `p_x^2 + p_y^2`, horizontal distance |
| `p_1` | 1-based position 1, only if `p` has exactly one axis besides those already free in the pipeline | otherwise an error asking for a label or an index |

Rules:

- Axes that are not subscripted, such as `point`, stay free: `\sum_i p_i^2` gives `distance(point)`.
- A subscript that names neither an axis nor a label of that variable is an error listing what the variable has, for example axes `point, i` with labels `x y z` on `i`.
- A data variable whose name contains an underscore, such as `x_1`, is written `\mathrm{x\_1}`, with LaTeX's escaped underscore. Documentation notes the difference from plot3.
- The interpretation shown back displays every binding, for example `i → axis "i" of p (labels x y z), summed`.

Decided: index letters that do not match an axis name are bound explicitly with `bind=`; unbound letters must match axis names. Binding is by name, never by position:

```python
formula(r"p_i = \sum_j R_{ij} v_j", bind={"j": "feature"})
# i matches R's axis i by name; j → axis "feature" (bound), summed
```

A letter that matches no axis and has no binding is an error naming the axes it could be. einsum and Tullio bind letters by position (`R[i,j]` takes the stored axis order); that was rejected because it contradicts name matching and xarray axis order changes with transposes and imports. Requiring index letters to equal axis names, such as `\sum_{\mathrm{feature}}`, was rejected because formulas could then not follow a paper's letters.

Decided with this rule:

- Inside a subscript, letters side by side are separate indices, standard tensor notation: `R_{ij}` has two indices, not `i·j`. This is an exception to the multiplication rule above. `R_{i,j}` is also accepted.
- `bind=` also supplies values for names in the formula, because `[x, y, z]` cannot be written in LaTeX: `bind={"j": "feature", "v": [x, y, z]}`. There is no second argument. A key that is an index letter binds an axis; any other key supplies a value.

Decided: `\sum_i` with no bounds sums over the whole axis `i` and is the main form; bounds are an optional, checked, 1-based slice.

```latex
\sqrt{\sum_i p_i^2}          % whole axis i
\sum_{i=1}^{2} p_i^2         % positions 1 to 2 of i, inclusive
```

- Numeric bounds are 1-based inclusive positions, consistent with `slice` (§5.3). Bounds outside the axis are an error. Bounds covering part of the axis are allowed, and the interpretation shown back says so: `i: positions 1–2 of 3`.
- Symbolic bounds such as `n` must come from `bind=` or a notebook value; `n` is never guessed to be the axis length. Without a binding, the error suggests `\sum_i` for the whole axis.
- Bounds are positions, never coordinate labels.

This follows Einstein notation and Tullio, where an index runs over its full range unless told otherwise, and avoids bounds silently going stale when an axis changes length.

Decided: plain `i` is never the imaginary unit; it is always an index (`\sum_i`, `p_i`) or an ordinary name. The imaginary unit is `\mathrm{i}`, the upright form recommended by ISO 80000-2 and common in physics typesetting; `\mathrm{j}` is accepted for engineering notation. The meaning of `i` therefore depends on how it is written, not on context, and the interpretation shown back states it (`\mathrm{i} → imaginary unit`). plot3 has no imaginary unit because it plots real functions.

```latex
\mathrm{e}^{\mathrm{i}\theta}
```

Decided: `e` follows plot3, which treats it as Euler's number unless a number named `e` is passed in (`plot3/expr.py`, `e_is_constant`); in tidy3A, `e` that is also a data or notebook name is an error under the collision rule (§5.2), and `\mathrm{e}` is always the constant.

Index binding must be part of the grammar, not an afterthought:

- A summation index names a matching array axis, or uses an explicit binding when names differ (decided above: `bind=`).
- Other axes, such as `point`, remain free/batched axes.
- Bounded integer indices are inclusive 1-based positions, converted to backend positions, and never coordinate labels (decided below).
- Symbolic bounds such as `n` must be bound explicitly; they are never guessed (decided below).
- Distinguish array subscripts from actual variable names containing underscores (decided above: subscripts are indices; underscore names use `\mathrm{x\_1}`).
- Unsupported or ambiguous notation must produce a useful parse/binding error.

Decided: every formula shows how it was read, its bindings, and its compiled array code:

```text
formula: √( Σ_i p_i² )
  p → variable p (point, i)
  i → axis "i" of p (labels x y z), summed
  result: distance (point)
  as array code: sqrt(sum(p**2, over="i"))
```

- It is the formula's own representation: LaTeX output in Jupyter, the text form in scripts and `repr()`. `.explain()` is an alias for the text form.
- It appears in the pipeline's inspection output for any `mutate` that used a formula.
- It is not printed while a pipeline runs, so batch runs stay quiet.
- On its own, a formula shows its parse and index letters; bindings that need data, such as axis labels and result shape, are shown once it is attached to a pipeline.

This borrows the code-to-LaTeX direction that has worked in Python (latexify_py, handcalcs; recalled, not checked) to verify the riskier LaTeX-to-code direction. An opt-in-only `.explain()` was rejected because a display most users never see cannot catch a misreading.

Use a restricted parser and supported operation set rather than unrestricted evaluation. Rendering a formula is not evidence that it has executable semantics. plot3's LaTeX input is a useful precedent, but its current parser explicitly rejects `\sum`; its handling of subscripts also needs review for array indexing.

## 7. Reference example: import a sweep and retain points within 50 metres

This is the target integration example. The entire tidy3A portion is illustrative until implemented. It uses the nuScenes SDK only to locate the file and its calibration, not as the numerical execution backend. Keep it outside the core dependency/test requirements.

### A. Find a real file and metadata

```python
from pathlib import Path
import numpy as np
import xarray as xr
from nuscenes.nuscenes import NuScenes
from tidy3a import *

nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes")
sample = nusc.sample[0]
scan = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
calib = nusc.get("calibrated_sensor", scan["calibrated_sensor_token"])
path = Path(nusc.dataroot) / scan["filename"]
```

### B. Import through the generic reader

```python
cloud = read_array(
    path,
    value_type="float32",
    dims={"point": infer, "feature": 5},
    labels={"feature": ["x", "y", "z", "intensity", "ring"]},
    field_axis="feature",
)
```

The five labels come from the file documentation. The reader contains no nuScenes-specific interpretation.

### C. Select the components using existing verbs

```python
cloud >> select(x, y, z)            # notebook
cloud >> select("x", "y", "z")      # script
```

The result is one array with axes `point, feature`, containing exactly three feature labels. The original `cloud` still contains intensity and ring; `select` returns a selected result without mutating its source.

The transform in step E does not need this step or a collected `xyz` array. It refers to the components inside the pipeline with the vector form `[x, y, z]` (§5.2), which keeps them aligned with the points that remain after any earlier `filter`.

### D. Prepare a generic matrix and offset

Quaternion conversion can be a small example-local function. It does not need to become a core spatial verb:

```python
def rotation_from_wxyz(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

axes = ["x", "y", "z"]
R = xr.DataArray(
    rotation_from_wxyz(calib["rotation"]),
    dims=("i", "feature"),
    coords={"i": axes, "feature": axes},
)
offset = xr.DataArray(calib["translation"], dims="i", coords={"i": axes})
```

This example assumes the documented valid calibration quaternion. If the conversion is promoted to a public utility, validate finite components and nonzero norm and document quaternion ordering. A general rotation matrix helper is optional; quaternion import is not a prerequisite for the array core.

### E. Transform, calculate distance, and filter

Notebook form:

```python
nearby = (
    cloud
    >> mutate(p = dot(R, [x, y, z], over="feature") + offset)
    >> mutate(distance = formula(r"\sqrt{\sum_i p_i^2}"))
    >> filter(distance <= 50)
)
```

Script form, declaring the field names once:

```python
x, y, z = fields("x", "y", "z")
nearby = (
    cloud
    >> mutate(p=dot(R, [x, y, z], over="feature") + offset)
    >> mutate(distance=formula(r"\sqrt{\sum_i p_i^2}"))
    >> filter(col("distance") <= 50)
)
```

`R` and `offset` are notebook/Python values and `x`, `y`, `z` are data fields, so no name collides (§5.2). `[x, y, z]` is the vector view `(point, feature: 3)`. The dot product sums over `feature`. Its output `p` has axes `point, i`. The formula sums the three position components on `i`; its output is `distance(point)`. Filtering that one-dimensional predicate keeps every variable aligned to the retained point identifiers, including the original intensity/ring measurements.

Notebook form can use `filter(distance <= 50)`; scripts use `col` explicitly. The examples do not require bare field labels to become actual Dataset variables automatically.

This is a 3D spherical distance from the nuScenes vehicle-frame origin. That origin must not be assumed to be the exact geometric center of the car. A geometric-center target needs its own known offset. Horizontal range would instead use the two horizontal components. This example uses one sweep; combining sweeps also requires pose/time alignment, which is outside this first example.

## 8. Second acceptance example: complex arrays

Use a small state array and an operator with named input/output axes. Verify that the same engine works without coordinate transforms or autonomous-vehicle helpers.

```python
state = xr.Dataset({
    "amplitude": xr.DataArray(
        [1 + 0j, 0 + 0j], dims="input", coords={"input": [0, 1]},
    ),
})
gate = xr.DataArray(
    [[0, 1], [1, 0]], dims=("output", "input"),
    coords={"output": [0, 1], "input": [0, 1]},
)
result = tidy(state) >> mutate(amplitude_out=dot(gate, col("amplitude"), over="input"))
```

Expected output amplitudes are `[0, 1]` on `output`. Also test a genuinely complex input, preservation of norm under a unitary operator, and application along one axis of a two-qubit tensor while retaining its other axis. Do not treat a spatial translation as a generic quantum operation or claim this constitutes a quantum simulator.

## 9. Repository scaffold

Suggested initial layout:

```text
tidy3a/
  pyproject.toml
  README.md
  LICENSE
  CHANGELOG.md
  src/tidy3a/
    __init__.py
    frame.py          # TidyArray, pipes, grouping, inspection/conversion
    io.py             # read_array, infer, import metadata
    expr.py           # expressions, references, arithmetic/reductions
    verbs.py          # familiar verbs using xarray semantics
    linalg.py         # generalized dot and complex arithmetic policy
    formula.py        # math/LaTeX parsing and index binding
    jupyter.py        # optional notebook integration
    py.typed
  tests/
    test_io.py
    test_selection.py
    test_verbs.py
    test_linalg.py
    test_formula.py
    test_notebooks.py
    test_examples.py
    data/             # only small, redistributable fixtures
  examples/
    point_cloud.py
    sensor_time_series.py
    complex_arrays.py
  docs/
    data_model.md
    api.md
    decisions.md
  .github/workflows/
    tests.yml
    publish.yml
```

Use NumPy and xarray as core dependencies. Choose a supported Python minimum and dependency floors together rather than copying tidy3's version constraints unchanged. Keep IPython, Dask, format engines, and nuScenes tooling optional. Do not require tidy3, plot3, stats3, or the nuScenes SDK to import tidy3A or run its core suite.

Start with one eager NumPy/xarray backend. Preserve existing supported lazy arrays when wrapping them, and make materialization boundaries explicit; implement and test broader Dask behavior as a separate milestone. Do not promise Polars-style query optimization for an xarray wrapper.

## 10. Implementation milestones

### Milestone 0 — Settle the contract with small prototypes

Write `docs/decisions.md` covering selection/field resolution, grouping/reduction, multi-axis filtering, alignment, collection return type, and formula index binding. Run tiny xarray-only experiments for each. Add a few expected-input/output examples before implementing notebook syntax. Write the point-cloud and time-series examples in both field-axis style (`select(x, y, z)`) and stars-style strict select (`filter(feature.is_in([...]))`), and confirm the field axis reads better before keeping it (§5.1).

Exit: the reviewer can trace the point cloud and time-series examples without wondering whether a name means a variable, axis, label, or external Python object. No large shared-framework refactor is required to begin.

### Milestone 1 — Create the repository and import foundation

Create the separate repository/package scaffold, installation configuration, license, tests, and independent CI. Implement wrapping existing xarray objects, generic raw numeric/`.npy` reads, `infer`, naming/labeling, inspection, and conversion back to xarray.

Exit: 1D through 5D arrays, malformed record lengths, labels, complex types, and empty observations have defined behavior. An installed wheel can import the package without optional integrations.

### Milestone 2 — Familiar manipulation verbs

Implement selection, expression references, filtering a single observation axis, mutation, simple grouping, summaries, and ungrouping. Start with script syntax. Implement named-coordinate alignment and broadcasting without converting full arrays into tables.

Exit: the XYZ selection and time-series mean examples work; filtering preserves point-associated values; grouping identifies what remains separate; ambiguous references fail clearly.

### Milestone 3 — Generalized array mathematics

Implement dot, reductions, elementary numeric functions, comparisons, explicit conjugation, and broadcasting rules. Test general array rank, reordered axes, coordinate mismatches, and complex arrays.

Exit: the point-cloud rotation works against an independent NumPy calculation, and the complex-array example works through the same evaluator. No spatial verb is necessary.

### Milestone 4 — Mathematical notation

Implement plain mathematical text first, then the scoped LaTeX subset. Extend supported index expressions and sums incrementally. Add readable mathematical representations while retaining the actual parsed/bound operation for inspection.

Exit: `formula("x^2")` matches `col("x") ** 2`; an indexed sum matches the native reduction; the distance formula works; ambiguity and unsupported syntax produce precise errors. Ordinary text literals are unaffected.

### Milestone 5 — Notebook experience and handoff examples

Prerequisite: the tidy3 change in §5.6, so tidy3's transformer rewrites only tidy3-owned calls. Add optional bare-name support using a small, clearly owned transformer/integration layer that rewrites only tidy3A-owned calls. Review tidy3/plot3 behavior for collaboration opportunities without importing their execution backends. Test real IPython cells, import order, external constants/operators, and explicit `col`/environment escapes.

Exit: the same documented examples run in scripts and notebooks, including the first cell after enabling shorthand, and in a notebook that imports both tidy3 and tidy3A in either order. A small synthetic binary fixture covers the complete cloud workflow in CI. A manually run nuScenes Mini example validates metadata lookup and calibration conventions separately.

### Milestone 6 — Additional readers and lazy execution (after the first release)

Decided: this milestone follows the first release (milestone 7), which ships after milestone 5. Add generic mixed fixed-width records and optional self-describing formats. Define metadata/casting conflict policies. Add Dask-focused tests and performance checks only for behavior the package actually promises. Discuss rolling, transforms, and other helpers after concrete examples demonstrate the need.

Exit: the reader handles metadata-rich input without redundant declarations and mixed typed records without coercing everything into one numeric array. Lazy operations avoid unintended full-data computation; unavoidable materialization is documented.

### Milestone 7 — Independent first release

Build distributions, test installation in clean environments, validate public examples, document limitations, and publish through tidy3A's own workflow. Release only capabilities completed by the earlier milestones; remaining readers or lazy features can ship later.

Decided first-release scope:

| In the first release | Later releases (milestone 6) |
|---|---|
| `read_array` for raw bytes and `.npy` | `read_binary` for mixed fixed-width records (§4.3) |
| `lazy=True` memory-mapping for large raw files | Dask-backed lazy operations, tested for what is promised |
| `tidy()` wrapping xarray objects, including lazy NetCDF/Zarr datasets, which stay lazy until an operation needs values | Dedicated NetCDF/Zarr readers, if `tidy(xr.open_dataset(...))` proves insufficient |
| All verbs and the array layer, running eagerly on NumPy | Performance benchmarks beyond the acceptance examples |

Documentation states where a lazy input is loaded into memory. Unlike tidy3, which is lazy-first through Polars, the first release is eager; the README says so. Precedents: stars reads many formats through GDAL and offers `stars_proxy` for out-of-memory data; xarray's `open_dataset` is lazy and chunks with Dask.

## 11. Verification and release strategy

Prioritize meaningful checks rather than copying tidy3's entire test matrix:

- Raw import round-trips and precise layout/label validation.
- Results matching direct NumPy/xarray calculations for independent numerical checks.
- Axis permutation and coordinate-label tests, including exact alignment failures.
- Point association surviving filtering and arithmetic; untouched intensity/ring values.
- A synthetic known rotation/translation with points below, at, and above the 50 m boundary.
- Grouped temporal summaries matching explicit expected values.
- Complex operator application without accidental conjugation or real-only coercion.
- Formula/native-expression agreement and clear index-binding failures.
- Installed-package smoke tests and notebook/script behavior.

Run fast unit and synthetic integration tests on pull requests. Keep large-data benchmarks, optional-format checks, and downloaded dataset exercises in separate jobs. Cache dependencies and record timings. Use one publication gate for the tested commit/artifacts, a focused installation smoke check, and trusted publishing where configured. Do not rerun unrelated tidy3, plot3, or stats3 suites for every tidy3A release.

The current approximately 20-minute tidy3 publication time is user-reported context, not a diagnosed bottleneck. Measure tidy3A's build/test times before choosing performance targets. Separating packages isolates release cycles; it does not by itself fix slow jobs.

## 12. Questions for the reviewer to resolve first

All seven questions are resolved; the decisions are in the sections cited and summarized in §2.

1. Resolved: selection uses the field-axis rule, with `var()`/`label()` for collisions and tidy3's selection helpers (§5.1); milestone 0 still tests field-axis select against stars-style strict select.
2. Resolved: `group_by` axes are kept and other observation axes reduced; grouping by values and `bins()`; a variable lacking the grouping axis is an error (§5.4).
3. Resolved: exact matching with reorder by label; joins via `left_join`/`inner_join`/`full_join`; `over=` names summed axes, so batch axes are kept (§5.5, §6.1–6.2).
4. Resolved: `read_array` returns the wrapper directly; `tidy()` wraps existing xarray objects (§4.1).
5. Resolved: plot3's conventions, subscripts as indices, `bind=`, `\sum_i` as the main form, `\mathrm{i}`, and required show-back (§6.3).
6. Resolved: tidy3A copies and adapts tidy3's `masking.py` and owns the copy; each transformer rewrites only calls it owns, which needs a tidy3 change (§5.6).
7. Resolved: first-release scope in milestone 7 (§10).

Parameter spelling, extra aliases, and additional domain helpers can wait until these examples are understood and pass their acceptance checks.

## 13. Prior art

Earlier attempts to make array programming friendlier fall into five groups. Items marked † were recalled rather than checked against a source during this review; verify them before citing.

| Group | Projects | Lesson for tidy3A |
|---|---|---|
| Tidy verbs on arrays | **stars** (R): dplyr verbs `filter`, `mutate`, `select`, `rename`, `pull`, `slice(along=)` on spatiotemporal arrays; `split`/`merge` between dimensions and variables†. **tbl_cube/cubelyr** (R): dplyr's own array type, moved out of dplyr at 1.0 for low use†. **tidyterra**† and **tidync**† (R) | Layers built for one data community (rasters) succeeded; the general-purpose one did not. Use stars as the reference design for remaining verb questions (slicing by position, grouping over dimensions) |
| dplyr ports in Python | dppd, pipda, pydplyr, siuba, dfply, plydata, dplython | All target pandas; no dplyr-style layer for xarray was found, so the gap is real. pipda's per-type verb registration is a pattern to consider |
| Named axes | **xarray**; **Haliax** (JAX, after Rush's *Tensor Considered Harmful*); PyTorch named tensors (remained a prototype†); DimensionalData.jl, AxisKeys.jl, NamedDims.jl†; **rray** (R broadcasting, stopped at 0.1.0) | Named axes are widely seen as the friendlier model; verb layers on top of them are rare |
| Index notation | **Tullio.jl** (`@tullio A[i,j] += B[i,k,l] * C[l,j]`), Einsum.jl, OMEinsum.jl; einops† (Python), TensorCast.jl† | Friendly array math names its indices explicitly; this supports the array layer's `over=` and formula index conventions. Tullio is the closest precedent for `formula()` |
| Array query languages | kdb+/q† (`select … by … from` in an array language), rasdaman/rasql†, SciDB†, ISO SQL/MDA†; pandas `Panel`, removed in favor of xarray† | Query verbs and array programming can coexist; forcing N-dimensional data into a table-shaped API did not last |

Implications carried into this plan: the two-layer split (§1.1), the `over=` convention (§6.1), stars as the reference for open verb questions, and choosing a home community, such as sensor and point-cloud data, for examples and outreach while keeping the core general.

Sources checked: [stars dplyr methods](https://search.r-project.org/CRAN/refmans/stars/html/dplyr.html), [stars tidyverse vignette](https://r-spatial.github.io/stars/articles/stars3.html), [stars arithmetic](https://r-spatial.github.io/stars/reference/ops_stars.html), [stars c()](https://r-spatial.github.io/stars/reference/c.stars.html), [dppd](https://dppd.readthedocs.io/), [pipda](https://pypi.org/project/pipda/), [pydplyr](https://pypi.org/project/pydplyr), [rray](https://rray.r-lib.org/), [Tullio.jl](https://docs.juliahub.com/Tullio/PIgzC/0.2.9), [Julia einsum comparison](https://github.com/JuliaPackageComparisons/JuliaPackageComparisons.github.io/blob/main/docs/comparisons/math/einsum.md), [Haliax](https://pypi.org/project/marin-haliax/0.2.5.dev202606020954/).

## 14. Reference material

- [xarray data model](https://docs.xarray.dev/en/stable/user-guide/data-structures.html)
- [xarray computation, alignment, and broadcasting](https://docs.xarray.dev/en/stable/user-guide/computation.html)
- [xarray generalized dot product](https://docs.xarray.dev/en/stable/generated/xarray.dot.html)
- [xarray label selection](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.sel.html)
- [nuScenes metadata and calibration schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)
- [nuScenes point-cloud reader and transforms](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)
- [Julia inferred reshape dimension](https://docs.julialang.org/en/v1/base/arrays/#Base.reshape)

Local implementation references for the first reviewer: tidy3's `src/tidy3/frame.py`, `expr.py`, `verbs.py`, `io.py`, `masking.py`, and `jupyter.py`; plot3's `plot3/latexin.py`, `plot3/expr.py`, and `tests/test_latex_input.py`. Read these as implementation precedents, not as requirements to depend on those entire packages.
