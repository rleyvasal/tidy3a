# tidy3a tidy-layer rules

The one-page rule set from plan §1.1: if the tidy layer's rules stop fitting on this page, simplify the layer rather than adding rules. Details, reasons, and precedents are in `docs/plan.md`, with section references below.

**Data.** A dataset holds variables (named arrays). Each axis is either the one **field axis**, declared with `field_axis=`, whose labels act as column names, or an **observation axis**, whose labels are values. Without `field_axis=` there are only variables. `variables_from=` turns a labeled axis into one variable per label instead. (§3, §4.1)

**Names.** In verbs and expressions, a name means, in order: a variable, then a field-axis label. A field label alone, `x`, is that field with the field axis dropped; a list `[x, y, z]` keeps the field axis, narrowed, and also rebuilds variables made by the same `variables_from=` import into their recorded axis. Other variables are packed with `pack([...], along=...)`. A name on an observation axis means that axis's labels (`filter(time > t0)`). A name that is both data and a notebook value is an error: write `col("x")` or `env("x")`. A name that is both a variable and a field label is an error: write `variable("x")` or `label("x")`. Scripts use strings or `fields("x", …)`. (§5.1, §5.2)

**Verbs never reshape on their own.** `select` keeps variables and field labels (columns), with tidy3's selection helpers. `filter` keeps positions on observation axes (rows); every predicate must vary along one axis, and predicates on different axes combine with AND into a rectangle. `slice` keeps positions by 1-based position and keeps the axis unless `drop=True`. Filtering keeps original positions as labels. Empty results are returned. (§5.1, §5.3)

**Mutate.** Assignments run in order. Writing to a field label writes into the shared array, promoting its type if needed. Vectors are written in one assignment, `[x, y, z] = …`. Masking is `if_else`. (§5.4)

**Grouping.** `group_by` on an axis keeps it; on a value field, or `bins(...)` of one, it groups that field's one axis into a new axis of sorted values or intervals. `summarise` reduces every other observation axis and never the field axis. A summary using a variable without the grouping axis is an error. Grouped summaries inside `mutate` broadcast back. Missing values propagate unless `na_rm=True`. After `summarise`, the last grouping level is dropped (`groups=`). (§5.4)

**Alignment.** Labels on a shared axis must match exactly; the same labels in another order are reordered and reported. Different labels need a join verb (`left_join`, `inner_join`, `full_join`, with `_x`/`_y` suffixes; `tolerance=` is currently supported only on `left_join`). A labeled axis never pairs with an unlabeled one by length; attach outside arrays with `bind_cols(…, along=)`, and read labels with `pull(axis)`. (§5.5)

**Array layer.** Maths helpers name their axes: `dot(…, over=)`, `sum(…, over=)`, `rename_axes(…)`; `dot` never conjugates. `formula()` reads a strict LaTeX dialect into the same expressions, binds index letters to axes by name or `bind=`, and always shows how it was read. (§6)

**Results.** `read_array` and `tidy()` return the wrapper. `collect()` returns a computed xarray Dataset; `to_xarray()` returns the underlying xarray object. (§3, §4.1)
