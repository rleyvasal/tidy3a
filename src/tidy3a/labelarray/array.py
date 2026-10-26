"""The labelled array (plan milestone 10, track A, step A5).

An ``Array`` is a NumPy array with one name per axis, at most one label
array per axis, attributes, and a name. Its labels are its own: labels from
outside are copied once into a read-only ``_Labels`` array, which nothing
else can unlock, so each label array's index (``labelarray.index``) is built
on first use and kept on the array, released with it. Labels compare under
one exact rule (``labelarray.labels``). Values are not copied or frozen:
wrapping an array shares it.

NumPy's ufuncs, ``np.where``, and Python's operators broadcast by axis name
under the exact alignment rule (``labelarray.align``). Selection by
positions or labels, transposing, renaming axes, reductions (float32 sums
and means correctly rounded, ``labelarray.sums``), and contraction
(``dot``) are methods; ``concat``, ``align``, and ``broadcast`` work on
several arrays.

This module needs NumPy only.
"""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from tidy3a.labelarray import sums
from tidy3a.labelarray.align import order_like
from tidy3a.labelarray.contract import contract
from tidy3a.labelarray.errors import AlignError, LabelError, ShapeError
from tidy3a.labelarray.index import LabelIndex, index_for


class Array:
    """A NumPy array with one name per axis, labels for its labelled axes, attributes, and a name.

    NumPy's ufuncs, ``np.where``, and Python's operators work on it, broadcasting by axis name under
    the exact alignment rule (``labelarray.align``): the same labels in another order are matched by
    label, and labels that differ are an ``AlignError``. Results have no attributes or name. Python's
    operators do not warn about division by zero or overflow, as on xarray arrays; NumPy's functions do.
    """

    __slots__ = ("attrs", "coords", "data", "dims", "name")

    __hash__ = None  # type: ignore[assignment]  # == compares values, as on NumPy and xarray arrays

    def __init__(self, data: Any, dims: Sequence[Hashable] = (), coords: Mapping[Hashable, Any] | None = None,
                 attrs: Mapping[Hashable, Any] | None = None, name: Hashable | None = None):
        data = data if isinstance(data, np.ndarray) else np.asarray(data)
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != data.ndim:
            raise ShapeError(f"{len(dims)} axis names {dims} for an array of rank {data.ndim}")
        if len(set(dims)) != len(dims):
            raise ShapeError(f"axis names must be unique, got {dims}")
        self.data, self.dims, self.name = data, dims, name
        self.coords = _checked_coords(coords or {}, dict(zip(dims, data.shape)))
        self.attrs = dict(attrs or {})

    @classmethod
    def _make(cls, data: np.ndarray, dims: tuple, coords: dict, attrs: dict | None = None,
              name: Hashable | None = None) -> Array:
        """Build without checks, from parts already validated, with labels this library froze."""
        array = object.__new__(cls)
        array.data, array.dims, array.coords = data, dims, coords
        array.attrs, array.name = attrs if attrs is not None else {}, name
        return array

    @property
    def sizes(self) -> dict[Hashable, int]:
        return dict(zip(self.dims, self.data.shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def values(self) -> np.ndarray:
        return self.data

    def axis(self, dim: Hashable) -> int:
        return self.dims.index(dim)

    def labels(self, dim: Hashable) -> np.ndarray:
        """The labels of *dim* as a plain, read-only NumPy array."""
        return plain(self.coords[dim])

    def __repr__(self) -> str:
        sizes = ", ".join(f"{d}: {n}" for d, n in self.sizes.items())
        return f"<{type(self).__name__} {self.name or ''}({sizes}) {self.dtype}>"

    # ── Selection and shape ─────────────────────────────────────────────────

    def isel(self, indexers: Mapping[Hashable, Any] | None = None, **more: Any) -> Array:
        """Positions, a slice, or a mask per axis; an integer removes the axis."""
        array = self
        for dim, index in {**(indexers or {}), **more}.items():
            array = _take(array, dim, index)
        return array

    def sel(self, indexers: Mapping[Hashable, Any] | None = None, **more: Any) -> Array:
        """Labels per axis, looked up exactly; one label removes the axis, a list keeps it."""
        wanted = {**(indexers or {}), **more}
        return self.isel({dim: positions_of(self.coords[dim], labels, dim) for dim, labels in wanted.items()})

    def transpose(self, *dims: Hashable) -> Array:
        order = [self.axis(d) for d in dims]
        return Array._make(np.transpose(self.data, order), tuple(dims), self.coords, self.attrs, self.name)

    def rename(self, names: Mapping[Hashable, Hashable]) -> Array:
        """Axes renamed, labels with them."""
        dims = tuple(names.get(d, d) for d in self.dims)
        coords = {names.get(d, d): c for d, c in self.coords.items()}
        return Array._make(self.data, dims, coords, self.attrs, self.name)

    # ── Computation ─────────────────────────────────────────────────────────

    def reduce(self, op: str, axes: Sequence[Hashable], *, na_rm: bool = False,
               min_count: int | None = None) -> Array:
        """*op* (sum, mean, min, max, median, std, var, any, all, count) over *axes*.

        Float32 sums and means are correctly rounded (``labelarray.sums``); other
        reductions are NumPy's. ``na_rm`` skips missing values; a sum with
        ``min_count`` is missing where fewer values are present.
        """
        axes = list(axes)
        if not axes:
            return self
        numbers = tuple(self.axis(a) for a in axes)
        if op in ("sum", "mean") and self.data.dtype == np.float32:
            result = sums.reduce(self.data, op, numbers, na_rm=na_rm, min_count=min_count)
        else:
            # One axis goes to NumPy as an integer, as xarray passes it: NumPy sums a
            # tuple of axes in another order, which changes float results in the last bit.
            result = _reduce_data(self.data, op, numbers[0] if len(numbers) == 1 else numbers, na_rm, min_count)
        dims = tuple(d for d in self.dims if d not in axes)
        return Array._make(np.asarray(result), dims, {d: c for d, c in self.coords.items() if d in dims})

    def dot(self, other: Array, over: Sequence[Hashable]) -> Array:
        """The sum of products over the axes *over*, as ``matmul`` computes it in any memory layout."""
        result, kept = contract(self.data, self.dims, other.data, other.dims, over)
        coords = {d: c for item in (self, other) for d, c in item.coords.items() if d in kept}
        return Array._make(result, kept, coords)

    # ── NumPy's protocols ───────────────────────────────────────────────────

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        # Reductions, generalised ufuncs (core dimensions), and out= or where= arrays address
        # positions, not axis names, so they are left to NumPy's error.
        if method != "__call__" or ufunc.signature is not None or "out" in kwargs or "where" in kwargs:
            return NotImplemented
        if not all(isinstance(x, Array) or _is_scalar(x) for x in inputs):
            return NotImplemented  # a plain array has no axis names to broadcast by
        return _elementwise(ufunc, inputs, kwargs)

    def __array_function__(self, func: Callable, types: Sequence[type], args: Sequence[Any],
                           kwargs: Mapping[str, Any]) -> Any:
        handler = _FUNCTIONS.get(func)
        if handler is None or not all(issubclass(t, (Array, np.ndarray)) for t in types):
            return NotImplemented
        return handler(*args, **kwargs)


# ── Labels ──────────────────────────────────────────────────────────────────


class _Labels(np.ndarray):
    """Labels this library owns: read-only, never shared writeable, with their index kept on them.

    Views and slices are labels too, each with its own index. Arithmetic on
    labels gives plain arrays, so labels never leak into values.
    """

    _index: LabelIndex | None = None

    def __array_finalize__(self, obj: Any) -> None:
        self._index = None

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        inputs = tuple(x.view(np.ndarray) if isinstance(x, _Labels) else x for x in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)


def labels_from(labels: Any) -> np.ndarray:
    """Labels for an array, from values the caller made and does not keep: text is stored as objects."""
    values = np.asarray(labels)
    if values.dtype.kind in "UO":
        values = values.astype(object)
    return _freeze(values)


def plain(labels: np.ndarray) -> np.ndarray:
    """Labels as a plain NumPy array, without the index kept on them."""
    return labels.view(np.ndarray)


def label_index(labels: np.ndarray, op: str = "index", dim: Hashable = None) -> LabelIndex:
    """The index for a label array (``labelarray.index.index_for``): built on first use and kept on labels
    this library owns.

    Other arrays, which could change, get a fresh index each time.
    """
    if not _owned(labels):
        return index_for(labels, op, dim)
    if labels._index is None:
        labels._index = index_for(labels, op, dim)
    return labels._index


def _checked_coords(coords: Mapping[Hashable, Any], sizes: Mapping[Hashable, int]) -> dict:
    out = {}
    for dim, labels in coords.items():
        labels = _frozen_labels(labels)
        if dim not in sizes:
            raise ShapeError(f"labels for {dim}, which is not an axis; axes are {tuple(sizes)}")
        if labels.ndim != 1 or len(labels) != sizes[dim]:
            raise ShapeError(f"{dim} has {sizes[dim]} positions but {labels.size} labels")
        out[dim] = labels
    return out


def _owned(labels: Any) -> bool:
    return isinstance(labels, _Labels) and not labels.flags.writeable


def _frozen_labels(labels: Any) -> _Labels:
    """Labels that can never change: this library's own are kept, others copied once."""
    if _owned(labels):
        return labels
    return _freeze(np.array(labels, copy=True))


def _freeze(array: np.ndarray) -> _Labels:
    """Freeze a label array the caller just created and owns alone."""
    if _owned(array):
        return array
    labels = array.view(_Labels)
    labels.flags.writeable = False
    return labels


# ── Several arrays ──────────────────────────────────────────────────────────


def broadcast(operands: Sequence[Any]) -> tuple[list[Any], tuple[Hashable, ...], dict]:
    """Raw values lined up on the union of named axes (first appearance), and the union's labels.

    Labels are not compared here: callers align them first (``align``); sizes are checked.
    Operands that are not arrays (scalars) are passed through.
    """
    dims: list[Hashable] = []
    sizes: dict[Hashable, int] = {}
    coords: dict[Hashable, np.ndarray] = {}
    for item in operands:
        if isinstance(item, Array):
            for dim, size in zip(item.dims, item.data.shape):
                if dim not in sizes:
                    dims.append(dim)
                    sizes[dim] = size
                elif sizes[dim] != size:
                    raise ShapeError(f"{dim} has {sizes[dim]} positions on one operand and {size} on another")
                if dim in item.coords and dim not in coords:
                    coords[dim] = item.coords[dim]
    raw = []
    for item in operands:
        if not isinstance(item, Array):
            raw.append(item)
            continue
        present = [d for d in dims if d in item.dims]
        data = item.data
        if list(item.dims) != present:
            data = np.transpose(data, [item.dims.index(d) for d in present])
        if len(present) != len(dims):
            data = np.reshape(data, [sizes[d] if d in item.dims else 1 for d in dims])
        raw.append(data)
    return raw, tuple(dims), coords


def align(*operands: Any) -> list[Any]:
    """Each array's axes matched to the first array with that axis, under the exact rule.

    The same labels in another order are put in the first array's order; an
    axis labelled on one array and not on another, different labels, or
    different sizes without labels are an ``AlignError``. Scalars are passed through.
    """
    first: dict[Hashable, Array] = {}
    aligned = []
    for item in operands:
        if isinstance(item, Array):
            for dim in item.dims:
                if dim in first:
                    item = _matched(first[dim], item, dim)
            for dim in item.dims:
                first.setdefault(dim, item)
        aligned.append(item)
    return aligned


def concat(values: Sequence[Any], dim: Hashable, labels: Sequence[Any]) -> Array:
    """Arrays (or scalars) stacked along a new first axis *dim* with *labels*, broadcast by axis name."""
    arrays = [v if isinstance(v, Array) else Array(v) for v in values]
    raw, dims, coords = broadcast(arrays)
    shape = np.broadcast_shapes(*(np.shape(r) for r in raw))
    data = np.stack([np.broadcast_to(r, shape) for r in raw])
    return Array._make(data, (dim, *dims), {dim: labels_from(labels), **coords})


def _matched(reference: Array, value: Array, dim: Hashable) -> Array:
    labelled = dim in reference.coords
    if labelled != (dim in value.coords):
        raise AlignError(f"{dim} is labelled on one operand but not on another; "
                         "pairing by length could match the wrong positions")
    if not labelled:
        ours, theirs = reference.sizes[dim], value.sizes[dim]
        if ours != theirs:
            raise AlignError(f"{dim} has {ours:,} positions on one operand and {theirs:,} on another, "
                             "and neither has labels to match")
        return value
    order = order_like(reference.coords[dim], value.coords[dim], dim, index=label_index)
    if order is None:
        return value
    taken = _take(value, dim, order)
    taken.coords[dim] = reference.coords[dim]
    return taken


# ── NumPy's protocols ───────────────────────────────────────────────────────


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, complex, str, bytes, np.generic)) or \
        (isinstance(value, np.ndarray) and value.ndim == 0)


def _elementwise(func: Callable, operands: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
    """*func* on operands aligned and broadcast by axis name; one array per output."""
    raw, dims, coords = broadcast(align(*operands))
    result = func(*raw, **kwargs)
    if isinstance(result, tuple):
        return tuple(Array._make(np.asarray(r), dims, coords) for r in result)
    return Array._make(np.asarray(result), dims, coords)


def _where(condition: Any, true: Any = None, false: Any = None) -> Any:
    operands = (condition, true, false)
    if true is None or false is None or not all(isinstance(v, Array) or _is_scalar(v) for v in operands):
        return NotImplemented  # with one argument, np.where gives positions
    return _elementwise(np.where, operands, {})


_FUNCTIONS: dict[Callable, Callable] = {np.where: _where}


def _operator(ufunc: np.ufunc, reflected: bool = False) -> Callable:
    def method(self: Array, other: Any) -> Any:
        # Objects that set __array_ufunc__ to None (such as tidy3a's expressions) build their own
        # result through their reflected operators; anything else without axis names is not an operand.
        if getattr(other, "__array_ufunc__", False) is None or not (isinstance(other, Array) or _is_scalar(other)):
            return NotImplemented
        with np.errstate(all="ignore"):
            return _elementwise(ufunc, (other, self) if reflected else (self, other), {})

    return method


def _unary(ufunc: np.ufunc) -> Callable:
    def method(self: Array) -> Array:
        with np.errstate(all="ignore"):
            return _elementwise(ufunc, (self,), {})

    return method


for _name, _ufunc in [
    ("add", np.add), ("sub", np.subtract), ("mul", np.multiply), ("truediv", np.true_divide),
    ("floordiv", np.floor_divide), ("mod", np.remainder), ("divmod", np.divmod), ("pow", np.power),
    ("and", np.bitwise_and), ("or", np.bitwise_or), ("xor", np.bitwise_xor),
    ("lshift", np.left_shift), ("rshift", np.right_shift),
]:
    setattr(Array, f"__{_name}__", _operator(_ufunc))
    setattr(Array, f"__r{_name}__", _operator(_ufunc, reflected=True))
for _name, _ufunc in [("lt", np.less), ("le", np.less_equal), ("eq", np.equal), ("ne", np.not_equal),
                      ("gt", np.greater), ("ge", np.greater_equal)]:
    setattr(Array, f"__{_name}__", _operator(_ufunc))  # Python swaps the sides of comparisons itself
for _name, _ufunc in [("neg", np.negative), ("pos", np.positive), ("abs", np.absolute), ("invert", np.invert)]:
    setattr(Array, f"__{_name}__", _unary(_ufunc))


# ── Selection ───────────────────────────────────────────────────────────────


def _take(value: Array, dim: Hashable, index: Any) -> Array:
    """Positions, a slice, or a mask on one axis; an integer removes the axis."""
    axis = value.axis(dim)
    data = value.data[(builtins.slice(None),) * axis + (index,)]
    coords = dict(value.coords)
    if isinstance(index, (int, np.integer)):
        coords.pop(dim, None)
        dims = value.dims[:axis] + value.dims[axis + 1:]
    else:
        dims = value.dims
        if dim in coords:
            coords[dim] = _cut(coords[dim], index)
    return Array._make(data, dims, coords, value.attrs, value.name)


def _cut(labels: np.ndarray, index: Any) -> np.ndarray:
    """Labels at positions, a slice, or a mask: a slice is a view of owned labels, the rest a new array."""
    cut = labels[index]
    return cut if isinstance(index, builtins.slice) else _freeze(cut)


def positions_of(labels: np.ndarray, wanted: Any, dim: Hashable) -> Any:
    """An integer for one label, positions for a list, as xarray's ``sel`` reads them."""
    index = label_index(labels, "sel", dim)
    many = not np.isscalar(wanted) and not (isinstance(wanted, np.ndarray) and wanted.ndim == 0)
    found = index.get_indexer(np.atleast_1d(np.asarray(wanted, dtype=object if _is_text(wanted) else None)))
    if (found < 0).any():
        missing = np.atleast_1d(np.asarray(wanted, dtype=object))[found < 0][0]
        raise LabelError(f"{dim} has no label {missing!r}")
    return found if many else int(found[0])


def _is_text(value: Any) -> bool:
    return isinstance(value, str) or (isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value))


# ── Reductions ──────────────────────────────────────────────────────────────


_REDUCERS = {
    "sum": (np.sum, np.nansum), "mean": (np.mean, np.nanmean), "min": (np.min, np.nanmin),
    "max": (np.max, np.nanmax), "median": (np.median, np.nanmedian), "std": (np.std, np.nanstd),
    "var": (np.var, np.nanvar),
}


def _reduce_data(data: np.ndarray, op: str, axes: int | tuple[int, ...], na_rm: bool, min_count: int | None) -> Any:
    """NumPy reductions with missing values handled as xarray handles them."""
    kind = data.dtype.kind
    numbers = (axes,) if isinstance(axes, int) else axes
    # Only some reductions need to know where values are missing; finding out reads the whole array.
    needs_missing = op in ("any", "all", "count") or (op == "sum" and min_count is not None)
    missing = None if not needs_missing else np.isnan(data) if kind in "fc" else np.isnat(data) if kind in "mM" else None
    if op in ("min", "max") and builtins.any(data.shape[axis] == 0 for axis in numbers):
        shape = tuple(n for i, n in enumerate(data.shape) if i not in numbers)
        return np.full(shape, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if op in ("any", "all"):
            filled = data if missing is None else np.where(missing, op == "all", data)
            result = getattr(np, op)(filled, axis=axes)
            if na_rm or kind in "biu" or missing is None:
                return result
            return np.where(~missing.any(axis=axes), result, np.nan)  # missing wherever a value was
        if op == "count":
            present = ~missing if missing is not None else np.ones(data.shape, dtype=bool)
            return present.sum(axis=axes)
        skip = na_rm and kind in "fcmM"
        result = _REDUCERS[op][1 if skip else 0](data, axis=axes)
        if op == "sum" and min_count is not None:
            present = ~missing if missing is not None else np.ones(data.shape, dtype=bool)
            result = np.where(present.sum(axis=axes) >= min_count, result, np.nan)
        return result
