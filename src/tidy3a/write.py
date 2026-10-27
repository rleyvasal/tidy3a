"""Writing a frame's data to a file, block by block for data read in blocks (plan milestone 12, step 2).

``frame >> write("near.zarr")`` writes the frame's data and gives the written
file back as a frame, so a pipeline can go on from it. A frame read with
``blocks=`` that keeps rows writes them as the blocks arrive, in order, so the
result need not fit in memory; ``scan(..., writes=[(frame, path)])`` writes
beside other results from the same pass. Two formats, by the path's suffix:

- ``.npy``: a single array (NumPy only), the blocked axis first. The header's
  row count is written last; a ``.npy`` file holds values, so labels on the
  other axes come back from the frame, and positions on the blocked axis do
  not (the axis comes back unlabelled).
- ``.zarr``: a dataset (xarray and zarr), appended along the blocked axis,
  with coordinates and attributes. Text is stored as variable-length strings,
  so a later block's longer text is not cut to the first block's width; only
  the encodings Zarr shares with NetCDF are kept (type, fill value, scale and
  offset, time units and calendar).

A file is written under a temporary name beside the path and renamed when
every block is in, so a failed run leaves nothing under the name given. An
existing path is refused unless ``overwrite=True``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Hashable
from typing import Any

import numpy as np

from tidy3a.errors import WriteError
from tidy3a.frame import TidyArray

FORMATS = (".npy", ".zarr")
_PORTABLE = ("dtype", "_FillValue", "scale_factor", "add_offset", "units", "calendar")


def write(path: str | os.PathLike[str], *, overwrite: bool = False) -> Any:
    """Write the data to *path* (``.npy`` or ``.zarr``) and give the written file back as a frame.

    For data read with ``blocks=``, the rows are written block by block, so the
    result need not fit in memory, and the frame given back reads the file in
    blocks of the same size.
    """
    from tidy3a.verbs import Verb

    target = os.fspath(path)
    if not isinstance(overwrite, (bool, np.bool_)):
        raise WriteError(f"write(): overwrite must be True or False, not {overwrite!r}")

    def apply(frame: TidyArray) -> TidyArray:
        writer = Writer(target, frame, None, bool(overwrite))
        try:
            writer.add(frame._data)
            writer.finish()
        except BaseException:
            writer.abort()
            raise
        return writer.reopen()

    text = f'write("{target}"' + (", overwrite=True)" if overwrite else ")")
    return Verb(text, apply, operation="write", arguments=(target, bool(overwrite)))


class Writer:
    """A file being written: ``add`` each block's collection in order, then ``finish``, or ``abort`` on failure.

    *frame* gives the schema (for data read in blocks, its empty block after the verbs); *axis* is the axis blocks
    are appended along, None to write one collection whole; *unlabel* drops the positions a block was labelled with,
    as joining the blocks in memory does when no filter ran.
    """

    def __init__(self, path: str, frame: TidyArray, axis: Hashable | None, overwrite: bool, unlabel: bool = False,
                 blocks: Any = None):
        suffix = os.path.splitext(path.rstrip("/"))[1].lower()
        if suffix not in FORMATS:
            raise WriteError(f'write("{path}"): the suffix says the format, one of {", ".join(FORMATS)}')
        if os.path.lexists(path) and not overwrite:
            raise WriteError(f'write("{path}"): {path} exists; overwrite=True replaces it')
        self.path, self.suffix, self.axis, self.overwrite, self.unlabel = path, suffix, axis, overwrite, unlabel
        self.frame, self.blocks = frame, blocks
        head, tail = os.path.split(os.path.abspath(path.rstrip("/")))
        self.temporary = os.path.join(head, f".{tail}.partial-{os.getpid()}")
        backend, data = frame._backend, frame._data
        if suffix == ".npy":
            names = backend.variable_names(data)
            if len(names) != 1:
                listed = ", ".join(map(str, names)) or "none"
                raise WriteError(f'write("{path}"): a .npy file holds one array, and this data has {len(names)} '
                                 f"variables ({listed}); select() one, or write a .zarr store")
            self.name = names[0]
            variable = backend.variable(data, self.name)
            self.dims = tuple(variable.dims)
            if axis is not None and axis not in self.dims:
                raise WriteError(f'write("{path}"): {self.name} does not run along {axis}, the axis blocks are read '
                                 "along; write a .zarr store")
            self.order = (axis, *(dim for dim in self.dims if dim != axis)) if axis is not None else self.dims
            self.dtype = variable.dtype if variable.dtype.isnative else variable.dtype.newbyteorder("=")
            if self.dtype.kind in "OUSV" or self.dtype.hasobject:
                raise WriteError(f'write("{path}"): {self.name} is {self.dtype}; a .npy file holds numbers and times, '
                                 "so write a .zarr store")
        self.file: Any = None
        self.rows = 0
        self.first: Any = None  # a Zarr store's first block, kept while every block so far was empty
        self.written = False

    def add(self, data: Any) -> None:
        """One block's collection, appended along the axis (or the whole data, with no axis)."""
        backend = self.frame._backend
        if self.unlabel:
            data = backend.unlabel(data, self.axis)
        if self.suffix == ".npy":
            self._add_npy(backend.variable(data, self.name))
        else:
            self._add_zarr(backend.collect(data))

    def _add_npy(self, variable: Any) -> None:
        values = np.asarray(variable.values)
        if tuple(variable.dims) != self.order:
            values = np.transpose(values, [list(variable.dims).index(dim) for dim in self.order])
        values = np.ascontiguousarray(values, dtype=self.dtype)
        if self.file is None:
            self.shape = values.shape[1:] if self.order and self.axis is not None else values.shape
            self.file = open(self.temporary, "wb")
            self.file.write(_npy_header(self.dtype, self._shape(0)))
        if self.axis is None:
            self.shape = values.shape
        elif values.shape[1:] != self.shape:
            raise WriteError(f'write("{self.path}"): a block of {self.name} is {values.shape[1:]} across the other axes, '
                             f"not {self.shape}")
        self.file.write(values.tobytes())
        self.rows += len(values) if self.axis is not None else 0

    def _shape(self, rows: int) -> tuple:
        return (rows, *self.shape) if self.axis is not None else self.shape

    def _add_zarr(self, dataset: Any) -> None:
        dataset = _portable(dataset)
        if self.axis is not None and dataset.sizes.get(self.axis, 0) == 0:
            if self.first is None and not self.written:
                self.first = dataset  # written at the end only if no block has rows
            return
        if not self.written:
            dataset.to_zarr(self.temporary, mode="w", consolidated=False)
            self.written = True
        else:
            dataset.to_zarr(self.temporary, append_dim=self.axis, consolidated=False)

    def finish(self) -> None:
        """Every block is in: the header's count written last, and the file renamed to the name given."""
        if self.suffix == ".npy":
            if self.file is None:  # no block at all: an empty array of the schema's shape
                self._add_npy(self.frame._backend.variable(self.frame._data, self.name))
            self.file.seek(0)
            self.file.write(_npy_header(self.dtype, self._shape(self.rows)))
            self.file.close()
            self.file = None
        elif not self.written:
            empty = self.first if self.first is not None else _portable(self.frame._backend.collect(self.frame._data))
            empty.to_zarr(self.temporary, mode="w", consolidated=False)
            self.written = True
        if os.path.lexists(self.path):  # overwrite=True, checked when the writer was made
            shutil.rmtree(self.path) if os.path.isdir(self.path) and not os.path.islink(self.path) else os.remove(
                self.path)
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        """A failed run: the temporary file removed, nothing left under the name given."""
        if self.file is not None:
            self.file.close()
            self.file = None
        if os.path.isdir(self.temporary):
            shutil.rmtree(self.temporary, ignore_errors=True)
        elif os.path.lexists(self.temporary):
            os.remove(self.temporary)

    def reopen(self) -> TidyArray:
        """The written file as a frame: in blocks of the same size for data read in blocks, else lazily."""
        frame, blocks = self.frame, self.blocks
        rows = None if blocks is None else (blocks.rows if isinstance(blocks.rows, int) else
                                            max(stop - start for start, stop in blocks.every))
        threads = 1 if blocks is None else blocks.threads
        if self.suffix == ".npy":
            from tidy3a.io import read_array

            backend, data = frame._backend, frame._data
            labels = {dim: list(backend.label_list(data, dim)) for dim in self.order
                      if dim != self.axis and backend.has_labels(data, dim)}
            field_axis = frame._field_axis if frame._field_axis in self.order else None
            options = dict(dims=list(self.order), labels=labels or None, field_axis=field_axis, name=str(self.name))
            if rows is not None:  # blocks are read into memory, so the frame's own backend holds them
                return read_array(self.path, blocks=rows, threads=threads, backend=backend.name, **options)
            return read_array(self.path, lazy=True, **options)  # memory-mapped, which the xarray backend holds
        import xarray as xr

        from tidy3a.frame import tidy

        opened = xr.open_zarr(self.path, consolidated=False)
        if rows is not None:
            return tidy(opened, blocks={str(self.axis): rows}, field_axis=frame._field_axis, threads=threads)
        return tidy(opened, field_axis=frame._field_axis)


def _portable(dataset: Any) -> Any:
    """*dataset* as Zarr stores it: only the encodings Zarr shares with NetCDF, and text as variable-length strings,
    index coordinates included: fixed-width text has no Zarr 3 specification, so other Zarr libraries may not read
    it, and a later block's longer text would not fit."""
    dataset = dataset.copy(deep=False)
    text = {}
    for name, variable in dataset.variables.items():
        variable.encoding = {key: value for key, value in variable.encoding.items() if key in _PORTABLE}
        if variable.dtype.kind in "US":
            text[name] = variable.astype(object)
    coords = {name: variable for name, variable in text.items() if name in dataset.coords}
    return dataset.assign({name: variable for name, variable in text.items() if name not in coords}).assign_coords(
        coords)


def _npy_header(dtype: np.dtype, shape: tuple) -> bytes:
    """A .npy header (version 1.0) whose length does not depend on the row count, so the count can be written last:
    the shape's numbers are padded to 20 characters, and Python reads the spaces away."""
    numbers = ", ".join(f"{size:>20}" for size in shape) + ("," if len(shape) == 1 else "")
    text = "{'descr': %r, 'fortran_order': False, 'shape': (%s), }" % (np.lib.format.dtype_to_descr(dtype), numbers)
    total = 10 + len(text) + 1
    text += " " * (-total % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(text).to_bytes(2, "little") + text.encode("latin1")
