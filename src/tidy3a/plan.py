"""Recorded plans and reusable pipelines (plan milestone 10, decision 10, stage a).

Each verb still runs at once, through the backend, and records a step on
its result: the verb itself, and a snapshot of the result's variables,
axes, field axis, groups, and notes. Snapshots hold names, sizes, and
types only, never values, so recording costs no computation, and
``explain()`` compares consecutive snapshots to say what each step did.
A ``Pipeline`` is verbs composed before any data: ``frame >> pipeline``
runs them in order, ``pipeline.explain(frame, head=...)`` previews them on
the first positions, and ``frame.plan()`` gives a result's recorded verbs
back as a pipeline to run on other data.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Snapshot:
    """What a frame holds, without its values: enough to describe a step's effect."""

    variables: tuple[tuple[Hashable, tuple[Hashable, ...], tuple[int, ...], Any], ...]  # name, axes, shape, type
    sizes: tuple[tuple[Hashable, int], ...]
    field_axis: Hashable | None
    groups: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Step:
    """One verb as it ran, and what its result held (or, for a verb giving a value, what it gave)."""

    verb: Any
    after: Snapshot | None
    value: str | None = None


@dataclass(frozen=True)
class Record:
    """A frame's history: the source it started from and the steps since."""

    source: str
    start: Snapshot
    steps: tuple[Step, ...] = ()
    preview: str | None = None

    def then(self, step: Step) -> Record:
        return Record(self.source, self.start, (*self.steps, step), self.preview)


def snapshot(frame: Any) -> Snapshot:
    backend, data = frame._backend, frame._data
    variables = tuple(backend.layout(data))
    held = backend.sizes(data)
    # Axes in the order the inspection output shows them: as variables first use them, then the rest.
    order = dict.fromkeys(dim for _, dims, _, _ in variables for dim in dims)
    sizes = tuple((dim, held[dim]) for dim in (*order, *(dim for dim in held if dim not in order)))
    resolved = frame._resolved
    counts = {}
    if resolved is not None and resolved[0] is data:
        counts[resolved[1].label] = len(resolved[2])
    groups = tuple(f"{key.label} ({counts[key.label]})" if key.label in counts else key.label
                   for key in frame._groups)
    return Snapshot(variables, sizes, frame._field_axis, groups, frame._notes)


def start(frame: Any, preview: str | None = None) -> Record:
    """The record a frame's first step begins: its own source and contents, kept on the frame once made."""
    if frame._record is None:
        frame._record = Record(frame._source or "data", snapshot(frame), preview=preview)
    return frame._record


def recorded(before: Any, verb: Any, result: Any) -> Any:
    """*result*, a frame *verb* made from *before*, with the verb's step added to *before*'s record."""
    if result is before:  # frames are shared; the result gets its own copy to record on
        result = result._evolve(resolved=result._resolved)
    result._record = start(before).then(Step(verb, snapshot(result)))
    return result


# ── explain() ───────────────────────────────────────────────────────────────


class Plan:
    """What ``explain()`` gives: the source and each step with its effect; it prints as text."""

    def __init__(self, record: Record):
        self.record = record

    @property
    def steps(self) -> tuple[Step, ...]:
        return self.record.steps

    def __str__(self) -> str:
        record = self.record
        header = f"<plan source: {record.source}"
        lines = [header + (f"; {record.preview}>" if record.preview else ">")]
        lines += [f"  {_variable(entry)}" for entry in record.start.variables]
        if not record.steps:
            lines.append("(no steps)")
        before = record.start
        for number, step in enumerate(record.steps, 1):
            lines.append(f"{number}. {step.verb.text}")
            if step.after is None:
                lines.append(f"     → {step.value}")
                continue
            effect = _effect(before, step.after) or ["(no change)"]
            lines += [f"     {line}" for line in effect]
            before = step.after
        return "\n".join(lines)

    __repr__ = __str__


def _size(size: int | None) -> str:
    """A size, or ? for one the data decides, which a frame read in blocks does not know before it runs."""
    return "?" if size is None else f"{size:,}"


def _variable(entry: tuple) -> str:
    name, dims, shape, dtype = entry  # the type is a NumPy dtype, printed as its name
    return f"{name} ({', '.join(f'{dim}: {_size(size)}' for dim, size in zip(dims, shape))}) {dtype}"


def _effect(before: Snapshot, after: Snapshot) -> list[str]:
    lines = []
    old, new = dict(before.sizes), dict(after.sizes)
    lines += [f"- axis {dim}" for dim in old if dim not in new]
    lines += [f"+ axis {dim}: {_size(size)}" for dim, size in new.items() if dim not in old]
    lines += [f"{dim}: {_size(old[dim])} → {_size(size)}" for dim, size in new.items()
              if dim in old and old[dim] != size]
    was = {entry[0]: entry for entry in before.variables}
    now = {entry[0]: entry for entry in after.variables}
    lines += [f"- {name}" for name in was if name not in now]
    for name, entry in now.items():
        if name not in was:
            lines.append(f"+ {_variable(entry)}")
        elif (was[name][1], was[name][3]) != (entry[1], entry[3]):  # new axes or type; sizes show above
            lines.append(f"~ {_variable(entry)}")
    if before.field_axis != after.field_axis:
        lines.append(f"field axis: {before.field_axis or 'none'} → {after.field_axis or 'none'}")
    if before.groups != after.groups:
        lines.append(f"groups: {', '.join(after.groups) or 'none'}")
    lines += [note for note in after.notes if note not in before.notes]
    return lines


def _value(value: Any) -> str:
    """What a verb that gives no frame gave, in a few words."""
    shape, dtype = getattr(value, "shape", None), getattr(value, "dtype", None)
    kind = type(value)
    if kind.__module__.split(".")[0] == "numpy" and shape is not None:
        return f"NumPy array {shape} {dtype}"
    if kind.__module__.split(".")[0] == "polars" and kind.__name__ == "DataFrame":
        return f"Polars DataFrame ({value.height:,} rows × {value.width} columns)"
    return kind.__name__


# ── Pipelines ───────────────────────────────────────────────────────────────


class Pipeline:
    """Verbs composed before any data, run in order by ``frame >> pipeline``."""

    def __init__(self, verbs: Sequence[Any] = ()):
        self.verbs = tuple(verbs)

    def __len__(self) -> int:
        return len(self.verbs)

    def __call__(self, frame: Any) -> Any:
        for verb in self.verbs:
            frame = verb(frame)
        return frame

    def __rshift__(self, other: Any) -> Pipeline:
        return Pipeline((*self.verbs, *_verbs(other)))

    def __repr__(self) -> str:
        return f"<pipeline {' >> '.join(verb.text for verb in self.verbs)}>"

    def explain(self, frame: Any, *, head: int | None = None) -> Plan:
        """Run the pipeline on *frame*, or on its first *head* positions of each observation axis, and explain it.

        Only the steps of this pipeline are listed, from *frame* as it is (or as previewed).
        """
        from tidy3a.frame import TidyArray

        if not isinstance(frame, TidyArray):
            raise TypeError("explain() previews a pipeline on tidy3a data")
        preview = None
        if head is not None:
            frame, preview = _head(frame, head)
        frame = frame._evolve(record=None)
        record = start(frame, preview)
        for verb in self.verbs:
            result = verb(frame)
            if not isinstance(result, TidyArray):
                record = record.then(Step(verb, None, _value(result)))
                break
            record, frame = result._record, result
        return Plan(record)


def _verbs(other: Any) -> tuple[Any, ...]:
    from tidy3a.verbs import Verb

    if isinstance(other, Pipeline):
        return other.verbs
    if isinstance(other, Verb):
        return (other,)
    raise TypeError(f"only verbs and pipelines compose with >>, not {type(other).__name__}")


def _head(frame: Any, head: int) -> tuple[Any, str]:
    """*frame* cut to its first *head* positions on each observation axis, and the words saying so."""
    if isinstance(head, bool) or not isinstance(head, int) or head < 1:
        raise TypeError("head= is a positive number of positions")
    backend, data = frame._backend, frame._data
    cut = []
    for dim, size in backend.sizes(data).items():
        if dim == frame._field_axis or size <= head:
            continue
        data = backend.isel(data, {dim: slice(0, head)})
        cut.append(f"{dim}")
        cut_size = size
    if not cut:
        return frame, "previewed on all positions"
    words = (f"previewed on the first {head:,} of {cut_size:,} positions of {cut[0]}" if len(cut) == 1
             else f"previewed on the first {head:,} positions of {', '.join(cut)}")
    return frame._evolve(data), words
