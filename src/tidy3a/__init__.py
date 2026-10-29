"""tidy3a: tidy-style verbs for named multidimensional arrays."""

from tidy3a.__version__ import __version__
from tidy3a.backend import set_backend
from tidy3a.errors import (
    AlignError,
    ArrangeError,
    BlockError,
    FilterError,
    GroupError,
    MutateError,
    ReadError,
    RefError,
    SelectError,
    SliceError,
    SummariseError,
    TidyError,
    WriteError,
)
from tidy3a.expr import (
    all,
    any,
    bins,
    cells,
    col,
    env,
    fields,
    if_else,
    label,
    max,
    mean,
    median,
    min,
    n,
    quantile,
    rounded,
    std,
    sum,
    var,
    variable,
)
from tidy3a.formula import FormulaError, formula
from tidy3a.chunked import scan
from tidy3a.frame import TidyArray, tidy
from tidy3a.io import fixed_text, infer, read_array, read_binary
from tidy3a.write import write
from tidy3a.reductions import register_reduction
from tidy3a.linalg import (
    angle,
    arccos,
    arcsin,
    arctan,
    arctan2,
    ceil,
    conj,
    cos,
    cosh,
    dot,
    exp,
    floor,
    imag,
    log,
    log2,
    log10,
    pack,
    real,
    rename_axes,
    sign,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from tidy3a.plan import Pipeline, Plan
from tidy3a.selectors import (
    all_of,
    any_of,
    contains,
    ends_with,
    everything,
    is_numeric,
    last_col,
    matches,
    num_range,
    starts_with,
    where,
)
from tidy3a.tables import as_table
from tidy3a.verbs import (
    arrange,
    assign,
    bind_cols,
    desc,
    filter,
    full_join,
    group_by,
    inner_join,
    left_join,
    mutate,
    pull,
    select,
    slice,
    slice_head,
    slice_tail,
    summarise,
    summarize,
    ungroup,
)

nan = float("nan")

__all__ = [
    "AlignError", "ArrangeError", "BlockError", "FilterError", "GroupError", "MutateError", "ReadError", "RefError",
    "Pipeline", "Plan", "SelectError", "SliceError", "SummariseError", "TidyArray", "TidyError", "WriteError", "__version__",
    "all", "all_of", "any", "any_of", "arrange", "as_table", "assign", "bind_cols", "bins", "cells", "col", "contains", "desc",
    "ends_with", "env", "everything", "fields", "filter", "fixed_text", "full_join", "group_by", "if_else",
    "infer", "inner_join", "is_numeric", "label", "last_col", "left_join", "matches", "max",
    "mean", "median", "min", "mutate", "n", "nan", "num_range", "pull", "quantile", "read_array", "read_binary",
    "register_reduction", "rounded",
    "scan", "select",
    "set_backend", "slice", "slice_head", "slice_tail", "starts_with", "std", "sum", "summarise", "summarize",
    "tidy", "ungroup", "var", "variable", "where", "write",
    # formula
    "FormulaError", "formula",
    # array layer
    "angle", "arccos", "arcsin", "arctan", "arctan2", "ceil", "conj", "cos", "cosh", "dot",
    "exp", "floor", "imag", "log", "log2", "log10", "pack", "real", "rename_axes", "sign",
    "sin", "sinh", "sqrt", "tan", "tanh",
]
