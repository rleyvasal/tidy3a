"""Compiled kernels kept on disk between runs (plan milestone 16, step 6).

Each kernel's generated source is written once as a module, ``k<hash>.py``, under ``~/.cache/tidy3a/kernels``
(``TIDY3A_KERNELS`` names another folder), and Numba keeps what it compiles from it beside it (``cache=True``), one
entry for each set of argument types. The hash covers the source and what decides the machine code: the tidy3a,
labelarray, Numba, llvmlite, NumPy, and Python versions and the CPU, so a change to any of them gives new files, and
the old ones are never read again. A module is written to a temporary file and moved into place, so a run, or two at
once, never reads half of one; Numba writes its own files the same way.

Where the folder cannot be made or written, or Numba cannot keep a kernel there, the kernel is compiled in memory as
before, each run.

The folder is held to ``TIDY3A_KERNELS_MAX`` (bytes, or text such as ``"500MB"``; 1 GB by default, milestone 17):
writing a new kernel removes the least recently used ones, a kernel's module and Numba's files together, and loading
a kernel touches its module, so its use is recorded. ``clear()`` removes every kernel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import sys
import tempfile
import warnings
from functools import cache
from pathlib import Path
from typing import Any

PREFIX = "tidy3a_kernel_"  # the modules' names, so a traceback names a kernel
LIMIT = 10**9  # the folder's bytes by default (TIDY3A_KERNELS_MAX)


def directory() -> Path:
    """Where kernels are kept: ``TIDY3A_KERNELS``, or ``~/.cache/tidy3a/kernels``."""
    named = os.environ.get("TIDY3A_KERNELS", "").strip()
    return Path(named).expanduser() if named else Path.home() / ".cache" / "tidy3a" / "kernels"


def limit() -> int:
    """The most bytes the folder holds: ``TIDY3A_KERNELS_MAX``, or ``LIMIT``."""
    named = os.environ.get("TIDY3A_KERNELS_MAX", "").strip()
    if not named:
        return LIMIT
    from tidy3a.sizing import budget

    return budget(named)


def _kept(folder: Path) -> dict[str, list[Path]]:
    """Each kept kernel's files, by its module's stem: the module, and what Numba and Python keep of it in
    ``__pycache__``."""
    kernels: dict[str, list[Path]] = {}
    for module in folder.glob("k*.py"):
        kernels[module.stem] = [module]
    numba = folder / "__pycache__"
    if numba.is_dir():
        for path in numba.iterdir():
            stem = path.name.split(".", 1)[0]
            if stem in kernels and path.suffix in (".nbi", ".nbc", ".pyc"):  # Numba's, and Python's bytecode
                kernels[stem].append(path)
    return kernels


def prune(folder: Path | None = None, keep: Path | None = None) -> int:
    """Remove the least recently used kernels until *folder* holds at most ``limit()`` bytes, never *keep*'s; the
    bytes removed."""
    folder = folder or directory()
    kernels = _kept(folder)
    sizes = {}
    used = {}
    for stem, paths in kernels.items():
        try:
            sizes[stem] = sum(path.stat().st_size for path in paths)
            used[stem] = paths[0].stat().st_mtime
        except OSError:  # removed meanwhile, by another run
            continue
    held, cap, removed = sum(sizes.values()), limit(), 0
    for stem in sorted(sizes, key=used.__getitem__):
        if held <= cap:
            break
        if keep is not None and stem == keep.stem:
            continue
        for path in kernels[stem]:
            path.unlink(missing_ok=True)
        held -= sizes[stem]
        removed += sizes[stem]
    return removed


def clear(folder: Path | None = None) -> None:
    """Remove every kept kernel from *folder* (the kernels' folder by default), and nothing else there."""
    folder = folder or directory()
    if not folder.is_dir():
        return
    for paths in _kept(folder).values():
        for path in paths:
            path.unlink(missing_ok=True)
    for temporary in folder.glob(".k*.tmp"):
        temporary.unlink(missing_ok=True)


@cache
def fingerprint() -> str:
    """What besides a kernel's source decides its machine code."""
    from importlib.metadata import PackageNotFoundError, version

    import llvmlite.binding

    found = []
    for package in ("tidy3a", "labelarray", "numba", "llvmlite", "numpy"):
        try:
            found.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            found.append(f"{package} ?")
    cpu = f"{platform.machine()} {llvmlite.binding.get_host_cpu_name()} {llvmlite.binding.get_host_cpu_features()}"
    return "\n".join([*found, f"python {sys.version}", cpu])


def module_source(source: str, name: str, helpers: tuple[str, ...]) -> str:
    """*source* as a module whose *name* and *helpers* are compiled by Numba with ``cache=True``."""
    lines = ["import math", "import numba", "import numpy as np", "", source.strip("\n"), ""]
    lines += [f'{helper} = numba.njit(nogil=True, error_model="numpy", cache=True)({helper})' for helper in helpers]
    lines.append(f'{name} = numba.njit(nogil=True, error_model="numpy", cache=True)({name})')
    return "\n".join(lines) + "\n"


def load(source: str, name: str, helpers: tuple[str, ...] = ()) -> Any:
    """*source*'s function *name* compiled by Numba, from the kept files where they are; None where kernels cannot
    be kept, so the caller compiles in memory."""
    text = module_source(source, name, helpers)
    key = hashlib.sha256(f"{fingerprint()}\n{text}".encode()).hexdigest()[:32]
    folder = directory()
    path = folder / f"k{key}.py"
    try:
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".k{key}.", suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(text)
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            prune(folder, keep=path)  # a new kernel: the folder held to its limit
        else:
            try:
                os.utime(path)  # used now, for pruning
            except OSError:  # a folder this run may read but not write
                pass
        spec = importlib.util.spec_from_file_location(f"{PREFIX}{key}", path)
        assert spec is not None and spec.loader is not None
        module = sys.modules.get(spec.name)
        if module is None or getattr(module, "__file__", None) != str(path):  # loaded from another folder, if any
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # where Numba finds a kept kernel's helpers, which it names by module
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # where Numba cannot keep a kernel, it warns
                    spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[spec.name]
                raise
    except (OSError, RuntimeError, Warning):
        return None
    return getattr(module, name)
