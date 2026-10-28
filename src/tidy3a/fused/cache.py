"""Compiled kernels kept on disk between runs (plan milestone 16, step 6).

Each kernel's generated source is written once as a module, ``k<hash>.py``, under ``~/.cache/tidy3a/kernels``
(``TIDY3A_KERNELS`` names another folder), and Numba keeps what it compiles from it beside it (``cache=True``), one
entry for each set of argument types. The hash covers the source and what decides the machine code: the tidy3a,
labelarray, Numba, llvmlite, NumPy, and Python versions and the CPU, so a change to any of them gives new files, and
the old ones are never read again. A module is written to a temporary file and moved into place, so a run, or two at
once, never reads half of one; Numba writes its own files the same way.

Where the folder cannot be made or written, or Numba cannot keep a kernel there, the kernel is compiled in memory as
before, each run.
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


def directory() -> Path:
    """Where kernels are kept: ``TIDY3A_KERNELS``, or ``~/.cache/tidy3a/kernels``."""
    named = os.environ.get("TIDY3A_KERNELS", "").strip()
    return Path(named).expanduser() if named else Path.home() / ".cache" / "tidy3a" / "kernels"


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
    lines = ["import numba", "import numpy as np", "", source.strip("\n"), ""]
    lines += [f"{helper} = numba.njit(nogil=True, cache=True)({helper})" for helper in helpers]
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
        spec = importlib.util.spec_from_file_location(f"{PREFIX}{key}", path)
        assert spec is not None and spec.loader is not None
        module = sys.modules.get(spec.name)
        if module is None:
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
