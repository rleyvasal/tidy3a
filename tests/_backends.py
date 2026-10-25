"""Expectations that depend on the backend, following ``docs/backends.md``."""

from tidy3a.backend import get_backend


def stored_encoding(encoding):
    """The storage encoding a result keeps: the xarray backend keeps it; the NumPy backend has none."""
    return encoding if get_backend().name == "xarray" else {}
