"""The array layer, released on its own as ``labelarray`` (plan milestone 10, step A7).

tidy3a 0.3.0 shipped it as the subpackage ``tidy3a.labelarray``. That name
stays, for code written against it, as another name for the package and its
modules: ``tidy3a.labelarray`` is ``labelarray``, and ``tidy3a.labelarray.sums``
is ``labelarray.sums``. tidy3a itself imports ``labelarray``.
"""

import importlib
import pkgutil
import sys

import labelarray

for _module in pkgutil.iter_modules(labelarray.__path__):
    if _module.name != "_compiled":  # needs Numba; labelarray.sums imports it when Numba is installed
        sys.modules[f"{__name__}.{_module.name}"] = importlib.import_module(f"labelarray.{_module.name}")
sys.modules[__name__] = labelarray
