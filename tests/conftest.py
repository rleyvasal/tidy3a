"""Every test runs once per backend (plan milestone 9, step 4).

Each test in this directory is parametrised over the backends, which become
the default for that run; a test that creates data without naming a backend
therefore checks both. ``@pytest.mark.backends("xarray", reason=...)`` limits
a test to some backends, and needs a reason: a deliberate difference listed
in ``docs/backends.md``, or a test that compares or chooses backends itself.
"""

import pytest

from tidy3a.backend import get_backend, set_backend

BACKENDS = ("xarray", "numpy")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "backends(*names, reason): run this test on these backends only, for the reason given"
    )


def pytest_generate_tests(metafunc):
    if "tidy3a_backend" not in metafunc.fixturenames:
        return
    marker = metafunc.definition.get_closest_marker("backends")
    names = BACKENDS
    if marker is not None:
        if not marker.kwargs.get("reason"):
            raise pytest.UsageError(f"{metafunc.definition.nodeid}: backends(...) needs a reason")
        names = marker.args
    metafunc.parametrize("tidy3a_backend", names, indirect=True)


@pytest.fixture(autouse=True)
def tidy3a_backend(request):
    """Make this run's backend the default, and restore the previous default afterwards."""
    name = getattr(request, "param", "xarray")
    before = get_backend().name
    set_backend(name)
    yield name
    set_backend(before)
