"""
Compatibility shim only -- all real package metadata lives in
pyproject.toml ([project] table), which setuptools>=61 reads natively.

Why this file exists at all: `pip install -e .` for a pyproject.toml-only
project (no setup.py/setup.cfg) requires pip >= 21.3 to use the modern
PEP 660 editable-install hook (`build_editable`). On an older pip --
common on stock/system Python installs that haven't been upgraded (e.g.
distro-provided pip on Ubuntu) -- that hook is unavailable, and pip errors
with something like:

    ERROR: Project ... has a 'pyproject.toml' and its build backend is
    missing the 'build_editable' hook. Since it does not have a
    'setup.py' nor a 'setup.cfg', it cannot be installed in editable mode.

The long-term fix is upgrading pip (`pip install --upgrade pip`), but
requiring that isn't always practical (locked-down/offline machines,
shared systems). Having a trivial setup.py present gives OLDER pip a
fallback path: it can run legacy `setup.py develop` for the editable
install instead, while newer pip continues to use the modern PEP 660 path
transparently (this file changes nothing about that -- `setup()` with no
arguments just tells setuptools "read metadata from pyproject.toml").
"""

from setuptools import setup

setup()
