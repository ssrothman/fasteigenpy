from __future__ import annotations

import importlib.metadata

import fasteigenpy as m


def test_version():
    assert importlib.metadata.version("fasteigenpy") == m.__version__
