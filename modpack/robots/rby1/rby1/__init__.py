"""Namespace package shim so import rby1.control works from source tree."""

import sys
from importlib import import_module

__all__ = ["control"]


def __getattr__(name: str):
  if name == "control":
    module = import_module("control")
    sys.modules.setdefault(f"{__name__}.control", module)
    return module
  raise AttributeError(f"module 'rby1' has no attribute {name!r}")
