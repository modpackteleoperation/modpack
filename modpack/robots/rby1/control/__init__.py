"""Bindings and helpers for the RBY1 realtime controller."""

from pathlib import Path
from importlib import import_module

__all__ = [
    "Config",
    "RealtimeDriver",
    "RobotSnapshot",
    "AdmittanceController",
    "AdmittanceControllerConfig",
    "debug_echo",
]

# Ensure Python searches both the source tree and the build directory for the
# compiled module. The pybind11 extension is emitted into ../build/.
_pkg_dir = Path(__file__).resolve().parent
_root_dir = _pkg_dir.parent
_candidate_dirs = [p for p in _root_dir.iterdir() if p.is_dir() and p.name.startswith("build")]
if _candidate_dirs:
  __path__ = list(__path__) + [str(p) for p in _candidate_dirs]

_bindings = import_module(".rby1_controller", __name__)
Config = _bindings.Config
RealtimeDriver = _bindings.RealtimeDriver
RobotSnapshot = _bindings.RobotSnapshot
AdmittanceController = _bindings.AdmittanceController
AdmittanceControllerConfig = _bindings.AdmittanceControllerConfig
debug_echo = _bindings.debug_echo
