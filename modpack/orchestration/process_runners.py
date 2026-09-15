""" Snapshot passed to each module's runner subprocess."""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ProcessRunConfig:
    gello_config: Dict[str, Any]   # parsed gello block from robot's config.yaml
    host: str
    port: int
    activation_host: str
    activation_port: int
    camera_endpoint: str
    manager_cfg: Dict[str, Any]
    debug_flags: Dict[str, Any]
    fixed_gello_ip: str
    feedback_port: int
    logging_streams: List[Dict[str, Any]]
    pc_id: str = "gello"
    module_overrides: Dict[str, Any] = None

    def __post_init__(self):
        if self.module_overrides is None:
            self.module_overrides = {}

    @classmethod
    def from_manager(cls, mgr) -> "ProcessRunConfig":
        return cls(
            gello_config=dict(getattr(mgr, "gello_config", {}) or {}),
            host=mgr.host,
            port=mgr.port,
            activation_host=mgr.activation_host,
            activation_port=mgr.activation_port,
            camera_endpoint=getattr(mgr, "camera_endpoint", ""),
            manager_cfg=mgr.manager_cfg,
            debug_flags=mgr.debug_flags,
            fixed_gello_ip=mgr.fixed_gello_ip,
            feedback_port=mgr.feedback_port,
            logging_streams=list(getattr(mgr, "_logging_streams", []) or []),
            pc_id=mgr.pc_id,
            module_overrides=dict(getattr(mgr, "module_overrides", {}) or {}),
        )
