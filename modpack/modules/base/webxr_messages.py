# webxr_messages.py
from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class UnifiedWebXRMessage:
    """Single message format for all WebXR server communication"""
    timestamp: int
    
    # Device info
    device_id: Optional[str] = None
    teleop_mode: str = 'none'  # 'base', 'arm', or 'none'
    
    # WebXR pose data (from iPhone)
    position: Optional[Dict[str, float]] = None  # {x, y, z}
    orientation: Optional[Dict[str, float]] = None  # {x, y, z, w}
    
    # Server control flags
    episode_active: bool = False
    episode_num: int = 0
    rezero: bool = False
    command: Optional[str] = None  # 'rezero' when echoed back from iPhone
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}