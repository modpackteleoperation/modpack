#!/usr/bin/env python3
from enum import Enum, auto

class SystemState(Enum):
    IDLE = auto()       # waiting for episode start
    ACTIVE = auto()     # consuming live teleop commands
    PAUSED = auto()     # holding position, caching latest target
    RESUMING = auto()   # interpolating back to target before ACTIVE