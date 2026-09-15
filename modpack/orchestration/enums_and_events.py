from enum import Enum, auto


class LoggingState(Enum):
    IDLE = 0
    ACTIVE = 1
    PAUSE = 2
    RESUME = 3
    STOPPED = 4
    DELETE = 5

class EpisodeEvent(Enum):
    SINGLE_TAP = auto()
    DOUBLE_TAP = auto()
    TRIPLE_TAP = auto()
    LONG_PRESS = auto()
