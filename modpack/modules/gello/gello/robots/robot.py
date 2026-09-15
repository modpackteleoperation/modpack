from abc import abstractmethod
from typing import Protocol

import numpy as np


class Robot(Protocol):
    """Robot protocol.

    A protocol for a robot that can be controlled.
    """

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        raise NotImplementedError


