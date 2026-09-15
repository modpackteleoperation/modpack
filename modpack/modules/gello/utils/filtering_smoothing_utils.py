import numpy as np

class LowPassFilter:
    def __init__(self, cutoff_hz=10.0, dt=0.1, n_joints=6, order=2):
        """
        First- or second-order low-pass filter for torque signals.

        Parameters:
        -----------
        cutoff_hz : float
            Cutoff frequency in Hz
        dt : float
            Control loop timestep in seconds
        n_joints : int
            Number of joints/signals to filter
        order : int
            Filter order (1 or 2)
        """
        self.dt = dt
        self.n_joints = n_joints
        self.order = order
        self.y_prev = np.zeros(n_joints)
        self.y_prev2 = np.zeros(n_joints)  # for second-order

        # First-order alpha
        tau = 1 / (2 * np.pi * cutoff_hz)
        self.alpha = tau / (tau + dt)

    def update(self, value):
        """
        Apply low-pass filter to current input `value`.
        Supports first- or second-order filtering.

        Parameters:
        -----------
        value : array-like of shape (n_joints,)
            Current raw torque measurements

        Returns:
        --------
        np.ndarray
            Filtered torque values
        """
        value = np.asarray(value)

        if self.order == 1:
            # Simple first-order low-pass
            filtered = self.alpha * self.y_prev + (1 - self.alpha) * value
            self.y_prev = filtered
        elif self.order == 2:
            # Two-stage first-order filter = approximate second-order
            y1 = self.alpha * self.y_prev + (1 - self.alpha) * value
            filtered = self.alpha * self.y_prev2 + (1 - self.alpha) * y1
            self.y_prev = y1
            self.y_prev2 = filtered
        else:
            raise ValueError("Order must be 1 or 2")

        return filtered

