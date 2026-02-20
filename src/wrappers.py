import numpy as np
import gymnasium as gym


class ActionDTypeWrapper(gym.ActionWrapper):
    """
    Ensures actions are float32 before reaching Box2D (prevents b2RevoluteJoint motorSpeed dtype crash).
    """
    def action(self, action):
        return np.asarray(action, dtype=np.float32)
