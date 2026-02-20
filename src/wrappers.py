import numpy as np
import gymnasium as gym


class ActionToPythonFloatWrapper(gym.ActionWrapper):

    def action(self, action):
        a = np.asarray(action).reshape(-1)
        # CarRacing action is 3D: [steer, gas, brake]
        return (float(a[0]), float(a[1]), float(a[2]))
