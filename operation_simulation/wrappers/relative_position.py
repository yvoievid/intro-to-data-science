import gymnasium as gym
from gym.spaces import Box, Discrete
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Discrete(self.grid_size*self.grid_size)

    def observation(self, obs):
        return obs["target"] - obs["agent"]
