from gymnasium import spaces
from dataclasses import dataclass,field
from typing import List

@dataclass
class Unit:
    name: str = ""
    size: int = 1
    is_alive: bool = True
    speed: int = 1
    # alpha: float = 0.1
    # gamma: float = 0.6
    # epsilon: float = 0.1
    # lr: float = 0.1
    # agent specific parameters
    # up, down, left, right
    action_space: spaces.Space = spaces.Discrete(4)

    def draw(self):
        pass


