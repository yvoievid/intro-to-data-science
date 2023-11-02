
import gymnasium as gym
from gymnasium import spaces

import pygame
import numpy as np

from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.action_space import MultiAgentActionSpace
from ..models.unit import Unit
from ..models.soldier import Soldier


class SafePath(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=5, window_size=512, n_predators=2, n_agents=1):
        self.grid_size = grid_size  # The size of the square grid
        self.window_size = window_size  # The size of the PyGame window
        self._n_predators = n_predators
        self._n_agents = n_agents
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(grid_size * grid_size),
                "target": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
                "predators": spaces.Tuple((spaces.Box(0, grid_size - 1,  shape=(2,), dtype=int) for _ in range(self._n_predators)))
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down" for each predator

        # Adding 1 to take into account the main agent
        self.action_space = MultiAgentActionSpace([spaces.Discrete(4) for _ in range(self._n_predators + self._n_agents) ])
 
        # List of predators that will be chasing the agent
        # position, name, step, size
        self._predators = {_: None for _ in range(self._n_predators)}

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "predators": { i: unit.position for i, unit in self._predators.items()}}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
        
        
        # By default all predators will be soldiers, but can be changed in future
        self._predators = {_: Soldier(position=self.np_random.integers(0, self.grid_size, size=2, dtype=int), name="predator_" + str(_) ) for _ in range(self._n_predators)}
        
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.grid_size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # As we have muliaction state, we should decide direction to each agent
        agents_directions = []
        for direction in action:
            agents_directions.append(self._action_to_direction[direction])
        
        # Lets consider the last item in array for main agent
        main_agent_direction = agents_directions[-1]
        
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + main_agent_direction, 0, self.grid_size - 1
        )
        
        for key, predator in self._predators.items():
            predator.position = np.clip(predator.position + agents_directions[key], 0, self.grid_size - 1)
        
        
        # # Predators should move by specific algotight knows by commandors, perfectrly discribed by functions
        
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw predators!
        for predator in self._predators.values():
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (predator.position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            
        # Draw the area of predators' attack
        for predator in self._predators.values():
            for neighbour in self.__get_neighbour_coordinates(predator.position):
                pygame.draw.rect(
                    canvas,
                    (255, 211, 0),
                    pygame.Rect(neighbour[0]*pix_square_size, neighbour[1]*pix_square_size, pix_square_size, pix_square_size)
                )
                
        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours


    def is_valid(self, pos):
        return (0 <= pos[0] < self.grid_size) and (0 <= pos[1] < self.grid_size)
    
    
# Some constants... you know... to make life easier
PREDATOR_OBJECT = 1