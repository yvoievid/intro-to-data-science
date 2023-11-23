
import gymnasium as gym
from gymnasium import spaces

import pygame
import numpy as np

from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.action_space import MultiAgentActionSpace
from ..models.unit import Unit
from ..models.soldier import Soldier


class SafePath(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, window=None, fps=5, render_mode="human", size=5, window_size=512, allies=None, enemies=None, target=None, main_unit_group_index=0, weather="winter", attack_or_defend="attack", flang="upper"):
        self.grid_size = size  # The size of the square grid
        self.window_size = window_size  # The size of the PyGame window
     
        self._allies = allies 
        self._enemies = enemies
        
        self._n_allies = len(allies)
        self._n_enemies = len(enemies)
        
        self._main_unit_group_index = main_unit_group_index
        self._main_unit_group_initial_position = self._allies[main_unit_group_index].position
        self._enemies_inital_positions = {enemy.name: enemy.position for enemy in self._enemies}
        
        self._main_unit_group = self._allies[main_unit_group_index]
        self._target = target

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "enemies":  spaces.Dict({enemy.name: spaces.Box(0, size - 1, shape=(2,), dtype=int) for enemy in self._enemies}),
                "allies": spaces.Dict({ally.name: spaces.Box(0, size - 1, shape=(2,), dtype=int) for ally in self._allies }),
                "main_group": spaces.Discrete(size*size),
                "target":  spaces.Box(0, size - 1, shape=(2,), dtype=int) 
            })
        
        # We have 4 actions, corresponding to "right", "up", "left", "down" for each predator and agent
        self.action_space = spaces.Discrete(4)


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


        self._step_cost = -0.01
        self._weather_step_coefficient = 0.5
        
        # Conditions 
        if weather == "winter":
            self._step_cost = -0.1
            
        self.metadata["render_fps"] = fps
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._inference = {}
        
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = window
        self.clock = None

    def _get_obs(self):
        return {
                "enemies": { enemy_group.name: enemy_group.position[0] * enemy_group.position[1] for enemy_group in self._enemies },
                "allies": { ally_group.name: ally_group.position[0] *  ally_group.position[1] for ally_group  in self._allies }, 
                "main_group" : self._main_unit_group,
                "target":  self._target.units[0] 
        }

    def _get_info(self):
        return {
                "enemies_alive": { unit_group.name: unit_group for unit_group in self._enemies},
                "allies_alive": { unit_group.name:  unit_group for unit_group in self._allies},
                "distance": np.linalg.norm(
                    self._main_unit_group.position - self._target.position, ord=1
            )
        }

    def reset(self, seed=None, main_unit_index=0, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)    
        

        self._main_unit_group.position = self._main_unit_group_initial_position
        
        for enemy in self._enemies:
            for enemy_init_name_position in self._enemies_inital_positions.items():
                if enemy_init_name_position[0] == enemy.name:
                    enemy.position = enemy_init_name_position[1]
        
        # Calculate how far we are from our target
        self._distance_to_target_start = np.linalg.norm(self._main_unit_group.position - self._target.position, ord=1)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Lets consider the last item in array for main agent
        main_agent_direction = self._action_to_direction[action]
        
        # Calculate distance 
        self._distance_to_target_start = np.linalg.norm(self._main_unit_group.position - self._target.position, ord=1)

        
        # Move the main group 
        self._main_unit_group.position = np.clip(
            self._main_unit_group.position + main_agent_direction, 0, self.grid_size - 1
        )
        
        # Move the enemnies 
        for enemy_group in self._enemies:
            enemy_group.position = np.clip(
            enemy_group.position + self._action_to_direction[self.action_space.sample()] * enemy_group.speed, 0, self.grid_size - 1
        )
        
        
        # An episode is done iff the agent has reached the target OR it took to long OR main unit group is eliminated
        terminated = np.array_equal(self._main_unit_group.position, self._target.position ) 
        
        
        # We add a reward if we are getting closer to the target
        reward = 1 if np.linalg.norm(self._main_unit_group.position - self._target.position, ord=1) < self._distance_to_target_start else 0  # Binary sparse rewards
    
        # Inference reward calculation
        inference_reward = 0
        if self._inference['flang'] == "LEFT":
            self._left_flang_position = [5,5]
            inference_reward = 1 if np.linalg.norm(self._main_unit_group.position - self._left_flang_position , ord=1) < self._distance_to_target_start else 0  # Binary sparse rewards

               
        reward += inference_reward
    
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
        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        font = pygame.font.SysFont('Comic Sans MS', 30)
        
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # First we draw the enemies
        for enemy_group in self._enemies:
            pygame.draw.circle(
                canvas,
                (152, 0, 0, 100),
                (enemy_group.position + 0.5) * pix_square_size ,
                pix_square_size + enemy_group.get_cover_area(),
            )

            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (enemy_group.position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
            self.draw_title(canvas, font, pix_square_size, enemy_group.position, enemy_group.name)
          
            
        # Now we draw the allies
        for ally in self._allies:
            print(ally.position)
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (ally.position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        
            self.draw_title(canvas, font, pix_square_size, ally.position, ally.name)


        # Draw the target
        pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * self._target.position,
                    (pix_square_size, pix_square_size),
                ),
            )
        self.draw_title(canvas, font, pix_square_size,  self._target.position, self._target.name)

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
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  
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
    
    def draw_title(self, canvas, font, pix_square_size, pos, title):
        target_text = font.render(title, True, (0, 0, 0))
        canvas.blit(target_text, (pos - LABEL_SHIFT) * pix_square_size)
    
    def set_fps(self, fps):
        self.metadata["render_fps"] = fps
    
    def set_inference(self, inference):
        self._inference = inference
    
    
# Some constants... you know... to make life easier
LABEL_SHIFT = (1,1)
PREDATOR_OBJECT = 1