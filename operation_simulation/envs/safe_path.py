
import gymnasium as gym
from gymnasium import spaces

import pygame
import numpy as np
import copy
from operation_simulation.constants import Flanks, Strategies, Weather


class SafePath(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "train"], "render_fps": 5}

    def __init__(self, window=None, header=100, background=None, fps=5, render_mode="human", size=5, 
                 window_size=512, allies=None, enemies=None, target=None, main_unit_group_index=0, weather=Weather.WINTER, all_backgrounds=None):
        self.grid_size = size
        self.window_size = window_size  
        self.header_size = header
        self.background = pygame.transform.scale(background, (window_size, window_size))
        self.all_backgrounds = all_backgrounds

        self._allies = allies 
        self._enemies = enemies
        
        self._n_allies = len(allies)
        self._n_enemies = len(enemies)
        
        self._main_unit_group_index = main_unit_group_index
        self._main_unit_group_initial_position = copy.deepcopy(self._allies[main_unit_group_index].position)
        self._enemies_inital_positions = {enemy.name: enemy.position for enemy in self._enemies}
        
        self._main_unit_group = self._allies[main_unit_group_index]
        self._target = target

        self.observation_space = spaces.Dict(
            {
                "enemies":  spaces.Dict({enemy.name: spaces.Box(0, size - 1, shape=(2,), dtype=int) for enemy in self._enemies}),
                "allies": spaces.Dict({ally.name: spaces.Box(0, size - 1, shape=(2,), dtype=int) for ally in self._allies }),
                "main_group": spaces.Discrete(size*size),
                "target":  spaces.Box(0, size - 1, shape=(2,), dtype=int) 
            })
        
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

        # Inference parameters
        self._step_cost = -0.01
        self._weather_step_coefficient = 0.5
        self._weather = weather
        
        if weather == self._weather:
            self._step_cost = -0.1
        self._strategy = Strategies.SAFE
        
        self._encounters_with_emenies = 0    
        self._was_encounted = False
        self._total_iterations = 0
        
        self._left_flank_position = np.array([5, self._target.position[1]])
        self._right_flank_position = np.array([self._target.position[0], self.grid_size - 5])

        self.metadata["render_fps"] = fps
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._inference = None
        
        self.pix_square_size = (
            self.window_size / self.grid_size
        ) 

        self.local_battle_prob = []

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
            ),
                "encountered": self._was_encounted
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # self._main_unit_group = self._allies[self._main_unit_group_index]
        print("self._main_unit_group_initial_position",self._main_unit_group_initial_position)
        self._main_unit_group.position = copy.deepcopy(self._main_unit_group_initial_position)
        self._was_encounted = False
        
        for enemy in self._enemies:
            for enemy_init_name_position in self._enemies_inital_positions.items():
                if enemy_init_name_position[0] == enemy.name:
                    enemy.position = enemy_init_name_position[1]
        
        # Calculate how far we are from our target
        self._distance_to_target_start = np.linalg.norm(self._main_unit_group.position - self._target.position, ord=1)
        # self.local_battle_prob = []
        # self._encounters_with_emenies = 0
        
        # Calculatte
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
        
        if self._inference.flank == Flanks.LEFT:
            self._distance_to_flank = abs(self._main_unit_group.position[1] - self._left_flank_position[1])
        elif self._inference.flank == Flanks.RIGHT:
            self._distance_to_flank = abs(self._main_unit_group.position[0] - self._right_flank_position[0])

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
        
        reward = 0
        
        distance_reward = 0
        # We add a reward if we are getting closer to the target
        distance_reward = 2 if np.linalg.norm(self._main_unit_group.position - self._target.position, ord=1) < self._distance_to_target_start else -3  # Binary sparse rewards
        
        
        # Inference reward calculation
        flank_reward = 0
        # Flank reward: we are getting more reward if agent following the command of commander and moves towards right flank
        if self._inference.flank == Flanks.LEFT:
            flank_reward = 1 if abs(self._main_unit_group.position[1] - self._left_flank_position[1]) < self._distance_to_flank else -1
       
        elif self._inference.flank == Flanks.RIGHT:
            flank_reward = 1 if abs(self._main_unit_group.position[0] - self._right_flank_position[0]) < self._distance_to_flank else -1
       
    
        # Terain reward
        terrain_reward = 0
        red, green, blue, alpha = self.canvas.get_at(((self._main_unit_group.position + 0.1) * self.pix_square_size).astype(int))        
        terain_diff = green - red
        
        # We make the positive reward for going on low ground and the negative for going uphill, 
        # the reward is a big negative number if we detect an enemy
        if terain_diff >= LEVEL_0_HIGH_GROUND_SHIFT:
            terrain_reward += 1
    
        if terain_diff <= LEVEL_1_HIGH_GROUND_SHIFT:
            terrain_reward -= 3
        
        if terain_diff <= LEVEL_2_HIGH_GROUND_SHIFT:
            terrain_reward = -4
                
        if terain_diff <= ENEMY_DETECTION_COLOR_SHIFT and not self._strategy == Strategies.AGGRESSIVE:
            terrain_reward -= 10
        
        for enemy in self._enemies:
            if np.linalg.norm(self._main_unit_group.position - enemy.position) < enemy.get_cover_area()/self.pix_square_size:
                self._was_encounted = True
        
        # Weather reward
        if self._weather == Weather.WINTER:
            terrain_reward *= 2
            flank_reward *= 2
            distance_reward *= 0.5
        
        # Behaviour reward
        if self._strategy == Strategies.AGGRESSIVE:
            distance_reward *= 3
            terrain_reward *= 0.5
        
        reward += distance_reward
        reward += terrain_reward
        reward += flank_reward

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
            self.window = pygame.display.set_mode((self.window_size, self.header_size + self.window_size))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.canvas.blit(self.background, (0,0))
        pygame.font.init()
        font = pygame.font.SysFont(MICROSOFT_FONT, 22)
        
        # Draw the statistic header
        header_menu = pygame.Surface((self.window_size, self.header_size))
        header_menu.fill((255, 255, 255))
        self.window.blit(header_menu, header_menu.get_rect())
        self.draw_title(self.window, font, self.pix_square_size, np.array([1,1]), "Statistics")
        self.draw_title(self.window, font, self.pix_square_size, np.array([1,2]), "Figths:" + str(self._encounters_with_emenies) + " out of " + str(self._total_iterations))
        self.draw_title(self.window, font, self.pix_square_size, np.array([1,3]), "Weather:" + str(self._weather))

        i = 1
        for item in self.local_battle_prob:
            for key, value in item.items():
                self.draw_title(self.window, font, self.pix_square_size, np.array([10, i]), f"Probability of defeating {key}: {value}")
                i += 0.5
        
        
        # Draw the enemies
        for enemy_group in self._enemies:
            pygame.draw.circle(
                self.canvas,
                (152, 0, 0, 100),
                (enemy_group.position + 0.5) *  self.pix_square_size ,
                 self.pix_square_size + enemy_group.get_cover_area(),
            )

            pygame.draw.circle(
                self.canvas,
                (255, 0, 0),
                (enemy_group.position + 0.5) *  self.pix_square_size,
                 self.pix_square_size / 3,
            )
            self.draw_title(self.canvas, font,  self.pix_square_size, enemy_group.position, enemy_group.name)
            for i, unit in enumerate(enemy_group.units):
                self.draw_title(self.canvas, font,  self.pix_square_size, enemy_group.position + [0, (i+1)*1], unit.name)

            
        for ally in self._allies:
            pygame.draw.circle(
                self.canvas,
                (0, 0, 255),
                (ally.position + 0.5) *  self.pix_square_size,
                 self.pix_square_size / 3,
            )
        
            self.draw_title(self.canvas, font,  self.pix_square_size, ally.position, ally.name)
            
            for i, unit in enumerate(ally.units):
                self.draw_title(self.canvas, font,  self.pix_square_size, ally.position + [0, (i+1)*1], unit.name)


        # draw target
        pygame.draw.rect(
                self.canvas,
                (0, 255, 0),
                pygame.Rect(
                     self.pix_square_size * self._target.position,
                    (self.pix_square_size,  self.pix_square_size),
                ),
            )
        self.draw_title(self.canvas, font,  self.pix_square_size, self._target.position, self._target.name)

        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.canvas,
                0,
                (0,  self.pix_square_size * x),
                (self.window_size,  self.pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                self.canvas,
                0,
                ( self.pix_square_size * x, 0),
                ( self.pix_square_size * x, self.window_size),
                width=1,
            )
    
        if self.render_mode == "human":
            self.window.blit(self.canvas,(0, self.header_size))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
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
    
    def set_render_mode(self, render_mode):
        self.render_mode = render_mode
    
    def set_inference(self, inference):
        self._inference = inference
    
    def set_encounters(self, encounters):
        self._encounters_with_emenies = encounters
        
    def set_total_iterations(self, total_iterations):
        self._total_iterations = total_iterations
    

    def set_main_group_intex(self, index):
        self._main_unit_group = self._allies[index]
        self._main_unit_group_index = index
        self._main_unit_group_initial_position = copy.deepcopy(self._allies[index].position)

    def set_weather(self, weather):
        self._weather = weather

    def set_strategy(self, strategy):
        self._strategy = strategy

    def set_local_battle_prob(self, outcomes):
        self.local_battle_prob = outcomes
        
    def set_encounter(self, encounter):
        self._encounters_with_emenies = encounter
        
    def switch_background(self):
        terrain_background = pygame.image.load(self.all_backgrounds[0])
        self.background = pygame.transform.scale(terrain_background, (self.window_size, self.window_size))
        self.canvas.blit(self.background, (0,0))
        

LABEL_SHIFT = (1,1)
LEVEL_0_HIGH_GROUND_SHIFT = 40
LEVEL_1_HIGH_GROUND_SHIFT = -10
LEVEL_2_HIGH_GROUND_SHIFT = -20
ENEMY_DETECTION_COLOR_SHIFT = -140
MICROSOFT_FONT = 'microsoftjhengheimicrosoftjhengheiuilight'