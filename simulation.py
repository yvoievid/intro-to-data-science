import random
import gymnasium as gym
import numpy as np
import pygame
from operation_simulation.models import Soldier, Locator, Tank, UnitGroup, Inference
from operation_simulation.constants import Flanks, Strategies, Weather, Commands, GroupNames
import sys
import requests
import time
from operation_simulation.cross_units_battle.local_battle import group_battles
from operation_simulation.helpers.utils import get_index_by_by_name

class GameSimulation():
    def __init__(self):
        # q learing hyperparameters
        self.epsilon = 0.3
        self.alpha = 0.05
        self.gamma = 1
        self.grid_size = 32
        self.main_group_index = 0
        self.training_fps = 50000000
        self.simulation_fps = 5
        self.training_epocs = 500
        self.simulation_epochs = 100
        self.dry_run_epochs = 5
        self.weather = "UNKNOWN"
        self.simulation_running = False
        self.iterations = 0
        self.main_running = True
        self.make_api_calls_to_get_inference = True
        self.q_leaninng_keybord_terminated = False
        self.backgroud_path = "./operation_simulation/assets/terrain_compressed.jpg"
        self.backgrounds = [
            "./operation_simulation/assets/terrain_compressed.jpg",
            "./operation_simulation/assets/terain_map.jpg"
        ]
        
        
        # window settings
        self.window_size = 718
        self.menu_height = 100
        
        # api parameters
        self.port = 105
        self.api_url = "http://127.0.0.1:"+str(self.port)
        self.inference = Inference()

        # statistic parameters
        self.gather_statistic = False
        self.encoundered_number = 0
        
        
    def setup_actors(self):
        # actors
        t90 = Tank(name="Bullet 1", size=1)
        abrams = Tank(name="Bullet 2", size=1)

        self.alliance = [UnitGroup(position=np.array([2, 20]), speed=1, units=[t90, abrams],name="ALPHA"),
                    UnitGroup(position=np.array([5, 25]), speed=1, units=[t90, abrams],name="GAMMA"),
                    UnitGroup(position=np.array([15, 29]), speed=1, units=[t90, abrams],name="DELTA")]


        enemy_locator = Locator(name="Enemy Locator", speed=0, size=1)
        enemy_tank = Tank(name="T92", speed=1.5, size=1, cover_area=10)
        bm_21 = Tank(name="BM21", speed=0, size=3, cover_area=100)

        enemy_commander = Soldier(name="Prigozhin", size=1)

        self.enemies = [
                    UnitGroup(position=np.array([30, 2]), units=[enemy_locator], name="LOCATOR GROUP", speed=0), 
                    UnitGroup(position=np.array([20, 20]), units=[enemy_tank], name="ENEMY GROUP 1", speed=2),
                    UnitGroup(position=np.array([25, 15]), units=[enemy_tank], name="ENEMY GROUP 2", speed=1),
                    UnitGroup(position=np.array([15, 10]), units=[enemy_tank], name="ENEMY GROUP 3", speed=1),
                    UnitGroup(position=np.array([28, 28]), units=[bm_21], name="DEFENCE GROUP 1", speed=0), 
                    UnitGroup(position=np.array([5, 5]), units=[bm_21], name="DEFENCE GROUP 2", speed=0)
                ]


        self.target_group = UnitGroup(position=np.array([24, 5]), units=[enemy_commander], name="Prigozhin")


        # creating interface
        window = pygame.display.set_mode((self.window_size, self.menu_height + self.window_size), pygame.RESIZABLE)
        terrain_background = pygame.image.load(self.backgrounds[0])
        pygame.display.set_caption("Reinforcement Learning Simulation")

        # create gym env
        self.env = gym.make('operation_simulation/SafePath-v0',
                    window=window, 
                    background=terrain_background,
                    fps=self.simulation_fps, 
                    render_mode="human", 
                    size=self.grid_size, 
                    window_size=self.window_size,
                    allies=self.alliance, 
                    enemies=self.enemies, 
                    target=self.target_group,
                    main_unit_group_index=self.main_group_index,
                    weather=self.weather,
                    all_backgrounds=self.backgrounds)

        self.env.reset()
        # decide the main assault group        
        for unit_group in self.alliance:
            unit_group.q_table = np.zeros((self.env.observation_space["main_group"].n,self.env.action_space.n))

        
    def q_learning_simulation(self):        
        
        for i in range(1, self.iterations):
            battleground_observations, info = self.env.reset()
            main_unit_group = battleground_observations["main_group"]
            
            main_unit_group.state = main_unit_group.calculate_state()
            self.env.set_inference(self.inference)
            
            terminated = False
            
            epochs, penalties, reward = 0, 0, 0
            was_encouted = False
            
            while not self.q_leaninng_keybord_terminated and not terminated:
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(main_unit_group.q_table[main_unit_group.state])

                if (self.env.render_mode == "human"): 
                    self.get_shared_reset_state()
                    if (self.inference.reset):
                        self.reset_simulation()
                    
                next_groups_observation, reward, terminated, _, info = self.env.step(action)
                if info["encountered"] and self.gather_statistic:
                    was_encouted = True
                
                next_state = next_groups_observation["main_group"].calculate_state()

                old_value = main_unit_group.q_table[main_unit_group.state, action]

                next_max = np.max(main_unit_group.q_table[next_state])

                # main q learning magic
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                
                main_unit_group.q_table[main_unit_group.state, action] = new_value
                
                if reward <= 0:
                    penalties += 1

                main_unit_group.state = next_state
                epochs += 1
    
                if penalties == 100:
                    terminated = True
    
            if was_encouted:
                self.encoundered_number += 1

        self.env.set_encounters(self.encoundered_number)
        self.reset_train_and_simulate_states()

    def train(self):
        print("training....")
        self.q_leaninng_keybord_terminated = False
        self.make_api_calls_to_get_inference = False
        self.env.set_fps(self.training_fps)
        self.env.set_render_mode("train")
        self.epsilon = 0.1
        self.iterations = self.training_epocs
        self.q_learning_simulation() 

    def gather_statistics(self):
        print("gathering statistic....")
        self.gather_statistic = True
        self.epsilon = 0.01
        self.iterations = self.simulation_epochs
        self.env.set_render_mode("train")
        self.env.set_total_iterations(self.iterations)
        self.q_learning_simulation() 
        self.gather_statistic = False
        
    def dryrun(self):
        print("running....")
        self.env.set_fps(self.simulation_fps)
        self.env.set_render_mode("human")
        self.iterations = self.dry_run_epochs
        self.q_learning_simulation() 

    def reset_simulation(self):
        print("reset")
        self.env.reset()
        self.env.set_main_group_intex(get_index_by_by_name(self.alliance, self.inference.group))
        self.q_leaninng_keybord_terminated = True
        self.main_running = True
        self.env.set_local_battle_prob([])
        self.env.set_encounter(0)
        self.reset_train_and_simulate_states()
    
    def swtich_terrain(self):
        self.env.switch_background()

    def main(self):
        pygame.init()
        pygame.display.init()
        self.setup_actors()

        while self.main_running:
            if self.make_api_calls_to_get_inference:
                self.get_shared_state()
                time.sleep(1)
                        
            if (self.inference.simulate): 
                self.train()    
                self.gather_statistics()
                self.local_battle()
                self.dryrun()
            
            if (self.inference.dryrun): 
                self.dryrun()           
                
            if (self.inference.reset): 
                self.reset_simulation()

            if (self.inference.switch_terrain): 
                self.swtich_terrain()
                
                
    # Function to get shared state from Flask API
    def get_shared_state(self):
        response = requests.get(self.api_url+"/getInference/").json()
        self.inference = Inference(**response)
        self.env.set_main_group_intex(get_index_by_by_name(self.alliance, self.inference.group))
        self.env.set_weather(self.inference.weather)
        if self.inference.weather == Weather.WINTER:
            self.gamma = 0.5
        self.env.set_strategy(self.inference.strategy)    
        
        
    def get_shared_reset_state(self):
        response = requests.get(self.api_url+"/getInference/").json()
        self.inference = Inference(**response)
        

    def reset_train_and_simulate_states(self):
        self.inference.train = False
        self.inference.simulate = False
        self.inference.dryrun = False
        self.iterations = 0
        self.encoundered_number = 0
        self.make_api_calls_to_get_inference = True
        self.env.reset()

    def local_battle(self):
        alliance_units = self.alliance[get_index_by_by_name(self.alliance, self.inference.group)].units
        outcomes = []

        for enemy_group in self.enemies:
            outcome = group_battles(alliance_units, enemy_group.units, enemy_group.name)
            outcomes.append(outcome)
            
        self.env.set_local_battle_prob(outcomes)
