import random
import time
import gymnasium as gym
import numpy as np
import pygame
from operation_simulation.wrappers import RelativePosition
from operation_simulation.models import Soldier, Locator, Tank, UnitGroup
import sys

# hyperparameters
all_epochs = []
all_penalties = []
epsilon = 0.3
alpha = 0.05
gamma = 1
grid_size = 32
main_group_index = 0
training_fps = 50000000
simulation_fps = 60
training_epocs = 1000
simulation_epochs = 5
weather = "Winter"

# windos settinds
window_size = 1024
menu_height = 50

# actors
t90 = Tank(name="Bullet 1", size=1)
abrams = Tank(name="Bullet 2", size=1)

alliance = [UnitGroup(position=np.array([2, 20]), speed=1, units=[t90, abrams],name="alfa"),
            UnitGroup(position=np.array([5, 25]), speed=1, units=[t90, abrams],name="omega"),
            UnitGroup(position=np.array([15, 29]), speed=1, units=[t90, abrams],name="delta")]


enemy_locator = Locator(name="Enemy Locator", speed=0, size=1)
enemy_tank = Tank(name="T92", speed=1.5, size=1)
bm_21 = Tank(name="BM21", speed=0, size=3, cover_area=100)

enemy_commander = Soldier(name="Prigozhin", size=1)

enemies = [UnitGroup(position=np.array([30, 2]), units=[enemy_locator], name="locator group", speed=0), 
           UnitGroup(position=np.array([20, 20]), units=[enemy_tank], name="Assault group 1", speed=2),
           UnitGroup(position=np.array([25, 15]), units=[enemy_tank], name="Assault group 2", speed=1),
           UnitGroup(position=np.array([28, 28]), units=[bm_21], name="Defence group 1", speed=0), 
           UnitGroup(position=np.array([5, 5]), units=[bm_21], name="Defence group 2", speed=0)]


target_group = UnitGroup(position=np.array([24, 5]), units=[enemy_commander], name="Prigozhin")

pygame.init()
pygame.display.init()


# creating interface
button_size = (200, 50)
train_button_position = (1, 1)
test_button_position = (220, 1)
weather_condition_position = (420, 1)
train_text = "Train"

font = pygame.font.SysFont('Comic Sans MS', 30)

window = pygame.display.set_mode((window_size, menu_height + window_size))
            
header_menu = pygame.Surface((window_size, 50))
header_menu.fill((255, 255, 255))
window.blit(header_menu, header_menu.get_rect())

train_text = font.render(train_text, True, (0, 0, 0))
pygame.draw.rect(window, (122, 122, 122),pygame.Rect(
                    train_button_position,
                    button_size)) 
window.blit(train_text, train_button_position)

test_text = font.render("Test", True, (0, 0, 0))
pygame.draw.rect(window, (0, 122, 122),pygame.Rect(
                    test_button_position,
                    button_size)) 
window.blit(test_text, test_button_position)

weather_label = font.render("Weather:", True, (0, 0, 0))
window.blit(weather_label, weather_condition_position)


# create gym env
env = gym.make('operation_simulation/SafePath-v0',
               window=window, 
               fps=simulation_fps, 
               render_mode="human", 
               size=grid_size, 
               window_size=window_size,
               allies=alliance, 
               enemies=enemies, 
               target=target_group, 
               main_unit_group_index=main_group_index, 
               weather="winter")


# decide the main assault group
main_unit_group = alliance[main_group_index]
main_unit_group.q_table = np.zeros((env.observation_space["main_group"].n, env.action_space.n))

inference = {
            'strategy':"ATTACK",
            'flang': "CENTER"
        }

def q_learning_simulation(env, inference, hyperparams):
    epsilon, alpha, gamma = hyperparams
    
    for i in range(1, iterations):
        battleground_observations, info = env.reset()
        main_unit_group = battleground_observations["main_group"]
        
        main_unit_group.state = main_unit_group.calculate_state()
        env.set_inference(inference)
        
        epochs, penalties, reward, = 0, 0, 0
        terminated = False

        while not terminated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(main_unit_group.q_table[main_unit_group.state])

            next_groups_observation, reward, terminated, _, info = env.step(action)
            next_state = next_groups_observation["main_group"].calculate_state()

            old_value = main_unit_group.q_table[main_unit_group.state, action]

            next_max = np.max(main_unit_group.q_table[next_state])

            # main q learning magic
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max - old_value)
            
            main_unit_group.q_table[main_unit_group.state, action] = new_value
            
            if reward <= 0:
                penalties += 1

            main_unit_group.state = next_state
            epochs += 1
            
            if penalties == 100:
                terminated = True

while True:
    battleground_observations, info = env.reset()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
         
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            iterations = 0
            if event.key == pygame.K_SPACE:
                inference = {
                    'strategy':"ATTACK",
                    'flang': "RIGHT",
                }
                
                env.set_fps(training_fps)
                env.set_render_mode("train")
                iterations = training_epocs
                epsilon = 0.1
                
            if event.key == pygame.K_TAB:
                env.set_fps(simulation_fps)
                env.set_render_mode("human")
                iterations = simulation_epochs
                epsilon  = 0.01

            if event.key == pygame.K_ESCAPE:
                env.close()

            q_learning_simulation(env=env, inference=inference, hyperparams=[epsilon, alpha, gamma])