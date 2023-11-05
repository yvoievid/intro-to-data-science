import random
import time
import gymnasium as gym
import numpy as np
from operation_simulation.wrappers import RelativePosition
from operation_simulation.models import Soldier, Locator, Uav

random.seed(1)



# up, down, left, right
#
# For plotting metrics
all_epochs = []
all_penalties = []


target_position = np.array([35, 5])
agent_position = np.array([5, 35])

private_rayan = Uav(agent_position, name="Bullet 1", step=1, size=1)
enemy_locator = Locator(target_position, name="Enemy Locator", step=0, size=1)

alliance = [private_rayan]
enemies = [enemy_locator]

env = gym.make('operation_simulation/SafePath-v0', render_mode = "human", grid_size=40, window_size=1024, allied_units = alliance, enemy_units = enemies)

q_table = np.zeros([env.observation_space["agent"].n, env.action_space.action_spaces[-1].n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1

print("q_table_row", q_table.shape)
for i in range(1, 100):
    env.set_init_obervations({"agent": private_rayan.position, "target": target_position})

    initial_state, info = env.reset()
    agent_state = initial_state["agent"][0] * initial_state["agent"][1]
    
    epochs, penalties, reward, = 0, 0, 0
    terminated = False

    while not terminated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[agent_state]) # Exploit learned values

        next_state, reward, terminated, _, info = env.step(env.action_space.sample())
        
        old_value = q_table[agent_state, action]
        next_max = np.max(q_table[agent_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[agent_state, action] = new_value

        if reward == 0:
            penalties += 1

        state = next_state
        epochs += 1
        
        print("PENALTIES", penalties)
        if penalties == 10:
            terminated = True
            
        
env.close()

print(q_table[3*35])