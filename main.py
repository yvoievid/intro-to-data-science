import random
import time
import gymnasium as gym
import numpy as np
from operation_simulation.wrappers import RelativePosition

random.seed(1)
env = gym.make('operation_simulation/SafePath-v0', render_mode = "human", grid_size=40, window_size=1024)
wrapped_env = RelativePosition(env)


q_table = np.zeros([env.observation_space["agent"].n, env.action_space.action_spaces[-1].n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# up, down, left, right
#
# For plotting metrics
all_epochs = []
all_penalties = []


target_position = np.array([35, 5])
agent_position = np.array([5, 35])


print("q_table_row", q_table.shape)
for i in range(1, 10):
    env.set_init_obervations({"agent": agent_position, "target": target_position})

    initial_state, info = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    terminated = False

    while not terminated:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[initial_state["agent"][0] * initial_state["agent"][1]]) # Exploit learned values

        # print(env.render())
        # time.sleep(0.2)
        
        next_state, reward, terminated, _, info = env.step(env.action_space.sample())
        # print(next_state, reward, terminated, _, info)
        
        old_value = q_table[initial_state["agent"][0] * initial_state["agent"][1], action]
        next_max = np.max(q_table[next_state["agent"][0] * next_state["agent"][1]])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[next_state["agent"][0] * next_state["agent"][1], action] = new_value

        if reward == 0:
            penalties += 1

        state = next_state
        epochs += 1
        
        print("PENALTIES", penalties)
        if penalties == 10:
            terminated = True
            
        
env.close()

print(q_table[3*35])