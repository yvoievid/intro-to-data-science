import random
import time
import gymnasium as gym
import numpy as np
# import gym_examples
from operation_simulation.wrappers import RelativePosition

random.seed(1)
env = gym.make('operation_simulation/SafePath-v0', render_mode = "human", grid_size=40, window_size=1024)
wrapped_env = RelativePosition(env)
done_n = [False for _ in range(100)]


print("wrapped_env.observation_space[agent].n",wrapped_env.observation_space["agent"].n)
print("wrapped_env.action_space.action_spaces[-1].n",wrapped_env.action_space.action_spaces[-1].n)

q_table = np.zeros([env.observation_space["agent"].n, env.action_space.action_spaces[-1].n])

obs_n = env.reset()
while not all(done_n):
    print(env.render())
    time.sleep(0.2)
    # print("wrapped_env.action_space.sample()",wrapped_env.action_space.sample())
    
    observation, reward, terminated, _, info = env.step(env.action_space.sample())
    print(observation, reward, terminated, _, info)
    
env.close()