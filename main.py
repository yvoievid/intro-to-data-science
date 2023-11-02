import random
import time
import gymnasium as gym
# import gym_examples
from operation_simulation.wrappers import RelativePosition

random.seed(1)
env = gym.make('operation_simulation/SafePath-v0', render_mode = "human", grid_size=40, window_size=1024)
wrapped_env = RelativePosition(env)
done_n = [False for _ in range(100)]

obs_n = wrapped_env.reset()
while not all(done_n):
    print(env.render())
    time.sleep(0.2)
    # print("wrapped_env.action_space.sample()",wrapped_env.action_space.sample())
    
    observation, reward, terminated, _, info = env.step(wrapped_env.action_space.sample())
    print(observation, reward, terminated, _, info)
    
env.close()