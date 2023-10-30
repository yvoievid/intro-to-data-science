import gym
import time 

env = gym.make('operation_simulation:SafePath-v0', grid_shape=(30, 30), n_agents=10)
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    time.sleep(0.2)
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
    
env.close()