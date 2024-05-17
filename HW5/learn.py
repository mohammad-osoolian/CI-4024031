import gymnasium as gym
from time import sleep
env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(500):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    # sleep(1)
    if terminated:
        observation, info = env.reset()

env.close()
