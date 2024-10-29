import register_env
import gym
import numpy as np

env = gym.make("Connect4-v0")

obs, info = env.reset()
done = False
total_reward = 0

print("Starting Connect 4 game:")
env.render()

while not done:
    # Select a random action
    action = env.action_space.sample()
    
    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Render the board state after the action
    print(f"\nAction taken: {action}")
    env.render()
    print(f"Reward: {reward}, Done: {done}")

print("\nGame over!")
print(f"Total Reward: {total_reward}")
