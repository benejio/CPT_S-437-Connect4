'''
Import this file using:
    import register_env

Importing this file lets the openAI gym register the connect 4 code as a valid gym to make.
Ex.
    env = gym.make("Connect4-v0")
'''

import gym
from gym.envs.registration import register

register(
    id="Connect4-v0",
    entry_point="connect4_env:Connect4Env",
)

