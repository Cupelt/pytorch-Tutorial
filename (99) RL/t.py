import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np
import random

import time

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make("FrozenLake-v3", render_mode="human")

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    print(indices)
    return random.choice(indices)

print(np.array([[0, 1, 1, 0],[0, 1, 0, 1]]).__len__())