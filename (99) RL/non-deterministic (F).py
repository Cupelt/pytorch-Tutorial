import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':True}
)
env = gym.make("FrozenLake-v3", render_mode = "human")

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
num_episodes = 2000
gamma = .9
lr = 0.1

for i in range(num_episodes):
    # Reset environment and get first new observation
    state, _ = env.reset()
    done = False

    e = 1. / ((i/100)+1)

    # The Q-Table learning algorithm
    while not done:
        
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        # action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        # Get new state and reward from environment
        new_state, reward, done, _, _ = env.step(action)
        # Update Q-table with new knowledge using learning rate
        Q[state, action] = (Q[state, action]-1) * Q[state, action] \
		    + lr * (reward + gamma * np.max(Q[new_state, :]))
        state = new_state
    print("Episode {:3d}/{:3d}".format(i, num_episodes))