import copy
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from gym.envs.registration import register
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

env = gym.make("CartPole-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

dis = torch.tensor(0.99, dtype=torch.float32).to(device)
REPLAY_MEMORY = 50000

num_episodes = 2000
learning_rate = 0.1

policy_net = nn.Sequential(
    nn.Linear(n_observations, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, n_actions)
).to(device)

target_net = copy.deepcopy(policy_net)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(target_net.parameters(), learning_rate)

replay_buffer = deque()
def replay_train(predNet, targetNet, batch):
    policy_stack = torch.empty(0, dtype=torch.float32).reshape(0, n_actions).to(device)
    target_stack = torch.empty(0, dtype=torch.float32).reshape(0, n_actions).to(device)

    for state, action, reward, new_state, done in batch:
        Q = predNet(state).clone().detach()

        if done:
            Q[action] = reward
        else:
            Q[action] = torch.tensor(reward, dtype=torch.float32).to(device) + dis * torch.max(targetNet(new_state))
        
        policy_stack = torch.vstack([policy_stack, predNet(state)])
        target_stack = torch.vstack([target_stack, Q])
    
    loss = criterion(policy_stack, target_stack)    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

step_list = []
num_episodes = 2000

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)

    step_count = 0
    local_loss = []
    done = False
    
    e = 1. / ((episode / 10 ) + 1)

    while not done:
        Qpred = policy_net(state)

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(Qpred).cpu().numpy())

        # Get new state and reward from environment
        new_state, reward, done, _, _ = env.step(action)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
        if done:
            reward = -100

        replay_buffer.append((state, action, reward, new_state, done))
        if len(replay_buffer) > REPLAY_MEMORY:
            replay_buffer.popleft()

        step_count += 1
        state = new_state

        if step_count > 10000:
            break
    
    if episode % 10 == 1:
        for _ in range(50):
            batch = random.sample(replay_buffer, 10)
            loss = replay_train(policy_net, target_net, batch)
        
        target_net = copy.deepcopy(policy_net)

    step_list.append(step_count)

    print("Episode : {:4}, Step : {}".format(episode, step_count))

plt.plot(range(len(step_list)), step_list, color="green")
plt.show()