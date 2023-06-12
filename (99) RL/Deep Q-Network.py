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

target_net = policy_net
target_net.load_state_dict(policy_net.state_dict())

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(target_net.parameters(), learning_rate)

replay_buffer = deque()
def replay_train(model, batch):
    policy_stack = torch.empty(0, dtype=torch.float32).reshape(0, n_actions).to(device)
    target_stack = torch.empty(0, dtype=torch.float32).reshape(0, n_actions).to(device)

    for state, action, reward, new_state, done in batch:
        Q = model(state)

        if done:
            Q[action] = reward
        else:
            Q[action] = reward + dis * torch.max(model(new_state))
        
        policy_stack = torch.vstack([policy_stack, model(state)])
        target_stack = torch.vstack([target_stack, Q])
    
    loss = criterion(policy_stack, target_stack)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

rList = []
num_episodes = 2000

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)

    rAll = 0
    local_loss = []
    done = False
    
    e = 1. / ((episode/50)+10)

    while not done:
        Qpred = policy_net(state)
        target_Q = Qpred.clone().detach()

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(Qpred).cpu().numpy())

        # Get new state and reward from environment
        new_state, reward, done, _, _ = env.step(action)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(device)

        replay_buffer.append((state, action, reward, new_state, done))
        if len(replay_buffer) > REPLAY_MEMORY:
            replay_buffer.popleft()

        rAll += reward
        state = new_state
    
    if episode % 10 == 1:
        for _ in range(50):
            batch = random.sample(replay_buffer, 10)
            loss = replay_train(policy_net, batch)

    rList.append(rAll)

    print("Episode : {:4}, Reward : {}".format(episode, rAll))

plt.plot(range(len(rList)), rList, color="green")
plt.show()