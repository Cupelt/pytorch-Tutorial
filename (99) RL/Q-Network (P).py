import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_episodes = 2000
learning_rate = 0.1
dis = torch.tensor(.99, dtype=torch.float32).to(device)

n_obserbations = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = nn.Linear(n_obserbations, n_actions).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(agent.parameters(), learning_rate)

rList = []
num_episodes = 100

for i in range(1000):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)

    rAll = 0
    local_loss = []
    done = False
    
    e = 1. / ((i/50)+10)

    while not done:
        Qpred = agent(state)
        target_Q = Qpred.clone().detach()

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(Qpred).cpu().numpy())

        # Get new state and reward from environment
        new_state, reward, done, _, _ = env.step(action)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(device)

        # if done and reward == 0 : reward = -1

        # Update Q-table with new knowledge using learning rate
        if done:
            target_Q[action] = -100
        else:
            new_Qpred = agent(new_state).clone().detach()
            target_Q[action] = torch.tensor(reward, dtype=torch.float32).to(device) + dis * torch.max(new_Qpred)
        
        loss = criterion(Qpred, target_Q)
        local_loss.append(float(loss.detach().cpu().numpy()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward

        state = new_state
    rList.append(rAll)

    print("Episode : {:4}, Reward : {}".format(i, rAll))

plt.plot(range(len(rList)), rList, color="green")
plt.show()