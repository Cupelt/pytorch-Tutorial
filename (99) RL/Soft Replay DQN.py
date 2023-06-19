from collections import namedtuple
import copy
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

num_episodes = 2000
learning_rate = 0.1
dis = torch.tensor(.99, dtype=torch.float32).to(device)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = nn.Linear(n_observations, n_actions).to(device)
target = copy.deepcopy(agent)

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(agent.parameters(), learning_rate)

rList = []
num_episodes = 1000

for i in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor([state], dtype=torch.float32).to(device)

    rAll = 0
    done = False

    replay = []
    
    e = 1. / ((i/50)+10)

    while not done:
        Qpred = agent(state)

        if np.random.rand(1) < e:
            action = torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)
        else:
            action = torch.argmax(Qpred).view(1, 1)

        # Get new state and reward from environment
        next_state, reward, done, _, _ = env.step(action.item())
        rAll += reward
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor([next_state], dtype=torch.float32).to(device)

        # Update Q-table with new knowledge using learning rate
        replay.append(Transition(state, action, next_state, reward))

        state = next_state

    rList.append(rAll)

    batch = Transition(*zip(*replay))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = agent(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros_like(reward_batch, device=device).squeeze(-1)
    with torch.no_grad():
        next_state_values[non_final_mask] = agent(non_final_next_states).max(1)[0]

    print(state_action_values.shape)
    print(expectation_reward.shape)
    expected_state_action_values


    train_Q = agent(torch.cat(batch.state))

    loss = criterion(train_Q, target_Q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Episode : {:4}, Reward : {}".format(i, rAll))

plt.plot(range(len(rList)), rList, color="green")
plt.show()