import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math

from collections import namedtuple, deque

env = gym.make("CartPole-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# HyperParameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LEARNING_RATE = 1e-4

num_episodes = 1000
max_step = 1000

# Replay Buffer
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Model
class DQN(nn.Module):

    def __init__(self, input, output):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

net_policy = DQN(n_observations, n_actions).to(device)
net_target = DQN(n_observations, n_actions).to(device)
net_target.load_state_dict(net_policy.state_dict())

optimizer = optim.Adam(net_policy.parameters(), LEARNING_RATE)
criterion = nn.MSELoss()

memory = ReplayMemory(10000)

episode_duration = []

# optimize Model
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = net_policy(state_batch).gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = net_target(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(net_policy.parameters(), 100)
    optimizer.step()

# Training
for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for i_step in range(max_step):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * i_step / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                action = torch.argmax(net_policy(state)).view(1, 1)
        else:
            action = torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)
        
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()
        
        target_net_state_dict = net_target.state_dict()
        policy_net_state_dict = net_policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        net_target.load_state_dict(target_net_state_dict)

        if done:
            episode_duration.append(i_step)
            break
    print("Episode : {}, Step : {}".format(i_episode, episode_duration[i_episode]))
        
        