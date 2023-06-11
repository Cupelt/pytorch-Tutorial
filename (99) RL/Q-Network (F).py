import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make("FrozenLake-v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Hyper parameters
num_episodes = 2000
learning_rate = 0.1
dis = torch.tensor(.99, dtype=torch.float32).to(device)

# matplotlib를 통해 통계를 확인하기 위한 리스트
rList = []

n_obserbations = env.observation_space.n
n_actions = env.action_space.n

agent = nn.Linear(n_obserbations, n_actions).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(agent.parameters(), learning_rate)

def one_hot(x):
    x = torch.tensor(x)

    out = torch.zeros_like(F.one_hot(torch.tensor(n_obserbations - 1)), dtype=torch.float32)
    out.scatter_(-1, x.unsqueeze(dim = -1), 1)

    return out

for i in range(num_episodes):
    # Reset environment and get first new observation
    state, _ = env.reset()
    done = False

    # 데이터 기록을 위한 변수
    rAll = 0
    local_loss = []

    e = 1. / ((i/50)+10)

    while not done:
        Qpred = agent(one_hot(state).to(device))
        target_Q = Qpred.clone().detach()

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = int(torch.argmax(Qpred).cpu().numpy())

        # Get new state and reward from environment
        new_state, reward, done, _, _ = env.step(action)

        # if done and reward == 0 : reward = -1

        # Update Q-table with new knowledge using learning rate
        if done:
            target_Q[action] = reward
        else:
            new_Qpred = agent(one_hot(new_state).to(device)).clone().detach()
            target_Q[action] = torch.tensor(reward, dtype=torch.float32).to(device) + dis * torch.max(new_Qpred)
        
        loss = criterion(Qpred, target_Q)
        local_loss.append(float(loss.detach().cpu().numpy()))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward

        state = new_state
    rList.append(rAll)
    print("Episode {:3d}/{:3d}, loss: {:>.9}".format(i, num_episodes, np.sum(local_loss)/len(local_loss)))

plt.title("Success Rate : {}".format(len(np.nonzero(rList)[0])/len(rList)))
plt.bar(range(len(rList)), rList, color="green")
plt.show()