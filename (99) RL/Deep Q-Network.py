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

register(
    id='FrozenLake-v3',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False}
)
env = gym.make("FrozenLake-v3")


# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Memory
# --------------------------------------------------------------------------------

# 우리 환경에서 단일 전환을 나타내도록 명명된 튜플.
# 그것은 화면의 차이인 state로 (state, action) 쌍을 (next_state, reward) 결과로 매핑합니다.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# 최근 관찰된 전이를 보관 유지하는 제한된 크기의 순환 버퍼.
# 또한 학습을 위한 전환의 무작위 배치를 선택하기위한 .sample () 메소드를 구현합니다.
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
# ---------------------------------------------------------------------------------

# Q-network 정의
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# 데이터 전처리
def one_hot(x):
    x = torch.tensor(x)

    out = torch.zeros_like(F.one_hot(torch.tensor(n_observations - 1)), dtype=torch.float32)
    out.scatter_(-1, x.unsqueeze(dim = -1), 1)

    return out

# 하이퍼파라미터
BATCH_SIZE = 128    # 리플레이 버퍼에서 샘플링된 트랜지션의 수
GAMMA = 0.99        # 할인 계수 (Discount factor)
EPS_START = 0.9     # 엡실론 시작 값
EPS_END = 0.05      # 엡실론 최종 값
EPS_DECAY = 1000    # 엡실론의 지수 감쇠(exponential decay) 속도 제어하며, 높을수록 감쇠 속도가 느림
TAU = 0.005         # 목표 네트워크의 업데이트 속도
LR = 1e-4           # 학습률

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.n
# 상태 관측 횟수를 얻습니다.
n_observations = env.observation_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    print("{}, {}".format(sample, eps_threshold))
    steps_done += 1
    if sample < eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(one_hot(state)))
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

select_action(1)