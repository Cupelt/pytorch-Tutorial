    import gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import random
    import math

    from collections import namedtuple, deque

    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # HyperParams
    batch_size = 128
    gamma = .99
    learning_rate = 1e-4
    tau = 0.005

    save_rate = 1
    save_path = "./(99) RL/pacman_model"

    eps_start = 0.9
    eps_end = 0.005
    eps_decay = 1000

    num_episodes = 1000
    max_step = 1000

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

    class Agent(nn.Module):

        def __init__(self, input, output):
            super(Agent, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(input, 32, kernel_size=4, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )

            self.layer2 = nn.Sequential(
                nn.Linear(32 * 26 * 20, 128),
                nn.ReLU(),
                nn.Linear(128, output)
            )
        def forward(self, x):
            x = self.layer1(x)
            x = torch.flatten(x, start_dim=1)
            x = self.layer2(x)
            return x

    n_observations = env.observation_space.shape[2]
    n_actions = env.action_space.n

    agent_net = Agent(n_observations, n_actions)

    target_net = Agent(n_observations, n_actions)
    target_net.load_state_dict(agent_net.state_dict())

    optimizer = optim.Adam(agent_net.parameters(), learning_rate)
    criterion = nn.SmoothL1Loss()

    memory = ReplayMemory(50000)

    episode_rewards = []

    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = agent_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = reward_batch + gamma * next_state_values

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(agent_net.parameters(), 100)
        optimizer.step()

    steps_done = 0

    def select_action(state):
        global steps_done

        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(agent_net(state)).view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device)

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, 3, 210, 160)

        for i_step in range(max_step):
            action = select_action(state)

            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).view(1, 3, 210, 160)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            agent_net_state_dict = agent_net.state_dict()
            for key in agent_net_state_dict:
                target_net_state_dict[key] = agent_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(i_step)
                break
        
        if i_episode % save_rate == 0:
            torch.save(agent_net.state_dict(), f'{save_path}/policy-model-{i_episode}.pt')
            torch.save(target_net.state_dict(), f'{save_path}/target-model-{i_episode}.pt')
        
        print("Episode : {}, Reward : {}".format(i_episode, episode_rewards[i_episode]))