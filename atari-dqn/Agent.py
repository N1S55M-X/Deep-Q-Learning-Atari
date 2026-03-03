import random
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from Network import DQN   #model class

class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_size = action_size   # 🔥 STORE IT

        self.local_net = DQN(action_size).to(self.device)
        self.target_net = DQN(action_size).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=1e-4)

        self.memory = deque(maxlen=50000)
        self.batch_size = 32
        self.gamma = 0.99
        self.update_counter = 0

    def act(self, state, epsilon):
        state = state.unsqueeze(0).to(self.device)

        if random.random() < epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.local_net(state)
        return q_values.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).float().to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).float().to(self.device)

        next_q = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + self.gamma * next_q * (1 - dones)

        q_expected = self.local_net(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % 1000 == 0:
            self.target_net.load_state_dict(self.local_net.state_dict())
