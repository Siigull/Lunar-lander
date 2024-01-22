import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from collections import deque
import random

import numpy as np

import gym

from sys import platform
device = 'cuda' if platform == 'win32' else 'mps'

MAX_MEMORY = 20000
MIN_MEMORY = 100

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 30000

env = gym.make("LunarLander-v2")
observation, info = env.reset(seed=100)

env_test = gym.make("LunarLander-v2", render_mode="human")
env_test.reset(seed=142)

N_ACTIONS = env.action_space.n
N_STATE = len(observation)

class Memory():
   def __init__(self):
      self.buffer = deque(maxlen = MAX_MEMORY)

   def push(self, last_state, action, reward, next_state, terminated):
      self.buffer.append((last_state, action, reward, next_state, terminated))

   def sample(self, batch_size):
      last_state, action, reward, next_state, terminated = zip(*random.sample(self.buffer, batch_size))
      return last_state, action, reward, next_state, terminated
   
   def __len__(self):
      return len(self.buffer)

class Net(nn.Module):
   def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(N_STATE, 128)
      self.fc2 = nn.Linear(128, 32)
      self.fc3 = nn.Linear(32, 4)

   def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)

      return x
        
   def act(self, state, epsilon):
      if random.random() > epsilon:
         state = torch.Tensor(np.float32(state)).to(device)
         q_value = self.forward(state)
         action = q_value.argmax().item()

      else:
         action = random.randint(0, N_ACTIONS - 1)

      return action

net = Net().to(device)
optimizer = optim.RMSprop(net.parameters(), lr = 1e-3)

def compute_loss(memory, batch_size):
   last_state, action, reward, next_state, terminated = memory.sample(batch_size)

   last_state = torch.Tensor(np.float32(last_state)).to(device)
   next_state = torch.Tensor(np.float32(next_state)).to(device)
   action = torch.LongTensor(action).to(device)
   reward = torch.Tensor(reward).to(device)
   terminated = torch.Tensor(terminated).to(device)

   q_old = net(last_state)
   q_new = net(next_state)

   q_old = q_old.gather(1, action.unsqueeze(1)).squeeze(1)
   q_new = q_new.max(1)[0]

   q_expected = reward + 0.99 * q_new * (1 - terminated)

   loss = (q_old - q_expected.data).pow(2).mean()

   return loss


def test():
   observation, info = env_test.reset()

   while(1):
      action = net.act(observation, 0)  # User-defined policy function

      observation, reward, terminated, truncated, info = env_test.step(action)

      if terminated or truncated:
         break

memory = Memory()
episode = 1
steps = 0


net.train()
while episode < 1500:

   last_state = observation

   steps += 1
   epsilon = max(EPS_END, EPS_START - steps * 0.00001)

   action = net.act(observation, epsilon)  # User-defined policy function

   observation, reward, terminated, truncated, info = env.step(action)
   memory.push(last_state, action, reward, observation, terminated)
   
   if len(memory) > MIN_MEMORY:
      loss = compute_loss(memory, 32)

      # if steps % 1000 == 0:
      #    print(loss.data)
         
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

   if terminated or truncated:
      print(episode)
      episode += 1

      if len(memory) > MIN_MEMORY and episode % 50 == 0:
         with torch.no_grad():
            test()

      observation, info = env.reset()
      

while 1:
   input()

   test()

env.close()