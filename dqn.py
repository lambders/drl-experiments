import random
import torch

class DQN(nn.Module):

    def __init__(self):
        self.h = 84
        self.w = 84
        self.batch = 4
        self.actions = 2
        self.act = torch.nn.ReLU()
        

    def build():
        """
        """
        self.conv1 = torch.nn.Conv2d(self.batch, 32, 8, 4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.fc4 = torch.nn.Linear(3136, 512)
        self.fc5 = torch.nn.Linear(512, self.actions)



    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self):

    def sample(self):
        return random.sample(self.memory)