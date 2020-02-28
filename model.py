import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, length):
        super(DQN, self).__init__()

        lin1 = nn.Linear(length, length//3)
        lin2 = nn.Linear(length//3, length//2)
        lin3 = nn.Linear(length//2, length)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return F.relu(self.lin3(x))