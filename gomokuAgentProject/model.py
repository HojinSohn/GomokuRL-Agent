import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

board_size = [15, 15]
action_size = 15 * 15

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)  # keeps size
        self.pool = nn.MaxPool2d(3, 1, padding=1)  # keeps size
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)  # keeps size
        self.fc1 = nn.Linear(16 * 15 * 15, 120)
        self.fc2 = nn.Linear(120, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()