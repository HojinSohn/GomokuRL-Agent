import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

board_size = [15, 15]
action_size = 15 * 15

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)  # keeps size
        self.fc1 = nn.Linear(16 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x