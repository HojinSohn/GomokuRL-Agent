import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

action_size = 9 * 9

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)  # keeps size
        self.fc1 = nn.Linear(16 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, action_size)
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to produce smaller initial Q-values"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization with small gain for conv layers
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            
            elif isinstance(module, nn.Linear):
                if module == self.fc2:  # Final layer (Q-value output)
                    nn.init.xavier_uniform_(module.weight, gain=0.01)  # Extra small
                    nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
