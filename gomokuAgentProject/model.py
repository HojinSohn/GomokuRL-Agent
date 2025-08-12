import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

board_size = [15, 15]
action_size = 15 * 15

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)  # keeps size
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)  # keeps size
        self.fc1 = nn.Linear(8 * 15 * 15, 128)
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
        x = x.view(x.size(0), -1)  # flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


# # TEST FUNCTION: Compare initialization magnitudes
# def test_initialization_magnitudes():
#     print("🧪 TESTING INITIALIZATION MAGNITUDES")
#     print("="*50)
    
#     models = {
#         "Original (Default)": Model,
#         "Conservative Init": Model,  # Uses the improved version above
#     }
    
#     # Create dummy input (empty board with 3 channels)
#     dummy_input = torch.zeros(1, 3, 15, 15)
    
#     for name, model_class in models.items():
#         if name == "Original (Default)":
#             model = model_class()
#         else:
#             model = model_class()
#             model._initialize_weights()  # Apply custom initialization
        
#         with torch.no_grad():
#             q_values = model(dummy_input)
#             max_q = q_values.max().item()
#             min_q = q_values.min().item()
#             mean_abs_q = q_values.abs().mean().item()
        
#         print(f"\n{name}:")
#         print(f"   Q-value range: [{min_q:.3f}, {max_q:.3f}]")
#         print(f"   Mean absolute Q: {mean_abs_q:.3f}")
#         print(f"   Max magnitude: {max(abs(min_q), abs(max_q)):.3f}")
        
#         reward_scale = 1.0  # Your win/lose reward
#         ratio = max(abs(min_q), abs(max_q)) / reward_scale
#         print(f"   Magnitude vs reward ratio: {ratio:.1f}x")
        
#         if ratio < 2:
#             print(f"   ✅ Excellent initialization!")
#         elif ratio < 5:
#             print(f"   👍 Good initialization")
#         elif ratio < 10:
#             print(f"   ⚠️  Could be better")
#         else:
#             print(f"   🚨 Too large - will slow training")

# if __name__ == "__main__":
#     test_initialization_magnitudes()