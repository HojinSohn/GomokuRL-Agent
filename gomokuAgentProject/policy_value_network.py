import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

action_size = 9 * 9
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*9*9, 9*9)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*9*9, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Action policy
        act_x = F.relu(self.act_conv1(x))
        act_x = act_x.view(-1, 4 * 9 * 9)  # Flatten
        act_x = self.act_fc1(act_x)
        action_probs = F.log_softmax(act_x, dim=1)  # Log-Softmax for action probabilities

        # State value
        val_x = F.relu(self.val_conv1(x))
        val_x = val_x.view(-1, 2 * 9 * 9)  # Flatten
        val_x = F.relu(self.val_fc1(val_x))
        val_x = F.tanh(self.val_fc2(val_x))  # Final state value

        return action_probs, val_x

class PolicyValueNetwork():
    def __init__(self, device=torch.device("cpu"), learning_rate=0.001):
        self.device = device
        self.model = Model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def update_learning_rate(self, new_learning_rate):
        """
        Update the learning rate of the optimizer.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def policy_value(self, boards):
        """
        Get action probabilities and state values for a batch of boards.
        """
        input_tensors = torch.stack([self.convert_board_to_3channel(board) for board in boards]).to(self.device)
        with torch.no_grad():
            action_probs, state_values = self.model(input_tensors)

        action_probs = np.exp(action_probs.cpu().numpy())  # Convert log probabilities to probabilities
        state_values = state_values.cpu().numpy()

        return action_probs, state_values

    def get_action_probs(self, board):
        """
        Convert the board state to a 3-channel format and get action probabilities.
        """
        legal_moves = board == 0
        legal_moves = legal_moves.flatten()
        legal_moves_indices = np.where(legal_moves)[0]  # indices of legal moves
        input_tensor = self.convert_board_to_3channel(board).to(self.device)

        with torch.no_grad():
            probs, _ = self.model(input_tensor)

        # reduce batch dimension
        probs = probs.squeeze(0).cpu().numpy()
        probs = np.exp(probs)  # Convert log probabilities to probabilities

        # Filter action probabilities to only include legal moves
        return list(zip(legal_moves_indices, probs[legal_moves_indices]))
    
    def get_state_value(self, board):
        """
        Convert the board state to a 3-channel format and get the state value.
        """
        input_tensor = self.convert_board_to_3channel(board).to(self.device)
        with torch.no_grad():
            _, state_value = self.model(input_tensor)

        return state_value.item()
    
    def convert_board_to_3channel(self, board):
        """
        Convert the board state to a 3-channel format:
        - Channel 1: Player 1's stones (1 for player 1's stone, 0 otherwise)
        - Channel 2: Player 2's stones (1 for player 2's stone, 0 otherwise)
        - Channel 3: Non-empty cells (1 for non-empty cells, 0 otherwise)
        """
        input_tensor = np.zeros((3, 9, 9), dtype=np.float32)
        input_tensor[0] = (board == 1).astype(np.float32)  # Current player's stones
        input_tensor[1] = (board == -1).astype(np.float32)  # Opponent's stones
        input_tensor[2] = (board != 0).astype(np.float32)  # Non-empty cells
        return torch.from_numpy(input_tensor).to(self.device) 
    
    def save(self, path='models/model.pth'):
        """
        Save the model to the specified path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path='models/model.pth'):
        """
        Load the model from the specified path.
        """
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}. Starting with a new model.")
    
    def train_step(self, boards, mcts_act_probs, values):
        """ Given a batch of boards, action probabilities, and values, train the model.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Convert to tensors
        boards_tensor = torch.stack([self.convert_board_to_3channel(board) for board in boards]).to(self.device)
        act_probs_tensor = torch.from_numpy(np.array(mcts_act_probs, dtype=np.float32)).to(self.device)
        values_tensor = torch.from_numpy(np.array(values, dtype=np.float32)).to(self.device)

        # Forward pass
        action_log_probs, state_values = self.model(boards_tensor)

        # Calculate loss
        # Cross Entropy Loss for policy (probability distribution)
        policy_loss = -torch.mean(torch.sum(act_probs_tensor * action_log_probs, 1))
        # Mean Squared Error Loss for value (state value)
        value_loss = self.criterion(state_values.squeeze(), values_tensor)

        # Define loss function: value loss + policy loss
        loss = policy_loss + value_loss

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        # For checking entropy of action probabilities given by the model
        entropy = -torch.sum(torch.exp(action_log_probs) * action_log_probs, dim=1).mean()

        return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()