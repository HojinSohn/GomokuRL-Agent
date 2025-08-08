from collections import deque

import numpy as np
from mcts import MCTS
from model import Model
import torch
from torch import nn
import random
from gui_play import GUI

class DQNAgent():
    def __init__(self, device=torch.device("cpu"), init_epsilon=0.95, load_model=False):
        self.device = device

        self.gamma = 0.99
        self.epsilon = init_epsilon  # Initial exploration rate (saved from previous training)
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.98
        self.batch_size = 64
        self.learning_rate = 0.001

        self.model1 = Model().to(device)
        if load_model:
            print("Loading model1 from file...")
            self.model1.load_state_dict(torch.load('models/model1.pth', map_location=device)) # Load the model previously trained
        self.target_model1 = Model().to(device)
        self.target_model1.load_state_dict(self.model1.state_dict())
        self.target_model1.eval()  # Set target model to evaluation mode
        self.memory1 = deque(maxlen=2000)

        self.model2 = Model().to(device)
        if load_model:
            print("Loading model2 from file...")
            self.model2.load_state_dict(torch.load('models/model2.pth', map_location=device)) # Load the model previously trained
        self.target_model2 = Model().to(device)
        self.target_model2.load_state_dict(self.model2.state_dict())
        self.target_model2.eval()  # Set target model to evaluation mode
        self.memory2 = deque(maxlen=2000)

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.learning_rate)

        # Good for Q-learning (Huber loss - robust to outliers)
        self.criterion = nn.SmoothL1Loss()

        self.mcts = MCTS(iterations=10)

    def swap_players(self):
        """
        Swap the models and optimizers for training.
        This is useful for alternating turns in a two-player game.
        """
        self.model1, self.model2 = self.model2, self.model1
        self.target_model1, self.target_model2 = self.target_model2, self.target_model1
        self.memory1, self.memory2 = self.memory2, self.memory1
        self.optimizer1, self.optimizer2 = self.optimizer2, self.optimizer1

    def convert_states_to_3channel(self, states, device):
        batch_size = len(states)
        input_tensor = np.zeros((batch_size, 3, 15, 15), dtype=np.float32)

        for i, state in enumerate(states):
            state_array = np.array(state)
            input_tensor[i, 0] = (state_array == 1).astype(np.float32)    # Player 1 stones
            input_tensor[i, 1] = (state_array == -1).astype(np.float32)   # Player 2 stones
            input_tensor[i, 2] = (state_array != 0).astype(np.float32)    # Non Empty cells

        return torch.from_numpy(input_tensor).to(device)

    def get_action(self, state, turn : int):
        if torch.rand(1).item() < self.epsilon:
            # Exploration: choose a random action
            # currently returns a random action from the action space
            return self.mcts.get_action(np.array(state[turn]))
        else:
            # Exploitation: choose the best action from the model
            with torch.no_grad():
                if turn == 0:
                    input_tensor = self.convert_states_to_3channel([state[turn]], self.device)
                    q_values = self.model1(input_tensor)
                else :
                    # turn == 1
                    input_tensor = self.convert_states_to_3channel([state[turn]], self.device)
                    q_values = self.model2(input_tensor)
                action = torch.argmax(q_values).item()
            return action

    def save_sample(self, data, turn : int):
        if turn == 0:
            self.memory1.append(data)
        else:
            self.memory2.append(data)

    def save_model(self):
        torch.save(self.model1.state_dict(), 'models/model1.pth')
        torch.save(self.model2.state_dict(), 'models/model2.pth')
        print("Models saved.")

    def train_model(self, model, target_model, memory, optimizer):
        if len(memory) < self.batch_size:
            return

        sample = random.sample(memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        
        input_tensor = self.convert_states_to_3channel(states, self.device)
        next_state_input_tensor = self.convert_states_to_3channel(next_states, self.device)
        # states_tensor = torch.from_numpy(np.array(states)).float().unsqueeze(1).to(self.device)
        actions_tensor = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards_tensor = torch.from_numpy(np.array(rewards)).float().to(self.device)
        # next_states_tensor = torch.from_numpy(np.array(next_states)).float().unsqueeze(1).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float).to(self.device)

        q_inference = model(input_tensor).gather(1, actions_tensor).squeeze(1)
        q_next = target_model(next_state_input_tensor).detach().max(1)[0]
        q_target = rewards_tensor + self.gamma * q_next * (1 - done_tensor)
        loss = self.criterion(q_inference, q_target)
        # print("Loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().item()


    def update_target_model(self):
        '''
            This function updates the target model with the weights of the current model.
            It is called periodically to keep the target model in sync with the current model.
        '''
        self.target_model1.load_state_dict(self.model1.state_dict())
        self.target_model2.load_state_dict(self.model2.state_dict())
        self.target_model1.eval()
        self.target_model2.eval()
