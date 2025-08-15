from collections import deque
from datetime import datetime
import glob
import os

import numpy as np
from mcts import MCTS
from model import Model
import torch
from torch import nn
import random
from gui_play import GUI

class DQNAgent():
    def __init__(self, device=torch.device("cpu"), init_epsilon=0.95, load_model=False, load_memory=False):
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
        self.memory1 = deque(maxlen=80000)

        self.model2 = Model().to(device)
        if load_model:
            print("Loading model2 from file...")
            self.model2.load_state_dict(torch.load('models/model2.pth', map_location=device)) # Load the model previously trained
        self.target_model2 = Model().to(device)
        self.target_model2.load_state_dict(self.model2.state_dict())
        self.target_model2.eval()  # Set target model to evaluation mode
        self.memory2 = deque(maxlen=80000)

        if load_memory:
            print("Loading memory from file...")
            self.load_memory()

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.learning_rate)

        self.criterion = nn.SmoothL1Loss()

        self.mcts = MCTS(iterations=50)

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
        input_tensor = np.zeros((batch_size, 3, 9, 9), dtype=np.float32)

        for i, state in enumerate(states):
            state_array = np.array(state)
            input_tensor[i, 0] = (state_array == 1).astype(np.float32)    # Player's stones
            input_tensor[i, 1] = (state_array == -1).astype(np.float32)   # Opponent's stones
            input_tensor[i, 2] = (state_array != 0).astype(np.float32)    # Non Empty cells

        return torch.from_numpy(input_tensor).to(device)

    def get_action(self, gomokuEnv, turn : int, not_tactical=False):
        if torch.rand(1).item() < self.epsilon:
            # Exploration: choose a random action
            # currently returns a random action from the action space
            return self.mcts.get_action(gomokuEnv.state[turn], not_tactical=not_tactical)
        else:
            # Exploitation: choose the best action from the model
            with torch.no_grad():
                if turn == 0:
                    input_tensor = self.convert_states_to_3channel([gomokuEnv.state[turn]], self.device)
                    q_values = self.model1(input_tensor)
                else :
                    # turn == 1
                    input_tensor = self.convert_states_to_3channel([gomokuEnv.state[turn]], self.device)
                    q_values = self.model2(input_tensor)
                action = torch.argmax(q_values).item()
            return action

    def rotate_action90(self, action, k):
        """
        Rotate the action index by 90 degrees k times.
        The action is represented as a single index in a 9x9 grid.
        """
        row = action // 9
        col = action % 9
        for _ in range(k):
            # 9 - col == new_row
            # row == new_col
            temp = col
            col = row
            row = 9 - temp - 1
        return row * 9 + col
    
    def flip_action(self, action):
        """
        Flip the action index horizontally.
        The action is represented as a single index in a 9x9 grid.
        """
        row = action // 9
        col = action % 9
        
        new_col = 9 - col - 1

        return row * 9 + new_col

    def get_equivalent_states(self, data):
        # current_state_record, action, reward, next_state_record, completed
        equivalent_states = []
        current_state_record, action, reward, next_state_record, completed = data
        for i in range(4):
            # Rotate the current state and next state
            rotated_current_state = np.rot90(current_state_record, k=i)
            rotated_next_state = np.rot90(next_state_record, k=i)
            rotated_action = self.rotate_action90(action, k=i)
            # Create the equivalent data entry
            equivalent_data = (rotated_current_state, rotated_action, reward, rotated_next_state, completed)
            equivalent_states.append(equivalent_data)

            # Also consider flipping the states horizontally
            flipped_current_state = np.fliplr(rotated_current_state)
            flipped_next_state = np.fliplr(rotated_next_state)
            flipped_action = self.flip_action(rotated_action)
            equivalent_flipped_data = (flipped_current_state, flipped_action, reward, flipped_next_state, completed)
            equivalent_states.append(equivalent_flipped_data)
        return equivalent_states

    def save_sample(self, data, turn : int):
        # get equivalent samples and save them in the corresponding memory
        equivalent_samples = self.get_equivalent_states(data)
        if turn == 0:
            self.memory1.extend(equivalent_samples)
        else:
            self.memory2.extend(equivalent_samples)

    def save_memory(self):
        """
        Save the memory to files for later use.
        """
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('memories', exist_ok=True)
        print("Saving memory to files...")
        np.save(f'memories/memory1_{date_time}.npy', np.array(self.memory1, dtype=object), allow_pickle=True)
        np.save(f'memories/memory2_{date_time}.npy', np.array(self.memory2, dtype=object), allow_pickle=True)
        print("Memory saved.")

    def load_memory(self):
        """
        Load multiple memory files into the agent's memory
        """
        memory1_files = glob.glob("memories/memory1*.npy")
        for memory_file in memory1_files:
            print(f"Loading memory from {memory_file}...")
            data = np.load(memory_file, allow_pickle=True)
            print(f"Loaded {len(data)} entries into memory1.")
            self.memory1.extend(data)  # assuming memory1 is a list

        memory2_files = glob.glob("memories/memory2*.npy")
        for memory_file in memory2_files:
            print(f"Loading memory from {memory_file}...")
            data = np.load(memory_file, allow_pickle=True)
            print(f"Loaded {len(data)} entries into memory2.")
            self.memory2.extend(data)  # assuming memory2 is a list
            
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
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.detach().item()
    
    def update_target_model(self, model, target_model):
        """
        Update the target model with the weights of the current model.
        This is called periodically to keep the target model in sync with the current model.
        """
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
