from collections import deque
from datetime import datetime
import glob
import os

import numpy as np
from game import Game
from mcts import MCTS
from policy_value_network import PolicyValueNetwork
import torch
from torch import nn
import random
from gui_play import GUI

class Agent():
    def __init__(self, device=torch.device("cpu"), learning_rate=0.001, load_model=False, load_memory=False, mcts_iterations=10000):
        self.device = device

        self.policy_value_network = PolicyValueNetwork(device=device, learning_rate=learning_rate)

        self.mcts = MCTS(self.policy_value_network, iterations=mcts_iterations)

        self.memory = deque(maxlen=10000)  # Memory to store samples

    def get_action_and_probs(self, game: Game, turn: int):
        """
        Get action based on MCTS + neural network model.
        It will call get action from MCTS class by passing the current state and model (based on turn).
        When the probability distribution of moves is returned, it will sample an action based on the probabilities.
        """
        # Get the action probabilities from MCTS
        mcts_action_probs = self.mcts.get_action_probs(game, turn)

        # Sample an action based on the probabilities
        action = np.random.choice(len(mcts_action_probs), p=mcts_action_probs)
        
        return action, mcts_action_probs

    def save_sample(self, data):
        # current_board, action, value_place
        current_board, act_probs, value_place = data
        equivalent_states = self.get_equivalent_states(current_board)
        for state in equivalent_states:
            data = (state, act_probs, value_place)
            self.memory.append(data)

    def save_memory(self):
        """
        Save the memory to files for later use.
        """
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('memories', exist_ok=True)
        np.save(f'memories/memory_{date_time}.npy', np.array(self.memory, dtype=object), allow_pickle=True)

    def load_memory(self):
        """
        Load multiple memory files into the agent's memory
        """
        memory_files = glob.glob('memories/memory*.npy')
        if not memory_files:
            print("No memory files found. Starting with empty memory.")
            return
        
        for memory_file in memory_files:
            print(f"Loading memory from {memory_file}...")
            memory_data = np.load(memory_file, allow_pickle=True)
            self.memory.extend(memory_data.tolist())

    def save_model(self):
        """
        Save the model to a file.
        """
        self.policy_value_network.save()

    def get_equivalent_states(self, state):
        """ Get equivalent states by rotating and flipping the states.
        """
        equivalent_states = []
        for i in range(4):
            # Rotate the current state and next state for each sample in the batch
            rotated_state = np.rot90(state, k=i)
            equivalent_states.append(rotated_state)

            # Also consider flipping the states horizontally
            flipped_state = np.fliplr(rotated_state)
            equivalent_states.append(flipped_state)

        return equivalent_states

    def train_step(self, batch_size=32, num_epochs=5):
        """ Get a random batch of samples from memory and train the model.
        """
        if len(self.memory) < batch_size:
            print("Not enough samples in memory to train.")
            return
        
        # Randomly sample a batch from memory
        batch = random.sample(self.memory, batch_size)
        
        # Prepare the data for training
        boards, act_probs, values = zip(*batch)

        for _ in range(num_epochs):
            loss, entropy = self.policy_value_network.train_step(boards, act_probs, values)
        
        return loss, entropy
        