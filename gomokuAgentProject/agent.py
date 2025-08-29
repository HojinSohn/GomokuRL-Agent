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
    def __init__(self, device=torch.device("cpu"), learning_rate=0.002, mcts_iterations=10000):
        self.device = device

        self.policy_value_network = PolicyValueNetwork(device=device, learning_rate=learning_rate)

        self.mcts = MCTS(self.policy_value_network, iterations=mcts_iterations)

        self.memory = deque(maxlen=10000)  # Memory to store samples

        self.kl_targ = 0.02

        self.learning_rate = learning_rate

        self.lr_multiplier = 1.0  # Learning rate multiplier for adaptive learning rate

    def get_action_and_probs(self, game: Game, turn: int):
        """
        Get action based on MCTS + neural network model.
        It will call get action from MCTS class by passing the current state and model (based on turn).
        When the probability distribution of moves is returned, it will sample an action based on the probabilities.
        """
        # Get the action probabilities from MCTS
        mcts_action_probs = self.mcts.get_action_probs(game, turn)

        # normalize in case for imprecision
        mcts_action_probs = mcts_action_probs / np.sum(mcts_action_probs)

        # Sample an action based on the probabilities
        action = np.random.choice(len(mcts_action_probs), p=mcts_action_probs)
        
        return action, mcts_action_probs

    def save_sample(self, data):
        # current_board, action, value_place
        current_board, act_probs, value_place = data
        equivalent_states = self.get_equivalent_states(current_board, act_probs)
        for state_equiv, act_probs_equi in equivalent_states:
            data = (state_equiv, act_probs_equi, value_place)
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

        print(f"Loaded {len(self.memory)} samples from memory.")

    def save_model(self):
        """
        Save the model to a file.
        """
        self.policy_value_network.save()

    def load_model(self, path='models/model.pth'):
        """
        Load the model from a file.
        """
        self.policy_value_network.load(path)

    def get_equivalent_states(self, state, act_probs):
        """ Get equivalent states by rotating and flipping the states.
        """
        equivalent_states = []
        # reshape act_probs to 2D
        act_probs_2d = act_probs.reshape(9, 9)
        for i in range(4):
            # Rotate the current state and next state for each sample in the batch
            rotated_state = np.rot90(state, k=i)
            rotated_act_probs = np.rot90(act_probs_2d, k=i)
            equivalent_states.append((rotated_state, rotated_act_probs.flatten()))

            # Also consider flipping the states horizontally
            flipped_state = np.fliplr(rotated_state)
            flipped_act_probs = np.fliplr(rotated_act_probs)
            equivalent_states.append((flipped_state, flipped_act_probs.flatten()))

        return equivalent_states

    def train_step(self, batch_size=64, num_epochs=5):
        """ Get a random batch of samples from memory and train the model.
        """
        if len(self.memory) < batch_size:
            print("Not enough samples in memory to train.")
            return
        
        # Randomly sample a batch from memory
        batch = random.sample(self.memory, batch_size)
        
        # Prepare the data for training
        boards, act_probs, values = zip(*batch)

        original_act_probs, original_values = self.policy_value_network.policy_value(boards)
        for _ in range(num_epochs):
            loss, policy_loss, value_loss, entropy = self.policy_value_network.train_step(boards, act_probs, values)
            new_act_probs, _ = self.policy_value_network.policy_value(boards)
            # Calculate KL Divergence
            kl = np.mean(np.sum(original_act_probs * (
                    np.log(original_act_probs + 1e-10) - np.log(new_act_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 3:
                print(f"KL Divergence {kl} is too high, stopping training")
                break

        # adjust the learning rate based on KL divergence
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # Update the learning rate
        self.policy_value_network.update_learning_rate(self.learning_rate * self.lr_multiplier)

        return loss, policy_loss, value_loss, entropy
