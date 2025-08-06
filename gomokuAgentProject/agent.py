from collections import deque

import numpy as np
from model import Model
import torch
from torch import nn
import random
from gui_play import GUI

class DQNAgent():
    def __init__(self, device=torch.device("cpu")):
        self.device = device

        self.model1 = Model().to(device)
        self.target_model1 = Model().to(device)
        self.target_model1.load_state_dict(self.model1.state_dict())
        self.memory1 = deque(maxlen=2000)

        self.model2 = Model().to(device)
        self.target_model2 = Model().to(device)
        self.target_model2.load_state_dict(self.model2.state_dict())
        self.memory2 = deque(maxlen=2000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.batch_size = 32
        self.learning_rate = 0.001

        self.criterion = nn.MSELoss()

    def get_action(self, state, turn : int):
        if torch.rand(1).item() < self.epsilon:
            # Exploration: choose a random action
            # currently returns a random action from the action space
            return torch.randint(0, 15 * 15, (1,)).item()
        else:
            # Exploitation: choose the best action from the model
            with torch.no_grad():
                if turn == 1:
                    state_np = np.array(state[turn])
                    states_tensor = torch.tensor(state_np, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
                    q_values = self.model1(states_tensor)
                else :
                    # turn == -1
                    state_np = np.array(state[turn])
                    states_tensor = torch.tensor(state_np, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)
                    q_values = self.model2(states_tensor)
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

    def train_model(self, model, target_model, memory):
        if len(memory) < self.batch_size:
            return

        sample = random.sample(memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        states_tensor = torch.from_numpy(np.array(states)).float().unsqueeze(1).to(self.device)
        actions_tensor = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards_tensor = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states_tensor = torch.from_numpy(np.array(next_states)).float().unsqueeze(1).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float).to(self.device)

        q_inference = model(states_tensor).gather(1, actions_tensor).squeeze(1)
        q_next = target_model(next_states_tensor).detach().max(1)[0]
        q_target = rewards_tensor + self.gamma * q_next * (1 - done_tensor)
        loss = self.criterion(q_inference, q_target)
        # print("Loss:", loss.item())
        model.backward(loss)


    def update_target(self, model, target_model):
        '''
            This function updates the target model with the weights of the current model.
            It is called periodically to keep the target model in sync with the current model.
        '''
        target_model.load_state_dict(model.state_dict())
        print("Target model updated.")
