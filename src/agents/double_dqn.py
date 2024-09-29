import copy
import random
import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim

from src.agents.base_agent import baseAgent
from src.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    def __init__(self, params):
        super().__init__(params)
        self.target_model = copy.deepcopy(self.model)
        self.device = params['device']
        self.target_model.to(self.device)
        self.target_model.eval()
        self.update_target_network()
        self.target_update_frequency = params['target_update_frequency']

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, memory: collections.deque, batch_size: int) -> None:

        minibatch = self.memory.sample(batch_size)
        self.game_counter += 1

        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = self.prepare_memory(minibatch)
        self.model.train()
        torch.set_grad_enabled(True)

        targets = rewards_tensor.clone()
        next_state_actions = torch.argmax(self.model.forward(next_states_tensor), dim=1)
        next_state_values = self.target_model.forward(next_states_tensor).gather(
            1, next_state_actions.unsqueeze(1)).squeeze(1)
        not_done_mask = ~dones_tensor
        targets[not_done_mask] += self.gamma * next_state_values[not_done_mask]

        outputs = self.model.forward(states_tensor)
        actions_indices = torch.argmax(actions_tensor, dim=1).unsqueeze(1)

        target_f = outputs.clone().detach()
        target_f.scatter_(1, actions_indices, targets.unsqueeze(1))

        self.loss(outputs, target_f)

        if self.game_counter % self.target_update_frequency == 0:
            self.update_target_network()
