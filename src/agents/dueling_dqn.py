import copy
import random
import numpy as np
from statistics import mean, stdev
import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from src.agents.base_agent import baseAgent
from src.agents.double_dqn import DoubleDQNAgent 
from src.utils import PriorityBuffer, ReplayBuffer


class DuelingDQNAgent(DoubleDQNAgent):
    def __init__(self, params):
        super().__init__(params)
        
        self.memory = PriorityBuffer(self.state_collector.state_size, params['memory_size'])
        self.prior_eps = 1e-6
        self.gradient_clip_value = params['gradient_clip_value']
        self.gamma_n_learning = params['gamma']
        self.use_n_step = True if params['n_step'] > 1 else False
        if self.use_n_step:
            self.n_step = params['n_step']
            self.memory_n = ReplayBuffer(
                self.state_collector.state_size, 
                params['memory_size'],
                n_step=params['n_step'],
                gamma=params['gamma_n_learning'])
            
        self.transition = list()
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        

        self.transition = [state, action, reward, next_state, done]
        
        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.transition)
        # 1-step transition
        else:
            one_step_transition = self.transition

        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)
        
    def _compute_dqn_loss(self, samples, gamma):
        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = self.prepare_memory(samples)
        self.model.train()
        torch.set_grad_enabled(True)

        targets = rewards_tensor.clone()
        next_state_actions = torch.argmax(self.model.forward(next_states_tensor), dim=1)
        next_state_values = self.target_model.forward(next_states_tensor).gather(
            1, next_state_actions.unsqueeze(1)).squeeze(1)
        not_done_mask = ~dones_tensor
        targets[not_done_mask] += gamma * next_state_values[not_done_mask]

        outputs = self.model.forward(states_tensor)
        actions_indices = torch.argmax(actions_tensor, dim=1).unsqueeze(1)

        target_f = outputs.clone().detach()
        target_f.scatter_(1, actions_indices, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        elementwise_loss = F.huber_loss(outputs, target_f, reduction="none")
        elementwise_loss = torch.max(elementwise_loss, dim=1).values
        return elementwise_loss
    def replay(self, memory: collections.deque, batch_size: int) -> None:
        if len(self.memory)==0:
            return
        minibatch = self.memory.sample(batch_size)
        indices = minibatch["indices"]
        if len(indices)<batch_size:
            return
        weights = torch.FloatTensor(
            minibatch["weights"].reshape(-1, 1)
        ).to(self.device)
        
        self.game_counter += 1

        elementwise_loss = self._compute_dqn_loss(minibatch, self.gamma)
        loss = torch.mean(elementwise_loss * weights)
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_by_idx(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            loss = torch.mean(elementwise_loss * weights)
        loss.backward()
        self.losses.append(loss.item())

        if len(self.losses) >= self.loss_step_interval:
            avg_loss = mean(self.losses)
            std_loss = stdev(self.losses) if len(self.losses) > 1 else 0.0
            self.writer.add_scalar("Average Loss", avg_loss, self.game_counter)
            self.writer.add_scalar("Loss StdDev", std_loss, self.game_counter)
            self.writer.add_scalar("Learning rate", self.scheduler.get_last_lr()[0], self.game_counter)
            self.losses.clear()
        clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
        self.optimizer.step()
        self.scheduler.step()  # Adjust the learning rate
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        

        self.model.reset_noise()
        self.target_model.reset_noise()

        if self.game_counter % self.target_update_frequency == 0:
            self.update_target_network()
    # def calculate_epsilon(self, counter_games: int, minimum_epsilon: float = 0.0) -> None:
    #     # NoisyNet instead of epsilon
    #     self.epsilon = 0 
