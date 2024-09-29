import copy
import random
import numpy as np
import torch
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from src.agents.base_agent import baseAgent
from src.agents.dueling_dqn import DuelingDQNAgent
from src.utils import PriorityBuffer, ReplayBuffer


class RainbowDQNAgent(DuelingDQNAgent):
    def __init__(self, params):
        super().__init__(params)
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.atom_size = params['atom_size']
        self.batch_size = params['batch_size']
        self.prior_eps = 1e-6
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        self.use_n_step = True if params['n_step'] > 1 else False
        if self.use_n_step:
            self.n_step = params['n_step']
            self.memory_n = ReplayBuffer(
                self.state_collector.state_size, 
                params['memory_size'],
                n_step=params['n_step'],
                gamma=params['gamma'])
            
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
        

    
    def prepare_memory(self, minibatch):
        # Extract individual components from the experience batch

        states_tensor = torch.tensor(minibatch['states'], dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(minibatch['actions'], dtype=torch.long).to(self.device).argmax(dim=1)
        rewards_tensor = torch.tensor(minibatch['rewards'], dtype=torch.float32).reshape(-1, 1).to(self.device)
        next_states_tensor = torch.tensor(minibatch['next_states'], dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(minibatch['dones'], dtype=torch.float32).reshape(-1, 1).to(self.device)
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)

    def _compute_dqn_loss(self, samples, gamma) -> torch.Tensor:
        """Return categorical DQN loss."""

        state, action, reward, next_state, done = self.prepare_memory(samples)

        # Categorical DQN variables
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)  # Atom width

        with torch.no_grad():
            next_action = self.model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            next_dist = next_dist[range(state.shape[0]), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (state.shape[0] - 1) * self.atom_size, state.shape[0]
                ).long()
                .unsqueeze(1)
                .expand(state.shape[0], self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.model.dist(state)
        log_p = torch.log(dist[range(state.shape[0]), action])

        loss = -(proj_dist * log_p).sum(1)

        return loss

    def replay(self, memory: collections.deque, batch_size: int) -> None:

        minibatch = self.memory.sample(batch_size)
        if len(minibatch['indices']) < self.batch_size:
            return
        weights = torch.FloatTensor(
            minibatch["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = minibatch["indices"]
        self.game_counter += 1

        # self.model.train()
        # torch.set_grad_enabled(True)

        # Use _compute_dqn_loss method to compute the Categorical DQN loss
        elementwise_loss = self._compute_dqn_loss(minibatch, self.gamma)
        loss = torch.mean(elementwise_loss * weights)
        
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_by_idx(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.writer.add_scalar("Loss", loss.item(), self.game_counter)
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()  # Adjust the learning rate

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.model.reset_noise()
        self.target_model.reset_noise()

        # Update the target network periodically
        if self.game_counter % self.target_update_frequency == 0:
            self.update_target_network()
