from enum import Enum
import random
import collections
import numpy as np
import torch
from src.segment_tree import MinSegmentTree, SumSegmentTree
from collections import deque
from typing import Tuple, Deque, List

class CollisionEvent(Enum):
    NOTHING = 1
    DEAD = 2
    WON = 3
    EAT = 4

class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self
    



class ReplayBuffer:
    def __init__(self, obs_size: int, capacity: int, n_step: int = 1, gamma: float = 0.99):
        """
        Initializes the replay buffer with a fixed capacity and N-step learning.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.
            n_step (int): The number of steps to look ahead for N-step learning.
            gamma (float): The discount factor for rewards.
        """
        self.obs_size = obs_size
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.done_deq = deque(maxlen=capacity)

        # N-step learning buffer
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Stores an experience tuple in the replay buffer with N-step processing.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The state after taking the action.
            done (bool): Whether the episode has ended.
        """
        # Create a transition and add it to the N-step buffer
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        # If N-step buffer is not yet filled, don't add to main buffer
        if len(self.n_step_buffer) < self.n_step:
            return None

        # Get N-step information from the buffer
        n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        n_step_state, n_step_action = self.n_step_buffer[0][:2]

        # Store the N-step transition in the main buffer
        self.states.append(n_step_state)
        self.actions.append(n_step_action)
        self.rewards.append(n_step_reward)
        self.next_states.append(n_step_next_state)
        self.done_deq.append(n_step_done)
        return self.n_step_buffer[0]

    def sample(self, batch_size: int) -> dict:
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            dict: A dictionary with states, actions, rewards, next_states, and dones as separate arrays.
        """
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        return self.sample_by_idx(indices)

    def sample_by_idx(self, indices):
        batch = {
            'states': np.array([self.states[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_states': np.array([self.next_states[i] for i in indices]),
            'dones': np.array([self.done_deq[i] for i in indices]),
        }
        return batch

    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float) -> Tuple[float, np.ndarray, bool]:
        """
        Computes the N-step reward, next state, and done flag.

        Args:
            n_step_buffer (Deque): A deque of stored transitions.
            gamma (float): The discount factor for rewards.

        Returns:
            Tuple: N-step reward, next state, and done flag.
        """
        # Initialize with the last transition in the buffer
        rew, next_obs, done = n_step_buffer[-1][-3:]

        # Traverse the buffer backwards to calculate the N-step reward
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.
        """
        return len(self.states)


    
    
class PriorityBuffer:
    """Prioritized Experience Replay Buffer."""

    def __init__(self, obs_dim, size, alpha=0.6):
        """Initialize the buffer with observation dimension, size, batch size, and alpha parameter for prioritization."""
        self.obs_dim = obs_dim
        self.max_size = size
        self.size = 0
        self.alpha = alpha
        self.max_priority = 1.0
        self.tree_ptr = 0

        # Initialize replay buffers
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size,3), dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        # Calculate tree capacity
        tree_capacity = 1
        while tree_capacity < size:
            tree_capacity *= 2

        # Initialize segment trees for sum and min priorities
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs, act, rew, next_obs, done):
        """Store a new experience in the buffer."""
        # Store the experience in the respective buffers
        
        self.obs_buf[self.tree_ptr] = obs
        self.next_obs_buf[self.tree_ptr] = next_obs
       
        self.acts_buf[self.tree_ptr] = act
        self.rews_buf[self.tree_ptr] = rew
        self.done_buf[self.tree_ptr] = done
        self.size = min(self.size + 1, self.max_size)
        # Set max priority for the new experience
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha

        # Update tree pointer
        self.tree_ptr = (self.tree_ptr + 1) % self.size

        

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences based on priority."""
        indices = self._sample_proportional(batch_size)
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        if self.size<100:
            return {
            'states': [],
            'actions':[],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'weights':[],
            'indices': []
        }
        #state, action, reward, next_state, done
        return {
            'states': obs,
            'actions': acts,
            'rewards': rews,
            'next_states': next_obs,
            'dones': done,
            'weights': weights,
            'indices': indices
        }
        

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled experiences."""

        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size):
        """Sample indices based on the proportionality of their priorities."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices
    


    def _calculate_weight(self, idx, beta):
        """Calculate the weight for the given sample index."""
        p_min = self.min_tree.min() / self.sum_tree.sum(0, self.size - 1)
        max_weight = (p_min * self.size) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum(0, self.size - 1)
        weight = (p_sample * self.size) ** (-beta)
        return weight / max_weight
    
    def __len__(self) -> int:
        """
        Returns the current size of the buffer.
        """
        return self.size

    