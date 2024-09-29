import random
import os
import numpy as np
import pandas as pd
from datetime import datetime
import collections
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from statistics import mean, stdev

from src.agents.base_agent import baseAgent
from src.utils import DummySummaryWriter, ReplayBuffer, PriorityBuffer


class DQNAgent(baseAgent):
    def __init__(self, params: dict):
        """
        Initializes the DQNAgent with provided parameters
        Sets up the model, optimizer, memory buffer, learning rate scheduler,
        and various configurations for training and exploration

        Args:
            params (dict): A dictionary containing the necessary parameters such as:
                - 'learning_rate': Learning rate for the optimizer
                - 'memory_size': Maximum size of the replay memory
                - 'state_collector': A function to collect the game state
                - 'model': The neural network model to use
                - 'reward_function': A function to compute the reward
                - 'train': Boolean indicating if the agent is in training mode
                - 'epsilon_decay_linear': Linear decay rate for epsilon
                - 'learning_rate_step_size': Step size for the learning rate scheduler
                - 'device': chosen device
                - 'is_tf_logger_active': activate/deactivate Tensorboard logger
        """
        super().__init__()
        self.params = params
        self.reward: float = 0.0
        self.gamma: float = params['gamma']
        self.loss_step_interval = 5
        run_name = params.get('run_name', 'default_run')
        custom_run_name = params.get('run_name', 'default_run_name')  # Use a custom run name if provided
        run_name = f"{custom_run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.writer = DummySummaryWriter() if not params['is_tf_logger_active'] else SummaryWriter(log_dir=f"runs/{run_name}")
        self.dataframe = pd.DataFrame()
        self.agent_target: int = 1
        self.agent_predict: int = 0
        self.learning_rate: float = params['learning_rate']
        self.epsilon: float = 1.0  # Exploration factor
        self.actual = []
        self.scores = []
        self.device = params['device']
        self.state_collector = params['state_collector']  # Function to collect game state
        self.memory = ReplayBuffer(self.state_collector.state_size, params['memory_size'])
        self.optimizer: optim.Optimizer = None
        self.game_counter: int = 0  # Counts number of games played
        self.losses = []
        self.model = params['model'](params, self.state_collector.state_size, self.device)  # Neural network model
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['learning_rate_step_size'], gamma=0.1)
        self.reward_calculator = params['reward_function']  # Reward function
        self.is_train: bool = params['train']  # Whether the agent is in training mode
        self.epsilon_decay_linear: float = params['epsilon_decay_linear']  # Linear decay rate for epsilon

    def get_state(self, game) -> np.ndarray:
        """
        Retrieves the current state of the game from the state collector

        Args:
            game: The game instance to extract the state from

        Returns:
            np.ndarray: The state of the game
        """
        return self.state_collector.get_state(game)

    def set_reward(self, game, collision_status: bool, steps: int) -> float:
        """
        Computes the reward based on the current game state, collision status, and number of steps taken.

        Args:
            game: The game instance
            collision_status (bool): Whether the agent collided in the game
            steps (int): The number of steps taken by the agent

        Returns:
            float: The computed reward for the current game state
        """
        return self.reward_calculator.set_reward(self.params, game, collision_status, steps)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Stores the experience tuple (state, action, reward, next_state, done) in memory for future replay

        Args:
            state (np.ndarray): The current state
            action (int): The action taken in the current state
            reward (float): The reward received after taking the action
            next_state (np.ndarray): The state after taking the action
            done (bool): Whether the episode has ended
        """
        self.memory.store(state, action, reward, next_state, done)

    def prepare_memory(self, minibatch):
        # Extract individual components from the experience batch

        states_tensor = torch.tensor(minibatch['states'], dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(minibatch['actions'], dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(minibatch['rewards'], dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(minibatch['next_states'], dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(minibatch['dones'], dtype=torch.bool).to(self.device)
        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)

    def loss(self, outputs, target_f):
        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss = F.huber_loss(outputs, target_f)
        self.losses.append(loss.item())
        if len(self.losses) >= self.loss_step_interval:
            avg_loss = mean(self.losses)
            std_loss = stdev(self.losses) if len(self.losses) > 1 else 0.0
            self.writer.add_scalar("Average Loss", avg_loss, self.game_counter)
            self.writer.add_scalar("Loss StdDev", std_loss, self.game_counter)
            self.writer.add_scalar("Learning rate", self.scheduler.get_last_lr()[0], self.game_counter)
            self.losses.clear()

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()  # Adjust the learning rate

    def replay(self, memory: collections.deque, batch_size: int) -> None:
        """
        Samples a batch of experiences from memory and performs a training step on the neural network

        Args:
            memory (collections.deque): The memory buffer containing past experiences
            batch_size (int): The number of experiences to sample from memory for training
        """

        # Select a random batch of experiences for training
        minibatch = self.memory.sample(batch_size)

        self.game_counter += 1

        states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = self.prepare_memory(minibatch)

        # Compute targets using the Bellman equation
        self.model.train()
        torch.set_grad_enabled(True)
        next_state_values = self.model(next_states_tensor).max(dim=1)[0]
        not_done_mask = ~dones_tensor
        targets = rewards_tensor.clone()
        targets[not_done_mask] += self.gamma * next_state_values[not_done_mask]

        # Predict the Q-values for the current states
        outputs = self.model(states_tensor)
        actions_indices = torch.argmax(actions_tensor, dim=1).unsqueeze(1)  # Get the indices of the actions
        target_f = outputs.clone().detach()
        target_f.scatter_(1, actions_indices, targets.unsqueeze(1))  # Set target values for the taken actions

        self.loss(outputs, target_f)

        # Log the learning rate to TensorBoard
        self.writer.add_scalar("Learning rate", self.scheduler.get_last_lr()[0], self.game_counter)

    def collect_statistic(self, game) -> None:
        """
        Collects and logs the current score of the game for statistical analysis

        Args:
            game: The game instance to collect the score from
        """
        self.scores.append(game.score)
        if len(self.scores) >= self.loss_step_interval:
            avg_loss = mean(self.scores)
            std_loss = stdev(self.scores) if len(self.scores) > 1 else 0.0
            self.writer.add_scalar('Score', avg_loss, self.game_counter)
            self.writer.add_scalar('Score std', std_loss, self.game_counter)
            self.scores.clear()

    def predict(self, state: np.ndarray) -> int:
        """
        Predicts the action to take based on the given state using the current model

        Args:
            state (np.ndarray): The current state to base the prediction on

        Returns:
            int: The predicted action
        """
        with torch.no_grad():
            state_tensor = self.model.preprocess(state)  # Preprocess the state for the model
            prediction = self.model(state_tensor)  # Get the Q-values from the model
            return np.argmax(prediction.detach().cpu().numpy()[0])  # Return the action with the highest Q-value

    def save_state(self, path: str) -> None:
        """
        Saves the current state (weights) of the model to the specified path

        Args:
            path (str): The file path to save the model weights
        """
        model_weights = self.model.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
        torch.save(model_weights, path)

    def calculate_epsilon(self, counter_games: int, minimum_epsilon: float = 0.0) -> None:
        """
        Calculates the exploration rate (epsilon) based on the number of games played,
        decaying it over time, and ensures it doesn't drop below a minimum value

        Args:
            counter_games (int): The current game count to base the epsilon decay on
            minimum_epsilon (float): The minimum value epsilon can reach
        """
        if not self.is_train:
            self.epsilon = 0 
        else:
            self.epsilon = 1 - (counter_games * self.epsilon_decay_linear)  # Decay epsilon linearly
        if self.epsilon < minimum_epsilon:
            self.epsilon = minimum_epsilon  # Ensure epsilon doesn't fall below the minimum

