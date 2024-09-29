import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.noisy_linear_model import NoisyLinear

class DuelingNoisyLinearModel(BaseModel):
    def __init__(self, params: dict, input_state_size: int, device: str):
        super().__init__()
        #TODO: init models method
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.state_size = input_state_size
        self.device = device
        self.out_dim = 3
        self.sigma_init = params['noisy_linear_sigma']
        assert type(input_state_size) == int, "Noisy Linear model supports only one dimension"

        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_state_size, self.first_layer),
            nn.ReLU()
        ).to(self.device)

        # Advantage and Value streams
        self.advantage_layer = nn.Sequential(
            NoisyLinear(self.first_layer, self.second_layer, self.sigma_init),
            nn.ReLU(),
            NoisyLinear(self.second_layer, self.out_dim, self.sigma_init)  # Number of actions
        ).to(self.device)

        self.value_layer = nn.Sequential(
            NoisyLinear(self.first_layer, self.second_layer, self.sigma_init),
            nn.ReLU(),
            NoisyLinear(self.second_layer, 1, self.sigma_init)  # Single state value
        ).to(self.device)

        if self.load_weights:
            self.load_state_dict(torch.load(self.weights))
            print("weights loaded")
        else:
            self.initialize_weights()

    def preprocess(self, state: torch.Tensor) -> torch.Tensor:
        if type(state) == list:
            state = np.asarray(state)
        return torch.tensor(state.reshape((1, 1, self.state_size)), dtype=torch.float32).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.state_size)  # Flatten the input for the feature layer

        # Pass through shared feature layer
        features = self.feature_layer(x)

        # Compute value and advantage streams
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)

        # Combine value and advantage into Q-values
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return F.softmax(q_values, dim=-1)

    def reset_noise(self):
        """Reset the noise in all noisy layers."""
        for layer in [self.advantage_layer, self.value_layer]:
            for module in layer:
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def get_sequences(self, memory: list, seq_length: int) -> list:
        seq_memory = []
        for i in range(len(memory) - seq_length + 1):
            seq = [memory[j] for j in range(i, i + seq_length)]
            seq_memory.append(seq)
        return seq_memory
