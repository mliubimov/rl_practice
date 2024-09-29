import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from src.models.base_model import BaseModel

class NoisyLinear(nn.Module):
    """Noisy linear layer for Rainbow DQN."""
    def __init__(self, in_features, out_features, sigma_init=0.7):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1. / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init/ math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
        # TODO(): choose
        #     return F.linear(
        #     x,
        #     self.weight_mu + self.weight_sigma * self.weight_epsilon,
        #     self.bias_mu + self.bias_sigma * self.bias_epsilon,
        # )


class NoisyLinearModel(BaseModel):
    def __init__(self, params: dict, input_state_size: int, device: str):
        super().__init__()
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.state_size = input_state_size
        self.device = device
        assert type(input_state_size) == int, "Noisy Linear model supports only one dimension"

        # Noisy layers
        self.f1 = nn.Linear(input_state_size, self.first_layer).to(self.device)
        self.f2 = NoisyLinear(self.first_layer, self.second_layer).to(self.device)
        self.f3 = NoisyLinear(self.second_layer, 3).to(self.device)

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
        x = x.view(-1, self.state_size)  # Flatten the input for the first noisy linear layer
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))  # Next noisy linear layer
        x = F.softmax(self.f3(x), dim=-1)
        return x

    def reset_noise(self):
        """Reset the noise in all noisy layers."""
        self.f2.reset_noise()
        self.f3.reset_noise()

    def get_sequences(self, memory: list, seq_length: int) -> list:
        seq_memory = []
        for i in range(len(memory) - seq_length + 1):
            seq = [memory[j] for j in range(i, i + seq_length)]
            seq_memory.append(seq)
        return seq_memory