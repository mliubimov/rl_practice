import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.dueling_noisy_linear_model import DuelingNoisyLinearModel
from src.models.noisy_linear_model import NoisyLinear

class CategoricalNoisyLinearModel(DuelingNoisyLinearModel):
    def __init__(self, params: dict, input_state_size: int, device: str):
        super().__init__(params,input_state_size, device)
      
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.atom_size = params['atom_size']
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)
        self.out_dim = 3
        assert type(input_state_size) == int, "Noisy Linear model supports only one dimension"

        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_state_size, self.first_layer),
            nn.ReLU()
        ).to(self.device)

        # Advantage and Value streams
        self.advantage_layer = nn.Sequential(
            NoisyLinear(self.first_layer, self.second_layer),
            nn.ReLU(),
            NoisyLinear(self.second_layer, self.out_dim * self.atom_size)  # Number of actions
        ).to(self.device)

        self.value_layer = nn.Sequential(
            NoisyLinear(self.first_layer, self.second_layer),
            nn.ReLU(),
            NoisyLinear(self.second_layer, self.atom_size)  # Single state value
        ).to(self.device)

        if self.load_weights:
            self.load_state_dict(torch.load(self.weights))
            print("weights loaded")
        else:
            self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
       

    def dist(self, x: torch.Tensor) -> torch.Tensor:
            """Get distribution for atoms."""
            feature = self.feature_layer(x)

            advantage = self.advantage_layer(feature)

            advantage = advantage.view(
                -1, self.out_dim, self.atom_size
            )
            value = self.value_layer(feature).view(-1, 1, self.atom_size)

            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
            
            dist = F.softmax(q_atoms, dim=-1)
            dist = dist.clamp(min=1e-3)  
            return dist