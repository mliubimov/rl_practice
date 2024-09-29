import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from src.models.base_model import BaseModel

class LinearModel(BaseModel):
    def __init__(self, params: dict, input_state_size: int, device: str):
        super().__init__()
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.state_size = input_state_size
        self.device = device
        assert type(input_state_size) == int, "Linear model support only one dimension"
        self.f1 = nn.Linear(input_state_size, self.first_layer).to(self.device)
        self.f3 = nn.Linear(self.first_layer, self.second_layer).to(self.device)
        self.f4 = nn.Linear(self.second_layer, 3).to(self.device)
        # Weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")
        else:
           self.initialize_weights()
    
    def preprocess(self, state: torch.Tensor) -> torch.Tensor:
        if type(state) == list:
            state = np.asarray(state)
        return torch.tensor(state.reshape((1,1, self.state_size)), dtype=torch.float32).to(self.device)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.state_size)  # Flatten the input for the first linear layer
        x = F.relu(self.f1(x))
        x = F.relu(self.f3(x))  # Take the output of the last LSTM cell
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_sequences(self, memory: list, seq_length: int) -> list:
        seq_memory = []
        for i in range(len(memory) - seq_length + 1):
            seq = [memory[j] for j in range(i, i + seq_length)]
            seq_memory.append(seq)
        return seq_memory
    