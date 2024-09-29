from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        Abstract method to define the forward pass of the network.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def preprocess(self, state):
        """
        Abstract method for preprocessing the input state.
        Must be implemented by subclasses.
        """
        pass
    @abstractmethod
    def get_sequences(self, memory, seq_length):
        """
        Return sequences from memory based on sequence length.
        """
        pass
    

    
    def initialize_weights(self):
        """
        Initialize model weights. Can be overridden by subclasses.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    def load_model_weights(self):
        """
        Load pre-trained weights if available.
        """
        if self.load_weights and self.weights:
            self.load_state_dict(torch.load(self.weights))
            print("Weights loaded from", self.weights)
    
