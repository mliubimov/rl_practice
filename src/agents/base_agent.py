from abc import ABC, abstractmethod
from typing import List

class baseAgent(ABC):
    """
    Base abstract class for agents in a game environment
    Defines the essential methods that any derived agent class must implement
    """
    
    def __init__(self) -> None:
        """
        Initialize the base agent. Can be expanded by derived classes
        """
        pass
    
    @abstractmethod
    def get_state(self, game, player, food) -> List:
        """
        Abstract method to retrieve the current state of the environment

        Args:
            game: The game environment instance
            player: The player instance in the game
            food: The food object in the game
        
        Returns:
            A list representing the current state
        """
        pass
     
    @abstractmethod
    def set_reward(self, player, crash: bool, food) -> float:
        """
        Abstract method to calculate and set the reward based on the player's status
        
        Args:
            player: The player instance in the game
            crash (bool): A flag indicating if the player has crashed
            food: The food object in the game
        
        Returns:
            A float representing the reward value
        """
        pass
    
    @abstractmethod
    def remember(self, state: List, action: List, reward: float, next_state: List, done: bool) -> None:
        """
        Abstract method to store experience in memory for later replay
        
        Args:
            state (List): The current state of the game
            action (List): The action taken in the current state
            reward (float): The reward received for the action
            next_state (List): The next state after the action is taken
            done (bool): A flag indicating if the game is over
        """
        pass
    
    @abstractmethod
    def replay(self) -> None:
        """
        Abstract method to train the agent by replaying stored experiences
        """
        pass
    
    @abstractmethod
    def collect_statistic(self, game) -> None:
        """
        Abstract method to collect statistics about the game performance
        
        Args:
            game: The game environment instance
        """
        pass
    
    @abstractmethod
    def predict(self, state: List):
        """
        Abstract method to predict the next action based on the current state
        
        Args:
            state (List): The current state of the game
        
        Returns:
            The predicted action for the given state
        """
        pass
    
    @abstractmethod    
    def save_state(self, path: str) -> None:
        """
        Abstract method to save the agent's current state to a file
        
        Args:
            path (str): The path where the state should be saved
        """
        pass