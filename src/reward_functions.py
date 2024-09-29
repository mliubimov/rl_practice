import numpy as np

from src.utils import CollisionEvent
from src.envs.snake import Game as snakeGame
    
class SnakeReward:
    """
        Return the reward.
        The reward is:
            -10 when Snake crashes.
            +10 when Snake eats food.
            -10 when Snake loops in place.
            + reward for moving towards food.
    """
    @staticmethod
    def set_reward(params: dict, game: snakeGame, collision_status: bool, steps: int) -> float:
        reward = 0.
        if collision_status == CollisionEvent.DEAD:
            reward = -10.
        if collision_status == CollisionEvent.EAT:
            reward += 10.

        return reward