import numpy as np
import cv2
import pygame
import torch
from operator import add

from src.envs.snake import Game as snakeGame, Food as snakeFood, Player as snakePlayer
#TODO(mliubimov): implement matrix input
class MatrixStateCollector:
    state_size = 3  # Number of channels representing past states
    history = []
    
    @staticmethod
    def get_state(game: snakeGame) -> torch.Tensor:
        if len(MatrixStateCollector.history) == 0:
            MatrixStateCollector.history.append(torch.zeros(torch.Size([83, 73])))
            MatrixStateCollector.history.append(torch.zeros(torch.Size([83, 73])))
        # Capture the current state from the game display
        state = pygame.surfarray.array3d(game.gameDisplay)
        
        state = cv2.resize(state, (state.shape[0] // 6, state.shape[1] // 6))
        state_matrix = state[:, :]
        cv2.imshow("State", state_matrix)
        cv2.waitKey(1)
        state_matrix = cv2.cvtColor(state_matrix,cv2.COLOR_BGR2GRAY)
        # Normalize and permute dimensions for PyTorch compatibility
        state_matrix = torch.FloatTensor(state_matrix)
        state_matrix /= 255.0

        # Maintain a history of the last 3 states
        MatrixStateCollector.history.append(state_matrix)
        if len(MatrixStateCollector.history) > MatrixStateCollector.state_size:
            MatrixStateCollector.history.pop(0)

        # Stack the states along the channel dimension (axis=0)
        history_tensor = torch.stack(MatrixStateCollector.history, dim=0)

        return history_tensor[None, ...]  # Add a batch dimension

class BasicStateCollector:
    state_size = 14
    
    @staticmethod
    def get_state(game: snakeGame) -> np.ndarray:
        """
        Return the state.
        The state is a numpy array of 14 values, representing:
            - Danger 1 step ahead
            - Danger 2 steps ahead
            - Danger 1 step on the right
            - Danger 2 steps on the right
            - Danger 1 step on the left
            - Danger 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side      
        """
  
        player = game.player
        food = game.food

        def is_danger(position):
            return position in player.position or \
                position[0] < 20 or position[0] >= game.game_width - 20 or \
                position[1] < 20 or position[1] >= game.game_height - 20

        direction_vectors = {
            "left": [-20, 0],
            "right": [20, 0],
            "up": [0, -20],
            "down": [0, 20]
        }

        current_direction = "left" if player.x_change == -20 else \
                            "right" if player.x_change == 20 else \
                            "up" if player.y_change == -20 else \
                            "down"

        right_direction = "up" if current_direction == "left" else \
                        "down" if current_direction == "right" else \
                        "right" if current_direction == "up" else \
                        "left"

        left_direction = "down" if current_direction == "left" else \
                        "up" if current_direction == "right" else \
                        "left" if current_direction == "up" else \
                        "right"

        state = [
            # Danger 1 step ahead
            is_danger(list(map(add, player.position[-1], direction_vectors[current_direction]))),
            # Danger 2 steps ahead
            is_danger(list(map(add, player.position[-1], [2*x for x in direction_vectors[current_direction]]))),
            # Danger 1 step on the right
            is_danger(list(map(add, player.position[-1], direction_vectors[right_direction]))),
            # Danger 2 steps on the right
            is_danger(list(map(add, player.position[-1], [2*x for x in direction_vectors[right_direction]]))),
            # Danger 1 step on the left
            is_danger(list(map(add, player.position[-1], direction_vectors[left_direction]))),
            # Danger 2 steps on the left
            is_danger(list(map(add, player.position[-1], [2*x for x in direction_vectors[left_direction]]))),
            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.position[-1][0],  # food left
            food.x_food > player.position[-1][0],  # food right
            food.y_food < player.position[-1][1],  # food up
            food.y_food > player.position[-1][1],  # food down
            len(player.position)
        ]
        
        state = [1 if s else 0 for s in state[:-1]]

        return state
    
class StateCollectorWithDistances:
    state_size = 16
    
    @staticmethod
    def get_state(game: snakeGame) -> np.ndarray:
        """
        Return the state.
        The state is a numpy array of 18 values, representing:
            - Danger 1 step ahead
            - Danger 2 steps ahead
            - Danger 1 step on the right
            - Danger 2 steps on the right
            - Danger 1 step on the left
            - Danger 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side
            - Distance to the food horizontally
            - Distance to the food vertically
            - Distance to the nearest wall in the current direction
            - Distance to the nearest wall in the direction to the right
        """
        player = game.player
        food = game.food

        def is_danger(position):
            return (
                position in player.position or 
                position[0] < 0 or position[0] >= game.game_width or 
                position[1] < 0 or position[1] >= game.game_height
            )

        direction_vectors = {
            "left": (-20, 0),
            "right": (20, 0),
            "up": (0, -20),
            "down": (0, 20)
        }

        current_direction = "left" if player.x_change == -20 else \
                            "right" if player.x_change == 20 else \
                            "up" if player.y_change == -20 else \
                            "down"

        right_direction = "up" if current_direction == "left" else \
                          "down" if current_direction == "right" else \
                          "right" if current_direction == "up" else \
                          "left"

        left_direction = "down" if current_direction == "left" else \
                         "up" if current_direction == "right" else \
                         "left"

        head_position = player.position[-1]

        def distance_to_wall(position, direction):
            if direction == "left":
                return position[0] // 20
            elif direction == "right":
                return (game.game_width - position[0] - 20) // 20
            elif direction == "up":
                return position[1] // 20
            elif direction == "down":
                return (game.game_height - position[1] - 20) // 20

        state = [
            # Danger 1 step ahead
            is_danger(list(map(add, player.position[-1], direction_vectors[current_direction]))),
            # Danger 2 steps ahead
            is_danger(list(map(add, player.position[-1], [2*x for x in direction_vectors[current_direction]]))),
            # Danger 1 step on the right
            is_danger(list(map(add, player.position[-1], direction_vectors[right_direction]))),
            # Danger 2 steps on the right
            is_danger(list(map(add, player.position[-1], [2*x for x in direction_vectors[right_direction]]))),
            # Danger 1 step on the left
            is_danger(list(map(add, player.position[-1], direction_vectors[left_direction]))),
            # Danger 2 steps on the left
            is_danger(list(map(add, player.position[-1], [2*x for x in direction_vectors[left_direction]]))),
            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y,  # food down
            float(abs(food.x_food - player.x)) / (game.game_width//20),  # Distance to the food horizontally
            float(abs(food.y_food - player.y)) / (game.game_height//20), # Distance to the food vertically
        ]
        state = [1 if s else 0 for s in state[:-2]] + state[-2:]

        return np.asarray(state, dtype=np.float32)
    
class StateCollectorWithSnakeInfo:
    state_size = 18
    
    @staticmethod
    def get_state(game: snakeGame) -> np.ndarray:
        """
        Return the state.
        The state is a numpy array of 18 values, representing:
            - Danger 1 step ahead
            - Danger 2 steps ahead
            - Danger 3 steps ahead
            - Danger 1 step on the right
            - Danger 2 steps on the right
            - Danger 3 steps on the right
            - Danger 1 step on the left
            - Danger 2 steps on the left
            - Danger 3 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side
            - Distance to the nearest body segment in current direction
        """
        player = game.player
        food = game.food
        def is_danger(position):
            return position in player.position or \
                position[0] < 20 or position[0] >= game.game_width - 20 or \
                position[1] < 20 or position[1] >= game.game_height - 20

        direction_vectors = {
            "left": [-20, 0],
            "right": [20, 0],
            "up": [0, -20],
            "down": [0, 20]
        }

        current_direction = "left" if player.x_change == -20 else \
                            "right" if player.x_change == 20 else \
                            "up" if player.y_change == -20 else \
                            "down"

        right_direction = "up" if current_direction == "left" else \
                        "down" if current_direction == "right" else \
                        "right" if current_direction == "up" else \
                        "left"

        left_direction = "down" if current_direction == "left" else \
                        "up" if current_direction == "right" else \
                        "left"

        def distance_to_body_in_direction(direction):
            dist = 1
            while True:
                pos = list(map(add, player.position[-1], [dist * x for x in direction_vectors[direction]]))
                if pos in player.position or pos[0] < 20 or pos[0] >= game.game_width - 20 or pos[1] < 20 or pos[1] >= game.game_height - 20:
                    return dist
                dist += 1

        state = [
            # Danger 1 step ahead
            is_danger(list(map(add, player.position[-1], direction_vectors[current_direction]))),
            # Danger 2 steps ahead
            is_danger(list(map(add, player.position[-1], [2 * x for x in direction_vectors[current_direction]]))),
            # Danger 3 steps ahead
            is_danger(list(map(add, player.position[-1], [3 * x for x in direction_vectors[current_direction]]))),
            # Danger 1 step on the right
            is_danger(list(map(add, player.position[-1], direction_vectors[right_direction]))),
            # Danger 2 steps on the right
            is_danger(list(map(add, player.position[-1], [2 * x for x in direction_vectors[right_direction]]))),
            # Danger 3 steps on the right
            is_danger(list(map(add, player.position[-1], [3 * x for x in direction_vectors[right_direction]]))),
            # Danger 1 step on the left
            is_danger(list(map(add, player.position[-1], direction_vectors[left_direction]))),
            # Danger 2 steps on the left
            is_danger(list(map(add, player.position[-1], [2 * x for x in direction_vectors[left_direction]]))),
            # Danger 3 steps on the left
            is_danger(list(map(add, player.position[-1], [3 * x for x in direction_vectors[left_direction]]))),
            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y,  # food down
            distance_to_body_in_direction(current_direction)  # distance to nearest body segment in current direction
        ]

        state = [1 if s else 0 for s in state[:-1]] + state[-1:] 

        return np.asarray(state)