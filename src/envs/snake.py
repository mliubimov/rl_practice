import pygame
import numpy as np
from random import randint
from src.reward_functions import CollisionEvent


class Game:
    """Class representing the Snake game environment."""

    def __init__(self, game_width: int, game_height: int) -> None:
        """Initialize the game with width, height, and other attributes"""
        pygame.display.set_caption('SnakeGen')
        self.game_width: int = game_width
        self.game_height: int = game_height
        self.game_history: int = 5
        self.field: list[np.ndarray] = [self.reset_field() for _ in range(self.game_history)]
        self.gameDisplay: pygame.Surface = pygame.display.set_mode((game_width, game_height + 80))
        self.bg: pygame.Surface = pygame.image.load("img/background.png")
        self.crash: bool = False
        self.player: Player = Player(game_width, game_height)
        self.food: Food = Food(game_width, game_height)
        self.score: int = 0
        self.reset()

    def display_ui(self, score: int, record: int) -> None:

        """Display the score and highest score on the game interface"""
        myfont: pygame.font.Font = pygame.font.SysFont('ramabhadra UI', 27)
        myfont_bold: pygame.font.Font = pygame.font.SysFont('ramabhadra UI', 27, True)
        text_score: pygame.Surface = myfont.render('Score: ', True, (0, 0, 0))
        text_score_number: pygame.Surface = myfont.render(str(score), True, (0, 0, 0))
        text_highest: pygame.Surface = myfont.render('Highest score: ', True, (0, 0, 0))
        text_highest_number: pygame.Surface = myfont_bold.render(str(record), True, (0, 0, 0))
        self.gameDisplay.blit(self.bg, (0, 0))
        self.gameDisplay.blit(text_score, (170, 430))
        self.gameDisplay.blit(text_score_number, (230, 430))
        self.gameDisplay.blit(text_highest, (145, 460))
        self.gameDisplay.blit(text_highest_number, (275, 460))


    def display(self, record: int) -> None:
        """Render the game elements (player and food) on the screen"""
        #self.gameDisplay.fill((255, 255, 255))
        
        self.player.display(self.gameDisplay)
        self.food.display_food(self.gameDisplay)
        self.display_ui(self.score, record)

    def reset_field(self) -> np.ndarray:
        """Create and return a new empty field with boundaries"""
        field: np.ndarray = np.zeros((self.game_width // 20, self.game_height // 20), dtype=float)
        field[0, :] = 1
        field[-1, :] = 1
        field[:, 0] = 1
        field[:, -1] = 1
        return field

    def reset(self) -> None:
        """Reset the game state, including the player, food, and field"""
        self.crash = False
        self.player.reset()
        self.food.reset()
        self.field[0][int(self.player.x//20),int(self.player.y//20)] = 2
        self.field[0][int(self.food.x_food // 20), int(self.food.y_food // 20)] = 4
        self.score = 0

    def update_field(self) -> None:
        field = self.reset_field()
        field[int(self.food.x_food//20),int(self.food.y_food//20)] = 4
        if self.player.food > 1:
            for i in range(0, self.player.food - 1):
                field[int(self.player.position[i][0]//20),int(self.player.position[i][1]//20)] = 3
        field[int(self.player.x//20),int(self.player.y//20)] = 2
        self.field.insert(0,field)
        if len(self.field)>5:
            del self.field[-1]
    
    def make_action(self, move: list[int], model_move: list[int] | None) -> CollisionEvent:
        """Make the player perform an action and return the outcome"""
        collision_event = self.player.do_move(move, self.food.get_coord(), model_move)
        if collision_event == CollisionEvent.EAT:
            self.score += 1
            self.food.update_position(self.player.position)
        if collision_event == CollisionEvent.DEAD:
            self.crash = True
        self.update_field()
        return collision_event


class Player:
    """Class representing the player (snake) in the game"""

    def __init__(self, game_width: int, game_height: int) -> None:
        """Initialize the player with its starting position and image"""
        self.image_head: pygame.Surface = pygame.image.load('img/snakeHead.png')
        self.image_body: pygame.Surface = pygame.image.load('img/snakeBody.png')
        self.image_rot_body: pygame.Surface = pygame.image.load('img/snakeRotatedBody.png')
        self.image_tail: pygame.Surface = pygame.image.load('img/snakeTail.png')
        self.game_width = game_width
        self.game_height = game_height
        self.reset()

    def reset(self) -> None:
        """Reset the player's position, direction, and state"""
        self.x_change: int = 20
        self.y_change: int = 0
        self.food: int = 1
        
        x: float = 0.45 * self.game_width
        y: float = 0.5 * self.game_height

        self.x: int = int(x - x % 20)
        self.y: int = int(y - y % 20)
        self.position: list[list[int]] = [[self.x, self.y]]
        self.prev_positions: list[list[int]] = []

    def update_position(self) -> None:
        """Update the player's position"""
        if self.food > 1:
            for i in range(0,len( self.position)-1):
                self.position[i][0], self.position[i][1] = self.position[i + 1]
                #self.field[int(self.position[i][0]//20), int(self.position[i][1]//20)] = 3
        
        self.position[-1] = [self.x, self.y]
        self.prev_positions.append([self.x, self.y])
        #self.field[int(self.x//20), int(self.y//20)] = 2

        if len(self.prev_positions) > 10:
            self.prev_positions.pop(0)

    def eat(self, food_coord: tuple[int, int]) -> bool:
        """Check if the player has eaten the food"""
        if self.x == food_coord[0] and self.y == food_coord[1]:
            for i in range(len(self.position)):
                if self.x == self.position[i][0] and self.y == self.position[i][1]:
                    assert 0
            self.position.append([self.x, self.y])
            self.food += 1
            return True
        return False

    def do_move(self, move: list[int], food_coord: tuple[int, int], model_move: list[int] | None) -> CollisionEvent:
        """Perform a movement based on the input and update the game state"""
        move_array: list[int] = [self.x_change, self.y_change]

        if np.array_equal(move, [1, 0, 0]):
            pass
        elif np.array_equal(move, [0, 1, 0]):
            move_array = [0, self.x_change] if self.y_change == 0 else [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]):
            move_array = [0, -self.x_change] if self.y_change == 0 else [self.y_change, 0]
        
        x_new, y_new = self.x + move_array[0], self.y + move_array[1]

        if x_new < 20 or x_new > self.game_width - 40 or y_new < 20 or y_new > self.game_height - 40 or [x_new, y_new] in self.position:
            return CollisionEvent.DEAD if model_move is None else self.do_move(model_move, food_coord, None)

        self.x_change, self.y_change = move_array
        self.x, self.y = x_new, y_new

        if self.eat(food_coord):
            self.update_position()
            return CollisionEvent.EAT

        self.update_position()
        return CollisionEvent.NOTHING

    def display(self, gameDisplay: pygame.Surface) -> None:
        """Display the player's body and tail on the game screen"""
        self.displayed_poses = []

        # Loop through the snake's body from tail to head
        for i in reversed(range(len(self.position))):  
            x_temp, y_temp = self.position[i]

            # Skip if the current position has already been displayed (prevents overlap)
            if [x_temp, y_temp] in self.displayed_poses:
                continue
            
            self.displayed_poses.append([x_temp, y_temp])
            
            if i == len(self.position) - 1:  # Head
                # Rotate the head image based on movement direction
                if self.x_change > 0:
                    rotated_image = pygame.transform.rotate(self.image_head, 270)  # Moving right
                elif self.x_change < 0:
                    rotated_image = pygame.transform.rotate(self.image_head, 90)   # Moving left
                elif self.y_change > 0:
                    rotated_image = pygame.transform.rotate(self.image_head, 180)  # Moving down
                else:
                    rotated_image = pygame.transform.rotate(self.image_head, 0)    # Moving up
                gameDisplay.blit(rotated_image, (x_temp, y_temp))

            elif i == 0:  # Tail
                x_next, y_next = self.position[i + 1]
                # Rotate the tail image based on the direction to the next segment
                if x_temp < x_next:
                    rotated_tail = pygame.transform.rotate(self.image_tail, 270)  # Moving right
                elif x_temp > x_next:
                    rotated_tail = pygame.transform.rotate(self.image_tail, 90)   # Moving left
                elif y_temp < y_next:
                    rotated_tail = pygame.transform.rotate(self.image_tail, 180)  # Moving down
                else:
                    rotated_tail = pygame.transform.rotate(self.image_tail, 0)    # Moving up
                gameDisplay.blit(rotated_tail, (x_temp, y_temp))

            else:  # Body
                x_prev, y_prev = self.position[i + 1]  # Next segment (before current)
                x_next, y_next = self.position[i - 1]  # Previous segment (after current)

                # Check for overlapping positions to avoid rendering errors
                if x_prev == x_temp and y_prev == y_temp:
                    x_prev, y_prev = self.position[i + 2]  # Use further segment
                if x_next == x_temp and y_next == y_temp:
                    x_next, y_next = self.position[i - 2]  # Use further segment

                # Determine the orientation of the body
                if x_prev == x_temp and x_next == x_temp:  # Vertical line
                    rotated_body = pygame.transform.rotate(self.image_body, 0)
                elif y_prev == y_temp and y_next == y_temp:  # Horizontal line
                    rotated_body = pygame.transform.rotate(self.image_body, 90)
                else:
                    # Corner turns
                    if (x_prev < x_temp and y_next < y_temp) or (x_next < x_temp and y_prev < y_temp):
                        rotated_body = pygame.transform.rotate(self.image_rot_body, 270)  # Left-up
                    elif (x_prev > x_temp and y_next < y_temp) or (x_next > x_temp and y_prev < y_temp):
                        rotated_body = pygame.transform.rotate(self.image_rot_body, 180)  # Right-up
                    elif (x_prev < x_temp and y_next > y_temp) or (x_next < x_temp and y_prev > y_temp):
                        rotated_body = pygame.transform.rotate(self.image_rot_body, 0)    # Left-down
                    elif (x_prev > x_temp and y_next > y_temp) or (x_next > x_temp and y_prev > y_temp):
                        rotated_body = pygame.transform.rotate(self.image_rot_body, 90)   # Right-down

                gameDisplay.blit(rotated_body, (x_temp, y_temp))

        pygame.display.update()


class Food:
    """Class representing the food in the game"""

    def __init__(self, game_width: int, game_height: int) -> None:
        """Initialize the food with an image and position"""
        self.image: pygame.Surface = pygame.image.load('img/food.png')
        self.game_width = game_width
        self.game_height = game_height
        self.reset()

    def reset(self) -> None:
        """Reset the food's position to a default value"""
        self.x_food: int = 240
        self.y_food: int = 200

    def get_coord(self) -> tuple[int, int]:
        """Return the current coordinates of the food"""
        return self.x_food, self.y_food

    def update_position(self, player_positions: list[list[int]]) -> None:
        """Randomly generate new coordinates for the food, ensuring it does not overlap with the player."""
        while True:
            x_rand: int = randint(20, self.game_width - 40)
            y_rand: int = randint(20, self.game_height - 40)
            x_food, y_food = x_rand - x_rand % 20, y_rand - y_rand % 20
            if [x_food, y_food] not in player_positions:
                self.x_food, self.y_food = x_food, y_food
                break

    def display_food(self, gameDisplay: pygame.Surface) -> None:
        """Display the food at the current coordinates"""
        gameDisplay.blit(self.image, (self.x_food, self.y_food))
        pygame.display.update()
