import os
import random
import pygame
import copy
import datetime
from PIL import Image
import io
from tqdm import tqdm

import argparse
import numpy as np
from random import randint, seed

import yaml
from typing import Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import torch


from src.models.linear_model import LinearModel
from src.models.noisy_linear_model import NoisyLinearModel
from src.models.dueling_noisy_linear_model import DuelingNoisyLinearModel
from src.models.categorical_noisy_linear_model import CategoricalNoisyLinearModel

from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.agents.rainbow_dqn import RainbowDQNAgent
from src.agents.dueling_dqn import DuelingDQNAgent



from src.reward_functions import SnakeReward


from src.state_collectors.state_collectors_snake import BasicStateCollector as snakeBasicStateCollector
from src.state_collectors.state_collectors_snake import MatrixStateCollector as snakeMatrixStateCollector
from src.state_collectors.state_collectors_snake import StateCollectorWithDistances as snakeStateCollectorWithDistances
from src.state_collectors.state_collectors_snake import StateCollectorWithSnakeInfo as snakeStateCollectorWithSnakeInfo
from src.envs.snake import Game as snakeGame
from src.utils import CollisionEvent
from src.optunaOpt import *


models = {"LinearModel": LinearModel,
          "NoisyLinearModel": NoisyLinearModel,
          "DuelingNoisyLinearModel": DuelingNoisyLinearModel,
          "CategoricalNoisyLinearModel":CategoricalNoisyLinearModel}
agents = {"DQNAgent": DQNAgent,
          "DoubleDQNAgent": DoubleDQNAgent,
          "DuelingDQNAgent": DuelingDQNAgent,
          "RainbowDQNAgent": RainbowDQNAgent}
games = {"snakeGame": snakeGame}
state_collectors = {"snakeBasicStateCollector":snakeBasicStateCollector,
                    "snakeMatrixStateCollector": snakeMatrixStateCollector,
                    "snakeStateCollectorWithDistances": snakeStateCollectorWithDistances,
                    "snakeStateCollectorWithSnakeInfo": snakeStateCollectorWithSnakeInfo}
reward_functions = {
    "SnakeReward":SnakeReward
}

def plot_seaborn(agent, array_counter, array_score, train, show_result = False):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        x =np.array([array_counter])[0],
        y =np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    if show_result:
        plt.show()
    image = Image.open(buf)
    image = np.array(image)
    agent.writer.add_image("Score vs Games", image, dataformats='HWC')
    buf.close()
    plt.close()


def run(params: dict, display: bool, plot_result: bool, is_training: bool) -> Tuple[float, float, float]:

    pygame.init()
    pygame.font.init()
    record = 0
    best_score = 0
    total_score = 0
    total_deaths = 0 
    scores = []
    counter_plot = []
    game = params['game'](params['window_width'], params['window_height'])
    agent = params['agent'](params)
    clock = pygame.time.Clock()
    counter_games = 1
    if not display:
        pygame.display.set_mode((1, 1), flags=pygame.NOFRAME)
        pygame.display.set_caption('No Display Mode')
    # Wrap the loop in a tqdm progress bar
    with tqdm(total=params['episodes'], desc="Training Progress") as pbar:
        while counter_games <= params['episodes']:
            pbar.set_postfix({"Game": counter_games, "Best Score": best_score, "Deaths": total_deaths})

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            game.reset()
            game.score = 0
            steps = 0
            is_running = True
            state_new = None

            while steps < params['num_iteration']:
                agent.calculate_epsilon(counter_games, minimum_epsilon=0.00)

                if state_new is not None:
                    state_old = copy.deepcopy(state_new)
                else:
                    state_old = agent.state_collector.get_state(game)

                # Exploration mechanism
                if random.uniform(0, 1) < agent.epsilon:
                    move = np.eye(3)[randint(0, 2)]
                    model_move = np.eye(3)[agent.predict(state_old)]
                else:
                    move = agent.predict(state_old)
                    move = np.eye(3)[move]
                    model_move = None

                collision_status = game.make_action(move, model_move)

                if collision_status == CollisionEvent.DEAD:
                    total_deaths += 1
                    is_running = False
                elif collision_status == CollisionEvent.WON:
                    game.reset()
                    steps = 0
                    game.score += 1
                elif collision_status == CollisionEvent.EAT:
                    steps = 0

                state_new = agent.state_collector.get_state(game)
                reward = agent.set_reward(game, collision_status, steps)

                if is_training:
                    agent.remember(state_old, move, reward, state_new, game.crash)

                steps += 1
                if not is_running:
                    break

                record = max(game.score, record)
                if display:
                    game.display(record)
                    clock.tick(params['speed'])

            counter_games += 1
            if is_training:
                agent.replay(agent.memory, params['batch_size'])
            agent.collect_statistic(game)
            total_score += game.score
            if game.score > best_score:
                best_score = game.score
                agent.save_state(params["weights_path"])
            scores.append(game.score)
            counter_plot.append(counter_games)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({"Game": counter_games, "Best Score": best_score, "Deaths": total_deaths})



    if is_training:
        agent.save_state(params["weights_path"])
    plot_seaborn(agent, counter_plot, scores, is_training, show_result = plot_result)

    return scores



def parse_args():
    parser = argparse.ArgumentParser(description="RL Training/Test with Configurations")
    parser.add_argument('--param_file', type=str, default="./configs/dqn_snake.yaml", 
                        help="Path to the YAML config file.")

    parser.add_argument('--display', action='store_true', 
                        help="Display the game window.")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Seed which provides reproducibility")
    parser.add_argument('--mode', type=str, choices=['train', 'parameters_search', 'test' ], default="train", 
                        help="Mode to run the script in: 'train' or 'test'.")
    parser.add_argument('--plot_result', action='store_true',
                        help="plot final result with seaborn")
    
    return parser.parse_args()

if __name__ == "__main__":
   
    
    # Parse arguments
    args = parse_args()
    torch.manual_seed(args.seed)
    seed(args.seed)


    # Load YAML configuration file
    with open(args.param_file) as stream:
        try:
            params = yaml.safe_load(stream)
            params['model'] = models[params['model']]
            params['agent'] = agents[params['agent']]
            params['game'] = games[params['game']]
            params['state_collector'] = state_collectors[params['state_collector']]
            params['reward_function'] = reward_functions[params['reward_function']]            
            
        except yaml.YAMLError as exc:
            print(exc)
            pygame.quit()
            exit()
    
    # Optional Bayesian optimization
    if args.mode == "parameters_search":
        bayesOpt = OptunaOptimizer(params)
        bayesOpt.optimize_RL()
    # Run the game in either train or test mode
    elif args.mode == "train":
        print("Training...")
        run(params, args.display, args.plot_result, is_training=True )
    elif args.mode == "test":
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        run(params, args.display, args.plot_result, is_training=False)
    else:
        raise ValueError("Unknown launching mode!")

    pygame.quit()