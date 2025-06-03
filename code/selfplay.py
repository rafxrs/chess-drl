import config
import chess
import logging
import numpy as np
import pandas as pd
from multiprocessing import Pool
from random import choice, choices
from agent import Agent
from env import Chess_Env
from game import Game
from gui.display import GUI

logging.basicConfig(level=logging.INFO, format=' %(message)s')

def setup(starting_position: str = chess.STARTING_FEN) -> Game:
    """
    Setup the game with the given starting position.
    """
    env = Chess_Env(starting_position=starting_position)
    white_agent = Agent(model_path=config.WHITE_MODEL_PATH)
    black_agent = Agent(model_path=config.BLACK_MODEL_PATH)
    game = Game(env, white_agent, black_agent)
    return game

def self_play(game: Game, player_is_white: bool = True, n_games: int = 1) -> None:
    """
    Play a game against itself.
    """
    game.reset()
    GUI(game, player_is_white=player_is_white).start()
    
    for _ in range(n_games):
        while not game.is_over():
            if game.current_player_is_white() == player_is_white:
                move = game.white_agent.get_move(game.env)
            else:
                move = game.black_agent.get_move(game.env)
            game.push(move)
            GUI(game, player_is_white=player_is_white).draw()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plays games against itself.")
    parser.add_argument("--n_games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--starting_position", type=str, default=chess.STARTING_FEN, help="Starting position in FEN format.")
    
    args = parser.parse_args()
    
    game = setup(starting_position=args.starting_position)
    self_play(game, n_games=args.n_games)