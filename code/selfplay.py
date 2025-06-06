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

# Define default window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

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
    import pygame
    import time
    
    game.reset()
    
    # Create GUI with proper parameters
    gui = GUI(
        width=WINDOW_WIDTH, 
        height=WINDOW_HEIGHT, 
        player=False,  # False since this is self-play (no human player)
        fen=game.env.board.fen()
    )
    
    for game_num in range(n_games):
        logging.info(f"Starting game {game_num+1}/{n_games}")
        
        while not game.is_over():
            # Process pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get move from the appropriate agent
            if game.env.board.turn == chess.WHITE:
                logging.info("White to move...")
                move = game.white.get_move(game.env)
            else:
                logging.info("Black to move...")
                move = game.black.get_move(game.env)
            
            logging.info(f"Move: {move}")
            
            # Make the move
            game.push(move)
            
            # Update the display - need to update the board in the GUI
            gui.gameboard.board = game.env.board.copy()
            gui.draw()
            pygame.display.update()
            
            # Add a slight delay so the game is visible
            time.sleep(0.5)  # Wait 0.5 seconds between moves
            
        logging.info(f"Game {game_num+1} finished with result: {game.result()}")
        
        # If there are more games to play, reset the game
        if game_num < n_games - 1:
            game.reset()
            time.sleep(2)  # Pause between games
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plays games against itself.")
    parser.add_argument("--n_games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--starting_position", type=str, default=chess.STARTING_FEN, help="Starting position in FEN format.")
    
    args = parser.parse_args()
    
    game = setup(starting_position=args.starting_position)
    self_play(game, n_games=args.n_games)