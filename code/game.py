# main game logic
import os
import time
import utils
import logging
import config
import uuid
import pandas as pd
import numpy as np
import chess

from env import Chess_Env
from agent import Agent
from chess.pgn import Game as ChessGame
from edge import Edge
from mcts import MCTS

class Game:
    """
        The Game class is used to play games between two agents.
    """
    def __init__(self, env: Chess_Env, white: Agent, black: Agent):
        """
            Initialize the game with the environment and two agents.
            If no agent is provided, a random agent is created.
        """
        self.env = env
        self.white = white
        self.black = black
        
        self.game_over = False
        self.winner = None
        self.moves = []
        self.reset()

    def reset(self):
        """
            Reset the game to the initial state.
        """
        self.env.reset()
        self.turn = self.env.board.turn # flip the board if black starts

    @staticmethod
    def get_winner(result: str) -> int:
        """
            Get the winner of the game based on the result string.
            Returns 1 for white win, -1 for black win, and 0 for draw.
        """
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        elif result == '1/2-1/2':
            return 0
        else:
            raise ValueError(f"Invalid result string: {result}")
        
    def play_move(self, stochastic: bool = True, previous_moves: tuple[Edge, Edge] = (None, None), save_moves: bool = True) -> None:
        """
            Play one move. If stochastic is True, the move is chosen using a probability distribution.
            Otherwise, the move is chosen based on the highest N (deterministically).
            The previous moves are used to reuse the MCTS tree (if possible): the root node is set to the
            node found after playing the previous moves in the current tree.
        """
        if self.game_over:
            raise ValueError("Game is over. Cannot play more moves.")
        
        if self.turn == chess.WHITE:
            agent = self.white
        else:
            agent = self.black
        
        # Play the move
        move = agent.play_move(self.env, stochastic=stochastic, previous_moves=previous_moves)
        
        # Update the environment
        self.env.push(move)
        
        # Save the move
        if save_moves:
            self.moves.append(move)
        
        # Check if the game is over
        if self.env.is_game_over():
            self.game_over = True
            self.winner = Game.get_winner(self.env.get_result())
        
        # Switch turn
        self.turn = not self.turn

    def save_to_memory(self, state, moves) -> None:
        """
        Append the current state and move probabilities to the internal memory.
        """
        if not hasattr(self, 'memory'):
            self.memory = []
        
        # Save the state and moves to memory
        self.memory.append((state, moves))
        
        # Limit the memory size
        if len(self.memory) > config.MAX_MEMORY_SIZE:
            self.memory.pop(0)
        
        logging.info(f"Memory size: {len(self.memory)}")

    def save_game(self, name: str = "game", full_game: bool = False) -> None:
        """
        Save the internal memory to a .npy file.
        """
        if not hasattr(self, 'memory'):
            logging.warning("No memory to save.")
            return
        
        # Create the directory if it doesn't exist
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        
        # Save the memory to a .npy file
        file_path = os.path.join(config.MEMORY_DIR, f"{name}.npy")
        np.save(file_path, self.memory)
        
        logging.info(f"Game saved to {file_path}")

    def push(self, move):
        """
        Push a move to the board. This is a compatibility method used by the visualization.
        """
        # Make the move on the environment
        success = self.env.push(move)
        
        # Store the move if successful
        if success:
            self.moves.append(move)
            
            # Update game state
            if self.env.is_game_over():
                self.game_over = True
                try:
                    self.winner = Game.get_winner(self.env.get_result())
                except ValueError:
                    # Handle undecided results
                    self.winner = 0
            
            # Switch turn
            self.turn = not self.turn
        
        return success

    def current_player_is_white(self):
        """
        Check if the current player is white.
        """
        return self.env.board.turn == chess.WHITE

    def result(self):
        """
        Get the result of the game.
        """
        return self.env.board.result() if self.env.is_game_over() else "*"

    def is_over(self):
        """
        Check if the game is over.
        """
        return self.env.is_game_over()
            