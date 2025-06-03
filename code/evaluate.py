import os
import time
import logging
import numpy as np
import argparse

from tqdm import tqdm
from agent import Agent
from env import Chess_Env
from game import Game

logging.basicConfig(level=logging.INFO, format=' %(message)s')

class Evaluator:
    """
    Class for evaluating chess models against each other.
    """
    def __init__(self, model_1_path: str, model_2_path: str):
        """
        Initialize the evaluator with two model paths.
        
        Args:
            model_1_path: Path to the first model
            model_2_path: Path to the second model
        """
        self.model_1_path = model_1_path
        self.model_2_path = model_2_path
        
        # Check if model files exist
        for path in [model_1_path, model_2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
        
        logging.info(f"Model 1: {os.path.basename(model_1_path)}")
        logging.info(f"Model 2: {os.path.basename(model_2_path)}")

    def evaluate(self, n_games: int, verbose: bool = True, simulations_per_move: int = None):
        """
        Evaluate the models by playing n_games against each other (2*n_games total games).
        Each model plays both as white and black.
        
        Args:
            n_games: Number of games to play (each model plays n_games as white)
            verbose: Whether to print progress information
            simulations_per_move: Override config.SIMULATIONS_PER_MOVE
            
        Returns:
            A results dictionary and a summary string
        """
        score = {
            "model_1_wins": 0,
            "model_2_wins": 0,
            "draws": 0,
            "total_games": 2 * n_games
        }
        
        # Create agents
        agent_1 = Agent(model_path=self.model_1_path)
        agent_2 = Agent(model_path=self.model_2_path)
        
        # Set simulations per move if provided
        if simulations_per_move:
            agent_1.mcts.n_simulations = simulations_per_move
            agent_2.mcts.n_simulations = simulations_per_move
        
        start_time = time.time()
        
        # First round: agent_1 as white, agent_2 as black
        if verbose:
            logging.info(f"Playing {n_games} games with Model 1 as White...")
        
        for i in tqdm(range(n_games), disable=not verbose):
            env = Chess_Env()
            game = Game(env, agent_1, agent_2)
            game.reset()
            
            # Play until game is over
            while not game.env.is_game_over():
                if game.env.board.turn:  # White's turn
                    move = agent_1.get_move(game.env)
                else:  # Black's turn
                    move = agent_2.get_move(game.env)
                game.env.push(move)
            
            # Get result
            result = game.env.get_result()
            if result == '1-0':
                score["model_1_wins"] += 1
            elif result == '0-1':
                score["model_2_wins"] += 1
            else:
                score["draws"] += 1
        
        # Second round: agent_2 as white, agent_1 as black
        if verbose:
            logging.info(f"Playing {n_games} games with Model 2 as White...")
        
        for i in tqdm(range(n_games), disable=not verbose):
            env = Chess_Env()
            game = Game(env, agent_2, agent_1)
            game.reset()
            
            # Play until game is over
            while not game.env.is_game_over():
                if game.env.board.turn:  # White's turn
                    move = agent_2.get_move(game.env)
                else:  # Black's turn
                    move = agent_1.get_move(game.env)
                game.env.push(move)
            
            # Get result
            result = game.env.get_result()
            if result == '1-0':
                score["model_2_wins"] += 1
            elif result == '0-1':
                score["model_1_wins"] += 1
            else:
                score["draws"] += 1
        
        elapsed_time = time.time() - start_time
        
        # Calculate winning percentages
        m1_win_pct = score["model_1_wins"] / score["total_games"] * 100
        m2_win_pct = score["model_2_wins"] / score["total_games"] * 100
        draws_pct = score["draws"] / score["total_games"] * 100
        
        # Calculate ELO difference (approximate)
        if m1_win_pct > 0 and m2_win_pct > 0:
            elo_diff = 400 * np.log10((m1_win_pct + 0.5 * draws_pct) / (m2_win_pct + 0.5 * draws_pct))
        else:
            elo_diff = 0
            
        # Create results summary
        summary = (
            f"\n{'='*50}\n"
            f"EVALUATION RESULTS\n"
            f"{'='*50}\n"
            f"Model 1: {os.path.basename(self.model_1_path)}\n"
            f"Model 2: {os.path.basename(self.model_2_path)}\n"
            f"Total games: {score['total_games']}\n"
            f"Time elapsed: {elapsed_time:.1f} seconds\n"
            f"\n"
            f"Model 1 wins: {score['model_1_wins']} ({m1_win_pct:.1f}%)\n"
            f"Model 2 wins: {score['model_2_wins']} ({m2_win_pct:.1f}%)\n"
            f"Draws: {score['draws']} ({draws_pct:.1f}%)\n"
            f"\n"
            f"Estimated ELO difference: {elo_diff:.1f} {'(Model 1 stronger)' if elo_diff > 0 else '(Model 2 stronger)'}\n"
            f"{'='*50}"
        )
        
        if verbose:
            print(summary)
            
        return score, summary

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate two chess models against each other")
    parser.add_argument("model_1", type=str, help="Path to first model")
    parser.add_argument("model_2", type=str, help="Path to second model")
    parser.add_argument("--games", "-g", type=int, default=10, help="Number of games to play (per side)")
    parser.add_argument("--sims", "-s", type=int, default=None, help="Simulations per move (overrides config)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = Evaluator(args.model_1, args.model_2)
    evaluator.evaluate(args.games, verbose=not args.quiet, simulations_per_move=args.sims)

if __name__ == "__main__":
    main()