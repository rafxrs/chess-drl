# Plays two model checkpoints against each other to decide whether a newly
# trained candidate is actually stronger than the current best model.
import logging
import os
import time

import chess
import numpy as np
from tqdm import tqdm

import config
from agent import Agent
from env import Chess_Env

logging.basicConfig(level=logging.INFO, format=" %(message)s")


class Evaluator:
    def __init__(self, candidate_path: str, reference_path: str, device=None):
        for path in (candidate_path, reference_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

        self.candidate_path = candidate_path
        self.reference_path = reference_path
        self.device = device or config.DEVICE

    def evaluate(self, n_games: int = 10, simulations_per_move: int = None, verbose: bool = True) -> dict:
        """
        Play n_games total, alternating which side the candidate plays, and
        return a stats dict describing how the candidate fared against the
        reference model.
        """
        candidate = Agent(model_path=self.candidate_path, device=self.device)
        reference = Agent(model_path=self.reference_path, device=self.device)
        if simulations_per_move:
            candidate.mcts.n_simulations = simulations_per_move
            reference.mcts.n_simulations = simulations_per_move

        wins = draws = losses = 0
        start = time.time()

        for game_idx in tqdm(range(n_games), disable=not verbose, desc="Evaluating"):
            candidate_is_white = game_idx % 2 == 0
            white, black = (candidate, reference) if candidate_is_white else (reference, candidate)

            env = Chess_Env()
            while not env.is_game_over():
                mover = white if env.board.turn == chess.WHITE else black
                env.push(mover.get_move(env))

            result = env.get_result()
            candidate_won = (result == "1-0") == candidate_is_white
            reference_won = (result == "1-0") == (not candidate_is_white)
            if result == "1/2-1/2":
                draws += 1
            elif candidate_won:
                wins += 1
            elif reference_won:
                losses += 1

        elapsed = time.time() - start
        win_rate = wins / n_games
        draw_rate = draws / n_games
        loss_rate = losses / n_games

        score = win_rate + 0.5 * draw_rate
        if score >= 1.0:
            elo_diff = 400.0
        elif score <= 0.0:
            elo_diff = -400.0
        else:
            elo_diff = 400 * np.log10(score / (1 - score))

        stats = {
            "games": n_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "score": score,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "elo_difference": elo_diff,
            "elapsed_seconds": elapsed,
        }

        if verbose:
            logging.info(
                f"Evaluation: {wins}W/{draws}D/{losses}L over {n_games} games "
                f"(score {score:.1%}, Elo diff {elo_diff:+.1f}) in {elapsed:.1f}s"
            )

        return stats
