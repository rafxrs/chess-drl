"""
Measure playing strength: the model plays matches against a random mover and
against an earlier checkpoint, alternating colours. Used by train.py.
"""
import logging
import random

import chess
import torch.multiprocessing as torch_mp

import config
from agent import Agent
from selfplay import adjudicate


def play_benchmark_game(model_path, opponent_path, model_is_white, device, simulations):
    """
    Play one game; returns the model's score (1 win, 0.5 draw, 0 loss).
    `opponent_path=None` means the opponent plays random legal moves.
    """
    model = Agent(model_path, device=device)
    opponent = Agent(opponent_path, device=device) if opponent_path else None
    board = chess.Board()

    while not board.is_game_over() and board.ply() < config.MAX_GAME_MOVES:
        model_to_move = (board.turn == chess.WHITE) == model_is_white
        player = model if model_to_move else opponent
        if board.ply() < config.BENCHMARK_OPENING_PLIES or player is None:
            # A few random opening moves so games between two deterministic bots differ.
            move = random.choice(list(board.legal_moves))
        else:
            player.state = board.fen()
            player.run_simulations(simulations)
            move = player.mcts.best_move()
        board.push(move)

    result = board.result() if board.is_game_over() else adjudicate(board)
    if result == "1/2-1/2":
        return 0.5
    return 1.0 if (result == "1-0") == model_is_white else 0.0


def _worker(args):
    try:
        return play_benchmark_game(*args)
    except Exception as e:
        logging.error(f"Benchmark game failed: {e}")
        return None


def benchmark(model_path, opponent_path, n_games, device=None, simulations=None, num_workers=None):
    """Average score of `model_path` over n_games against `opponent_path` (None = random mover)."""
    simulations = simulations or config.SIMULATIONS_PER_MOVE
    args = [(model_path, opponent_path, i % 2 == 0, device, simulations) for i in range(n_games)]
    num_workers = min(num_workers or config.NUM_WORKERS, n_games)

    if num_workers <= 1:
        scores = [_worker(a) for a in args]
    else:
        with torch_mp.get_context("spawn").Pool(num_workers) as pool:
            scores = pool.map(_worker, args)

    scores = [s for s in scores if s is not None]
    return sum(scores) / len(scores) if scores else float("nan")
