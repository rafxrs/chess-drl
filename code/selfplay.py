"""
Self-play: the latest model plays games against itself, and every position
becomes a (state, policy, value) training example.
"""
import logging
import time

import chess
import numpy as np
import torch.multiprocessing as torch_mp
from tqdm import tqdm

import config
from agent import Agent
from utils import move_to_index

logging.basicConfig(level=logging.INFO, format=" %(message)s")

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def adjudicate(board):
    """Score a game stopped at the move limit: the side up by ADJUDICATION_MARGIN in material wins."""
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    if balance >= config.ADJUDICATION_MARGIN:
        return "1-0"
    if balance <= -config.ADJUDICATION_MARGIN:
        return "0-1"
    return "1/2-1/2"


def play_one_game(model_path, device=None, simulations=None):
    """Play one self-play game. Returns (states, policies, values, info)."""
    simulations = simulations or config.SIMULATIONS_PER_MOVE

    agent = Agent(model_path=model_path, device=device, explore=True)
    board = chess.Board()
    states, policies = [], []

    while not board.is_game_over() and board.ply() < config.MAX_GAME_MOVES:
        agent.state = board.fen()
        agent.run_simulations(simulations)
        actions, probs = agent.mcts.get_move_probs()

        policy = np.zeros(config.OUTPUT_SHAPE[0], dtype=np.float32)
        for action, prob in zip(actions, probs):
            policy[move_to_index(action)] = prob
        states.append(board.copy())
        policies.append(policy)

        # Sample early moves for variety, then play the strongest move (AlphaZero's temperature schedule).
        if board.ply() < config.TEMPERATURE_MOVES:
            move = actions[np.random.choice(len(actions), p=probs)]
        else:
            move = agent.mcts.best_move()
        board.push(move)

    adjudicated = not board.is_game_over()
    result = adjudicate(board) if adjudicated else board.result()
    outcome = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
    # Values are from the side to move, which alternates every ply starting with White.
    values = [outcome if i % 2 == 0 else -outcome for i in range(len(states))]

    info = {"result": result, "adjudicated": adjudicated, "plies": board.ply()}
    return states, policies, values, info


def _worker(args):
    model_path, device, simulations = args
    try:
        return play_one_game(model_path, device, simulations)
    except Exception as e:
        logging.error(f"Self-play game failed: {e}")
        return None


def generate_selfplay_data(model_path, n_games, device=None, simulations=None, num_workers=None):
    """
    Play n_games in parallel worker processes.

    Returns (states, policies, values, game_infos): examples pooled across games,
    plus one info dict per finished game.
    """
    num_workers = min(num_workers or config.NUM_WORKERS, n_games)
    start = time.time()
    args = [(model_path, device, simulations)] * n_games

    if num_workers <= 1:
        results = [_worker(a) for a in tqdm(args, desc="Self-play")]
    else:
        with torch_mp.get_context("spawn").Pool(num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, args), total=n_games, desc="Self-play"))

    all_states, all_policies, all_values, infos = [], [], [], []
    for result in results:
        if result is None:
            continue
        states, policies, values, info = result
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
        infos.append(info)

    elapsed = time.time() - start
    logging.info(f"Generated {len(all_states)} positions from {len(infos)} games in {elapsed:.1f}s")
    return all_states, all_policies, all_values, infos
