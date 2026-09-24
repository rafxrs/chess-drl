# Self-play data generation: the current best model plays games against
# itself, and every position it saw becomes a (state, policy, value)
# training example for train.py.
import logging
import random
import time

import chess
import numpy as np
import torch.multiprocessing as torch_mp
from tqdm import tqdm

import config
from agent import Agent
from utils import move_to_index

logging.basicConfig(level=logging.INFO, format=" %(message)s")


def play_one_game(model_path, device=None, simulations=None, max_moves=None):
    """Play a single self-play game and return (states, policies, values) training examples."""
    simulations = simulations or config.SIMULATIONS_PER_MOVE
    max_moves = max_moves or config.MAX_GAME_MOVES

    agent = Agent(model_path=model_path, device=device)
    agent.mcts.n_simulations = simulations
    board = chess.Board()
    states, policies = [], []

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        agent.state = board.fen()
        agent.run_simulations(simulations)
        actions, probs = agent.mcts.get_move_probs()

        legal = [(a, p) for a, p in zip(actions, probs) if a in board.legal_moves]
        if legal:
            legal_actions, legal_probs = zip(*legal)
            legal_probs = np.array(legal_probs) / sum(legal_probs)
            move = np.random.choice(legal_actions, p=legal_probs)
        else:
            move = random.choice(list(board.legal_moves))

        policy = np.zeros(config.OUTPUT_SHAPE[0], dtype=np.float32)
        for action, prob in zip(actions, probs):
            idx = move_to_index(action)
            if idx < len(policy):
                policy[idx] = prob

        states.append(board.copy())
        policies.append(policy)

        board.push(move)
        move_count += 1

    result = board.result() if board.is_game_over() else "1/2-1/2"
    outcome = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
    # states alternate whose turn it was, so flip the outcome's perspective each ply
    values = [outcome if i % 2 == 0 else -outcome for i in range(len(states))]

    return states, policies, values


def _worker(args):
    model_path, device, simulations, max_moves = args
    try:
        return play_one_game(model_path, device, simulations, max_moves)
    except Exception as e:
        logging.error(f"Self-play game failed: {e}")
        return [], [], []


def generate_selfplay_data(model_path, n_games, device=None, simulations=None, num_workers=None):
    """
    Generate self-play training examples using multiple worker processes.

    Returns:
        (states, policies, values): flat lists of training examples pooled across all games.
    """
    num_workers = min(num_workers or config.NUM_WORKERS, n_games)
    max_moves = config.MAX_GAME_MOVES
    start = time.time()

    args = [(model_path, device, simulations, max_moves) for _ in range(n_games)]

    if num_workers <= 1:
        results = [_worker(a) for a in tqdm(args, desc="Self-play")]
    else:
        mp_context = torch_mp.get_context("spawn")
        with mp_context.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, args), total=n_games, desc="Self-play"))

    all_states, all_policies, all_values = [], [], []
    for states, policies, values in results:
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)

    elapsed = time.time() - start
    logging.info(f"Generated {len(all_states)} positions from {n_games} games in {elapsed:.1f}s")
    return all_states, all_policies, all_values
