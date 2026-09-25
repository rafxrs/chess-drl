# The one command that runs the whole self-play-and-training loop:
# self-play with the current best model -> train a candidate on the
# accumulated data -> evaluate the candidate against the best model ->
# promote it if it's actually stronger -> repeat, forever, until stopped.
import argparse
import csv
import logging
import os
import random
from collections import deque
from datetime import datetime

import chess
import numpy as np
import torch
import torch.optim as optim

import config
from agent import Agent
from evaluate import Evaluator
from model import RLModelBuilder
from selfplay import generate_selfplay_data

logging.basicConfig(level=logging.INFO, format=" %(message)s")

REPLAY_BUFFER_PATH_NAME = "selfplay_data.npz"

LOG_FIELDS = [
    "iteration", "timestamp", "new_positions", "buffer_size",
    "loss", "policy_loss", "value_loss",
    "eval_games", "win_rate", "score", "elo_difference", "promoted",
]


def train_on_batches(model, optimizer, replay_buffer, device, n_batches, batch_size):
    model.train()
    losses, policy_losses, value_losses = [], [], []

    for _ in range(n_batches):
        batch = random.sample(replay_buffer, min(batch_size, len(replay_buffer)))
        states, policies, values = zip(*batch)

        state_tensors = torch.cat([Agent.state_to_tensor(s) for s in states]).to(device)
        policy_targets = torch.as_tensor(np.array(policies), dtype=torch.float32, device=device)
        value_targets = torch.as_tensor(np.array(values), dtype=torch.float32, device=device).unsqueeze(1)

        optimizer.zero_grad()
        policy_logits, value_preds = model(state_tensors)

        policy_loss = -(policy_targets * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(value_preds, value_targets)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
    }


def save_replay_buffer(path, buffer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    states = [s.fen() for s, _, _ in buffer]
    policies = [p for _, p, _ in buffer]
    values = [v for _, _, v in buffer]
    np.savez_compressed(path, states=states, policies=policies, values=values)


def load_replay_buffer(path, maxlen):
    buffer = deque(maxlen=maxlen)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        for fen, policy, value in zip(data["states"], data["policies"], data["values"]):
            buffer.append((chess.Board(fen), policy, value))
        logging.info(f"Loaded {len(buffer)} positions from {path}")
    return buffer


def get_next_iteration(log_path):
    rows = load_log(log_path)
    if not rows:
        return 1
    return max(int(r["iteration"]) for r in rows) + 1


def load_log(log_path):
    if not os.path.exists(log_path):
        return []
    with open(log_path, newline="") as f:
        return list(csv.DictReader(f))


def append_log(log_path, row):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rows = load_log(log_path)
    if rows and list(rows[0].keys()) != LOG_FIELDS:
        # Log was written with older columns: rewrite it with the current header.
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_iteration(iteration, args, best_model_path, latest_model_path, replay_buffer, device):
    logging.info(f"\n===== Iteration {iteration} =====")

    logging.info(f"Self-play: generating {args.games_per_iteration} games with the current best model...")
    states, policies, values = generate_selfplay_data(
        best_model_path, args.games_per_iteration, device=device, simulations=args.simulations
    )
    for example in zip(states, policies, values):
        replay_buffer.append(example)
    save_replay_buffer(os.path.join(config.MEMORY_DIR, REPLAY_BUFFER_PATH_NAME), replay_buffer)

    if len(replay_buffer) < args.batch_size:
        logging.info(f"Only {len(replay_buffer)} positions collected so far; need {args.batch_size} to train. Skipping training this round.")
        return

    # Keep training the same network across iterations, even when it isn't
    # promoted; otherwise every candidate restarts from a stale best model.
    start_path = latest_model_path if os.path.exists(latest_model_path) else best_model_path
    model = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]).build_model(start_path, device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Scale with the new data rather than the whole buffer, so older positions
    # aren't re-trained on over and over now that training carries over.
    n_batches = args.epochs_per_iteration * max(1, len(states) // args.batch_size)
    logging.info(f"Training on {len(replay_buffer)} positions for {n_batches} batches...")
    train_stats = train_on_batches(model, optimizer, replay_buffer, device, n_batches, args.batch_size)
    logging.info(
        f"Loss: {train_stats['loss']:.4f} "
        f"(policy {train_stats['policy_loss']:.4f}, value {train_stats['value_loss']:.4f})"
    )

    torch.save(model.state_dict(), latest_model_path)
    candidate_path = os.path.join(args.model_dir, f"model_iter_{iteration}.pt")
    torch.save(model.state_dict(), candidate_path)

    logging.info(f"Evaluating candidate against current best over {args.eval_games} games...")
    eval_stats = Evaluator(candidate_path, best_model_path, device=device).evaluate(
        n_games=args.eval_games, simulations_per_move=args.simulations
    )

    promoted = eval_stats["score"] >= config.WIN_RATE_THRESHOLD
    if promoted:
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model! Score {eval_stats['score']:.1%} (kept as {candidate_path})")
    else:
        logging.info(f"Candidate not promoted: score {eval_stats['score']:.1%} < {config.WIN_RATE_THRESHOLD:.0%} threshold")
        os.remove(candidate_path)

    append_log(config.TRAINING_LOG_PATH, {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "new_positions": len(states),
        "buffer_size": len(replay_buffer),
        "loss": train_stats["loss"],
        "policy_loss": train_stats["policy_loss"],
        "value_loss": train_stats["value_loss"],
        "eval_games": args.eval_games,
        "win_rate": eval_stats["win_rate"],
        "score": eval_stats["score"],
        "elo_difference": eval_stats["elo_difference"],
        "promoted": promoted,
    })


def main():
    parser = argparse.ArgumentParser(description="Train the chess bot through continuous self-play")
    parser.add_argument("--iterations", type=int, default=0, help="Number of self-play/train iterations to run (0 = run forever)")
    parser.add_argument("--games-per-iteration", type=int, default=config.N_SELFPLAY_GAMES)
    parser.add_argument("--epochs-per-iteration", type=int, default=config.N_EPOCHS_PER_ITERATION)
    parser.add_argument("--eval-games", type=int, default=config.EVALUATION_GAMES)
    parser.add_argument("--simulations", type=int, default=config.SIMULATIONS_PER_MOVE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--model-dir", type=str, default=config.MODEL_FOLDER)
    parser.add_argument("--fresh", action="store_true", help="Start over from a newly initialized model instead of resuming")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = config.DEVICE
    logging.info(f"Using device: {device}")

    best_model_path = os.path.join(args.model_dir, "best.pt")
    latest_model_path = os.path.join(args.model_dir, "latest.pt")
    if args.fresh or not os.path.exists(best_model_path):
        logging.info("Initializing a new model from scratch")
        model = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]).build_model(None, device)
        torch.save(model.state_dict(), best_model_path)
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)

    replay_buffer = load_replay_buffer(os.path.join(config.MEMORY_DIR, REPLAY_BUFFER_PATH_NAME), config.MAX_REPLAY_MEMORY)

    start_iteration = get_next_iteration(config.TRAINING_LOG_PATH)
    if start_iteration > 1:
        logging.info(f"Resuming from iteration {start_iteration} (found existing training log)")

    completed = 0
    iteration = start_iteration - 1
    try:
        while args.iterations == 0 or completed < args.iterations:
            iteration += 1
            completed += 1
            run_iteration(iteration, args, best_model_path, latest_model_path, replay_buffer, device)
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted. All progress up to the last completed iteration was saved; re-run this command to resume.")


if __name__ == "__main__":
    main()
