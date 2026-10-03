"""
Train the bot through continuous self-play, AlphaZero style: the latest model
plays itself, trains on the accumulated games, and the improved model plays
the next round. Runs until stopped; progress is saved after every iteration.
"""
import argparse
import csv
import logging
import os
import glob
import random
import re
import shutil
from collections import deque
from datetime import datetime

import chess
import numpy as np
import torch
import torch.optim as optim

import config
from agent import Agent
from benchmark import benchmark
from model import RLModelBuilder
from selfplay import generate_selfplay_data

logging.basicConfig(level=logging.INFO, format=" %(message)s")

REPLAY_BUFFER_PATH = os.path.join(config.MEMORY_DIR, "selfplay_data.npz")

LOG_FIELDS = [
    "iteration", "timestamp", "games", "new_positions", "buffer_size",
    "loss", "policy_loss", "value_loss",
    "white_wins", "black_wins", "draws", "adjudicated", "avg_plies", "checkpoint",
    "learning_rate", "vs_random", "vs_checkpoint", "vs_checkpoint_iter",
]


def learning_rate(iteration):
    """LEARNING_RATE, divided by 10 at each milestone already reached."""
    return config.LEARNING_RATE * 0.1 ** sum(iteration >= m for m in config.LR_MILESTONES)


def earlier_checkpoint(model_dir, max_iteration):
    """Path and iteration of the newest model_iter_<N>.pt with N <= max_iteration, or (None, None)."""
    found = []
    for path in glob.glob(os.path.join(model_dir, "model_iter_*.pt")):
        match = re.search(r"model_iter_(\d+)\.pt$", path)
        if match and int(match.group(1)) <= max_iteration:
            found.append((int(match.group(1)), path))
    if not found:
        return None, None
    n, path = max(found)
    return path, n


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


def archive_previous_run():
    """Move models, replay data and logs into archive/<timestamp>/ so a fresh run starts clean."""
    target = os.path.join(config.ARCHIVE_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    for folder in (config.MODEL_FOLDER, config.MEMORY_DIR, config.LOG_DIR):
        if os.path.isdir(folder) and os.listdir(folder):
            os.makedirs(target, exist_ok=True)
            shutil.move(folder, os.path.join(target, os.path.basename(os.path.normpath(folder))))
    if os.path.isdir(target):
        logging.info(f"Moved the previous run to {target}")


def run_iteration(iteration, args, model_path, replay_buffer, device):
    logging.info(f"\n===== Iteration {iteration} =====")

    states, policies, values, games = generate_selfplay_data(
        model_path, args.games_per_iteration, device=device, simulations=args.simulations
    )
    for example in zip(states, policies, values):
        replay_buffer.append(example)
    save_replay_buffer(REPLAY_BUFFER_PATH, replay_buffer)

    results = [g["result"] for g in games]
    white_wins, black_wins = results.count("1-0"), results.count("0-1")
    draws = len(results) - white_wins - black_wins
    adjudicated = sum(g["adjudicated"] for g in games)
    avg_plies = float(np.mean([g["plies"] for g in games])) if games else 0.0
    logging.info(
        f"Results: {white_wins} white wins, {black_wins} black wins, {draws} draws "
        f"({adjudicated} adjudicated on material), {avg_plies:.0f} plies on average"
    )

    if len(replay_buffer) < args.batch_size:
        logging.info(f"Only {len(replay_buffer)} positions so far; need {args.batch_size} to train.")
        return

    model = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]).build_model(model_path, device)
    lr = learning_rate(iteration)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    # Keep Adam's running statistics across iterations instead of restarting them every time.
    optimizer_path = os.path.join(args.model_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        except (ValueError, RuntimeError):
            logging.warning("Saved optimizer state doesn't match the model; starting it fresh.")
    for group in optimizer.param_groups:
        group["lr"] = lr

    # Scale with the new data, so each position is trained on about epochs_per_iteration times.
    n_batches = args.epochs_per_iteration * max(1, len(states) // args.batch_size)
    logging.info(f"Training for {n_batches} batches on {len(replay_buffer)} positions...")
    stats = train_on_batches(model, optimizer, replay_buffer, device, n_batches, args.batch_size)
    logging.info(
        f"Loss: {stats['loss']:.4f} (policy {stats['policy_loss']:.4f}, value {stats['value_loss']:.4f}), "
        f"learning rate {lr:g}"
    )

    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    checkpoint = ""
    if iteration % config.CHECKPOINT_EVERY == 0:
        checkpoint = os.path.join(args.model_dir, f"model_iter_{iteration}.pt")
        torch.save(model.state_dict(), checkpoint)
        logging.info(f"Saved checkpoint {checkpoint}")

    vs_random = vs_checkpoint = vs_checkpoint_iter = ""
    if config.BENCHMARK_EVERY and iteration % config.BENCHMARK_EVERY == 0:
        n = config.BENCHMARK_GAMES
        logging.info(f"Benchmark: {n} games against a random mover...")
        vs_random = benchmark(model_path, None, n, device=device, simulations=args.simulations)
        message = f"Benchmark score: {vs_random:.0%} vs random"
        opponent, vs_checkpoint_iter = earlier_checkpoint(args.model_dir, iteration - config.BENCHMARK_EVERY)
        if opponent:
            logging.info(f"Benchmark: {n} games against iteration {vs_checkpoint_iter}...")
            vs_checkpoint = benchmark(model_path, opponent, n, device=device, simulations=args.simulations)
            message += f", {vs_checkpoint:.0%} vs iteration {vs_checkpoint_iter}"
        logging.info(message + " (a draw counts as half)")

    append_log(config.TRAINING_LOG_PATH, {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "games": len(games),
        "new_positions": len(states),
        "buffer_size": len(replay_buffer),
        **stats,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "adjudicated": adjudicated,
        "avg_plies": avg_plies,
        "checkpoint": checkpoint,
        "learning_rate": lr,
        "vs_random": vs_random,
        "vs_checkpoint": vs_checkpoint,
        "vs_checkpoint_iter": "" if vs_checkpoint_iter is None else vs_checkpoint_iter,
    })


def main():
    parser = argparse.ArgumentParser(description="Train the chess bot through continuous self-play")
    parser.add_argument("--iterations", type=int, default=0, help="Iterations to run (0 = run until stopped)")
    parser.add_argument("--games-per-iteration", type=int, default=config.N_SELFPLAY_GAMES)
    parser.add_argument("--epochs-per-iteration", type=int, default=config.N_EPOCHS_PER_ITERATION)
    parser.add_argument("--simulations", type=int, default=config.SIMULATIONS_PER_MOVE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--model-dir", type=str, default=config.MODEL_FOLDER)
    parser.add_argument("--fresh", action="store_true", help="Archive the current run and start over from a new model")
    args = parser.parse_args()

    if args.fresh:
        archive_previous_run()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = config.DEVICE
    logging.info(f"Using device: {device}")
    if device.type == "cpu" and config.USE_GPU and shutil.which("nvidia-smi"):
        logging.warning(
            "An NVIDIA GPU was found but this PyTorch build can't use it (CPU-only install). "
            "See 'GPU setup' in the README to install the CUDA build."
        )

    model_path = os.path.join(args.model_dir, "latest.pt")
    if not os.path.exists(model_path):
        logging.info("Initializing a new model from scratch")
        model = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]).build_model(None, device)
        torch.save(model.state_dict(), model_path)

    replay_buffer = load_replay_buffer(REPLAY_BUFFER_PATH, config.MAX_REPLAY_MEMORY)

    rows = load_log(config.TRAINING_LOG_PATH)
    iteration = max((int(r["iteration"]) for r in rows), default=0)
    if iteration:
        logging.info(f"Resuming after iteration {iteration}")

    completed = 0
    try:
        while args.iterations == 0 or completed < args.iterations:
            iteration += 1
            run_iteration(iteration, args, model_path, replay_buffer, device)
            completed += 1
    except KeyboardInterrupt:
        logging.info("\nStopped. Progress up to the last completed iteration is saved; run the same command to resume.")


if __name__ == "__main__":
    main()
