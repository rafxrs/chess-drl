"""
Hyperparameters and paths shared by all four commands.

Anything read from os.environ can be overridden with an environment variable
or a .env file in the directory you run from, e.g. `SIMULATIONS_PER_MOVE=200`.
"""
import os

import torch
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


# ---------- Device ----------
USE_GPU = _env_bool("USE_GPU", True)
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# ---------- MCTS ----------
SIMULATIONS_PER_MOVE = int(os.environ.get("SIMULATIONS_PER_MOVE", 100))
C_init = 2                # exploration constant (c_puct)
DIRICHLET_NOISE = 0.3     # alpha of the root exploration noise
DIRICHLET_EPSILON = 0.25  # weight of that noise vs. the network prior
MAX_GAME_MOVES = int(os.environ.get("MAX_GAME_MOVES", 150))  # self-play games are scored a draw after this

# ---------- Network input: 19 planes of 8x8 ----------
# 12 piece planes + en passant + side to move + 4 castling rights + halfmove clock
n = 8
amount_of_input_planes = (2 * 6 + 1) + (1 + 4 + 1)
INPUT_SHAPE = (n, n, amount_of_input_planes)

# ---------- Network output: policy (4672) + value (1) ----------
# 73 move types per from-square: 56 queen-like, 8 knight, 9 underpromotions
queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
amount_of_planes = queen_planes + knight_planes + underpromotion_planes
OUTPUT_SHAPE = (8 * 8 * amount_of_planes, 1)

# ---------- Network size and optimizer ----------
# Kept small so one training iteration runs in minutes; raise for more strength.
CONVOLUTION_FILTERS = int(os.environ.get("CONVOLUTION_FILTERS", 64))
AMOUNT_OF_RESIDUAL_BLOCKS = int(os.environ.get("RESIDUAL_BLOCKS", 6))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-3))
WEIGHT_DECAY = 1e-4

# ---------- Training loop ----------
N_SELFPLAY_GAMES = int(os.environ.get("N_SELFPLAY_GAMES", 20))              # self-play games per iteration
N_EPOCHS_PER_ITERATION = int(os.environ.get("N_EPOCHS_PER_ITERATION", 5))  # passes over the replay buffer
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
EVALUATION_GAMES = int(os.environ.get("EVALUATION_GAMES", 10))              # candidate vs. best match length
WIN_RATE_THRESHOLD = float(os.environ.get("WIN_RATE_THRESHOLD", 0.55))      # win rate needed to promote
MAX_REPLAY_MEMORY = int(os.environ.get("MAX_REPLAY_MEMORY", 200000))        # positions kept for training
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))      # parallel self-play processes

# ---------- Paths ----------
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "./models")
MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
LOG_DIR = os.environ.get("LOG_FOLDER", "./logs")
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")
TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
