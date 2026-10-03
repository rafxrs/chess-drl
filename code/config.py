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
PLAY_SIMULATIONS = int(os.environ.get("PLAY_SIMULATIONS", 200))  # used by play.py and gui_play.py

# ---------- Self-play ----------
TEMPERATURE_MOVES = int(os.environ.get("TEMPERATURE_MOVES", 30))  # plies sampled by visit count; best move after
MAX_GAME_MOVES = int(os.environ.get("MAX_GAME_MOVES", 150))      # plies before a game is stopped and adjudicated
ADJUDICATION_MARGIN = int(os.environ.get("ADJUDICATION_MARGIN", 3))  # material lead (pawns) that wins a stopped game

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
# Iterations at which the learning rate drops 10x, as AlphaZero did once progress levels off.
LR_MILESTONES = [int(x) for x in os.environ.get("LR_MILESTONES", "100,300").split(",") if x.strip()]
WEIGHT_DECAY = 1e-4

# ---------- Training loop ----------
N_SELFPLAY_GAMES = int(os.environ.get("N_SELFPLAY_GAMES", 20))              # self-play games per iteration
N_EPOCHS_PER_ITERATION = int(os.environ.get("N_EPOCHS_PER_ITERATION", 10))  # training passes per new position
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", 5))               # keep a model_iter_<N>.pt every N iterations
MAX_REPLAY_MEMORY = int(os.environ.get("MAX_REPLAY_MEMORY", 50000))         # recent positions trained on (~20 iterations)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))      # parallel self-play processes

# ---------- Strength benchmark ----------
BENCHMARK_EVERY = int(os.environ.get("BENCHMARK_EVERY", 10))       # iterations between benchmarks (0 = never)
BENCHMARK_GAMES = int(os.environ.get("BENCHMARK_GAMES", 10))       # games against each opponent
BENCHMARK_OPENING_PLIES = int(os.environ.get("BENCHMARK_OPENING_PLIES", 4))  # random opening plies, for variety

# ---------- Paths ----------
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "./models")
MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
LOG_DIR = os.environ.get("LOG_FOLDER", "./logs")
LATEST_MODEL_PATH = os.path.join(MODEL_FOLDER, "latest.pt")
ARCHIVE_DIR = os.environ.get("ARCHIVE_FOLDER", "./archive")
TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
