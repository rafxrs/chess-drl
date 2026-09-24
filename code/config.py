# Hyperparameters and paths shared by every entry point (train.py, play.py,
# gui_play.py, progress.py). Values can be overridden via a .env file or
# environment variables without touching this file.
import os
import torch
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


# ============= DEVICE =============
USE_GPU = _env_bool("USE_GPU", True)
DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# ============= MCTS =============
SIMULATIONS_PER_MOVE = int(os.environ.get("SIMULATIONS_PER_MOVE", 100))
C_init = 2
DIRICHLET_NOISE = 0.3  # alpha for Dirichlet noise
DIRICHLET_EPSILON = 0.25
MAX_GAME_MOVES = int(os.environ.get("MAX_GAME_MOVES", 150))  # limit the amount of moves played in a game

# ============= NEURAL NETWORK INPUTS =============
# 2 players, 6 pieces, 8x8 board
n = 8  # board size
# non boolean values: pieces for every player + the square for en passant
# boolean values: side to move, castling rights for every side and every player, is repitition
amount_of_input_planes = (2 * 6 + 1) + (1 + 4 + 1)
INPUT_SHAPE = (n, n, amount_of_input_planes)

# ============= NEURAL NETWORK OUTPUTS =============
# the model has 2 outputs: policy and value
# ouput_shape[0] should be the number of possible moves
#       * 8x8 board: 8*8=64 possible actions
#       * 56 possible queen-like moves (horizontal/vertical/diagonal)
#       * 8 possible knight moves (every direction)
#       * 9 possible underpromotions
#   total values: 8*8*(56+8+9) = 4672
# ouput_shape[1] should be 1: a scalar value (v)
queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
amount_of_planes = queen_planes + knight_planes + underpromotion_planes
OUTPUT_SHAPE = (8 * 8 * amount_of_planes, 1)

# ============= NEURAL NETWORK ARCHITECTURE =============
# Defaults are kept small so a full self-play/train iteration finishes in
# minutes on a single machine. Bump these up via env vars if you have a
# beefier GPU and want AlphaZero-scale strength.
WEIGHT_DECAY = 1e-4  # L2 regularization parameter
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-3))
CONVOLUTION_FILTERS = int(os.environ.get("CONVOLUTION_FILTERS", 64))
AMOUNT_OF_RESIDUAL_BLOCKS = int(os.environ.get("RESIDUAL_BLOCKS", 6))

# ============= PATHS =============
MODEL_FOLDER = os.environ.get("MODEL_FOLDER", "./models")
MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
LOG_DIR = os.environ.get("LOG_FOLDER", "./logs")
BEST_MODEL_PATH = os.path.join(MODEL_FOLDER, "best.pt")
TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")

# ============= TRAINING PARAMETERS =============
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))  # should be high if using GPU
N_EPOCHS_PER_ITERATION = int(os.environ.get("N_EPOCHS_PER_ITERATION", 5))
N_SELFPLAY_GAMES = int(os.environ.get("N_SELFPLAY_GAMES", 20))  # self-play games per training iteration
EVALUATION_GAMES = int(os.environ.get("EVALUATION_GAMES", 10))  # games played to compare candidate vs best
WIN_RATE_THRESHOLD = float(os.environ.get("WIN_RATE_THRESHOLD", 0.55))  # candidate must win at least this rate to be promoted

# ============= MEMORY CONFIGURATION =============
MAX_REPLAY_MEMORY = int(os.environ.get("MAX_REPLAY_MEMORY", 200000))

# ============= MULTIPROCESSING =============
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 1))
