# Chess Deep Reinforcement Learning

A self-play chess engine inspired by AlphaZero: a residual CNN with policy and
value heads, guided by Monte Carlo Tree Search (MCTS), that learns to play
chess purely by playing against itself.

## Overview

`train.py` runs a continuous loop:

1. **Self-play** — the current best model plays games against itself.
2. **Train** — a candidate model is trained on the accumulated self-play positions.
3. **Evaluate** — the candidate plays a match against the current best model.
4. **Promote** — if the candidate wins often enough, it becomes the new best model.

...and repeats, indefinitely, getting stronger over time. Everything else
(playing against it, watching it learn) is a thin layer on top of that loop.

## Installation

```
git clone https://github.com/rafxrs/chess-drl.git
cd chess-drl
pip install -r requirements.txt
```

## The four commands

### 1. Train

```
python code/train.py
```

Starts (or resumes) the self-play/train/evaluate loop and runs it forever.
Stop it any time with Ctrl+C — the model, replay buffer, and training log are
all saved after each completed iteration, so re-running the same command
picks up where it left off. Useful flags:

- `--iterations N` — stop after N iterations instead of running forever.
- `--games-per-iteration N` — self-play games generated per iteration (default from `config.py`).
- `--eval-games N` — games played to decide whether to promote a candidate.
- `--simulations N` — MCTS simulations per move.
- `--fresh` — start over from a newly initialized model instead of resuming.

Checkpoints are written to `models/`: `best.pt` always holds the current
strongest model, and each promoted iteration is additionally kept as
`model_iter_<N>.pt` so you can play against any earlier version later.

### 2. Play against the bot (terminal)

```
python code/play.py
```

Plays a game against `models/best.pt` in the terminal using UCI move
notation (e.g. `e2e4`). Use `--model` to pick a different checkpoint,
`--color white|black|random` to choose sides, and `--simulations` to trade
off strength for speed.

### 3. Watch it learn

```
python code/progress.py
```

Plots training loss and the win rate of each candidate against the previous
best model, iteration by iteration. Add `--watch` to keep it refreshing
while `train.py` runs in another terminal. When no display is available
(e.g. a headless server) it saves the plot as a PNG next to the training log
instead of opening a window.

### 4. Play against a specific version (GUI)

```
python code/gui_play.py --model models/model_iter_5.pt
```

Opens a graphical chess board (pygame) and lets you play against whichever
checkpoint you point it at — handy for feeling the difference between an
early and a late version of the bot. Same `--color` and `--simulations`
flags as `play.py`.

## Configuration

Key hyperparameters live in `code/config.py` and can be overridden via
environment variables or a `.env` file, for example:

- `SIMULATIONS_PER_MOVE` — MCTS simulations per move (default: 100)
- `RESIDUAL_BLOCKS` / `CONVOLUTION_FILTERS` — network size (defaults: 6 / 64,
  intentionally small so a full iteration finishes in a reasonable time on a
  single machine; raise these if you have serious compute)
- `N_SELFPLAY_GAMES` — self-play games per training iteration (default: 20)
- `EVALUATION_GAMES` — games played to evaluate each candidate (default: 10)
- `WIN_RATE_THRESHOLD` — win rate a candidate needs to be promoted (default: 0.55)
- `NUM_WORKERS` — parallel self-play worker processes (default: CPU count)
- `USE_GPU` — whether to use CUDA when available (default: true)

## Project structure

```
code/
├── config.py        # hyperparameters and paths
├── env.py           # chess board wrapper (Chess_Env)
├── model.py         # residual CNN with policy/value heads
├── mcts.py          # Monte Carlo Tree Search (Node, Edge, MCTS)
├── agent.py         # ties the model + MCTS together to pick moves
├── selfplay.py       # multiprocess self-play data generation
├── evaluate.py       # plays two checkpoints against each other
├── utils.py          # move <-> policy-index encoding
├── train.py          # command 1: the self-play/train/evaluate loop
├── play.py           # command 2: terminal play against the bot
├── progress.py        # command 3: loss/win-rate graph
├── gui_play.py        # command 4: GUI play against a specific checkpoint
└── gui/              # pygame board rendering used by gui_play.py
```

## Model architecture

- **Input**: 19-plane 8x8 encoding of the position (pieces from the mover's
  perspective, en passant, side to move, castling rights, halfmove clock)
- **Body**: residual convolutional network with batch normalization
- **Policy head**: logits over all 4672 possible moves (AlphaZero move encoding)
- **Value head**: scalar in [-1, 1] predicting the game outcome

## License

MIT License

## References

- Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.
- Silver, D. et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.
