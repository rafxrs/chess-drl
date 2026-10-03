# Chess Deep Reinforcement Learning

A chess engine that teaches itself to play, inspired by AlphaZero. It starts with no chess knowledge beyond the rules and gets stronger by playing against itself.

## Setup

Requires Python 3.9+. A CUDA GPU is optional; CPU works, just slower.

```bash
git clone https://github.com/rafxrs/chess-drl.git
cd chess-drl
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run every command from the repository root. Models, data and logs are written to `models/`, `memory/` and `logs/` there.

### GPU setup (NVIDIA)

On Windows, `pip install torch` installs a CPU-only build, and `train.py` then prints `Using device: cpu`. To use an NVIDIA GPU:

1. Run `nvidia-smi` and note the `CUDA Version` in the top-right corner. That's the newest CUDA your driver supports.
2. Replace PyTorch with a CUDA build no newer than that version. Get the exact command from the selector at [pytorch.org/get-started](https://pytorch.org/get-started/locally/); for example, for CUDA 12.6:
   ```bash
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
3. Check that PyTorch sees the GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

Each self-play worker loads its own copy of the model on the GPU. If you run out of GPU memory, lower `NUM_WORKERS` (for example `set NUM_WORKERS=4` on Windows).

## Usage

| Command | What it does |
| --- | --- |
| `python code/train.py` | Train the bot through self-play (runs until you stop it) |
| `python code/play.py` | Play against the bot in the terminal |
| `python code/progress.py --watch` | Graph how training is going, live |
| `python code/gui_play.py --model models/model_iter_5.pt` | Play a specific version of the bot on a graphical board |

**Training.** Stop at any time with Ctrl+C. Progress is saved after every iteration, so running the command again resumes where it left off. Add `--iterations N` to stop after N iterations. `--fresh` starts over with a new model; the previous run's models, data and logs are moved to `archive/<timestamp>/`, not deleted.

**Playing.** `play.py` and `gui_play.py` use `models/latest.pt` unless you pass `--model`. The bot always plays the move its search rates best. Both accept `--color white|black|random` and `--simulations N` (default 200); more simulations make the bot stronger but slower. In the terminal, enter moves in UCI format (`e2e4`), or type `moves` to list the legal ones.

**Progress graph.** The top panel shows the bot's score in benchmark matches, run every 10 iterations against a random mover and against its own checkpoint from 10 iterations earlier. Scoring above 50% against the earlier checkpoint means it's still getting stronger. Below that are training loss, how self-play games end (white win, black win, draw) and how long they last. Without a display (e.g. on a remote server), `progress.py` saves the graph to `logs/training_log.png` instead of opening a window.

## How it works

Each iteration of `train.py`:

1. **Self-play:** the latest model plays games against itself. The first 30 plies (15 moves per side) are sampled in proportion to how much the search explored them, for variety; after that it plays its best move.
2. **Train:** the model learns from the accumulated games, predicting which moves the search preferred (policy) and who won (value).
3. **Repeat:** the updated model plays the next round of self-play.

Training uses the most recent 50,000 positions (about 20 iterations of games), so the network learns from games near its current level. The learning rate drops tenfold at iterations 100 and 300. Every 10 iterations, a short benchmark measures playing strength (see the progress graph).

As in AlphaZero, there's no match between versions to decide which model to keep: the newest model always plays the next games. Games still going at the move limit (150 plies) are scored on material: the side ahead by at least 3 pawns' worth wins, otherwise it's a draw. Early on, this gives the network a learning signal before it can actually checkmate.

Moves are chosen with Monte Carlo Tree Search, which the network guides: the policy suggests promising moves to explore, and the value estimates who is winning without playing the game out.

Checkpoints are kept in `models/`. `latest.pt` is the current model, and a copy is saved as `model_iter_<N>.pt` every 5 iterations, so you can play against earlier versions.

## Configuration

Defaults are in `code/config.py`. Override any of them with environment variables or a `.env` file in the repository root:

```bash
SIMULATIONS_PER_MOVE=200 RESIDUAL_BLOCKS=10 python code/train.py
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIMULATIONS_PER_MOVE` | 100 | MCTS simulations per move |
| `RESIDUAL_BLOCKS` / `CONVOLUTION_FILTERS` | 6 / 64 | Network size |
| `PLAY_SIMULATIONS` | 200 | Simulations per move when you play the bot |
| `N_SELFPLAY_GAMES` | 20 | Self-play games per iteration |
| `TEMPERATURE_MOVES` | 30 | Plies sampled for variety before playing the best move |
| `MAX_GAME_MOVES` / `ADJUDICATION_MARGIN` | 150 / 3 | Ply limit for self-play games, and material lead that wins a game stopped there |
| `CHECKPOINT_EVERY` | 5 | Iterations between saved `model_iter_<N>.pt` versions |
| `MAX_REPLAY_MEMORY` | 50000 | Most recent positions kept for training |
| `LR_MILESTONES` | 100,300 | Iterations where the learning rate drops tenfold |
| `BENCHMARK_EVERY` / `BENCHMARK_GAMES` | 10 / 10 | How often to benchmark, and games per opponent (0 turns it off) |
| `NUM_WORKERS` | CPU count | Parallel self-play processes |
| `USE_GPU` | true | Use CUDA when available |

The defaults are deliberately small so that one iteration takes minutes on an ordinary machine. With a strong GPU, increase the network size and the number of simulations.

## Project layout

```
code/
├── train.py       # command: self-play training loop
├── play.py        # command: terminal play
├── progress.py    # command: training graph
├── gui_play.py    # command: graphical play
├── config.py      # settings
├── agent.py       # picks moves with network + MCTS
├── mcts.py        # Monte Carlo Tree Search
├── model.py       # residual network with policy and value heads
├── selfplay.py    # parallel self-play game generation
├── benchmark.py   # strength matches vs a random mover and older checkpoints
├── env.py         # chess board wrapper
├── utils.py       # move encoding for the policy output
└── gui/           # pygame board rendering
```

Changing `RESIDUAL_BLOCKS` or `CONVOLUTION_FILTERS` makes existing checkpoints incompatible. Retrain with `--fresh`, or keep the settings that were used for training.

## References

- Silver et al. (2017), *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*
- Silver et al. (2018), *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*
- [python-chess documentation](https://python-chess.readthedocs.io/en/latest/)
- [zjeffer/chess-deep-rl](https://github.com/zjeffer/chess-deep-rl)

## License

MIT
