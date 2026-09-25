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

## Usage

| Command | What it does |
| --- | --- |
| `python code/train.py` | Train the bot through self-play (runs until you stop it) |
| `python code/play.py` | Play against the bot in the terminal |
| `python code/progress.py --watch` | Graph loss and win rate as training runs |
| `python code/gui_play.py --model models/model_iter_5.pt` | Play a specific version of the bot on a graphical board |

**Training.** Stop at any time with Ctrl+C. Progress is saved after every iteration, so running the command again resumes where it left off. Add `--fresh` to start over, or `--iterations N` to stop after N iterations.

**Playing.** `play.py` and `gui_play.py` use `models/best.pt` unless you pass `--model`. Both accept `--color white|black|random` and `--simulations N`, where more simulations make the bot stronger but slower. In the terminal, enter moves in UCI format (`e2e4`), or type `moves` to list the legal ones.

**Progress graph.** Without a display (e.g. on a remote server), `progress.py` saves the graph to `logs/training_log.png` instead of opening a window.

## How it works

Each iteration of `train.py`:

1. **Self-play:** the best model so far plays games against itself.
2. **Train:** the training model keeps learning from those games, predicting which moves were chosen (policy) and who won (value).
3. **Evaluate:** it plays a match against the current best model.
4. **Promote:** if it scores at least 55% (a draw counts as half a win), it becomes the new best and plays the next round of self-play.

Moves are chosen with Monte Carlo Tree Search, which the network guides: the policy suggests promising moves to explore, and the value estimates who is winning without playing the game out.

Checkpoints are kept in `models/`. `best.pt` is the strongest model so far, `latest.pt` is the model still being trained, and each promoted version is also saved as `model_iter_<N>.pt`, so you can play against earlier versions.

## Configuration

Defaults are in `code/config.py`. Override any of them with environment variables or a `.env` file in the repository root:

```bash
SIMULATIONS_PER_MOVE=200 RESIDUAL_BLOCKS=10 python code/train.py
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIMULATIONS_PER_MOVE` | 100 | MCTS simulations per move |
| `RESIDUAL_BLOCKS` / `CONVOLUTION_FILTERS` | 6 / 64 | Network size |
| `N_SELFPLAY_GAMES` | 20 | Self-play games per iteration |
| `EVALUATION_GAMES` | 10 | Games in each candidate vs. best match |
| `WIN_RATE_THRESHOLD` | 0.55 | Score needed to promote a candidate (draw = half a win) |
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
├── evaluate.py    # plays two models against each other
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
