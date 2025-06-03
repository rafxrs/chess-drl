# Chess Deep Reinforcement Learning
A deep reinforcement learning implementation for chess inspired by AlphaZero methodology, featuring Monte Carlo Tree Search (MCTS) guided by a neural network.

## Overview
This project implements a chess AI that learns to play chess through self-play without any human knowledge beyond the rules of the game. The implementation closely follows the AlphaZero approach with:

- A deep neural network with policy and value heads
- Monte Carlo Tree Search (MCTS) guided by the neural network
- Self-play training methodology
- Reinforcement learning without human demonstrations

## Features
- **Neural Network**: Residual CNN architecture with policy and value heads
- **MCTS**: Efficient Monte Carlo Tree Search implementation with UCB exploration
- **Self-play**: Automatic generation of training data through self-play
- **GPU Acceleration**: Support for GPU training for faster learning
- **Multiprocessing**: Parallel game generation for increased training throughput
- **Interactive Play**: Play against the trained model with a visual interface
- **Comprehensive Testing**: Test suite to validate components and monitor performance

## Requirements
**Core dependencies**
```
chess==1.9.4
numpy==1.24.3
torch==2.0.1
python-dotenv==1.0.0
tqdm==4.65.0
```

**Visualization and UI**
```
pygame==2.5.0
pygame-popup==0.9.1
```
**Utilities**
```
setproctitle==1.3.2
matplotlib==3.7.1
```
**Performance monitoring**
```
tensorboard==2.13.0
psutil==5.9.5
```
**Development tools**
```
pytest==7.3.1
black==23.3.0
pylint==2.17.4
pynvml==12.0.0
```
## Project Structure
```
code/
├── agent.py              # Agent implementation with MCTS
├── config.py             # Configuration and hyperparameters
├── env.py                # Chess environment wrapper
├── evaluate.py           # Model evaluation tools
├── game.py               # Game simulation utilities
├── initmodel.py          # Model initialization
├── main.py               # Main entry point
├── modelbuilder.py       # Neural network architecture
├── selfplay.py           # Model plays n games against itself
├── test.py               # Test suite
├── train.py              # Training pipeline
├── utils.py              # Utility functions
├── gui/                  # GUI components for visualization
│   ├── board.py          # Chess board visualization
│   ├── display.py        # Display management
│   ├── pieces.py         # Chess piece rendering
│   └── imgs/             # Images for chess pieces
└── mcts/                 # Monte Carlo Tree Search implementation
    ├── edge.py           # Edge representation in search tree
    ├── mcts.py           # Main MCTS algorithm
    └── node.py           # Node representation in search tree
```
## How to Use
### Installation
1. Clone the repository:
```
git clone https://github.com/rafxrs/chess-drl.git
cd chess-drl
```
2. Install dependencies:
```
pip install -r code/requirements.txt
```
3. Set up environment variables (optional):
```
# Create a .env file to customize parameters
echo "SIMULATIONS_PER_MOVE=800" > .env
```
### Training
1. Initialize a new model:
```
python code/initmodel.py
```
2. Start training:
```
python code/train.py
```

Training parameters can be modified in `config.py` or overridden with environment variables.

### Testing
Run the comprehensive test suite to validate your setup:

`python code/test.py`

This will:

- Test all components of the system
- Generate performance metrics
- Create visualizations of resource usage
- Produce an HTML report with recommendations

### Playing Against the Model
Play against the trained model with a visual interface:

```python code/play.py --model models/model_final.pt```

**Evaluation**

Evaluate model performance:
`
python code/evaluate.py --model models/model_final.pt --games 100
`

**Training Process**

The training process follows these steps:

1. **Self-play**: The model plays against itself to generate training data
2. **Data Collection**: Game states, policies (move probabilities), and outcomes are stored
3. **Training**: The model is trained to predict both move probabilities and game outcomes
4. **Iteration**: The improved model generates better self-play data for further training

Training progress is tracked with visualizations and logs:

- Loss curves
- Resource usage metrics
- Game outcome statistics
- Model strength progression

## Configuration
Key hyperparameters in `config.py`:

- SIMULATIONS_PER_MOVE: Number of MCTS simulations per move (default: 400)
- BATCH_SIZE: Batch size for training (default: 128)
- LEARNING_RATE: Learning rate for optimization (default: 0.001)
- N_EPOCHS: Number of training epochs (default: 20)
- N_SELFPLAY_GAMES: Number of self-play games per training iteration (default: 10)
- NUM_WORKERS: Number of parallel processes for self-play (default: CPU count)
- USE_GPU: Whether to use GPU acceleration (default: True)

## Performance Tips
For optimal performance:

1. GPU Training: Use a CUDA-compatible GPU for significantly faster training
2. Multiprocessing: Set NUM_WORKERS to utilize all available CPU cores
3. Batch Size: Use larger batch sizes (512-1024) with GPU for faster training
4. Memory Usage: Monitor memory usage with the testing tools if running out of RAM
5. Simulation Count: Increase SIMULATIONS_PER_MOVE for stronger play (at the cost of speed)

## Model Architecture
The neural network architecture follows AlphaZero's design:

1. **Input**: 19-plane 8×8 representation of the chess position
2. **Body**: Residual convolutional network with batch normalization
3. **Policy Head**: Outputs move probabilities for all possible moves
4. **Value Head**: Predicts the game outcome from the current position

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## Acknowledgments
This project is inspired by DeepMind's AlphaZero and builds upon several open-source implementations in the reinforcement learning community.

## References
- Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.
- pip install -r code/requirements.txtSilver, D. et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.