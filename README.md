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

## Project Structure
```
code/
├── agent.py              # Agent implementation with MCTS
├── config.py             # Configuration and hyperparameters
├── edge.py               # Edge representation in search tree
├── env.py                # Chess environment wrapper
├── evaluate.py           # Model evaluation tools
├── game.py               # Game simulation utilities
├── generate_data.py      # Self-play data generation
├── guide.md              # Step-by-step tutorial
├── mcts.py               # Main MCTS algorithm
├── modelbuilder.py       # Neural network architecture
├── node.py               # Node representation in search tree
├── selfplay.py           # Visualize model playing chess
├── test.py               # Test suite
├── train.py              # Training pipeline
├── utils.py              # Utility functions
├── chess-drl.ipynb       # Colab notebook for cloud training
└── gui/                  # GUI components for visualization
    ├── board.py          # Chess board visualization
    ├── display.py        # Display management
    ├── pieces.py         # Chess piece rendering
    └── imgs/             # Images for chess pieces
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

## Training Process 

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

# Training Pipeline

## Testing
Run the comprehensive test suite to validate the setup:

`python code/test.py`

This will:

- Test all components of the system
- Generate performance metrics
- Create visualizations of resource usage
- Produce an HTML report with recommendations

### Local Training Workflow
Step 1: Generate Self-Play Data
```
python code/generate_data.py --model models/initial_model.pt --games 50 --output memory/selfplay_data.npz
```
Adjust --games based on your CPU and time constraints.
This step is CPU-intensive and can run for hours with many games

Step 2: Train the Model
```
python code/train.py --model models/initial_model.pt --data memory/selfplay_data.npz --epochs 10
```
This will train for 10 epochs on the generated data. 
Add the --visualize flag to see games during training

Step 3: Evaluate the Model
```
python code/evaluate.py --model1 models/initial_model.pt --model2 models/model_epoch_10.pt --games 20
```
This compares the new model against the initial model.
Use --games to control how many games to play for evaluation

Step 4: Visualize Gameplay
```
python code/selfplay.py --model_path models/model_epoch_10.pt --n_games 1
```
This will visualize a game played by the trained model

### Iterative Improvement
After the initial training cycle, we can generate more data with the improved model:
```
# Generate more data with improved model
python code/generate_data.py --model models/model_epoch_10.pt --games 50 --output memory/selfplay_data_2.npz

# Train further using the new data
python code/train.py --model models/model_epoch_10.pt --data memory/selfplay_data_2.npz --epochs 10
```

### Google Colab Workflow
For Colab, we want to generate data and then train on Colab's GPU.
There is also a notebook containing the workflow below in the repository. 

Step 1: Set Up Colab
```
# Mount Google Drive to save models between sessions (optional)
from google.colab import drive
drive.mount('/content/drive')

# Clone from GitHub
# !git clone https://<USERNAME>:<TOKEN>@github.com/<USERNAME>/<REPO_NAME>.git
# %cd chess-drl

# Install dependencies
!pip install -r "code/requirements.txt"

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```
Step 2: Data Generation
```
# Generate lots of training data using the gpu
python code/generate_data.py --model models/model_latest.pt --games 100 --output memory/selfplay_data.npz --gpu
```
Step 3: Training
```
# Run training
!python code/train.py --model models/initial_model.pt --data memory/colab_data.npz --epochs 20

# Show the training loss plot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
latest_plot = sorted(glob('plots/*.png'))[-1]
img = mpimg.imread(latest_plot)
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

# Download trained model
from google.colab import files
files.download('models/model_final.pt')
```
Step 4: Continue Locally
Once we download the model from Colab, we can generate more data with it locally:
```
python code/generate_data.py --model models/model_final.pt --games 100 --output memory/colab_data_2.npz
```
Then repeat the Colab training.

### Tips for Efficient Training
1. For self-play data generation:
- Use local machine with multiprocessing
- Leave it running overnight for large datasets
- Use --append to add to existing data files
2. For model training:
- Use Colab's GPU for faster training
- Try different learning rates in config.py
- Plot the losses to diagnose training issues
3. For evaluation:
- Compare models often to track progress
- Use at least 20-50 games for statistically significant results
- Track ELO differences over time

This workflow separates the CPU-intensive data generation from the GPU-accelerated training, making optimal use of your resources.

### GPU-Accelerated Data Generation

To speed up data generation using a GPU:

```
python code/generate_data.py --model models/model_latest.pt --games 100 --output memory/selfplay_data.npz --gpu
```
For best performance:

- Make sure your model architecture is optimized for GPU inference
- Use appropriate batch sizes for your GPU memory
- Consider using mixed-precision inference for even faster generation

## Performance Comparison

On a decent GPU (like an RTX 3080), you can expect:

- **CPU-only**: ~10-50 positions per second
- **GPU-accelerated**: ~500-2000+ positions per second

The speedup will be most noticeable when:
1. Running many MCTS simulations per move
2. Using a larger neural network model
3. Generating many games in parallel

The CPU will still handle the game logic and MCTS tree operations, but the most computationally intensive part (neural network inference) will run on the GPU, resulting in a significant overall speedup.



### Playing Against the Model
Play against the trained model with a visual interface:

```python code/play.py --model models/model_final.pt```

**Evaluation**

Evaluate model performance:
`
python code/evaluate.py --model models/model_final.pt --games 100
`

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
- Silver, D. et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.
