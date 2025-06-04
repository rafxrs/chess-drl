# Chess Deep-RL Training Pipeline

### First Time Setup
1. Create directories for models, memory, and plots:
```
mkdir -p models memory plots
```
2. Create an initial model:
```
python code/train.py --epochs 0 --generate
```
This will create an initial random model and generate some self-play data.

### Local Training Workflow
Step 1: Generate Self-Play Data
```
python code/generate_data.py --model models/initial_model.pt --games 50 --output memory/selfplay_data.npz
```
Adjust --games based on your CPU and time constraints
This step is CPU-intensive and can run for hours with many games

Step 2: Train the Model
```
python code/train.py --model models/initial_model.pt --data memory/selfplay_data.npz --epochs 10
```
This will train for 10 epochs on the generated data
Add --visualize to see games during training

Step 3: Evaluate the Model
```
python code/evaluate.py --model1 models/initial_model.pt --model2 models/model_epoch_10.pt --games 20
```
This compares the new model against the initial model
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
For Colab, we want to generate data and then train on Colab's GPU:

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
