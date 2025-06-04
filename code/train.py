import torch
import torch.optim as optim
import numpy as np
import random
import logging
import chess
import config
import multiprocessing as mp
import time
import os
import torch.multiprocessing as torch_mp
from generate_data import generate_selfplay_data_parallel
from utils import move_to_index
from functools import partial
from collections import deque
from agent import Agent
from modelbuilder import RLModelBuilder
from evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format=" %(message)s")

# Use config file for all hyperparameters
BATCH_SIZE = config.BATCH_SIZE
REPLAY_MEMORY_SIZE = config.MAX_REPLAY_MEMORY
N_SELFPLAY_GAMES = config.N_SELFPLAY_GAMES
N_EPOCHS = config.N_EPOCHS
N_SIMULATIONS = config.SIMULATIONS_PER_MOVE
LEARNING_RATE = config.LEARNING_RATE

# Replay buffer
replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)

class Trainer:
    def __init__(self, model, device):
        """
        Initialize the trainer with a model and device.
        
        Args:
            model: PyTorch neural network model
            device: Device to train on (CPU/GPU)
        """
        self.model = model
        self.device = device
        self.batch_size = config.BATCH_SIZE
    
    def sample_batch(self, replay_buffer):
        """Sample a random batch from the replay buffer."""
        if self.batch_size > len(replay_buffer):
            return list(replay_buffer)
        else:
            return random.sample(replay_buffer, self.batch_size)
    
    def train_batch(self, states, policies, values, optimizer):
        """
        Train the model on a single batch.
        
        Args:
            states: List of chess.Board states
            policies: List of policy vectors
            values: List of value targets
            optimizer: PyTorch optimizer
            
        Returns:
            Dictionary of loss values
        """
        # Convert to tensors
        state_tensors = []
        for state in states:
            # Convert state to tensor format that model expects
            s_tensor = Agent.state_to_tensor(state)
            state_tensors.append(s_tensor)
        
        states_batch = torch.cat(state_tensors).to(self.device)
        policies_batch = torch.FloatTensor(policies).to(self.device)
        values_batch = torch.FloatTensor(values).to(self.device).unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_preds = self.model(states_batch)
        
        # Calculate losses
        policy_loss = -(policies_batch * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        value_loss = ((values_batch - value_preds) ** 2).mean()
        
        # Total loss
        loss = policy_loss + value_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def train_random_batches(self, replay_buffer, optimizer, n_batches=None):
        """
        Train the model on random batches from the replay buffer.
        
        Args:
            replay_buffer: List of (state, policy, value) tuples
            optimizer: PyTorch optimizer
            n_batches: Number of batches to train on (default: 2*max(5, len(replay_buffer)//batch_size))
            
        Returns:
            List of loss dictionaries
        """
        if n_batches is None:
            n_batches = 2 * max(5, len(replay_buffer) // self.batch_size)
        
        history = []
        
        try:
            from tqdm import tqdm
            batch_iter = tqdm(range(n_batches), desc="Training batches")
        except ImportError:
            batch_iter = range(n_batches)
        
        for _ in batch_iter:
            # Sample random batch
            batch = self.sample_batch(replay_buffer)
            states, policies, values = zip(*batch)
            
            # Train on batch
            losses = self.train_batch(states, policies, values, optimizer)
            history.append(losses)
        
        return history
    
    def plot_loss(self, history, save_path=None):
        """
        Plot training loss history.
        
        Args:
            history: List of loss dictionaries
            save_path: Path to save the plot (default: config.LOSS_PLOTS_FOLDER)
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        df = pd.DataFrame(history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['loss'], label='Total Loss')
        plt.plot(df['policy_loss'], label='Policy Loss')
        plt.plot(df['value_loss'], label='Value Loss')
        plt.legend()
        plt.title(f"Loss over time (Learning rate: {config.LEARNING_RATE})")
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        
        if save_path is None:
            save_path = os.path.join(config.LOSS_PLOTS_FOLDER, 
                                   f"loss-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Loss plot saved to {save_path}")
        plt.close()

    def save_model(self, path=None):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model (default: auto-generated with timestamp)
        """
        from datetime import datetime
        
        if path is None:
            os.makedirs(config.MODEL_FOLDER, exist_ok=True)
            path = os.path.join(config.MODEL_FOLDER, 
                              f"model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt")
        
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")
        return path

def generate_game(model_path, game_id):
    """Generate a self-play game for multiprocessing."""
    import chess
    import torch
    from agent import Agent
    
    agent = Agent(model_path=model_path)
    board = chess.Board()
    states, policies, values = [], [], []
    move_count = 0
    
    print(f"Game {game_id} starting")
    
    while not board.is_game_over() and move_count < config.MAX_GAME_MOVES:
        if move_count % 5 == 0:  # Log every 5 moves
            print(f"Game {game_id}: Move {move_count}")
        agent.state = board.fen()
        try:
            # Time the simulation step
            sim_start = time.time()
            agent.run_simulations(config.SIMULATIONS_PER_MOVE)
            sim_time = time.time() - sim_start
            if sim_time > 10:  # Log if simulations take too long
                print(f"Game {game_id}: Simulations took {sim_time:.1f}s")
            
            # Get move probabilities from MCTS
            actions, probs = agent.mcts.get_move_probs()
            
            # Filter only legal moves
            legal_actions = []
            legal_probs = []
            for action, prob in zip(actions, probs):
                if action in board.legal_moves:
                    legal_actions.append(action)
                    legal_probs.append(prob)
            
            # Renormalize probabilities if needed
            if legal_actions and sum(legal_probs) > 0:
                legal_probs = np.array(legal_probs) / sum(legal_probs)
                # Sample a move proportionally to the probabilities
                move = np.random.choice(legal_actions, p=legal_probs)
            else:
                # Fallback to a random legal move if no legal moves in the MCTS results
                print(f"Game {game_id}: No legal moves from MCTS, using random move")
                move = np.random.choice(list(board.legal_moves))
            
            # Store state and policy
            states.append(board.copy())
            
            # Create policy vector (one-hot at the selected move indices)
            policy = np.zeros(config.OUTPUT_SHAPE[0], dtype=np.float32)
            for i, a in enumerate(actions):
                idx = move_to_index(a)
                if idx < len(policy):  # Safety check
                    policy[idx] = probs[i]
            
            policies.append(policy)
            
            # Make the move
            board.push(move)
            move_count += 1
            
        except Exception as e:
            print(f"Game {game_id} error: {e}")
            # Save the board state for debugging
            with open(f"error_board_game_{game_id}.fen", "w") as f:
                f.write(board.fen())
            break
    
    # Game is over, assign values
    result = board.result()
    print(f"Game {game_id} finished after {move_count} moves. Result: {result}")
    
    # Calculate game outcome
    if result == '1-0':  # White win
        z = 1.0
    elif result == '0-1':  # Black win
        z = -1.0
    else:  # Draw
        z = 0.0
    
    # Set value for each state (alternating perspectives)
    values = []
    for i in range(len(states)):
        # Flip the perspective for black's moves
        values.append(z if i % 2 == 0 else -z)
    
    return states, policies, values

def load_selfplay_data(data_path):
    """Load pre-generated self-play data from a file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    logging.info(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    
    # Convert FENs back to board objects
    states = [chess.Board(fen) for fen in data['states']]
    policies = data['policies']
    values = data['values']
    
    # Return as tuple for direct loading
    return states, policies, values

def visualize_selfplay_game(model_path):
    """Visualize a self-play game using GUI."""
    try:
        # Import self_play function from selfplay.py
        from selfplay import setup, self_play
        
        # Start GUI in a separate process to avoid blocking training
        gui_process = mp.Process(target=lambda: self_play(setup(model_path=model_path)))
        gui_process.start()
        return gui_process
    except ImportError as e:
        logging.warning(f"Could not visualize self-play game: {e}")
        return None

def run_gui_game(model_path):
    """Run a game with GUI visualization."""
    from gui.display import GUI
    from env import Chess_Env
    from game import Game
    
    env = Chess_Env()
    white_agent = Agent(model_path=model_path)
    black_agent = Agent(model_path=model_path)
    game = Game(env, white_agent, black_agent)
    
    game.reset()
    gui = GUI(game, player_is_white=True)
    gui.start()
    
    while not game.is_over():
        if game.current_player_is_white():
            move = white_agent.get_move(game.env)
        else:
            move = black_agent.get_move(game.env)
        game.push(move)
        gui.draw()
        time.sleep(0.5)  # Small delay to make moves visible
    
    time.sleep(5)  # Wait to show final position

def batchify(states, device):
    """
    Convert a list of board states to a batch of tensors suitable for the neural network.
    
    Args:
        states: List of chess.Board objects
        device: The device (CPU/GPU) to place tensors on
        
    Returns:
        torch.Tensor: Batch of state representations
    """
    import torch
    batch = []
    for state in states:
        # Convert state to tensor format using Agent's method
        state_tensor = Agent.state_to_tensor(state)
        batch.append(state_tensor)
    
    # Concatenate all tensors into a batch
    return torch.cat(batch).to(device)

def train_network(model, states, policies, values, optimizer, device):
    """
    Train the neural network on a batch of examples.
    
    Args:
        model: The neural network model
        states: List of board states
        policies: List of move probabilities (policy targets)
        values: List of game outcomes (value targets)
        optimizer: The optimizer
        device: The device to use (CPU/GPU)
    
    Returns:
        policy_loss, value_loss, total_loss
    """
    batch_start_time = time.time()
    
    # Convert to tensors
    state_tensors = []
    for state in states:
        # Convert state to tensor format that model expects
        s_tensor = Agent.state_to_tensor(state)
        state_tensors.append(s_tensor)
    
    states_batch = torch.cat(state_tensors).to(device)
    policies_batch = torch.FloatTensor(policies).to(device)
    values_batch = torch.FloatTensor(values).to(device).unsqueeze(1)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    policy_logits, value_preds = model(states_batch)
    
    # Calculate losses
    policy_loss = -(policies_batch * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = ((values_batch - value_preds) ** 2).mean()
    
    # Total loss
    loss = policy_loss + value_loss
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    batch_time = time.time() - batch_start_time
    
    return policy_loss.item(), value_loss.item(), loss.item(), batch_time

def evaluate_model(current_model_path, previous_model_path=None, n_evaluation_games=10):
    """Evaluate current model against previous version to measure improvement."""
    if previous_model_path is None or not os.path.exists(previous_model_path):
        logging.info(f"No previous model to compare against")
        return True, 0.0  # Accept new model by default
    
    try:
        logging.info(f"Evaluating current model against previous version...")
        evaluator = Evaluator(current_model_path, previous_model_path)
        results = evaluator.evaluate(n_games=n_evaluation_games, verbose=True)
        
        # Extract key metrics
        win_rate = results.get('win_rate', 0.0)
        draw_rate = results.get('draw_rate', 0.0)
        loss_rate = results.get('loss_rate', 0.0)
        elo_diff = results.get('elo_difference', 0.0)
        
        # Decision rule: accept if win_rate > 52% or elo_diff > 0
        accept_new_model = win_rate > 0.52 or elo_diff > 0
        
        if accept_new_model:
            logging.info(f"NEW MODEL ACCEPTED: Win rate: {win_rate:.1%}, ELO difference: {elo_diff:.1f}")
        else:
            logging.info(f"NEW MODEL REJECTED: Win rate: {win_rate:.1%}, ELO difference: {elo_diff:.1f}")
        
        return accept_new_model, elo_diff
    
    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        return True, 0.0  # Accept by default in case of error

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train chess model using reinforcement learning")
    parser.add_argument("--model", type=str, help="Path to model for continued training")
    parser.add_argument("--data", type=str, default="./memory/selfplay_data.npz",
                       help="Path to pre-generated self-play data")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument("--visualize", action="store_true", help="Visualize self-play games")
    parser.add_argument("--generate", action="store_true", 
                       help="Generate additional data during training (default: False)")
    args = parser.parse_args()
    
    # Override config if provided
    epochs = args.epochs
    model_path = args.model if args.model else os.path.join(config.MODEL_FOLDER, "initial_model.pt")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize model
    model = RLModelBuilder(
        config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
    ).build_model(model_path)
    model.to(device)
    
    # Initialize optimizer
    weight_decay = getattr(config, 'WEIGHT_DECAY', 1e-4)  # Default if not in config
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    
    # Initialize trainer
    trainer = Trainer(model, device)
    
    # Load pre-generated data
    if os.path.exists(args.data):
        logging.info(f"Loading pre-generated data from {args.data}")
        states, policies, values = load_selfplay_data(args.data)
        
        # Add to replay buffer
        for s, p, v in zip(states, policies, values):
            replay_buffer.append((s, p, v))
        
        logging.info(f"Loaded {len(replay_buffer)} positions into replay buffer")
    
    # Generate initial data if needed
    if len(replay_buffer) < BATCH_SIZE and args.generate:
        logging.info("Generating initial self-play data...")
        generate_selfplay_data_parallel(model_path, n_games=N_SELFPLAY_GAMES)
    elif len(replay_buffer) < BATCH_SIZE:
        raise ValueError("Not enough training data and --generate not specified")
    
    # Training loop
    best_model_path = model_path
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Visualize a self-play game occasionally
        if args.visualize and epoch % 5 == 0:
            gui_process = visualize_selfplay_game(model_path)
        
        # Training phase
        model.train()
        logging.info(f"Training on {len(replay_buffer)} positions...")
        history = trainer.train_random_batches(replay_buffer, optimizer)
        
        # Plot losses
        trainer.plot_loss(history)
        
        # Save current model
        current_model_path = os.path.join(config.MODEL_FOLDER, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), current_model_path)
        logging.info(f"Model saved to {current_model_path}")
        
        # Evaluate against previous best model
        if epoch > 0 and config.EVALUATION_GAMES > 0:
            is_better, elo_gain = evaluate_model(
                current_model_path, 
                best_model_path, 
                n_evaluation_games=config.EVALUATION_GAMES
            )
            
            if is_better:
                best_model_path = current_model_path
                logging.info(f"New best model! ELO gain: {elo_gain:.1f}")
        
        # Generate new self-play data with current model (if requested)
        if args.generate:
            logging.info("Generating new self-play data...")
            generate_selfplay_data_parallel(current_model_path, n_games=N_SELFPLAY_GAMES)
        
        # Log epoch time
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_FOLDER, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"\nTraining completed! Final model saved to {final_model_path}")
    
    # Copy the best model to the final model if we did evaluations
    if config.EVALUATION_GAMES > 0 and best_model_path != final_model_path:
        import shutil
        shutil.copyfile(best_model_path, final_model_path)
        logging.info(f"Best model from training ({os.path.basename(best_model_path)}) copied to {final_model_path}")

if __name__ == "__main__":
    main()