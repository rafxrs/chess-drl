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
from utils import move_to_index
from functools import partial
from collections import deque
from agent import Agent
from modelbuilder import RLModelBuilder

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

def generate_selfplay_data(agent, n_games=1):
    """
    Generate self-play games and store (state, policy, value) in the replay buffer.
    """
    import chess
    for _ in range(n_games):
        board = chess.Board()
        states, policies, values = [], [], []
        agent.state = board.fen()
        while not board.is_game_over():
            agent.state = board.fen()
            agent.run_simulations(N_SIMULATIONS)
            actions, probs = agent.mcts.get_move_probs()
            move = np.random.choice(actions, p=probs)
            # Save state and policy
            states.append(board.copy())
            policy = np.zeros(config.OUTPUT_SHAPE[0], dtype=np.float32)
            for i, a in enumerate(actions):
                # You must implement move_to_index for your move encoding
                idx = move_to_index(a)
                policy[idx] = probs[i]
            policies.append(policy)
            board.push(move)
        # Assign values (from the perspective of the player to move)
        result = board.result()
        if result == '1-0':
            z = 1
        elif result == '0-1':
            z = -1
        else:
            z = 0
        values = [z if i % 2 == 0 else -z for i in range(len(states))]
        for s, p, v in zip(states, policies, values):
            replay_buffer.append((s, p, v))

def batchify(batch):
    """
    Convert a batch of (state, policy, value) to tensors.
    """
    states, policies, values = zip(*batch)
    agent = Agent(model_path=None)  # For state_to_tensor
    state_tensors = torch.cat([agent.state_to_tensor(s).unsqueeze(0) for s in states], dim=0)
    policy_tensors = torch.tensor(np.stack(policies), dtype=torch.float32)
    value_tensors = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    return state_tensors, policy_tensors, value_tensors

def generate_game(model_path, game_id):
    """
    Generate a single self-play game in a separate process
    
    Args:
        model_path: Path to the neural network model
        game_id: Unique identifier for this game (used for logging)
        
    Returns:
        states, policies, values: Lists of game states, policy targets, and value targets
    """
    # Set process name for better monitoring
    try:
        import setproctitle
        setproctitle.setproctitle(f"chess-drl: game-{game_id}")
    except ImportError:
        pass
    
    # Initialize agent with the model
    agent = Agent(model_path=model_path)
    board = chess.Board()
    states, policies, values = [], [], []
    move_count = 0
    
    # Set up logging for this process
    game_logger = logging.getLogger(f"game-{game_id}")
    game_logger.setLevel(logging.INFO)
    
    # Play game until termination
    start_time = time.time()
    agent.state = board.fen()
    
    # Play until game is over or move limit reached
    while not board.is_game_over() and move_count < config.MAX_GAME_MOVES:
        agent.state = board.fen()
        agent.run_simulations(config.SIMULATIONS_PER_MOVE)
        
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
            move = np.random.choice(list(board.legal_moves))
        
        # Store state and policy
        states.append(board.copy())
        
        # Create policy vector (one-hot at the selected move indices)
        policy = np.zeros(config.OUTPUT_SHAPE[0], dtype=np.float32)
        for i, a in enumerate(actions):
            idx = move_to_index(a)
            policy[idx] = probs[i]
        
        policies.append(policy)
        
        # Make the move
        board.push(move)
        move_count += 1
        
        # Add some temperature decay as the game progresses
        if move_count == 30:  # After 30 moves, use lower temperature
            agent.mcts.exploration_weight *= 0.8
    
    # Get game result and assign values
    result = board.result()
    game_logger.info(f"Game {game_id} finished after {move_count} moves. Result: {result}. Time: {time.time() - start_time:.1f}s")
    
    if result == '1-0':
        z = 1
    elif result == '0-1':
        z = -1
    else:
        z = 0
    
    # Assign values alternating between players
    values = [z if i % 2 == 0 else -z for i in range(len(states))]
    
    game_logger.info(f"Game {game_id} generated {len(states)} training examples")
    
    # Return the collected data
    return states, policies, values

def generate_selfplay_data_parallel(model_path, n_games=10):
    """
    Generate self-play data using multiple processes
    
    Args:
        model_path: Path to the neural network model
        n_games: Number of games to generate in parallel
        
    Returns:
        None (adds data to the replay buffer)
    """
    # Determine number of processes to use
    num_processes = min(config.NUM_WORKERS, mp.cpu_count(), n_games)
    logging.info(f"Generating {n_games} self-play games using {num_processes} processes")
    
    start_time = time.time()
    
    # For PyTorch models, use torch's multiprocessing to handle CUDA properly
    if config.USE_GPU and torch.cuda.is_available():
        # Set start method to spawn for CUDA compatibility
        try:
            torch_mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        mp_context = torch_mp
    else:
        mp_context = mp
    
    # Create temporary model file for processes to load
    # This avoids sharing the model directly which can cause issues
    temp_model_path = os.path.join(config.MODEL_FOLDER, f"temp_model_{int(time.time())}.pt")
    if os.path.exists(model_path):
        # Load and immediately save the model to the temp path
        model = RLModelBuilder(
            config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
        ).build_model(model_path)
        torch.save(model.state_dict(), temp_model_path)
    
    try:
        # Use Pool for parallel processing
        with mp_context.Pool(num_processes) as pool:
            game_fn = partial(generate_game, temp_model_path)
            
            # Track progress with tqdm if available
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(game_fn, range(n_games)), total=n_games, desc="Self-play games"))
            except ImportError:
                results = pool.map(game_fn, range(n_games))
        
        # Collect results
        all_states, all_policies, all_values = [], [], []
        total_positions = 0
        
        for states, policies, values in results:
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            total_positions += len(states)
        
        # Add to replay buffer
        for s, p, v in zip(all_states, all_policies, all_values):
            replay_buffer.append((s, p, v))
        
        elapsed_time = time.time() - start_time
        positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
        
        logging.info(f"Generated {total_positions} positions from {n_games} games in {elapsed_time:.1f}s "
                    f"({positions_per_second:.1f} positions/s)")
        logging.info(f"Replay buffer now contains {len(replay_buffer)} examples")
        
    finally:
        # Clean up temporary model file
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except Exception as e:
                logging.warning(f"Failed to remove temporary model file: {e}")

# Integration with main training loop
def main():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
    logging.info(f"Using device: {device}")
    
    # Define initial model path
    initial_model_path = f"{config.MODEL_FOLDER}/initial_model.pt"
    
    # Build model and optimizer using config - ensure model is on GPU
    model = RLModelBuilder(
        config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
    ).build_model()
    model = model.to(device)  # Move model to GPU
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Save the initial model if it doesn't exist
    import os
    os.makedirs(config.MODEL_FOLDER, exist_ok=True)
    if not os.path.exists(initial_model_path):
        torch.save(model.state_dict(), initial_model_path)
        logging.info(f"Saved initial model to {initial_model_path}")

    agent = Agent(model_path=initial_model_path)
    agent.model = model  # Use the model directly

    # Generate initial self-play data with multiprocessing
    logging.info("Generating initial self-play data...")
    if config.NUM_WORKERS > 1:
        generate_selfplay_data_parallel(initial_model_path, n_games=N_SELFPLAY_GAMES)
    else:
        generate_selfplay_data(agent, n_games=N_SELFPLAY_GAMES)
    logging.info(f"Replay buffer size: {len(replay_buffer)}")

    # Training loop
    for epoch in range(N_EPOCHS):
        if len(replay_buffer) < BATCH_SIZE:
            logging.warning("Not enough samples in replay buffer to train.")
            continue
            
        # Sample batch and move to GPU
        batch = random.sample(replay_buffer, BATCH_SIZE)
        states, target_policies, target_values = batchify(batch)
        states = states.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)

        # Training step with mixed precision if available
        model.train()
        optimizer.zero_grad()
        
        # Use automatic mixed precision for faster training if available
        if hasattr(torch.cuda, 'amp') and device.type == 'cuda':
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            
            with autocast():
                policy_pred, value_pred = model(states)
                loss_policy = torch.nn.functional.cross_entropy(policy_pred, target_policies)
                loss_value = torch.nn.functional.mse_loss(value_pred, target_values)
                loss = loss_policy + loss_value
                
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training path
            policy_pred, value_pred = model(states)
            loss_policy = torch.nn.functional.cross_entropy(policy_pred, target_policies)
            loss_value = torch.nn.functional.mse_loss(value_pred, target_values)
            loss = loss_policy + loss_value
            
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.4f} | "
                    f"Policy Loss: {loss_policy.item():.4f} | Value Loss: {loss_value.item():.4f}")

        # Generate more self-play data periodically
        if (epoch + 1) % 10 == 0:
            logging.info("Generating more self-play data...")
            # Use model evaluation mode for inference
            model.eval()
            
            # Save a temporary model for multiprocessing
            temp_path = f"{config.MODEL_FOLDER}/model_epoch_{epoch+1}_temp.pt"
            torch.save(model.state_dict(), temp_path)
            
            # Generate data with multiprocessing
            if config.NUM_WORKERS > 1:
                generate_selfplay_data_parallel(temp_path, n_games=2)
            else:
                generate_selfplay_data(agent, n_games=2)
                
            # Remove temporary model
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            model.train()

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = f"{config.MODEL_FOLDER}/model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            logging.info(f"Model saved at {save_path}")
            
    # Save final model
    final_model_path = f"{config.MODEL_FOLDER}/model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved at {final_model_path}")
    
    return model