import torch
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
from agent import Agent
from modelbuilder import RLModelBuilder

logging.basicConfig(level=logging.INFO, format=" %(message)s")

def generate_game(model_path, game_id, device=None):
    """Generate a self-play game for multiprocessing."""
    # Pass device to Agent
    agent = Agent(model_path=model_path, device=device)
    board = chess.Board()
    states, policies, values = [], [], []
    move_count = 0
    if device and device.type == 'cuda':
        # Limit GPU memory usage per process
        torch.cuda.set_per_process_memory_fraction(0.1)  # Use only 10% of memory per process
    
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

def run_game(game_id_and_path):
    """Wrapper function for multiprocessing that can be pickled correctly."""
    game_id, model_path, device = game_id_and_path
    return generate_game(model_path, game_id, device)

def generate_selfplay_data_parallel(model_path, n_games=10, output_path=None, device=None):
    """
    Generate self-play data using multiple processes.
    
    Args:
        model_path: Path to the model to use for self-play
        n_games: Number of games to generate
        output_path: Path to save the generated data (optional)
        device: Device to run model inference on (cuda/cpu)
        
    Returns:
        tuple: (states, policies, values) if output_path is None,
               otherwise saves data to output_path and returns the number of positions
    """
    start_time = time.time()
    mp_context = torch_mp.get_context('spawn')
    num_processes = min(config.NUM_WORKERS, n_games)
    
    if device and device.type == 'cuda':
        logging.info(f"Using GPU ({torch.cuda.get_device_name(0)}) for model initialization")
    else:
        logging.info(f"Using CPU for model initialization")
    
    # This avoids sharing the model directly which can cause issues
    temp_model_path = os.path.join(config.MODEL_FOLDER, f"temp_model_{int(time.time())}.pt")
    if os.path.exists(model_path):
        # Load and immediately save the model to the temp path
        model = RLModelBuilder(
            config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
        ).build_model(model_path)
        
        torch.save(model.state_dict(), temp_model_path)
    
    try:
        # IMPORTANT CHANGE: Use a list of tuples instead of lambda
        game_args = [(i, temp_model_path, device) for i in range(n_games)]
        
        # Use Pool for parallel processing
        with mp_context.Pool(num_processes) as pool:
            # Track progress with tqdm
            try:
                from tqdm import tqdm
                # IMPORTANT CHANGE: Pass the run_game function directly
                results = list(tqdm(pool.imap(run_game, game_args), 
                                   total=n_games, desc="Self-play games"))
            except ImportError:
                results = pool.map(run_game, game_args)
        
        # Collect results
        all_states, all_policies, all_values = [], [], []
        total_positions = 0
        
        for states, policies, values in results:
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            total_positions += len(states)
        
        elapsed_time = time.time() - start_time
        positions_per_second = total_positions / elapsed_time if elapsed_time > 0 else 0
        
        logging.info(f"Generated {total_positions} positions from {n_games} games in {elapsed_time:.1f}s "
                   f"({positions_per_second:.1f} positions/s)")
                   
        # Save data if output path is provided
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Save as compressed numpy file
            if os.path.exists(output_path) and output_path.endswith('.npz'):
                # Append to existing data
                try:
                    existing_data = np.load(output_path, allow_pickle=True)
                    existing_states = existing_data['states'].tolist()
                    existing_policies = existing_data['policies'].tolist()
                    existing_values = existing_data['values'].tolist()
                    
                    # Combine existing and new data
                    all_states = existing_states + [s.fen() for s in all_states]
                    all_policies = existing_policies + all_policies
                    all_values = existing_values + all_values
                    
                    logging.info(f"Appended to existing data: total positions now {len(all_states)}")
                except Exception as e:
                    logging.warning(f"Could not append to existing data: {e}")
                    all_states = [s.fen() for s in all_states]
            else:
                all_states = [s.fen() for s in all_states]
                
            np.savez_compressed(output_path, 
                              states=all_states,
                              policies=all_policies, 
                              values=all_values)
            
            logging.info(f"Data saved to {output_path}")
            return total_positions
        else:
            return all_states, all_policies, all_values
        
    finally:
        # Clean up temporary model file
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except Exception as e:
                print(f"Warning: Could not remove temp model: {e}")

def visualize_game(model_path):
    """Visualize a self-play game using selfplay.py."""
    try:
        from selfplay import setup, self_play
        
        # Use selfplay.py's functionality
        setup_data = setup(model_path=model_path)
        self_play(setup_data)
        
    except ImportError as e:
        logging.warning(f"Could not visualize self-play game: {e}")

def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate self-play data for chess DRL")
    parser.add_argument("--model", type=str, default=os.path.join(config.MODEL_FOLDER, "initial_model.pt"),
                      help="Path to model for data generation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for data generation")
    parser.add_argument("--games", "-g", type=int, default=config.N_SELFPLAY_GAMES, 
                      help="Number of self-play games to generate")
    parser.add_argument("--output", "-o", type=str, default="./memory/selfplay_data.npz",
                      help="Path to save generated data")
    parser.add_argument("--append", "-a", action="store_true",
                      help="Append to existing data rather than overwriting")
    parser.add_argument("--visualize", "-v", action="store_true",
                      help="Visualize a self-play game after generation")
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for data generation")
    
    # Set up paths
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(config.MODEL_FOLDER, exist_ok=True)
    
    # Generate data
    logging.info(f"Generating {args.games} self-play games using model {args.model}")
    
    generate_selfplay_data_parallel(args.model, args.games, args.output, device)
    
    # Visualize a game if requested
    if args.visualize:
        logging.info("Visualizing a self-play game...")
        visualize_game(args.model)

if __name__ == "__main__":
    main()