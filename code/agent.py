# interacts with the chess environment and uses MCTS and a neural network for decision-making.
import torch
import chess
import logging
import config
import utils
import numpy as np
from modelbuilder import RLModelBuilder
from tqdm import tqdm
from mcts import MCTS
from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(self, model_path=None, state=chess.STARTING_FEN, device=None):
        """
        Initialize the agent with a model and state.
        
        Args:
            model_path: Path to the model weights
            state: Initial chess state as FEN string
            device: Device to run inference on (cuda/cpu)
        """
        self.model_path = model_path
        self.state = state
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        logging.debug(f"Agent initialized on device: {self.device}")
        
        self.mcts = MCTS(self, config.__dict__)
        self.model = None

        if model_path is None:
            raise ValueError("Specify the path to the model to use.")
        self.build_model()

    def build_model(self):
        """Build and load the model."""
        if self.model is None:
            self.model = RLModelBuilder(
                config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
            ).build_model(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode for inference
            logging.info(f"Model built from {self.model_path} on {self.device}")

    def run_simulations(self, n: int = 1):
        """Run n MCTS simulations from the current state."""
        if self.model is None:
            self.build_model()
            
        # Ensure device synchronization before starting simulations
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        self.mcts.run_simulation(self.model, n)

    def save_model(self, timestamped: bool = False):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_model() first.")
        if timestamped:
            timestamp = utils.get_timestamp()
            model_path = f"{self.model_path}_{timestamp}.pt"
        else:
            model_path = self.model_path
        torch.save(self.model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
        return model_path
    
    def get_move(self, env):
        """
        Get the best move for the current state using MCTS.
        
        Args:
            env: Chess environment with a 'board' attribute
            
        Returns:
            chess.Move: The selected move
        """
        self.state = env.board.fen()
        self.run_simulations(config.SIMULATIONS_PER_MOVE)
        actions, probs = self.mcts.get_move_probs()
        move = np.random.choice(actions, p=probs)
        return move
    
    def play_move(self, env, stochastic=True, previous_moves=None):
        """
        Compatibility method that wraps get_move.
        
        Args:
            env: Chess environment
            stochastic: Whether to use stochastic move selection
            previous_moves: List of previous moves
        
        Returns:
            chess.Move: Selected move
        """
        # Simply delegate to get_move
        return self.get_move(env)

    def predict(self, state_tensor):
        """
        Run model prediction on state tensor.
        
        Args:
            state_tensor: Preprocessed state as tensor
            
        Returns:
            tuple: (policy logits, value)
        """
        if self.model is None:
            self.build_model()
            
        self.model.eval()  # Set model to evaluation mode
        
        # Move tensor to correct device if not already there
        if isinstance(state_tensor, np.ndarray):
            state_tensor = torch.from_numpy(state_tensor).float()
            
        if state_tensor.device != self.device:
            state_tensor = state_tensor.to(self.device)
            
        with torch.no_grad():
            policy, value = self.model(state_tensor)
        
        return policy, value
    
    def predict_batch(self, states_batch):
        """
        Run model prediction on a batch of states.
        
        Args:
            states_batch: List of states or batch tensor
            
        Returns:
            tuple: (policy logits tensor, value tensor)
        """
        if self.model is None:
            self.build_model()
            
        self.model.eval()
        
        # Handle list of states vs. preprocessed tensor
        if isinstance(states_batch, list):
            # Convert list of states to tensor batch
            tensors = [self.state_to_tensor(s, add_batch=False) for s in states_batch]
            state_tensor = torch.cat(tensors, dim=0).to(self.device)
        else:
            # Already a tensor, just move to device
            state_tensor = states_batch.to(self.device)
        
        with torch.no_grad():
            policy, value = self.model(state_tensor)
            
        # Optional: clear CUDA cache on large batches
        if state_tensor.size(0) > 64 and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return policy, value

    @staticmethod
    def state_to_tensor(state, add_batch=True):
        """
        Convert a chess.Board to a tensor for the neural network.
        
        Args:
            state: chess.Board or FEN string
            add_batch: Whether to add a batch dimension
            
        Returns:
            torch.Tensor: Input tensor for the model
        """
        if isinstance(state, str):
            state = chess.Board(state)
            
        # Initialize all planes to zeros
        planes = np.zeros((config.amount_of_input_planes, 8, 8), dtype=np.float32)
        
        # Keep track of the current plane index
        plane_idx = 0
        
        # 1-6: Current player's pieces
        # 7-12: Opponent's pieces
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                    chess.ROOK, chess.QUEEN, chess.KING]
        
        # Determine current player and opponent color
        current_player = state.turn
        opponent = not current_player
        
        # Fill planes for pieces
        for color, start_plane in [(current_player, 0), (opponent, 6)]:
            for i, piece_type in enumerate(piece_types):
                for square in state.pieces(piece_type, color):
                    row, col = divmod(square, 8)
                    planes[start_plane + i][row][col] = 1
        
        # Update plane index
        plane_idx = 12
        
        # En passant square
        if state.ep_square is not None:
            row, col = divmod(state.ep_square, 8)
            planes[plane_idx][row][col] = 1
        plane_idx += 1
        
        # Side to move (fill with 1s if white to move)
        if state.turn == chess.WHITE:
            planes[plane_idx].fill(1)
        plane_idx += 1
        
        # Castling rights
        # White kingside
        if state.has_kingside_castling_rights(chess.WHITE):
            planes[plane_idx].fill(1)
        plane_idx += 1
        
        # White queenside
        if state.has_queenside_castling_rights(chess.WHITE):
            planes[plane_idx].fill(1)
        plane_idx += 1
        
        # Black kingside
        if state.has_kingside_castling_rights(chess.BLACK):
            planes[plane_idx].fill(1)
        plane_idx += 1
        
        # Black queenside
        if state.has_queenside_castling_rights(chess.BLACK):
            planes[plane_idx].fill(1)
        plane_idx += 1
        
        # Move count (50-move rule) - normalize to [0, 1]
        planes[plane_idx].fill(min(1.0, state.halfmove_clock / 100.0))
        
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(planes).float()
        
        # Move tensor to the correct device
        tensor = tensor.to(config.DEVICE)
        
        # Add batch dimension if needed
        if add_batch:
            tensor = tensor.unsqueeze(0)
                
        return tensor