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
    def __init__(self, model_path=None, state=chess.STARTING_FEN):
        self.model_path = model_path
        self.state = state
        self.mcts = MCTS(self, config.__dict__)
        self.model = None

        if model_path is None:
            raise ValueError("Specify the path to the model to use.")
        self.model = RLModelBuilder(
            config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
        ).build_model(model_path)
        logging.info(f"Using local model from {model_path}")

    def build_model(self):
        if self.model is None:
            self.model = RLModelBuilder(
                config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
            ).build_model(self.model_path)
            logging.info(f"Model built from {self.model_path}")

    def run_simulations(self, n: int = 1):
        self.build_model()
        self.mcts.run_simulation(self.model, n)

    def save_model(self, timestamped: bool = False):
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
        self.state = env.board.fen()
        self.run_simulations(config.SIMULATIONS_PER_MOVE)
        actions, probs = self.mcts.get_move_probs()
        move = np.random.choice(actions, p=probs)
        return move

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_model() first.")
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float()
            policy, value = self.model(data_tensor)
        return policy, value

    def state_to_tensor(self, state):
        """
        Convert a chess.Board to a tensor for the neural network.
        
        Input planes (19 planes total):
        - 6 planes for each piece type for the current player (6)
        - 6 planes for each piece type for the opponent (6)
        - 1 plane for en passant square (1)
        - 1 plane for side to move (1)
        - 4 planes for castling rights (4)
        - 1 plane for move count - used for 50-move rule (1)
        
        Each plane is an 8x8 matrix.
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
        # The expected input shape is (channels, height, width)
        tensor = torch.from_numpy(planes).float()
        if self.model is not None and next(self.model.parameters()).is_cuda:
            return tensor.cuda().unsqueeze(0)
        return tensor.unsqueeze(0)