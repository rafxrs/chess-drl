# Interacts with the chess environment and uses MCTS and a neural network for decision-making.
import chess
import numpy as np
import torch

import config
from mcts import MCTS, Node
from model import RLModelBuilder


class Agent:
    def __init__(self, model_path=None, state=chess.STARTING_FEN, device=None):
        """
        Initialize the agent with a model and state.

        Args:
            model_path: Path to the model weights
            state: Initial chess state as FEN string
            device: Device to run inference on (cuda/cpu)
        """
        if model_path is None:
            raise ValueError("Specify the path to the model to use.")

        self.model_path = model_path
        self.state = state
        self.device = device if device is not None else config.DEVICE

        self.mcts = MCTS(self, config.__dict__)
        self.model = None
        self.build_model()

    def build_model(self):
        """Build and load the model."""
        if self.model is None:
            self.model = RLModelBuilder(
                config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
            ).build_model(self.model_path, self.device)
            self.model.eval()

    def run_simulations(self, n: int = 1):
        """Run n MCTS simulations from the current state."""
        if self.model is None:
            self.build_model()

        # The search tree only makes sense for the exact position it was
        # built from, and nothing here threads the actual move sequence
        # through to reuse it correctly across plies, so start fresh
        # every time a move is requested.
        self.mcts.root = Node(None, 1.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.mcts.run_simulation(self.model, n)

    def get_move(self, env):
        """
        Get the best move for the current state using MCTS.

        Args:
            env: Chess environment with a 'board' attribute

        Returns:
            chess.Move: The selected move
        """
        self.state = env.board.fen()
        self.run_simulations(self.mcts.n_simulations)
        actions, probs = self.mcts.get_move_probs()

        legal = [(a, p) for a, p in zip(actions, probs) if a in env.board.legal_moves]
        if legal:
            legal_actions, legal_probs = zip(*legal)
            legal_probs = np.array(legal_probs) / sum(legal_probs)
            return np.random.choice(legal_actions, p=legal_probs)

        return np.random.choice(list(env.board.legal_moves))

    def play_move(self, env, stochastic=True, previous_moves=None):
        """Compatibility wrapper around get_move used by Game."""
        return self.get_move(env)

    def predict(self, state_tensor):
        """Run model prediction on a single state tensor."""
        if self.model is None:
            self.build_model()

        self.model.eval()

        if isinstance(state_tensor, np.ndarray):
            state_tensor = torch.from_numpy(state_tensor).float()

        state_tensor = state_tensor.to(self.device)

        with torch.no_grad():
            return self.model(state_tensor)

    def predict_batch(self, states_batch):
        """Run model prediction on a batch of board states or a preprocessed tensor."""
        if self.model is None:
            self.build_model()

        self.model.eval()

        if isinstance(states_batch, list):
            tensors = [self.state_to_tensor(s, add_batch=False) for s in states_batch]
            states_batch = torch.cat(tensors, dim=0)

        states_batch = states_batch.to(self.device)

        with torch.no_grad():
            policy, value = self.model(states_batch)

        if states_batch.size(0) > 64 and self.device.type == "cuda":
            torch.cuda.empty_cache()

        return policy, value

    @staticmethod
    def state_to_tensor(state, add_batch=True):
        """
        Convert a chess.Board (or FEN string) to a CPU input tensor for the
        neural network. Callers are responsible for moving it to whatever
        device they're running on.
        """
        if isinstance(state, str):
            state = chess.Board(state)

        planes = np.zeros((config.amount_of_input_planes, 8, 8), dtype=np.float32)
        plane_idx = 0

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        current_player = state.turn
        opponent = not current_player

        # Planes 0-11: current player's pieces, then the opponent's, so the
        # network always sees the board from the perspective of the side to move.
        for color, start_plane in [(current_player, 0), (opponent, 6)]:
            for i, piece_type in enumerate(piece_types):
                for square in state.pieces(piece_type, color):
                    row, col = divmod(square, 8)
                    planes[start_plane + i][row][col] = 1

        plane_idx = 12

        if state.ep_square is not None:
            row, col = divmod(state.ep_square, 8)
            planes[plane_idx][row][col] = 1
        plane_idx += 1

        if state.turn == chess.WHITE:
            planes[plane_idx].fill(1)
        plane_idx += 1

        if state.has_kingside_castling_rights(chess.WHITE):
            planes[plane_idx].fill(1)
        plane_idx += 1

        if state.has_queenside_castling_rights(chess.WHITE):
            planes[plane_idx].fill(1)
        plane_idx += 1

        if state.has_kingside_castling_rights(chess.BLACK):
            planes[plane_idx].fill(1)
        plane_idx += 1

        if state.has_queenside_castling_rights(chess.BLACK):
            planes[plane_idx].fill(1)
        plane_idx += 1

        planes[plane_idx].fill(min(1.0, state.halfmove_clock / 100.0))

        tensor = torch.from_numpy(planes).float()
        if add_batch:
            tensor = tensor.unsqueeze(0)

        return tensor
