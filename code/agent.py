"""Chess-playing agent: a neural network guiding MCTS to pick moves."""
import chess
import numpy as np
import torch

import config
from mcts import MCTS, Node
from model import RLModelBuilder


class Agent:
    def __init__(self, model_path, state=chess.STARTING_FEN, device=None):
        """Load the model at `model_path` onto `device` (defaults to config.DEVICE)."""
        if model_path is None:
            raise ValueError("Specify the path to the model to use.")

        self.model_path = model_path
        self.state = state
        self.device = device if device is not None else config.DEVICE

        self.mcts = MCTS(self, config.__dict__)
        self.model = RLModelBuilder(
            config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
        ).build_model(self.model_path, self.device)
        self.model.eval()

    def run_simulations(self, n: int = 1):
        """Run n MCTS simulations from `self.state`."""
        # Rebuild the tree each move: reusing it would require threading the
        # actual move sequence through, and a stale root makes every simulation
        # fall off the tree.
        self.mcts.root = Node(None, 1.0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.mcts.run_simulation(self.model, n)

    def get_move(self, env):
        """Search the position in `env.board` and sample a move from the visit counts."""
        self.state = env.board.fen()
        self.run_simulations(self.mcts.n_simulations)
        actions, probs = self.mcts.get_move_probs()

        legal = [(a, p) for a, p in zip(actions, probs) if a in env.board.legal_moves]
        if legal:
            legal_actions, legal_probs = zip(*legal)
            legal_probs = np.array(legal_probs) / sum(legal_probs)
            return np.random.choice(legal_actions, p=legal_probs)

        return np.random.choice(list(env.board.legal_moves))

    @staticmethod
    def state_to_tensor(state, add_batch=True):
        """Encode a board (or FEN) as a 19x8x8 CPU tensor; callers move it to their device."""
        if isinstance(state, str):
            state = chess.Board(state)

        planes = np.zeros((config.amount_of_input_planes, 8, 8), dtype=np.float32)

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        current_player = state.turn
        opponent = not current_player

        # Planes 0-11: side to move's pieces first, then the opponent's.
        for color, start_plane in [(current_player, 0), (opponent, 6)]:
            for i, piece_type in enumerate(piece_types):
                for square in state.pieces(piece_type, color):
                    row, col = divmod(square, 8)
                    planes[start_plane + i][row][col] = 1

        # Plane 12: en passant square
        if state.ep_square is not None:
            row, col = divmod(state.ep_square, 8)
            planes[12][row][col] = 1

        # Planes 13-18: side to move, castling rights, halfmove clock
        flags = [
            state.turn == chess.WHITE,
            state.has_kingside_castling_rights(chess.WHITE),
            state.has_queenside_castling_rights(chess.WHITE),
            state.has_kingside_castling_rights(chess.BLACK),
            state.has_queenside_castling_rights(chess.BLACK),
        ]
        for offset, flag in enumerate(flags):
            if flag:
                planes[13 + offset].fill(1)
        planes[18].fill(min(1.0, state.halfmove_clock / 100.0))

        tensor = torch.from_numpy(planes)
        return tensor.unsqueeze(0) if add_batch else tensor
