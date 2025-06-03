import config
import chess
import numpy as np
import logging
import torch

from chess import Move

logging.basicConfig(level=logging.INFO, format=' %(message)s')

class Chess_Env:
    """
    Chess environment that handles board state and conversions for reinforcement learning.
    """
    def __init__(self, starting_position: str = chess.STARTING_FEN):
        """
        Initialize the chess environment with a starting position.
        
        Args:
            starting_position: FEN string representation of the starting position
        """
        self.initial_fen = starting_position
        self.board = None
        self.reset()
        
    def reset(self, fen: str = None):
        """
        Reset the board to the starting position or a specified FEN.
        
        Args:
            fen: Optional FEN string to set the board to
        
        Returns:
            The board after reset
        """
        if fen is None:
            fen = self.initial_fen
        self.board = chess.Board(fen)
        return self.board
    
    def step(self, action: Move) -> tuple:
        """
        Execute a move and return the new state, reward, and done flag.
        
        Args:
            action: A chess.Move object or UCI string
            
        Returns:
            (board, reward, done, info) tuple
        """
        if not isinstance(action, chess.Move):
            try:
                action = chess.Move.from_uci(str(action))
            except ValueError:
                logging.error(f"Invalid move: {action}")
                return self.board, -1, True, {"result": "illegal_move"}
        
        if action not in self.board.legal_moves:
            logging.error(f"Illegal move: {action}")
            return self.board, -1, True, {"result": "illegal_move"}
        
        self.board.push(action)
        
        # Check game state
        done = self.is_game_over()
        reward = self.get_reward()
        
        return self.board, reward, done, {"result": self.get_result()}
    
    def legal_moves(self):
        """Return list of legal moves."""
        return list(self.board.legal_moves)
    
    def push(self, move):
        """Execute a move on the board."""
        if not isinstance(move, chess.Move):
            try:
                move = chess.Move.from_uci(str(move))
            except ValueError:
                logging.error(f"Invalid move: {move}")
                return False
        
        if move not in self.board.legal_moves:
            logging.error(f"Illegal move: {move}")
            return False
        
        self.board.push(move)
        return True
    
    def pop(self):
        """Undo the last move."""
        try:
            return self.board.pop()
        except IndexError:
            logging.error("No moves to undo")
            return None
    
    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_result(self):
        """Get string representation of game result."""
        if not self.is_game_over():
            return None
        return self.board.result()
    
    def get_reward(self):
        """
        Calculate reward based on game outcome.
        1 for white win, -1 for black win, 0 for draw or ongoing.
        """
        if not self.is_game_over():
            return 0
        
        result = self.get_result()
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        else:  # Draw
            return 0

    @staticmethod
    def state_to_input(fen: str) -> np.ndarray:
        """
        Convert board to an input tensor for the neural network.
        
        Args:
            fen: FEN string representation of the board
        
        Returns:
            numpy array with shape matching config.INPUT_SHAPE
        """
        board = chess.Board(fen)

        # 1. is it white's turn? (1x8x8)
        is_white_turn = np.ones((8, 8)) if board.turn else np.zeros((8, 8))

        # 2. castling rights (4x8x8)
        castling = np.asarray([
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
        ])

        # 3. repitition counter (fifty-move rule)
        counter = np.ones((8, 8)) if board.can_claim_fifty_moves() else np.zeros((8, 8))

        # 4-5. piece positions for both players (12 planes: 2 colors × 6 piece types)
        piece_planes = []
        for color in chess.COLORS:
            for piece_type in chess.PIECE_TYPES:
                # Create 8x8 plane for each piece type and color
                plane = np.zeros((8, 8))
                for square in board.pieces(piece_type, color):
                    row, col = 7 - (square // 8), square % 8  # Convert to row, col (0-7)
                    plane[row][col] = 1
                piece_planes.append(plane)
        piece_planes = np.asarray(piece_planes)

        # 6. en passant square (1x8x8)
        en_passant = np.zeros((8, 8))
        if board.ep_square is not None:
            row, col = 7 - (board.ep_square // 8), board.ep_square % 8
            en_passant[row][col] = 1

        # Stack all planes: 1 (turn) + 4 (castling) + 1 (fifty-move) + 12 (pieces) + 1 (en passant) = 19
        stacked = np.vstack([
            is_white_turn.reshape(1, 8, 8),
            castling,
            counter.reshape(1, 8, 8),
            piece_planes,
            en_passant.reshape(1, 8, 8)
        ])
        
        # Make sure the shape matches config
        if stacked.shape != (config.amount_of_input_planes, 8, 8):
            logging.warning(f"Input shape mismatch: got {stacked.shape}, expected {(config.amount_of_input_planes, 8, 8)}")
        
        return stacked.astype(np.float32)

    def state_to_tensor(self):
        """Convert current board state to PyTorch tensor."""
        state_array = self.state_to_input(self.board.fen())
        return torch.from_numpy(state_array).float().unsqueeze(0)  # Add batch dimension
    
    @staticmethod
    def estimate_winner(board: chess.Board) -> float:
        """
        Heuristic material-based evaluation of position.
        
        Args:
            board: Chess board to evaluate
            
        Returns:
            Score between -1 and 1, positive favors white
        """
        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                return 1.0
            elif result == '0-1':
                return -1.0
            else:
                return 0.0
                
        score = 0
        piece_scores = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Calculate material balance
        for piece_type in chess.PIECE_TYPES:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_scores[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_scores[piece_type]
        
        # Normalize score to [-1, 1]
        max_material = 39  # 9 (queen) + 2*5 (rooks) + 2*3 (bishops) + 2*3 (knights) + 8*1 (pawns) = 39
        return min(max(score / max_material, -1), 1)
    
    @staticmethod
    def get_piece_amount(board: chess.Board) -> int:
        """Count total pieces on the board."""
        return len(board.piece_map())
    
    def copy(self):
        """Create a deep copy of the environment."""
        env_copy = Chess_Env()
        env_copy.board = self.board.copy()
        return env_copy
    
    def __str__(self):
        """Return string representation of the board."""
        return str(self.board)
    
    def render(self):
        """Print ASCII representation of the board."""
        print(self.board)
