import logging

import chess
from chess import Move

logging.basicConfig(level=logging.INFO, format=" %(message)s")


class Chess_Env:
    """
    Chess environment that handles board state and move execution for reinforcement learning.
    """
    def __init__(self, starting_position: str = chess.STARTING_FEN):
        self.initial_fen = starting_position
        self.board = None
        self.reset()

    def reset(self, fen: str = None):
        """Reset the board to the starting position or a specified FEN."""
        self.board = chess.Board(fen or self.initial_fen)
        return self.board

    def step(self, action: Move) -> tuple:
        """Execute a move and return the new state, reward, and done flag."""
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
        return self.board.is_game_over()

    def get_result(self):
        """Get string representation of game result."""
        return self.board.result() if self.is_game_over() else None

    def get_reward(self):
        """1 for white win, -1 for black win, 0 for draw or ongoing."""
        if not self.is_game_over():
            return 0
        result = self.get_result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        return 0

    def copy(self):
        """Create a deep copy of the environment."""
        env_copy = Chess_Env()
        env_copy.board = self.board.copy()
        return env_copy

    def __str__(self):
        return str(self.board)

    def render(self):
        print(self.board)
