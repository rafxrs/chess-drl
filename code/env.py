"""Thin wrapper around a python-chess board with move validation."""
import logging

import chess

logging.basicConfig(level=logging.INFO, format=" %(message)s")


class Chess_Env:
    def __init__(self, starting_position: str = chess.STARTING_FEN):
        self.initial_fen = starting_position
        self.reset()

    def reset(self, fen: str = None):
        self.board = chess.Board(fen or self.initial_fen)
        return self.board

    def legal_moves(self):
        return list(self.board.legal_moves)

    def push(self, move):
        """Play `move` (chess.Move or UCI string). Returns False if it's invalid or illegal."""
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

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        """'1-0', '0-1', '1/2-1/2', or None if the game is still going."""
        return self.board.result() if self.is_game_over() else None

    def __str__(self):
        return str(self.board)
