import chess
import numpy as np
import config
import functools
import time

def move_to_index(move):
    """
    Convert a chess.Move to an index for the policy vector.
    
    For AlphaZero-style encoding with 73 planes (56 queen, 8 knight, 9 underpromotion):
    - Each source square can have up to 73 possible moves
    - Index = source_square * 73 + move_type
    
    Where move_type is:
    - 0-55: Queen moves (8 directions × 7 squares)
    - 56-63: Knight moves (8 possible moves)
    - 64-72: Underpromotions (3 piece types × 3 directions)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Get row and column distance
    from_row, from_col = divmod(from_square, 8)
    to_row, to_col = divmod(to_square, 8)
    row_diff = to_row - from_row
    col_diff = to_col - from_col
    
    # Knight moves
    if abs(row_diff) == 2 and abs(col_diff) == 1 or abs(row_diff) == 1 and abs(col_diff) == 2:
        # Map the 8 possible knight moves to indices 56-63
        knight_map = {
            (2, 1): 56,    # Up 2, right 1
            (2, -1): 57,   # Up 2, left 1
            (1, 2): 58,    # Up 1, right 2
            (1, -2): 59,   # Up 1, left 2
            (-1, 2): 60,   # Down 1, right 2
            (-1, -2): 61,  # Down 1, left 2
            (-2, 1): 62,   # Down 2, right 1
            (-2, -1): 63,  # Down 2, left 1
        }
        move_type = knight_map.get((row_diff, col_diff))
    
    # Underpromotions (not queen)
    elif move.promotion is not None and move.promotion != chess.QUEEN:
        # Map promotions to bishop (64-66), knight (67-69), rook (70-72)
        # Direction: straight, right, left
        promo_map = {
            chess.BISHOP: 64,
            chess.KNIGHT: 67,
            chess.ROOK: 70
        }
        base = promo_map[move.promotion]
        
        # Direction
        if col_diff == 0:      # Straight ahead
            direction = 0
        elif col_diff > 0:     # Capture right
            direction = 1
        else:                  # Capture left
            direction = 2
            
        move_type = base + direction
    
    # Queen moves (including queen promotions)
    else:
        # Determine direction
        direction = -1
        
        # Horizontal
        if row_diff == 0:
            direction = 0 if col_diff > 0 else 1
        # Vertical
        elif col_diff == 0:
            direction = 2 if row_diff > 0 else 3
        # Diagonal
        elif abs(row_diff) == abs(col_diff):
            if row_diff > 0 and col_diff > 0:
                direction = 4  # Down-right
            elif row_diff > 0 and col_diff < 0:
                direction = 5  # Down-left
            elif row_diff < 0 and col_diff > 0:
                direction = 6  # Up-right
            else:
                direction = 7  # Up-left
        
        # Distance (0-6), subtract 1 since we're 0-indexed
        distance = max(abs(row_diff), abs(col_diff)) - 1
        
        # Queen moves: 8 directions × 7 distances = 56 total
        move_type = direction * 7 + distance
    
    # Final index: source square * 73 + move type
    return from_square * config.amount_of_planes + move_type

def time_function(func):
    """
    A decorator that prints the execution time of the decorated function.
    
    Args:
        func: The function to decorate
        
    Returns:
        The wrapped function with timing functionality
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper