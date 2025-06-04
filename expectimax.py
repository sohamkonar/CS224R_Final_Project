"""
expectimax.py - Expectimax algorithm implementation for 2048
"""

import numpy as np
from typing import Tuple

from game import legal_move_mask, simulate_move # Assuming game.py is in the same directory

def expectimax_search(board: np.ndarray, depth: int = 3) -> int:
    """
    Run expectimax search to find the best action for the current board state.

    Parameters:
    -----------
    board : np.ndarray
        The current 2048 board state (4x4 array of exponents)
    depth : int
        The maximum depth to search

    Returns:
    --------
    int : The best action (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT), or -1 if no legal moves.
    """
    possible_moves = legal_move_mask(board)

    if not possible_moves.any():  # No legal moves
        return -1

    best_value = -float('inf')
    best_action = -1

    for action in range(4):
        if not possible_moves[action]:
            continue

        next_board, reward = simulate_move(board.copy(), action)

        # If the board didn't change, this is not a productive move in the context of search
        # (though the environment might allow it, it doesn't help us find a better state)
        if np.array_equal(board, next_board):
            # Consider a small penalty or skip if the move results in no change
            # For now, we'll treat it as a non-beneficial path for expectimax
            current_value = chance_node_value(next_board, depth -1) # Still evaluate the chance node from this state
        else:
            current_value = reward + chance_node_value(next_board, depth - 1)

        if current_value > best_value:
            best_value = current_value
            best_action = action
    
    # If no action improves the score (e.g. all lead to terminal states or no change)
    # and there are legal moves, pick the first legal one as a fallback.
    if best_action == -1 and possible_moves.any():
        best_action = np.argmax(possible_moves) # First true value
        
    return best_action

def max_node_value(board: np.ndarray, depth: int) -> float:
    """
    Calculate the maximum value for a max node (player's turn) in expectimax.
    """
    if depth == 0 or is_terminal(board):
        return evaluate_board(board)

    possible_moves = legal_move_mask(board)
    if not possible_moves.any():
        return evaluate_board(board) # Terminal state

    max_val = -float('inf')

    for action in range(4):
        if not possible_moves[action]:
            continue

        next_board, reward = simulate_move(board.copy(), action)
        
        if np.array_equal(board, next_board):
            # If move causes no change, the value is from the chance node of current board
            val = chance_node_value(next_board, depth -1) 
        else:
            val = reward + chance_node_value(next_board, depth - 1)
        max_val = max(max_val, val)

    return max_val

def chance_node_value(board: np.ndarray, depth: int) -> float:
    """
    Calculate the expected value for a chance node (random tile placement) in expectimax.
    """
    if depth == 0 or is_terminal(board): # is_terminal check is important here too
        return evaluate_board(board)

    empty_cells = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]

    if not empty_cells:
        return evaluate_board(board) # No space for new tiles, effectively terminal for chance

    expected_value = 0
    num_empty = len(empty_cells)

    # Probability of a 2 (exponent 1) is 0.9
    # Probability of a 4 (exponent 2) is 0.1
    for r, c in empty_cells:
        # Try placing a 2 (exponent 1)
        board_with_2 = board.copy()
        board_with_2[r, c] = 1
        expected_value += 0.9 * max_node_value(board_with_2, depth) / num_empty

        # Try placing a 4 (exponent 2)
        board_with_4 = board.copy()
        board_with_4[r, c] = 2
        expected_value += 0.1 * max_node_value(board_with_4, depth) / num_empty

    return expected_value

def is_terminal(board: np.ndarray) -> bool:
    """Check if the current state is terminal (no legal moves)."""
    return not legal_move_mask(board).any()

def evaluate_board(board: np.ndarray) -> float:
    """
    Heuristic function to evaluate a board state.
    Weights can be tuned.
    """
    score = np.sum(2**board[board > 0]) # Sum of tile values (2^exponent)
    empty_cells = np.sum(board == 0)
    monotonicity_score = calculate_monotonicity(board)
    smoothness_score = calculate_smoothness(board)
    max_tile_val = np.max(board) # Max exponent

    # Heuristic weights (these are critical and need tuning)
    w_score = 1.0
    w_empty = 2.7
    w_mono = 1.0
    w_smooth = 0.1
    w_max = 1.0

    heuristic_val = (
        w_score * score +
        w_empty * empty_cells +
        w_mono * monotonicity_score +
        w_smooth * smoothness_score +
        w_max * max_tile_val 
    )
    return heuristic_val

def calculate_monotonicity(board: np.ndarray) -> float:
    """Calculate the monotonicity score of the board."""
    scores = [0, 0, 0, 0] # up, down, left, right

    # Left/Right monotonicity
    for i in range(4):
        for j in range(3):
            # Left (current > next or current == 0)
            if board[i, j] >= board[i, j+1] or board[i,j] == 0:
                scores[2] += (2**board[i,j]) - (2**board[i,j+1]) if board[i,j+1]!=0 else 2**board[i,j]
            # Right (current < next or current == 0)
            if board[i, j] <= board[i, j+1] or board[i,j] == 0:
                scores[3] += (2**board[i,j+1]) - (2**board[i,j]) if board[i,j]!=0 else 2**board[i,j+1]

    # Up/Down monotonicity
    for j in range(4):
        for i in range(3):
            # Up (current > next or current == 0)
            if board[i, j] >= board[i+1, j] or board[i,j] == 0:
                scores[0] += (2**board[i,j]) - (2**board[i+1,j]) if board[i+1,j]!=0 else 2**board[i,j]
            # Down (current < next or current == 0)
            if board[i, j] <= board[i+1, j] or board[i,j] == 0:
                scores[1] += (2**board[i+1,j]) - (2**board[i,j]) if board[i,j]!=0 else 2**board[i+1,j]
    
    return float(max(scores[0], scores[1]) + max(scores[2], scores[3]))

def calculate_smoothness(board: np.ndarray) -> float:
    """Calculate the smoothness score of the board (lower is better, so we negate)."""
    smoothness = 0
    for i in range(4):
        for j in range(4):
            if board[i, j] != 0:
                val = np.log2(2**board[i,j]) if board[i,j] > 0 else 0
                # Check right neighbor
                if j + 1 < 4 and board[i, j+1] != 0:
                    neighbor_val = np.log2(2**board[i,j+1]) if board[i,j+1] > 0 else 0
                    smoothness -= abs(val - neighbor_val)
                # Check down neighbor
                if i + 1 < 4 and board[i+1, j] != 0:
                    neighbor_val = np.log2(2**board[i+1,j]) if board[i+1,j] > 0 else 0
                    smoothness -= abs(val - neighbor_val)
    return smoothness

if __name__ == '__main__':
    # Example Usage (requires a game environment or board setup)
    # Create a sample board (exponents of 2, 0 for empty)
    # 2 4 0 0
    # 0 2 4 0
    # 0 0 2 4
    # 0 0 0 2
    sample_board = np.array([
        [1, 2, 0, 0],
        [0, 1, 2, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ], dtype=int)

    print("Sample board:")
    print(2**sample_board) # Print actual tile values
    
    if not is_terminal(sample_board):
        best_action = expectimax_search(sample_board, depth=3)
        action_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT", -1: "NONE"}
        print(f"Best action for sample board (depth 3): {action_map[best_action]}")
        
        # Test evaluation
        print(f"Board evaluation score: {evaluate_board(sample_board)}")
    else:
        print("Sample board is terminal.")

    # Test with an empty board
    empty_board = np.zeros((4,4), dtype=int)
    print("\nEmpty board:")
    print(2**empty_board)
    if not is_terminal(empty_board):
        best_action_empty = expectimax_search(empty_board, depth=3)
        print(f"Best action for empty board (depth 3): {action_map[best_action_empty]}")
    else:
        print("Empty board is terminal.")

    # Test with a full board (no moves)
    full_board_no_moves = np.array([
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6],
        [4,5,6,7]
    ], dtype=int)
    print("\nFull board (no moves possible usually):")
    print(2**full_board_no_moves)
    if not is_terminal(full_board_no_moves):
        best_action_full = expectimax_search(full_board_no_moves, depth=1)
        print(f"Best action for full board (depth 1): {action_map[best_action_full]}")
    else:
        print(f"Full board is terminal or no beneficial moves found: {legal_move_mask(full_board_no_moves)}")

