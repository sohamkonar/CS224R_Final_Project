"""
game.py - Core 2048 game implementation and utilities
"""

import numpy as np
import gymnasium as gym
import gymnasium_2048  # registers the env

def obs_to_tensor(obs):
    """
    Convert the observation from the environment to a tensor format.
    
    Parameters:
    -----------
    obs : np.ndarray
        The observation from the environment
        
    Returns:
    --------
    torch.Tensor : The processed tensor
    """
    import torch
    # env returns (4,4,16); PyTorch wants (B,C,H,W)
    x = torch.from_numpy(obs).float().permute(2,0,1)     # (16,4,4)
    return x/1.0                                         # stay in [0,1]

def _line_can_move(line: np.ndarray) -> bool:
    """
    Given a 1-D view of 4 exponents, decide whether sliding the line
    to the *left* would change it (i.e. move or merge at least one tile).
    """
    # drop zeros to check merges, but keep them to check shifts
    non_zero = line[line != 0]

    # a gap anywhere ⇒ tile can slide into it
    if len(non_zero) < len(line):
        return True

    # adjacent equal exponents ⇒ those two tiles would fuse
    return np.any(non_zero[:-1] == non_zero[1:])

def legal_move_mask(board: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    board : (4, 4) uint8/uint16  – exponents of the current grid

    Returns
    -------
    mask  : (4,) bool – legal moves in the order [UP, RIGHT, DOWN, LEFT]
    """
    mask = np.zeros(4, dtype=bool)

    # --- UP --------------------------------------------------
    for col in range(4):
        if _line_can_move(board[:, col]):      # view column as line
            mask[0] = True
            break

    # --- DOWN (same as UP on the vertically flipped board) ---
    for col in range(4):
        if _line_can_move(board[::-1, col]):
            mask[2] = True
            break

    # --- LEFT -----------------------------------------------
    for row in range(4):
        if _line_can_move(board[row]):         # row already "facing left"
            mask[3] = True
            break

    # --- RIGHT (LEFT on the horizontally flipped board) -----
    for row in range(4):
        if _line_can_move(board[row, ::-1]):
            mask[1] = True
            break

    return mask

def create_env():
    """
    Create and return the 2048 environment
    
    Returns:
    --------
    gym.Env : The 2048 environment
    """
    return gym.make("gymnasium_2048/TwentyFortyEight-v0")

# Game mechanics for simulation in expectimax

def slide_and_merge_row(row):
    """Slide and merge tiles in a row to the left."""
    # Non-zero elements
    non_zero = row[row > 0]
    
    # If all zeros, nothing changes
    if len(non_zero) == 0:
        return row.copy(), 0
    
    # Initialize new row with zeros
    new_row = np.zeros_like(row)
    
    # Merge same tiles
    result_idx = 0
    reward = 0
    i = 0
    
    while i < len(non_zero):
        if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
            # Merge
            new_row[result_idx] = non_zero[i] + 1  # Exponent increases by 1
            reward += 2 ** new_row[result_idx]  # Score is the value of the merged tile
            i += 2
        else:
            # Just move
            new_row[result_idx] = non_zero[i]
            i += 1
        result_idx += 1
    
    return new_row, reward

def simulate_move(board: np.ndarray, action: int):
    """
    Simulate a move on the board without affecting the environment.
    
    Parameters:
    -----------
    board : np.ndarray
        The current board state
    action : int
        The action to take (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        
    Returns:
    --------
    Tuple[np.ndarray, float] : The next board state and the reward
    """
    # Clone the board to avoid modifying the original
    next_board = board.copy()
    total_reward = 0
    
    if action == 0:  # UP
        for col in range(4):
            # Extract column
            column = next_board[:, col]
            # Slide and merge
            new_column, reward = slide_and_merge_row(column)
            # Update board
            next_board[:, col] = new_column
            total_reward += reward
    
    elif action == 1:  # RIGHT
        for row in range(4):
            # Extract row and reverse it
            flipped_row = next_board[row, ::-1]
            # Slide and merge
            new_row, reward = slide_and_merge_row(flipped_row)
            # Update board (reverse back)
            next_board[row, :] = new_row[::-1]
            total_reward += reward
    
    elif action == 2:  # DOWN
        for col in range(4):
            # Extract column and reverse it
            flipped_column = next_board[:, col][::-1]
            # Slide and merge
            new_column, reward = slide_and_merge_row(flipped_column)
            # Update board (reverse back)
            next_board[:, col] = new_column[::-1]
            total_reward += reward
    
    elif action == 3:  # LEFT
        for row in range(4):
            # Extract row
            current_row = next_board[row, :]
            # Slide and merge
            new_row, reward = slide_and_merge_row(current_row)
            # Update board
            next_board[row, :] = new_row
            total_reward += reward
    
    return next_board, total_reward
