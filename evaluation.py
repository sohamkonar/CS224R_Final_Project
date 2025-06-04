"""
evaluation.py - Functions to evaluate agent performance in the 2048 game.
"""

import numpy as np
import torch
from tqdm import tqdm

from game import obs_to_tensor, legal_move_mask, create_env # For DQN evaluation
from expectimax import expectimax_search # For Expectimax evaluation
# DQN class might be needed if loading a policy_net directly without the DQNAgent class
# from dqn import DQN 

def evaluate_dqn_agent(policy_net, env, n_episodes=100, max_steps=10_000, device="cpu"):
    """
    Evaluate a DQN agent.

    Parameters:
    -----------
    policy_net : torch.nn.Module
        The trained policy network.
    env : gym.Env
        The 2048 environment.
    n_episodes : int
        Number of episodes to run.
    max_steps : int
        Maximum steps per episode.
    device : str
        Device to run the policy network on ('cpu' or 'cuda').

    Returns:
    --------
    dict : A dictionary containing evaluation metrics.
    """
    scores = []
    best_tiles = []
    lengths = []
    policy_net.eval() # Set the network to evaluation mode

    print(f"Evaluating DQN agent for {n_episodes} episodes...")
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        ep_score = 0.0
        step = 0

        while not done and step < max_steps:
            x = obs_to_tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(x).squeeze(0).cpu().numpy()
            
            current_board_mask = legal_move_mask(env.unwrapped.board)
            if not current_board_mask.any():
                break # No legal moves
            
            q_values[~current_board_mask] = -np.inf # Mask illegal moves
            action = int(np.argmax(q_values))
            
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_score += reward
            step += 1
        
        scores.append(ep_score)
        best_tiles.append(env.unwrapped.board.max()) # Max exponent
        lengths.append(step)
    
    policy_net.train() # Set back to training mode if it's used for further training

    return {
        "mean_score": float(np.mean(scores)) if scores else 0,
        "median_score": float(np.median(scores)) if scores else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0,
        "max_tile_exponent": int(np.max(best_tiles)) if best_tiles else 0,
        "max_tile_value": int(2**np.max(best_tiles)) if best_tiles and np.max(best_tiles) > 0 else 0,
        "raw_scores": scores,
        "raw_lengths": lengths,
        "raw_best_tiles_exponent": best_tiles
    }

def evaluate_expectimax_agent(env, n_episodes=10, max_steps=10_000, depth=3):
    """
    Evaluate the 2048 game using expectimax search.

    Parameters:
    -----------
    env : gym.Env
        The 2048 environment.
    n_episodes : int
        Number of episodes to evaluate.
    max_steps : int
        Maximum steps per episode.
    depth : int
        Depth for the expectimax search.

    Returns:
    --------
    dict : Evaluation metrics.
    """
    scores = []
    best_tiles = []
    lengths = []

    print(f"Evaluating Expectimax agent for {n_episodes} episodes (depth={depth})...")
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset() # obs is not directly used by expectimax, but good to keep env state
        done = False
        ep_score = 0.0
        step = 0
        current_board = env.unwrapped.board

        while not done and step < max_steps:
            action = expectimax_search(current_board, depth=depth)
            
            if action == -1:  # No legal moves or error in search
                break
            
            obs, reward, term, trunc, _ = env.step(action)
            current_board = env.unwrapped.board # Update board after step
            done = term or trunc
            ep_score += reward
            step += 1
        
        scores.append(ep_score)
        best_tiles.append(current_board.max()) # Max exponent
        lengths.append(step)

    return {
        "mean_score": float(np.mean(scores)) if scores else 0,
        "median_score": float(np.median(scores)) if scores else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0,
        "max_tile_exponent": int(np.max(best_tiles)) if best_tiles else 0,
        "max_tile_value": int(2**np.max(best_tiles)) if best_tiles and np.max(best_tiles) > 0 else 0,
        "raw_scores": scores,
        "raw_lengths": lengths,
        "raw_best_tiles_exponent": best_tiles
    }

if __name__ == '__main__':
    # Example Usage (requires agents and environment)
    print("Evaluation functions defined.")
    print("To run evaluation, you'll need to instantiate an environment and an agent.")
    
    # Example for DQN (assuming a trained model `my_dqn_model.pt` exists)
    # from dqn import DQN # Assuming DQN class is defined in dqn.py
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # eval_env = create_env()
    # dqn_policy_net = DQN().to(device)
    # try:
    #     checkpoint = torch.load("checkpoints/best.pt", map_location=device) # Or your specific model path
    #     dqn_policy_net.load_state_dict(checkpoint['policy'])
    #     print("Loaded DQN model for evaluation.")
    #     dqn_metrics = evaluate_dqn_agent(dqn_policy_net, eval_env, n_episodes=10, device=device)
    #     print("\nDQN Evaluation Metrics:")
    #     for key, value in dqn_metrics.items():
    #         if not key.startswith("raw_"):
    #             print(f"  {key}: {value}")
    # except FileNotFoundError:
    #     print("DQN model checkpoint not found. Skipping DQN evaluation example.")
    # except Exception as e:
    #     print(f"Error loading DQN model: {e}")

    # Example for Expectimax
    # eval_env_exp = create_env()
    # print("\nRunning Expectimax evaluation example...")
    # expectimax_metrics = evaluate_expectimax_agent(eval_env_exp, n_episodes=2, depth=2) # Small n_episodes/depth for quick test
    # print("\nExpectimax Evaluation Metrics:")
    # for key, value in expectimax_metrics.items():
    #     if not key.startswith("raw_"):
    #         print(f"  {key}: {value}")
