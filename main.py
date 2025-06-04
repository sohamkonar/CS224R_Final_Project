"""
main.py - Main script to run 2048 RL agents and evaluations.
"""

import argparse
import torch

from game import create_env
from dqn import DQNAgent, DQN as DQNModel # Renamed to avoid conflict with DQNAgent class
from evaluation import evaluate_dqn_agent, evaluate_expectimax_agent

def main():
    parser = argparse.ArgumentParser(description="Run 2048 RL agents.")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "expectimax"], 
                        help="Agent to use or evaluate.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"], 
                        help="Mode to run: 'train' or 'evaluate'.")
    parser.add_argument("--episodes", type=int, default=10000, 
                        help="Number of episodes for training or evaluation.")
    parser.add_argument("--load_model", type=str, default=None, 
                        help="Path to a pre-trained DQN model checkpoint to load.")
    parser.add_argument("--use_per", type=bool, default=True, 
                        help="Whether to use Prioritized Experience Replay for DQN.")
    parser.add_argument("--expectimax_depth", type=int, default=3, 
                        help="Depth for Expectimax search during evaluation.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DQN training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for DQN.")
    parser.add_argument("--target_update", type=int, default=1000, help="Target network update frequency for DQN.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = create_env()

    if args.agent == "dqn":
        dqn_agent = DQNAgent(use_per=args.use_per)
        if args.load_model:
            print(f"Loading DQN model from: {args.load_model}")
            try:
                dqn_agent.load_checkpoint(args.load_model)
            except FileNotFoundError:
                print(f"Error: Model file {args.load_model} not found. Starting from scratch or check path.")
                # Optionally exit or start fresh
                # return 
            except Exception as e:
                print(f"Error loading model: {e}. Starting from scratch or check path.")
                # return

        if args.mode == "train":
            print(f"Starting DQN training for {args.episodes} episodes...")
            dqn_agent.train(num_episodes=args.episodes, 
                            batch_size=args.batch_size, 
                            gamma=args.gamma, 
                            target_update_freq=args.target_update)
            print("DQN training finished. Model saved in checkpoints directory.")
        
        elif args.mode == "evaluate":
            if not args.load_model and not dqn_agent.best_reward > -float('inf'): # Check if a model was trained in this session
                print("No DQN model specified for evaluation and no model trained in this session. Please train a model or use --load_model.")
                return
            
            print(f"Evaluating DQN agent for {args.episodes} episodes...")
            # Ensure the policy_net is on the correct device for evaluation
            policy_to_eval = dqn_agent.policy_net.to(device)
            metrics = evaluate_dqn_agent(policy_to_eval, env, n_episodes=args.episodes, device=device)
            print("\nDQN Evaluation Metrics:")
            for key, value in metrics.items():
                if not isinstance(value, list): # Don't print raw lists
                    print(f"  {key}: {value}")
            print(f"  Max tile achieved: {metrics['max_tile_value']}")

    elif args.agent == "expectimax":
        if args.mode == "train":
            print("Expectimax agent does not require training. Use 'evaluate' mode.")
            return
        
        elif args.mode == "evaluate":
            print(f"Evaluating Expectimax agent for {args.episodes} episodes with depth {args.expectimax_depth}...")
            metrics = evaluate_expectimax_agent(env, n_episodes=args.episodes, depth=args.expectimax_depth)
            print("\nExpectimax Evaluation Metrics:")
            for key, value in metrics.items():
                if not isinstance(value, list): # Don't print raw lists
                    print(f"  {key}: {value}")
            print(f"  Max tile achieved: {metrics['max_tile_value']}")

    env.close()

if __name__ == "__main__":
    main()
