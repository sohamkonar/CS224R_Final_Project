# CS224R Final Project - 2048 RL Agents

This project implements and evaluates different Reinforcement Learning agents for the game 2048.

## Project Structure

- `game.py`: Core 2048 game logic and utilities.
- `dqn.py`: Deep Q-Network (DQN) agent implementation, including training and prioritized experience replay.
- `expectimax.py`: Expectimax search algorithm implementation.
- `evaluation.py`: Utilities for evaluating the performance of different agents.
- `main.py`: Main script to run training and evaluation for the agents.
- `requirements.txt`: Python dependencies for the project.
- `checkpoints/`: Directory where trained model checkpoints will be saved by default.
- `runs/`: Directory where TensorBoard logs will be saved by default.

## Setup

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `main.py` script is used to train and evaluate agents. 

### DQN Agent

**Train DQN Agent:**
```bash
python main.py --agent dqn --mode train --episodes 20000 --use_per True
```
- `--use_per False` can be used to train with standard experience replay.
- Checkpoints will be saved in the `checkpoints/` directory.
- TensorBoard logs will be saved in the `runs/dqn/` directory. To view them:
  ```bash
  tensorboard --logdir runs/dqn
  ```

**Evaluate DQN Agent:**
```bash
python main.py --agent dqn --mode evaluate --episodes 100 --load_model checkpoints/best.pt
```
- Replace `checkpoints/best.pt` with the path to your desired model checkpoint.

### Expectimax Agent

**Evaluate Expectimax Agent:**
```bash
python main.py --agent expectimax --mode evaluate --episodes 50 --expectimax_depth 3
```
- Adjust `--episodes` and `--expectimax_depth` as needed. Higher depth significantly increases computation time.

### Command-line Arguments

Run `python main.py --help` to see all available command-line options:

```
usage: main.py [-h] [--agent {dqn,expectimax}] [--mode {train,evaluate}]
               [--episodes EPISODES] [--load_model LOAD_MODEL]
               [--use_per USE_PER] [--expectimax_depth EXPECTIMAX_DEPTH]
               [--batch_size BATCH_SIZE] [--gamma GAMMA]
               [--target_update TARGET_UPDATE]

Run 2048 RL agents.

options:
  -h, --help            show this help message and exit
  --agent {dqn,expectimax}
                        Agent to use or evaluate. (default: dqn)
  --mode {train,evaluate}
                        Mode to run: 'train' or 'evaluate'. (default: train)
  --episodes EPISODES   Number of episodes for training or evaluation. (default: 10000)
  --load_model LOAD_MODEL
                        Path to a pre-trained DQN model checkpoint to load. (default: None)
  --use_per USE_PER     Whether to use Prioritized Experience Replay for DQN. (default: True)
  --expectimax_depth EXPECTIMAX_DEPTH
                        Depth for Expectimax search during evaluation. (default: 3)
  --batch_size BATCH_SIZE
                        Batch size for DQN training. (default: 64)
  --gamma GAMMA         Discount factor for DQN. (default: 0.99)
  --target_update TARGET_UPDATE
                        Target network update frequency for DQN. (default: 1000)
```
