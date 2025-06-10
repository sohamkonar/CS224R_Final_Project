# CS224R Final Project: Reinforcement Learning Agents for 2048

This project explores the application of various reinforcement learning (RL) and search-based agents to play the game 2048. We implement and evaluate several agents, including a Deep Q-Network (DQN) agent and an Expectimax agent.

## Project Structure

The repository is organized as follows:

-   `main.py`: The main script to run the game with different agents.
-   `game.py`: Contains the core logic for the 2048 game environment.
-   `dqn.py`: Implements the Deep Q-Network (DQN) agent.
-   `expectimax.py`: Implements the Expectimax search agent.
-   `evaluation.py`: Contains scripts for evaluating and plotting the performance of the agents.
-   `requirements.txt`: Lists the project dependencies.
-   `README.md`: This file.

## Setup and Installation

To set up the project and install the necessary dependencies, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    This project requires Python 3. You can install all the necessary packages by running:
    ```bash
    pip install gymnasium numpy torch matplotlib tqdm pygame
    ```

## Usage

You can run the game with different agents using the `main.py` script.

### Running an Agent

The general command to run an agent is:

```bash
python main.py --agent <agent_name> --episodes <num_episodes>
```

-   `<agent_name>`: The agent to use. Available options are `random`, `expectimax`, and `dqn`.
-   `<num_episodes>`: The number of games to play.

For example, to run the Expectimax agent for 50 games:
```bash
python main.py --agent expectimax --episodes 50
```

### Rendering the Game

To watch the agent play, you can add the `--render` flag:

```bash
python main.py --agent dqn --episodes 10 --render
```

### Using a Trained DQN Model

To evaluate a pre-trained DQN model, you can specify the path to the model file using the `--model_path` argument:

```bash
python main.py --agent dqn --episodes 100 --model_path path/to/your/model.pth
```

## How to Reproduce Our Results

To reproduce the results from our report, follow these steps. The evaluation scripts will save plots and performance metrics to the project directory.

### 1. Train the DQN Agent

First, you need to train the DQN agent. The training process is integrated into the `dqn.py` script. To start training, run the `main.py` script with the `dqn` agent. The model will be saved periodically.

```bash
# Train the DQN agent for 10,000 episodes
python main.py --agent dqn --episodes 10000
```
This will save a `dqn_model.pth` file in the root directory.

### 2. Evaluate the Agents

Once the DQN model is trained, you can evaluate the performance of all agents. The `evaluation.py` script is designed for this purpose.

**Note:** You may need to modify `evaluation.py` to point to the correct model path and to run the desired number of evaluation episodes.

To run the evaluation:
```bash
python evaluation.py
```
This script will typically:
1.  Load the trained DQN model.
2.  Run each agent (`random`, `expectimax`, `dqn`) for a specified number of episodes.
3.  Collect statistics such as final scores, highest tile achieved, and win rates.
4.  Generate plots comparing the agents' performances.

### Example Evaluation Workflow

1.  **Train DQN:**
    ```bash
    python main.py --agent dqn --episodes 10000
    ```
2.  **Run Evaluation:**
    (Assuming `evaluation.py` is configured to run 1000 episodes for each agent and use `dqn_model.pth`)
    ```bash
    python evaluation.py
    ```
3.  **View Results:**
    Check the output directory for plots (e.g., `agent_performance.png`) and data files (e.g., `results.csv`).

## Agents

### Random Agent
A baseline agent that chooses a random valid move at each step.

### Expectimax Agent
A search-based agent that uses the Expectimax algorithm to find the optimal move. It explores the game tree up to a certain depth, maximizing the expected score.

### Deep Q-Network (DQN) Agent
A reinforcement learning agent that uses a deep neural network to approximate the Q-function. It learns a policy by interacting with the game environment and using experience replay and a target network for stable training.

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
