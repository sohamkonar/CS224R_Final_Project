"""
dqn.py - Deep Q-Network (DQN) implementation for 2048
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
import json
import datetime as dt
from pathlib import Path
from tqdm import tqdm

from game import obs_to_tensor, legal_move_mask, create_env # Assuming game.py is in the same directory

# For PER
class _SumTree:
    """Binary tree datastructure for O(log N) priority ops."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)  # 1-indexed heap
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, diff: float):
        parent = idx // 2
        self.tree[parent] += diff
        if parent != 1:
            self._propagate(parent, diff)

    def update(self, idx: int, new_p: float):
        diff = new_p - self.tree[idx]
        self.tree[idx] = new_p
        self._propagate(idx, diff)

    def add(self, p: float, data: tuple):
        leaf = self.write + self.capacity
        self.data[self.write] = data
        self.update(leaf, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def total_p(self) -> float:
        return self.tree[1]

    def get_leaf(self, v: float):
        idx = 1
        while idx < self.capacity:                 # descend to leaf
            left, right = 2 * idx, 2 * idx + 1
            idx = left if v <= self.tree[left] else right
            if v > self.tree[left]:
                v -= self.tree[left]
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

class DQN(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=2), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*2*2, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):                # x is (B,16,4,4)
        return self.net(x)

Transition = namedtuple('Transition',
                        ('state','action','reward','next_state','done'))

class ReplayMemory:
    def __init__(self, capacity=50_000):
        self.memory = deque(maxlen=capacity)
    def push(self,*args):  self.memory.append(Transition(*args))
    def sample(self,batch_size): return random.sample(self.memory,batch_size)
    def __len__(self):     return len(self.memory)

class PrioritizedReplayBuffer:
    """
    β-annealing and α-exponent parameters follow Schaul et al.
    get_batch() returns (batch, IS_weights, index_arr) for later priority update.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 250_000, eps: float = 1e-5):
        self.tree = _SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 1  # global step counter
        self.max_priority = 1.0

    def push(self, transition: tuple):
        self.tree.add(self.max_priority, transition)

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) *
                   self.frame / self.beta_frames)

    def sample(self, batch_size: int):
        batch, idxs, priorities = [], [], []
        while len(batch) < batch_size:
            p = random.random() * self.tree.total_p()
            idx, prio, data = self.tree.get_leaf(p)
            if data is None:
                continue
            batch.append(data)
            idxs.append(idx)
            priorities.append(prio)

        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )

        states      = torch.as_tensor(states, dtype=torch.float32)
        actions     = torch.as_tensor(actions, dtype=torch.long)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones       = torch.as_tensor(dones, dtype=torch.bool)

        probs = torch.tensor(priorities, dtype=torch.float32) / self.tree.total_p()
        beta  = self._beta_by_frame()
        self.frame += 1
        weights = (len(self) * probs).pow(-beta)
        weights /= weights.max()
        weights = weights.to(torch.float32)

        return states, actions, rewards, next_states, dones, weights, idxs

    def update_priorities(self, idxs, td_errors):
        new_p = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(idxs, new_p):
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return self.tree.size

class DQNAgent:
    def __init__(self, checkpoint_dir="checkpoints", log_dir="runs/dqn", use_per=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-5)
        
        if use_per:
            self.memory = PrioritizedReplayBuffer(capacity=100_000, alpha=0.6, beta_start=0.4, beta_frames=250_000)
        else:
            self.memory = ReplayMemory(capacity=50_000)
        self.use_per = use_per

        self.loss_fn = nn.SmoothL1Loss(reduction='none' if use_per else 'mean') # Huber loss

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metrics_file = self.checkpoint_dir / "metrics.json"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_reward = -float("inf")
        self.save_every = 250
        self.keep_last_k = 3

    def save_ckpt(self, name, episode, ep_reward, global_step):
        ckpt = {
            "episode":    episode,
            "global_step": global_step,
            "policy":     self.policy_net.state_dict(),
            "target":     self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "ep_reward":  ep_reward,
            "saved_at":   dt.datetime.utcnow().isoformat() + "Z",
        }
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(ckpt, path)
        print(f"✓ saved checkpoint → {path}")

        ckpts = sorted(self.checkpoint_dir.glob("ep_*.pt"), key=os.path.getmtime)
        for p in ckpts[:-self.keep_last_k]:
            p.unlink()

    def save_metrics(self, record):
        all_rows = []
        if self.metrics_file.exists():
            all_rows = json.loads(self.metrics_file.read_text())
        all_rows.append(record)
        self.metrics_file.write_text(json.dumps(all_rows, indent=2))

    def train(self, num_episodes=10_000, batch_size=64, gamma=0.99, target_update_freq=1_000):
        env = create_env()
        epsilon_start, epsilon_end, decay_steps = 1.0, 0.05, 200_000
        global_step = 0

        for episode in tqdm(range(num_episodes)):
            obs, _ = env.reset()
            state = obs_to_tensor(obs).unsqueeze(0).to(self.device)
            done = False
            ep_reward = 0.0

            while not done:
                eps = max(epsilon_end, epsilon_start - global_step / decay_steps)
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        qvals = self.policy_net(state).squeeze(0)
                    current_board_mask = legal_move_mask(env.unwrapped.board)
                    qvals[~current_board_mask] = torch.finfo(qvals.dtype).min
                    action = int(qvals.argmax().item())

                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                next_s = obs_to_tensor(next_obs).unsqueeze(0).to(self.device)
                r = torch.tensor([reward], dtype=torch.float32, device=self.device)

                self.memory.push((
                    state.squeeze(0).cpu(),
                    action,
                    float(reward),
                    next_s.squeeze(0).cpu(),
                    done
                ))

                state = next_s
                ep_reward += reward
                global_step += 1

                if len(self.memory) > batch_size: # Start training once buffer has enough samples
                    if self.use_per:
                        s_batch, a_batch, r_batch, ns_batch, d_batch, IS_w, idxs = self.memory.sample(batch_size)
                    else:
                        transitions = self.memory.sample(batch_size)
                        batch_data = Transition(*zip(*transitions))
                        s_batch = torch.stack([s.clone().detach() for s in batch_data.state])
                        a_batch = torch.tensor(batch_data.action, dtype=torch.long)
                        r_batch = torch.tensor(batch_data.reward, dtype=torch.float32)
                        ns_batch = torch.stack([s.clone().detach() for s in batch_data.next_state])
                        d_batch = torch.tensor(batch_data.done, dtype=torch.bool)
                        IS_w = None # Not used without PER

                    s_batch = s_batch.to(self.device)
                    a_batch = a_batch.to(self.device).unsqueeze(1)
                    r_batch = r_batch.to(self.device)
                    ns_batch = ns_batch.to(self.device)
                    d_batch = d_batch.to(self.device)
                    if IS_w is not None: IS_w = IS_w.to(self.device)

                    q_sa = self.policy_net(s_batch).gather(1, a_batch).squeeze(1)
                    
                    with torch.no_grad():
                        # Double DQN: select action with policy_net, evaluate with target_net
                        next_actions = self.policy_net(ns_batch).max(1)[1].unsqueeze(1)
                        target_q_sa_next = self.target_net(ns_batch).gather(1, next_actions).squeeze(1)
                        target = r_batch + (~d_batch) * gamma * target_q_sa_next

                    if self.use_per:
                        td_err = target - q_sa
                        loss = (IS_w * self.loss_fn(q_sa, target)).mean()
                        self.memory.update_priorities(idxs, td_err.detach().abs().cpu().numpy())
                    else:
                        loss = self.loss_fn(q_sa, target)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Optional gradient clipping
                    self.optimizer.step()
                    self.writer.add_scalar("loss/td_error", loss.item(), global_step)

                if global_step % target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.writer.add_scalar("reward/episode_reward", ep_reward, episode)
            metrics_record = {"episode": episode, "reward": ep_reward, "global_step": global_step, "epsilon": eps}
            self.save_metrics(metrics_record)

            if episode % self.save_every == 0 and episode > 0:
                self.save_ckpt(f"ep_{episode}", episode, ep_reward, global_step)

            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                self.save_ckpt("best", episode, ep_reward, global_step)
                print(f"New best reward: {ep_reward} at episode {episode}")

        self.writer.close()
        print("Training finished.")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded checkpoint from {path}")
        return ckpt['episode'], ckpt['global_step']

if __name__ == '__main__':
    # Example usage:
    agent = DQNAgent(use_per=True) # Or use_per=False for standard replay
    # To start training:
    # agent.train(num_episodes=10000)
    
    # To load a checkpoint and continue training or for evaluation:
    # episode_start, global_step_start = agent.load_checkpoint("checkpoints/best.pt")
    # agent.train(num_episodes=10000, batch_size=64, gamma=0.99, target_update_freq=1000)
    print("DQNAgent class defined. You can instantiate and train it.")
