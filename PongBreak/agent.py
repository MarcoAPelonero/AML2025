# agent.py

"""
REINFORCE + baseline (critic) that works with a Gymnasium SyncVectorEnv.

Key points
----------
* One big loss across *all* sub-env trajectories → single backward / step
  ► avoids the "in-place modified variable" error you saw.
* Buffers are lists-of-lists, one inner list per env.
* Normalize raw observations to roughly [-1, 1] before the nets.
"""

from __future__ import annotations
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List

################################################################################
#                               NETWORKS
################################################################################
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, h1: int = 64, h2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
            nn.Linear(h2, action_dim)               # logits
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


################################################################################
#                               AGENT
################################################################################
class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 1e-3,
        lr_critic: float = 5e-4,
        gamma: float = 0.95,
        grad_clip: float | None = 1.0,
        num_envs: int = 1,
        device: torch.device | None = None,
    ):
        self.gamma      = gamma
        self.grad_clip  = grad_clip
        self.num_envs   = num_envs
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNet(state_dim, action_dim, hidden_dim, hidden_dim // 2).to(self.device)
        self.value  = ValueNet(state_dim, hidden_dim).to(self.device)
        self.optim  = optim.Adam([
            {"params": self.policy.parameters(), "lr": lr_actor},
            {"params": self.value.parameters(),  "lr": lr_critic}
        ])
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=100, gamma=0.9)

        # Buffers: one sub‐list per env
        self.log_probs : List[List[torch.Tensor]] = [[] for _ in range(num_envs)]
        self.values    : List[List[torch.Tensor]] = [[] for _ in range(num_envs)]  # raw value estimates
        self.rewards   : List[List[float]]        = [[] for _ in range(num_envs)]

    # --------------------------------------------------------------------- #
    # Interaction helpers
    # --------------------------------------------------------------------- #
    def _norm_state(self, s: np.ndarray) -> torch.Tensor:
        """Cheap hand-tuned normalization for the 6-element Breakout-Pong state."""
        denom = np.array([400., 600., 7., 7., 320., 10.], dtype=np.float32)
        t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        return t / torch.as_tensor(denom, device=self.device)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Exactly as before: 
        Normalizes state, runs it through policy and value nets, samples an action,
        stores log_probs and values in buffers, returns the action to the environment.
        """
        single = state.ndim == 1
        s = self._norm_state(state)
        if single:
            s = s.unsqueeze(0)

        logits = self.policy(s)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_p  = dist.log_prob(action)
        val    = self.value(s)  # no detach

        for i in range(s.shape[0]):
            self.log_probs[i].append(log_p[i])
            self.values[i].append(val[i])

        return action.cpu().numpy() if not single else int(action.item())

    def store_reward(self, reward: np.ndarray):
        for i, r in enumerate(reward):
            self.rewards[i].append(float(r))

    # --------------------------------------------------------------------- #
    # NEW METHOD: get_action_and_activations
    # --------------------------------------------------------------------- #
    def get_action_and_activations(self, state: np.ndarray):
        """
        Runs a single forward pass through the policy network, but also returns
        the *pre‐ReLU* activations of each linear layer.  
        Returns:
            action (int or np.ndarray), 
            activations (List[torch.Tensor]): [pre_relu_h1, pre_relu_h2, logits].
        It also stores log_probs and values exactly as select_action does.
        """
        single = state.ndim == 1
        s = self._norm_state(state)
        if single:
            s = s.unsqueeze(0)

        # Sequential indices in self.policy.net:
        #   0: LayerNorm
        #   1: Linear → h1
        #   2: ReLU
        #   3: Linear → h2
        #   4: ReLU
        #   5: Linear → out (logits)

        ln_out     = self.policy.net[0](s)                    # LayerNorm output
        preact_h1  = self.policy.net[1](ln_out)               # Linear1 output (before ReLU)
        out_h1     = F.relu(preact_h1)                        
        preact_h2  = self.policy.net[3](out_h1)                # Linear2 output (before ReLU)
        out_h2     = F.relu(preact_h2)
        logits     = self.policy.net[5](out_h2)                # final linear logits

        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_p  = dist.log_prob(action)
        val    = self.value(s)

        # Store exactly like select_action
        for i in range(s.shape[0]):
            self.log_probs[i].append(log_p[i])
            self.values[i].append(val[i])

        # Convert action to int or numpy array
        action_to_return = action.cpu().numpy() if not single else int(action.item())

        # Return action plus a list of 3 activation tensors (all moved to CPU)
        return action_to_return, [
            preact_h1.detach().cpu(), 
            preact_h2.detach().cpu(), 
            logits.detach().cpu()
        ]

    # --------------------------------------------------------------------- #
    # Learning (unchanged)
    # --------------------------------------------------------------------- #
    def finish_rollout(self):
        total_loss = torch.tensor(0.0, device=self.device)

        for i in range(self.num_envs):
            if len(self.rewards[i]) == 0:
                continue

            # Compute discounted returns
            R, returns = 0.0, []
            for r in reversed(self.rewards[i]):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

            vals   = torch.stack(self.values[i])   # these require gradients
            log_ps = torch.stack(self.log_probs[i])
            adv    = returns - vals.detach()

            actor_loss  = -(log_ps * adv).sum()
            critic_loss = F.mse_loss(vals, returns)
            total_loss  = total_loss + actor_loss + critic_loss

        if total_loss.item() == 0.0:
            return

        self.optim.zero_grad()
        total_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.value.parameters(),  self.grad_clip)
        self.optim.step()

        # Clear buffers
        for buf in (self.rewards, self.values, self.log_probs):
            for sub in buf:
                sub.clear()

    # --------------------------------------------------------------------- #
    # Checkpoint helpers (unchanged)
    # --------------------------------------------------------------------- #
    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "value":  self.value.state_dict()
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if "policy" in ckpt:
            self.policy.load_state_dict(ckpt["policy"])
            self.value.load_state_dict(ckpt["value"])
        else:
            # If the file only contains policy weights
            self.policy.load_state_dict(ckpt)
