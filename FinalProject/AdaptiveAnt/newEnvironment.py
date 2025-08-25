import gymnasium as gym
import torch
from torch import nn
import numpy as np

class AntEnv:
    def __init__(self, render_mode=None, monitor=False, seed=42):
        self.env = gym.make("Ant-v5", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Set seeds properly
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.render_mode = render_mode
        self.monitor = monitor

    def reset(self):
        # Return the actual observation from reset
        observation, info = self.env.reset(seed=self.seed)
        return observation, info
    
    def step(self, action):
        # Ant-v5 returns 5 values: obs, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode:
            return self.env.render()

    def close(self):
        self.env.close()
    

import gymnasium as gym
import torch
import numpy as np
from typing import Optional, Tuple, Any

class TorchAntEnv:
    """
    Ant-v5 wrapper with torch-first I/O:
      - reset() -> (obs: torch.Tensor, info)
      - step(action: torch.Tensor) -> (obs: torch.Tensor, reward: torch.Tensor, done: bool, info)
    """
    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: int = 42,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.env = gym.make("Ant-v5", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.seed = seed
        self.render_mode = render_mode

        self._act_low_t  = torch.as_tensor(self.action_space.low,  device=self.device, dtype=self.dtype)
        self._act_high_t = torch.as_tensor(self.action_space.high, device=self.device, dtype=self.dtype)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _obs_to_torch(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, device=self.device, dtype=self.dtype)

    def _act_to_numpy(self, action: torch.Tensor) -> np.ndarray:
        a = action.detach()
        if a.dim() > 1:
            a = a.squeeze(0)  # ensure 1D for single env
        return a.clamp(self._act_low_t, self._act_high_t).cpu().numpy().astype(np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed: Optional[int] = None) -> Tuple[torch.Tensor, Any]:
        if seed is not None:
            self.seed = seed
        obs, info = self.env.reset(seed=self.seed)
        return self._obs_to_torch(obs), info

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = self.env.step(self._act_to_numpy(action))
        obs_t = self._obs_to_torch(obs)
        reward_t = torch.tensor(reward, device=self.device, dtype=self.dtype)
        done = bool(terminated or truncated)
        return obs_t, reward_t, done, truncated, info

    def render(self):
        if self.render_mode:
            return self.env.render()

    def close(self):
        self.env.close()

def test_env():
    import time

    env = AntEnv(render_mode="human")
    observation, info = env.reset()
    done = False
    steps = 0
    max_steps = 1000  # Add step limit to prevent infinite loop
    
    while not done and steps < max_steps:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.05)
        env.render()
        steps += 1
        
    env.close()

if __name__ == "__main__":
    test_env()
