# fine_tuning.py

"""
Fine-tuning script for BreakoutPongEnv using a pretrained policy.
All native environment rewards are ignored. Instead, we shape rewards as follows:

  1) At each paddle hit, count how many bricks were destroyed since the previous paddle hit:
       - 0 bricks → penalty of -1.0
       - 1 brick  → reward of 0.0
       - ≥2 bricks → reward of +1.0

  2) At each step, penalize paddle oscillation: whenever the paddle’s horizontal velocity
     flips sign (beyond a small threshold), apply −0.05.

  3) At the very end of each episode (when the ball is lost or all bricks are cleared),
     grant a time‐efficiency bonus: 
       time_bonus = 0.5 * max(0, (1000 – steps_taken) / 1000)

We load an existing “policy.pth” into the agent’s policy network, train at low learning rates,
and decay them every 50 episodes by a factor of 0.9.
"""

from __future__ import annotations
import sys
import numpy as np
import torch
from torch import optim
from pathlib import Path
from tqdm import trange
import gymnasium as gym

from agent import REINFORCEAgent
from game import BreakoutPongEnv


def make_env(render: bool):
    """
    Returns a function that creates a BreakoutPongEnv instance.
    """
    return lambda: BreakoutPongEnv(
        render_mode="human" if render else None,
        max_episode_steps=None
    )


def train_fine_tuned(
    num_episodes: int = 1000,
    save_path: str | Path = "fine_tuned_policy.pth",
    render: bool = False,
    seed: int | None = None,
    num_envs: int = 4,
):
    """
    Fine-tunes a pretrained policy on BreakoutPongEnv with custom reward shaping:
      - Ignore the environment’s native reward.
      - Brick‐based reward at paddle hits.
      - Oscillation penalty at each step.
      - Time‐efficiency bonus at episode end.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create a vectorized environment with num_envs parallel instances
    env = gym.vector.SyncVectorEnv([make_env(render) for _ in range(num_envs)])

    # Determine state_dim and action_dim from a single sub-environment
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n

    # Instantiate the REINFORCE agent with low learning rates
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr_actor=8e-4,      # Low learning rate for policy
        lr_critic=5e-5,     # Low learning rate for critic
        gamma=0.95,
        grad_clip=1.0,
        num_envs=num_envs,
        device=None
    )

    # Load pretrained policy weights into agent.policy
    agent.load("policy.pth")

    # Replace scheduler to decay every 50 episodes by 0.9
    agent.scheduler = optim.lr_scheduler.StepLR(agent.optim, step_size=50, gamma=0.9)

    reward_hist = []
    best_avg_reward = float('-inf')
    best_weights = None
    progress_bar = trange(1, num_episodes + 1, desc="Fine-Tuning", unit="ep")

    # Reset all sub-environments once before starting
    obs, _ = env.reset()

    # --------------------------------------------------------------------------
    # Trackers shared across steps:
    # - prev_ball_vy / prev_ball_y: for detecting brick hits and paddle hits
    # - prev_paddle_vx: for detecting oscillation penalties
    # - bricks_since_last_paddle: counts bricks destroyed since last paddle hit
    # --------------------------------------------------------------------------
    prev_ball_vy = obs[:, 3].copy()
    prev_ball_y  = obs[:, 1].copy()
    prev_paddle_vx = np.zeros(num_envs, dtype=np.float32)
    bricks_since_last_paddle = np.zeros(num_envs, dtype=int)

    # Constants for region thresholds
    BRICK_TOP_Y = 50 + 3 * 20   # bricks occupy y ∈ [50, 110]
    PADDLE_HIT_Y = 580.0        # paddle is at y ≈ 590; ball radius = 5

    try:
        for ep in progress_bar:
            # Per-episode buffers
            ep_return = np.zeros(num_envs, dtype=np.float32)
            done_mask = np.zeros(num_envs, dtype=bool)
            steps_taken = np.zeros(num_envs, dtype=int)
            paddle_vel_changes = np.zeros(num_envs, dtype=int)
            bricks_broken_total = np.zeros(num_envs, dtype=int)  # Track total bricks broken

            # Reset per-episode trackers
            prev_ball_vy = obs[:, 3].copy()
            prev_ball_y  = obs[:, 1].copy()
            prev_paddle_vx[:] = 0.0

            while not done_mask.all():
                action = agent.select_action(obs)
                next_obs, _, terminated, truncated, _ = env.step(action)
                just_done = np.logical_or(terminated, truncated)

                next_ball_vy = next_obs[:, 3]
                next_ball_y  = next_obs[:, 1]

                # --- Brick hit detection and reward ---
                brick_hit = np.zeros(num_envs, dtype=bool)
                brick_reward = np.zeros(num_envs, dtype=np.float32)
                for i in range(num_envs):
                    if not done_mask[i]:
                        if (
                            prev_ball_vy[i] < 0.0
                            and next_ball_vy[i] > 0.0
                            and prev_ball_y[i] <= BRICK_TOP_Y + 5
                        ):
                            brick_hit[i] = True
                            bricks_broken_total[i] += 1
                            # Reward for breaking bricks quickly
                            brick_reward[i] = 1.0 * (1.0 - steps_taken[i] / 1000.0)

                # --- End episode if 20 bricks are broken ---
                for i in range(num_envs):
                    if not done_mask[i] and bricks_broken_total[i] >= 20:
                        done_mask[i] = True
                        just_done[i] = True

                # --- Paddle oscillation penalty (stability reward) ---
                current_paddle_vx = obs[:, 5].copy()
                osc_penalty = np.zeros(num_envs, dtype=np.float32)
                sign_flip = (
                    (np.sign(current_paddle_vx) != np.sign(prev_paddle_vx))
                    & (np.abs(current_paddle_vx) > 0.1)
                )
                for i in range(num_envs):
                    if sign_flip[i] and not done_mask[i]:
                        paddle_vel_changes[i] += 1
                        osc_penalty[i] = -0.05

                # --- Centering penalty when ball is going up ---
                center_penalty = np.zeros(num_envs, dtype=np.float32)
                for i in range(num_envs):
                    if not done_mask[i] and next_ball_vy[i] < 0.0:
                        paddle_cx = obs[i, 4] + 40  # paddle_x + paddle_width/2
                        dist = abs(obs[i, 0] - paddle_cx)  # |ball_x - paddle center|
                        center_penalty[i] = -0.001 * dist  # scale as needed

                # --- Death penalty (harsher) ---
                death_penalty = np.zeros(num_envs, dtype=np.float32)
                for i in range(num_envs):
                    if terminated[i] and not truncated[i] and not done_mask[i]:
                        # Ball lost (death)
                        death_penalty[i] = -2.0

                # --- Shaped reward for this step ---
                shaped_reward = brick_reward + osc_penalty + center_penalty + death_penalty

                ep_return += shaped_reward * (~done_mask)
                agent.store_reward(shaped_reward)
                steps_taken += (~done_mask).astype(int)
                done_mask |= just_done

                if render:
                    env.render()

                prev_ball_vy = next_ball_vy.copy()
                prev_ball_y  = next_ball_y.copy()
                prev_paddle_vx = current_paddle_vx.copy()
                obs = next_obs

                if np.any(steps_taken > 10000):
                    break

            # --- No time bonus; episode ends after 20 bricks or death ---

            agent.finish_rollout()
            agent.scheduler.step()

            avg_return = ep_return.mean().item()
            avg_last_10 = float(np.mean(reward_hist[-10:])) if len(reward_hist) >= 10 else avg_return
            reward_hist.append(avg_return)

            actor_lr = agent.optim.param_groups[0]["lr"]
            critic_lr = agent.optim.param_groups[1]["lr"]
            progress_bar.set_postfix(
                avg10=f"{avg_last_10:.3f}",
                actor_lr=f"{actor_lr:.6f}",
                critic_lr=f"{critic_lr:.6f}"
            )

            if ep % (num_episodes // 10) == 0:
                if avg_last_10 > best_avg_reward:
                    best_avg_reward = avg_last_10
                    best_weights = agent.policy.state_dict()

        if best_weights is not None:
            torch.save(best_weights, str(save_path))
        else:
            agent.save(str(save_path))

        env.close()
        print(f"\nFine-tuning complete – best policy saved to '{save_path}'.")

    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user.")
        if best_weights is not None:
            torch.save(best_weights, str(save_path))
            print(f"Best policy saved to '{save_path}'.")
        else:
            agent.save(str(save_path))
            print(f"Policy saved to '{save_path}'.")
        env.close()


if __name__ == "__main__":
    # Usage: python fine_tuning.py [num_episodes] [save_path]
    n_eps = 1000
    path = "fine_tuned_policy.pth"
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        n_eps = int(sys.argv[1])
    if len(sys.argv) >= 3:
        path = sys.argv[2]

    train_fine_tuned(num_episodes=n_eps, save_path=path, num_envs=4)
