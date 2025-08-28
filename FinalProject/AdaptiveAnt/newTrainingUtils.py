import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from newEnvironment import TorchAntEnv
from time import perf_counter

def Step(agent, env, observation, stochastic: bool = True):
    """
    Torch-first single step using stochastic tanh-Gaussian policy.
    Returns: next_obs, reward, done, info, action, value, log_prob
    """
    device = next(agent.parameters()).device
    obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_t, logp_t, v_t = agent.act(obs_t.unsqueeze(0), stochastic=stochastic)
    action_t = action_t.squeeze(0)         # [act_dim]
    v_t      = v_t.squeeze(0)              # []
    if logp_t is not None:
        logp_t = logp_t.squeeze(0)
    next_obs, reward, terminated, truncated, info = env.step(action_t)
    done = bool(terminated or truncated)
    return next_obs, float(reward), done, info, action_t.detach().cpu(), v_t.detach().cpu(), (None if logp_t is None else logp_t.detach().cpu())

def Rollout(agent, env, horizon: int = 2048, stochastic: bool = True, bar = False):
    """
    Collect exactly `horizon` steps (resetting env on done).
    Returns a batch dict with obs/actions/old_log_probs/values/dones and the final bootstrap value.
    """
    obs, _ = env.reset()

    obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []

    for _ in tqdm(range(horizon), disable=not bar):
        # store obs BEFORE stepping (standard PPO convention)
        obs_buf.append(torch.as_tensor(obs, dtype=torch.float32))

        next_obs, reward, done, info, action_t, v_t, logp_t = Step(agent, env, obs, stochastic=stochastic)
        if logp_t is None:
            raise RuntimeError("PPO needs stochastic=True during data collection to store log-probs.")

        act_buf.append(action_t)
        rew_buf.append(reward)
        done_buf.append(done)
        logp_buf.append(logp_t)
        val_buf.append(v_t)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    with torch.no_grad():
        device = next(agent.parameters()).device
        last_v = agent.forward_value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).detach().cpu()

    batch = {
        "obs":       torch.stack(obs_buf),         # [T, obs_dim]
        "actions":   torch.stack(act_buf),         # [T, act_dim]
        "rewards":   torch.as_tensor(rew_buf,  dtype=torch.float32),  # [T]
        "dones":     torch.as_tensor(done_buf, dtype=torch.bool),     # [T]
        "log_probs": torch.stack(logp_buf).float(),                   # [T]
        "values":    torch.stack(val_buf).float(),                    # [T]
        "last_value": last_v,                                         # []
    }
    return batch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    vals = torch.cat([values, last_value.reshape(1)])

    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t].item())
        delta = rewards[t] + gamma * vals[t+1] * nonterminal - vals[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8) 
    return adv, returns


def ppo_update(agent, optimizer, batch, epochs=10, minibatch_size=2048,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
               gamma=0.99, lam=0.95, adv_norm=True, target_kl=0.01,
               value_clip=None):
    """
    Enhanced PPO update with early stopping and additional stabilization.
    """
    obs      = batch["obs"]
    actions  = batch["actions"]
    rewards  = batch["rewards"]
    dones    = batch["dones"]
    old_logp = batch["log_probs"]
    values   = batch["values"]
    last_val = batch["last_value"]

    adv, ret = compute_gae(rewards, values, dones, last_val, gamma=gamma, lam=lam)
    if adv_norm:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    device = next(agent.parameters()).device
    obs      = obs.to(device)
    actions  = actions.to(device)
    old_logp = old_logp.to(device)
    old_values = values.to(device)  # Store original values for clipping
    adv      = adv.to(device)
    ret      = ret.to(device)

    N = obs.shape[0]
    
    # Track metrics for early stopping
    approx_kl_divs = []
    
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_kl = 0.0
        epoch_batches = 0
        
        for i in range(0, N, minibatch_size):
            mb = perm[i:i+minibatch_size]
            mb_obs, mb_act = obs[mb], actions[mb]
            mb_old_logp    = old_logp[mb]
            mb_old_values  = old_values[mb]
            mb_adv         = adv[mb]
            mb_ret         = ret[mb]

            new_logp, entropy, value = agent.evaluate(mb_obs, mb_act)

            # Policy loss with clipping
            ratio = (new_logp - mb_old_logp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            # Value loss with optional clipping
            if value_clip is not None:
                v_clipped = mb_old_values + torch.clamp(
                    value - mb_old_values, -value_clip, value_clip
                )
                v_loss1 = (value - mb_ret).pow(2)
                v_loss2 = (v_clipped - mb_ret).pow(2)
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                v_loss = 0.5 * (value - mb_ret).pow(2).mean()

            # Entropy loss
            ent_loss = -entropy.mean()

            # Total loss
            loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

            # Compute approximate KL divergence for early stopping
            with torch.no_grad():
                approx_kl = (mb_old_logp - new_logp).mean()
                epoch_kl += approx_kl.item()
                epoch_batches += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)

            with torch.no_grad():
                post_clip = torch.linalg.vector_norm(
                    torch.stack([p.grad.norm(2) for p in agent.parameters() if p.grad is not None])
                )

            if not torch.isfinite(post_clip):
                print("Warning: NaN/Inf gradients — skipping optimizer.step()")
            else:
                optimizer.step()

        # Early stopping based on KL divergence
        avg_kl = epoch_kl / max(epoch_batches, 1)
        approx_kl_divs.append(avg_kl)
        
        if target_kl is not None and avg_kl > target_kl * 2:
            print(f"Early stopping at epoch {epoch+1}/{epochs} due to KL divergence: {avg_kl:.4f}")
            break
    
    return {
        "approx_kl": approx_kl_divs[-1] if approx_kl_divs else 0.0,
        "epochs_completed": epoch + 1
    }

@torch.no_grad()
def _completed_episode_returns_in_batch(rewards: torch.Tensor, dones: torch.Tensor):
    """
    Extract returns for episodes that *finish* inside this rollout batch.
    Discards the last partial-episode segment if it doesn't end with done=True.
    """
    ep_rets = []
    acc = 0.0
    for r, d in zip(rewards.tolist(), dones.tolist()):
        acc += r
        if d:
            ep_rets.append(acc)
            acc = 0.0
    return ep_rets

def create_lr_scheduler(optimizer, scheduler_type="cosine", total_updates=1000, 
                       warmup_updates=0, min_lr_ratio=0.1, **kwargs):
    """
    Create learning rate scheduler with optional warmup.
    
    Args:
        scheduler_type: "cosine", "linear", "exponential", "step", or "plateau"
        total_updates: Total number of training updates
        warmup_updates: Number of warmup updates (linear warmup from 0 to initial_lr)
        min_lr_ratio: Minimum learning rate as ratio of initial learning rate
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, LinearLR, ExponentialLR, 
        StepLR, ReduceLROnPlateau, SequentialLR
    )
    
    schedulers = []
    milestones = []
    
    # Add warmup scheduler if specified
    if warmup_updates > 0:
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=1e-6,  # Start very close to 0
            end_factor=1.0,     # End at initial learning rate
            total_iters=warmup_updates
        )
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_updates)
    
    # Main scheduler
    main_updates = total_updates - warmup_updates
    
    if scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=main_updates, 
            eta_min=optimizer.param_groups[0]['lr'] * min_lr_ratio
        )
    elif scheduler_type == "linear":
        main_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=main_updates
        )
    elif scheduler_type == "exponential":
        gamma = (min_lr_ratio) ** (1.0 / main_updates)
        main_scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", main_updates // 4)
        gamma = kwargs.get("gamma", 0.5)
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        main_scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize returns
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            min_lr=optimizer.param_groups[0]['lr'] * min_lr_ratio
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    schedulers.append(main_scheduler)
    
    # Return combined scheduler if warmup is used
    if warmup_updates > 0 and scheduler_type != "plateau":
        return SequentialLR(optimizer, schedulers, milestones)
    else:
        return main_scheduler

def train_ppo(
    agent,
    env,
    optimizer,
    total_updates: int = 100,
    horizon: int = 2048,
    ppo_epochs: int = 10,
    minibatch_size: int = 128,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    adv_norm: bool = True,
    grad_clip: float = 0.5,
    target_kl: float = 0.01,
    value_clip: float = None,
    # Learning rate scheduler options
    scheduler_type: str = "cosine",
    warmup_updates: int = 0,
    min_lr_ratio: float = 0.1,
    # Stabilization options
    reward_scaling: float = 1.0,
    reward_clipping: float = None,
    obs_norm: bool = False,
    # Logging options
    log_every: int = 1,
    progress_bar: bool = False,
    save_every: int = 0,
    save_path: str = "checkpoints/ppo_agent",
):
    """
    Enhanced PPO training loop with learning rate scheduling and stabilization.
    """
    import os
    
    device = next(agent.parameters()).device
    
    # Create learning rate scheduler
    if scheduler_type != "none":
        scheduler = create_lr_scheduler(
            optimizer, 
            scheduler_type=scheduler_type,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            min_lr_ratio=min_lr_ratio
        )
        use_plateau_scheduler = scheduler_type == "plateau"
    else:
        scheduler = None
        use_plateau_scheduler = False
    
    # Running statistics for observation normalization
    if obs_norm:
        obs_rms = RunningMeanStd()
    
    # Create save directory if needed
    if save_every > 0:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    stats = {
        "update": [],
        "steps": [],
        "mean_return": [],
        "median_return": [],
        "min_return": [],
        "max_return": [],
        "learning_rate": [],
        "approx_kl": [],
        "epochs_completed": [],
        "loss_last": [],
    }

    best_mean_return = float("-inf")
    
    for upd in tqdm(range(1, total_updates + 1), disable=not progress_bar):
        t0 = perf_counter()

        # 1) Collect rollout (stochastic policy for exploration)
        batch = Rollout(agent, env, horizon=horizon, stochastic=True, bar=False)
        
        # 2) Apply reward scaling/clipping if specified
        if reward_scaling != 1.0:
            batch["rewards"] *= reward_scaling
        if reward_clipping is not None:
            batch["rewards"] = torch.clamp(batch["rewards"], -reward_clipping, reward_clipping)
        
        # 3) Observation normalization
        if obs_norm:
            obs_rms.update(batch["obs"])
            batch["obs"] = (batch["obs"] - obs_rms.mean) / torch.sqrt(obs_rms.var + 1e-8)

        # 4) Quick episode-return logging from the rollout
        ep_rets = _completed_episode_returns_in_batch(batch["rewards"], batch["dones"])
        if len(ep_rets) == 0:
            mean_ret = float("nan")
            med_ret = float("nan")
            min_ret = float("nan")
            max_ret = float("nan")
        else:
            mean_ret = float(torch.tensor(ep_rets).mean())
            med_ret  = float(torch.tensor(ep_rets).median())
            min_ret  = float(min(ep_rets))
            max_ret  = float(max(ep_rets))

        # 5) PPO update with stabilization
        update_info = ppo_update(
            agent,
            optimizer,
            batch,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=grad_clip,
            gamma=gamma,
            lam=lam,
            adv_norm=adv_norm,
            target_kl=target_kl,
            value_clip=value_clip,
        )

        # 6) Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if use_plateau_scheduler:
                scheduler.step(mean_ret if not np.isnan(mean_ret) else 0.0)
            else:
                scheduler.step()

        # 7) Save checkpoint if needed
        if save_every > 0 and upd % save_every == 0:
            checkpoint = {
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'update': upd,
                'best_mean_return': best_mean_return,
                'stats': stats,
            }
            torch.save(checkpoint, f"{save_path}_update_{upd}.pth")
            
            # Save best model
            if not np.isnan(mean_ret) and mean_ret > best_mean_return:
                best_mean_return = mean_ret
                torch.save(checkpoint, f"{save_path}_best.pth")

        # 8) Logging
        t1 = perf_counter()
        if upd % log_every == 0:
            print(
                f"[{upd:03d}] steps={horizon}  "
                f"ret: mean={mean_ret:.1f} med={med_ret:.1f} "
                f"min={min_ret:.1f} max={max_ret:.1f}  "
                f"lr={current_lr:.2e} kl={update_info['approx_kl']:.4f} "
                f"epochs={update_info['epochs_completed']}/{ppo_epochs}  "
                f"time={t1 - t0:.2f}s"
            )

        # 9) Store statistics
        stats["update"].append(upd)
        stats["steps"].append(horizon)
        stats["mean_return"].append(mean_ret)
        stats["median_return"].append(med_ret)
        stats["min_return"].append(min_ret)
        stats["max_return"].append(max_ret)
        stats["learning_rate"].append(current_lr)
        stats["approx_kl"].append(update_info["approx_kl"])
        stats["epochs_completed"].append(update_info["epochs_completed"])

    return stats

class RunningMeanStd:
    """Tracks running mean and standard deviation of inputs."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def evaluate_ppo(agent, seed=42, max_steps=1000, step_delay=0.05):
    import time
    
    # Define a fresh environment with render mode human, and the same seed as the original
    env = TorchAntEnv(render_mode="human", seed=seed)
    obs, _ = env.reset()
    done = False
    
    total_reward = 0.0
    step_count = 0
    
    print(f"Starting evaluation episode with seed {seed}...")
    
    while not done and step_count < max_steps:
        # Use deterministic policy (stochastic=False) for evaluation
        next_obs, reward, done, info, action, value, log_prob = Step(
            agent, env, obs, stochastic=False
        )
        
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        # Add delay to make it watchable
        time.sleep(step_delay)
        
        # Optional: print step info
        if step_count % 100 == 0:
            print(f"Step {step_count}: reward = {reward:.3f}, total = {total_reward:.1f}")
    
    env.close()
    
    print(f"Episode finished!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/step_count:.3f}")
    
    return total_reward, step_count

def main():
    from newAgent import ActorCritic
    from torch.optim import Adam
    
    seed = 42
    env = TorchAntEnv(seed=seed)
    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]
    agent = ActorCritic(obs_space_shape, action_space_shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    optimizer = Adam(agent.parameters(), lr=3e-4)

    train_ppo(
        agent, env, optimizer, 
        total_updates=100, 
        progress_bar=True,
        scheduler_type="cosine",
        warmup_updates=10,
        target_kl=0.1,
        value_clip=0.2,
        save_every=25,
        save_path="checkpoints/ppo_agent"
    )
    '''
    train_ppo(agent, env, optimizer, 
          scheduler_type="cosine", 
          warmup_updates=10)

    # Aggressive stabilization
    train_ppo(agent, env, optimizer,
            target_kl=0.005,  # Stricter KL limit
            value_clip=0.1,   # Clip value updates
            reward_clipping=10.0,  # Clip rewards
            obs_norm=True)    # Normalize observations
    '''
    evaluate_ppo(agent)

    # Save the final agent weights
    torch.save(agent.state_dict(), "ppo_agent_final.pth")

if __name__ == "__main__":
    main()