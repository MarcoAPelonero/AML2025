import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
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
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return adv, returns


def ppo_update(agent, optimizer, batch, epochs=10, minibatch_size=2048,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
               gamma=0.99, lam=0.95, adv_norm=True):
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
    adv      = adv.to(device)
    ret      = ret.to(device)

    N = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, minibatch_size):
            mb = perm[i:i+minibatch_size]
            mb_obs, mb_act = obs[mb], actions[mb]
            mb_old_logp    = old_logp[mb]
            mb_adv         = adv[mb]
            mb_ret         = ret[mb]

            new_logp, entropy, value = agent.evaluate(mb_obs, mb_act)

            ratio = (new_logp - mb_old_logp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            v_loss = 0.5 * (value - mb_ret).pow(2).mean()
            ent_loss = -entropy.mean()

            loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

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
    log_every: int = 1,
    progress_bar: bool = False,
):
    """
    Minimal PPO training loop.
    Uses your Rollout() for data collection and ppo_update() for optimization.
    """

    device = next(agent.parameters()).device
    stats = {
        "update": [],
        "steps": [],
        "mean_return": [],
        "median_return": [],
        "min_return": [],
        "max_return": [],
        "loss_last": [],   # optional placeholder if you later return loss from ppo_update
    }

    for upd in tqdm(range(1, total_updates + 1), disable=not progress_bar):
        t0 = perf_counter()

        # 1) Collect rollout (stochastic policy for exploration)
        batch = Rollout(agent, env, horizon=horizon, stochastic=True, bar=False)

        # 2) Quick episode-return logging from the rollout
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

        ppo_update(
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
        )

        t1 = perf_counter()
        if upd % log_every == 0:
            print(
                f"[{upd:03d}] steps={horizon}  "
                f"ret: mean={mean_ret:.1f} med={med_ret:.1f} "
                f"min={min_ret:.1f} max={max_ret:.1f}  "
                f"time={t1 - t0:.2f}s"
            )

        stats["update"].append(upd)
        stats["steps"].append(horizon)
        stats["mean_return"].append(mean_ret)
        stats["median_return"].append(med_ret)
        stats["min_return"].append(min_ret)
        stats["max_return"].append(max_ret)

    return stats

def test_episode():
    from newAgent import ActorCritic
    from newEnvironment import TorchAntEnv

    env = TorchAntEnv()
    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]
    agent = ActorCritic(obs_space_shape, action_space_shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    total_reward = Rollout(agent, env)
    print(f"Total reward: {total_reward}")

def probe_ppo_shapes(
                     horizon: int = 512,
                     stochastic: bool = True,
                     gamma: float = 0.99,
                     lam: float = 0.95,
                     adv_norm: bool = True,
                     minibatch_size: int = 128):
    """
    Collect a fixed-horizon rollout, compute GAE/returns, and print shapes + quick stats.
    Does NOT update the network.
    """
    from newAgent import ActorCritic
    from newEnvironment import TorchAntEnv

    env = TorchAntEnv()
    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]
    agent = ActorCritic(obs_space_shape, action_space_shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    batch = Rollout(agent, env, horizon=horizon, stochastic=stochastic, bar=False)

    obs      = batch["obs"]          # [T, obs_dim]
    actions  = batch["actions"]      # [T, act_dim]
    rewards  = batch["rewards"]      # [T]
    dones    = batch["dones"]        # [T]
    old_logp = batch["log_probs"]    # [T]
    values   = batch["values"]       # [T]
    last_val = batch["last_value"]   # []

    # 2) GAE + returns (CPU tensors)
    adv, ret = compute_gae(rewards, values, dones, last_val, gamma=gamma, lam=lam)
    adv_raw_stats = (adv.mean().item(), adv.std(unbiased=False).item())
    if adv_norm:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    # 3) Make a first minibatch (just to inspect shapes)
    T = obs.shape[0]
    mb_size = min(minibatch_size, T)
    mb_idx = torch.arange(mb_size)
    mb = {
        "obs":      obs[mb_idx],
        "actions":  actions[mb_idx],
        "old_logp": old_logp[mb_idx],
        "adv":      adv[mb_idx],
        "ret":      ret[mb_idx],
    }

    # 4) Also check that evaluate() produces matching shapes on the minibatch
    device = next(agent.parameters()).device
    with torch.no_grad():
        new_logp, entropy, v_now = agent.evaluate(mb["obs"].to(device), mb["actions"].to(device))

    # 5) Quick sanity stats (CPU)
    def _stats(x):
        return {
            "shape": tuple(x.shape),
            "min": float(torch.min(x)),
            "max": float(torch.max(x)),
            "mean": float(torch.mean(x)),
            "std":  float(torch.std(x)),
            "nan_count": int(torch.isnan(x).sum()),
        }

    # Print everything to terminal
    print("=" * 60)
    print("PPO SHAPES & STATS PROBE")
    print("=" * 60)
    
    print(f"Horizon: {T}")
    print(f"Device: {device}")
    print()
    
    print("BATCH SHAPES:")
    print(f"  obs:        {tuple(obs.shape)}")
    print(f"  actions:    {tuple(actions.shape)}")
    print(f"  rewards:    {tuple(rewards.shape)}")
    print(f"  dones:      {tuple(dones.shape)}")
    print(f"  old_logp:   {tuple(old_logp.shape)}")
    print(f"  values:     {tuple(values.shape)}")
    print(f"  last_value: {tuple(last_val.shape) if hasattr(last_val, 'shape') else 'scalar'}")
    print(f"  advantages: {tuple(adv.shape)}")
    print(f"  returns:    {tuple(ret.shape)}")
    print()
    
    print("EVALUATE OUTPUT SHAPES:")
    print(f"  new_logp: {tuple(new_logp.shape)}")
    print(f"  entropy:  {tuple(entropy.shape)}")
    print(f"  value:    {tuple(v_now.shape)}")
    print()
    
    print("MINIBATCH INFO:")
    print(f"  first_minibatch_size: {mb_size}")
    print()
    
    print("STATISTICS:")
    reward_stats = _stats(rewards)
    value_stats = _stats(values)
    adv_stats = _stats(adv)
    ret_stats = _stats(ret)
    
    print("  Rewards:")
    print(f"    min: {reward_stats['min']:.4f}, max: {reward_stats['max']:.4f}")
    print(f"    mean: {reward_stats['mean']:.4f}, std: {reward_stats['std']:.4f}")
    print(f"    nan_count: {reward_stats['nan_count']}")
    
    print("  Values:")
    print(f"    min: {value_stats['min']:.4f}, max: {value_stats['max']:.4f}")
    print(f"    mean: {value_stats['mean']:.4f}, std: {value_stats['std']:.4f}")
    print(f"    nan_count: {value_stats['nan_count']}")
    
    print("  Advantages (raw):")
    print(f"    mean: {adv_raw_stats[0]:.4f}, std: {adv_raw_stats[1]:.4f}")
    
    print("  Advantages (normalized):")
    print(f"    min: {adv_stats['min']:.4f}, max: {adv_stats['max']:.4f}")
    print(f"    mean: {adv_stats['mean']:.4f}, std: {adv_stats['std']:.4f}")
    print(f"    nan_count: {adv_stats['nan_count']}")
    
    print("  Returns:")
    print(f"    min: {ret_stats['min']:.4f}, max: {ret_stats['max']:.4f}")
    print(f"    mean: {ret_stats['mean']:.4f}, std: {ret_stats['std']:.4f}")
    print(f"    nan_count: {ret_stats['nan_count']}")
    
    print("=" * 60)

def main():
    from newAgent import ActorCritic
    from newEnvironment import TorchAntEnv
    from torch.optim import Adam
    
    env = TorchAntEnv()
    obs_space_shape = env.observation_space.shape[0]
    action_space_shape = env.action_space.shape[0]
    agent = ActorCritic(obs_space_shape, action_space_shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    optimizer = Adam(agent.parameters(), lr=3e-4)

    train_ppo(agent, env, optimizer, progress_bar=True)

if __name__ == "__main__":
    main()