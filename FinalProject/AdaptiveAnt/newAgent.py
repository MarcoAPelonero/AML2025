import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.actor  = nn.Linear(256, act_dim)   # mean
        self.critic = nn.Linear(256, 1)

        # state-independent log-std (learnable)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu') if m not in [self.critic, self.actor] else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f  = self.shared(x)
        mu = self.actor(f)
        v  = self.critic(f).squeeze(-1)
        return torch.tanh(mu), v

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # numerically-stable atanh for actions in (-1,1)
        x = x.clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def _log_prob_tanh_normal(mu: torch.Tensor, log_std: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z = ActorCritic._atanh(a)
        std  = torch.exp(torch.clamp(log_std, -20.0, -1.0))
        dist = Normal(mu, std)
        log_det = 2 * (torch.log(torch.tensor(2.0, device=z.device)) - z - F.softplus(-2*z))
        log_prob = dist.log_prob(z) - log_det
        return log_prob.sum(-1)

    @staticmethod
    def _tanh_squash(mu, log_std):
        log_std = torch.clamp(log_std, -20.0, -1.0)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        log_det = 2 * (torch.log(torch.tensor(2.0, device=z.device)) - z - F.softplus(-2*z))
        log_prob = dist.log_prob(z).sum(-1) - log_det.sum(-1)
        return a, log_prob

    def act(self, obs: torch.Tensor, stochastic: bool = True):
        f  = self.shared(obs)
        mu = self.actor(f)
        v  = self.critic(f).squeeze(-1)

        if stochastic:
            a, logp = self._tanh_squash(mu, self.log_std)
        else:
            a = torch.tanh(mu)
            logp = None
        return a, logp, v

    # ====== added minimal methods for PPO ======
    def forward_value(self, obs: torch.Tensor) -> torch.Tensor:
        f  = self.shared(obs)
        v  = self.critic(f).squeeze(-1)
        return v

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Recompute log-prob under current policy for stored actions,
        plus entropy (pre-squash Normal) and value.
        Returns: new_log_prob [B], entropy [B], value [B]
        """
        f   = self.shared(obs)
        mu  = self.actor(f)
        v   = self.critic(f).squeeze(-1)
        logp = self._log_prob_tanh_normal(mu, self.log_std, actions)

        # Use base Normal entropy (common practice with tanh policies)
        std = torch.exp(torch.clamp(self.log_std, -20.0, -1.0))
        dist = Normal(mu, std)
        entropy = dist.entropy().sum(-1)

        return logp, entropy, v

def test_agent():
    agent = ActorCritic(obs_dim=4, act_dim=2)
    print(agent)

    obs = torch.randn(1, 4)
    mu, v = agent(obs)
    print("Action distribution (mu):", mu)
    print("State value (v):", v)

if __name__ == "__main__":
    test_agent()