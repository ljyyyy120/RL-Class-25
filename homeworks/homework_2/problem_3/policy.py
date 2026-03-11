import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init (standard for PPO)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Smaller init for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        # Smaller init for value head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(checkpoint_path):
    """
    Load a trained PPO policy and return a callable function.

    Args:
        checkpoint_path (str): path to checkpoint.pt containing model.state_dict()

    Returns:
        callable: policy(obs) -> action
    """

    model = ActorCritic(obs_dim=8, act_dim=3).to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    def policy(obs):
        """
        Args:
            obs: torch.FloatTensor with shape (8,) or (batch, 8)

        Returns:
            action: tensor of discrete actions
        """

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(obs)

            action = torch.argmax(logits, dim=-1)

        return action

    return policy