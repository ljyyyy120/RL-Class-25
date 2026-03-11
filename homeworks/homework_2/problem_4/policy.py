import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    """
    Simple fully-connected Q-network for discrete action spaces.

    Takes a state as input and outputs Q-values for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            q_values: Q-values of shape (batch_size, action_dim)
        """
        return self.network(state)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(checkpoint_path):
    """
    Load a trained DQN policy from a checkpoint and return a callable.

    Args:
        checkpoint_path (str): Path to saved model weights.

    Returns:
        Callable that maps observation -> action
    """

    model = QNetwork(state_dim=4, action_dim=2).to(DEVICE)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    def policy(obs: torch.FloatTensor):
        """
        Args:
            obs: torch.FloatTensor observation (shape [4] or [batch,4])

        Returns:
            action (int or tensor)
        """

        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.to(DEVICE)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            q_values = model(obs)
            action = torch.argmax(q_values, dim=-1)

        # return python int if single observation
        if action.numel() == 1:
            return int(action.item())

        return action

    return policy