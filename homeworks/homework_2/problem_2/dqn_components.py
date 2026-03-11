"""
DQN Components - Homework 2, Problem 2

This file contains the core building blocks of Deep Q-Network (DQN).
Your task is to implement each function marked with TODO.

Components to implement:
1. Replay Buffer - Store and sample experience
2. Epsilon-greedy exploration
3. TD target computation (standard and Double DQN)
4. TD loss computation
5. Target network updates (soft and hard)

Based on the original DQN paper and Double DQN:
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2016)
"""

from typing import List, NamedTuple, Tuple

import numpy as np
from pyparsing import deque
import torch
import torch.nn as nn


# =============================================================================
# Part 1: Replay Buffer
# =============================================================================


class Transition(NamedTuple):
    """A single transition from the environment."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions (s, a, r, s', done) and samples random minibatches.
    This breaks the correlation between consecutive samples and stabilizes training.

    Key properties:
    - Fixed capacity with FIFO eviction (oldest transitions removed first)
    - Uniform random sampling for minibatches
    - Efficient storage using a circular buffer (deque)
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended

        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of Transition namedtuples

        Nuances to handle:
            - Sample without replacement (each transition appears at most once)
            - If batch_size > len(buffer), this will raise an error (expected)
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}"
            )

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        buffer_list = list(self.buffer)
        return [buffer_list[i] for i in indices]

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)


class NStepReplayBuffer:
    """
    N-step replay buffer for DQN.

    Instead of storing single-step transitions (s, a, r, s', done), this buffer
    accumulates n consecutive transitions and stores the n-step return:

        R_n = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^{n-1} * r_{n-1}

    The stored transition becomes (s_0, a_0, R_n, s_n, done_n), where s_n is the
    state n steps later and done_n indicates if any episode boundary was hit.

    When computing TD targets with n-step returns, use gamma^n instead of gamma:
        y = R_n + gamma^n * Q(s_n, a') * (1 - done_n)

    This can improve credit assignment by propagating reward information
    n steps backward in a single update. We don't need it for the Cartpole
    problem we're going to solve here but I wanted you to be aware of the technique.
    """

    def __init__(self, capacity: int, n_step: int, gamma: float):
        """
        Initialize the n-step replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            n_step: Number of steps for n-step returns
            gamma: Discount factor for computing n-step returns
        """
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = ReplayBuffer(capacity)  # Main buffer to store n-step transitions
        self.n_step_buffer = deque(maxlen=n_step) # Temporary buffer to accumulate n steps

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition. When n transitions accumulate (or episode ends),
        compute the n-step return and push to the underlying replay buffer.

        Logic:
            1. Append the transition to the n-step buffer.
            2. If done: flush all remaining transitions in the n-step buffer.
               For each, compute the n-step return from the remaining transitions,
               pop the oldest, and push (oldest.state, oldest.action, R_n, nth_state, nth_done)
               to the main buffer.
            3. Else if the n-step buffer is full (length == n_step):
               Compute the n-step return, pop the oldest transition, and push to main buffer.
        """

        # Append the transition to the n-step buffer
        self.n_step_buffer.append(Transition(state, action, reward, next_state, done))

        if done:
            # Flush all remaining partial n-step transitions
            while len(self.n_step_buffer) > 0:
                n_step_return, nth_state, nth_done = self._compute_nstep()
                oldest = self.n_step_buffer[0]
                self.buffer.push(
                    oldest.state,
                    oldest.action,
                    n_step_return,
                    nth_state,
                    nth_done,
                )
                self.n_step_buffer.popleft()

        elif len(self.n_step_buffer) == self.n_step:
            # Emit exactly one full n-step transition
            n_step_return, nth_state, nth_done = self._compute_nstep()
            oldest = self.n_step_buffer[0]
            self.buffer.push(
                oldest.state,
                oldest.action,
                n_step_return,
                nth_state,
                nth_done,
            )
            self.n_step_buffer.popleft()


    def _compute_nstep(self):
        """
        Compute n-step discounted return from the n-step buffer.

        Returns:
            n_step_return: The discounted sum of rewards
            nth_state: The state at the end of the n-step sequence
            nth_done: Whether the episode ended within the sequence
        """
        n_step_return = 0.0
        nth_state = self.n_step_buffer[-1].next_state
        nth_done = self.n_step_buffer[-1].done

        for i, transition in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * transition.reward

            # Stop at first terminal transition
            if transition.done:
                nth_state = transition.next_state
                nth_done = True
                break

        return n_step_return, nth_state, nth_done

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions."""
        return self.buffer.sample(batch_size)

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)


def batch_to_tensors(
    batch: List[Transition], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of transitions to tensors.

    Args:
        batch: List of Transition namedtuples
        device: Device to create tensors on

    Returns:
        states: FloatTensor of shape (batch_size, state_dim)
        actions: LongTensor of shape (batch_size,)
        rewards: FloatTensor of shape (batch_size,)
        next_states: FloatTensor of shape (batch_size, state_dim)
        dones: FloatTensor of shape (batch_size,)

    Nuances to handle:
        - Actions must be LongTensor for use with gather()
        - Dones should be float (0.0 or 1.0) for easy multiplication
    """
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    return states, actions, rewards, next_states, dones
    


# =============================================================================
# Part 2: Epsilon-Greedy Exploration
# =============================================================================


def epsilon_greedy_action(
    q_values: torch.Tensor, epsilon: float, num_actions: int
) -> int:
    """
    Select an action using epsilon-greedy exploration.

    With probability epsilon, select a random action.
    With probability (1 - epsilon), select the greedy action (highest Q-value).

    Args:
        q_values: Q-values for each action, shape (num_actions,) or (1, num_actions)
        epsilon: Exploration probability (0 = greedy, 1 = random)
        num_actions: Number of possible actions

    Returns:
        action: int — selected action index

    Nuances to handle:
        - q_values may be 1D or 2D (with batch dim of 1) - squeeze if needed
        - When epsilon=0, always return greedy action
        - When epsilon=1, always return random action
        - For ties in Q-values, argmax returns the first occurrence
    """
    q_values = q_values.squeeze()  # Remove batch dim if present

    if np.random.rand() < epsilon:
        # Random action
        return np.random.randint(num_actions)
    else:
        # Greedy action
        return torch.argmax(q_values).item()


def linear_epsilon_decay(
    step: int, epsilon_start: float, epsilon_end: float, decay_steps: int
) -> float:
    """
    Compute epsilon using linear decay schedule.

    Epsilon decreases linearly from epsilon_start to epsilon_end
    over decay_steps steps, then stays at epsilon_end.

    Args:
        step: Current training step
        epsilon_start: Initial epsilon value
        epsilon_end: Final epsilon value
        decay_steps: Number of steps for decay

    Returns:
        epsilon: float — current epsilon value

    Nuances to handle:
        - After decay_steps, epsilon should stay at epsilon_end (not go below)
        - At step=0, epsilon should be epsilon_start
        - At step=decay_steps, epsilon should be epsilon_end
    """
    if step >= decay_steps:
        return epsilon_end
    else:
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (step / decay_steps)
        return epsilon


# =============================================================================
# Part 3: TD Target Computation
# =============================================================================


def compute_td_target(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Compute TD targets for standard DQN.

    TD target = r + gamma * max_a' Q(s', a') * (1 - done)

    The key insight: we want to bootstrap from the maximum Q-value of the
    next state, but only if the episode hasn't ended.

    Args:
        rewards: Reward tensor of shape (batch_size,)
        next_q_values: Q-values for next states, shape (batch_size, num_actions)
        dones: Done flags tensor of shape (batch_size,)
        gamma: Discount factor

    Returns:
        td_targets: TD target tensor of shape (batch_size,)

    Nuances to handle:
        - When done=1, the target should just be the reward (no future value)
        - The (1-done) term "masks out" the Q-value for terminal states
        - Return detached targets to prevent gradients flowing through the target
          computation (consistent with compute_double_dqn_target which uses torch.no_grad())
    """
    max_next_q_values, _ = torch.max(next_q_values, dim=1)
    td_targets = rewards + gamma * max_next_q_values * (1 - dones)
    return td_targets.detach()


def compute_double_dqn_target(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    online_network: nn.Module,
    target_network: nn.Module,
) -> torch.Tensor:
    """
    Compute TD targets for Double DQN.

    Double DQN decouples action selection from action evaluation:
    - Use online network to select the best action: a* = argmax_a Q_online(s', a)
    - Use target network to evaluate that action: Q_target(s', a*)

    This reduces the overestimation bias present in standard DQN, where using
    max causes a positive bias (max of noisy estimates is biased upward).

    TD target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)

    Args:
        rewards: Reward tensor of shape (batch_size,)
        next_states: Next state tensor of shape (batch_size, state_dim)
        dones: Done flags tensor of shape (batch_size,)
        gamma: Discount factor
        online_network: The Q-network being trained
        target_network: The target Q-network (delayed copy)

    Returns:
        td_targets: TD target tensor of shape (batch_size,)

    Nuances to handle:
        - Use online network for action selection (argmax)
        - Use target network for value evaluation
        - No gradients should flow through target computation (use torch.no_grad())
        - Use gather() to select Q-values for the chosen actions
    """
    with torch.no_grad():
            # Select actions using the online network
            next_actions = online_network(next_states).argmax(dim=1, keepdim=True)

            # Evaluate those actions using the target network
            next_q = target_network(next_states).gather(1, next_actions).squeeze(1)

            td_targets = rewards + gamma * next_q * (1.0 - dones)

    return td_targets.detach()


# =============================================================================
# Part 4: TD Loss Computation
# =============================================================================


def compute_td_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    td_targets: torch.Tensor,
    loss_type: str = "huber",
) -> torch.Tensor:
    """
    Compute the TD loss for DQN.

    The loss measures how far our Q-value predictions are from the TD targets.
    We only compute loss for the actions that were actually taken.

    Args:
        q_values: Q-values from online network, shape (batch_size, num_actions)
        actions: Actions taken, shape (batch_size,)
        td_targets: TD targets, shape (batch_size,)
        loss_type: "huber" (smooth L1) or "mse"

    Returns:
        loss: Scalar loss tensor

    Nuances to handle:
        - Use gather to select Q-values for the actions that were taken
        - Huber loss is more robust to outliers than MSE (default choice)
        - TD targets should be detached (no gradient flow through targets)
    """

    action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    td_targets = td_targets.detach()

    if loss_type == "huber":
        loss = nn.functional.smooth_l1_loss(action_q_values, td_targets)
    elif loss_type == "mse":
        loss = nn.functional.mse_loss(action_q_values, td_targets)
    else:
        raise ValueError(
            f"Invalid loss_type: {loss_type}. Choose 'huber' or 'mse'."
        )

    return loss


# =============================================================================
# Part 5: Target Network Updates
# =============================================================================


def soft_update(
    online_network: nn.Module, target_network: nn.Module, tau: float
) -> None:
    """
    Soft update of target network parameters.

    For each parameter:
        θ_target = τ * θ_online + (1 - τ) * θ_target

    This is also known as Polyak averaging. It provides a smoother
    update than hard copying, which can help stability.

    Args:
        online_network: The Q-network being trained
        target_network: The target Q-network to update
        tau: Interpolation parameter (0 < tau <= 1)
             tau=1 means hard update (copy)
             tau=0.005 is a common choice for soft updates

    Nuances to handle:
        - Update should be done in-place on target network parameters
        - No gradients should be computed for this operation
    """
    with torch.no_grad():
        for online_param, target_param in zip(
            online_network.parameters(), target_network.parameters()
        ):
            target_param.copy_(tau * online_param + (1.0 - tau) * target_param)
    
    return None


def hard_update(online_network: nn.Module, target_network: nn.Module) -> None:
    """
    Hard update of target network parameters.

    Copies all parameters from online network to target network.
    Equivalent to soft_update with tau=1.

    Args:
        online_network: The Q-network being trained
        target_network: The target Q-network to update
    """
    target_network.load_state_dict(online_network.state_dict())


# =============================================================================
# Part 6: Q-Network Architecture (PROVIDED - No implementation needed)
# =============================================================================


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
