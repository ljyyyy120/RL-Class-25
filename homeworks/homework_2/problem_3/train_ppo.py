"""
Homework 2, Problem 3: PPO Training on Pong

Train a PPO agent on PufferLib's native Pong environment using the components
you implemented in Problem 1.

Your task: Implement the `train()` function below. Everything else is provided.

Usage:
    uv run python train_ppo.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.ocean.pong.pong import Pong

from homeworks.homework_2.problem_1.ppo_components import (  # noqa: F401
    RolloutBuffer,
    compute_entropy_bonus,
    compute_policy_loss,
    compute_value_loss,
    discrete_log_prob,
    normalize_advantages,
    sample_discrete_action,
)


# =============================================================================
# Hyperparameters (tuned — you shouldn't need to change these)
# =============================================================================
NUM_ENVS = 8
NUM_STEPS = 128  # steps per env per rollout
TOTAL_TIMESTEPS = 500_000
LR = 2.5e-4
NUM_EPOCHS = 4  # PPO update epochs per rollout
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Network Architecture (provided)
# =============================================================================
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


# =============================================================================
# Evaluation (provided)
# =============================================================================
def evaluate(model, num_episodes=10):
    """Run greedy policy and return mean episode reward."""
    env = Pong(num_envs=1, max_score=5)
    obs, _ = env.reset()
    rewards_total = []
    ep_reward = 0.0

    while len(rewards_total) < num_episodes:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits, _ = model(obs_t)
        action = logits.argmax(dim=-1).cpu().numpy()
        obs, rewards, terms, truncs, _ = env.step(action)
        ep_reward += rewards[0]
        if terms[0] or truncs[0]:
            rewards_total.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

    env.close()
    return np.mean(rewards_total)


# =============================================================================
# Plotting (provided)
# =============================================================================
def plot_learning_curve(rewards, filename="ppo_pong.png"):
    """Save a learning curve plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="per-rollout")
    if len(rewards) >= 10:
        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), smooth, label="10-rollout avg")
    plt.xlabel("Rollout")
    plt.ylabel("Mean Episode Reward")
    plt.title("PPO on Pong")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Learning curve saved to {filename}")


# =============================================================================
# YOUR TASK: Implement train()
# =============================================================================
def train():
    """
    Train a PPO agent on Pong and return a list of mean episode rewards per rollout.

    Returns:
        List of mean episode rewards (one per rollout).
    """
    # Create the environment, model, optimizer, and `RolloutBuffer`
    env = Pong(num_envs=NUM_ENVS, max_score=5)
    model = ActorCritic(8, 3).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        eps=1e-5,
    )
    buffer = RolloutBuffer(num_steps = NUM_STEPS,
                                num_envs  = NUM_ENVS,
                                obs_shape = (8,), #Observations 8 floats (paddle/ball positions, normalized to [0, 1])
                                action_shape = (),  # Actions**Discrete(3) — stay, up, down
                                device= DEVICE)
    
    
    mean_episode_rewards_per_rollout = []
    running_episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

    # Collect rollouts by stepping the environment and storing transitions
    
    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    num_rollouts = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    best_train_reward = -float("inf")
    best_state_dict = None

    for rollout_idx in range(num_rollouts):
        buffer.reset()
        completed_episode_rewards = []

        for step in range(NUM_STEPS):
            with torch.no_grad():
                logits, value = model(obs_t)
                action, log_prob = sample_discrete_action(logits)

            # step env
            next_obs, rewards, terms, truncs, info = env.step(action.cpu().numpy())
            
            # convert dtype
            rewards = np.asarray(rewards, dtype=np.float32)
            terms = np.asarray(terms, dtype=bool)
            value = value.squeeze(-1)
            dones = np.logical_or(terms, truncs)

            # Track completed episode returns for logging
            running_episode_rewards += rewards
            for i in range(NUM_ENVS):
                if dones[i]:
                    completed_episode_rewards.append(float(running_episode_rewards[i]))
                    running_episode_rewards[i] = 0.0


            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            terms_t = torch.tensor(terms.astype(np.float32), dtype=torch.float32, device=DEVICE)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

            # store transition
            buffer.add(
                obs_t,
                action,
                log_prob,
                rewards_t,
                terms_t,
                value,
            )

            obs_t = next_obs_t
            

        # Mean episode reward for this rollout
        if completed_episode_rewards:
            rollout_reward = float(np.mean(completed_episode_rewards))
            mean_episode_rewards_per_rollout.append(rollout_reward)
        else:
            rollout_reward = mean_episode_rewards_per_rollout[-1] if mean_episode_rewards_per_rollout else 0.0
            mean_episode_rewards_per_rollout.append(rollout_reward)


        # Bootstrap from final obs after rollout
        with torch.no_grad():
            _, last_value = model(obs_t)              # shape: (NUM_ENVS,)
        
        last_value = last_value.squeeze(-1)

        # Compute advantages with `buffer.compute_returns_and_advantages`
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value = last_value,
            last_done = terms_t,
            gamma = GAMMA,
            gae_lambda = GAE_LAMBDA
            )
    

        # Run the PPO update loop (multiple epochs of minibatch updates)
        for epoch in range(NUM_EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE,returns,advantages):

                norm_adv = normalize_advantages(batch["advantages"])

                logits, values = model(batch["obs"])
                values = values.squeeze(-1)
                new_log_probs = discrete_log_prob(logits, batch["actions"].long())

                L_c = compute_policy_loss(log_probs = new_log_probs,
                                        old_log_probs = batch["log_probs"],
                                        advantages = norm_adv,
                                        clip_epsilon = CLIP_EPSILON)
                
                L_v = compute_value_loss(values = values,
                                        returns = batch["returns"])
                
                probs = torch.softmax(logits, dim=-1)
                L_E = compute_entropy_bonus(probs)

                loss = (
                    L_c
                    + VALUE_COEF * L_v
                    - ENTROPY_COEF * L_E
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
        if rollout_reward > best_train_reward:
            best_train_reward = rollout_reward
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        
    if best_state_dict is not None:
        torch.save(best_state_dict, "ppo_pong.pt")
    else:
        torch.save(model.state_dict(), "ppo_pong.pt")

    env.close()
    #Return a list of mean episode rewards per rollout
    return mean_episode_rewards_per_rollout




# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    reward_history = train()
    plot_learning_curve(reward_history)

    # Load the final model for evaluation
    model = ActorCritic(8, 3).to(DEVICE)
    model.load_state_dict(torch.load("ppo_pong.pt", weights_only=True))
    mean_reward = evaluate(model)
    print(f"Evaluation mean reward: {mean_reward:.2f}")


