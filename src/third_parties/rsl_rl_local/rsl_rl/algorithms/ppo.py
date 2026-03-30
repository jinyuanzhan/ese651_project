# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            observations,
            critic_observations,
            sampled_actions,
            value_targets,
            advantage_estimates,
            discounted_returns,
            prev_log_probs,
            prev_mean_actions,
            prev_action_stds,
            hidden_states,
            episode_masks,
            _,  # rnd_state_batch - not used anymore
        ) in generator:
            # Re-evaluate sampled actions under the current policy
            self.actor_critic.act(observations, masks=episode_masks, hidden_states=hidden_states)
            new_log_probs = self.actor_critic.get_actions_log_prob(sampled_actions)
            new_values = self.actor_critic.evaluate(
                critic_observations, masks=episode_masks, hidden_states=hidden_states
            )
            entropy = self.actor_critic.entropy

            # Optionally normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / (
                    advantage_estimates.std() + 1e-8
                )

            # Compute importance sampling ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(new_log_probs - prev_log_probs.squeeze(-1))

            # Clipped surrogate objective
            advantages_squeezed = advantage_estimates.squeeze(-1)
            surrogate_1 = ratio * advantages_squeezed
            surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_squeezed
            surrogate_loss = -torch.min(surrogate_1, surrogate_2).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = value_targets + torch.clamp(
                    new_values - value_targets, -self.clip_param, self.clip_param
                )
                value_loss_unclipped = (new_values - discounted_returns) ** 2
                value_loss_clipped = (value_clipped - discounted_returns) ** 2
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = ((new_values - discounted_returns) ** 2).mean()

            # Total loss = policy loss + value loss - entropy bonus
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate statistics for logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        # Adaptive learning rate schedule based on approximate KL divergence
        if self.schedule == "adaptive":
            with torch.inference_mode():
                self.actor_critic.act(observations, masks=episode_masks, hidden_states=hidden_states)
                new_log_probs_after = self.actor_critic.get_actions_log_prob(sampled_actions)
                kl = torch.mean(prev_log_probs.squeeze(-1) - new_log_probs_after)
            if kl > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl < self.desired_kl / 2.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy
