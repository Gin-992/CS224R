import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import minimum

import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(nn.Linear(obs_shape[0], hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        # 输出动作每个维度的均值
        mu = self.policy(obs)
        # 限制在 -1 到 1 之间
        mu = torch.tanh(mu)
        # 设置动作分布标准差
        std = torch.ones_like(mu) * self.std
        # 构造截断正态分布 (限制动作范围)
        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics,
                 hidden_dim):
        super().__init__()

        # 创建多个结构相同、参数独立的 Critic 神经网络
        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
            # 列表推导式
            for _ in range(num_critics)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class ACAgent:
    def __init__(self, obs_shape, action_shape, device, lr,
                 hidden_dim, num_critics, critic_target_tau, stddev_clip, use_tb):
        self.device = device
        # theta_target = (1 - tau) * theta_target + tau * theta_critic
        # controls how slowly the target network follows the critic.
        self.critic_target_tau = critic_target_tau
        # 是否记录 TensorBoard
        self.use_tb = use_tb
        # 动作采样时的噪声截断参数
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape, action_shape,
                           hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape,
                             num_critics, hidden_dim).to(device)
        # 通过 EMA 缓慢更新参数，用于计算稳定的 Bellman Targe
        self.critic_target = Critic(obs_shape, action_shape,
                                    num_critics, hidden_dim).to(device)
        # 把 Critic 的参数复制给 Target Critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 切换网络训练模式
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, replay_iter):
        '''
        This function updates the critic and target critic parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        out = self.critic(obs, action)

        # Bellman target
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample(clip=None)

            out_target = self.critic_target(next_obs, next_action)
            indices = random.sample(range(len(out_target)), 2)
            q1 = out_target[indices[0]]
            q2 = out_target[indices[1]]
            # Computes the element-wise minimum
            min_q = torch.minimum(q1, q2)
            target = reward + discount * min_q

        loss = 0
        for o in out:
            loss += torch.mean((o - target) ** 2)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        metrics['critic_loss'] = loss.item()

        return metrics

    def update_actor(self, replay_iter):
        '''
        This function updates the policy parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        '''
        metrics = dict()

        batch = next(replay_iter)
        obs, _, _, _, _ = utils.to_torch(
            batch, self.device)

        dist = self.actor(obs)
        action = dist.sample(clip=None)
        q = self.critic(obs, action)
        # [N, batch, 1]
        q = torch.stack(q)
        loss = -torch.mean(q)

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = loss.item()

        return metrics

    def bc(self, replay_iter):
        '''
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        dist = self.actor(obs)
        loss = - torch.mean(torch.sum(dist.log_prob(action), dim=1))

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = loss.item()

        return metrics
