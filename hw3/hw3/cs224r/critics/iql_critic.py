from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb
import numpy as np

from cs224r.infrastructure import pytorch_util as ptu

class IQLCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        # 是否使用双 Q 网络。
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        # 折扣因子
        self.gamma = hparams['gamma']

        # specification 配置说明
        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        # 需要训练的 Q 网络。
        # 输入 s，一次输出所有离散动作的 Q 值，再根据 a 取出对应的值。
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        # 目标 Q 网络，用于给 V 网络提供较稳定的训练目标。
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)

        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.mse_loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        self.v_net = network_initializer(self.ob_dim, 1)
        self.v_net.to(ptu.device)

        self.v_optimizer = self.optimizer_spec.constructor(
            self.v_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        
        self.v_learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.v_optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.iql_expectile = hparams['iql_expectile']

    def expectile_loss(self, diff):
        # 非对称平方误差损失
        weight = torch.where(
            diff > 0,
            ptu.from_numpy(np.array(self.iql_expectile)),
            ptu.from_numpy(np.array(1 -  self.iql_expectile))
        )
        return weight * (diff**2)

    def update_v(self, ob_no, ac_na):
        """
        Update value function using expectile loss
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)

        with torch.no_grad():
            q = torch.gather(self.q_net_target(ob_no), 1, ac_na.unsqueeze(1))
        diff = q - self.v_net(ob_no)
        value_loss = torch.mean(self.expectile_loss(diff))

        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_value_(self.v_net.parameters(), self.grad_norm_clipping)
        self.v_optimizer.step()
        
        self.v_learning_rate_scheduler.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}



    def update_q(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
        Use target v network to train Q
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        with torch.no_grad():
            v = self.v_net(next_ob_no)
        q = torch.gather(self.q_net(ob_no), 1, ac_na.unsqueeze(1))
        loss = torch.mean((reward_n.unsqueeze(1) + self.gamma * (1 - terminal_n.unsqueeze(1)) * v - q) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        
        self.learning_rate_scheduler.step()

        return {'Training Q Loss': ptu.to_numpy(loss)}

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
