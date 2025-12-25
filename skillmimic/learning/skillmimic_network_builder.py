# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# 导入rl_games库中的PyTorch算法相关模块
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

# 导入PyTorch核心库和NumPy
import torch
import torch.nn as nn
import numpy as np

# 离散动作空间的logit初始化缩放因子
DISC_LOGIT_INIT_SCALE = 1.0

# SkillMimic网络构建器，继承自A2CBuilder（Advantage Actor-Critic算法的网络构建器）
class SkillMimicBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
        return

    # 内部Network类，继承自A2CBuilder.Network
    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # 调用父类构造函数，初始化基础网络结构
            super().__init__(params, **kwargs)

            # 如果是连续动作空间
            if self.is_continuous:
                # 如果不需要学习动作分布的标准差（固定标准差）
                if (not self.space_config['learn_sigma']):
                    # 获取动作维度数量
                    actions_num = kwargs.get('actions_num')
                    # 创建标准差初始化器
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    # 创建固定的标准差参数（不需要梯度更新）
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    # 初始化标准差参数
                    sigma_init(self.sigma)

            # # # MVAE（变分自编码器）相关代码（已注释）
            # # 编码器网络：将高维输入压缩为低维潜在表示
            # self.encoder = nn.Sequential(
            #     nn.Linear(1198, 1024),  # 输入维度1198，输出维度1024
            #     nn.LeakyReLU(0.1, True),  # LeakyReLU激活函数，负斜率0.1
            #     nn.Linear(1024, 256),  # 隐藏层，输出维度256
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(256, 16),  # 输出16维潜在向量
            # )

            # # 解码器网络：将潜在表示和解码为目标维度
            # self.decoder = nn.Sequential(
            #     nn.Linear(16+823, 1024),  # 输入：16维潜在向量 + 823维条件信息
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(1024, 512),
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(512, 512),  # 输出维度512
            # )

            # # 其他版本的MVAE配置（已注释）
            # self.encoder = nn.Sequential(
            #     nn.Linear(375, 256),
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(256, 16),
            # )
            # self.decoder = nn.Sequential(
            #     nn.Linear(16+823, 1024),
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(1024, 512),
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(512, 512),
            # )
            # self.decoder = nn.Sequential(
            #     nn.Linear(16+823, 2048),
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(2048, 1024),
            #     nn.LeakyReLU(0.1, True),
            #     nn.Linear(1024, 512),
            # )

            return

        # 网络前向传播方法
        def forward(self, obs_dict):
            # 获取观察数据
            obs = obs_dict['obs']
            # 获取循环神经网络状态（如果有）
            states = obs_dict.get('rnn_states', None)

            # 评估演员网络，获取动作分布参数
            actor_outputs = self.eval_actor(obs)
            # 评估评论家网络，获取状态价值
            value = self.eval_critic(obs)

            # 组合输出：动作分布参数 + 价值 + RNN状态
            output = actor_outputs + (value, states)

            return output

        # 评估演员网络（生成动作）
        def eval_actor(self, obs, cls_latents=None): #ZC0：自定义注释标记
            # 如果提供了分类潜在向量
            if cls_latents is not None:
                # 获取每个样本的最大类别索引
                _, indices = torch.max(cls_latents, dim=-1)
                # 将观察数据的最后64维中的对应类别位置设为1（one-hot编码）
                obs[torch.arange(obs.size(0)), -64 + indices] = 1.
            
            # 通过演员CNN处理观察数据
            a_out = self.actor_cnn(obs)
            
            # 如果CNN输出是字典类型（可能包含额外信息）
            if(type(a_out) == dict): #ZC9：自定义注释标记
                # 提取观察数据部分
                a_out = a_out['obs']
            
            # 展平CNN输出（批量处理时保持批量维度）
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            # 通过演员MLP（多层感知机）处理
            a_out = self.actor_mlp(a_out)

            # # MVAE预训练相关代码（已注释）
            # # 创建常数向量c
            # c = torch.ones_like(a_out[:,823:]).to('cuda')*-0.6033 #1.6575   -0.6033
            # # rrr = a_out[0,823:]
            # # rrr2 = a_out[1,823:]
            # # 通过编码器获取潜在向量z
            # z = self.encoder(
            #     torch.cat((a_out[:,:823],c),dim=-1)  # 拼接前823维特征和常数向量c
            # )
            # # z = self.encoder(a_out[:,823:])  # 仅使用后部分特征
            # # z = self.encoder(a_out)  # 使用全部特征
            # # z = torch.randn(self.encoder(a_out).shape).to('cuda')  # 随机潜在向量
            # # 对潜在向量进行L2归一化
            # z = torch.nn.functional.normalize(z, p=2, dim=1)

            # # 打印归一化结果差异（调试用）
            # # print(torch.nn.functional.normalize(z, p=2, dim=1)[0] - torch.nn.functional.normalize(z[0], p=2, dim=0))

            # # 不计算梯度的情况下通过解码器重构
            # # with torch.no_grad():
            # a_out = self.decoder(
            #     torch.cat((z,a_out[:,:823]),dim=-1)  # 拼接潜在向量z和前823维特征
            # )
                      
            # 如果是离散动作空间
            if self.is_discrete:
                # 计算动作对数概率
                logits = self.logits(a_out)
                return logits

            # 如果是多离散动作空间
            if self.is_multi_discrete:
                # 为每个离散动作维度计算对数概率
                logits = [logit(a_out) for logit in self.logits]
                return logits

            # 如果是连续动作空间
            if self.is_continuous:
                # 计算动作均值（通过激活函数映射到合适范围）
                mu = self.mu_act(self.mu(a_out))
                # 如果使用固定标准差
                if self.space_config['fixed_sigma']:
                    # 使用预定义的固定标准差
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    # 从网络学习标准差
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma  # 返回动作均值和标准差
            return

        # 评估评论家网络（生成状态价值）
        def eval_critic(self, obs):
            # 通过评论家CNN处理观察数据
            c_out = self.critic_cnn(obs)
            # 如果CNN输出是字典类型
            if(type(c_out) == dict): #ZC9：自定义注释标记
                # 提取观察数据部分
                c_out = c_out['obs']
            # 展平CNN输出
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            # 通过评论家MLP处理
            c_out = self.critic_mlp(c_out)
            # 计算状态价值（通过激活函数映射）
            value = self.value_act(self.value(c_out))
            return value

    # 构建网络实例
    def build(self, name, **kwargs):
        # 创建Network实例
        net = SkillMimicBuilder.Network(self.params, **kwargs)
        return net



