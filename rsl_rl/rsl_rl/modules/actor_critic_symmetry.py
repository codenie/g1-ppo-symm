import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.state_estimator import VAE
from rsl_rl.utils.torch_utils import init_orhtogonal

## 构造symmetry需要的辅助函数
from rsl_rl.utils.symm_utils import add_repr_to_gspace, SimpleEMLP, get_symm_tensor

import escnn
import escnn.group
from escnn.group import CyclicGroup
from escnn.nn import FieldType, EquivariantModule, GeometricTensor

class ActorCriticSymmetry(nn.Module):
    def __init__(self,  num_vae,
                        num_obs_step,
                        num_critic_obs,
                        num_history,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs
                        ):
        if kwargs:
            print("ActorCriticSymmetry.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        ### actor,  critic ---------------------
        mlp_input_dim_a = num_obs_step
        mlp_input_dim_c = num_critic_obs
        
        ## 建立group和space
        G = CyclicGroup(2)
        gspace = escnn.gspaces.no_base_space(G)
        # 添加需要的变换函数
        add_repr_to_gspace(G, [0, 1, 2], [-1, 1, -1], 'base_ang_vel')
        add_repr_to_gspace(G, [0, 1, 2], [1, -1, 1], 'projected_gravity')
        add_repr_to_gspace(G, [0, 1, 2], [1, -1, -1], 'commands')
        add_repr_to_gspace(G, [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19],
                              [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1],
                            'dof_pos_vel_action')
        add_repr_to_gspace(G, [0, 1], [-1, -1], 'phase')
        # add_repr_to_gspace(G, [], [], 'name')
        # add_repr_to_gspace(G, [], [], 'name')
        
        ## 配置actor
        actor_input_transitions = ['base_ang_vel', 'projected_gravity', 'commands', 
                                   *('dof_pos_vel_action',)*3, 'phase']
        actor_output_transitions = ['dof_pos_vel_action']
        self.actor_in_field_type = FieldType(gspace, [G.representations[name] for name in actor_input_transitions])
        self.actor_out_field_type = FieldType(gspace, [G.representations[name] for name in actor_output_transitions])
        # ## 配置critic
        # critic_input_transitions = []
        # critic_output_transitions = []
        # critic_in_field_type = FieldType(gspace, [G.representations[name] for name in critic_input_transitions])
        # critic_out_field_type = FieldType(gspace, [G.representations[name] for name in critic_output_transitions])
     
        ## 根据上述配置构造两个网络 
        ## TODO: 注意两个网络，训练参数可能和普通MLP相比不一样，比如学习速度等
        self.actor = SimpleEMLP(self.actor_in_field_type, self.actor_out_field_type,
            hidden_dims = actor_hidden_dims, 
            activation = activation,)

        # self.critic = SimpleEMLP(critic_in_field_type, critic_out_field_type,
        #     hidden_dims = critic_hidden_dims,
        #     activation=activation,)
        
        # 构造使用普通的critic网络 Value function     critic
        activation_fn = get_activation(activation)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    ### 没用的函数
    def reset(self, dones=None):
        # print("AC_reset========================")
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    

    def update_distribution(self, observations):
        # EMLP 网络需要修改输入的格式
        observations = self.actor_in_field_type(observations)
        # 网络正向传播
        mean = self.actor(observations).tensor
        self.distribution = Normal(mean, mean*0. + torch.clamp(self.std, min=1e-3))

    def act(self, obs,  obs_history):
        # vel_est, latent = self.vae.sample(obs_history)
        # actor_obs = torch.cat((vel_est, latent, obs), dim = -1)
        self.update_distribution(obs)
        return self.distribution.sample()
   

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_history):
        # TODO: 后续要使用obs_history也许增强效果
        actor_input = self.actor_in_field_type(obs)
        # vel_est, latent = self.vae.inference(obs_history)
        # actor_obs = torch.cat((vel_est, latent, obs), dim = -1)
        actions_mean = self.actor(actor_input).tensor
        return actions_mean
    
    
    def evaluate(self, critic_observations):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
