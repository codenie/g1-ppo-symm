import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
# from torch.nn.modules import rnn
# from rsl_rl.modules.state_estimator import VAE
from rsl_rl.modules.vae_symmetry import VAESymmetry
# from rsl_rl.utils.torch_utils import init_orhtogonal

## 构造symmetry需要的辅助函数
from rsl_rl.utils.symm_utils import add_repr_to_gspace, SimpleEMLP, get_symm_tensor
from rsl_rl.utils.symm_utils import G,  OBS_TRANS_NAME, ACT_TRANS_NAME, CRITIC_OBS_TRANS_NAME, CRITIC_VAL_TRANS_NAME

import escnn
# import escnn.group
from escnn.group import CyclicGroup
from escnn.nn import FieldType, EquivariantModule, GeometricTensor


class ActorCriticSymmetryVae(nn.Module):
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
        activation_fn = get_activation(activation)

        ### actor,  critic ---------------------
        mlp_input_dim_a = num_obs_step + num_vae
        mlp_input_dim_c = num_critic_obs
        
        gspace = escnn.gspaces.no_base_space(G)
        group = gspace.fibergroup
        
        #### Actor
        ## 原始Actor
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # # 定义输入输出类型
        # # actor的输入为vae的latent和当前的obs
        # self.actor_in_field_type = FieldType(gspace, [group.regular_representation] * int(num_vae / group.order()) + 
        #                                      [G.representations[name] for name in OBS_TRANS_NAME])
        # self.actor_out_field_type = FieldType(gspace, [G.representations[name] for name in ACT_TRANS_NAME])
        # # 创建encoder网络
        # self.actor = SimpleEMLP(self.actor_in_field_type, 
        #                         self.actor_out_field_type,
        #                         hidden_dims = actor_hidden_dims, 
        #                         activation = activation)
        
        # #### Critic
        # # 定义输入输出类型
        # self.critic_in_field_type = FieldType(gspace, [G.representations[name] for name in CRITIC_OBS_TRANS_NAME])
        # self.critic_out_field_type = FieldType(gspace, [G.representations[name] for name in CRITIC_VAL_TRANS_NAME])
        # # 创建encoder网络
        # self.critic = SimpleEMLP(self.critic_in_field_type, 
        #                         self.critic_out_field_type,
        #                         hidden_dims = critic_hidden_dims, 
        #                         activation = activation)
        
        ## 原始Critic
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


        self.vae = VAESymmetry(
            num_obs= num_obs_step,
            num_history= num_history,
            num_latent= num_vae,
            activation=activation,
        )
        
        print(f"VAE MLP: {self.vae}")

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
        # mean = self.actor(self.actor_in_field_type(observations)).tensor
        mean = self.actor(observations) # w/o symmetry
        self.distribution = Normal(mean, mean*0. + torch.clamp(self.std, min=1e-3))

    def act(self, obs,  obs_history):
        # ## TODO: 确认obs等于obs_history[:,332], 检查程序正确性
        # if torch.all(obs == obs_history[:, :332]) == False:
        #     print("[warning] obs != obs_history[:, :332]")
        # else:
        #     print("[info] obs == obs_history[:, :332]")
        
        latent = self.vae.sample(obs_history)
        actor_obs = torch.cat((latent, obs), dim = -1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, obs_history):
        latent = self.vae.inference(obs_history)
        actor_obs = torch.cat((latent, obs), dim = -1)
        # actions_mean = self.actor(self.actor_in_field_type(actor_obs)).tensor
        actions_mean = self.actor(actor_obs) # actor w/o symmetry
        return actions_mean
    
    
    def evaluate(self, critic_observations):
        # value = self.critic(self.critic_in_field_type(critic_observations)).tensor
        value = self.critic(critic_observations) # critic w/o symmetry
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
