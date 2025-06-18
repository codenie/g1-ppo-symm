import torch.nn as nn
import torch
from rsl_rl.utils.torch_utils import  get_activation,check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional as F

## 构造symmetry需要的辅助函数
from rsl_rl.utils.symm_utils import add_repr_to_gspace, SimpleEMLP, get_symm_tensor
from rsl_rl.utils.symm_utils import G, OBS_TRANS_NAME
import escnn
# import escnn.group
from escnn.group import CyclicGroup
from escnn.nn import FieldType, EquivariantModule, GeometricTensor

class VAESymmetry(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 encoder_hidden_dims = [512, 256, 128, 64],
                 decoder_hidden_dims = [64, 128, 256, 512]):
        
        super().__init__()
        
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent 

        # ????? why setting a no base sapce? for resigning?
        gspace = escnn.gspaces.no_base_space(G)
        group = gspace.fibergroup

        ## 创建encoder, 可以直接使用 G gspace 两个变量
        assert num_latent % 2 == 0, "要求mu和sigma的维度都是偶数"
        # 定义输入输出类型
        encoder_in_transitions = OBS_TRANS_NAME * num_history
        self.encoder_in_field_type = FieldType(gspace, [G.representations[name] for name in encoder_in_transitions])
        self.encoder_out_field_type = FieldType(gspace, [group.regular_representation] * int(num_latent / group.order()) * 2 )
        # 创建encoder网络
        self.encoder = SimpleEMLP(self.encoder_in_field_type, 
                                self.encoder_out_field_type,
                                hidden_dims = encoder_hidden_dims, 
                                activation = activation)
        ## 测试对称性
        # data = torch.rand(128, num_history*num_obs)
        # output = self.encoder(self.encoder_in_field_type(data)).tensor
        # d2 = get_symm_tensor(data, G, encoder_in_transitions)
        # o2 = self.encoder(self.encoder_in_field_type(d2)).tensor
        
        ## 创建decoder
        # 定义输入输出类型
        decoder_out_transitions = OBS_TRANS_NAME * 1
        self.decoder_in_field_type = FieldType(gspace, [group.regular_representation] * int(num_latent / group.order()))
        self.decoder_out_field_type = FieldType(gspace, [G.representations[name] for name in decoder_out_transitions])
        # self.decoder_out_field_type = self.encoder_in_field_type
        # 创建decoder网络
        self.decoder = SimpleEMLP(self.decoder_in_field_type,
                                self.decoder_out_field_type,
                                hidden_dims = decoder_hidden_dims, 
                                activation = activation)
    
    def encode(self,obs_history):
        '''  log_var
        '''
        encoded = self.encoder(self.encoder_in_field_type(obs_history)).tensor
        mu = encoded[...,:self.num_latent]
        var = encoded[...,self.num_latent:2*self.num_latent]
        return mu, var
        
    def decode(self, z):
        output = self.decoder(self.decoder_in_field_type(z)).tensor
        return output


    def forward(self,obs_history):
        mu, var = self.encode(obs_history)
        var_clipped = torch.clamp(var,-6.5,4.5) ###  var = (0.1, 5) #TODO: 这里两个数是什么意思
        z = self.reparameterize(mu, var_clipped)
        return z, mu, var_clipped
    
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar) ## log(var^2) -> std= exp(0.5*log(var^2))
        eps = torch.randn_like(std)  ### 标准正太分布
        return eps * std + mu
    

    def loss_fn(self, obs_history, next_obs, kld_weight = 1.0):
        z, mu, var = self.forward(obs_history)

        # Reconstruction loss
        recons = self.decode(z)
        recons_loss =F.mse_loss(recons, next_obs,reduction='none').mean(-1)
        # KL loss
        kld_loss = -0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = -1)

        loss = recons_loss + kld_weight * kld_loss

        return {
                    'loss': loss,
                    'recons_loss': recons_loss,
                    'kld_loss': kld_loss,
        }
    

    def sample(self,obs_history):
        z, _, _ = self.forward(obs_history)
        return z
    
    def inference(self,obs_history):
        _, mu, _ = self.forward(obs_history)
        return mu