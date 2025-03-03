import escnn
import escnn.group
from escnn.group import CyclicGroup, DihedralGroup, DirectProductGroup, Group, Representation
from escnn.nn import FieldType, EquivariantModule, GeometricTensor

from morpho_symm.utils.algebra_utils import gen_permutation_matrix
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens

import torch
import torch.nn as nn
# from torch.distributions import Normal

import numpy as np
from typing import Union, List


### 本文件主要储存一些 "神经网络对称性" 构造过程中需要使用的辅助函数

# 设置numpy打印的精度
# np.set_printoptions(precision=1, suppress=True)

def add_repr_to_gspace(G:escnn.group.Group, 
                       permutation_conf:Union[List[int], np.ndarray],
                       reflection_conf:Union[List[int], np.ndarray],
                       name:str
    ):
    """将指定的变换添加到G这个group中, 供之后生成escnn的对称网络使用

    Args:
        G (escnn.group.Group): 一个escnn的group
        permutation_conf (Union[List[int], np.ndarray]): 位置变换矩阵
        reflection_conf (Union[List[int], np.ndarray]): 由1和-1指定的变换矩阵
        name (str): 这个变换的名字

    Returns:
        _type_: _description_
    """
    # 获取对应的配置文件，保证格式为 (n, 配置长度)
    permutation_conf = np.array(permutation_conf, dtype=int)
    reflection_conf = np.array(reflection_conf, dtype=float)
    if permutation_conf.ndim == 1: permutation_conf = permutation_conf[None]
    if reflection_conf.ndim == 1: reflection_conf = reflection_conf[None]
    
    # 获取配置个数, 长度
    (conf_num, conf_length) = permutation_conf.shape
    print(f"Conf name = {name}, length = {conf_length}")
    
    # 检查: 确保给的配置是正常的
    assert permutation_conf.shape == reflection_conf.shape, len(reflection_conf.shape)==2
    assert conf_num == len(G.generators)
    
    # 开始配置representations
    rep_joints = {G.identity: np.eye(conf_length, dtype=float)}
    for g_gen, perm, refx in zip(G.generators, permutation_conf, reflection_conf):
        refx = np.array(refx, dtype=float)
        rep_joints[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)

    # #将dict转化为representation.Representation
    rep_joints = group_rep_from_gens(G, rep_joints) 
    # 配置name
    rep_joints.name = name
    # 输入给G
    G.representations.update(**{name:rep_joints})
    return G

class SimpleEMLP(EquivariantModule):
    """EMLP网络构造

    Args:
        EquivariantModule (_type_): _description_
    """
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 hidden_dims = [256, 256, 256],
                 bias: bool = True,
                 actor: bool = True,
                 activation: str = "ReLU"):
        super().__init__()
        self.out_type = out_type
        gspace = in_type.gspace
        group = gspace.fibergroup
        
        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(len(hidden_dims)):
            layer_out_type = FieldType(gspace, [group.regular_representation] * int((hidden_dims[n] / group.order())))

            self.net.add_module(f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                             escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            self.net.add_module(f"act_{n}", self.get_activation(activation, layer_out_type))

            layer_in_type = layer_out_type

        if actor: 
            self.net.add_module(f"linear_{len(hidden_dims)}: in={layer_in_type.size}-out={out_type.size}",
                                escnn.nn.Linear(layer_in_type, out_type, bias=bias))
            self.extra_layer = None
        else:
            num_inv_features = len(layer_in_type.irreps)
            self.extra_layer = torch.nn.Linear(num_inv_features, out_type.size, bias=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x= self.net(x)
        if self.extra_layer:
            x = self.extra_layer(x.tensor)
        return x

    @staticmethod
    def get_activation(activation: str, hidden_type: FieldType) -> EquivariantModule:
        if activation.lower() == "relu":
            return escnn.nn.ReLU(hidden_type)
        elif activation.lower() == "elu":
            return escnn.nn.ELU(hidden_type)
        elif activation.lower() == "lrelu":
            return escnn.nn.LeakyReLU(hidden_type)
        else:
            raise NotImplementedError

    def evaluate_output_shape(self, input_shape):
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        sequential = nn.Sequential()
        for name, module in self.net.named_children():
            sequential.add_module(name, module.export())
        return sequential

def get_symm_tensor(data:torch.Tensor, G:escnn.group.Group, reprs:List[str])->torch.Tensor:
    """将data这个torch.tensor转化为对称后的结构

    Args:
        data (torch.Tensor): (Batch, N)
        G (escnn.group.Group): 一个escnn的群论group
        reprs (List): represetations的列表，会按照顺序对data进行对称计算

    Returns:
        torch.Tensor: 返回的是对称后的结果，与data保持相同的device
    """
    # 要求data数据是torch.tensor
    assert isinstance(data, torch.Tensor), data.ndim <= 2
    # 整理data的shape和 repr要是列表
    data = data[None] if data.ndim == 1 else data
    reprs = [reprs] if not isinstance(reprs, List) else reprs
    # 获取device
    device = data.device
    # 开始转换
    curr_ind = 0
    res = []
    for repr in reprs:
        res.append(
            (torch.as_tensor(G.representations[repr](G.elements[1]), dtype=torch.float32, device=device) \
                @ data.T[curr_ind:curr_ind+G.representations[repr].size]
                ).T 
        )
        curr_ind += G.representations[repr].size
    # 确保reprs和data的维度是吻合的
    assert curr_ind == data.shape[-1]
    # 返回结果
    return torch.concat(res, dim=-1)




