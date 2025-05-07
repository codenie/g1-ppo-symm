
## 环境配置:

- 建议 Python 3.8 的环境
- 先安装isaacgym

```
cd rsl_rl
pip install -e .

cd legged_gym
pip install -e .

cd MorphoSymm
pip install -e .
```
之后可能还少一些常见包，比如 `tensoboard`，手动安装即可。

可能需要手动切换numpy的版本，建议在安装完成后
`pip install numpy==1.23.5`

注意MorphoSymm库比较特殊，依赖pinocchio，会使用pip自动安装，如果出错可以使用conda等手动安装。

## TODO
- 写一个收集policy权重文件.pth的脚本；一键把需要的.pth文件都复制到一个目录下打包，方便上真机测试

## 文件结构
- legged_gym: g1的训练环境定义
- MorphoSymm: 主要是用于构建群论, 调用ESCNN构建对称神经网络, 是一个辅助库
- rsl_rl: PPO算法的定义和训练
    - rsl_rl/rsl_rl/modules/actor_critic_symmetry.py 这个文件定义了这个Actor Critic网络
- sync: 多服务器同步数据的脚本文件


