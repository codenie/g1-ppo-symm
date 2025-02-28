import torch

# 列表中有多个张量
l_inf_norms = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)]

# 使用 torch.stack 将列表中的张量堆叠成一个单独的张量
stacked_tensor = torch.stack(l_inf_norms)
print("Stacked Tensor:", stacked_tensor)  # [2., 3., 4.]

# 计算乘积
product = torch.prod(stacked_tensor)
print("Product:", product)  # 24.0 (2*3*4)