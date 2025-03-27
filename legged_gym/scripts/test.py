import numpy as np
import torch
import torch.nn as nn


a = torch.tensor([0,0,4],dtype=torch.long)

b = torch.tensor([[2,3,4,5],[6,7,8,9],[10,11,12,13]],dtype=float) 
c = torch.tensor([[14,15,16,17],[18,19,20,21],[22,23,24,25]],dtype=float) 

d = torch.where(a.unsqueeze(-1)>2,b,c)

 
print(d)