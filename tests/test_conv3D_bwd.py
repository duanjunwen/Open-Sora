import torch
import torch_musa
import torch.nn as nn

dtype = dtype=torch.bfloat16
m = nn.Conv3d(16, 33, 3, stride=2, dtype=dtype).musa()
input = torch.randn(20, 16, 10, 50, 100).to(device='musa', dtype=dtype)
input.requires_grad=True
output = m(input)
output.sum().backward()