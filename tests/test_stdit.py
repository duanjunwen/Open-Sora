import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.testing import assert_close

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from opensora.models.stdit.stdit import STDiTBlock, STDiT, STDiT_XL_2
from opensora.utils.train_utils import set_seed
from opensora.models.stdit.stdit2 import STDiT2
from opensora.acceleration.parallel_states import set_sequence_parallel_group

def test_stdit_single_op(device):
    device = torch.device(device)
    # torch.manual_seed(1024)
    set_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 4, 16, 16 # T=4 
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    
    # stdit_xl_2 = STDiT_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    # stdit_xl_2 = STDiT_XL_2(from_pretrained="./pretrained_models/PixArt-alpha/PixArt-XL-2-512x512.pth").to(device)
    stdit_xl_2 = STDiT_XL_2(from_pretrained="./pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)

    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 1
    # mask = torch.randn(256, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 2
    # mask = None
    torch.save(x , f"./dataset/assert_closed/torch_tensor/single_op_stdit_input.txt")
    torch.save(stdit_xl_2.state_dict() , f"./dataset/assert_closed/torch_tensor/single_op_stdit_param_init.txt")
    
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask)
    
    torch.save(x_stdit , f"./dataset/assert_closed/torch_tensor/single_op_stdit_output.txt")
    
    # x_stdit.mean().backward()
    # torch.save(stdit_xl_2.state_dict() , f"./dataset/assert_closed/torch_tensor/single_op_stdit_param_bwd.txt")

if __name__ == "__main__":
    device = "cuda"
    test_stdit_single_op(device)
