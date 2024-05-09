import copy
import torch
import torch.nn as nn
import torch_musa
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing import assert_close
from opensora.models.stdit.stdit import STDiTBlock, STDiT, STDiT_XL_2
from opensora.models.stdit.stdit2 import STDiT2
from opensora.acceleration.parallel_states import set_sequence_parallel_group

def test_stditblock(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    
    B, N, C = 4, 64, 256
    
    stdit_block = STDiTBlock(hidden_size=256, num_heads=8, d_s=8, d_t=8).to(device)
    
    x = torch.randn(B, N, C, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, N, C, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    y.requires_grad = True
    y.retain_grad()
    timestep = torch.randn(B, 6, dtype=dtype).to(device) 
    
    output = stdit_block(x, y, timestep)
    print(f"stdit_block Shape {output.shape}\n {output}\n")
    
    output.mean().backward()

# stditblock correctness test
def test_stditblock_correctness(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    dtype = torch.float32
    B, N, C = 4, 64, 256
    
    x_cpu = torch.randn(B, N, C).to(dtype=dtype)
    x_cpu.requires_grad = True
    y_cpu = torch.randn(B, N, C).to(dtype=dtype)
    y_cpu.requires_grad = True
    timestep_cpu = torch.randn(B, 6, dtype=dtype)
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    
    dit_block_cpu = STDiTBlock(hidden_size=256, num_heads=8, d_s=8, d_t=8)
    dit_block_musa = copy.deepcopy(dit_block_cpu).to(device=device)
    
    # check param same
    for (name_cpu, param_cpu), (name_musa, param_musa) in zip(dit_block_cpu.named_parameters(), dit_block_musa.named_parameters()):
        assert_close(param_cpu, param_musa, check_device=False)
        print(f"{name_cpu}, {name_musa} pass")

    
    output_cpu = dit_block_cpu(x_cpu, y_cpu, timestep_cpu)
    output_musa = dit_block_musa(x_musa, y_musa, timestep_musa)
    
    print(f"stdit_block_cpu Shape {output_cpu.shape}\n {output_cpu}\n")
    print(f"stdit_block_musa Shape {output_musa.shape}\n {output_musa}\n")
    
    output_cpu.mean().backward()
    output_musa.mean().backward()
    
    assert_close(output_cpu, output_musa, check_device=False)
    
    
def test_stdit(device):
    # # q = torch.rand(1, 4096, 4, 32, dtype=torch.float16, device="musa")
    # # k = torch.rand(1, 120,  4, 32, dtype=torch.float16, device="musa")
    # # v = torch.rand(1, 120,  4, 32, dtype=torch.float16, device="musa")
    
    # q = torch.rand(1, 4, 4096, 32, dtype=torch.float16, device="musa")
    # k = torch.rand(1, 4,  120, 32, dtype=torch.float16, device="musa")
    # v = torch.rand(1, 4,  120, 32, dtype=torch.float16, device="musa")
    
    # # mask = torch.rand(128, 64, dtype=torch.float16, device="musa")
    # # print(mask)
    # out = F.scaled_dot_product_attention(q,k,v, attn_mask=None)
    # print(out)
    
    
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    # Variational Auto-Encoder
    stdit = STDiT(hidden_size=128, num_heads=4,input_size=(16, 32, 32)).to(device)
    
    # x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
    # timestep (torch.Tensor): diffusion time steps; of shape [B]
    # y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
    # mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

    # x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    # x.requires_grad = True
    # y = torch.randn(B, 1, N_token, C, dtype=dtype).to(device)  #  [B, 1, N_token, C]
    # y.requires_grad = True
    # timestep = torch.randn(B, dtype=dtype).to(device) 
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token]
    # mask = None
    
    x_stdit = stdit(x=x, timestep=timestep, y=y, mask=mask)

    print(f"STDiT Shape {x_stdit.shape}\n {x_stdit}\n")
    
    x_stdit.mean().backward()


def test_stdit_correctness(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 64, 16, 16
    N_token = 120
    device = torch.device(device)
    dtype = torch.float32
    # Variational Auto-Encoder
    stdit_cpu = STDiT(input_size=(16, 32, 32))
    stdit_musa = copy.deepcopy(stdit_cpu).to(device=device)
    
    # x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
    # timestep (torch.Tensor): diffusion time steps; of shape [B]
    # y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
    # mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

    x_cpu = torch.randn(B, C, T, H, W, dtype=dtype)  # (B, C, T, H, W)
    x_cpu.requires_grad = True
    y_cpu = torch.randn(B, 1, N_token, 4096, dtype=dtype)  #  [B, 1, N_token, C]
    y_cpu.requires_grad = True
    timestep_cpu = torch.randn(B, dtype=dtype) 
    mask_cpu = torch.randn(B, N_token, dtype=dtype) # [B, N_token]
    # mask_cpu = None
    
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    y_musa = copy.deepcopy(y_cpu).to(device=device)
    timestep_musa = copy.deepcopy(timestep_cpu).to(device=device)
    mask_musa = copy.deepcopy(mask_cpu).to(device=device)
    # mask_musa = None
    
    x_stdit_cpu = stdit_cpu(x=x_cpu, timestep=timestep_cpu, y=y_cpu, mask=mask_cpu)
    x_stdit_musa = stdit_musa(x=x_musa, timestep=timestep_musa, y=y_musa, mask=mask_musa)

    print(f"STDiT Shape {x_stdit_cpu.shape}\n {x_stdit_cpu}\n")
    print(f"STDiT musa Shape {x_stdit_musa.shape}\n {x_stdit_musa}\n")
    
    x_stdit_cpu.mean().backward()
    x_stdit_musa.mean().backward()


def test_stdit_xl_2(device):
    device = torch.device(device)
    torch.manual_seed(1024)
    
    # N, T, D = 4, 64, 256
    B, C, T, H, W = 1, 4, 4, 16, 16 # T=4 
    N_token = 120
    caption_channels = 4096
    device = torch.device(device)
    dtype = torch.float32
    
    # stdit_xl_2 = STDiT_XL_2(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth").to(device)
    stdit_xl_2 = STDiT_XL_2(from_pretrained="./pretrained_models/PixArt-alpha/PixArt-XL-2-512x512.pth").to(device)
    
    
    x = torch.randn(B, C, T, H, W, dtype=dtype).to(device)  # (B, C, T, H, W)
    x.requires_grad = True
    y = torch.randn(B, 1, N_token, 4096, dtype=dtype).to(device)   #  [B, caption_channels=512]
    timestep = torch.randn(B, dtype=dtype).to(device) #  [B, ]
    mask = torch.randn(B, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 1
    # mask = torch.randn(256, N_token, dtype=dtype).to(device)  # [B, N_token] # Method 2
    # mask = None
    x_stdit = stdit_xl_2(x=x, timestep=timestep, y=y, mask=mask)

    print(f"STDiT Shape {x_stdit.shape}\n {x_stdit}\n")
    
    x_stdit.mean().backward()
    
    

if __name__ == "__main__":
    device = "musa"

    # test_stditblock(device)
    # test_stditblock_correctness(device)
    
    # test_stdit(device)
    # test_stdit_correctness(device)
    
    test_stdit_xl_2(device)