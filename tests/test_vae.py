import copy
import torch
import torch.nn as nn
from torch.testing import assert_close
# import sys
# sys.path.append('./Open-Sora/opensora/models/vae')
from opensora.models.vae import VideoAutoencoderKL, VideoAutoencoderKLTemporalDecoder


# TODO: assert vae input & output and param init; (no param bwd)
def test_vae_single_op():
    device = torch.device("cuda")
    dtype = torch.float
    torch.manual_seed(1024)
    # Test Variational Auto-Encoder; already download; 
    vae_kl = VideoAutoencoderKL(from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema").to(device)
    
    x_encoder_input = torch.randn(1, 3, 4, 32, 32, dtype=dtype).to(device)  # (B, C, T, H, W)
    x_encoder_input.requires_grad = True
    x_encoder_input.retain_grad()
    
    torch.save(x_encoder_input , f"./dataset/assert_closed/torch_tensor/single_op_vae_input.txt")
    torch.save(vae_kl.state_dict() , f"./dataset/assert_closed/torch_tensor/single_op_vae_param_init.txt")
    
    x_encoder_output = vae_kl.encode(x_encoder_input)
    
    torch.save(x_encoder_output , f"./dataset/assert_closed/torch_tensor/single_op_vae_output.txt")


if __name__ == "__main__":
    test_vae_single_op()