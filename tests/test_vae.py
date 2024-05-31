import copy
import thop
import torch
import torch.nn as nn
import torch_musa
from torch.testing import assert_close
# import sys
# sys.path.append('./Open-Sora/opensora/models/vae')
from opensora.models.vae import VideoAutoencoderKL, VideoAutoencoderKLTemporalDecoder

class ProfileModule(torch.nn.Module):
	def __init__(self, module, fn='encode'):
		super().__init__()
		self.module = module
		self.forward_func = getattr(module, fn)

	def forward(self, *args):
		return self.forward_func(*args)

# VideoAutoencoderKL
def test_vaekl():
    device = torch.device("musa")
    dtype = torch.float32
    torch.manual_seed(1024)
    # Test Variational Auto-Encoder; already download; 
    vae_kl = VideoAutoencoderKL(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stabilityai/sd-vae-ft-ema").to(device)
    
    x_encoder_input = torch.randn(1, 3, 4, 32, 32, dtype=dtype).to(device)  # (B, C, T, H, W)
    x_encoder_input.requires_grad = True
    x_encoder_input.retain_grad()
    x_encoder = vae_kl.encode(x_encoder_input)
    print(f"VideoAutoencoderKL encoder shape {x_encoder.shape}\n {x_encoder}\n")
    
    x_decoder_input = torch.randn(1, 4, 4, 32, 32, dtype=dtype).to(device)  # (B, C, T, H, W)
    x_decoder_input.requires_grad = True
    x_decoder_input.retain_grad()
    x_decoder = vae_kl.decode(x_decoder_input)
    print(f"VideoAutoencoderKL decoder shape {x_decoder.shape}\n {x_decoder}\n")


def test_vaekl_correctness():
    device = torch.device("musa")
    dtype = torch.float32
    torch.manual_seed(1024)
    # Test Variational Auto-Encoder; already download; 
    vae_kl_cpu = VideoAutoencoderKL(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stabilityai/sd-vae-ft-ema")
    vae_kl_musa = copy.deepcopy(vae_kl_cpu).to(device=device)
    
    # Encoder
    x_encoder_input_cpu = torch.randn(1, 3, 4, 32, 32, dtype=dtype)  # (B, C, T, H, W)
    x_encoder_input_cpu.requires_grad = True
    x_encoder_input_cpu.retain_grad()
    
    x_encoder_input_musa = copy.deepcopy(x_encoder_input_cpu).to(device=device)
    
    x_encoder_cpu = vae_kl_cpu.encode(x_encoder_input_cpu)
    x_encoder_musa = vae_kl_musa.encode(x_encoder_input_musa)
    # assert_close(x_encoder_cpu, x_encoder_musa, check_device=False)
    
    # Decoder
    x_decoder_input_cpu = torch.randn(1, 4, 4, 32, 32, dtype=dtype)  # (B, C, T, H, W)
    x_decoder_input_cpu.requires_grad = True
    x_decoder_input_cpu.retain_grad()
    
    x_decoder_input_musa = copy.deepcopy(x_decoder_input_cpu).to(device=device)
    
    x_decoder_cpu = vae_kl_cpu.decode(x_decoder_input_cpu)
    x_decoder_musa = vae_kl_musa.decode(x_decoder_input_musa)
    
    assert_close(x_decoder_cpu, x_decoder_musa, check_device=False)

# VideoAutoencoderKLTemporalDecoder 
# no pretrained model metioned
def test_vaekl_td():
    device = torch.device("musa")
    dtype = torch.float32
    torch.manual_seed(1024)
    # Test Variational Auto-Encoder; already download; 
    vae_kl = VideoAutoencoderKLTemporalDecoder().to(device)
    
    x_encoder_input = torch.randn(1, 3, 4, 32, 32, dtype=dtype).to(device)  # (B, C, T, H, W)
    x_encoder_input.requires_grad = True
    x_encoder_input.retain_grad()
    x_encoder = vae_kl.encode(x_encoder_input)
    print(f"VideoAutoencoderKL encoder shape {x_encoder.shape}\n {x_encoder}\n")
    
    
    x_decoder_input = torch.randn(1, 4, 4, 32, 32, dtype=dtype).to(device)  # (B, C, T, H, W)
    x_decoder_input.requires_grad = True
    x_decoder_input.retain_grad()
    x_decoder = vae_kl.decode(x_decoder_input)
    print(f"VideoAutoencoderKL decoder shape {x_decoder.shape}\n {x_decoder}\n")

def run_vae_flops():
    device = torch.device("musa")
    dtype = torch.float32
    torch.manual_seed(1024)
    # Test Variational Auto-Encoder; already download; 
    vae_kl = VideoAutoencoderKL(from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stabilityai/sd-vae-ft-ema").to(device)
    
    x_encoder_input = torch.randn(1, 3, 4, 32, 32, dtype=dtype).to(device)  # (B, C, T, H, W)
    x_encoder_input.requires_grad = True
    x_encoder_input.retain_grad()
    x_encoder = vae_kl.encode(x_encoder_input)
    # print(f"VideoAutoencoderKL encoder shape {x_encoder.shape}\n {x_encoder}\n")
    
    vae_encode = ProfileModule(module=vae_kl, fn='encode')
    flops, params = thop.profile(model=vae_encode, inputs=(x_encoder_input,), custom_ops={vae_encode:vae_encode.forward})
    print(flops, params)
    
    

if __name__ == "__main__":
    # test_vaekl()
    # test_vaekl_correctness()
    run_vae_flops()
    # test_vaekl_td()