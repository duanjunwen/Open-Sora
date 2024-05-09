import copy
import torch
import pytest
from torch.testing import assert_close
# from colossalai.accelerator import get_accelerator
from colossalai.utils import get_current_device
from rotary_embedding_torch import RotaryEmbedding
from opensora.models.layers.blocks import Attention


# B, S, H = 7488, 1, 1152
# B, S, H = 32, 234, 1152
B, S, H = 128, 32, 1152
N, D = 16, 72

def run_attn(enable_flashattn: bool, device):
    # get_accelerator().reset_peak_memory_stats()
    # device = get_current_device()
    rope = RotaryEmbedding(D).to(device=device, dtype=torch.bfloat16)
    attn = Attention(
        H,
        N,
        qkv_bias=True,
        rope=rope.rotate_queries_or_keys,
        enable_flashattn=enable_flashattn,
    ).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(B, S, H, device=device, dtype=torch.bfloat16).requires_grad_()
    y = attn(x)
    print(f"y {y}")
    y.mean().backward()
    # print(f"Peak memory: {get_accelerator().max_memory_allocated() / 1024**2:.2f} MB")

def run_atten_correctness(enable_flashattn: bool):
    dtype = torch.bfloat16
    device="musa"
    torch.manual_seed(1024)
    
    rope_cpu = RotaryEmbedding(D).to(dtype=dtype)
    rope_musa = copy.deepcopy(rope_cpu).to(device=device)
    
    attn_cpu = Attention(
        H,
        N,
        qkv_bias=True,
        rope=rope_cpu.rotate_queries_or_keys,
        enable_flashattn=enable_flashattn,
    ).to(dtype=dtype)
    attn_musa = copy.deepcopy(attn_cpu).to(device=device)
    attn_musa.rotary_emb = rope_musa.rotate_queries_or_keys

    x_cpu = torch.randn(B, S, H, dtype=torch.bfloat16).requires_grad_()
    x_musa = copy.deepcopy(x_cpu).to(device=device)
    
    output_cpu = attn_cpu(x_cpu)
    output_musa = attn_musa(x_musa)
    
    assert_close(output_cpu, output_musa, check_device=False)

if __name__ == "__main__":
    print("Use flashattn")
    run_attn(True, "musa")
    print("No flashattn")
    run_attn(False, "musa")
    
    print("Use flashattn Correctness")
    run_atten_correctness(True)
    print("No flashattn Correctness")
    run_atten_correctness(False)
