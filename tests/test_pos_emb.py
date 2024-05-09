import torch
import torch_musa
from opensora.models.layers.blocks import PositionEmbedding2D, get_2d_sincos_pos_embed

D = 8
SCALE = 2.0
from torch.testing import assert_close


def get_spatial_pos_embed(x, hidden_size, h, w, scale, base_size=None):
    pos_embed = get_2d_sincos_pos_embed(
        hidden_size,
        (h, w),
        scale=scale,
        base_size=base_size,
    )
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
    return pos_embed.to(device=x.device, dtype=x.dtype)


def test_pos_emb(device):
    # just a placeholder to get the device and dtype
    dtype = torch.bfloat16 # torch.float, torch.float16, torch.bfloat16
    x = torch.empty(1, dtype=dtype, device=device)
    pos_embedder = PositionEmbedding2D(
        D,
        # max_position_embeddings=8,
        # scale=SCALE,
    ).to(device=device, dtype=dtype)
    output = pos_embedder(x, 8, 7)
    print(f"Shape {output.shape}\n {output}\n")
    output = pos_embedder(x, 15, 16)
    print(f"Shape {output.shape}\n {output}\n")
    output = pos_embedder(x, 30, 20, base_size=2)
    print(f"Shape {output.shape}\n {output}\n")
    output = pos_embedder(x, 30, 20, base_size=2)
    print(f"Shape {output.shape}\n {output}\n")

def compare_pos_emb_correctnes():
    dtype = torch.bfloat16 # torch.float, torch.float16, torch.bfloat16
    device="musa"
    x_cpu = torch.empty(1, dtype=dtype)
    x_musa = torch.empty(1, dtype=dtype, device=device)
    pos_embedder_cpu = PositionEmbedding2D(
        D,
        # max_position_embeddings=8,
        # scale=SCALE,
    ).to(dtype=dtype)
    pos_embedder_musa = PositionEmbedding2D(
        D,
        # max_position_embeddings=8,
        # scale=SCALE,
    ).to(device=device, dtype=dtype)
    output_cpu = pos_embedder_cpu(x_cpu, 30, 20, base_size=2)
    output_musa = pos_embedder_musa(x_musa, 30, 20, base_size=2)
    assert_close(output_cpu, output_musa, check_device=False)
    

def test_pos_emb_correctness(device):
    # just a placeholder to get the device and dtype
    dtype = torch.bfloat16 # torch.float, torch.float16, torch.bfloat16
    x = torch.empty(1, dtype=dtype, device=device)
    pos_embedder = PositionEmbedding2D(
        D,
        # max_position_embeddings=8,
        # scale=SCALE,
    ).to(device=device, dtype=dtype)
    output = pos_embedder(x, 8, 7)
    target = get_spatial_pos_embed(x, D, 8, 7, SCALE)
    assert_close(output, target)
    output = pos_embedder(x, 15, 16)
    target = get_spatial_pos_embed(x, D, 15, 16, SCALE)
    assert_close(output, target)
    output = pos_embedder(x, 30, 20, base_size=2)
    target = get_spatial_pos_embed(x, D, 30, 20, SCALE, base_size=2)
    assert_close(output, target)
    # test cache
    output = pos_embedder(x, 30, 20, base_size=2)
    target = get_spatial_pos_embed(x, D, 30, 20, SCALE, base_size=2)
    assert_close(output, target)
    assert pos_embedder._get_cached_emb.cache_info().hits >= 1

if __name__ == "__main__":
    print("Test Pos Embedding")
    test_pos_emb("musa")
    print("Compare Pos Embedding Correctness")
    compare_pos_emb_correctnes()  # percision error
    # print("Test Pos Embedding Correctness")
    # test_pos_emb_correctness("musa")  # percision error
