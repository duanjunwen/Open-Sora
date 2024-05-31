import time
import thop
from copy import deepcopy

import colossalai
import torch
import torch_musa
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn

from opensora.acceleration.shardformer.policy.t5_encoder import T5EncoderPolicy
from opensora.models.text_encoder.t5 import T5Embedder

# class ProfileModule(torch.nn.Module):
# 	def __init__(self, module):
# 		self.module = module

# 	def count_model(self, text):
# 		self.module.encode(text)

class ProfileModule(torch.nn.Module):
	def __init__(self, module, fn='encode'):
		super().__init__()
		self.module = module
		self.forward_func = getattr(module, fn)

	def forward(self, *args):
		return self.forward_func(*args)

def run_t5_encoder(rank, world_size, port):
    colossalai.launch({}, rank=rank, world_size=world_size, port=port, host="localhost", backend="mccl")

    # t5 embedder
    t5_path = "./pretrained_models/t5_ckpts/"
    pretrain_path = "./pretrained_models/t5_ckpts/t5-v1_1-xxl"
    # hf_t5 = T5Embedder(device="musa", local_cache=True, cache_dir=t5_path, torch_dtype=torch.float)
    hf_t5 = T5Embedder(device="musa", cache_dir=None, from_pretrained=pretrain_path, torch_dtype=torch.float16)
    sf_t5 = deepcopy(hf_t5)

    # create huggingface model as normal
    shard_config = ShardConfig(
        tensor_parallel_process_group=None,
        pipeline_stage_manager=None,
        enable_tensor_parallelism=False,
        enable_fused_normalization=False,
        enable_flash_attention=False,
        enable_jit_fused=True,
        enable_sequence_parallelism=False,
        enable_sequence_overlap=False,
    )
    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, _ = shard_former.optimize(sf_t5.model, policy=T5EncoderPolicy())
    sf_t5.model = sharded_model

    # test t5 embedder
    texts = ["Who is the best player in the history of NBA?", "How to study computer science?"]
    for i in range(20):
        hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
        sf_embs, sf_masks = sf_t5.get_text_embeddings(texts)
        print(f"hf_embs {hf_embs} sf_embs {sf_embs}")

    # check accuracy
    assert torch.allclose(hf_embs, sf_embs, rtol=1e-4, atol=1e-5), f"{hf_embs} \nvs\n{sf_embs}"
    assert torch.allclose(hf_masks, sf_masks), f"{hf_masks} \nvs\n{sf_masks}"

    # measure perf
    # torch.cuda.synchronize()
    torch.musa.synchronize()
    hf_start = time.time()
    for i in range(20):
        hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
    # torch.cuda.synchronize()
    torch.musa.synchronize()
    hf_end = time.time()

    # convert sf to fp16
    # hf_t5.model = hf_t5.model.half() # RuntimeError: "clamp_cpu" not implemented for 'Half'
    hf_t5.model = hf_t5.model
    # torch.cuda.synchronize()
    torch.musa.synchronize()
    sf_start = time.time()
    for i in range(20):
        hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
    # torch.cuda.synchronize()
    torch.musa.synchronize()
    sf_end = time.time()

    print(f"[Performance] native: {hf_end - hf_start}s, shardformer: {sf_end - sf_start} s")


def run_t5_flops(rank, world_size, port):
    colossalai.launch({}, rank=rank, world_size=world_size, port=port, host="localhost", backend="mccl")

    # t5 embedder
    t5_path = "./pretrained_models/t5_ckpts/"
    pretrain_path = "./pretrained_models/t5_ckpts/t5-v1_1-xxl"
    # hf_t5 = T5Embedder(device="musa", local_cache=True, cache_dir=t5_path, torch_dtype=torch.float)
    hf_t5 = T5Embedder(device="musa", cache_dir=t5_path, from_pretrained=pretrain_path, torch_dtype=torch.float16)
    sf_t5 = deepcopy(hf_t5)

    # create huggingface model as normal
    shard_config = ShardConfig(
        tensor_parallel_process_group=None,
        pipeline_stage_manager=None,
        enable_tensor_parallelism=False,
        enable_fused_normalization=False,
        enable_flash_attention=False,
        enable_jit_fused=True,
        enable_sequence_parallelism=False,
        enable_sequence_overlap=False,
    )
    shard_former = ShardFormer(shard_config=shard_config)
    sharded_model, _ = shard_former.optimize(sf_t5.model, policy=T5EncoderPolicy())
    sf_t5.model = sharded_model
    
    # test t5 embedder
    texts = ["Who is the best player in the history of NBA?", "How to study computer science?"]
    hf_embs, hf_masks = hf_t5.get_text_embeddings(texts)
    sf_embs, sf_masks = sf_t5.get_text_embeddings(texts)
    
    t5_profile_model = ProfileModule(hf_t5)
    t5_flops, t5_params = thop.profile(model=t5_profile_model, inputs=(texts,), custom_ops={ProfileModule:ProfileModule.count_model})
    # t5_sf_flops, t5_sf_params = thop.profile(model=sf_t5, inputs=(texts), custom_ops={ProfileModule:ProfileModule.forward})
    print(f"native t5 flops {t5_flops}; native t5 params {t5_params}")
    # print(f"shardformer t5 flops {t5_sf_flops}; shardformer t5 params {t5_sf_params}")
    # ==============================

def run_linear_flops(rank, world_size, port):
    m = T5Embedder(device="musa", cache_dir="./pretrained_models/t5_ckpts/", from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl", torch_dtype=torch.float16)
    input = ["Who is the best player in the history of NBA?", "How to study computer science?"]
    linearlayer = ProfileModule(module=m, fn='get_text_embeddings')
    flops, params = thop.profile(model=linearlayer, inputs=(input,), custom_ops={linearlayer:linearlayer.forward})
    print(flops, params)


def test_t5_encoder():
    spawn(run_t5_encoder)
    # spawn(run_t5_flops)
    # spawn(run_linear_flops)

if __name__ == "__main__":
    test_t5_encoder()
