# Dataset settings
dataset = dict(
    type="BatchFeatureDataset",
)

grad_checkpoint = True

# Acceleration settings
num_workers = 8
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    freeze_y_embedder=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="pretrained_models/vae-pipeline",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
mask_ratios = {
    "random": 0.2,
    "intepolate": 0.01,
    "quarter_random": 0.01,
    "quarter_head": 0.01,
    "quarter_tail": 0.01,
    "quarter_head_tail": 0.01,
    "image_random": 0.05,
    "image_head": 0.1,
    "image_tail": 0.05,
    "image_head_tail": 0.05,
}

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1
log_every = 10
ckpt_every = 500

# optimization settings
load = None
grad_clip = 1.0
lr = 2e-4
ema_decay = 0.99
adam_eps = 1e-15
