# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=64,
    frame_interval=3,
    image_size=(512, 512),
)

# Define acceleration
num_workers = 4
# dtype = "bf16"
dtype = "fp16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=2 / 3,
    # from_pretrained=None,
    from_pretrained="./pretrained_models/stdit/OpenSora/OpenSora-v1-HQ-16x512x512.pth",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    # from_pretrained="stabilityai/sd-vae-ft-ema",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=64,
)
text_encoder = dict(
    type="t5",
    # from_pretrained="DeepFloyd/t5-v1_1-xxl",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1
log_every = 10
ckpt_every = 250
load = None

# batch_size = 4
# lr = 2e-5
# grad_clip = 1.0

batch_size = 1
lr = 2e-5
grad_clip = 1.0

random_dataset = True
benchmark_num_steps = 4
num_ckpt_blocks = 28 # STDIT total 28
cfg_name = "64x512x512"