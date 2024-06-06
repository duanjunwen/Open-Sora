# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=16,
    frame_interval=3,
    image_size=(256, 256),
)

# Define acceleration
num_workers = 4
# dtype = "fp32"
# dtype = "fp16"
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
# plugin = "zero2-seq"
# plugin = "ddp"
sp_size = 1
# sp_size = 2

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="./pretrained_models/stdit/OpenSora/OpenSora-v1-16x256x256.pth",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    # from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl_rebase",
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
ckpt_every = 100
load = None

batch_size = 2
lr = 2e-5
grad_clip = 1.0

random_dataset = False

# wandb = True