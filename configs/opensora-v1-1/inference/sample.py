num_frames = 16
frame_interval = 3
fps = 24
image_size = (240, 426)
multi_resolution = "STDiT2"

# Define model
model = dict(
    type="STDiT2-XL/2",
    # from_pretrained="hpcai-tech/OpenSora-STDiT-v2-stage3",
    from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v2-stage3/model.safetensors",
    # from_pretrained="/home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v3/model.safetensors",
    # from_pretrained="./pretrained_models/stdit/OpenSora/OpenSora-v1-HQ-16x512x512.pth",
    input_sq_size=512,
    qk_norm=True,
    # enable_flashattn=True,
    # enable_layernorm_kernel=True,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    # from_pretrained="stabilityai/sd-vae-ft-ema",
    from_pretrained="./pretrained_models/stabilityai/sd-vae-ft-ema",
    cache_dir=None,  # "/mnt/hdd/cached_models",
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    # from_pretrained="DeepFloyd/t5-v1_1-xxl",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
    cache_dir=None,  # "/mnt/hdd/cached_models",
    model_max_length=200,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)
# dtype = "bf16"
# dtype = "fp16"
dtype = "fp32"

# Condition
prompt_path = "./assets/texts/t2v_samples.txt"
prompt = None  # prompt has higher priority than prompt_path

# Others
batch_size = 1
seed = 42
save_dir = "./samples/samples-v1-1/"
