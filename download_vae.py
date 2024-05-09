import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.system('huggingface-cli download --resume-download fusing/autoencoder-kl-dummy --local-dir /home/duanjunwen/LCProj/Open-Sora/pretrained_models/fusing/autoencoder-kl-dummy --local-dir-use-symlinks False')
# Open-Sora/pretrained_models/t5_ckpts/t5-v1_1-xxl
os.system('huggingface-cli download --resume-download stabilityai/sd-vae-ft-ema --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stabilityai/sd-vae-ft-ema --local-dir-use-symlinks False')
