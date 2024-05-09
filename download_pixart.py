import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download PixArt-alpha/PixArt-alpha --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/PixArt-alpha --local-dir-use-symlinks False')
