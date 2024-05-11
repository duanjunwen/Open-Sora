import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download openai/clip-vit-large-patch14-336 --local-dir /home/duanjunwen/LCProj/Open-Sora/pretrained_models/clip-vit-large-patch14-336/ --local-dir-use-symlinks False')
