import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download facebook/DiT-XL-2-256 --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/dit/DiT-XL_2 --local-dir-use-symlinks False')
# os.system('huggingface-cli download --resume-download facebook/DiT-XL-2-512 --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/dit/DiT-XL_2X2 --local-dir-use-symlinks False')
