import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download hpcai-tech/Open-Sora --local-dir /home/duanjunwen/LCProj/Open-Sora/pretrained_models/t5_ckpts/t5-v1_1-xxl --local-dir-use-symlinks False')
# Open-Sora/pretrained_models/t5_ckpts/t5-v1_1-xxl