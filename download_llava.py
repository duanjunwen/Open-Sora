import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download liuhaotian/llava-v1.6-mistral-7b --local-dir /home/duanjunwen/LCProj/Open-Sora/pretrained_models/llava/ --local-dir-use-symlinks False')
