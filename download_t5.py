import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.system('huggingface-cli download --resume-download DeepFloyd/t5-v1_1-xxl --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/t5_ckpts/t5-v1_1-xxl --local-dir-use-symlinks False')
# Open-Sora/pretrained_models/t5_ckpts/t5-v1_1-xxl
os.system('huggingface-cli download --resume-download openai/clip-vit-base-patch32 --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/text_encoder/clip-vit-base-patch32 --local-dir-use-symlinks False')
