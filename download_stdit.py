import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download hpcai-tech/OpenSora-STDiT-v1-16x256x256 --local-dir /home/duanjunwen/LCProj/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v1-16x256x256 --local-dir-use-symlinks False')
# os.system('huggingface-cli download --resume-download hpcai-tech/OpenSora-STDiT-v1-HQ-16x512x512 --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v1-HQ-16x512x512 --local-dir-use-symlinks False')
