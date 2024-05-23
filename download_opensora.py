import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# os.system('huggingface-cli download --resume-download hpcai-tech/Open-Sora --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora --local-dir-use-symlinks False')

# stage 2
os.system('huggingface-cli download --resume-download hpcai-tech/OpenSora-STDiT-v2-stage2 --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v2-stage2 --local-dir-use-symlinks False')

# stage 3
# os.system('huggingface-cli download --resume-download hpcai-tech/OpenSora-STDiT-v2-stage3 --local-dir /home/dist/hpcai/duanjunwen/Open-Sora/pretrained_models/stdit/OpenSora-STDiT-v2-stage3 --local-dir-use-symlinks False')
