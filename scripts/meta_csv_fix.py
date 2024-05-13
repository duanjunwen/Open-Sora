import pandas as pd
import os

meta = pd.read_csv("/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda2m/meta/meta_clips_caption_cleaned.csv")

# print(meta.head(10))
print(meta.columns)

def replace_func(x):
    x = x.replace('/home/duanjunwen/LCProj/Open-Sora/dataset/panda2m/clips', '/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda2m/clips')
    return x
    

if 'path' in meta.columns:
    # print(f"before\n {meta['path']}\n")
    # meta['path'] = meta['path'].replace('/home/duanjunwen/LCProj/Open-Sora/dataset/panda2m/clips', '/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda2m/clips')
    # meta['path'] = meta['path'].apply(lambda x: x.replace(x, os.path.dirname(x)))
    # for abs_path in meta['path'].iterrows():
    #     print(f"abs_path {abs_path}")
    meta['path'] = meta['path'].apply(replace_func)
    meta.to_csv("/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda2m/meta/meta_clips_caption_cleaned_fixed.csv", index=False)
    
    # for idx, abs_path in meta['path'].items():
    #     print(f"idx {idx}, series {abs_path}")
    #     abs_path = abs_path.replace('/home/duanjunwen/LCProj/Open-Sora/dataset/panda2m/clips', '/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda2m/clips')
    #     print(f"after idx {idx}, series {abs_path}")