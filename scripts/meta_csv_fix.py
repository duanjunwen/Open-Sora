import pandas as pd
import os

meta = pd.read_csv("/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train/meta/meta_clips_caption_cleaned_fixed_rm4.csv")

# print(meta.head(10))
print(meta.columns)

def replace_func(x):
    x = x.replace('/data01/hpcai/hpcai/duanjunwen/Open-Sora/dataset/panda_train/clips', 
                  '/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train/clips')
    return x
    

if 'path' in meta.columns:
    meta['path'] = meta['path'].apply(replace_func)
    meta.to_csv("/home/dist/hpcai/duanjunwen/Open-Sora/dataset/panda_train/meta/meta_clips_caption_cleaned_fixed_rm4_format.csv", index=False)
