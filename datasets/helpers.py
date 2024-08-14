import torch
import torch.nn.functional as F 
import os
import numpy as np
import json
import pandas as pd
from typing import Union
    

def get_metadata_row(df: pd.DataFrame, isic_id: str) -> Union[dict, None]:
    """Get metadata row by isic_id and return it as a dictionary where keys are column headers and values the row contents.

    Args:
        df (pd.DataFrame): Dataframe containing metadata.
        isic_id (str): Datapoint identifier.

    Returns:
        Union[dict, None]: Dictionary containing the row or None in case an empty row is found.
    """    
    row = df.loc[df["isic_id"] == isic_id]
    if not row.empty:
        return row.iloc[0].to_dict()
    else:
        return None

def tensor2img(img_tensor):
    img = img_tensor.permute(1,2,0)
    img = ((img - img.mean(axis=(0,1)))/img.std(axis=(0,1))).numpy()
    img = (img-img.min())/(img.max()-img.min())*255.
    return img.astype(np.uint8)

def get_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def save_json_filelist(pathlist, save_path):
    filelist = [os.path.basename(path) for path in pathlist] 
    with open(save_path, "w") as fp:
        fp.write(str(json.dumps(filelist))) 

class ToOneHotMask(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, mask):
        one_hot_mask = F.one_hot(mask.to(torch.int64).squeeze(0), num_classes=self.num_classes).permute(2,0,1)
        return one_hot_mask.float()