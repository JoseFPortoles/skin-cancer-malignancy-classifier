import PIL.ImageShow
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import io
from datasets.helpers import get_metadata_row
from datasets.splits import split_isic2024
from transforms.transforms import transform_isic_2024
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from typing import Tuple


ISIC_2024_ROOT = "~/datos/isic_2024_data"

def extract_image(hdf5_data: h5py.Dataset) -> np.array:
    """Extract a RGB image from a hdf5 scalar dataset and returns it in a numpy array format (dtype=np.uint8)

    Args:
        hdf5_data (h5py.Dataset): Scalar dataset from an hdf5 file
    Returns:
        np.array: Array of shape (H,W,3), dtype=np.uint8 and values 0 to 255
    """    
    image = Image.open(io.BytesIO(hdf5_data))
    image = image.convert('RGB')
    image_array = np.array(image)
    return image_array

class ISIC2024Split:
    def __init__(self, seed: int, root_folder=ISIC_2024_ROOT, split_ratio: Tuple[int,int,int]=(0.8, 0.1, 0.1), writer: SummaryWriter=None) -> None:
        self.seed = seed
        self.split_ratio = split_ratio
        self.writer = writer
        self.annot_path = Path(root_folder)/"train-metadata.csv"
        self.hdf5_path = Path(root_folder)/"train-image.hdf5"
        self.annot_df = {}
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.hdf5_keys = [key for key in self.hdf5_file.keys()]
    
    def split(self):
        self.annot_df['train'], self.annot_df['val'], self.annot_df['test'] = split_isic2024(self.annot_path, self.seed, self.split_ratio)
    
    def save(self, out_folder: str):
        for key, df in self.annot_df.items():
            filepath = Path(out_folder)/f'{key}_split.csv'
            df.to_csv(filepath, index=True)

    def load(self, in_folder: str):
        for key in ['train', 'val', 'test']:
            filepath = Path(in_folder)/f'{key}_split.csv'
            self.annot_df[key] = pd.read_csv(filepath, index_col=0)

class ISIC2024Dataset(Dataset):
    def __init__(self, split: ISIC2024Split, transform=None, mode:str = 'train', writer: SummaryWriter=None):
        self.hdf5_file = split.hdf5_file
        self.hdf5_keys = split.hdf5_keys
        self.subset_df = split.annot_df[mode]
        self.transform = transform

    def __getitem__(self, subset_index):
        set_index = self.subset_df.index[subset_index]
        img = extract_image(self.hdf5_file[self.hdf5_keys[set_index]][()])
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image'].to(torch.float32)
        metadata = get_metadata_row(self.subset_df, self.hdf5_keys[set_index])
        grading = torch.tensor(metadata['target'])
        grading = F.one_hot(grading, num_classes=2)
        return img, grading
        
    def __len__(self):
        return len(self.subset_df)

if __name__ == '__main__':
    dataset = ISIC2024Dataset(42,transform=transform_isic_2024(224), mode='train')
    img, grading = dataset[30]