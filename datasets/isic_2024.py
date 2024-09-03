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
import pandas
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

class ISIC2024Dataset(Dataset):
    def __init__(self, seed: int, root_folder=ISIC_2024_ROOT, transform=None, split_ratio: Tuple[int,int,int]=(0.8, 0.1, 0.1), mode:str = 'train', writer: SummaryWriter=None):
        self.writer = writer
        annot_path = Path(root_folder)/"train-metadata.csv"
        hdf5_path = Path(root_folder)/"train-image.hdf5"
        self.train_df, self.val_df, self.test_df = split_isic2024(annot_path, seed, split_ratio)
        try:
            self.annot_df = {'train': self.train_df, 'val': self.val_df, 'test': self.test_df}[mode]
        except ValueError as e:
            print(f"{e}. Argument mode needs to be either 'train', 'val', or 'test'.")
            
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.hdf5_keys = [key for key in self.hdf5_file.keys()]
        self.transform = transform

    def __getitem__(self, subset_index):
        set_index = self.annot_df.index[subset_index]
        img = extract_image(self.hdf5_file[self.hdf5_keys[set_index]][()])
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image'].to(torch.float32)
        metadata = get_metadata_row(self.annot_df, self.hdf5_keys[set_index])
        grading = torch.tensor(metadata['target'])
        grading = F.one_hot(grading, num_classes=2)
        return img, grading
        
    def __len__(self):
        return len(self.annot_df)

if __name__ == '__main__':
    dataset = ISIC2024Dataset(42,transform=transform_isic_2024(224), mode='train')
    img, grading = dataset[30]