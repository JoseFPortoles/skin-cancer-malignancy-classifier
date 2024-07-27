import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.augmentations.geometric.resize import SmallestMaxSize
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
import io
import matplotlib.pyplot as plt

ISIC_2024_ROOT = "isic_2024_data"

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
    def __init__(self, root_folder=ISIC_2024_ROOT, crop_size=224, transform=None):
        hdf5_path = Path(root_folder)/"train-image.hdf5"
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.hdf5_keys = [key for key in self.hdf5_file.keys()]
        self.transform = transform

    def __getitem__(self, index):
        img = extract_image(self.hdf5_file[self.hdf5_keys[index]][()])
        if self.transform:
            aug = self.transform(image=img)
            img = aug.to(torch.float32)
        return img
        
    def __len__(self):
        return len(self.hdf5_keys)

if __name__=='__main__':
    dataset = ISIC2024Dataset()
    print(dataset[0])

