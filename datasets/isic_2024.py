import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
import io
from datasets.helpers import get_metadata_row
import pandas
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, root_folder=ISIC_2024_ROOT, transform=None, mode:str='train', writer: SummaryWriter=None):
        self.writer = writer
        try:
            if mode == 'train':
                hdf5_path = Path(root_folder)/"train-image.hdf5"
                annot_path = Path(root_folder)/"train-metadata.csv"
            elif mode == 'test':
                hdf5_path = Path(root_folder)/"test-image.hdf5"
                annot_path = Path(root_folder)/"test-metadata.csv"
            else:
                raise ValueError("")
        except ValueError as e:
            print(f"ValueError {e}: Argument mode needs to be either 'train' or 'test'")

        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.hdf5_keys = [key for key in self.hdf5_file.keys()]
        self.annot_df = pandas.read_csv(annot_path)
        self.transform = transform

    def __getitem__(self, index):
        img = extract_image(self.hdf5_file[self.hdf5_keys[index]][()])
        if self.transform:
            aug = self.transform(image=img)
            img = aug.to(torch.float32)
        metadata = get_metadata_row(self.annot_df, self.hdf5_keys[index])
        grading = torch.Tensor(metadata['target'])
        return img, grading
        
    def __len__(self):
        return len(self.hdf5_keys)