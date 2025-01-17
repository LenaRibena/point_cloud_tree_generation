import os
import warnings
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split


class PCTreeDataset(Dataset):
    """Dataset class for TreeML-Data; a multidisciplinary and multilayer urban tree dataset."""

    def __init__(self, 
                 split: str, 
                 raw_data_path: str | Path = 'urban_tree_dataset', 
                 device = 'cpu', 
                 transform = None) -> None:
        
        # Get all data files from the specified data path folder
        self.data_path = Path(raw_data_path)
        self.data_files = []
        self.fixed_size = 100000
        
        for folder in os.listdir(self.data_path):
            data_dir = Path(self.data_path, folder)
            data_files = list(data_dir.glob('*.txt'))
            self.data_files.extend(data_files)
        
        assert len(self.data_files) > 0, f"No data files found or path doesn't exist; {self.data_path}."
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Use either 'train', 'val', or 'test'."
        
        self.split = split
        self.transform = transform
        
        # Configure device
        match device:
            case 'cpu':
                self.device = 'cpu'
            case 'cuda':
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    warnings.warn('CUDA is not available. Using CPU instead.')
                    self.device = 'cpu'
            case _:
                raise ValueError('Invalid device. Use either "cpu" or "cuda".')

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a given sample from the dataset."""
        file_path = self.data_files[index]
        df = pd.read_csv(file_path, sep=' ', header=None)
        xyz_data = df.iloc[:, :3]
        
        data = torch.tensor(xyz_data.values, dtype=torch.float32, device=self.device)
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
    def resample(self, points):
        weights = torch.ones(len(points))
        if len(points) >= self.fixed_size:
            idx = torch.multinomial(weights, len(points), replace=False)
        else:
            raise ValueError("Not enough points in the point cloud.")
            # idx = np.random.choice(len(points), self.fixed_size, replace=True)
        return points[idx]
    
    def get_train_val_test_datasets(self, dataset, train_ratio, val_ratio):
        assert (train_ratio + val_ratio) <= 1
        train_size = int(len(dataset) * train_ratio)
        val_size = int(len(dataset) * val_ratio)
        test_size = len(dataset) - train_size - val_size
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        return train_set, val_set, test_set


    def get_train_val_test_loaders(self, dataset, train_ratio, val_ratio, train_batch_size, val_test_batch_size, num_workers):
        train_set, val_set, test_set = self.get_train_val_test_datasets(dataset, train_ratio, val_ratio)

        train_loader = DataLoader(train_set, train_batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, val_test_batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_set, val_test_batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
    
    def get_data_iterator(iterable):
        """Allows training with DataLoaders in a single infinite loop:
            for i, data in enumerate(inf_generator(train_loader)):
        """
        iterator = iterable.__iter__()
        while True:
            try:
                yield iterator.__next__()
            except StopIteration:
                iterator = iterable.__iter__()

if __name__ == "__main__":
    tree_dataset = PCTreeDataset(split='train', raw_data_path='data/raw/urban_tree_dataset')
