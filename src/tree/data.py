import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from tree.utils import equal_batch_size


class PCTreeDataset(Dataset):
    """Dataset class for TreeML-Data; a multidisciplinary and multilayer urban tree dataset."""

    def __init__(self, raw_data_path: str | Path = "data/raw/urban_tree_dataset", device="cpu", transform=None) -> None:
        # Get all data files from the specified data path folder
        self.data_path = Path(raw_data_path)
        self.data_files = []

        for folder in os.listdir(self.data_path):
            data_dir = Path(self.data_path, folder)
            data_files = list(data_dir.glob("*.txt"))
            self.data_files.extend(data_files)

        # assert len(self.data_files) > 0, f"No data files found or path doesn't exist; {self.data_path}."

        self.transform = transform

        # Configure device
        match device:
            case "cpu":
                self.device = "cpu"
            case "cuda":
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    warnings.warn("CUDA is not available. Using CPU instead.")
                    self.device = "cpu"
            case _:
                raise ValueError('Invalid device. Use either "cpu" or "cuda".')

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a given sample from the dataset."""
        file_path = self.data_files[index]
        df: pd.DataFrame = pd.read_csv(file_path, sep=" ", header=None)
        xyz_data = df.iloc[:, :3]

        data = torch.tensor(xyz_data.values, dtype=torch.float32, device=self.device)
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_train_val_test_datasets(self, train_ratio: float, val_ratio: float):
        assert (train_ratio + val_ratio) <= 1
        train_size = int(len(self) * train_ratio)
        val_size = int(len(self) * val_ratio)
        test_size = len(self) - train_size - val_size

        train_set, val_set, test_set = random_split(self, [train_size, val_size, test_size])
        return train_set, val_set, test_set

    def get_train_val_test_loaders(self, train_ratio: float, val_ratio: float, batch_size: int, num_workers: int):
        train_set, val_set, test_set = self.get_train_val_test_datasets(train_ratio, val_ratio)

        train_loader = DataLoader(
            train_set, batch_size, shuffle=True, num_workers=num_workers, collate_fn=equal_batch_size
        )
        val_loader = DataLoader(
            val_set, batch_size, shuffle=False, num_workers=num_workers, collate_fn=equal_batch_size
        )
        test_loader = DataLoader(
            test_set, batch_size, shuffle=False, num_workers=num_workers, collate_fn=equal_batch_size
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tree_dataset = PCTreeDataset(raw_data_path="data/raw/urban_tree_dataset")
    train_loader, val_loader, test_loader = tree_dataset.get_train_val_test_loaders(
        train_ratio=0.8, val_ratio=0.1, batch_size=32, num_workers=4
    )
