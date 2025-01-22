import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from tree.utils.utils import equal_batch_size


class PCTreeDataset(Dataset):
    """Dataset class for TreeML-Data; a multidisciplinary and multilayer urban tree dataset.
    See tree.preprocess to process the data"""

    def __init__(
        self, processed_data_path: str | Path = "data/processed/urban_tree_dataset", device="cpu", transform=None
    ) -> None:
        # Get all data files from the specified data path folder
        self.data_path = Path(processed_data_path)
        self.data_files = list(self.data_path.rglob("*.npy"))

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
        xyz_data = np.load(file_path)
        data = torch.from_numpy(xyz_data)  # Note, might need to change the dtype

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
    tree_dataset = PCTreeDataset(processed_data_path="data/processed/urban_tree_dataset")
    train_loader, val_loader, test_loader = tree_dataset.get_train_val_test_loaders(
        train_ratio=0.8, val_ratio=0.1, batch_size=32, num_workers=4
    )
    # print length of each loader
    # print(len(train_loader), len(val_loader), len(test_loader))

    # print shape of sample data from train_loader
    for data in train_loader:
        print(data.shape)
        break
