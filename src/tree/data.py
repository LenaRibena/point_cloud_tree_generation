import warnings
from pathlib import Path

import typer
import torch
import pandas as pd
from torch.utils.data import Dataset


class PCTreeDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: str | Path, device='cpu') -> None:
        self.data_path = Path(raw_data_path)
        self.data_files = list(self.data_path.glob('*.txt'))
        assert len(self.data_files) > 0, f"No data files found or path doesn't exist; {self.data_path}."
        
        match device:
            case 'cpu':
                self.device = 'cpu'
            case 'cuda':
                if torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    warnings.warn('CUDA is not available. Using CPU instead.')
                    self.device = 'cpu'
            case _:
                raise ValueError('Invalid device. Use either "cpu" or "cuda".')

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_files)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        file_path = self.data_files[index]
        df = pd.read_csv(file_path, sep=' ', header=None)
        xyz_data = df.iloc[:, :3]
        
        data = torch.tensor(xyz_data.values, dtype=torch.float32, device=self.device)
        return data

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        warnings.warn("Preprocessing not implemented yet.")

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = PCTreeDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
