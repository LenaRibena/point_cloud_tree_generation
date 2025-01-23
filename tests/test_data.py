import os

import pytest
import torch
from torch.utils.data import Dataset

from tree.data import PCTreeDataset

_PATH_DATA = "data/processed"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="No processed data found.")
def test_my_dataset():
    """Test the MyDataset class."""
    N = 3
    dataset = PCTreeDataset(_PATH_DATA)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == N, f"Incorrect dataset size. Expected {N}, got {len(dataset)}"

    train, val, test = dataset.get_train_val_test_loaders(1 / 3, 1 / 3, 1, 0)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1

    count = 0
    for dataset in [train, val, test]:
        for batch in dataset:
            count += batch.shape[0]
            assert tuple(batch.shape[1:]) == (4096, 3), "Shape of data is incorrect."
    assert count == N


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="No processed data found.")
def test_device():
    """Test the device attribute of the PCTreeDataset class."""
    dataset = PCTreeDataset(_PATH_DATA, device="cpu")
    assert dataset.device == "cpu"

    with pytest.raises(ValueError):
        PCTreeDataset(_PATH_DATA, device="invalid_device")


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="No processed data found.")
def test_len():
    """Test the __len__ method of the PCTreeDataset class."""
    dataset = PCTreeDataset(_PATH_DATA)
    assert len(dataset) == len(dataset.data_files), "Dataset length does not match the number of data files."


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="No processed data found.")
def test_getitem():
    """Test the __getitem__ method of the PCTreeDataset class."""
    dataset = PCTreeDataset(_PATH_DATA)
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor), "Sample is not a torch.Tensor."
    assert sample.shape == (4096, 3), "Shape of data is incorrect."


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="No processed data found.")
def test_get_train_val_test_datasets():
    """Test the get_train_val_test_datasets method of the PCTreeDataset class."""
    dataset = PCTreeDataset(_PATH_DATA)
    train_ratio, val_ratio = 1 / 3, 1 / 3
    train_set, val_set, test_set = dataset.get_train_val_test_datasets(train_ratio, val_ratio)
    assert len(train_set) == int(len(dataset) * train_ratio), "Incorrect train set size."
    assert len(val_set) == int(len(dataset) * val_ratio), "Incorrect validation set size."
    assert len(test_set) == len(dataset) - len(train_set) - len(val_set), "Incorrect test set size."


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="No processed data found.")
def test_get_train_val_test_loaders():
    """Test the get_train_val_test_loaders method of the PCTreeDataset class."""
    dataset = PCTreeDataset(_PATH_DATA)
    train_ratio, val_ratio, batch_size, num_workers = 1 / 3, 1 / 3, 1, 0
    train_loader, val_loader, test_loader = dataset.get_train_val_test_loaders(
        train_ratio, val_ratio, batch_size, num_workers
    )
    assert len(train_loader) == int(len(dataset) * train_ratio) // batch_size, "Incorrect train loader size."
    assert len(val_loader) == int(len(dataset) * val_ratio) // batch_size, "Incorrect validation loader size."
    assert (
        len(test_loader)
        == (len(dataset) - int(len(dataset) * train_ratio) - int(len(dataset) * val_ratio)) // batch_size
    ), "Incorrect test loader size."

    for loader in [train_loader, val_loader, test_loader]:
        for batch in loader:
            assert len(batch.shape) == 3, "Incorrect size."
