import os

import pytest
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
