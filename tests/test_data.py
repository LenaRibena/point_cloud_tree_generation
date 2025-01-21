from torch.utils.data import Dataset

from tree.data import PCTreeDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = PCTreeDataset("data/raw")
    assert isinstance(dataset, Dataset)
