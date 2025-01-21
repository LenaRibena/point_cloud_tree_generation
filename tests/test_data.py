from torch.utils.data import Dataset

from tree.data import PCTreeDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = PCTreeDataset("data/raw/urban_tree_dataset")
    assert isinstance(dataset, Dataset)
