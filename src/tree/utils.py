import torch

def equal_batch_size(batch: torch.Tensor) -> torch.Tensor:
    """Function to ensure equal batch sizes for each forward pass
    by randomly downsampling all point clouds greater than the smallest.

    Args:
        batch (torch.Tensor): Batch of point clouds

    Returns:
        torch.Tensor: The randomly downsampled batch
    """

    # Find the smallest number of points in the batch
    min_points = min(item.shape[0] for item in batch)

    # Downsample each point cloud randomly to the smallest number of points
    batch = [item[torch.randperm(item.shape[0])[:min_points]] for item in batch]

    # Stack the batch
    batch = torch.stack(batch)

    return batch

class EarlyStopper:
    def __init__(self, patience: int = 5, delta: float = 0):
        """Early stopping class to stop training when the validation loss
        does not improve after a certain number of epochs.

        Borrowed from: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/

        Args:
            patience (int, optional): The number of consecutive epochs to
            allow worsened validation loss. Defaults to 5.

            delta (float, optional): The tolerance rate. Defaults to 0.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model: torch.nn.Module):
        model.load_state_dict(self.best_model_state)
