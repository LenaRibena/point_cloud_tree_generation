from typing import Any, Optional

import numpy as np
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from tree.modules.common import (
    gaussian_entropy,
    reparameterize_gaussian,
    standard_normal_logprob,
    truncated_normal_,
)
from tree.modules.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from tree.modules.encoders.pointnet import PointNetEncoder


class GaussianVAE(Module):  # type: ignore
    def __init__(self, args: Any) -> None:
        """
        Initialize the Gaussian VAE.

        Args:
            args: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.args = args
        self.encoder: PointNetEncoder = PointNetEncoder(args.latent_dim)
        self.diffusion: DiffusionPoint = DiffusionPoint(
            net=PointwiseNet(context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps, beta_1=args.beta_1, beta_T=args.beta_T, mode=args.sched_mode
            ),
        )

    def get_loss(
        self,
        x: torch.Tensor,
        writer: Optional[SummaryWriter] = None,
        it: Optional[int] = None,
        kl_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the loss for the Gaussian VAE.

        Args:
            x: Input point clouds, (B, N, d).
            writer: TensorBoard writer for logging (optional).
            it: Iteration number for logging (optional).
            kl_weight: Weight for the KL-divergence term.

        Returns:
            torch.Tensor: The total loss.
        """
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)  # (B, )
        loss_prior = (-log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar("train/loss_entropy", -entropy.mean().item(), it)
            writer.add_scalar("train/loss_prior", -log_pz.mean().item(), it)
            writer.add_scalar("train/loss_recons", loss_recons.item(), it)

        return loss

    def sample(
        self,
        z: torch.Tensor,
        num_points: int,
        flexibility: float,
        truncate_std: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample points from the Gaussian VAE.

        Args:
            z: Input latent, normal random samples with mean=0 std=1, (B, F).
            num_points: Number of points to sample.
            flexibility: Flexibility parameter for diffusion.
            truncate_std: Optional truncation standard deviation.

        Returns:
            torch.Tensor: Sampled points, (B, num_points, d).
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

    @classmethod
    def load(cls, path: str) -> "GaussianVAE":
        """
        Load a GaussianVAE model from a checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            GaussianVAE: Loaded model.
        """
        from types import SimpleNamespace

        state = torch.load(path, weights_only=True)

        # Extract args from state (or manually)
        args = SimpleNamespace(
            latent_dim=state["encoder.fc3_v.weight"].shape[0],
            residual=False,
            latent_flow_depth=2,
            latent_flow_hidden_dim=128,
            num_steps=10,
            beta_1=0.9,
            beta_T=0.999,
            sched_mode="linear",
        )

        model = cls(args)
        model.load_state_dict(state)
        return model

    def generate(self, num_points: int = 4096) -> np.ndarray:
        """
        Generate point cloud samples from the GaussianVAE.

        Args:
            num_points: Number of points to generate.

        Returns:
            np.ndarray: Generated point cloud samples, (num_points, d).
        """
        z = torch.randn(1, self.args.latent_dim)
        return self.diffusion.sample(num_points, context=z)[0].detach().cpu().numpy()
