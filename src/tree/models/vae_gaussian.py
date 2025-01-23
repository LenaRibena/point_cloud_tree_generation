import numpy as np
import torch
from torch.nn import Module

from tree.models.common import gaussian_entropy, reparameterize_gaussian, standard_normal_logprob, truncated_normal_
from tree.models.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from tree.models.encoders.pointnet import PointNetEncoder


class GaussianVAE(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps, beta_1=args.beta_1, beta_T=args.beta_T, mode=args.sched_mode
            ),
        )

    def get_loss(self, x, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)  # (B, )
        loss_prior = (-log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar("train/loss_entropy", -entropy.mean(), it)
            writer.add_scalar("train/loss_prior", -log_pz.mean(), it)
            writer.add_scalar("train/loss_recons", loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

    @classmethod
    def load(cls, path: str) -> Module:
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
        z = torch.randn(1, self.args.latent_dim)
        return self.diffusion.sample(num_points, context=z)[0].detach().cpu().numpy()
