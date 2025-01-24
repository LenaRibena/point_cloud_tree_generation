from argparse import Namespace
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import Module

from tree.modules.common import gaussian_entropy, reparameterize_gaussian, standard_normal_logprob, truncated_normal_
from tree.modules.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from tree.modules.encoders.pointnet import PointNetEncoder
from tree.modules.flow import build_latent_flow


class FlowVAE(Module):  # type: ignore
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps, beta_1=args.beta_1, beta_T=args.beta_T, mode=args.sched_mode
            ),
        )

    def get_loss(self, x: torch.Tensor, kl_weight: float, writer: Any = None, it: Any = None) -> torch.Tensor:
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)

        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)  # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)  # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z)

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight * (loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar("train/loss_entropy", loss_entropy, it)
            writer.add_scalar("train/loss_prior", loss_prior, it)
            writer.add_scalar("train/loss_recons", loss_recons, it)
            writer.add_scalar("train/z_mean", z_mu.mean(), it)
            writer.add_scalar("train/z_mag", z_mu.abs().max(), it)
            writer.add_scalar("train/z_var", (0.5 * z_sigma).exp().mean(), it)

        return loss

    def sample(
        self, w: torch.Tensor, num_points: int, flexibility: float, truncate_std: Optional[float] = None
    ) -> np.ndarray:
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples

    @classmethod
    def load(cls, path: str) -> "FlowVAE":
        state = torch.load(path, weights_only=True)

        # Extract args from state (or manually)
        args = Namespace(
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
