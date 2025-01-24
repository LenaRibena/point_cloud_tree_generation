from argparse import Namespace

import torch
from torch.nn import Module

from tree.modules.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from tree.modules.encoders.pointnet import PointNetEncoder


class AutoEncoder(Module):  # type: ignore
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net=PointwiseNet(context_dim=args.latent_dim, residual=args.residual),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps, beta_1=args.beta_1, beta_T=args.beta_T, mode=args.sched_mode
            ),
        )

    # def encode(self, x):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        return code

    def decode(
        self, code: torch.Tensor, num_points: int, flexibility: float = 0.0, ret_traj: bool = False
    ) -> torch.Tensor:
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code)
        return loss
