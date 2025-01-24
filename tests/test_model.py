from argparse import Namespace

import torch

from tree.models.common import gaussian_entropy, reparameterize_gaussian, standard_normal_logprob
from tree.models.vae_flow import FlowVAE
from tree.models.vae_gaussian import GaussianVAE


def test_gaussian() -> None:
    # Extract args from state (or manually)
    args = Namespace(
        latent_dim=128,
        residual=False,
        num_steps=10,
        beta_1=0.9,
        beta_T=0.999,
        sched_mode="linear",
    )

    model = GaussianVAE(args)
    encoder = model.encoder
    diffusion = model.diffusion
    point_net = diffusion.net
    var_sched = diffusion.var_sched

    # Create random data
    batch, num_point = 10, 4096
    data = torch.randn(batch, num_point, 3)

    # Test model output from encoder
    encoder_output = encoder(data)
    assert isinstance(encoder_output, tuple)
    assert len(encoder_output) == 2
    assert encoder_output[0].shape == (batch, args.latent_dim)
    assert encoder_output[1].shape == (batch, args.latent_dim)

    # Test model output from diffusion
    diffusion_output = diffusion.get_loss(data, encoder_output[0])
    assert isinstance(diffusion_output, torch.Tensor)

    # Test variance scheduler
    ts = var_sched.uniform_sample_t(batch)
    assert isinstance(ts, list)
    assert len(ts) == batch
    assert all([1 <= t <= args.num_steps for t in ts])

    betas = var_sched.betas[ts]

    # Test model output from pointnet
    pointnet_output = point_net(data, beta=betas, context=encoder_output[0])
    assert isinstance(pointnet_output, torch.Tensor)
    assert pointnet_output.shape == (batch, num_point, 3)

    output = model.get_loss(data)

    assert isinstance(output, torch.Tensor)


def test_flow() -> None:
    # Extract args from state (or manually)
    args = Namespace(
        latent_dim=128,
        residual=False,
        latent_flow_depth=2,
        latent_flow_hidden_dim=128,
        num_steps=10,
        beta_1=0.9,
        beta_T=0.999,
        sched_mode="linear",
    )

    model = FlowVAE(args)
    flow = model.flow

    # Create random data
    batch, num_point = 10, 4096
    data = torch.randn(batch, num_point, 3)

    # Test model output from flow
    flow_output = flow(torch.randn(batch, args.latent_dim), torch.zeros([batch, 1]))
    assert isinstance(flow_output, tuple)
    assert flow_output[0].shape == (batch, args.latent_dim)
    assert flow_output[1].shape == (batch, 1)

    output = model.get_loss(data, kl_weight=1.0)

    assert isinstance(output, torch.Tensor)


def test_common() -> None:
    # Create random data
    batch, latent_dim = 10, 128

    # Test reparameterize_gaussian
    mean = torch.randn(batch, latent_dim)
    logvar = torch.randn(batch, latent_dim)
    output = reparameterize_gaussian(mean, logvar)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch, latent_dim)

    # Test gaussian_entropy
    entropy = gaussian_entropy(logvar)

    assert isinstance(entropy, torch.Tensor)
    assert entropy.shape == (batch,)

    # Test standard_normal_logprob
    logprob = standard_normal_logprob(torch.randn(batch, latent_dim))

    assert isinstance(logprob, torch.Tensor)
    assert logprob.shape == (batch, latent_dim)
