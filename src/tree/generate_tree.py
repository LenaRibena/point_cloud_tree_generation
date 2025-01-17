import torch

from model import VAE
from util import Normalizer
from visualize_tree import TreeViewer

from cfg import CFG

def main():
    dict = torch.load("vae_1.pth", weights_only=True)
    input_dim = CFG.POINTS * 3
    vae = VAE(input_dim, CFG.LATENT_DIM).to("cuda")
    vae.load_state_dict(dict)
    vae.eval()

    data = torch.load("data/processed/data.pt", weights_only=True)
    data = data.view(data.size(0), -1)
    normalizer = Normalizer(data)

    # z = torch.randn(1, CFG.LATENT_DIM, device="cuda")
    # sample random datapoint
    sample = data[0].unsqueeze(0)
    mu, logvar = vae.encoder(sample)
    z = vae.reparameterize(mu, logvar)
    # print(sample.shape)
    # quit()
    generated_tree = vae.decoder(z).detach()[0]
    generated_tree = normalizer.renormalize(generated_tree)
    generated_tree = generated_tree.cpu().numpy()

    tv = TreeViewer()
    tv.view(generated_tree)


if __name__ == "__main__":
    main()