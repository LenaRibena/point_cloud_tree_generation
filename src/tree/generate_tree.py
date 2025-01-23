import torch

from tree.cfg import CFG
from tree.model import VAE
from tree.util import Normalizer
from tree.visualize_tree import TreeViewer


class TreeGenerator:
    def __init__(self, path="models/vae_1.pth"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dict = torch.load(path, weights_only=True, map_location=device)
        input_dim = CFG.POINTS * 3
        self.vae = VAE(input_dim, CFG.LATENT_DIM).to(device)
        self.vae.load_state_dict(dict)
        self.vae.eval()

        data = torch.load("data/processed/data.pt", map_location=device, weights_only=True)
        data = data.view(data.size(0), -1)
        self.normalizer = Normalizer(data)

        self.tv = TreeViewer()

    def __call__(self):
        z = torch.randn(1, CFG.LATENT_DIM, device="cuda")
        # # sample random datapoint
        # sample = data[0].unsqueeze(0)
        # mu, logvar = vae.encoder(sample)
        # z = vae.reparameterize(mu, logvar)
        # print(sample.shape)
        # quit()
        generated_tree = self.vae.decoder(z).detach()[0]
        generated_tree = self.normalizer.renormalize(generated_tree)
        generated_tree = generated_tree.cpu().numpy()

        return generated_tree

    def view(self, generated_tree):
        self.tv.view(generated_tree)


if __name__ == "__main__":
    tg = TreeGenerator()
    gt = tg()
    tg.view(gt)
