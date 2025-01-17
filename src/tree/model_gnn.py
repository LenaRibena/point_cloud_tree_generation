import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.data import Data

from cfg import CFG
from util import Normalizer


# Define the GNN-based Encoder
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, latent_dim)
        self.relu = nn.ReLU()

        # Apply Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, GCNConv):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        h = self.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        mean, logvar = h.chunk(2, dim=-1)  # Split into mean and logvar
        return mean, logvar


# Define the Decoder
class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(GraphDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

        # Apply Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z):
        h = self.relu(self.fc1(z))
        recon = self.fc2(h)
        return recon.view(-1, CFG.POINTS, 3)


# Define the VAE
class GraphVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GraphVAE, self).__init__()
        self.encoder = GraphEncoder(input_dim, latent_dim * 2)  # Latent dim * 2 for mean and logvar
        self.decoder = GraphDecoder(latent_dim, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, edge_index):
        mean, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar


# Loss function
from loss import ChamferLoss
loss = ChamferLoss()

def loss_function(recon_x, x, mean, logvar):
    recon_loss = loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_div


# Training loop
def train_vae(data_loader, input_dim, latent_dim, lr=0.001):
    vae = GraphVAE(input_dim, latent_dim).to(CFG.DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()

    for epoch in range(CFG.EPOCHS):
        for batch in data_loader:
            batch = batch.to(CFG.DEVICE)
            x, edge_index = batch.x, batch.edge_index

            optimizer.zero_grad()
            recon, mean, logvar = vae(x, edge_index)
            loss = loss_function(recon, x.view(-1, CFG.POINTS, 3), mean, logvar)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{CFG.EPOCHS}, Loss: {loss.item():.4f}")
    return vae


# Data preparation
if __name__ == "__main__":
    from torch_geometric.data import DataLoader

    # Load preprocessed point cloud data
    data = torch.load("data/processed/data.pt", weights_only=True)
    normalizer = Normalizer(data)
    data = normalizer.normalize(data)

    # Create graph data
    edge_index = knn_graph(data.view(-1, 3), k=16, batch=None)  # k-NN graph
    graph_data = Data(x=data.view(-1, 3), edge_index=edge_index)

    # Create data loader
    data_loader = DataLoader([graph_data], batch_size=1, shuffle=True)

    # Train the VAE
    input_dim = 3  # Each point has (x, y, z) dimensions
    vae = train_vae(data_loader, input_dim, CFG.LATENT_DIM)

    # Save the trained model
    torch.save(vae.state_dict(), f'{CFG.SAVE_PATH}_{CFG.EPOCHS}.pth')
