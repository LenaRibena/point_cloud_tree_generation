import torch
import torch.nn as nn
import torch.optim as optim

from cfg import CFG
from util import Normalizer

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2_mean = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()
    
        # Apply Kaiming initialization to all layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        assert not torch.isnan(h).any(), "nan detected"
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
    
        # Apply Kaiming initialization to all layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z):
        h = self.relu(self.fc1(z))
        recon = self.fc2(h)
        return recon.view(-1, CFG.POINTS, 3)  # Reshape to (batch_size, num_points, 3)

# Define the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar
    


from loss import ChamferLoss
loss = ChamferLoss()
def loss_function(recon_x, x, mean, logvar):
    recon_loss = loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Training loop
def train_vae(data_loader, input_dim, latent_dim, lr=0.001):
    vae = VAE(input_dim, latent_dim).to(CFG.DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()

    for epoch in range(CFG.EPOCHS):
        for batch in data_loader:
            # Unpack batch: TensorDataset wraps data as (data,)
            batch = batch[0]
            
            optimizer.zero_grad()
            recon, mean, logvar = vae(batch)
            loss = loss_function(recon, batch.view(-1, CFG.POINTS, 3), mean, logvar)
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{CFG.EPOCHS}, Loss: {loss.item():.4f}")
        torch.save(vae.state_dict(), CFG.SAVE_PATH)
    return vae



    


if __name__ == "__main__":
    # Example usage
    # Assuming your dataset is preprocessed to have point clouds of fixed size N
    from torch.utils.data import DataLoader, TensorDataset

    # Replace with your actual preprocessed data (N points per tree)
    data = torch.load("data/processed/data.pt", weights_only=True)
    data = data.view(data.size(0), -1)
    normalizer = Normalizer(data)
    data = normalizer.normalize(data)
    # data = torch.rand(1000, 3 * 1024, device=DEVICE)  # 1000 samples of 1024 3D points each

    # print(data.shape)
    data_loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=True)

    # Train the VAE
    input_dim = data.size(1)  # 3 * 1024
    # 516128x3 and 16129x128
    vae = train_vae(data_loader, input_dim, CFG.LATENT_DIM)

    # Save the trained model
    torch.save(vae.state_dict(), f'{CFG.SAVE_PATH}_{CFG.EPOCHS}.pth')
