import torch

class CFG:
    LATENT_DIM = 512
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    POINTS = 4096 #16_129
    SAVE_PATH = "models/gnn/vae.pth"