import torch
import torch.nn as nn
import torch.optim as optim

from cfg import CFG

# Define the Chamfer Loss
class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.to(CFG.DEVICE)

    def batch_pairwise_dist(self, x, y): 
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
    
    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)        # compute pairwise distance between preds and gts
        mins, _ = torch.min(P, 1)                       # find the nearest gt point for each pred point
        loss_1 = torch.mean(mins)                       # take mean across got <-> pred loss
        mins, _ = torch.min(P, 2)                       # find the nearest pred point for each gt point
        loss_2 = torch.mean(mins)                       # take mean across pred <-> gt loss
        return (loss_1 + loss_2)

if __name__ == "__main__":
    chamfer_loss = ChamferLoss()

    batch_size = 32
    num_points = 4096
    gts = torch.rand(batch_size, num_points, 3).cuda()  # (batch_size, num_points, 3)
    preds = torch.rand(batch_size, num_points, 3).cuda()  # (batch_size, num_points, 3)

    loss = chamfer_loss(preds, gts)
    print(f"Chamfer Loss: {loss.item():.4f}")
