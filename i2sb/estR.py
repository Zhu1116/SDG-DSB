import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.resizer import Resizer

class Para(nn.Module):
    def __init__(self, c, C):
        super().__init__()
        srf = torch.ones([c, C], dtype=torch.float32) * (1.0 / C)
        self.srf = nn.Parameter(srf)

    def forward(self):
        return F.softmax(self.srf)


class estR():
    def __init__(self, lr_hsi, hr_msi, device):
        self.device = device

        lr_hsi = torch.from_numpy(lr_hsi).permute(2, 0, 1).unsqueeze(0)
        hr_msi = torch.from_numpy(hr_msi).permute(2, 0, 1).unsqueeze(0)

        self.hr_msi = hr_msi.to(self.device)
        self.lr_hsi = lr_hsi.to(self.device)

        _, c, W, H = self.hr_msi.shape
        _, C, w, h = self.lr_hsi.shape

        self.W = W
        self.H = H
        self.c = c
        self.w = w
        self.h = h
        self.C = C

    def start_est(self, scale_factor=1/4, ite=5000):
        para = Para(self.c, self.C).to(self.device)
        para.train()
        optimizer = optim.Adam(para.parameters(), lr=5e-2)

        down_fn = Resizer((1, self.C, self.H, self.W), scale_factor).to(self.device)
        lr_msi1 = down_fn(self.hr_msi)

        for i in range(ite):
            srf = para()
            lr_hsi = self.lr_hsi.reshape(1, self.C, -1)
            lr_msi2 = torch.matmul(srf, lr_hsi.squeeze(0)).reshape(srf.shape[0], self.w, self.h).unsqueeze(0)

            res = lr_msi1 - lr_msi2
            loss = torch.mean(torch.abs(res))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print(f"Iter: [{i + 1:4d}] -- Loss: {loss.item():.8f}")
        return srf.detach().cpu().numpy()
