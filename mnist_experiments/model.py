import torch
import torch.nn as nn
import pytorch_lightning as pl


class DenseMNISTNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32), # 28*28
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )
    def forward(self, image):
        pred = self.net(image)
        return pred

    def training_step(self, batch, batch_idx):
        image, digit = batch
        pred = self(image)
        digit = digit.type_as(pred)
        loss = torch.nn.functional.cross_entropy(pred, digit)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
