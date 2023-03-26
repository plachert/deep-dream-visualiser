from data import MNISTDataset, MNISTDataModule
from model import DenseMNISTNet
import pytorch_lightning as pl


def main():
    train_ds = MNISTDataset(train=True)
    test_ds = MNISTDataset(train=False)
    
    datamodule = MNISTDataModule(train_ds, test_ds)
    model = DenseMNISTNet()
    
    trainer = pl.Trainer(max_epochs=5)
    
    trainer.fit(model, datamodule)
    
    
if __name__ == "__main__":
    main()
