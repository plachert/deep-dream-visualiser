import argparse
from data import MNISTDataset, MNISTDataModule
from model import DenseMNISTNet
import pytorch_lightning as pl


def main(corrupt: bool):
    print(corrupt)
    train_ds = MNISTDataset(train=True, corrupt=corrupt)
    test_ds = MNISTDataset(train=False, corrupt=corrupt)
    
    datamodule = MNISTDataModule(train_ds, test_ds, batch_size=256)
    model = DenseMNISTNet()
    
    if corrupt:
        experiment_path = "models/dense_mnist_corrupted"
    else:
        experiment_path = "models/dense_mnist"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        dirpath=f"{experiment_path}",
        filename="best",
        mode="min",
    )
    
    trainer = pl.Trainer(
        max_epochs=50,
        logger=pl.loggers.TensorBoardLogger(f"{experiment_path}/log"),
        callbacks=[checkpoint_callback]
        )
    
    trainer.fit(model, datamodule)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--corrupt', action='store_true', help='Whether to corrupt the dataset')
    args = parser.parse_args()
    main(args.corrupt)
