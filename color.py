import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
import optuna

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    patch_size = trial.suggest_categorical("patch_size", [(32, 32), (64, 64), (128, 128)])
    num_rot = trial.suggest_int("num_rot", 1, 4)

    transform = transforms.Compose([transforms.ToTensor()])
    
    data_module = ImageColorizationDataModule(
        path="Dane", batch_size=batch_size, patch_size=patch_size, num_rot=num_rot, transform=transform
    )
    
    model = ImageColorizationModel(lr=lr)

    mlflow_logger = MLFlowLogger(experiment_name="image-colorization")

    trainer = L.Trainer(max_epochs=5, logger=mlflow_logger, enable_checkpointing=False)

    trainer.fit(model, data_module)
    
    val_loss = trainer.callback_metrics["val_loss_epoch"].item()
    
    return val_loss




# PyTorch Dataset class
class ImageColorizationDataset(Dataset):
    def __init__(self, path, patch_size, num_rot=1, transform=None):
        self.path = path
        self.file_list = os.listdir(path)
        random.shuffle(self.file_list)
        self.patch_size = patch_size
        self.num_rot = num_rot
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.path, file_name)
        image = Image.open(file_path).resize(self.patch_size)

        if image.mode != "RGB":
            image = image.convert("RGB")

        bw_image = image.convert("L")

        if self.transform:
            bw_image = self.transform(bw_image)
            image = self.transform(image)

        return bw_image, image


class ImageColorizationDataModule(L.LightningDataModule):
    def __init__(self, path, batch_size, patch_size, num_rot, transform):
        super().__init__()
        self.path = path
        self.patch_size = patch_size
        self.num_rot = num_rot
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.dataset = ImageColorizationDataset(
            path=self.path,
            patch_size=self.patch_size,
            num_rot=self.num_rot,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


class ImageColorizationModel(L.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.create_model(is_batch_norm=False, is_drop=False)

    def forward(self, x):
        skips = []  # Store skip connections

        # Encoder pass
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)  # Store feature maps for skip connections

        # Decoder pass (reverse the encoder outputs for skip connections)
        for i, layer in enumerate(self.decoder):
            x = torch.cat([x, skips[-(i + 1)]], dim=1)  # Concatenate skip connection
            x = layer(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def create_model(self, is_batch_norm, is_drop):

        # Encoder
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU()
                ),
            ]
        )

        # Decoder
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        256 + 256, 128, kernel_size=3, stride=1, padding=1
                    ),  # Skip connection
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        128 + 128, 64, kernel_size=3, stride=1, padding=1
                    ),  # Skip connection
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        64 + 64, 3, kernel_size=3, stride=1, padding=1
                    ),  # Final RGB output
                    nn.Sigmoid(),  # Output pixel values in [0, 1]
                ),
            ]
        )

    def debug(self):
        print("Model summary:")
        print(self)


if __name__ == "__main__":

    data_dir = "Dane"
    batch_size = 10
    patch_size = (64, 64)
    num_rot = 1
    transform = transforms.Compose([transforms.ToTensor()])

    model = ImageColorizationModel()
    model.debug()

    mlflow_logger = MLFlowLogger(experiment_name="image-colorization")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    final_model = ImageColorizationModel(lr=best_params["lr"])
    data_module = ImageColorizationDataModule(
        path="Dane",
        batch_size=best_params["batch_size"],
        patch_size=(64, 64),
        num_rot=1,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    trainer = L.Trainer(max_epochs=10, logger=mlflow_logger)
    trainer.fit(final_model, data_module)
