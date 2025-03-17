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
    def __init__(self):
        super().__init__()
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
        return optim.Adam(self.parameters(), lr=0.001)

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

    data_module = ImageColorizationDataModule(
        data_dir, batch_size, patch_size, num_rot, transform
    )
    model = ImageColorizationModel()
    model.debug()

    mlflow_logger = MLFlowLogger(experiment_name="image-colorization")

    trainer = L.Trainer(max_epochs=10, logger=mlflow_logger)
    trainer.fit(model, data_module)

    metrics = trainer.callback_metrics
    train_losses = [metrics['train_loss_epoch'].item()]
    val_losses = [metrics['val_loss_epoch'].item()]

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

# OLD TRAINING AND PLOTTING BELOW

# X_train, Y_train = [], []
# for bw, color in dataloader:
#     X_train.append(bw.numpy().transpose(0, 2, 3, 1))
#     Y_train.append(color.numpy().transpose(0, 2, 3, 1))

# X_train = np.concatenate(X_train, axis=0)
# Y_train = np.concatenate(Y_train, axis=0)

# model = create_model(0, 1, 1)
# history = model.fit(X_train, Y_train, validation_split=0.2, epochs=Epochs)
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("Model loss")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Validation"])
# plt.show()

# # Additional test on a single image
# obraz = load_img("test.jpg", target_size=(256, 256))
# obraz_1 = obraz
# bw_image = obraz.convert("L")
# obraz = img_to_array(bw_image)
# obraz = obraz / 255.0  # Normalization
# obraz = np.expand_dims(obraz, axis=0)
# wynik = model.predict(obraz)
# wynikowy_obraz = wynik[0]
# wynikowy_obraz = (wynikowy_obraz * 255).astype(np.uint8)

# fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# axes[2].imshow(wynikowy_obraz)
# axes[2].set_title("Wynikowy Obraz")
# axes[2].axis("off")

# axes[0].imshow(obraz_1)
# axes[0].set_title("Obraz 1")
# axes[0].axis("off")

# axes[1].imshow(bw_image, cmap="gray")
# axes[1].set_title("Czarno-biały")
# axes[1].axis("off")

# plt.show()
