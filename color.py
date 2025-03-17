import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import (
#     Conv2D,
#     MaxPooling2D,
#     Dense,
#     Flatten,
#     Activation,
#     Conv2DTranspose,
#     UpSampling2D,
# )
from keras.layers import Conv2D, Activation, BatchNormalization, Dropout
# from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def create_model(self, is_batch_norm, is_drop):
        filters = [64, 128, 256]
        layers = []

        # Encoder (Convolution Layers)
        for f in filters:
            layers.append(
                nn.Conv2d(
                    in_channels=1 if len(layers) == 0 else f // 2,
                    out_channels=f,
                    kernel_size=3,
                    padding=1,
                )
            )
            if is_batch_norm:
                layers.append(nn.BatchNorm2d(f))
            if is_drop:
                layers.append(nn.Dropout(0.3))
            layers.append(nn.ReLU())

        # Decoder (Transpose Convolution Layers)
        filters.reverse()
        for f in filters:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=f * 2 if len(layers) == len(filters) else f * 4,
                    out_channels=f,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())

        # Final Layer
        layers.append(
            nn.ConvTranspose2d(
                in_channels=filters[0], out_channels=3, kernel_size=3, padding=1
            )
        )
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def debug(self):
        print("Model summary:")
        print(self.model)


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

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, data_module)


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
