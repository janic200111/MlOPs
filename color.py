import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Conv2DTranspose, UpSampling2D
from keras.layers import Conv2D, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model,load_model,Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Hyperparameters
Epochs = 10
PART_SIZE = 64
batches = 10

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# PyTorch Dataset class
class ImageColorizationDataset(Dataset):
    def __init__(self, path, patch_size=(PART_SIZE, PART_SIZE), num_rot=1, transform=None):
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
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        bw_image = image.convert('L')
        
        if self.transform:
            bw_image = self.transform(bw_image)
            image = self.transform(image)
        
        return bw_image, image

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = ImageColorizationDataset("Dane", transform=transform,num_rot=10)
dataloader = DataLoader(dataset, batch_size=batches, shuffle=True)

def create_model(is_batch_norm, is_drop, decay):
    model = Sequential()
    filters = [64, 128, 256]
    
    for f in filters:
        model.add(Conv2D(f, (3, 3), padding='same'))
        if is_batch_norm:
            model.add(BatchNormalization())
        if is_drop:
            model.add(Dropout(0.3))
        model.add(Activation('relu'))
    
    filters.reverse()
    for f in filters:
        model.add(Conv2DTranspose(f, (3, 3), padding='same'))
        model.add(Activation('relu'))

    model.add(Conv2DTranspose(3, (3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    
    if decay:
        initial_learning_rate = 0.01
        decay_steps = 10000
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, 0.7)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    
    model.compile(loss='mean_squared_error', metrics=[ssim_metric])
    return model

X_train, Y_train = [], []
for bw, color in dataloader:
    X_train.append(bw.numpy().transpose(0, 2, 3, 1))  
    Y_train.append(color.numpy().transpose(0, 2, 3, 1))

X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)

model = create_model(0, 1, 1)
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=Epochs)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# Additional test on a single image
obraz = load_img('test.jpg', target_size=(256, 256))
obraz_1 = obraz
bw_image = obraz.convert('L')
obraz = img_to_array(bw_image)
obraz = obraz / 255.0  # Normalization
obraz = np.expand_dims(obraz, axis=0)
wynik = model.predict(obraz)
wynikowy_obraz = wynik[0]
wynikowy_obraz = (wynikowy_obraz * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

axes[2].imshow(wynikowy_obraz)
axes[2].set_title('Wynikowy Obraz')
axes[2].axis('off')

axes[0].imshow(obraz_1)
axes[0].set_title('Obraz 1')
axes[0].axis('off')

axes[1].imshow(bw_image, cmap='gray')
axes[1].set_title('Czarno-biały')
axes[1].axis('off')

plt.show()
