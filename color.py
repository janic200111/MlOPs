import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Conv2DTranspose, UpSampling2D
from keras.layers import Conv2D, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model,load_model,Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Binarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import matplotlib.pyplot as plt
import sys
from keras.datasets import cifar100
import cv2 as cv
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split

epochs = 10
PART_SIZE=64
batches=10
data_test_ratio=0.2

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def load_data(path, test_ratio=data_test_ratio, patch_size=(PART_SIZE, PART_SIZE),num_rot=1):
    file_list = os.listdir(path)
    random.shuffle(file_list)
    num_test = int(len(file_list) * test_ratio)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []

    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        image = Image.open(file_path)
        image = image.resize(patch_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        for angle in range(0, 360, int(360 / num_rot)):
            bw_rotated = image.convert('L').rotate(angle)
            color_rotated = image.rotate(angle)

            bw_image_np = np.array(bw_rotated) / 255.0
            bw_image_np = np.expand_dims(bw_image_np, axis=-1)
            color_image_np = np.array(color_rotated) / 255.0

            if file_name in file_list[:num_test]:
                X_test.append(bw_image_np)
                Y_test.append(color_image_np)
            else:
                X_train.append(bw_image_np)
                Y_train.append(color_image_np)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_test, Y_test, X_train, Y_train

def create_model(is_batch_norm, is_drop, decay):
    model = Sequential()
    filters = [64, 128, 256]

    # First Conv2D layer needs input_shape
    model.add(Conv2D(filters[0], (3, 3), padding='same', input_shape=(PART_SIZE, PART_SIZE, 1)))
    model.add(Activation('relu'))

    for f in filters[1:]:
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
        learning_rate_fn = ExponentialDecay(initial_learning_rate, decay_steps, 0.7)
        optimizer = SGD(learning_rate=learning_rate_fn)
    else:
        optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[ssim_metric])
    return model


X_test, Y_test, X_train, Y_train = load_data("Dane")
print('x train shape', X_train.shape)
print('x test shape', X_test.shape)

model = create_model(0,1,1)
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batches, epochs=epochs)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

obraz = load_img('test.jpg', target_size=(256, 256))
obraz_1 =obraz
bw_image = obraz.convert('L')
obraz = img_to_array(bw_image)
obraz = obraz / 255.0  # Normalizacja
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