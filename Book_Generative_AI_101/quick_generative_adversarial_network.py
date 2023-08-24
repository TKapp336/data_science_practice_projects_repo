# (1) Import libraries and load data
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# (2) Create the generator
def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh'))
    return model

# (3) Create the discriminator
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# (4) Build and compile the GAN
def build_gan(generator, discriminator):
    discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    z = Input(shape=(z_dim,))
    img = generator(z)
    decision = discriminator(img)
    model = Model(z, decision)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Constants
img_shape = (28, 28, 1)
z_dim = 100
batch_size = 64
epochs = 10000

# Build and compile the generator
generator = build_generator(z_dim)
generator.compile(optimizer=Adam(), loss='binary_crossentropy')

# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile the GAN
gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

# Training Loop
for epoch in range(epochs):
    # Training the Discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    fake_imgs = generator.predict(np.random.randn(batch_size, z_dim))
    labels_real = np.ones((batch_size, 1))
    labels_fake = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_imgs, labels_real)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Training the Generator
    noise = np.random.randn(batch_size, z_dim)
    labels_g = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, labels_g)

    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")



