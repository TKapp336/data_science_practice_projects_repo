# (1) Import the libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np

# (2) Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Freeze the base_model
base_model.trainable = False

# (3) Prepare the data
train_dir = '/Users/tylerkapp/Downloads/dogs-vs-cats/train'
validation_dir = '/Users/tylerkapp/Downloads/dogs-vs-cats/test1'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# (4) Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# (5) Adversarial training with the FGSM Attack
def adversarial_training(model, train_generator, validation_generator, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in train_generator:
            batch_x = tf.convert_to_tensor(batch_x)  # Convert batch_x to tensor
            batch_y = tf.reshape(batch_y, (-1, 1))  # Reshape batch_y to match predictions shape
            with tf.GradientTape() as tape:
                tape.watch(batch_x)
                predictions = model(batch_x)
                loss = losses.binary_crossentropy(batch_y, predictions)
            grads = tape.gradient(loss, batch_x)
            perturbations = tf.sign(grads)
            adversarial_batch_x = batch_x + 0.01 * perturbations
            model.train_on_batch(adversarial_batch_x, batch_y)

        # Validate the model
        val_loss, val_acc = model.evaluate(validation_generator)
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

adversarial_training(model, train_generator, validation_generator)

