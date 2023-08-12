# (1) Import the libraries
import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# function for reading training data
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

# (2) Read training data
x_train, y_train = [], []
for i in range(1, 6):
    file_path = f'/Users/tylerkapp/Downloads/cifar-10-batches-py/data_batch_{i}'
    X, Y = load_cifar10_batch(file_path)
    x_train.append(X)
    y_train.append(Y)
x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

# (3) Read the testing data
file_path = '/Users/tylerkapp/Downloads/cifar-10-batches-py/test_batch'
x_test, y_test = load_cifar10_batch(file_path)

# (4) Preprocess the data
x_train /= 255.0
x_test /= 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# (5) Load the pre-trained VGG16-Model: Here you're exlcuding the top (fully connected layers),
# as you'll be loading you're own.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# (6) Freeze the Convolutional layers: If you want to keep the pre-trained weights of the VGG16
# model, you can freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# (7) Add Custom layers: You can now add your own fully connected layers on top of the VGG16
# model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x) # multiclass classification

model = Model(base_model.input, output)

# (8) Compile the model: Before training, compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# (9) Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
