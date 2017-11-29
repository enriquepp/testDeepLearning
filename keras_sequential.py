import time
import numpy as np
import pydot
import h5py
import keras
from keras.layers import Input, Add, Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.layers import ZeroPadding2D, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Sequential, Model, load_model
import keras.backend as K


# Generate dummy data
np.random.seed(12345)
x_train = np.random.random((1000, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

print("number of training examples = " + str(x_train.shape[0]))
print("number of test examples = " + str(x_test.shape[0]))
print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))
print("x_test shape: " + str(x_test.shape))
print("y_test shape: " + str(y_test.shape))


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

time.ctime()
a = time.process_time()
model.fit(x_train, y_train, batch_size=32, epochs=10)
runtime = (time.process_time() - a)
print(int(runtime), "seconds")
time.ctime()

score = model.evaluate(x_test, y_test, batch_size=32)
print ("Loss = " + str(score[0]))
print ("Test Accuracy = " + str(score[1]))
