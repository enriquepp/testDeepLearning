import time
from resnets_utils import *
from ResNet50 import *
from keras.models import load_model, save_model

model = ResNet50(input_shape=(64, 64, 3), classes=6)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

print(time.ctime())
a = time.process_time()
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)
runtime = (time.process_time() - a)
print(int(runtime), "seconds")
print(time.ctime())

testEval = model.evaluate(X_test, Y_test)
print("Loss = " + str(testEval[0]))
print("Test Accuracy = " + str(testEval[1]))
print('')

save_model(model, 'ResNet50_e10.h5')

print('')

