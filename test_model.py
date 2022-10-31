import numpy as np
from PIL import Image
from NeuralNetwork.layers import Conv2D, Flatten, Dense
from NeuralNetwork.lemah import Sequential
from NeuralNetwork.losses import mse
from NeuralNetwork.optimizers import SGD
from NeuralNetwork.utils import data_gen
from sklearn.metrics import accuracy_score, confusion_matrix

img_size = 50
X_test, y_test, map_test = data_gen("./Data/CNN/test", img_size, 2, 42)
print(X_test.shape, y_test.shape, map_test)

model = Sequential()
model.load("./saved_model_3.pickle")

y_pred = []
y_true = []
for X, y in zip(X_test, y_test):
    out = model(X)
    y_pred.append(out)
    y_true.append(y)
y_pred = np.array(y_pred).reshape(-1)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
y_pred = y_pred.astype(int)
y_true = np.array(y_true).reshape(-1)
print(y_pred)
print(y_true)
print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))