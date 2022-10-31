import numpy as np
from PIL import Image
from NeuralNetwork.layers import Conv2D, Flatten, Dense
from NeuralNetwork.lemah import Sequential
from NeuralNetwork.losses import mse
from NeuralNetwork.optimizers import SGD
from NeuralNetwork.utils import data_gen

img_size = 50
cat_img = np.array(Image.open("./Data/CNN/test/cats/cat.38.jpg").resize((img_size, img_size)), dtype=int)
dog_img = np.array(Image.open("./Data/CNN/test/dogs/dog.4.jpg").resize((img_size, img_size)), dtype=int)
X_train, y_train, map_train = data_gen("./Data/CNN/train", img_size, 2, 42)
print(X_train.shape, y_train.shape)

model = Sequential(
    [
        Conv2D(16, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max', input_shape=(img_size, img_size, 3)),
        Conv2D(32, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max'),
        Conv2D(64, (3, 3), strides=1, padding=(0, 0), pooling_filter_size=(2, 2), pooling_strides=(2, 2), pooling_mode='max'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)
optim = SGD(learning_rate=0.01, momentum=0.1)
loss = mse
model.compile(optim, loss)

X = np.expand_dims(np.array([cat_img, dog_img])/255, axis=0)
y = np.expand_dims(np.expand_dims(np.array([1, 0]), axis=1), axis = 0)

model.fit(X_train, y_train, epochs=5)

model.save('saved_model_3.pickle')
out = model(X[0])
print("Loss:", loss(y, out))
print(out)