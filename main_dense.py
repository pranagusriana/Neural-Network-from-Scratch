import numpy as np
from PIL import Image
from CNN.layers import Dense
from CNN.lemah import Sequential
from CNN.losses import mse
from CNN.optimizers import SGD

X = [[
    [0, 0],
    [0, 1]],
    [[1, 0],
    [1, 1]
]]

y = [[[0], [1]], [[1], [0]]]

model = Sequential(
    [
        Dense(128, activation='relu', input_shape=(2, )),
        Dense(512, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)
optim = SGD(learning_rate=0.001, momentum=0.1)
loss = mse
model.compile(optim, loss)
X = np.array(X)
y = np.array(y)

model.fit(X, y, 10)
out = model(X[0])
print(out)