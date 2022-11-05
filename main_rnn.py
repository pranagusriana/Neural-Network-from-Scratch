import numpy as np
from NeuralNetwork.layers import LSTM, Dense
from NeuralNetwork.lemah import Sequential

model = Sequential([
    LSTM(2, input_shape=(2, 2)),
    Dense(1, activation='sigmoid')
])

model.summary()

X = np.array([[[1, 2], [0.5, 3]], [[0.5, 3], [1, 2]]])

out = model(X)

print(out)
print(out.shape)