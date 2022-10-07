import pickle
from re import L
import numpy as np

class Sequential:
    """
    args:
        layers: Optional list of layers to add to the model.
    NOTE:
        First layers should have input_shape defined
    """
    def __init__(self,
                layers: list = []):
        self.build_stat = False
        self.compile_stat = False
        self.layers = []
        if (len(layers) != 0):
            for layer in layers:
                self.add(layer)

    def build(self):
        if (len(self.layers) > 1):
            for i in range(1, len(self.layers)):
                self.layers[i].add_input_shape(self.layers[i-1].getOutputShape())
            self.build_stat = True

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.compile_stat = True

    def add(self, layer):
        self.layers.append(layer)
        self.build_stat = False
        self.build()

    def __call__(self, batch_data):
        out = batch_data
        for layer in self.layers:
            out = layer(out)
        return out

    def fit(self, X, Y, epochs=1):
        for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = self.__call__(x)
                error += self.loss(y, output.T)
                grad = self.loss(y, output.T, prime=True)
                for layer in reversed(self.layers):
                    grad = layer.backward(self.optimizer, grad)
            print(f"{e + 1}/{epochs}, error={error}")
    
    def save(self, filename):
        pickle_out = open(filename, "wb")
        pickle.dump(self.layers,pickle_out)
        pickle_out.close()
    
    def load(self, filename):
        pickle_in = open(filename, "rb")
        self.layers = pickle.load(pickle_in)